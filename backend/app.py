"""
app.py
=========================================================
FastAPI application for the NLP Text Summarizer website.

Responsibilities:
  * User registration / login / logout backed by SQLite
    (users.db), with bcrypt password hashing.
  * Simple opaque session-token authentication (a random
    token issued at login, stored server-side, sent by the
    frontend as `Authorization: Bearer <token>`).
  * POST /api/summarize — runs the NLP pipeline from
    summarizer.py on user-supplied text.
  * GET  /api/health — basic liveness/readiness check.

All summarization logic itself lives in summarizer.py and
is untouched here; this file is purely the API + auth layer
around it.
=========================================================
"""

import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import bcrypt
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, field_validator

import summarizer as nlp_summarizer

# =========================================================
# CONFIG
# =========================================================

DB_PATH = Path(__file__).parent / "users.db"
SESSION_LIFETIME_HOURS = 24
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# =========================================================
# PASSWORD HASHING (bcrypt, used directly)
# =========================================================
# bcrypt only uses the first 72 bytes of a password, which is
# a hard limit of the algorithm itself, not something this
# code can lift. Passwords are capped at 128 chars on the
# RegisterRequest model, so this is only relevant for very
# long passwords.

def hash_password(password: str) -> str:
    password_bytes = password.encode("utf-8")[:72]
    hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    password_bytes = password.encode("utf-8")[:72]
    try:
        return bcrypt.checkpw(password_bytes, password_hash.encode("utf-8"))
    except ValueError:
        return False

app = FastAPI(
    title="NLP Text Summarizer API",
    description="REST API for the NLP Text Summarizer web app",
    version="1.0.0",
)

# Frontend is static HTML/CSS/JS served separately (e.g. via
# a simple local file server or VS Code Live Server), so CORS
# is opened up for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# DATABASE
# =========================================================

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )


# =========================================================
# SCHEMAS
# =========================================================

class RegisterRequest(BaseModel):
    full_name: str = Field(..., min_length=1, max_length=120)
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=128)
    confirm_password: str = Field(..., min_length=8, max_length=128)

    @field_validator("username")
    @classmethod
    def username_no_spaces(cls, value: str) -> str:
        cleaned = value.strip()
        if " " in cleaned:
            raise ValueError("Username cannot contain spaces.")
        if len(cleaned) < 3:
            raise ValueError("Username must be at least 3 characters long.")
        return cleaned.lower()

    @field_validator("full_name")
    @classmethod
    def full_name_clean(cls, value: str) -> str:
        return value.strip()


class LoginRequest(BaseModel):
    identifier: str = Field(..., min_length=1, description="Email or username")
    password: str = Field(..., min_length=1)


def normalize_identifier(value: str) -> str:
    return value.strip().lower()


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    summary_ratio: float = Field(0.30, ge=0.05, le=0.9)


class UserOut(BaseModel):
    id: int
    full_name: str
    email: str
    username: str


class AuthResponse(BaseModel):
    message: str
    token: Optional[str] = None
    user: Optional[UserOut] = None


class SummarizeResponse(BaseModel):
    summary: str
    nltk_sentences: list
    spacy_sentences: list
    nltk_tokens: list
    spacy_pos: list
    nltk_pos: list
    nltk_entities: list
    spacy_entities: list
    sentence_scores: dict
    spacy_tokens: list
    error: Optional[str] = None


# =========================================================
# AUTH HELPERS
# =========================================================

def create_session(user_id: int) -> str:
    token = secrets.token_hex(32)
    expires_at = (
        datetime.now(timezone.utc) + timedelta(hours=SESSION_LIFETIME_HOURS)
    ).isoformat()

    with get_db() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
            (token, user_id, expires_at),
        )

    return token


def get_current_user(authorization: Optional[str] = Header(None)) -> sqlite3.Row:
    """
    Reads the Authorization: Bearer <token> header, validates
    the session, and returns the associated user row. Raises
    401 for any missing/invalid/expired token.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header.",
        )

    token = authorization.split(" ", 1)[1].strip()

    with get_db() as conn:
        session_row = conn.execute(
            "SELECT * FROM sessions WHERE token = ?", (token,)
        ).fetchone()

        if session_row is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session not found. Please log in again.",
            )

        expires_at = datetime.fromisoformat(session_row["expires_at"])
        if datetime.now(timezone.utc) > expires_at:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired. Please log in again.",
            )

        user_row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (session_row["user_id"],)
        ).fetchone()

        if user_row is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account no longer exists.",
            )

        return user_row


def user_row_to_out(row: sqlite3.Row) -> UserOut:
    return UserOut(
        id=row["id"], full_name=row["full_name"], email=row["email"], username=row["username"]
    )


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
def on_startup():
    init_db()
    try:
        nlp_summarizer.initialize_nlp_engine()
    except nlp_summarizer.SpacyModelMissingError as exc:
        # Don't crash the whole server — surface a clear error
        # on the first /api/summarize call instead. This lets
        # /api/health still respond while the model is fixed.
        print(f"[STARTUP WARNING] {exc}")


# =========================================================
# ROUTES — HEALTH
# =========================================================

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "NLP Text Summarizer API"}


# =========================================================
# ROUTES — AUTH
# =========================================================

@app.post("/api/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest):
    if payload.password != payload.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    normalized_email = normalize_identifier(str(payload.email))
    normalized_username = normalize_identifier(payload.username)

    with get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE LOWER(email) = ? OR LOWER(username) = ?",
            (normalized_email, normalized_username),
        ).fetchone()

        if existing:
            raise HTTPException(
                status_code=409,
                detail="An account with that email or username already exists.",
            )

        password_hash = hash_password(payload.password)

        conn.execute(
            """
            INSERT INTO users (full_name, email, username, password_hash, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                payload.full_name.strip(),
                normalized_email,
                normalized_username,
                password_hash,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    return AuthResponse(message="Registration successful. Please log in.")


@app.post("/api/login", response_model=AuthResponse)
def login(payload: LoginRequest):
    identifier = normalize_identifier(payload.identifier)

    with get_db() as conn:
        user_row = conn.execute(
            "SELECT * FROM users WHERE LOWER(email) = ? OR LOWER(username) = ?",
            (identifier, identifier),
        ).fetchone()

        if user_row is None or not verify_password(payload.password, user_row["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid username/email or password.")

        token = create_session(user_row["id"])

    return AuthResponse(
        message="Login successful.",
        token=token,
        user=user_row_to_out(user_row),
    )


@app.post("/api/logout")
def logout(authorization: Optional[str] = Header(None)):
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
        with get_db() as conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))

    return {"message": "Logged out."}


@app.get("/api/me", response_model=UserOut)
def me(current_user: sqlite3.Row = Depends(get_current_user)):
    return user_row_to_out(current_user)


# =========================================================
# ROUTES — SUMMARIZATION
# =========================================================

@app.post("/api/summarize", response_model=SummarizeResponse)
def summarize(
    payload: SummarizeRequest,
    current_user: sqlite3.Row = Depends(get_current_user),
):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    if len(payload.text.strip()) < 20:
        raise HTTPException(
            status_code=400,
            detail="Text is too short to summarize. Please provide more content.",
        )

    try:
        result = nlp_summarizer.summarizer(payload.text, payload.summary_ratio)
    except nlp_summarizer.SpacyModelMissingError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        # Never leak a raw traceback to the client.
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while summarizing the text. Please try again.",
        )

    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])

    return result
