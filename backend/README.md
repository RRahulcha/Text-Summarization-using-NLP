# NLP Text Summarizer — Backend

FastAPI service that exposes the NLP extractive summarizer (NLTK + spaCy) as a
REST API, plus a small SQLite-backed authentication system.

## Files

| File               | Purpose                                                            |
|---------------------|---------------------------------------------------------------------|
| `app.py`            | FastAPI app: routes, auth, request/response models, DB access.     |
| `summarizer.py`     | The NLP summarization engine (refactored from `textsummary.py`).   |
| `requirements.txt`  | Python dependencies.                                                |
| `users.db`          | SQLite database — created automatically on first run.              |

## Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

NLTK resources (`punkt`, `stopwords`, POS tagger, NE chunker, etc.) are
downloaded automatically the first time the server starts.

## Run

```bash
uvicorn app:app --reload --port 8000
```

The API is now available at `http://127.0.0.1:8000`. Interactive docs are at
`http://127.0.0.1:8000/docs`.

## Endpoints

See the root `README.md` for the full endpoint reference.

## Notes

- Passwords are hashed with `bcrypt` before being stored — never in plain text.
- Sessions are opaque random tokens stored in a `sessions` table, sent by the
  frontend as `Authorization: Bearer <token>`. They expire after 24 hours.
- `/api/summarize` requires a valid session token.
