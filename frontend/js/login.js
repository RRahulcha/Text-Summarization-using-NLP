// =========================================================
// login.js — Login page logic
// =========================================================

const LOGIN_API_URL = window.API_BASE_URL
  ? `${window.API_BASE_URL}/api/login`
  : "http://127.0.0.1:8000/api/login";

document.addEventListener("DOMContentLoaded", () => {
  showRegisteredBanner();

  const form = document.getElementById("loginForm");
  if (!form) return;

  form.addEventListener("submit", handleLoginSubmit);
});

function showRegisteredBanner() {
  const params = new URLSearchParams(window.location.search);
  if (params.get("registered") === "1") {
    const infoAlert = document.getElementById("infoAlert");
    if (infoAlert) {
      infoAlert.textContent = "Account created successfully. Please log in.";
      infoAlert.classList.add("is-visible");
    }
  }
}

async function handleLoginSubmit(event) {
  event.preventDefault();

  clearAllErrors();
  hideAlert();

  const identifier = document.getElementById("identifier").value.trim();
  const password = document.getElementById("password").value;

  let valid = true;

  if (!identifier) {
    setFieldError("identifier", "Enter your email or username.");
    valid = false;
  }

  if (!password) {
    setFieldError("password", "Password is required.");
    valid = false;
  }

  if (!valid) return;

  const submitBtn = document.getElementById("loginSubmit");
  setLoading(submitBtn, true, "Logging in...");

  try {
    const response = await fetch(LOGIN_API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ identifier, password }),
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      showAlert(data.detail || "Invalid username/email or password.");
      return;
    }

    // Store session token + user info for the dashboard.
    localStorage.setItem("nlp_summarizer_token", data.token);
    localStorage.setItem("nlp_summarizer_user", JSON.stringify(data.user));

    window.location.href = "dashboard.html";
  } catch (err) {
    showAlert("Could not reach the server. Is the backend running?");
  } finally {
    setLoading(submitBtn, false, "Login");
  }
}

// ---------------------------------------------------------
// Small UI helpers (duplicated locally so each page's JS
// file stays fully self-contained, per the project spec)
// ---------------------------------------------------------

function setFieldError(fieldId, message) {
  const input = document.getElementById(fieldId);
  const errorEl = document.getElementById(`${fieldId}Error`);
  if (input) input.classList.add("is-invalid");
  if (errorEl) errorEl.textContent = message;
}

function clearAllErrors() {
  document.querySelectorAll(".form-field input").forEach((el) => el.classList.remove("is-invalid"));
  document.querySelectorAll(".field-error").forEach((el) => (el.textContent = ""));
}

function showAlert(message) {
  const alertEl = document.getElementById("formAlert");
  if (!alertEl) return;
  alertEl.textContent = message;
  alertEl.classList.add("is-visible");
}

function hideAlert() {
  const alertEl = document.getElementById("formAlert");
  if (!alertEl) return;
  alertEl.classList.remove("is-visible");
}

function setLoading(button, isLoading, label) {
  if (!button) return;
  button.disabled = isLoading;
  button.textContent = label;
}
