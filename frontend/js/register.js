// =========================================================
// register.js — Registration page logic
// =========================================================

const REGISTER_API_URL = window.API_BASE_URL
  ? `${window.API_BASE_URL}/api/register`
  : "http://127.0.0.1:8000/api/register";

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("registerForm");
  if (!form) return;

  form.addEventListener("submit", handleRegisterSubmit);
});

async function handleRegisterSubmit(event) {
  event.preventDefault();

  clearAllErrors();
  hideAlert();

  const fullName = document.getElementById("fullName").value.trim();
  const email = document.getElementById("email").value.trim();
  const username = document.getElementById("username").value.trim();
  const password = document.getElementById("password").value;
  const confirmPassword = document.getElementById("confirmPassword").value;

  const isValid = validateRegisterForm({
    fullName, email, username, password, confirmPassword,
  });

  if (!isValid) return;

  const submitBtn = document.getElementById("registerSubmit");
  setLoading(submitBtn, true, "Creating account...");

  try {
    const response = await fetch(REGISTER_API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        full_name: fullName,
        email,
        username,
        password,
        confirm_password: confirmPassword,
      }),
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      showAlert(data.detail || "Registration failed. Please try again.");
      return;
    }

    showAlert("Account created! Redirecting to login...", "success");
    setTimeout(() => {
      window.location.href = "login.html?registered=1";
    }, 900);
  } catch (err) {
    showAlert("Could not reach the server. Is the backend running?");
  } finally {
    setLoading(submitBtn, false, "Register");
  }
}

function validateRegisterForm({ fullName, email, username, password, confirmPassword }) {
  let valid = true;

  if (!fullName) {
    setFieldError("fullName", "Full name is required.");
    valid = false;
  }

  if (!email) {
    setFieldError("email", "Email is required.");
    valid = false;
  } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    setFieldError("email", "Enter a valid email address.");
    valid = false;
  }

  if (!username) {
    setFieldError("username", "Username is required.");
    valid = false;
  } else if (/\s/.test(username) || username.length < 3) {
    setFieldError("username", "At least 3 characters, no spaces.");
    valid = false;
  }

  if (!password) {
    setFieldError("password", "Password is required.");
    valid = false;
  } else if (password.length < 8) {
    setFieldError("password", "Use at least 8 characters.");
    valid = false;
  }

  if (!confirmPassword) {
    setFieldError("confirmPassword", "Please confirm your password.");
    valid = false;
  } else if (password && confirmPassword !== password) {
    setFieldError("confirmPassword", "Passwords do not match.");
    valid = false;
  }

  return valid;
}

// ---------------------------------------------------------
// Shared small UI helpers (also used conceptually by login.js)
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

function showAlert(message, type = "error") {
  const alertEl = document.getElementById("formAlert");
  if (!alertEl) return;
  alertEl.textContent = message;
  alertEl.classList.remove("alert--error", "alert--success");
  alertEl.classList.add(type === "success" ? "alert--success" : "alert--error", "is-visible");
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
