// =========================================================
// dashboard.js — Dashboard page logic
// =========================================================

const API_BASE = window.API_BASE_URL || "http://127.0.0.1:8000";
const SUMMARIZE_URL = `${API_BASE}/api/summarize`;
const LOGOUT_URL = `${API_BASE}/api/logout`;

let lastSummaryText = "";

document.addEventListener("DOMContentLoaded", () => {
  if (!requireAuth()) return;

  renderUserName();
  setupCounts();
  setupButtons();
  setupAnalysisToggle();
  setupTabs();
});

// =========================================================
// AUTH GUARD
// =========================================================

function requireAuth() {
  const token = localStorage.getItem("nlp_summarizer_token");
  if (!token) {
    window.location.href = "login.html";
    return false;
  }
  return true;
}

function getToken() {
  return localStorage.getItem("nlp_summarizer_token");
}

function renderUserName() {
  const userRaw = localStorage.getItem("nlp_summarizer_user");
  const nameEl = document.getElementById("dashUserName");
  if (!userRaw || !nameEl) return;

  try {
    const user = JSON.parse(userRaw);
    nameEl.textContent = user.full_name || user.username || "Account";
  } catch (e) {
    nameEl.textContent = "Account";
  }
}

// =========================================================
// WORD / CHARACTER COUNTS
// =========================================================

function setupCounts() {
  const textarea = document.getElementById("originalText");
  const counter = document.getElementById("originalCount");

  const update = () => {
    const text = textarea.value;
    const words = countWords(text);
    counter.textContent = `${words} words · ${text.length} chars`;
  };

  textarea.addEventListener("input", update);
  update();
}

function countWords(text) {
  const trimmed = text.trim();
  return trimmed.length === 0 ? 0 : trimmed.split(/\s+/).length;
}

// =========================================================
// BUTTONS: Clear / Summarize / Copy / Download / Logout
// =========================================================

function setupButtons() {
  document.getElementById("clearBtn").addEventListener("click", handleClear);
  document.getElementById("summarizeBtn").addEventListener("click", handleSummarize);
  document.getElementById("copyBtn").addEventListener("click", handleCopy);
  document.getElementById("downloadBtn").addEventListener("click", handleDownload);
  document.getElementById("logoutBtn").addEventListener("click", handleLogout);
}

function handleClear() {
  const textarea = document.getElementById("originalText");
  textarea.value = "";
  textarea.dispatchEvent(new Event("input"));
  resetSummaryPanel();
  hideDashAlert();
}

function resetSummaryPanel() {
  const output = document.getElementById("summaryOutput");
  output.classList.remove("is-loading");
  output.innerHTML = `<p class="panel__placeholder" id="summaryPlaceholder">Your summary will appear here once you click “Summarize Text.”</p>`;
  document.getElementById("summaryCount").textContent = "0 words";
  document.getElementById("copyBtn").disabled = true;
  document.getElementById("downloadBtn").disabled = true;
  document.getElementById("analysisSection").hidden = true;
  lastSummaryText = "";
}

async function handleSummarize() {
  const textarea = document.getElementById("originalText");
  const text = textarea.value.trim();
  const ratio = parseFloat(document.getElementById("summaryRatio").value);
  const summarizeBtn = document.getElementById("summarizeBtn");
  const output = document.getElementById("summaryOutput");

  hideDashAlert();

  if (!text) {
    showDashAlert("Please enter some text to summarize.");
    return;
  }

  if (text.length < 20) {
    showDashAlert("Text is too short to summarize. Add a bit more content.");
    return;
  }

  summarizeBtn.disabled = true;
  summarizeBtn.textContent = "Analyzing text...";
  output.classList.add("is-loading");
  output.innerHTML = `<span>Analyzing text...</span>`;

  try {
    const response = await fetch(SUMMARIZE_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${getToken()}`,
      },
      body: JSON.stringify({ text, summary_ratio: ratio }),
    });

    if (response.status === 401) {
      showDashAlert("Your session expired. Please log in again.");
      setTimeout(() => (window.location.href = "login.html"), 1200);
      return;
    }

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      output.classList.remove("is-loading");
      output.innerHTML = `<p class="panel__placeholder">No summary yet.</p>`;
      showDashAlert(data.detail || "Something went wrong while summarizing.");
      return;
    }

    displaySummary(data);
  } catch (err) {
    output.classList.remove("is-loading");
    output.innerHTML = `<p class="panel__placeholder">No summary yet.</p>`;
    showDashAlert("Could not reach the server. Is the backend running?");
  } finally {
    summarizeBtn.disabled = false;
    summarizeBtn.textContent = "Summarize Text";
  }
}

function displaySummary(data) {
  const output = document.getElementById("summaryOutput");
  output.classList.remove("is-loading");

  lastSummaryText = data.summary || "";

  if (!lastSummaryText) {
    output.innerHTML = `<p class="panel__placeholder">No summary could be generated for this text.</p>`;
    document.getElementById("summaryCount").textContent = "0 words";
    document.getElementById("copyBtn").disabled = true;
    document.getElementById("downloadBtn").disabled = true;
    return;
  }

  output.innerHTML = `<p>${escapeHtml(lastSummaryText)}</p>`;
  document.getElementById("summaryCount").textContent = `${countWords(lastSummaryText)} words`;
  document.getElementById("copyBtn").disabled = false;
  document.getElementById("downloadBtn").disabled = false;

  renderAnalysis(data);
}

async function handleCopy() {
  if (!lastSummaryText) return;
  const copyBtn = document.getElementById("copyBtn");

  try {
    await navigator.clipboard.writeText(lastSummaryText);
    const original = copyBtn.textContent;
    copyBtn.textContent = "Copied!";
    setTimeout(() => (copyBtn.textContent = original), 1400);
  } catch (err) {
    showDashAlert("Could not copy to clipboard.");
  }
}

function handleDownload() {
  if (!lastSummaryText) return;

  const blob = new Blob([lastSummaryText], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "summary.txt";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

async function handleLogout() {
  const token = getToken();

  try {
    if (token) {
      await fetch(LOGOUT_URL, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });
    }
  } catch (err) {
    // Ignore network errors on logout — clear local session regardless.
  } finally {
    localStorage.removeItem("nlp_summarizer_token");
    localStorage.removeItem("nlp_summarizer_user");
    window.location.href = "login.html";
  }
}

// =========================================================
// NLP ANALYSIS DISPLAY
// =========================================================

function renderAnalysis(data) {
  const section = document.getElementById("analysisSection");
  section.hidden = false;

  // Sentence scores table, sorted highest first.
  const scoresBody = document.querySelector("#scoresTable tbody");
  scoresBody.innerHTML = "";
  const entries = Object.entries(data.sentence_scores || {}).sort((a, b) => b[1] - a[1]);
  entries.forEach(([sentence, score]) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${escapeHtml(sentence)}</td><td class="score-cell">${score.toFixed(2)}</td>`;
    scoresBody.appendChild(tr);
  });

  renderList("nltkSentences", data.nltk_sentences);
  renderChips("nltkTokens", data.nltk_tokens, "chip--blue");
  renderChips("nltkPos", (data.nltk_pos || []).map(([w, tag]) => `${w} / ${tag}`), "chip--accent");
  renderChips("nltkEntities", (data.nltk_entities || []).map((e) => `${e.text} / ${e.label}`), "chip--amber");

  renderList("spacySentences", data.spacy_sentences);
  renderChips("spacyTokens", data.spacy_tokens, "chip--blue");
  renderChips("spacyPos", (data.spacy_pos || []).map(([w, tag]) => `${w} / ${tag}`), "chip--accent");
  renderChips("spacyEntities", (data.spacy_entities || []).map((e) => `${e.text} / ${e.label}`), "chip--amber");
}

function renderList(elementId, items) {
  const el = document.getElementById(elementId);
  if (!el) return;
  el.innerHTML = "";
  (items || []).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    el.appendChild(li);
  });
}

function renderChips(elementId, items, chipClass) {
  const el = document.getElementById(elementId);
  if (!el) return;
  el.innerHTML = "";
  (items || []).slice(0, 400).forEach((item) => {
    const span = document.createElement("span");
    span.className = `chip ${chipClass}`;
    span.textContent = item;
    el.appendChild(span);
  });
}

// =========================================================
// ANALYSIS TOGGLE + TABS
// =========================================================

function setupAnalysisToggle() {
  const toggle = document.getElementById("analysisToggle");
  const body = document.getElementById("analysisBody");

  toggle.addEventListener("click", () => {
    const isOpen = body.hidden === false;
    body.hidden = isOpen;
    toggle.setAttribute("aria-expanded", String(!isOpen));
  });
}

function setupTabs() {
  const tabs = document.querySelectorAll(".tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.getAttribute("data-tab");

      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("is-active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("is-active"));

      tab.classList.add("is-active");
      document.querySelector(`.tab-panel[data-panel="${target}"]`).classList.add("is-active");
    });
  });
}

// =========================================================
// ALERTS + UTIL
// =========================================================

function showDashAlert(message) {
  const el = document.getElementById("dashAlert");
  el.textContent = message;
  el.classList.add("is-visible");
}

function hideDashAlert() {
  const el = document.getElementById("dashAlert");
  el.classList.remove("is-visible");
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
