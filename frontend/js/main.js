// =========================================================
// main.js — Home page behavior
// =========================================================

document.addEventListener("DOMContentLoaded", () => {
  setupNavToggle();
  setupHeroAnnotation();
});

/**
 * Mobile nav toggle — shared pattern reused on every page
 * that includes #navToggle / #navLinks.
 */
function setupNavToggle() {
  const toggle = document.getElementById("navToggle");
  const links = document.getElementById("navLinks");

  if (!toggle || !links) return;

  toggle.addEventListener("click", () => {
    const isOpen = links.classList.toggle("is-open");
    toggle.setAttribute("aria-expanded", String(isOpen));
  });
}

/**
 * Purely decorative hero preview: shows an example sentence
 * with its most "important" word highlighted, to illustrate
 * what the dashboard's summarizer does. No backend call.
 */
function setupHeroAnnotation() {
  const el = document.getElementById("annotatedSentence");
  if (!el) return;

  const before = "The Reserve Bank of India raised interest rates on ";
  const highlighted = "Tuesday";
  const after = " to curb inflation across the country.";

  el.innerHTML =
    escapeHtml(before) +
    "<mark>" + escapeHtml(highlighted) + "</mark>" +
    escapeHtml(after);
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
