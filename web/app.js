// BearLM frontend logic: theme toggle, ask flow, markdown rendering, states.
(function () {
  "use strict";

  const root = document.documentElement;
  const toggle = document.getElementById("themeToggle");
  const form = document.getElementById("askForm");
  const input = document.getElementById("q");
  const submit = form.querySelector(".submit");
  const suggestions = document.getElementById("suggestions");
  const result = document.getElementById("result");

  // --- Theme -----------------------------------------------------------------
  toggle.addEventListener("click", function () {
    const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("bearlm-theme", next); } catch (e) {}
  });

  // --- Rendering -------------------------------------------------------------
  // Markdown/math/citation rendering lives in render.js (shared with Projects).
  function showLoading() {
    result.innerHTML =
      '<div class="card reveal">' +
      '<div class="card-label">Searching course materials</div>' +
      '<div class="skeleton"><span></span><span></span><span></span></div>' +
      "</div>";
  }

  function showError(message) {
    result.innerHTML =
      '<div class="card error reveal">' +
      '<div class="card-label">Something went wrong</div>' +
      '<div class="answer-body"><p></p></div>' +
      "</div>";
    result.querySelector(".answer-body p").textContent = message;
  }

  function showAnswer(data) {
    // The Ask page always queries the course corpus, so no source badge here.
    window.BearLM.renderResult(result, data, { showBadge: false });
  }

  // --- Ask flow --------------------------------------------------------------
  let pending = false;
  let lastOptimized = null;  // {original, optimized} — set when a suggestion is accepted

  async function ask(question) {
    if (pending || !question.trim()) return;
    pending = true;
    submit.disabled = true;
    if (suggestions) suggestions.classList.add("hidden");
    showLoading();

    // If this question was produced by the optimizer, remember the original.
    const originalMsg = (lastOptimized && question === lastOptimized.optimized)
      ? lastOptimized.original : null;
    lastOptimized = null;

    try {
      const rewrite = !!(rewriteToggle && rewriteToggle.checked);
      const resp = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question, rewrite: rewrite }),
      });
      if (!resp.ok) throw new Error("Request failed (" + resp.status + ")");
      const data = await resp.json();
      const provenance = {
        original: originalMsg,
        sent: originalMsg ? question : null,
        rewritten: data.rewritten_query || null,
      };
      window.BearLM.renderResult(result, data, { showBadge: false, provenance: provenance });
    } catch (err) {
      showError("Couldn't reach BearLM. Is the server running? (" + err.message + ")");
    } finally {
      pending = false;
      submit.disabled = false;
    }
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    ask(input.value);
  });

  // --- Prompt optimizer (suggest-then-confirm) + query-rewrite toggle --------
  const improveBtn = document.getElementById("improveBtn");
  const rewriteToggle = document.getElementById("rewriteToggle");
  const suggestPanel = document.getElementById("suggestPanel");
  if (improveBtn && suggestPanel) {
    improveBtn.addEventListener("click", async function () {
      const text = input.value.trim();
      if (!text) { input.focus(); return; }
      improveBtn.disabled = true;
      const label = improveBtn.textContent;
      improveBtn.textContent = "✨ Improving…";
      window.BearLM.suggestMessage(suggestPanel, "Optimizing your prompt…");
      try {
        const optimized = await window.BearLM.optimizePrompt(text);
        if (!optimized) { suggestPanel.hidden = true; }
        else window.BearLM.showSuggestion(suggestPanel, optimized, function (t) {
          input.value = t; input.focus();
          lastOptimized = { original: text, optimized: t };  // remember for provenance
        });
      } catch (e) {
        window.BearLM.suggestMessage(suggestPanel, e.message);
      } finally {
        improveBtn.disabled = false;
        improveBtn.textContent = label;
      }
    });
  }

  if (suggestions) {
    suggestions.querySelectorAll(".chip").forEach(function (chip) {
      chip.addEventListener("click", function () {
        input.value = chip.textContent;
        input.focus();
        ask(chip.textContent);
      });
    });
  }

  // Populate the footer corpus list from the live index (new textbooks appear
  // automatically). Falls back to the static list if the request fails.
  (async function () {
    const el = document.getElementById("corpusList");
    if (!el) return;
    try {
      const resp = await fetch("/api/courses");
      const data = await resp.json();
      if (data.courses && data.courses.length) {
        el.textContent = data.courses.join(" · ");
      }
    } catch (e) {}
  })();
})();
