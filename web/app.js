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

  async function ask(question) {
    if (pending || !question.trim()) return;
    pending = true;
    submit.disabled = true;
    if (suggestions) suggestions.classList.add("hidden");
    showLoading();

    try {
      const resp = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question }),
      });
      if (!resp.ok) throw new Error("Request failed (" + resp.status + ")");
      const data = await resp.json();
      showAnswer(data);
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
