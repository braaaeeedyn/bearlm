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

  // --- Rendering helpers -----------------------------------------------------
  // The model sometimes writes math in bare parens — ( \alpha ), ( \nabla f(w) ),
  // ( w ) — which KaTeX can't detect. Wrap those in $...$ so it can, without
  // touching prose like (course section p.159) or (weights), or existing math.
  function mathifyParens(text) {
    const spans = [];
    // 1. Stash existing math spans so we never edit inside them.
    let t = text.replace(
      /\$\$[\s\S]*?\$\$|\$[^$\n]+?\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)/g,
      function (m) {
        spans.push(m);
        return "\u0000" + (spans.length - 1) + "\u0000";
      }
    );
    // 2. Square-bracket display math with a command → $$...$$
    //    (the negative lookahead avoids markdown links like [text](url)).
    t = t.replace(
      /\[\s*([^\[\]]*\\[a-zA-Z]+[^\[\]]*?)\s*\](?!\()/g,
      function (_, inner) {
        return "$$" + inner.trim() + "$$";
      }
    );
    // 3. Parens containing a LaTeX command → inline math (allows one nested
    //    paren level, e.g. ( \nabla f(w) )).
    t = t.replace(
      /\(\s*((?:[^()]|\([^()]*\))*\\[a-zA-Z]+(?:[^()]|\([^()]*\))*)\s*\)/g,
      function (_, inner) {
        return "$" + inner.trim() + "$";
      }
    );
    // 4. A lone single variable in spaced parens: ( w ), ( x_i ) — but not
    //    f(w) (letter before "(") and not list markers like (a).
    t = t.replace(
      /(^|[^A-Za-z0-9\\])\(\s+([A-Za-z](?:_[A-Za-z0-9]+)?)\s+\)/g,
      function (_, pre, v) {
        return pre + "$" + v + "$";
      }
    );
    // 5. Restore the stashed math spans.
    return t.replace(/\u0000(\d+)\u0000/g, function (_, i) {
      return spans[+i];
    });
  }

  function renderMarkdown(text) {
    const prepared = mathifyParens(text || "");
    const raw = window.marked ? window.marked.parse(prepared) : prepared;
    return window.DOMPurify ? window.DOMPurify.sanitize(raw) : raw;
  }

  function renderMath(el) {
    if (!window.renderMathInElement) return;
    try {
      window.renderMathInElement(el, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\[", right: "\\]", display: true },
          { left: "\\(", right: "\\)", display: false },
        ],
        throwOnError: false,
      });
    } catch (e) {}
  }

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
    const card = document.createElement("div");
    card.className = "card reveal";
    const label = document.createElement("div");
    label.className = "card-label";
    label.textContent = "Answer";
    const bodyEl = document.createElement("div");
    bodyEl.className = "answer-body";
    bodyEl.innerHTML = renderMarkdown(data.answer || "");
    renderMath(bodyEl);
    card.appendChild(label);
    card.appendChild(bodyEl);

    result.innerHTML = "";
    result.appendChild(card);

    if (data.citations && data.citations.length) {
      const sources = document.createElement("div");
      sources.className = "sources reveal";
      const slabel = document.createElement("div");
      slabel.className = "card-label";
      slabel.textContent = "Sources";
      const tags = document.createElement("div");
      tags.className = "tags";
      data.citations.forEach(function (c) {
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = (c.course || "") + " · " + (c.section || "");
        tags.appendChild(tag);
      });
      sources.appendChild(slabel);
      sources.appendChild(tags);
      result.appendChild(sources);
    }
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
