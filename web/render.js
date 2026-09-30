// Shared answer rendering for BearLM (Ask page + Projects page).
// Exposes window.BearLM.{renderMarkdown, renderMath, renderResult}.
(function () {
  "use strict";

  // The model sometimes writes math in bare parens; wrap those so KaTeX sees them.
  function mathifyParens(text) {
    const spans = [];
    let t = text.replace(
      /\$\$[\s\S]*?\$\$|\$[^$\n]+?\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)/g,
      function (m) { spans.push(m); return "\u0000" + (spans.length - 1) + "\u0000"; }
    );
    t = t.replace(/\[\s*([^\[\]]*\\[a-zA-Z]+[^\[\]]*?)\s*\](?!\()/g,
      function (_, inner) { return "$$" + inner.trim() + "$$"; });
    t = t.replace(/\(\s*((?:[^()]|\([^()]*\))*\\[a-zA-Z]+(?:[^()]|\([^()]*\))*)\s*\)/g,
      function (_, inner) { return "$" + inner.trim() + "$"; });
    t = t.replace(/(^|[^A-Za-z0-9\\])\(\s+([A-Za-z](?:_[A-Za-z0-9]+)?)\s+\)/g,
      function (_, pre, v) { return pre + "$" + v + "$"; });
    return t.replace(/\u0000(\d+)\u0000/g, function (_, i) { return spans[+i]; });
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

  // Build a deep-link URL for a citation given the answer's source + context.
  // Returns null when the citation isn't a page-anchored PDF we can open.
  function citationLink(c, source, ctx) {
    const src = c.source || "";
    const pageMatch = /^p\.(\d+)$/i.exec(c.section || "");
    if (!src.toLowerCase().endsWith(".pdf") || !pageMatch) return null;
    const page = pageMatch[1];
    const f = encodeURIComponent(src);
    if (source === "project" && ctx && ctx.slug) {
      return "/projects/" + encodeURIComponent(ctx.slug) + "/pdfs/" + f + "#page=" + page;
    }
    if (source === "upload" && ctx && ctx.uid) {
      return "/uploads/" + encodeURIComponent(ctx.uid) + "/" + f + "#page=" + page;
    }
    if (c.course) {
      return "/source/" + encodeURIComponent(c.course) + "/" + f + "#page=" + page;
    }
    return null;
  }

  // Small badge showing where the answer came from.
  var SOURCE_BADGE = {
    project: { text: "From project", cls: "badge-project" },
    corpus: { text: "From course corpus", cls: "badge-corpus" },
    upload: { text: "From your PDF", cls: "badge-upload" },
    none: { text: "Not found in sources", cls: "badge-none" },
  };

  // A "what was sent to the model" panel: your message vs. the optimized prompt
  // and/or the rewritten retrieval query. prov = { original, sent, rewritten }.
  function provenanceNode(prov) {
    if (!prov) return null;
    const rows = [];
    if (prov.original) rows.push(["Your message", prov.original]);
    if (prov.sent && prov.sent !== prov.original) rows.push(["Optimized prompt sent", prov.sent]);
    if (prov.rewritten) rows.push(["Retrieval query used", prov.rewritten]);
    if (!rows.length) return null;
    const box = document.createElement("div");
    box.className = "provenance";
    rows.forEach(function (r) {
      const row = document.createElement("div");
      row.className = "prov-row";
      const k = document.createElement("span");
      k.className = "prov-key"; k.textContent = r[0];
      const v = document.createElement("span");
      v.className = "prov-val"; v.textContent = r[1];
      row.appendChild(k); row.appendChild(v);
      box.appendChild(row);
    });
    return box;
  }

  // Render an answer (+ optional provenance + source badge + citations) into `resultEl`.
  // data: { answer, citations, source }.  ctx: { slug?, uid?, showBadge?, provenance? }.
  function renderResult(resultEl, data, ctx) {
    ctx = ctx || {};
    const source = data.source || "corpus";
    resultEl.innerHTML = "";
    const prov = provenanceNode(ctx.provenance);
    if (prov) resultEl.appendChild(prov);

    const card = document.createElement("div");
    card.className = "card reveal";
    const label = document.createElement("div");
    label.className = "card-label";
    label.textContent = "Answer";
    card.appendChild(label);

    if (ctx.showBadge && SOURCE_BADGE[source]) {
      const b = SOURCE_BADGE[source];
      const badge = document.createElement("span");
      badge.className = "src-badge " + b.cls;
      badge.textContent = b.text;
      label.appendChild(badge);
    }

    const bodyEl = document.createElement("div");
    bodyEl.className = "answer-body";
    bodyEl.innerHTML = renderMarkdown(data.answer || "");
    renderMath(bodyEl);
    card.appendChild(bodyEl);
    resultEl.appendChild(card);

    if (data.citations && data.citations.length) {
      const sources = document.createElement("div");
      sources.className = "sources reveal";
      const slabel = document.createElement("div");
      slabel.className = "card-label";
      slabel.textContent = "Sources";
      const tags = document.createElement("div");
      tags.className = "tags";
      data.citations.forEach(function (c) {
        const text = (c.course || "") + " · " + (c.section || "");
        const url = citationLink(c, source, ctx);
        if (url) {
          const a = document.createElement("a");
          a.className = "tag tag-link";
          a.href = url;
          a.target = "_blank";
          a.rel = "noopener noreferrer";
          a.title = "Open " + (c.source || "") + " at " + (c.section || "");
          a.textContent = text;
          tags.appendChild(a);
        } else {
          const tag = document.createElement("span");
          tag.className = "tag";
          tag.textContent = text;
          tags.appendChild(tag);
        }
      });
      sources.appendChild(slabel);
      sources.appendChild(tags);
      resultEl.appendChild(sources);
    }
  }

  // Build a citation tag element (link if it's a page-anchored PDF, else plain).
  function citationTag(c, source, ctx) {
    const text = (c.course || "") + " · " + (c.section || "");
    const url = citationLink(c, source, ctx);
    if (url) {
      const a = document.createElement("a");
      a.className = "tag tag-link";
      a.href = url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.title = "Open " + (c.source || "") + " at " + (c.section || "");
      a.textContent = text;
      return a;
    }
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = text;
    return tag;
  }

  // Prompt optimizer: call the API and render a suggest-then-confirm panel.
  async function optimizePrompt(text) {
    const resp = await fetch("/api/optimize", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: text }),
    });
    if (!resp.ok) throw new Error("Couldn't optimize (" + resp.status + ")");
    return (await resp.json()).optimized || "";
  }

  function showSuggestion(panel, optimized, onUse) {
    panel.hidden = false;
    panel.innerHTML = "";
    const label = document.createElement("div");
    label.className = "suggest-label"; label.textContent = "Suggested prompt";
    const body = document.createElement("div");
    body.className = "suggest-text"; body.textContent = optimized;
    const actions = document.createElement("div");
    actions.className = "suggest-actions";
    const use = document.createElement("button");
    use.type = "button"; use.className = "tool-btn primary"; use.textContent = "Use this";
    use.addEventListener("click", function () { onUse(optimized); panel.hidden = true; });
    const keep = document.createElement("button");
    keep.type = "button"; keep.className = "tool-btn"; keep.textContent = "Keep mine";
    keep.addEventListener("click", function () { panel.hidden = true; });
    actions.appendChild(use); actions.appendChild(keep);
    panel.appendChild(label); panel.appendChild(body); panel.appendChild(actions);
  }

  function suggestMessage(panel, message) {
    panel.hidden = false;
    panel.innerHTML = "";
    const m = document.createElement("div");
    m.className = "suggest-text"; m.textContent = message;
    panel.appendChild(m);
  }

  window.BearLM = {
    renderMarkdown: renderMarkdown,
    renderMath: renderMath,
    renderResult: renderResult,
    citationTag: citationTag,
    sourceBadge: SOURCE_BADGE,
    optimizePrompt: optimizePrompt,
    showSuggestion: showSuggestion,
    suggestMessage: suggestMessage,
    provenanceNode: provenanceNode,
  };
})();
