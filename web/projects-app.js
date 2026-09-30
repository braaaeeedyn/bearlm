// BearLM Projects: collapsible sidebar, project list (newest→oldest), per-project
// reference PDFs + multiple chats, multi-turn grounded conversation, reset context,
// and a stale-project (>1 week) delete prompt on open.
(function () {
  "use strict";

  const root = document.documentElement;
  const $ = function (id) { return document.getElementById(id); };
  const STALE_DAYS = 7;

  // --- theme + sidebar collapse ----------------------------------------------
  $("themeToggle").addEventListener("click", function () {
    const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("bearlm-theme", next); } catch (e) {}
  });
  function setSidebar(collapsed) {
    root.setAttribute("data-sidebar", collapsed ? "collapsed" : "expanded");
    try { localStorage.setItem("bearlm-sidebar", collapsed ? "collapsed" : "expanded"); } catch (e) {}
  }
  $("sidebarToggle").addEventListener("click", function () { setSidebar(true); });
  $("sidebarOpen").addEventListener("click", function () { setSidebar(false); });

  // --- state -----------------------------------------------------------------
  let currentSlug = null;
  let currentChatId = null;
  let busy = false;
  let chatLastOptimized = null;  // {original, optimized} for provenance

  // --- helpers ---------------------------------------------------------------
  function show(el) { el.classList.remove("hidden"); }
  function hide(el) { el.classList.add("hidden"); }
  async function jsonFetch(url, opts) {
    const resp = await fetch(url, opts);
    if (!resp.ok) {
      let detail = "Request failed (" + resp.status + ")";
      try { const j = await resp.json(); if (j.detail) detail = j.detail; } catch (e) {}
      throw new Error(detail);
    }
    return resp.status === 204 ? null : resp.json();
  }
  function daysOld(iso) {
    if (!iso) return 0;
    return (Date.now() - new Date(iso).getTime()) / 86400000;
  }
  function relTime(iso) {
    const d = daysOld(iso);
    if (d < 1 / 24) return "just now";
    if (d < 1) return Math.floor(d * 24) + "h ago";
    if (d < 7) return Math.floor(d) + "d ago";
    return new Date(iso).toLocaleDateString();
  }

  // --- projects list ---------------------------------------------------------
  async function loadProjects(selectAfter) {
    const list = $("projectList");
    try {
      const data = await jsonFetch("/api/projects");
      list.innerHTML = "";
      if (!data.projects.length) {
        list.innerHTML = '<p class="side-empty">No projects yet.</p>';
      } else {
        data.projects.forEach(function (p) {
          const item = document.createElement("button");
          item.type = "button";
          item.className = "side-item";
          item.dataset.slug = p.slug;
          item.innerHTML =
            '<span class="side-item-name"></span>' +
            '<span class="side-item-meta">' + p.pdfs.length + ' PDF' + (p.pdfs.length === 1 ? '' : 's') +
            ' · ' + (p.chats ? p.chats.length : 0) + ' chat' + ((p.chats && p.chats.length === 1) ? '' : 's') + '</span>';
          item.querySelector(".side-item-name").textContent = p.name;
          item.addEventListener("click", function () { selectProject(p.slug); });
          list.appendChild(item);
        });
      }
      if (selectAfter) selectProject(selectAfter);
      return data.projects;
    } catch (e) {
      list.innerHTML = '<p class="side-empty">Couldn\'t load. (' + e.message + ')</p>';
      return [];
    }
  }

  async function createProject() {
    const input = $("newProjectName");
    const name = input.value.trim();
    if (!name || busy) return;
    busy = true;
    try {
      const p = await jsonFetch("/api/projects", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name }),
      });
      input.value = "";
      await loadProjects(p.slug);
    } catch (e) { alert(e.message); }
    finally { busy = false; }
  }

  // --- project detail (sidebar) ----------------------------------------------
  function showProjectsList() {
    currentSlug = null; currentChatId = null;
    hide($("sideProject")); show($("projectsSection"));
    hide($("chatView")); show($("chatEmpty"));
    document.querySelectorAll(".side-item").forEach(function (n) { n.classList.remove("is-active"); });
  }

  async function selectProject(slug) {
    currentSlug = slug; currentChatId = null;
    hide($("projectsSection")); show($("sideProject"));
    hide($("chatView")); show($("chatEmpty"));
    await refreshProject();
  }

  async function refreshProject() {
    if (!currentSlug) return;
    const p = await jsonFetch("/api/projects/" + encodeURIComponent(currentSlug));
    $("sideProjectName").textContent = p.name;
    renderDocList(p.pdfs);
    renderChatList(p.chats || []);
  }

  let pdfView = "list";
  try { pdfView = localStorage.getItem("bearlm-pdfview") || "list"; } catch (e) {}
  let lastPdfs = [];

  function pdfUrl(filename) {
    return "/projects/" + encodeURIComponent(currentSlug) + "/pdfs/" + encodeURIComponent(filename);
  }

  function removeBtn(filename) {
    const x = document.createElement("button");
    x.className = "side-doc-x"; x.type = "button"; x.title = "Remove PDF"; x.textContent = "✕";
    x.addEventListener("click", function (e) { e.preventDefault(); e.stopPropagation(); removePdf(filename); });
    return x;
  }

  function renderDocList(pdfs) {
    lastPdfs = pdfs || [];
    const wrap = $("projDocList");
    wrap.setAttribute("data-view", pdfView);
    wrap.innerHTML = "";
    if (!lastPdfs.length) { wrap.innerHTML = '<p class="side-empty">No PDFs yet.</p>'; return; }

    lastPdfs.forEach(function (d) {
      if (pdfView === "grid") {
        // Clickable thumbnail image that opens the PDF in a new tab.
        const tile = document.createElement("div");
        tile.className = "side-tile";
        const a = document.createElement("a");
        a.className = "side-tile-link"; a.href = pdfUrl(d.filename);
        a.target = "_blank"; a.rel = "noopener noreferrer"; a.title = "Open " + d.filename;
        const img = document.createElement("img");
        img.className = "side-tile-img"; img.loading = "lazy"; img.alt = d.filename;
        img.src = pdfUrl(d.filename) + "/thumb";
        const cap = document.createElement("span");
        cap.className = "side-tile-name"; cap.textContent = d.filename;
        a.appendChild(img); a.appendChild(cap);
        tile.appendChild(a); tile.appendChild(removeBtn(d.filename));
        wrap.appendChild(tile);
      } else {
        // List row: click the name to open the PDF in a new tab.
        const row = document.createElement("div");
        row.className = "side-doc";
        const a = document.createElement("a");
        a.className = "side-doc-open"; a.href = pdfUrl(d.filename);
        a.target = "_blank"; a.rel = "noopener noreferrer"; a.title = "Open " + d.filename;
        a.textContent = d.filename;
        const meta = document.createElement("span");
        meta.className = "side-doc-meta"; meta.textContent = d.chunks;
        row.appendChild(a); row.appendChild(meta); row.appendChild(removeBtn(d.filename));
        wrap.appendChild(row);
      }
    });
  }

  function togglePdfView() {
    pdfView = pdfView === "list" ? "grid" : "list";
    try { localStorage.setItem("bearlm-pdfview", pdfView); } catch (e) {}
    renderDocList(lastPdfs);
  }

  async function removePdf(filename) {
    if (!currentSlug || busy) return;
    if (!confirm('Remove "' + filename + '" from this project? Its chunks will be deleted.')) return;
    busy = true;
    try {
      await jsonFetch("/api/projects/" + encodeURIComponent(currentSlug) + "/pdfs/" + encodeURIComponent(filename), { method: "DELETE" });
      await refreshProject();
      await loadProjects();
    } catch (e) { alert(e.message); }
    finally { busy = false; }
  }

  function renderChatList(chats) {
    const wrap = $("chatList");
    wrap.innerHTML = "";
    if (!chats.length) { wrap.innerHTML = '<p class="side-empty">No chats yet — start one.</p>'; return; }
    chats.forEach(function (c) {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "side-chat" + (c.id === currentChatId ? " is-active" : "");
      item.innerHTML = '<span class="side-chat-title"></span>' +
        '<span class="side-chat-meta">' + relTime(c.updated) + '</span>' +
        '<span class="side-chat-x" title="Delete chat">✕</span>';
      item.querySelector(".side-chat-title").textContent = c.title;
      item.addEventListener("click", function (e) {
        if (e.target.classList.contains("side-chat-x")) { deleteChat(c.id); return; }
        openChat(c.id);
      });
      wrap.appendChild(item);
    });
  }

  async function addProjectPdf(fileInput) {
    const file = fileInput.files && fileInput.files[0];
    if (!file || !currentSlug || busy) return;
    busy = true;
    const wrap = $("projDocList");
    const note = document.createElement("p");
    note.className = "side-empty"; note.textContent = "Embedding " + file.name + "…";
    wrap.prepend(note);
    try {
      const form = new FormData(); form.append("file", file);
      await jsonFetch("/api/projects/" + encodeURIComponent(currentSlug) + "/pdfs", { method: "POST", body: form });
      await refreshProject();
      await loadProjects();
    } catch (e) { alert(e.message); }
    finally { fileInput.value = ""; busy = false; }
  }

  async function deleteProject() {
    if (!currentSlug || busy) return;
    if (!confirm("Delete this project, its PDFs and all chats? This can't be undone.")) return;
    busy = true;
    try {
      await jsonFetch("/api/projects/" + encodeURIComponent(currentSlug), { method: "DELETE" });
      showProjectsList();
      await loadProjects();
    } catch (e) { alert(e.message); }
    finally { busy = false; }
  }

  async function resetContext() {
    if (!currentSlug || busy) return;
    if (!confirm("Reset context: delete all chats in this project (keeping your PDFs)?\n\nThis clears conversation history to reduce drift/hallucination.")) return;
    busy = true;
    try {
      await jsonFetch("/api/projects/" + encodeURIComponent(currentSlug) + "/reset", { method: "POST" });
      currentChatId = null;
      hide($("chatView")); show($("chatEmpty"));
      await refreshProject();
      await loadProjects();
    } catch (e) { alert(e.message); }
    finally { busy = false; }
  }

  // --- chats -----------------------------------------------------------------
  async function newChat() {
    if (!currentSlug || busy) return;
    busy = true;
    try {
      const c = await jsonFetch("/api/projects/" + encodeURIComponent(currentSlug) + "/chats", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}),
      });
      await refreshProject();
      openChat(c.id);
    } catch (e) { alert(e.message); }
    finally { busy = false; }
  }

  async function deleteChat(chatId) {
    if (!currentSlug || busy) return;
    if (!confirm("Delete this chat?")) return;
    busy = true;
    try {
      await jsonFetch("/api/projects/" + encodeURIComponent(currentSlug) + "/chats/" + chatId, { method: "DELETE" });
      if (chatId === currentChatId) { currentChatId = null; hide($("chatView")); show($("chatEmpty")); }
      await refreshProject();
    } catch (e) { alert(e.message); }
    finally { busy = false; }
  }

  async function openChat(chatId) {
    currentChatId = chatId;
    hide($("chatEmpty")); show($("chatView"));
    document.querySelectorAll(".side-chat").forEach(function (n) { n.classList.remove("is-active"); });
    const p = await jsonFetch("/api/projects/" + encodeURIComponent(currentSlug));
    renderChatList(p.chats || []);
    try {
      const chat = await jsonFetch("/api/projects/" + encodeURIComponent(currentSlug) + "/chats/" + chatId);
      renderConversation(chat.messages || []);
    } catch (e) { renderConversation([]); }
    $("chatQ").focus();
  }

  // --- conversation rendering ------------------------------------------------
  function ctx() { return { slug: currentSlug }; }

  function appendMessage(msg) {
    const scroll = $("chatScroll");
    const wrap = document.createElement("div");
    wrap.className = "msg msg-" + (msg.role === "user" ? "user" : "assistant");

    if (msg.role === "user") {
      wrap.textContent = msg.content;
    } else {
      // provenance (your message vs. optimized prompt / rewritten query) — live turn only
      if (msg.provenance) {
        const p = window.BearLM.provenanceNode(msg.provenance);
        if (p) wrap.appendChild(p);
      }
      const body = document.createElement("div");
      body.className = "answer-body";
      body.innerHTML = window.BearLM.renderMarkdown(msg.content || "");
      window.BearLM.renderMath(body);
      // source badge
      const badge = window.BearLM.sourceBadge[msg.source || "corpus"];
      if (badge) {
        const b = document.createElement("span");
        b.className = "src-badge " + badge.cls;
        b.textContent = badge.text;
        wrap.appendChild(b);
      }
      wrap.appendChild(body);
      // citations
      if (msg.citations && msg.citations.length) {
        const tags = document.createElement("div");
        tags.className = "tags msg-cites";
        msg.citations.forEach(function (c) { tags.appendChild(window.BearLM.citationTag(c, msg.source, ctx())); });
        wrap.appendChild(tags);
      }
    }
    scroll.appendChild(wrap);
    scroll.scrollTop = scroll.scrollHeight;
    return wrap;
  }

  function renderConversation(messages) {
    const scroll = $("chatScroll");
    scroll.innerHTML = "";
    if (!messages.length) {
      scroll.innerHTML = '<p class="chat-hint">New chat — ask anything about this project’s PDFs. The whole conversation is kept as context.</p>';
      return;
    }
    messages.forEach(appendMessage);
  }

  async function sendMessage(question) {
    if (busy || !question.trim()) return;
    // ensure a chat exists
    if (!currentChatId) { await newChat(); if (!currentChatId) return; }
    busy = true;
    const input = $("chatQ");
    input.value = "";
    // clear the "new chat" hint if present
    const hint = $("chatScroll").querySelector(".chat-hint");
    if (hint) hint.remove();
    const originalMsg = (chatLastOptimized && question === chatLastOptimized.optimized)
      ? chatLastOptimized.original : null;
    chatLastOptimized = null;
    appendMessage({ role: "user", content: question });
    const thinking = appendMessage({ role: "assistant", content: "_…thinking…_", source: null });
    thinking.classList.add("msg-thinking");
    try {
      const rewrite = !!($("chatRewriteToggle") && $("chatRewriteToggle").checked);
      const data = await jsonFetch("/api/projects/" + encodeURIComponent(currentSlug) + "/chats/" + currentChatId + "/ask", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: question, rewrite: rewrite }),
      });
      thinking.remove();
      appendMessage({
        role: "assistant", content: data.answer, citations: data.citations, source: data.source,
        provenance: { original: originalMsg, sent: originalMsg ? question : null, rewritten: data.rewritten_query || null },
      });
      refreshProject();  // update chat title/updated in sidebar
    } catch (e) {
      thinking.remove();
      appendMessage({ role: "assistant", content: "⚠️ " + e.message, source: "none" });
    } finally { busy = false; }
  }

  // --- stale-project prompt on open ------------------------------------------
  async function promptStaleProjects(projects) {
    let dismissed = {};
    try { dismissed = JSON.parse(sessionStorage.getItem("bearlm-stale-dismissed") || "{}"); } catch (e) {}
    for (const p of projects) {
      if (daysOld(p.created) < STALE_DAYS || dismissed[p.slug]) continue;
      const ok = confirm('Project "' + p.name + '" is over a week old.\n\nDelete it now? (Keeping it will not remove your PDFs.)');
      if (ok) {
        try { await jsonFetch("/api/projects/" + encodeURIComponent(p.slug), { method: "DELETE" }); } catch (e) {}
      } else {
        dismissed[p.slug] = true;
        try { sessionStorage.setItem("bearlm-stale-dismissed", JSON.stringify(dismissed)); } catch (e) {}
      }
    }
  }

  // --- wiring ----------------------------------------------------------------
  $("newProjectBtn").addEventListener("click", createProject);
  $("newProjectName").addEventListener("keydown", function (e) { if (e.key === "Enter") createProject(); });
  $("backToProjects").addEventListener("click", showProjectsList);
  $("addPdfBtn").addEventListener("click", function () { $("projPdfInput").click(); });
  $("projPdfInput").addEventListener("change", function () { addProjectPdf(this); });
  $("pdfViewToggle").addEventListener("click", togglePdfView);
  $("deleteProjectBtn").addEventListener("click", deleteProject);
  $("resetCtxBtn").addEventListener("click", resetContext);
  $("newChatBtn").addEventListener("click", newChat);
  $("chatForm").addEventListener("submit", function (e) { e.preventDefault(); sendMessage($("chatQ").value); });

  // Prompt optimizer for the chat input
  const chatImprove = $("chatImproveBtn"), chatSuggest = $("chatSuggestPanel"), chatInput = $("chatQ");
  if (chatImprove && chatSuggest) {
    chatImprove.addEventListener("click", async function () {
      const text = chatInput.value.trim();
      if (!text) { chatInput.focus(); return; }
      chatImprove.disabled = true;
      const label = chatImprove.textContent; chatImprove.textContent = "✨ Improving…";
      window.BearLM.suggestMessage(chatSuggest, "Optimizing your prompt…");
      try {
        const optimized = await window.BearLM.optimizePrompt(text);
        if (!optimized) chatSuggest.hidden = true;
        else window.BearLM.showSuggestion(chatSuggest, optimized, function (t) {
          chatInput.value = t; chatInput.focus();
          chatLastOptimized = { original: text, optimized: t };
        });
      } catch (e) {
        window.BearLM.suggestMessage(chatSuggest, e.message);
      } finally {
        chatImprove.disabled = false; chatImprove.textContent = label;
      }
    });
  }

  // --- init ------------------------------------------------------------------
  (async function () {
    const projects = await loadProjects();
    promptStaleProjects(projects);
  })();
})();
