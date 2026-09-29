// BearLM Projects page: manage projects, add PDFs, project-scoped grounded chat,
// and a one-off "chat with a PDF" upload mode. Rendering is shared via render.js.
(function () {
  "use strict";

  const root = document.documentElement;
  const $ = function (id) { return document.getElementById(id); };

  // --- theme -----------------------------------------------------------------
  $("themeToggle").addEventListener("click", function () {
    const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("bearlm-theme", next); } catch (e) {}
  });

  // --- shared state ----------------------------------------------------------
  let currentSlug = null;
  let currentUploadId = null;
  let busy = false;

  // --- small UI helpers ------------------------------------------------------
  function loadingCard(el, label) {
    el.innerHTML =
      '<div class="card reveal"><div class="card-label">' + label + '</div>' +
      '<div class="skeleton"><span></span><span></span><span></span></div></div>';
  }
  function errorCard(el, message) {
    el.innerHTML =
      '<div class="card error reveal"><div class="card-label">Something went wrong</div>' +
      '<div class="answer-body"><p></p></div></div>';
    el.querySelector(".answer-body p").textContent = message;
  }
  async function jsonFetch(url, opts) {
    const resp = await fetch(url, opts);
    if (!resp.ok) {
      let detail = "Request failed (" + resp.status + ")";
      try { const j = await resp.json(); if (j.detail) detail = j.detail; } catch (e) {}
      throw new Error(detail);
    }
    return resp.json();
  }
  function show(el) { el.classList.remove("hidden"); }
  function hide(el) { el.classList.add("hidden"); }

  // --- view switching --------------------------------------------------------
  function showEmpty() {
    currentSlug = null;
    show($("projEmpty")); hide($("projView")); hide($("uploadView"));
    markActive();
  }
  function showProjectView() {
    hide($("projEmpty")); show($("projView")); hide($("uploadView"));
  }
  function showUploadView() {
    currentSlug = null;
    hide($("projEmpty")); hide($("projView")); show($("uploadView"));
    markActive();
  }

  // --- sidebar / project list ------------------------------------------------
  function markActive() {
    document.querySelectorAll(".proj-item").forEach(function (n) {
      n.classList.toggle("is-active", n.dataset.slug === currentSlug);
    });
  }

  async function loadProjects() {
    const list = $("projectList");
    try {
      const data = await jsonFetch("/api/projects");
      list.innerHTML = "";
      if (!data.projects.length) {
        list.innerHTML = '<p class="proj-list-empty">No projects yet.</p>';
        return;
      }
      data.projects.forEach(function (p) {
        const item = document.createElement("button");
        item.type = "button";
        item.className = "proj-item";
        item.dataset.slug = p.slug;
        item.innerHTML =
          '<span class="proj-item-name"></span>' +
          '<span class="proj-item-meta">' + (p.pdfs.length) + ' PDF' + (p.pdfs.length === 1 ? '' : 's') + '</span>';
        item.querySelector(".proj-item-name").textContent = p.name;
        item.addEventListener("click", function () { selectProject(p.slug); });
        list.appendChild(item);
      });
      markActive();
    } catch (e) {
      list.innerHTML = '<p class="proj-list-empty">Couldn\'t load projects. (' + e.message + ')</p>';
    }
  }

  async function createProject() {
    const input = $("newProjectName");
    const name = input.value.trim();
    if (!name || busy) return;
    busy = true;
    try {
      const p = await jsonFetch("/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name }),
      });
      input.value = "";
      await loadProjects();
      selectProject(p.slug);
    } catch (e) {
      alert(e.message);
    } finally {
      busy = false;
    }
  }

  // --- project detail --------------------------------------------------------
  async function selectProject(slug) {
    currentSlug = slug;
    currentUploadId = null;
    showProjectView();
    markActive();
    $("projResult").innerHTML = "";
    $("projQ").value = "";
    await refreshProject();
  }

  async function refreshProject() {
    if (!currentSlug) return;
    try {
      const p = await jsonFetch("/api/projects/" + encodeURIComponent(currentSlug));
      $("projName").textContent = p.name;
      renderDocList(p.pdfs);
    } catch (e) {
      errorCard($("projResult"), e.message);
    }
  }

  function renderDocList(pdfs) {
    const wrap = $("projDocList");
    wrap.innerHTML = "";
    if (!pdfs.length) {
      wrap.innerHTML = '<p class="proj-doc-empty">No PDFs yet — add one to ground this project.</p>';
      return;
    }
    pdfs.forEach(function (d) {
      const row = document.createElement("div");
      row.className = "proj-doc";
      row.innerHTML =
        '<span class="proj-doc-name"></span>' +
        '<span class="proj-doc-meta">' + d.chunks + ' chunks</span>';
      row.querySelector(".proj-doc-name").textContent = d.filename;
      wrap.appendChild(row);
    });
  }

  async function addProjectPdf(fileInput) {
    const file = fileInput.files && fileInput.files[0];
    if (!file || !currentSlug || busy) return;
    busy = true;
    const wrap = $("projDocList");
    const note = document.createElement("p");
    note.className = "proj-doc-empty";
    note.textContent = "Embedding " + file.name + "… this can take a moment.";
    wrap.prepend(note);
    try {
      const form = new FormData();
      form.append("file", file);
      await jsonFetch("/api/projects/" + encodeURIComponent(currentSlug) + "/pdfs", {
        method: "POST", body: form,
      });
      await refreshProject();
      await loadProjects();
    } catch (e) {
      errorCard($("projResult"), e.message);
    } finally {
      fileInput.value = "";
      busy = false;
    }
  }

  async function deleteProject() {
    if (!currentSlug || busy) return;
    if (!confirm("Delete this project and all its PDFs? This can't be undone.")) return;
    busy = true;
    try {
      await jsonFetch("/api/projects/" + encodeURIComponent(currentSlug), { method: "DELETE" });
      showEmpty();
      await loadProjects();
    } catch (e) {
      alert(e.message);
    } finally {
      busy = false;
    }
  }

  async function askProject(question) {
    if (!currentSlug || busy || !question.trim()) return;
    busy = true;
    loadingCard($("projResult"), "Searching your project");
    try {
      const data = await jsonFetch("/api/projects/" + encodeURIComponent(currentSlug) + "/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question }),
      });
      window.BearLM.renderResult($("projResult"), data, { slug: currentSlug, showBadge: true });
    } catch (e) {
      errorCard($("projResult"), e.message);
    } finally {
      busy = false;
    }
  }

  // --- quick upload mode -----------------------------------------------------
  async function doUpload(file) {
    if (!file || busy) return;
    busy = true;
    loadingCard($("uploadResult"), "Reading " + file.name);
    try {
      const form = new FormData();
      form.append("file", file);
      const data = await jsonFetch("/api/uploads", { method: "POST", body: form });
      currentUploadId = data.upload_id;
      const badge = $("uploadName");
      badge.textContent = data.filename + " · " + data.chunks + " chunks";
      badge.hidden = false;
      $("uploadResult").innerHTML = "";
      show($("uploadAskForm"));
      $("uploadQ").focus();
    } catch (e) {
      errorCard($("uploadResult"), e.message);
    } finally {
      busy = false;
    }
  }

  async function askUpload(question) {
    if (!currentUploadId || busy || !question.trim()) return;
    busy = true;
    loadingCard($("uploadResult"), "Reading your PDF");
    try {
      const data = await jsonFetch("/api/uploads/" + encodeURIComponent(currentUploadId) + "/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question }),
      });
      window.BearLM.renderResult($("uploadResult"), data, { uid: currentUploadId, showBadge: true });
    } catch (e) {
      errorCard($("uploadResult"), e.message);
    } finally {
      busy = false;
    }
  }

  // --- wiring ----------------------------------------------------------------
  $("newProjectBtn").addEventListener("click", createProject);
  $("newProjectName").addEventListener("keydown", function (e) { if (e.key === "Enter") createProject(); });

  $("addPdfBtn").addEventListener("click", function () { $("projPdfInput").click(); });
  $("projPdfInput").addEventListener("change", function () { addProjectPdf(this); });
  $("deleteProjectBtn").addEventListener("click", deleteProject);
  $("projAskForm").addEventListener("submit", function (e) { e.preventDefault(); askProject($("projQ").value); });

  $("quickPdfBtn").addEventListener("click", showUploadView);
  $("uploadPickBtn").addEventListener("click", function () { $("uploadInput").click(); });
  $("uploadInput").addEventListener("change", function () { if (this.files[0]) doUpload(this.files[0]); });
  $("uploadAskForm").addEventListener("submit", function (e) { e.preventDefault(); askUpload($("uploadQ").value); });

  // drag & drop for the quick-upload dropzone
  const drop = $("uploadDrop");
  ["dragover", "dragenter"].forEach(function (ev) {
    drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.add("is-drag"); });
  });
  ["dragleave", "drop"].forEach(function (ev) {
    drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.remove("is-drag"); });
  });
  drop.addEventListener("drop", function (e) {
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) doUpload(f);
  });

  // --- init ------------------------------------------------------------------
  loadProjects();
})();
