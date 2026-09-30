// BearLM analytics: theme toggle + stat tiles + hand-built SVG charts.
(function () {
  "use strict";

  const root = document.documentElement;

  // --- Theme toggle (shared behavior with the Ask page) ----------------------
  const toggle = document.getElementById("themeToggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("bearlm-theme", next); } catch (e) {}
    });
  }

  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  function emptyNote(msg) {
    return '<p class="empty-note">' + esc(msg) + "</p>";
  }

  // --- Stat tiles ------------------------------------------------------------
  function renderTiles(stats) {
    const answered = stats.total - stats.uncited;
    const tiles = [
      { value: stats.total, label: "Questions asked" },
      { value: stats.courses_covered, label: "Courses cited" },
      { value: answered, label: "Answered from corpus" },
      { value: stats.uncited, label: "No source found" },
    ];
    document.getElementById("tiles").innerHTML = tiles
      .map(function (t) {
        return (
          '<div class="tile"><div class="tile-value">' +
          esc(t.value) +
          '</div><div class="tile-label">' +
          esc(t.label) +
          "</div></div>"
        );
      })
      .join("");
  }

  // --- Horizontal bar chart: most-cited courses ------------------------------
  function renderCourseChart(rows) {
    const el = document.getElementById("courseChart");
    if (!rows.length) { el.innerHTML = emptyNote("No citations logged yet."); return; }

    const W = 680, rowH = 40, labelW = 116, valueW = 40;
    const barMax = W - labelW - valueW - 16;
    const max = Math.max.apply(null, rows.map(function (r) { return r.count; }));
    const H = rows.length * rowH;

    let svg = '<svg class="svg-chart" viewBox="0 0 ' + W + " " + H +
      '" role="img" preserveAspectRatio="xMinYMin meet">';
    rows.forEach(function (r, i) {
      const y = i * rowH;
      const len = max ? Math.max((r.count / max) * barMax, 2) : 2;
      const cy = y + rowH / 2;
      svg +=
        '<text class="chart-label" x="0" y="' + cy + '" dominant-baseline="middle">' + esc(r.course) + "</text>" +
        '<rect class="bar" x="' + labelW + '" y="' + (y + 8) + '" width="' + len + '" height="' + (rowH - 16) + '" rx="4">' +
        "<title>" + esc(r.course) + ": " + esc(r.count) + " question(s)</title></rect>" +
        '<text class="chart-value" x="' + (labelW + len + 8) + '" y="' + cy + '" dominant-baseline="middle">' + esc(r.count) + "</text>";
    });
    svg += "</svg>";
    el.innerHTML = svg;
  }

  // --- Column chart: questions per day ---------------------------------------
  function renderDayChart(rows) {
    const el = document.getElementById("dayChart");
    if (!rows.length) { el.innerHTML = emptyNote("No questions logged yet."); return; }

    const W = 680, H = 220, padB = 30, padT = 14, gap = 8;
    const n = rows.length;
    const colW = Math.min((W - (n - 1) * gap) / n, 72);
    const totalW = n * colW + (n - 1) * gap;
    const x0 = (W - totalW) / 2;
    const max = Math.max.apply(null, rows.map(function (r) { return r.count; }));
    const plotH = H - padB - padT;

    let svg = '<svg class="svg-chart" viewBox="0 0 ' + W + " " + H +
      '" role="img" preserveAspectRatio="xMidYMin meet">';
    rows.forEach(function (r, i) {
      const x = x0 + i * (colW + gap);
      const h = max ? Math.max((r.count / max) * plotH, 2) : 2;
      const y = H - padB - h;
      const short = r.date.slice(5); // MM-DD
      svg +=
        '<rect class="bar" x="' + x + '" y="' + y + '" width="' + colW + '" height="' + h + '" rx="4">' +
        "<title>" + esc(r.date) + ": " + esc(r.count) + " question(s)</title></rect>" +
        '<text class="chart-value" x="' + (x + colW / 2) + '" y="' + (y - 6) + '" text-anchor="middle">' + esc(r.count) + "</text>" +
        '<text class="chart-axis" x="' + (x + colW / 2) + '" y="' + (H - padB + 16) + '" text-anchor="middle">' + esc(short) + "</text>";
    });
    svg += "</svg>";
    el.innerHTML = svg;
  }

  // --- Most-asked area ("needs the most help") + practice questions ----------
  function renderFocus(stats) {
    const panel = document.getElementById("focusPanel");
    const area = stats.top_area;
    if (!area || !area.course) { panel.hidden = true; return; }
    panel.hidden = false;
    document.getElementById("focusCourse").textContent = area.course;
    let sub = area.count + " question" + (area.count === 1 ? "" : "s") +
      " · " + area.share + "% of answered questions";
    if (area.top_units && area.top_units.length) {
      sub += " · hot spots: " + area.top_units.map(function (u) { return u.section; }).join(", ");
    }
    document.getElementById("focusSub").textContent = sub;
  }

  async function generatePractice() {
    const btn = document.getElementById("practiceBtn");
    const out = document.getElementById("practice");
    btn.disabled = true;
    const label = btn.textContent;
    btn.textContent = "Generating…";
    out.innerHTML = '<p class="empty-note">Writing practice questions from the most-asked area… (a few seconds)</p>';
    try {
      const res = await (await fetch("/api/practice", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ n: 5 }),
      })).json();
      if (!res.questions || !res.questions.length) {
        out.innerHTML = emptyNote(res.message || "No questions could be generated yet.");
      } else {
        out.innerHTML =
          '<div class="practice-head">Practice — ' + esc(res.area) + '</div><ol class="practice-list">' +
          res.questions.map(function (qa) {
            return '<li><div class="practice-q">' + esc(qa.question) + '</div>' +
              '<details class="practice-a"><summary>Show answer</summary><div>' + esc(qa.answer) + '</div></details></li>';
          }).join("") + "</ol>";
      }
    } catch (e) {
      out.innerHTML = emptyNote("Couldn't generate practice questions.");
    } finally {
      btn.disabled = false;
      btn.textContent = label;
    }
  }

  function renderUnanswered(list) {
    const el = document.getElementById("unanswered");
    if (!list.length) {
      el.innerHTML = emptyNote("Every logged question found a source. 🎯");
      return;
    }
    el.innerHTML =
      '<ul class="gap-list">' +
      list.map(function (q) { return "<li>" + esc(q) + "</li>"; }).join("") +
      "</ul>";
  }

  // --- Load ------------------------------------------------------------------
  (async function () {
    try {
      const stats = await (await fetch("/api/stats")).json();
      renderTiles(stats);
      renderFocus(stats);
      renderCourseChart(stats.by_course || []);
      renderDayChart(stats.per_day || []);
      renderUnanswered(stats.recent_unanswered || []);
      const pb = document.getElementById("practiceBtn");
      if (pb) pb.addEventListener("click", generatePractice);
    } catch (e) {
      document.getElementById("tiles").innerHTML = emptyNote(
        "Couldn't load analytics. Is the server running?"
      );
    }
  })();

  // Footer corpus list (mirrors the Ask page).
  (async function () {
    const elc = document.getElementById("corpusList");
    if (!elc) return;
    try {
      const data = await (await fetch("/api/courses")).json();
      if (data.courses && data.courses.length) elc.textContent = data.courses.join(" · ");
    } catch (e) {}
  })();
})();
