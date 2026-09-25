# BearLM — Progress & Roadmap

_Last updated: 2026-09-25 (Stages 6–7 complete: FastAPI web app + analytics)_

## ⭐ START HERE NEXT SESSION
Stages 4–7 are **done and working**. The app is now a **FastAPI backend + a
hand-built HTML/CSS/JS frontend** (the old Streamlit app has been removed). Ask a
question, get a cited, LaTeX-rendered answer; an analytics page charts usage.

Run it locally:
```
uv run uvicorn bearlm.api:app --reload      # http://localhost:8000
```

**Next up: Stage 8 — deployment.** FastAPI does NOT fit Streamlit Community Cloud;
target Render / Railway / Fly.io (they run `uvicorn`). Add a Dockerfile or start
command, set `OPENAI_API_KEY` + `MONGODB_URI` as platform secrets, and allowlist
the host in Atlas Network Access.

---

BearLM is a RAG-based Q&A assistant grounded in Berkeley course materials. Ask a
question, get a cited answer (course + section). Stack: **FastAPI** + MongoDB Atlas
Vector Search + OpenAI (embeddings + chat), a custom light/dark web UI, and a
usage-analytics page. See `PROJECT_HANDOFF.md` for full background.

---

## Status at a glance

| # | Milestone | State |
|---|-----------|-------|
| 0 | Environment (uv, `.env`, Atlas cluster, deps) | ✅ Done |
| 1 | Corpus collected in `data/raw/` | ✅ Done (CS189 still pending) |
| 2 | Scaffolds written (`config`, `db`, `doc_processor`) | ✅ Done |
| 3 | **Stage 4 — Ingestion pipeline** | ✅ Done & run — 2,241 chunks in Atlas |
| 4 | Atlas Vector Search index (`vector_index`) | ✅ Done — ACTIVE (1536 dims, cosine) |
| 5 | **Stage 5 — Retrieval + RAG core** (`rag.py`) | ✅ Done & tested — cited answers |
| 6 | **Stage 6 — Web UI** (FastAPI + custom frontend) | ✅ **Done** — light/dark, LaTeX, cited answers |
| 7 | **Stage 7 — Analytics** (`/analytics`) | ✅ **Done** — tiles + SVG charts from `query_logs` |
| 8 | Stage 8 — Cloud deployment | 🔜 **NEXT** |
| 9 | Stage 9 — Polish + README | ⬜ Not started |

---

## Architecture (current)

**Backend — `src/bearlm/`**
- `config.py` — secrets (`.env`) + constants (models, chunk sizes, DB/collection names).
- `db.py` — Atlas client; `get_chunks_collection()`, `get_query_logs_collection()`, `ping()`.
- `doc_processor.py` — ingestion: extract PDF (`pdfplumber`) / markdown / `.ipynb`,
  chunk (~500 tok / ~50 overlap, word-approx), embed (`text-embedding-3-small`),
  insert into `bearlm.chunks`. Run: `uv run python -m bearlm.doc_processor`.
- `rag.py` — `retrieve()` ($vectorSearch, top-5) → `build_prompt()` → `answer()`
  returns `Answer(text, citations)`. `CHAT_MODEL = "gpt-4o-mini"`. The system prompt
  asks the model to wrap math in `$…$` / `$$…$$`.
- `api.py` — **FastAPI app** (`bearlm.api:app`). Serves the frontend and:
  - `POST /api/ask` → `{answer, citations}`, logs each question to `bearlm.query_logs`.
  - `GET /api/stats` → aggregated analytics (total, uncited, by-course, per-day, recent unanswered).
  - `GET /api/courses` → distinct corpus courses (feeds the dynamic footer).
  - `GET /` → `index.html`; `GET /analytics` → `analytics.html`.

**Frontend — `web/`** (no framework, no build step)
- `index.html` / `app.js` — Ask page: hero, pill search, suggestion chips, answer card,
  citation tags. Renders markdown (`marked`) + math (`KaTeX`), sanitized (`DOMPurify`).
  A post-processing step wraps the model's bare `( … )` / `[ … ]` LaTeX in `$…$`.
- `analytics.html` / `analytics.js` — stat tiles + hand-built **SVG** charts (course
  bars, per-day columns) + unanswered-questions list. No charting dependency.
- `styles.css` — the Cohere-inspired design system: light + dark via `[data-theme]`
  (☀/☾ toggle, persisted to `localStorage`), 22px cards, Space Grotesk / Inter /
  Space Mono, cool-gray borders, blue (`#1863dc`) only on interaction.
- `vendor/` — `marked`, `DOMPurify`, and full `KaTeX` (css/js/fonts) vendored
  **locally** — zero external runtime dependencies (deploy-safe). Only Google Fonts
  loads from the network.

**Data (MongoDB Atlas `bearlm`)**
- `chunks` — 2,241 embedded chunks; Vector Search index `vector_index` (1536, cosine), ACTIVE.
- `query_logs` — `{question, timestamp (UTC), courses: [...]}`, written per question.

---

## What's next — in order

### ▶ Stage 8 — Cloud deployment — NEXT
- Add a `Dockerfile` or platform start command running `uvicorn bearlm.api:app --host 0.0.0.0 --port $PORT`.
- Deploy on Render / Railway / Fly.io.
- Set `OPENAI_API_KEY` + `MONGODB_URI` as platform secrets (never commit `.env`).
- Atlas Network Access: allowlist the host (`0.0.0.0/0` if the platform has no fixed egress IP).
- ✅ Checkpoint: public URL answers a question end-to-end and shows analytics.

### Stage 9 — Polish + README
- Fill the empty `README.md`: what it is, stack, `uv sync`, run commands, screenshots, live URL.

---

## Adding more textbooks later
1. Drop file(s) into `data/raw/<course>/` (folder name → the `course` citation label).
2. Re-run `uv run python -m bearlm.doc_processor`.
3. The Atlas index auto-indexes new docs; the frontend course list updates via `/api/courses`.

Notes: only **PDF / `.md` / `.ipynb`** are extracted (other formats need a new
extractor); `ingest()` currently wipes + re-embeds the whole corpus each run (cheap
now, could be made incremental); Atlas M0 free tier is 512 MB (lots of headroom).

## Backlog / optional
- Source CS189 corpus (26 note PDFs) and re-run ingestion.
- Incremental ingestion (embed only new/changed files).
- `plotly` dependency is now unused (charts are hand-built SVG) — can be removed.
- Split markdown on headings for sharper citations; real `tiktoken` token counting.
