# BearLM — Progress & Roadmap

_Last updated: 2026-09-09 (Stage 4 code written & verified)_

## ⭐ START HERE NEXT SESSION
Stage 4 code is **written and verified** (compiles clean; `db.ping()` returns
`True` against live Atlas). Not yet run. Do these, in order:

1. **Run the ingestion pipeline** (spends OpenAI embedding credits, writes to Atlas):
   ```
   uv run python -m bearlm.doc_processor
   ```
   Expect per-course chunk counts then a total. cs188 is a 32 MB PDF — it's slow.
2. **Create the Atlas Vector Search index** (manual, in the Atlas web UI):
   collection `bearlm.chunks`, field `embedding`, **1536** dims, **cosine**.
3. Then begin **Stage 5 — Retrieval + RAG** (new file `src/bearlm/rag.py`; ask
   Claude to scaffold it the same guided way).

---


A RAG-based Q&A assistant grounded in Berkeley course materials. Ask a question,
get a cited answer (course + section). Stack: MongoDB Atlas Vector Search +
OpenAI embeddings, Streamlit UI, analytics dashboard, cloud deploy.

See `PROJECT_HANDOFF.md` for full background. This file tracks **what's done and
what's next, in order.**

---

## Status at a glance

| # | Milestone | State |
|---|-----------|-------|
| 0 | Environment (uv, `.env`, Atlas cluster, deps) | ✅ Done |
| 1 | Corpus collected in `data/raw/` | ✅ Done (CS189 still pending) |
| 2 | Scaffolds written (`config`, `db`, `doc_processor`) | ✅ Done |
| 3 | **Stage 4 — Ingestion pipeline** | ✅ Code done & verified — 🔜 **RUN IT** (see top) |
| 4 | Stage 5 — Retrieval + RAG core | ⬜ Not started (no file yet) |
| 5 | Atlas Vector Search index (manual, in UI) | ⬜ After ingestion runs |
| 6 | Stage 6 — Basic UI (Streamlit) | ⬜ Not started |
| 7 | Stage 7 — Analytics dashboard | ⬜ Not started |
| 8 | Stage 8 — Cloud deployment | ⬜ Not started |
| 9 | Stage 9 — Polish + README | ⬜ Not started |
| — | Frontend visual design / styling | ⬜ **Last** — deprioritized |

---

## What's done

- [x] `uv` project, Python 3.13, `.venv/`, all deps declared + locked (`uv sync`).
- [x] `.env` has real `OPENAI_API_KEY` + `MONGODB_URI` (gitignored).
- [x] Atlas free-tier cluster `BearLM-cluster` live. DB/collections not yet created.
- [x] Corpus in `data/raw/`: cs61a, cs61b, cs70, cs188 (PDFs); data8, data100 (markdown repos).
- [x] `src/bearlm/config.py`, `db.py`, `doc_processor.py` — **implemented by hand
      and verified** (all compile; `db.ping()` → `True` against live Atlas).
- [ ] Ingestion **not yet run** — `bearlm.chunks` collection is still empty.
- [ ] Atlas Vector Search index **not yet created**.
- [ ] **Nothing committed to git yet** — make the first commit when convenient.

---

## What's next — in order

### ▶ Stage 4 — Ingestion pipeline  (code DONE — just needs running)
- [x] `config.py`, `db.py`, `doc_processor.py` implemented & verified.
- [ ] **Run** `uv run python -m bearlm.doc_processor` → fills `bearlm.chunks`.
- [ ] **Create Atlas Vector Search index** on `embedding`, 1536 dims, cosine
      (manual, Atlas UI). Only after chunks exist.

### Stage 5 — Retrieval + RAG core  _(new file: `src/bearlm/rag.py`)_
- Embed the user's question (reuse `embed_batch`).
- `$vectorSearch` aggregation against `bearlm.chunks` → top-k chunks.
- Build a prompt: system instruction + retrieved chunks + question.
- Call the chat model; return answer **plus citations** (course + section from metadata).
- ✅ Checkpoint: a hard-coded question in `__main__` returns a cited answer in the terminal.

### Stage 6 — Basic UI  _(new file: `app.py` or `src/bearlm/app.py`)_
- Streamlit: text box → call `rag.answer()` → show answer + citations.
- Log each query to `bearlm.query_logs` (question, timestamp, which courses were cited).
- ✅ Checkpoint: `uv run streamlit run app.py` works locally.

### Stage 7 — Analytics dashboard  _(Streamlit page / `plotly`)_
- Read `query_logs`; charts: queries over time, most-cited courses, etc.

### Stage 8 — Cloud deployment
- Streamlit Community Cloud or Render. Set `OPENAI_API_KEY` + `MONGODB_URI` as
  platform secrets (don't commit `.env`). Whitelist the host IP in Atlas.

### Stage 9 — Polish + README
- Fill in the empty `README.md`: what it is, setup, `uv sync`, run commands, screenshots.

### LAST — Frontend visual design (deprioritized)
- Styling, theming, layout polish on the Streamlit app. Functionality first;
  make it look good only once everything works.

---

## Backlog / optional
- Source CS189 corpus (26 note PDFs) and re-run ingestion.
- Future-course placeholders: CS61C, Data 88S, Data 101, Data 140.
- Upgrade markdown extraction to split on headings for sharper citations.
- Swap word-count chunking for real `tiktoken` token counting.
