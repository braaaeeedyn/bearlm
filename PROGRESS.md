# BearLM — Progress & Roadmap

_Last updated: 2026-09-17 (Stage 5 RAG core complete & tested)_

## ⭐ START HERE NEXT SESSION
Stages 4 and 5 are **done and working**. The corpus is ingested (2,241 chunks in
Atlas), the Vector Search index `vector_index` is **ACTIVE**, and `rag.py` returns
cited answers (smoke test passes). **Next up: Stage 6 — the Streamlit UI.**

1. **Build `app.py`** (new file, project root or `src/bearlm/`): a text box that
   calls `rag.answer()` and shows the answer + citations. Log each query to a new
   `bearlm.query_logs` collection (question, timestamp, courses cited) — this
   feeds the Stage 7 dashboard. Ask Claude to scaffold it the same guided way.
   - Checkpoint: `uv run streamlit run app.py` works locally.
2. Then **Stage 7 — Analytics dashboard** (read `query_logs`, chart with plotly).

**Uncommitted work in the tree:** the `.ipynb` ingestion fix in `doc_processor.py`
and the completed `rag.py`. Commit when convenient (Stage 5 complete).

**Quick re-test of what's built:**
```
uv run python -m bearlm.rag        # prints a cited answer to a sample question
```

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
| 3 | **Stage 4 — Ingestion pipeline** | ✅ **Done & run** — 2,241 chunks in Atlas |
| 4 | Atlas Vector Search index (`vector_index`) | ✅ **Done — ACTIVE** (1536 dims, cosine) |
| 5 | **Stage 5 — Retrieval + RAG core** (`rag.py`) | ✅ **Done & tested** — returns cited answers |
| 6 | Stage 6 — Basic UI (Streamlit) | 🔜 **NEXT** (no file yet) |
| 7 | Stage 7 — Analytics dashboard | ⬜ Not started |
| 8 | Stage 8 — Cloud deployment | ⬜ Not started |
| 9 | Stage 9 — Polish + README | ⬜ Not started |
| — | Frontend visual design / styling | ⬜ **Last** — deprioritized |

---

## What's done

- [x] `uv` project, Python 3.13, `.venv/`, all deps declared + locked (`uv sync`).
- [x] `.env` has real `OPENAI_API_KEY` + `MONGODB_URI` (gitignored).
- [x] Atlas free-tier cluster `BearLM-cluster` live; `bearlm.chunks` collection populated.
- [x] Corpus in `data/raw/`: cs61a, cs61b, cs70, cs188 (PDFs); data8, data100 (Jupyter Book repos).
- [x] `src/bearlm/config.py`, `db.py`, `doc_processor.py` — implemented & verified.
- [x] **`.ipynb` support added** to `doc_processor.py` (`extract_notebook`) — the
      Data 8 / Data 100 textbooks live in notebooks, not markdown; keeps markdown + code cells.
- [x] **Ingestion run** — `uv run python -m bearlm.doc_processor` inserted **2,241 chunks**:
      cs188 258 · cs61a 462 · cs61b 298 · cs70 429 · data100 422 · data8 372.
- [x] **Atlas Vector Search index `vector_index`** created & **ACTIVE** on `embedding`,
      1536 dims, cosine (JSON editor, "bring your own embeddings").
- [x] **`src/bearlm/rag.py`** implemented & tested — `answer()` returns a grounded
      answer + deduped citations; smoke test (`uv run python -m bearlm.rag`) passes.
- [x] First commit exists (`c389fe0`, repo `braaaeeedyn/bearlm`).
- [ ] **Uncommitted:** the `.ipynb` fix + `rag.py` — commit when convenient.

---

## What's next — in order

### ▶ Stage 6 — Basic UI  _(new file: `app.py` or `src/bearlm/app.py`)_ — NEXT
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
