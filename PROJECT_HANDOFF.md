# BearLM — Project Handoff / Context for Claude Code

## What this project is
BearLM: a RAG-based Q&A assistant grounded in Berkeley course materials. Users ask questions and get cited answers (course + section). Built with MongoDB Atlas Vector Search, deployed to the cloud, with a visualization dashboard for query analytics.

**Resume goals this project hits:** RAG, MongoDB, cloud deployment, data visualization.

**Reference/inspiration only:** an open-source NotebookLM clone (Milvus + Zep + audio/podcast features) — NOT being used directly. Only borrowing architectural ideas (chunking pattern, RAG pipeline structure). This build uses MongoDB instead of Milvus, a fixed corpus instead of user uploads, and skips audio/podcast/memory-layer features entirely.

## Current status: Stage 5 of 9 complete (see full plan below)
- [x] Stage 0: Repo, venv (uv), MongoDB Atlas cluster, OpenAI key — all set up
- [x] Stage 1–2: Corpus collected into `data/raw/` (CS61A, CS61B, CS70, CS188, Data 8, Data 100). CS189 still pending.
- [x] Stage 4: Ingestion pipeline — **done & run**. `doc_processor.py` (incl. `.ipynb`
      support) inserted 2,241 chunks into `bearlm.chunks`. Vector Search index
      `vector_index` is ACTIVE (embedding field, 1536 dims, cosine).
- [x] Stage 5: Retrieval + RAG core — **done & tested**. `rag.py`'s `answer()`
      returns a grounded answer + citations; smoke test passes.
- [ ] Stage 6: Basic UI (Streamlit) — **NEXT STEP** (`app.py`; log queries to `query_logs`)
- [ ] Stage 7: Visualization dashboard (read `query_logs`, plotly)
- [ ] Stage 8: Cloud deployment
- [ ] Stage 9: Polish + README
- [ ] Backlog: CS189 corpus; future-course placeholders (CS61C, Data 88S, Data 101, Data 140)

## Environment
- Python 3.13 via `uv` (venv at `.venv/`), Windows machine using Git Bash (MINGW64)
- Dependencies are already declared in `pyproject.toml` and locked in `uv.lock`: `openai`, `pdfplumber`, `plotly`, `pymongo[srv]`, `pymupdf`, `python-dotenv`, `streamlit`. Run `uv sync` to install into `.venv/`. (Note: `pymupdf` is included in addition to `pdfplumber` — either can be used for PDF extraction.)
- `.env` already has `OPENAI_API_KEY` and `MONGODB_URI` filled in (real values, gitignored)
- MongoDB Atlas: free-tier cluster `BearLM-cluster` is active, database not yet created (need to create `bearlm` database + `chunks` and `query_logs` collections in Stage 4/6)
- **Git: no commits yet.** Branch `main` exists but has zero commits — every file is currently untracked. First commit still needs to be made.
- **Package layout:** source lives in the uv package `src/bearlm/` (see `[project.scripts] bearlm = "bearlm:main"` in `pyproject.toml`). New modules go under `src/bearlm/`, not the top of `src/`.

## Corpus collected (in `data/raw/`)
| Course | Format | Location |
|---|---|---|
| CS61A | PDF | `data/raw/cs61a/61a_txtbk.pdf` |
| CS61B | PDF | `data/raw/cs61b/61b_txtbk.pdf` |
| CS70 | PDF | `data/raw/cs70/cs70_txtbk.pdf` |
| CS188 | PDF | `data/raw/cs188/cs188_txtbk.pdf` |
| Data 8 | Markdown/MyST (cloned repo) | `data/raw/data8/chapters/` |
| Data 100 | Markdown/MyST (cloned repo) | `data/raw/data100/content/` |

Note: CS70 was added opportunistically (had the file) — not in the original 6-course MVP scope but fine to include.

Future/not-yet-implemented courses (per plan): CS61C, Data 88S, Data 101, Data 140.

## Immediate next step (Stage 6: Basic UI — Streamlit)
Build `app.py` (project root or `src/bearlm/app.py`) that:
1. Shows a text box for the user's question
2. Calls `rag.answer(question)` → gets back `Answer(text, citations)`
3. Displays the answer and its citations (course + section)
4. Logs each query to a new `bearlm.query_logs` collection:
   `{question, timestamp, courses: [...cited courses...]}` — this data feeds Stage 7
5. Checkpoint: `uv run streamlit run app.py` runs it locally in the browser

### Already built (Stages 4–5), for reference
- `doc_processor.py`: extracts PDF (`pdfplumber`) + markdown + **notebooks** (`.ipynb`),
  chunks (~500 tok / ~50 overlap, word-approximated), embeds (`text-embedding-3-small`),
  inserts `{text, embedding, course, filename, section}`. Run once → 2,241 chunks.
- Atlas Vector Search index `vector_index` on `embedding`, 1536 dims, cosine — ACTIVE.
- `rag.py`: `retrieve()` ($vectorSearch, top-5) → `build_prompt()` → `answer()` returns
  a cited answer. Uses `CHAT_MODEL = "gpt-4o-mini"`. Re-test: `uv run python -m bearlm.rag`.

## Full stage plan reference
1. Core corpus (CS61A, CS61B, Data 8, CS188) — done (CS70 also added opportunistically)
2. Data 100 corpus — done. CS189 corpus — NOT done (still needs sourcing; 26 separate note PDFs)
3. Future courses placeholder (CS61C, Data 88S, Data 101, Data 140) — not started
4. Ingestion pipeline — **done & run** (2,241 chunks; `vector_index` ACTIVE)
5. Retrieval + RAG core — **done & tested** (`rag.py`)
6. Basic UI (Streamlit) — **next**
7. Visualization dashboard
8. Cloud deployment (Streamlit Community Cloud or Render)
9. Polish + README

## Naming
Project name: **BearLM** (Cal + LM, subject area implied by CS/DS course scope).

## Notes for Claude Code
- Don't touch `.env` values, just read/reference `.env.example` structure
- CS 189 (26 separate note PDFs) still needs to be sourced — not yet downloaded
- Keep ingestion code modular per format (PDF extractor vs. Markdown extractor) since corpus mixes both
