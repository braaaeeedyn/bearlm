# BearLM — Project Handoff / Context

## What this project is
BearLM: a RAG-based Q&A assistant grounded in Berkeley course materials. Users ask
questions and get cited answers (course + section). Built on **FastAPI + MongoDB
Atlas Vector Search + OpenAI**, with a custom light/dark web UI and a usage-analytics
page.

**Resume goals this project hits:** RAG, MongoDB, cloud deployment, data visualization.

## Current status: Stages 6–7 complete (of 9)
- [x] Stage 0: Repo, venv (uv), MongoDB Atlas cluster, OpenAI key.
- [x] Stage 1–2: Corpus in `data/raw/` (CS61A, CS61B, CS70, CS188, Data 8, Data 100). CS189 pending.
- [x] Stage 4: Ingestion — `doc_processor.py` inserted **2,241 chunks**; index `vector_index` ACTIVE.
- [x] Stage 5: RAG core — `rag.py`'s `answer()` returns grounded answer + citations.
- [x] Stage 6: **Web UI** — FastAPI (`api.py`) + custom frontend (`web/`); light/dark,
      LaTeX rendering, cited answers. **(Replaced the earlier Streamlit app, now removed.)**
- [x] Stage 7: **Analytics** — `/analytics` page: stat tiles + SVG charts from `query_logs`.
- [ ] Stage 8: **Cloud deployment — NEXT** (Render / Railway / Fly.io; not Streamlit Cloud).
- [ ] Stage 9: Polish + README (`README.md` is still empty).

## Run it
```
uv sync                                   # install deps into .venv/
uv run uvicorn bearlm.api:app --reload    # http://localhost:8000  (Ask + /analytics)
uv run python -m bearlm.doc_processor     # (re)ingest the corpus
uv run python -m bearlm.rag               # RAG smoke test in the terminal
```

## Environment
- Python 3.13 via `uv` (venv at `.venv/`), Windows / Git Bash.
- Key deps (`pyproject.toml`): `fastapi`, `uvicorn[standard]`, `openai`, `pdfplumber`,
  `pymongo[srv]`, `pymupdf`, `python-dotenv`. (`streamlit` removed; `plotly` now unused.)
- `.env` holds `OPENAI_API_KEY` + `MONGODB_URI` (real values, gitignored — never commit).
- MongoDB Atlas free-tier cluster `BearLM-cluster`; database `bearlm` with `chunks`
  (2,241 docs, Vector Search index `vector_index`, 1536-dim cosine) and `query_logs`.
- Package layout: source in `src/bearlm/`; frontend static files in `web/`.

## Architecture
**Backend `src/bearlm/`:** `config.py` (secrets + constants), `db.py` (Atlas helpers),
`doc_processor.py` (PDF/markdown/`.ipynb` → chunk → embed → `chunks`), `rag.py`
(`retrieve` → `build_prompt` → `answer`; `gpt-4o-mini`; prompt asks for `$…$` math),
`api.py` (FastAPI: `POST /api/ask`, `GET /api/stats`, `GET /api/courses`, serves `web/`).

**Frontend `web/`** (no framework/build): `index.html`+`app.js` (Ask page), `analytics.html`
+`analytics.js` (tiles + hand-built SVG charts), `styles.css` (Cohere-inspired design
system, light/dark via `[data-theme]`, ☀/☾ toggle), `vendor/` (`marked`, `DOMPurify`,
`KaTeX` — all local). Design source of truth: `design.md`.

## Corpus (in `data/raw/`)
| Course | Format | Location |
|---|---|---|
| CS61A | PDF | `data/raw/cs61a/61a_txtbk.pdf` |
| CS61B | PDF | `data/raw/cs61b/61b_txtbk.pdf` |
| CS70 | PDF | `data/raw/cs70/cs70_txtbk.pdf` |
| CS188 | PDF | `data/raw/cs188/cs188_txtbk.pdf` |
| Data 8 | Jupyter/MyST repo | `data/raw/data8/` |
| Data 100 | Jupyter/MyST repo | `data/raw/data100/` |

CS189 (26 note PDFs) not yet sourced. Adding a course: drop files in
`data/raw/<course>/`, re-run `doc_processor` (only PDF / `.md` / `.ipynb` are extracted).

## Immediate next step (Stage 8 — Deploy)
1. Add a `Dockerfile` or platform start command: `uvicorn bearlm.api:app --host 0.0.0.0 --port $PORT`.
2. Deploy to Render / Railway / Fly.io.
3. Set `OPENAI_API_KEY` + `MONGODB_URI` as platform secrets.
4. Allowlist the host in Atlas Network Access (`0.0.0.0/0` if no fixed egress IP).
5. Checkpoint: public URL answers a question end-to-end and renders analytics.

## Notes
- Don't touch `.env` values; reference `.env.example` for structure.
- `ingest()` wipes and re-embeds the whole corpus on each run (cheap now; could be incremental).
- Keep the design consistent with `design.md` (Cohere-inspired; blue only on interaction).
