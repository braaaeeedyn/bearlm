# BearLM — Project Context & Handoff

_Last updated: 2026-09-30. This is the source-of-truth overview of what BearLM is,
how it's built, and everything you need to know to run, extend, or explain it._

---

## 1. What it is

**BearLM** is a **fully local, zero-cost, grounded RAG (retrieval-augmented
generation) Q&A assistant** over UC Berkeley CS/Data-Science course materials,
plus a **Projects workspace** where users bring their own PDFs and hold grounded,
multi-turn conversations over them.

Everything runs on the user's machine — **no API keys, no cloud** — behind a
`BACKEND=local|openai` config flag. `local` is the default and primary path; the
original `openai` path (OpenAI + MongoDB Atlas) is retained behind the flag for
comparison but is not the focus.

Design goals: **grounded answers with citations, anti-hallucination, and a real
full-stack product** (web UI, analytics, evaluation), not just a script.

---

## 2. Stack

| Concern | Local backend (default) | OpenAI backend (legacy, behind flag) |
|---|---|---|
| Orchestration | LangChain | raw OpenAI SDK |
| Embeddings | Ollama `nomic-embed-text` (768-dim) | `text-embedding-3-small` (1536-dim) |
| Generator | Ollama `llama3.1:8b` (temp 0, num_ctx 8192) | `gpt-4o-mini` |
| Vector store | Chroma (on-disk `./db`) | MongoDB Atlas `$vectorSearch` |
| Keyword search | BM25 (`rank_bm25`) | — |
| Reranking | `bge-reranker-base` cross-encoder (sentence-transformers) | — |
| Chunking | token-based `RecursiveCharacterTextSplitter` (tiktoken) | word-window (legacy) |
| Eval judge | RAGAS with local Ollama judge | — |
| Backend | FastAPI + Uvicorn | same |
| Frontend | hand-built vanilla JS (marked, DOMPurify, KaTeX) — no framework | same |
| Analytics | SQLite (`data/analytics.db`) | MongoDB `query_logs` |

Python 3.13, managed with **uv**. Windows dev machine (RTX 4060, 8 GB VRAM).

---

## 3. Corpus

~**10,394 chunks across 10 courses**, 768-dim, stored in Chroma collection
`bearlm_chunks`:

`cs61a · cs61b · cs70 · cs188 · cs189 · data8 · data100 · data101 · data144 · genML`

Sources live in `data/raw/<course>/` (PDFs, Markdown, and Jupyter notebooks;
some are cloned repos). The `course` label is the folder name. Chunk ids are
`{course}:{n}`; metadata is `{course, section, source, chunk_id}`.

> Chunk-count history (for context): legacy word splitter = 2,241 (6 courses);
> token splitter = 2,581 (6 courses); current = 10,394 (10 courses).

---

## 4. Architecture / modules (`src/bearlm/`)

| File | Responsibility |
|---|---|
| `config.py` | `BACKEND` flag + **all** constants/env knobs (see §7) |
| `db.py` | MongoDB Atlas helpers (openai backend only) |
| `doc_processor.py` | extract (PDF/MD/ipynb) + chunk + OpenAI-embed + Atlas ingest |
| `text_splitter.py` | token-based `RecursiveCharacterTextSplitter` (tiktoken length fn) |
| `local_store.py` | Ollama embeddings + **multi-collection** Chroma, ingest (resilient batched), per-PDF ingest, collection drop/count, distinct courses |
| `search.py` | **hybrid search**: BM25 + dense fused with Reciprocal Rank Fusion; per-course filtering; BM25 index cached per collection |
| `rerank.py` | `bge-reranker-base` cross-encoder (`rerank`, `rerank_scored`) |
| `rag.py` | retrieval + `answer` (corpus) + `answer_grounded` (projects/uploads: confidence gate, project-first fallback, multi-turn history, corpus scope), **query rewriting**, **prompt optimizer**, meta-question bypass |
| `projects.py` | projects, chats (multi-turn), ad-hoc uploads, per-PDF collections, corpus-scope setting |
| `analytics.py` | SQLite query logging, aggregation/stats, most-asked area, grounded practice-question generation |
| `api.py` | FastAPI: all endpoints, static host, no-cache middleware, BM25 startup pre-warm |

**Frontend (`web/`)** — no framework/build:
`index.html` (Ask), `projects.html` (Projects app-shell), `analytics.html`,
`render.js` (shared: markdown/math/citations/provenance/optimizer),
`app.js`, `projects-app.js`, `analytics.js`, `styles.css`, `vendor/`
(marked, DOMPurify, KaTeX — all local). Static asset URLs are versioned (`?v=N`)
and served with `Cache-Control: no-cache`.

---

## 5. Features

**Ask page** (course corpus Q&A):
- Grounded answers with **deduplicated citations that deep-link to the exact PDF page** (`#page=N`).
- **Prompt optimizer** ("Improve prompt", suggest-then-confirm) and **query-rewrite** toggle.
- **Provenance**: shows "Your message" vs. "Optimized prompt sent" / "Retrieval query used".
- Rich rendering: **Markdown tables → HTML tables, fenced code → code boxes, LaTeX → KaTeX**.

**Projects** (the primary surface):
- Collapsible sidebar; projects listed newest→oldest.
- Per-project **reference PDFs** with **list & grid (thumbnail) views**, open-in-new-tab, and remove.
- **Multiple multi-turn chats** per project (persistent history; whole conversation fed back as context; follow-ups resolve pronouns).
- **Reset context** (clear chats, keep PDFs).
- **Per-project corpus scoping**: checkboxes choose which course subjects the project may fall back on (none = project-only).
- **Stale-project prompt**: projects >1 week old prompt for deletion on open.

**Ad-hoc PDF upload** ("Chat with a PDF"): single-PDF, strictly grounded, ephemeral.

**Anti-hallucination** (projects/uploads): reranker **confidence gate**
(`RERANK_MIN_SCORE=0.02`) drops weak matches and refuses when nothing clears the
bar; strict prompt with exact refusal; always-on citations; meta/summary
questions ("what's in this pdf") bypass the gate.

**Retrieval quality**: hybrid (BM25 + dense, RRF) + cross-encoder reranking; optional query rewriting.

**Analytics** (local, SQLite): usage stats, most-cited courses, per-day volume,
unanswered questions, **"needs the most help" area**, and **auto-generated,
grounded practice questions** for that area.

---

## 6. How to run

```bash
# 1. Install Ollama (https://ollama.com/download), then pull models
ollama pull llama3.1:8b          # generator (qwen2.5:3b if under ~8 GB VRAM)
ollama pull nomic-embed-text     # 768-dim embeddings

# 2. Python deps
uv sync                          # (or: pip install -r requirements.txt)

# 3. Configure (BACKEND=local is default; no keys needed)
cp .env.example .env

# 4. Ingest the corpus into Chroma
BACKEND=local uv run python -m bearlm.local_store           # token splitter (~10.4k chunks over data/raw)
BACKEND=local uv run python -m bearlm.local_store --verify  # expect dim 768

# 5. Run
BACKEND=local uv run uvicorn bearlm.api:app --reload        # http://localhost:8000
```

Pages: `/` (Ask), `/projects`, `/analytics`, `/api/docs` (FastAPI).

---

## 7. Config knobs (`config.py` / `.env`)

`BACKEND` (local|openai) · `OLLAMA_BASE_URL/CHAT_MODEL/EMBED_MODEL/NUM_CTX` ·
`CHROMA_DIR/COLLECTION` · `RETRIEVAL_K` · `RERANK_ENABLED/MODEL/CANDIDATES/TOP_N` ·
`RERANK_MIN_SCORE` (confidence gate, 0.02) · `HYBRID_SEARCH_ENABLED` (true) ·
`RRF_K` (60) · `QUERY_REWRITE_ENABLED` (false; per-request override) ·
`PROJECT_FALLBACK_TO_CORPUS` (true) · `MIN_RELEVANCE` (reserved) ·
`PROJECTS_DIR` · `UPLOADS_DIR` · `ANALYTICS_DB` ·
OpenAI backend only: `OPENAI_API_KEY`, `MONGODB_URI`.

---

## 8. Key HTTP endpoints

```
POST /api/ask                                    ask the course corpus
POST /api/optimize                               prompt optimizer (suggest)
GET  /api/courses                                live course list (from Chroma)
GET  /api/stats        POST /api/practice        analytics + generated practice Qs
POST/GET /api/projects   GET/DELETE /api/projects/{slug}
POST/DELETE /api/projects/{slug}/pdfs[/{file}]   add / remove a PDF
POST /api/projects/{slug}/corpus                 set corpus-fallback subjects
POST /api/projects/{slug}/reset                  clear chats, keep PDFs
POST /api/projects/{slug}/chats   GET/DELETE .../chats/{id}
POST /api/projects/{slug}/chats/{id}/ask         multi-turn grounded ask
POST/DELETE /api/uploads[/{id}]   POST .../{id}/ask
GET  /source/{course}/{file}                     serve a corpus PDF (deep-link)
GET  /projects/{slug}/pdfs/{file}[/thumb]        serve a project PDF / thumbnail
```

---

## 9. Data layout (all gitignored)

```
data/raw/<course>/        # corpus sources (PDF/MD/ipynb; some cloned repos)
data/projects/<slug>/     # project.json (pdfs, chats, corpus_courses) + pdfs/
data/uploads/<id>/        # ad-hoc upload PDFs
db/                       # Chroma persistent store (~150 MB)
data/analytics.db         # SQLite query log
eval/out/                 # eval artifacts (retrieval.json, answers.csv, ragas*.json)
.env                      # secrets (openai backend only)
```

---

## 10. Evaluation & results (`eval/`)

- `generate_questions.py` — builds a **50-question benchmark from real chunks** (each question written from a chunk, so its `(course, section, chunk_id)` is exact gold).
- `retrieval_eval.py` — **recall@1/3/5 + MRR ablation**: dense → hybrid → hybrid+rerank.
- `answer_eval.py` — end-to-end answers + latency → CSV for manual grading (`--limit N`).
- `ragas_eval.py` — **RAGAS faithfulness + answer_relevancy** with a **local Ollama judge** (needs a small `langchain_community.chat_models.vertexai` shim; memory-heavy — run with `HYBRID_SEARCH_ENABLED=false RERANK_ENABLED=false` to lighten).
- `report.py` — combined summary (retrieval ablation + latency/cost).

**Measured (real):**

| | dense | hybrid | hybrid + rerank |
|---|---|---|---|
| recall@1 | 0.56 | 0.60 | **0.82** |
| recall@5 | 0.76 | 0.84 | **0.86** |
| MRR | 0.64 | 0.70 | **0.84** |
| RAGAS faithfulness | 0.55 | — | **0.83** |
| RAGAS answer_relevancy | 0.80 | — | **0.97** |

Answer latency (8 GB RTX 4060): median ~6 s, p95 ~26 s (first-call warmup).
Cost: **$0/query** local. RAGAS runs are n=6 with a small local judge (defensible,
but that's the caveat if pressed).

---

## 11. Gotchas / things to know

- **First corpus query after restart builds the BM25 index (~40 s over ~10k docs)**; it's pre-warmed in a background thread at startup and cached thereafter.
- **Ingest wipes and rebuilds** the whole store (not incremental); a few minutes. Batched with retry/backoff to survive Windows socket exhaustion (`WSAENOBUFS`).
- **RAGAS** pulls a heavy dep tree (langchain 1.x, langgraph, pandas, pyarrow, datasets) and is memory-intensive — the app still imports fine, but run RAGAS with other apps closed / hybrid+rerank off on 8 GB.
- **Static assets** use `?v=N` + `no-cache`; bump `?v=` in the HTML when you must force a refetch.
- **Adding a course**: drop files in `data/raw/<course>/`, re-run `bearlm.local_store`.
- **Reranker/gate calibration**: bge scores are sigmoid [0,1]; off-topic ≈ 0.0, real matches ≥ ~0.04, so the gate is 0.02.
- Root also has legacy docs: `DESIGN.md` (Cohere-inspired design system), `PROGRESS.md`, `README.md` (user-facing).

---

## 12. Change history (newest first)

- **c0df613** Rich answer formatting — code blocks, LaTeX, Markdown tables (prompt + CSS + fence-unwrap).
- **fbfe0f6 / 69471bd** Per-project corpus scoping (choose fallback subjects) + `no-cache` static + `?v=` cache-bust.
- **27ad026** Prompt provenance (original vs optimized/rewritten shown with answers).
- **5a7caeb / d044e9d / f39b91d** RAGAS eval (local judge) + recorded results.
- **5fafb5b** Generated 50-Q benchmark + dense/hybrid/rerank recall/MRR ablation.
- **36f3b36** Local SQLite analytics + most-asked area + auto practice questions.
- **3da9eec** Hybrid search (BM25+dense RRF), query rewriting, prompt optimizer.
- **b248082** Projects: remove/open PDFs + list/grid thumbnail views (fixed chunk-id collision).
- **976620b** Projects: collapsible sidebar, multi-turn chats, reset context, stale-project prompt.
- **8df1d2e** Fixed grounded gate over-refusing (threshold 0.1→0.02 + meta-question bypass).
- **1201990 / 1a0fd34** Local RAG stack + Projects/uploads/anti-hallucination + docs.
- **bb0cfcf** (and earlier) Original OpenAI + MongoDB Atlas era: Stage 4 ingestion → Stage 5 RAG → Streamlit → FastAPI web app + analytics (2,241 chunks, 6 courses).

---

## 13. Known gaps / possible next steps

- **Answer-correctness grading** is still manual (RAGAS covers faithfulness/relevance; `answer_eval.py` exports a CSV for hand grading).
- **Deployment**: local models complicate PaaS; currently a local-run project (could containerize Ollama + app).
- **Not implemented** (from the technique wishlist): LangSmith tracing (cloud; cuts against local ethos), LoRA/QLoRA fine-tuning, an MCP server.
- Minor: `pyproject.toml` has a placeholder description and a dead `bearlm:main` script entry.
- File-picker/drag-drop PDF upload is not automation-testable (OS dialog), but the add-PDF API path is tested.
