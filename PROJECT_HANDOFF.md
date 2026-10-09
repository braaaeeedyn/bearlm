# BearLM — Project Context & Handoff

_Last updated: 2026-10-08. This is the source-of-truth overview of what BearLM is,
how it's built, and everything you need to know to run, extend, or explain it._

---

## 1. What it is

**BearLM** is a **fully local, zero-cost, grounded RAG (retrieval-augmented
generation) Q&A assistant** over UC Berkeley CS/Data-Science course materials,
plus a **Projects workspace** where users bring their own PDFs and hold grounded,
multi-turn conversations over them.

Everything runs on the user's machine — **no API keys, no cloud** — behind a
`BACKEND=local|openai` config flag. `local` is the primary path and the value
`.env.example` sets; the code default in `config.py` is `openai`, so without a
`.env` you must set `BACKEND=local` yourself. The original `openai` path (OpenAI + MongoDB Atlas) is retained behind the flag for
comparison but is not the focus.

Design goals: **grounded answers with citations, anti-hallucination, and a real
full-stack product** (web UI, analytics, evaluation), not just a script.

---

## 2. Stack

| Concern | Local backend (primary) | OpenAI backend (legacy, behind flag) |
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
- **Reset context** (clear chats and retrieval memory, keep PDFs).
- **Retrieval memory**: chunk ids behind past answers (in `project.json`) are re-added to the rerank pool for follow-ups; they still must clear the gate.
- **Near-repeat reuse**: opt-in, off by default (`NEAR_REPEAT_THRESHOLD=0`). Follow-ups are retrieved as `previous turn + follow-up`, which embeds up to 0.977 cosine from the previous turn, so no safe threshold separates them from true repeats (`eval/speed_experiments.json`). When set > 0, a question at least that cosine-similar to an earlier one in the same project reuses its retrieval and gate outcome (in-memory, dropped on reset / PDF or scope change).
- **Stage timings**: `Answer.timings` records per-stage seconds (rewrite, search, rerank, generate, total).
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

# 3. Configure (.env.example sets BACKEND=local; the config.py default is openai; local needs no keys)
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
`CHROMA_DIR/COLLECTION` · `RETRIEVAL_K` · `RERANK_ENABLED/MODEL/CANDIDATES/TOP_N`
(true / `BAAI/bge-reranker-base` / 8 / 4) ·
`RERANK_MIN_SCORE` (confidence gate, 0.02) · `HYBRID_SEARCH_ENABLED` (true) ·
`RRF_K` (60) · `QUERY_REWRITE_ENABLED` (false; per-request override) ·
`PROJECT_FALLBACK_TO_CORPUS` (true) · `PROJECT_MEMORY_SIZE/MERGE` (40/8; 0 disables) ·
`NEAR_REPEAT_THRESHOLD` (0 = off; cosine for project near-repeat reuse) · `MIN_RELEVANCE` (reserved) ·
`PROJECTS_DIR` · `UPLOADS_DIR` · `ANALYTICS_DB` ·
OpenAI backend only: `OPENAI_API_KEY`, `MONGODB_URI`.
`OLLAMA_BASE_URL` defaults to `http://127.0.0.1:11434`, not `localhost`: on this Windows machine every new
connection to `localhost` costs ~2 s (likely IPv6 `::1` refused, then IPv4 fallback; Ollama binds 127.0.0.1).

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
- `retrieval_eval.py` — **recall@1/3/5 (id-level), precision@1/3/5 (page-level) and MRR ablation**: dense → hybrid → hybrid+rerank, plus a `gate` section (in-scope vs `OFF_TOPIC` top rerank scores against `RERANK_MIN_SCORE`, and known-hard probes such as "What is a B+ tree?").
- `answer_eval.py` — automated answer metrics per question: refusal, citation correctness (inline `(course section)` cites that match a retrieved chunk), judge faithfulness/completeness (local llama3.1:8b, noisy), token counts and per-stage timings (`rewrite_s`, `search_s`, `rerank_s`, `generate_s`). Writes a CSV (`--limit N`, `--no-judge`).
- `ragas_eval.py` — **RAGAS faithfulness + answer_relevancy** with a **local Ollama judge** (needs a small `langchain_community.chat_models.vertexai` shim; memory-heavy — run with `HYBRID_SEARCH_ENABLED=false RERANK_ENABLED=false` to lighten).
- `report.py` — combined summary (retrieval, answer quality, latency, $/query); `--json` prints it as one JSON object (the metric suite: retrieval, answers incl. `uncited_rate` and stage medians/p95, cost per query).
- `speed_experiments.py` — live speed experiments (`--stages`, `--rerank-sweep`, `--near-repeat`, `--decide` with fixed keep/reject criteria); results in `eval/speed_experiments.json`.

Metric suite run (all `BACKEND=local`):
```
uv run python eval/retrieval_eval.py --out eval/out/metrics/retrieval.json
uv run python eval/answer_eval.py --out eval/out/metrics/answers.csv     # delete the CSV first
uv run python eval/report.py --out-dir eval/out/metrics --json
```

**Measured (real; 2026-10-08, n=50 generated questions, pool 8, bge-reranker-base, rewrite off):**

| | dense | hybrid | hybrid + rerank |
|---|---|---|---|
| recall@1 | 0.56 | 0.60 | **0.82** |
| recall@5 | 0.76 | 0.86 | **0.86** |
| precision@1 | 0.64 | 0.62 | **0.82** |
| MRR | 0.64 | 0.70 | **0.84** |

Answers (n=50, judge on): refusal rate 0.02, faithfulness 0.83, completeness 0.88
(self-graded by the same llama3.1:8b, so trends only), citation correctness 0.18,
uncited rate 0.82 (most answers carry no inline `(course section)` cite).
Latency (RTX 4060 for the LLM, CPU reranker, `OLLAMA_BASE_URL` 127.0.0.1; n=50, 2026-10-08): median 5.3 s,
p95 9.9 s; stage medians search 0.11 s, rerank 1.94 s, generate 2.85 s, rewrite 0. Prompt ~1,730 tokens,
output ~123 tokens. Cost: **$0/query** local API spend; OpenAI estimate $0.00033/query.
Gate: 2/50 in-scope below 0.02, 5/10 off-topic at or above it. RAGAS (older, n=6):
faithfulness 0.55 → 0.83 and answer_relevancy 0.80 → 0.97, dense vs hybrid+rerank.

---

## 11. Gotchas / things to know

- **First corpus query after restart builds the BM25 index (~40 s over ~10k docs)**; it's pre-warmed in a background thread at startup and cached thereafter.
- **Ingest wipes and rebuilds** the whole store (not incremental); a few minutes. Batched with retry/backoff to survive Windows socket exhaustion (`WSAENOBUFS`).
- **RAGAS** pulls a heavy dep tree (langchain 1.x, langgraph, pandas, pyarrow, datasets) and is memory-intensive — the app still imports fine, but run RAGAS with other apps closed / hybrid+rerank off on 8 GB.
- **Static assets** use `?v=N` + `no-cache`; bump `?v=` in the HTML when you must force a refetch.
- **Adding a course**: drop files in `data/raw/<course>/`, re-run `bearlm.local_store`.
- **Reranker/gate calibration**: bge scores are sigmoid [0,1] and the gate is 0.02. On this corpus the gate is leaky: 5 of the 10 off-topic probes in `_common.OFF_TOPIC` score ≥ 0.02 (top scores up to 0.68, e.g. Pride and Prejudice, basketball), while 2 of 50 in-scope questions fall below it (`retrieval_eval.py` `gate` section, n=50 + 10, 2026-10-08).
- **The reranker runs on CPU**: the installed torch is a CPU build (`torch.cuda.is_available()` is False despite the RTX 4060), so bge reranking at 8 candidates takes a median 1.95 s per answer (p95 2.08 s; `--stages`, 20 questions, 2026-10-08), about 38% of a 5.1 s answer. A CUDA torch build is the likely win, but it is a dependency change and has not been made.
- **Use `127.0.0.1`, not `localhost`, for Ollama.** Pooled connections drop after a few seconds idle, and reconnecting to `localhost` costs ~2 s on this Windows machine (probe: `embed_query` after 8 s idle took 2.08 s at `localhost` vs 0.04 s at 127.0.0.1, 4/4 samples; likely IPv6 fallback). `rag._get_llm()` builds a new `ChatOllama` per call, so every generation and judge call paid it too: every new connection pays ~2.2 s even with no idle gap (judge probe: a one-word chat took 2.57 s at `localhost` vs 0.37 s at 127.0.0.1, n=4). With the 127.0.0.1 default, in-answer search went 2.15 → 0.09 s and generate 4.83 → 2.66 s (stage medians, n=20, 2026-10-08).
- Root docs: this file (how it works now), `PROGRESS.md` (append-only running log of changes and measurements), `README.md` (user-facing), `DESIGN.md` (legacy Cohere-inspired design system).

---

## 12. History

What changed, why, and what was measured, run by run: see `PROGRESS.md` (append-only running log).

---

## 13. Known gaps / possible next steps

Ranked by the metric suite (2026-10-08); retrieval shows no regression (recall@5 0.86, MRR 0.84).
1. **GPU torch** (rerank median 1.94 s ≥ 1.5 s, 36% of the 5.3 s median): install a CUDA torch build so bge runs on the RTX 4060. Dependency change; needs the user's OK.
2. **Answer-generation latency** (generate 2.85 s = 53% of the 5.3 s median): try `RERANK_TOP_N` 4 → 3, a lower `OLLAMA_NUM_CTX`, or streaming; measure each.
3. **Gate recalibration** (5/10 off-topic ≥ 0.02): max off-topic is 0.68, and a gate there would drop 8/50 in-scope questions instead of 2, so a plain threshold change fails the "≤ 1 more in-scope refused" rule. Needs a better off-topic signal, not just a higher number.
4. **Citation format** (uncited 0.82, citation correctness 0.18): change the prompt/context header format so the model cites `(course section)`.
5. **BM25 tokenizer** for symbol terms (`B+`, `C++`): the "What is a B+ tree?" probe has no B+ tree chunk in the top 4.
6. Faithfulness 0.83 and in-scope refusal 0.02 are within bounds; `PROJECT_MEMORY_SIZE` default (40) is still unproven (see PROGRESS.md run 2).

Other:
- **Deployment**: local models complicate PaaS; currently a local-run project (could containerize Ollama + app).
- **Not implemented** (from the technique wishlist): LangSmith tracing (cloud; cuts against local ethos), LoRA/QLoRA fine-tuning, an MCP server.
- Minor: `pyproject.toml` has a placeholder description and a dead `bearlm:main` script entry.
- File-picker/drag-drop PDF upload is not automation-testable (OS dialog), but the add-PDF API path is tested.
