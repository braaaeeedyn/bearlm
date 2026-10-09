# BearLM — Project Context & Handoff

_Last updated: 2026-10-09 (code at commit b124165). **This file is the current
state**: how BearLM works now, what it is, how it's built, and what you need to
run, extend or explain it. It is rewritten in place when anything changes.
`PROGRESS.md` is the history (what changed, why, and what was measured);
`README.md` is setup for anyone who clones the repo._

---

## 1. What it is

**BearLM** is a **fully local, zero-cost, grounded RAG (retrieval-augmented
generation) Q&A assistant** over UC Berkeley CS/Data-Science course materials,
plus a **Projects workspace** where users bring their own PDFs and hold grounded,
multi-turn conversations over them.

Everything runs on the user's machine (Ollama + Chroma + FastAPI). There are
**no API keys, accounts or cloud services**, and no `.env` is needed: every
setting has a working default. It is a single-user app meant to run on one
computer; the repository is public on GitHub.

Design goals: **grounded answers with citations, anti-hallucination, and a real
full-stack product** (web UI, analytics, evaluation), not just a script.

---

## 2. Stack

| Concern | Choice |
|---|---|
| Orchestration | LangChain |
| Embeddings | Ollama `nomic-embed-text` (768-dim) |
| Generator | Ollama `llama3.1:8b` (temp 0, num_ctx 8192, num_predict 1024) |
| Vector store | Chroma (on-disk `./db`) |
| Keyword search | BM25 (`rank_bm25`), fused with dense via Reciprocal Rank Fusion |
| Reranking | `bge-reranker-base` cross-encoder (sentence-transformers; CUDA torch on Windows/Linux) |
| Chunking | token-based `RecursiveCharacterTextSplitter` (tiktoken), 500 tokens / 50 overlap |
| Eval judge | the same local `llama3.1:8b` |
| Backend | FastAPI + Uvicorn |
| Frontend | hand-built vanilla JS (marked, DOMPurify, KaTeX), no framework or build |
| Analytics | SQLite (`data/analytics.db`) |

Python 3.13, managed with **uv** (`pyproject.toml` + `uv.lock`). Dev machine:
Windows, RTX 4060 (8 GB VRAM).

---

## 3. Corpus

**10,394 chunks across 10 courses**, 768-dim, in Chroma collection `bearlm_chunks`:

`cs61a · cs61b · cs70 · cs188 · cs189 · data8 · data100 · data101 · data144 · genML`

- **Public:** the chunk text and metadata ship in the repo as
  `data/corpus_chunks.jsonl.gz` (~4.6 MB; one JSON object per line:
  `{id, text, metadata: {course, section, source, chunk_id}}`, no embeddings).
  `python -m bearlm.local_store` embeds it into `./db` on the user's machine.
- **Private:** the original course files (PDFs, Markdown, notebooks) live in
  `data/raw/<course>/` on the owner's machine only (gitignored, never committed).
  `--raw` re-chunks them; `--export` rewrites the shipped chunk file from the
  current store.

Chunk ids are `{course}:{n}` (stable, so eval gold labels reference them). The
`course` label is the `data/raw` folder name; `section` is `p.N` for PDFs or the
file stem for Markdown/notebooks; `source` is the filename (used for PDF deep links).

---

## 4. Architecture / modules (`src/bearlm/`)

| File | Responsibility |
|---|---|
| `config.py` | all settings and their env overrides (see §7) |
| `doc_processor.py` | extract text from PDF (per page), Markdown and notebooks in `data/raw`; build per-course chunks |
| `text_splitter.py` | token-based `RecursiveCharacterTextSplitter` (tiktoken length fn) |
| `local_store.py` | Ollama embeddings + multi-collection Chroma; build the corpus store from the shipped chunks (`ingest_export`) or `data/raw` (`ingest_raw`); `export_chunks`; per-PDF ingest for projects/uploads; collection drop/count/fetch-by-id; distinct courses |
| `search.py` | hybrid search: BM25 + dense fused with RRF; per-course filtering; BM25 index cached per collection; tokenizer keeps `B+`, `C++`, `C#` |
| `rerank.py` | `bge-reranker-base` cross-encoder (`rerank`, `rerank_scored`); `RERANK_DEVICE` picks cuda (fp16) or cpu, falling back to cpu if the cuda load fails |
| `rag.py` | `answer` (Ask page), `answer_grounded` (projects/uploads: confidence gate, project-first fallback, multi-turn history, corpus scope, retrieval memory), `_generate` (one retry on an Ollama 5xx/stream error, then `LLMUnavailable`), citation normalizing and filtering, query rewriting, prompt optimizer, meta-question bypass |
| `projects.py` | projects, chats, ad-hoc uploads, per-PDF collections, corpus-scope setting, retrieval memory |
| `analytics.py` | SQLite query logging, stats, most-asked area, grounded practice-question generation |
| `api.py` | FastAPI: all endpoints, static host, no-cache middleware, BM25 pre-warm in the `lifespan` startup, 503 handlers for Ollama failures |

**Frontend (`web/`):** `index.html` (Ask), `projects.html` (Projects app shell),
`analytics.html`, `render.js` (shared: markdown/math/citations/provenance/optimizer),
`app.js`, `projects-app.js`, `analytics.js`, `styles.css` (Berkeley night theme: night-blue `#152237` base, California Gold accent; light mode via `[data-theme="light"]`),
`vendor/` (marked, DOMPurify, KaTeX, all local). Static asset URLs are versioned
(`?v=N`) and served with `Cache-Control: no-cache`.

---

## 5. Features

**Ask page** (course corpus Q&A):
- Grounded answers with deduplicated source chips that deep-link to the PDF page
  (`#page=N`) when the PDF exists locally in `data/raw`; otherwise the link
  returns a 404 explaining the course files aren't published.
- **Inline citations**: each context chunk is headed `Source (course section):`
  and the prompts ask for `(course section)` after the claims it supports.
  `rag._normalize_citations` rewrites bracketed, labelled or spaced markers
  (`[cs188 p.89]`, `(Source: cs188 p.89)`) into `(cs188 p.89)`, only when every
  ref in the group is a context chunk. Then `rag._drop_outside_citations` removes
  cites that name no context chunk: it rewrites a `(..)`/non-link `[..]` group only
  when every part looks like `<course> <section with a digit>`, where the course
  is `cs…`/`data…` plus digits or a context course. Math and prose such as
  `(R2 = 0.87)` or `(see cs61a p.3 for details)` are left alone. It only removes
  refs, never adds any. Limits: a fabricated cite to another prefix (`stat134`) or
  a non-digit course (`genML`, a project slug) outside the context is kept, and a
  prose or math group shaped like a cite, e.g. `(cs61b has 3 projects)` or
  `(cs2 = 5)`, is deleted when that course is outside the context. Both steps
  skip refusals.
- No "Sources:" line is appended to answer text: the source chips already list
  the context, and appending them would claim attribution the model never made.
- **Prompt optimizer** ("Improve prompt", suggest-then-confirm) and a query-rewrite toggle.
- **Provenance**: shows "Your message" vs. "Optimized prompt sent" / "Retrieval query used".
- Rich rendering: Markdown tables → HTML tables, fenced code → code boxes, LaTeX → KaTeX.

**Projects** (the primary surface):
- Collapsible sidebar; projects listed newest→oldest.
- Per-project reference PDFs with list & grid (thumbnail) views, open-in-new-tab, and remove.
- Multiple persistent multi-turn chats per project (recent turns fed back as context; follow-ups are retrieved as `previous user turn + question`).
- **Reset context** clears chats and retrieval memory, keeps PDFs.
- **Retrieval memory**: chunk ids behind past answers (stored in `project.json`, LRU-capped) are re-added to the rerank pool for later questions; they still must clear the gate.
- **Per-project corpus scoping**: choose which course subjects a project may fall back on (none = project-only).
- **Stale-project prompt**: projects >1 week old prompt for deletion on open.

**Ad-hoc PDF upload** ("Chat with a PDF"): single PDF, strictly grounded, ephemeral.

**Anti-hallucination**: the reranker **confidence gate** (`RERANK_MIN_SCORE=0.02`)
applies only to projects and uploads (`answer_grounded`): it drops chunks below
the bar and refuses when nothing clears it. The Ask page (`rag.answer`) has no
gate. The gate is loose by decision (5/10 off-topic probes score ≥ 0.02), so the
main safeguard is the strict prompt with its exact refusal plus inline citations
to context chunks. Meta/summary questions ("what's in this pdf") bypass the gate.

**Answer length**: every LLM call is capped at `OLLAMA_NUM_PREDICT` (1024) tokens;
an answer that hits the cap ends with "_(Answer cut off at the length limit.)_".

**Analytics** (local SQLite): usage stats, most-cited courses, per-day volume,
unanswered questions, a "needs the most help" area, and grounded practice
questions for that area.

---

## 6. How to run

See `README.md` "Setup" for the full first-time steps. In short:

```bash
ollama pull nomic-embed-text && ollama pull llama3.1:8b
uv sync
uv run python -m bearlm.local_store       # build ./db from data/corpus_chunks.jsonl.gz (~3.5 min on the RTX 4060)
uv run uvicorn bearlm.api:app             # http://localhost:8000
```

Store maintenance: `--raw` (re-chunk `data/raw`), `--export` (write the shipped
chunk file), `--verify` (count + dim; expect 10394 / 768). Every build wipes and
rebuilds the corpus store (project and upload collections are untouched).

Pages: `/` (Ask), `/projects`, `/analytics`, `/api/docs` (FastAPI).

---

## 7. Config knobs (`config.py` / `.env`)

`OLLAMA_BASE_URL` (`http://127.0.0.1:11434`) · `OLLAMA_CHAT_MODEL` (llama3.1:8b) ·
`OLLAMA_EMBED_MODEL` (nomic-embed-text) · `OLLAMA_NUM_CTX` (8192) ·
`OLLAMA_NUM_PREDICT` (1024; output cap) · `OLLAMA_TIMEOUT_S` (120; stall timeout:
max wait to connect or for the next streamed chunk, not a total time limit) ·
`CHROMA_DIR` (`./db`) · `CHROMA_COLLECTION` (bearlm_chunks) · `TOKENIZER_ENCODING`
(cl100k_base) · `RETRIEVAL_K` (4) · `RERANK_ENABLED/MODEL/CANDIDATES/TOP_N`
(true / `BAAI/bge-reranker-base` / 8 / 4) · `RERANK_DEVICE` (auto) ·
`RERANK_MIN_SCORE` (0.02) · `HYBRID_SEARCH_ENABLED` (true) · `RRF_K` (60) ·
`QUERY_REWRITE_ENABLED` (false; per-request override) ·
`PROJECT_FALLBACK_TO_CORPUS` (true) · `PROJECT_MEMORY_SIZE/MERGE` (40/8; 0 disables) ·
`PROJECTS_DIR` · `UPLOADS_DIR` · `ANALYTICS_DB`.

`OLLAMA_BASE_URL` uses `127.0.0.1`, not `localhost`: on Windows every new
connection to `localhost` costs ~2 s (likely an IPv6 attempt before IPv4), and
`rag._get_llm()` builds a new client per call.

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
GET  /source/{course}/{file}                     serve a corpus PDF from data/raw (404 + explanation if absent)
GET  /projects/{slug}/pdfs/{file}[/thumb]        serve a project PDF / thumbnail
```

The three ask endpoints return **503** with JSON `{"detail": "..."}` when Ollama
fails (not running, out of memory, a stall, or an embed failure during
retrieval). The message names Ollama and `OLLAMA_BASE_URL` and points to README
"Troubleshooting"; the pages show it as is. A failed chat turn is not saved.

---

## 9. Data layout

```
data/corpus_chunks.jsonl.gz   # TRACKED: public course chunk text + metadata
data/raw/<course>/            # gitignored: private course files (owner's machine only)
data/projects/<slug>/         # gitignored: project.json (pdfs, chats, corpus_courses, memory) + pdfs/
data/uploads/<id>/            # gitignored: ad-hoc upload PDFs
db/                           # gitignored: Chroma store (~150 MB), built locally
data/analytics.db             # gitignored: SQLite query log
eval/out/                     # gitignored: eval artifacts
.env                          # gitignored, optional: setting overrides
```

---

## 10. Evaluation (`eval/`)

- `generate_questions.py`: builds a 50-question benchmark from real chunks (each question written from a chunk, so its `(course, section, chunk_id)` is exact gold) → `questions_generated.jsonl`.
- `retrieval_eval.py`: recall@1/3/5 (id-level), precision@1/3/5 (page-level) and MRR for dense → hybrid → hybrid+rerank, plus a `gate` section (in-scope vs `_common.OFF_TOPIC` top rerank scores against `RERANK_MIN_SCORE`, and a "What is a B+ tree?" probe).
- `answer_eval.py`: per question: refusal, citation correctness (inline cites that match a retrieved chunk), judge faithfulness/completeness (local llama3.1:8b, noisy), token counts, per-stage timings and `context_ids`. `[ERROR]` rows are not judged or scored. Writes a CSV (`--limit N`, `--no-judge`) plus an `answers.meta.json` sidecar (store, config, Ollama VRAM split, torch/CUDA state).
- `report.py`: combined summary; `--json` prints retrieval, answer metrics (incl. `uncited_rate`), stage medians/p95, `generate_tok_per_s`, cost per query ($0) and `run_meta`.

**Current results** (n=50, RTX 4060, 2026-10-09):

| | dense | hybrid | hybrid + rerank |
|---|---|---|---|
| recall@1 | 0.56 | 0.60 | **0.82** |
| recall@5 | 0.76 | 0.86 | **0.86** |
| precision@1 | 0.64 | 0.62 | **0.82** |
| MRR | 0.64 | 0.70 | **0.84** |

Answers: faithfulness 0.84, completeness 0.87 (self-graded by the same 8B model,
so trends only), citation correctness 0.81, uncited 0.19 (9/48), refusal 0.04,
cites in context 55/55. Latency: median ~3.6–3.8 s, p95 ~12.4 s; generate median
~3.0 s at ~31 tok/s. Cost $0/query. Gate: 2/50 in-scope below 0.02, 5/10
off-topic at or above it.

Tests: `uv run --with pytest python -m pytest -q tests` (offline; LLM, Chroma and
retrieval mocked).

---

## 11. Gotchas / things to know

- **First corpus query after a restart builds the BM25 index** (~40 s over ~10k chunks); it's pre-warmed in a background thread at startup and cached thereafter.
- **Store builds wipe and rebuild the corpus collection** (not incremental), batched with retry/backoff to survive Windows socket exhaustion (`WSAENOBUFS`).
- **VRAM is tight on 8 GB**: llama3.1:8b (5.8 GB) plus the fp16 reranker (~0.66 GB reserved) use ~7.3 of 8.2 GB. Each process that loads the reranker holds its own copy, so don't run the API server and an eval at the same time.
- **Runtime CUDA OOM**: Ollama occasionally hits `CUDA error: out of memory` on `/api/chat` and llama-server reloads. It's not prevented: `rag._generate` retries once on an Ollama 5xx/stream error; if that fails, the ask endpoints return 503 and the eval writes an `[ERROR]` row. A 4xx (model not pulled), a connection error (Ollama down) or a stall is not retried. Mitigations: `OLLAMA_NUM_CTX=6144`, `RERANK_DEVICE=cpu`, a smaller chat model.
- **Output cap and timeout**: `rag._get_llm()` passes `num_predict=1024` and an httpx timeout of 120 s to every LLM call. `ChatOllama.invoke` streams, so the timeout bounds each connect or next-chunk wait; the cap is what bounds a long generation (without it, a degenerate generation once ran to 46,625 tokens).
- **Any builtin `ConnectionError` reaching the API becomes the Ollama 503**; only the Ollama client raises it in this app.
- **Eval reproducibility**: the judge is deterministic and copies of `db/` are equivalent; about 1 in 50 answers can still differ from the first token on an identical context, and OOM `[ERROR]` rows are excluded from scoring. Compare runs via `answers.meta.json` and `context_ids`.
- **Static assets** use `?v=N` + `no-cache`; bump `?v=` in the HTML to force a refetch.
- **Changing `OLLAMA_EMBED_MODEL`** requires rebuilding the store.
- Root docs: this file (how it works now), `PROGRESS.md` (append-only history), `README.md` (user-facing setup).

---

## 12. History

What changed, why, and what was measured, run by run: see `PROGRESS.md`.

---

## 13. Roadmap and open decisions

**Roadmap** (ranked):
1. **Generation latency**: try `RERANK_TOP_N` 4 → 3, a lower `OLLAMA_NUM_CTX`, or streaming responses to the UI; measure each.
2. **VRAM headroom**: measure `OLLAMA_NUM_CTX=6144` for OOM rate and quality.
3. **Reranker runtime CUDA OOM**: no fallback once the model is loaded (only the load falls back to cpu).
4. **Tests for `/api/practice`, `/api/stats` and `/api/courses`** (need real SQLite/Chroma data).
5. **`PROJECT_MEMORY_SIZE`** (40) is still unproven.
6. **One-command start or a health endpoint.**
7. **Remaining uncited answers** (9/48, mostly short facts, numbers, formulas, code or tables): try structured citations, which would also remove `_drop_outside_citations`' limits.

**Needs your decision:**
- Multi-user use: there is no auth, and projects are shared by everyone on the server.
- Optional cloud tooling (e.g. LangSmith) versus the local-only design.

Decided: hosting is the owner's own computer; the repo is public on GitHub with
chunk text only (course PDFs stay private).

Not planned: LoRA/QLoRA fine-tuning, an MCP server.
