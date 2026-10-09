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
- **Inline citations**: each context chunk is headed `Source (course section):` and the prompts ask for `(course section)` after the claims it supports (`; ` between sources, parentheses, never a source outside the context). `rag._normalize_citations` then rewrites bracketed, labelled or spaced markers (`[cs188 p.89]`, `(Source: cs188 p.89)`, `(cs188 p. 89)`) into `(cs188 p.89)`, but only when every ref in the group is a context chunk; anything else is left byte-identical, so it never adds a citation. Applied on the Ask and project/upload paths, not to refusals.
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

**Anti-hallucination**: the reranker **confidence gate** (`RERANK_MIN_SCORE=0.02`)
applies only to projects and uploads (`answer_grounded`); it drops chunks below the
bar and refuses when nothing clears it. The Ask page (`rag.answer`) is not gated.
The gate is leaky by decision: some off-topic questions pass it (5/10 probes score
≥ 0.02), so the safeguard is the strict prompt with its exact refusal plus inline
citations to the context chunks. Meta/summary questions ("what's in this pdf")
bypass the gate.

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
`OLLAMA_NUM_PREDICT` (1024; output-token cap per generation) · `OLLAMA_TIMEOUT_S` (120; stall timeout: max wait to connect or for the next streamed chunk, not a total time limit) ·
`CHROMA_DIR/COLLECTION` · `RETRIEVAL_K` · `RERANK_ENABLED/MODEL/CANDIDATES/TOP_N`
(true / `BAAI/bge-reranker-base` / 8 / 4) ·
`RERANK_MIN_SCORE` (confidence gate, 0.02) · `RERANK_DEVICE` (`auto` = cuda if torch sees a GPU, else cpu; fp16 on cuda, cpu fallback if loading fails) · `HYBRID_SEARCH_ENABLED` (true) ·
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
- `answer_eval.py` — automated answer metrics per question: refusal, citation correctness (inline `(course section)` cites that match a retrieved chunk), judge faithfulness/completeness (local llama3.1:8b, noisy), token counts, per-stage timings (`rewrite_s`, `search_s`, `rerank_s`, `generate_s`) and `context_ids` (the context's chunk ids). `[ERROR]` rows are not judged or scored. Writes a CSV (`--limit N`, `--no-judge`) plus an `answers.meta.json` sidecar: Chroma dir and count, questions sha256, config, judge flag, Ollama `/api/ps` VRAM split, torch/CUDA/rerank device, nvidia-smi.
- `ragas_eval.py` — **RAGAS faithfulness + answer_relevancy** with a **local Ollama judge** (needs a small `langchain_community.chat_models.vertexai` shim; memory-heavy — run with `HYBRID_SEARCH_ENABLED=false RERANK_ENABLED=false` to lighten).
- `report.py` — combined summary (retrieval, answer quality, latency, $/query); `--json` prints it as one JSON object (the metric suite: retrieval, answers incl. `uncited_rate`, stage medians/p95, `generate_tok_per_s`, cost per query, and `run_meta` from `answers.meta.json`).
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

The same on the CPU and GPU reranker. Gate: 2/50 in-scope below 0.02, 5/10 off-topic at or above it.
The "What is a B+ tree?" probe has a B+ tree chunk in the top 4 (top score 0.995).

Answers (n=50, judge on, 2026-10-08, one complete run with the 1024-token cap after a cold Ollama start;
f51cec1 baseline → current): uncited rate 0.82 → 0.25, citation correctness 0.18 → 0.8125,
faithfulness 0.835 → 0.8396, completeness 0.876 → 0.8729 (self-graded by the same llama3.1:8b, so trends
only), refusal rate 0.02 → 0.04; 0 `[ERROR]` rows, 0 answers truncated by the cap.
Latency (RTX 4060 for the LLM and the reranker, `OLLAMA_BASE_URL` 127.0.0.1): median 4.2 s, p95 12.8 s;
stage medians search 0.17 s, rerank 0.06 s, generate 3.24 s (29 tokens/s), rewrite 0 (an earlier n=50 run
of the same code measured median 3.5 s, generate 2.97 s). The f51cec1 baseline
was median 5.3 s with the CPU reranker (rerank 1.94 s); the same current code on CPU torch measured 6.0 s
(rerank 2.26 s). Prompt ~1,780 tokens, output ~128 tokens. Cost: **$0/query** local API spend; OpenAI
estimate $0.00033/query. RAGAS (older, n=6): faithfulness 0.55 → 0.83 and answer_relevancy 0.80 → 0.97,
dense vs hybrid+rerank.

---

## 11. Gotchas / things to know

- **First corpus query after restart builds the BM25 index (~40 s over ~10k docs)**; it's pre-warmed in a background thread at startup and cached thereafter.
- **Ingest wipes and rebuilds** the whole store (not incremental); a few minutes. Batched with retry/backoff to survive Windows socket exhaustion (`WSAENOBUFS`).
- **RAGAS** pulls a heavy dep tree (langchain 1.x, langgraph, pandas, pyarrow, datasets) and is memory-intensive — the app still imports fine, but run RAGAS with other apps closed / hybrid+rerank off on 8 GB.
- **Static assets** use `?v=N` + `no-cache`; bump `?v=` in the HTML when you must force a refetch.
- **Adding a course**: drop files in `data/raw/<course>/`, re-run `bearlm.local_store`.
- **Reranker/gate calibration**: bge scores are sigmoid [0,1] and the gate is 0.02. On this corpus the gate is leaky: 5 of the 10 off-topic probes in `_common.OFF_TOPIC` score ≥ 0.02 (top scores up to 0.68, e.g. Pride and Prejudice, basketball), while 2 of 50 in-scope questions fall below it (`retrieval_eval.py` `gate` section, n=50 + 10, 2026-10-08). Kept as is by the user's decision: a higher threshold would refuse more in-scope questions, and the strict prompt plus inline citations are the safeguard. The gate only applies to projects/uploads.
- **The reranker runs on the GPU**: `pyproject.toml` pulls torch `2.14.0+cu130` from the `pytorch-cu130` index on Windows/Linux (PyPI torch on macOS), and `RERANK_DEVICE=auto` loads bge on cuda in fp16. Rerank median 2.26 s → 0.06 s and answer median 6.0 → 4.2 s (n=50, 2026-10-08; 3.5 s in an earlier run). Kept after one complete 50-row run with no `[ERROR]` rows and no OOM. VRAM is tight: llama3.1:8b (5.81 GB) stays fully on the GPU with the reranker loaded in either order, at ~7.3 of 8.2 GB used (torch reserves ~0.66 GB). Each process that loads the reranker holds its own copy, so don't run the API server and an eval at the same time. The CUDA wheel adds ~2.4 GB to `.venv` (torch 507 MB → 2.9 GB). Set `RERANK_DEVICE=cpu` to force CPU.
- **Eval reproducibility**: the answer suite is repeatable on the same store. Copies of `db/` taken with no BearLM process running are equivalent: md5-identical files, the same contexts on 50/50 rows, and the same 0.8347 / 0.8755 / 0.1837 means. The judge is deterministic: 0 score differences on identical answers. Two things still move the figures. (1) Ollama sometimes hits a runtime `CUDA error: out of memory` on `/api/chat` and restarts llama-server; that row becomes an `[ERROR]` row, which is now excluded from scoring. Check for `[ERROR]` rows before comparing runs. (2) About 1 in 50 answers can differ from the first token on an identical context, depending on Ollama's request history. Each run's `answers.meta.json` records the store, config and VRAM split, and `context_ids` separates retrieval changes from generation changes.
- **Output cap and timeout**: `rag._get_llm()` passes `num_predict=OLLAMA_NUM_PREDICT` (1024) and an httpx timeout of `OLLAMA_TIMEOUT_S` (120 s) to every LLM call (answers, rewrite, optimizer, judge, practice, question generation). `ChatOllama.invoke` streams, so the timeout bounds each connect or wait for the next chunk, not the whole request; the 1024-token cap is what bounds a long generation, so a request takes at most about model load + prompt eval + 1024 tokens. Without a cap, a degenerate generation context-shifted forever (up to 46,625 tokens seen after an Ollama crash-reload). The longest legitimate answer seen was 815 tokens, so 1024 leaves ~25% headroom and bounds a runaway to ~31 s. When an answer hits the cap (`done_reason == "length"`), `_answer_local` and `answer_grounded` append `rag.TRUNCATED_NOTE` ("_(Answer cut off at the length limit.)_"). A timeout raises: the API returns 500 and the eval writes an `[ERROR]` row.
- **Runtime CUDA OOM**: Ollama occasionally hits `CUDA error: out of memory` on `/api/chat`, and llama-server crashes and reloads. This predates CUDA torch: it was seen on CPU torch (19:24 and 19:47) and on CUDA torch (21:18) on 2026-10-08. VRAM headroom is small (~1.1–1.5 GiB free with llama3.1:8b and nomic loaded, less with the reranker on cuda). It is not prevented. With the output cap and the timeout, it now ends as an `[ERROR]` eval row or a 500 instead of a hang. Before the cap, the first generation after the reload could run away (n_gen 46,625).
- **Use `127.0.0.1`, not `localhost`, for Ollama.** Pooled connections drop after a few seconds idle, and reconnecting to `localhost` costs ~2 s on this Windows machine (probe: `embed_query` after 8 s idle took 2.08 s at `localhost` vs 0.04 s at 127.0.0.1, 4/4 samples; likely IPv6 fallback). `rag._get_llm()` builds a new `ChatOllama` per call, so every generation and judge call paid it too: every new connection pays ~2.2 s even with no idle gap (judge probe: a one-word chat took 2.57 s at `localhost` vs 0.37 s at 127.0.0.1, n=4). With the 127.0.0.1 default, in-answer search went 2.15 → 0.09 s and generate 4.83 → 2.66 s (stage medians, n=20, 2026-10-08).
- Root docs: this file (how it works now), `PROGRESS.md` (append-only running log of changes and measurements), `README.md` (user-facing), `DESIGN.md` (legacy Cohere-inspired design system).

---

## 12. History

What changed, why, and what was measured, run by run: see `PROGRESS.md` (append-only running log).

---

## 13. Known gaps / possible next steps

Ranked by the metric suite (2026-10-08); retrieval shows no regression (recall@5 0.86, MRR 0.84).
1. **Answer-generation latency** (generate 3.24 s = 77% of the 4.2 s median now that rerank is 0.06 s): try `RERANK_TOP_N` 4 → 3, a lower `OLLAMA_NUM_CTX`, or streaming; measure each.
2. **Remaining uncited answers** (uncited 0.25): a quarter of answers still carry no inline cite. `report.py`'s `uncited_rate` does not recognise `genML` cites (`_COURSE_RE` needs a digit), so it overstates this: on the iter-2 complete run, 3 of the 12 uncited rows cite genML (Ctrl-t, GMM ellipse angle, GMM covariance_type), so about 0.19 (9/48) is truly uncited.
3. Faithfulness and in-scope refusal (0.04) are within bounds; `PROJECT_MEMORY_SIZE` default (40) is still unproven (see PROGRESS.md run 2).

Other:
- **Deployment**: local models complicate PaaS; currently a local-run project (could containerize Ollama + app).
- **Not implemented** (from the technique wishlist): LangSmith tracing (cloud; cuts against local ethos), LoRA/QLoRA fine-tuning, an MCP server.
- Minor: `pyproject.toml` has a placeholder description and a dead `bearlm:main` script entry.
- File-picker/drag-drop PDF upload is not automation-testable (OS dialog), but the add-PDF API path is tested.
