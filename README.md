# BearLM 🐻

A **retrieval-augmented Q&A assistant** grounded in UC Berkeley CS/DS course
materials (CS61A, CS61B, CS70, CS188, CS189, Data 8, Data 100, plus your own
added courses). Ask a question, get an answer that is **cited back to the source**
(course + section, deep-linked to the exact PDF page) and **refuses rather than
guessing** when the material doesn't cover it.

Beyond the shared course corpus, BearLM has **Projects** — bring your own PDFs,
hold multi-turn grounded conversations over them, and get project-first answers
that fall back to the course corpus only when your documents don't cover the
question. See [Projects & your own PDFs](#projects--your-own-pdfs).

BearLM runs in two interchangeable backends, selected by the `BACKEND` env var:

| | `BACKEND=local` (default) | `BACKEND=openai` |
|---|---|---|
| Embeddings | `nomic-embed-text` (Ollama), 768-dim | `text-embedding-3-small`, 1536-dim |
| Generator | `llama3.1:8b` (Ollama) | `gpt-4o-mini` |
| Vector store | Chroma (`./db`, on-disk) | MongoDB Atlas Vector Search |
| Reranker | `bge-reranker-base` (optional) | — |
| Cost / query | **$0** (self-hosted) | ~$0.0006 (estimate) |

The local stack needs no API keys and no cloud services.

---

## Architecture

```
                         ┌──────────────────── ingest (offline) ────────────────────┐
  data/raw/*.pdf,.md,.ipynb ─► extract ─► RecursiveCharacterTextSplitter (token len)
                                              │  metadata: {course, section, source}
                                              ▼
                                     OllamaEmbeddings(nomic-embed-text)  ─►  Chroma (./db)
                         └───────────────────────────────────────────────────────────┘

                         ┌──────────────────── query (online) ──────────────────────┐
  question ─► Chroma retriever (k=4, or 8→rerank→4 if RERANK_ENABLED)
                    │
                    ▼
             ChatPromptTemplate (strict: answer only from context;
             else "I don't know based on the course materials.")
                    │
                    ▼
             ChatOllama(llama3.1:8b, temperature=0, num_ctx=8192)
                    │
                    ▼
             StrOutputParser()  ─►  answer  +  deduped (course, section) citations
                         └───────────────────────────────────────────────────────────┘

  chain (LCEL):  retriever | prompt | llm | StrOutputParser()      (rag.build_chain)
  web:           FastAPI (bearlm.api) serves web/ + POST /api/ask, GET /api/stats
```

Backend selection lives in `src/bearlm/config.py` (`BACKEND=local|openai`). The
OpenAI/Atlas path is kept intact behind the flag for side-by-side comparison.

---

## Setup (local backend)

### 1. Install Ollama
Ollama is **not** auto-installed. Get it from <https://ollama.com/download>, or:

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
# Windows: download and run OllamaSetup.exe from the site above
```

Confirm it's running: `ollama --version` and `ollama list`.

### 2. Pull the models
```bash
ollama pull nomic-embed-text        # 768-dim embeddings
ollama pull llama3.1:8b             # generator  (needs ~8GB VRAM / RAM)
# On a machine under ~8GB VRAM, use a smaller generator instead:
ollama pull qwen2.5:3b             # then set OLLAMA_CHAT_MODEL=qwen2.5:3b in .env
```

### 3. Install Python deps
```bash
uv sync                             # preferred (Python 3.13)
# or: pip install -r requirements.txt
```

### 4. Configure
```bash
cp .env.example .env
# BACKEND=local is the default; no API keys required.
```

### 5. Ingest the corpus into Chroma
```bash
BACKEND=local uv run python -m bearlm.local_store            # token splitter (~2,581 chunks)
BACKEND=local uv run python -m bearlm.local_store --legacy-splitter   # reproduces exactly 2,241
BACKEND=local uv run python -m bearlm.local_store --verify  # print stored count + dim (expect 768)
```

> **Chunk count note:** the token-based `RecursiveCharacterTextSplitter` produces
> **2,581** chunks over this corpus; the original word-window splitter produces
> **2,241**. Use `--legacy-splitter` if you specifically want the 2,241 baseline.

### 6. Run the app
```bash
BACKEND=local uv run uvicorn bearlm.api:app --reload   # http://localhost:8000
BACKEND=local uv run python -m bearlm.rag              # terminal smoke test
```

### Running the OpenAI backend instead
Set `BACKEND=openai` and provide `OPENAI_API_KEY` + `MONGODB_URI` in `.env`
(the Atlas `chunks` collection + `vector_index` must already exist).

> Adding a course to the shared corpus: drop `.pdf`/`.md`/`.ipynb` files (or clone
> a repo) into `data/raw/<course>/`, then re-run `bearlm.local_store`. The course
> label is the folder name. Re-ingest rebuilds the whole store (it is not
> incremental). The current corpus is ~10.4k chunks across 10 courses.

---

## Projects & your own PDFs

Open `/projects` in the app (local backend only). Everything here is grounded and
local — no keys, no cloud.

**Projects** (persistent) — a named folder of your PDFs plus its own vector
collection:
- Add PDFs from the sidebar; each is embedded into the project's collection.
- Hold **multiple chats** per project — each a persistent, multi-turn conversation.
  The whole conversation is fed back as context every turn, so follow-ups
  ("what about its top speed?") resolve against history.
- **Project-first retrieval:** answers come from your PDFs when they cover the
  question, and fall back to the course corpus otherwise
  (`PROJECT_FALLBACK_TO_CORPUS`).
- **Reset context** clears a project's chats but keeps its PDFs — a one-click way
  to cut drift/hallucination from long histories.
- On open, projects older than a week prompt you to delete them.

**Chat with a PDF** (ad-hoc) — upload one PDF for a throwaway session answered
**strictly** from that document (no corpus fallback).

**Anti-hallucination** applies to both:
- **Strict prompt** + an exact refusal string when the answer isn't in context.
- **Confidence gate** — a `bge-reranker` cross-encoder scores each retrieved
  chunk; anything below `RERANK_MIN_SCORE` is dropped, and if nothing clears the
  bar the answer is refused rather than guessed. (Summary/"what's in this doc"
  questions bypass the gate, since they're topically orthogonal to any one passage.)
  The gate is leaky by design: some off-topic questions clear it (5 of 10 probes
  score ≥ 0.02), so the strict prompt and inline citations to the retrieved chunks
  are the real safeguard. The Ask page (corpus Q&A) has no gate.
- **Always-on citations** that deep-link to the source PDF at the cited page.

Storage: projects live under `data/projects/<slug>/` (PDFs + `project.json` with
chats); uploads under `data/uploads/`. Both are gitignored.

---

## Configuration

All settings have env overrides (see `.env.example`); defaults live in
`src/bearlm/config.py`.

| Variable | Default | What it does |
|---|---|---|
| `BACKEND` | `openai` | `local` (Ollama+Chroma) or `openai` (OpenAI+Atlas). `.env` sets `local`. |
| `OLLAMA_CHAT_MODEL` | `llama3.1:8b` | Local generator (`qwen2.5:3b` for <8 GB VRAM). |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Local embeddings (768-dim). |
| `OLLAMA_NUM_CTX` | `8192` | Generator context window. |
| `OLLAMA_NUM_PREDICT` | `1024` | Max tokens per generation; a cut-off answer ends with "_(Answer cut off at the length limit.)_". |
| `OLLAMA_TIMEOUT_S` | `120` | Maximum wait (seconds) for Ollama to connect or send the next streamed chunk; a stall timeout, not a total time limit (`OLLAMA_NUM_PREDICT` bounds a long generation). |
| `CHROMA_DIR` | `./db` | On-disk Chroma store. |
| `RETRIEVAL_K` | `4` | Chunks fed to the LLM. |
| `RERANK_ENABLED` | `true` | Rerank corpus retrieval (retrieve `RERANK_CANDIDATES` → keep 4). |
| `RERANK_CANDIDATES` | `8` | Candidates scored by the reranker (`BAAI/bge-reranker-base`, `RERANK_MODEL`); 8 measured ~2x faster than 15 at equal recall (`eval/speed_experiments.json`; the sweep's first config had warm-up-inflated search time, corrected ~6.5 s vs 2.9 s search+rerank). |
| `RERANK_MIN_SCORE` | `0.02` | Confidence gate for project/upload answers. |
| `RERANK_DEVICE` | `auto` | Reranker device: `auto` (cuda if torch sees a GPU, else cpu), `cpu` or `cuda`. On cuda it loads in fp16 and falls back to cpu if loading fails. |
| `PROJECT_FALLBACK_TO_CORPUS` | `true` | Let project answers fall back to the corpus. |
| `PROJECT_MEMORY_SIZE` | `40` | Per-project memory of relevant chunk ids (LRU; `0` disables). |
| `PROJECT_MEMORY_MERGE` | `8` | Remembered chunks added to each query's rerank pool. |
| `NEAR_REPEAT_THRESHOLD` | `0` | Off by default. When > 0, a project question this cosine-close to an earlier one reuses its retrieval and gate outcome (in-memory). Rejected as a default: follow-ups (retrieved as previous turn + follow-up) reach 0.977 against the previous turn, so no threshold avoids false reuse while keeping half the true repeats (`eval/speed_experiments.json`). |
| `MIN_RELEVANCE` | `0.30` | Vector-relevance floor (reserved). |

For the OpenAI backend, `OPENAI_API_KEY` and `MONGODB_URI` are required.

---

## Evaluation

All eval lives in `eval/`. The question set is **generated from real corpus
chunks** — each question is written from a specific chunk, so that chunk's
`(course, section, chunk_id)` is the exact gold. `recall@k` then measures whether
retrieval surfaces the true source chunk (a synthetic-but-grounded benchmark:
LLM-written questions, exact gold labels). `eval/questions.jsonl` is a small
hand-drafted template you can extend by hand.

```bash
# 1. Generate an eval set from the corpus (5 questions/course → eval/questions_generated.jsonl)
BACKEND=local uv run python eval/generate_questions.py --per-course 5

# 2. Retrieval quality (recall@k, precision@k, MRR): dense vs hybrid vs hybrid+rerank
BACKEND=local uv run python eval/retrieval_eval.py --questions eval/questions_generated.jsonl

# 3. End-to-end answers: refusal, citation correctness, faithfulness/completeness
#    (local LLM judge; --no-judge to skip), tokens, latency (--limit N to sample)
BACKEND=local uv run python eval/answer_eval.py --questions eval/questions_generated.jsonl --limit 10

# 4. Summary (--out-dir DIR to read artifacts elsewhere)
uv run python eval/report.py
```

Retrieval and answers are scored separately. `recall@k` is id-level (gold chunk
in the top k); `precision@k` is page-level (a top-k chunk counts if it shares the
gold course+section). Answer metrics are computed on the exact chunks the answer
was built from; faithfulness/completeness are noisy `llama3.1:8b` judge estimates.
OpenAI `$/query` = list prices x measured token counts (still an estimate).
Tests: `uv run --with pytest python -m pytest -q tests`.

### Retrieval results (n=50, generated set, pool=15)

Each technique measurably helps — hybrid lifts recall, the reranker sharpens the
top rank:

| metric | dense | hybrid | hybrid + rerank |
|---|---|---|---|
| recall@1 | 0.560 | 0.600 | **0.820** |
| recall@3 | 0.680 | 0.780 | **0.860** |
| recall@5 | 0.760 | 0.840 | **0.860** |
| MRR | 0.644 | 0.700 | **0.840** |

**Answering** (`llama3.1:8b` on an 8 GB RTX 4060, 10-question sample): median
**6.0 s**, p95 **26 s** (p95 is first-call model warmup; steady-state is ~4–8 s).
**Cost / query:** local **$0.0000** (self-hosted); OpenAI baseline ~$0.0006
(estimate — see `report.py`).

### Automated answer grading with RAGAS

`eval/ragas_eval.py` scores answer quality automatically with
[RAGAS](https://docs.ragas.io) — using the **local Ollama model as the judge**
(no OpenAI). Two reference-free metrics (no human-written gold answer needed):

- **faithfulness** (0–1): is every claim in the answer supported by the retrieved
  context? Low = hallucination. This automates the manual "hallucinated" grade.
- **answer_relevancy** (0–1): does the answer actually address the question?

```bash
BACKEND=local uv run python eval/ragas_eval.py --limit 8   # slow: several judge calls/sample
```

**Results** (n=6, local `llama3.1:8b` judge) — retrieval quality drives grounding:

| metric | dense-only | hybrid + reranker |
|---|---|---|
| answer_relevancy | 0.80 | **0.97** |
| faithfulness | 0.55 | **0.83** |

Turning on hybrid + reranking (recall@1 0.56 → 0.82) lifts faithfulness 0.55 →
0.83 and relevancy 0.80 → 0.97 on the same questions: better context in → better-
grounded answers out. (The hybrid run scored 5/6 — on one sample the small local
judge couldn't emit parseable output, so RAGAS dropped it; a stronger judge would
score all 6.)

> Memory note: RAGAS + `llama3.1:8b` + the retrieval stack is heavy on an 8 GB
> machine — run with other apps closed, or disable hybrid/reranker
> (`HYBRID_SEARCH_ENABLED=false RERANK_ENABLED=false`) to lighten the answer step.
> Scores land in `eval/out/ragas.json`.

---

## Project layout

```
src/bearlm/
  config.py          # BACKEND flag + all constants (OpenAI + local + projects)
  db.py              # MongoDB Atlas helpers (openai backend)
  doc_processor.py   # extract + chunk + OpenAI-embed + Atlas ingest
  text_splitter.py   # token-based RecursiveCharacterTextSplitter (local)
  local_store.py     # Ollama embeddings + multi-collection Chroma + ingest
  rerank.py          # bge-reranker-base CrossEncoder (rerank + scored)
  rag.py             # retrieve/answer + answer_grounded (gate, fallback, chat)
  projects.py        # projects, chats, ad-hoc uploads, per-PDF collections
  api.py             # FastAPI: ask, projects/chats, uploads, PDF serving
web/                 # hand-built frontend, light/dark, shared render.js
  index.html         # Ask page (course corpus Q&A)
  projects.html      # Projects: sidebar + multi-turn chat
  analytics.html     # usage analytics (OpenAI backend)
eval/                # generate_questions + retrieval/answer eval + report
```

### Key HTTP endpoints

```
POST /api/ask                                   ask the course corpus
GET  /api/courses                               live course list (from Chroma)
POST /api/projects            GET /api/projects  create / list projects
GET/DELETE /api/projects/{slug}                 fetch / delete a project
POST /api/projects/{slug}/pdfs                  add a PDF to a project
POST /api/projects/{slug}/reset                 clear chats, keep PDFs
POST /api/projects/{slug}/chats                 new chat
GET/DELETE /api/projects/{slug}/chats/{id}      fetch / delete a chat
POST /api/projects/{slug}/chats/{id}/ask        multi-turn grounded ask
POST/DELETE /api/uploads[/{id}]                 ad-hoc PDF upload / cleanup
GET  /source/{course}/{file}                    serve a corpus PDF (deep-link)
GET  /projects/{slug}/pdfs/{file}               serve a project PDF (deep-link)
```
