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
  question ─► Chroma retriever (k=4, or 15→rerank→4 if RERANK_ENABLED)
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
| `CHROMA_DIR` | `./db` | On-disk Chroma store. |
| `RETRIEVAL_K` | `4` | Chunks fed to the LLM. |
| `RERANK_ENABLED` | `false` | Rerank corpus retrieval (retrieve 15 → keep 4). |
| `RERANK_MIN_SCORE` | `0.02` | Confidence gate for project/upload answers. |
| `PROJECT_FALLBACK_TO_CORPUS` | `true` | Let project answers fall back to the corpus. |
| `MIN_RELEVANCE` | `0.30` | Vector-relevance floor (reserved). |

For the OpenAI backend, `OPENAI_API_KEY` and `MONGODB_URI` are required.

---

## Evaluation

All eval lives in `eval/`. **You must write/verify the question set yourself** —
`eval/questions.jsonl` ships with 10 auto-drafted templates and a marker; expand
it to 50–100 verified questions before trusting the numbers.

```bash
# 1. Retrieval quality (recall@1/3/5, MRR) — with and without the reranker
BACKEND=local uv run python eval/retrieval_eval.py

# 2. End-to-end answers + latency, for manual grading (exports a CSV)
uv run python eval/answer_eval.py --both       # runs local + openai on same Qs
#   then open eval/out/answers.csv and fill the `grade` column
#   (correct | partial | wrong | hallucinated)

# 3. Summary table
uv run python eval/report.py
```

`report.py` reads the artifacts above; it never fabricates numbers and shows `-`
for anything not yet measured. The OpenAI $/query figure is an **estimate** whose
assumptions you can edit at the top of `report.py`.

### Results (fill in after running the eval yourself)

| Config | recall@5 | MRR | median latency | p95 latency | $/query |
|---|---|---|---|---|---|
| local (vector) | _TODO_ | _TODO_ | _TODO_ | _TODO_ | $0.0000 |
| local (+reranker) | _TODO_ | _TODO_ | _TODO_ | _TODO_ | $0.0000 |
| openai (baseline) | _TODO_ | _TODO_ | _TODO_ | _TODO_ | ~$0.0006 (est.) |

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
eval/                # questions.jsonl + retrieval/answer eval + report
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
