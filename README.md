# BearLM 🐻

A **retrieval-augmented Q&A assistant** grounded in UC Berkeley CS/DS course
materials (CS61A, CS61B, CS70, CS188, Data 8, Data 100). Ask a question, get an
answer that is **cited back to the source** (course + section) and refuses to
answer when the corpus doesn't cover it.

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
  config.py          # BACKEND flag + all constants (OpenAI + local)
  db.py              # MongoDB Atlas helpers (openai backend)
  doc_processor.py   # extract + chunk + OpenAI-embed + Atlas ingest
  text_splitter.py   # token-based RecursiveCharacterTextSplitter (local)
  local_store.py     # Ollama embeddings + Chroma store + local ingest
  rerank.py          # bge-reranker-base CrossEncoder (local, optional)
  rag.py             # retrieve/answer, backend-dispatched; LCEL build_chain
  api.py             # FastAPI: /api/ask, /api/stats, serves web/
web/                 # hand-built frontend (index + analytics, light/dark)
eval/                # questions.jsonl + retrieval/answer eval + report
```
