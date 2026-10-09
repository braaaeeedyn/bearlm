# BearLM 🐻

A **local, private study assistant** for UC Berkeley CS/DS courses (CS61A, CS61B,
CS70, CS188, CS189, Data 8, Data 100, Data 101, Data 144, and a Python data-science
handbook). Ask a question and get an answer **grounded in the course material and
cited back to it** (course + page or section). When the material doesn't cover a
question, BearLM says so instead of guessing.

**Projects** let you bring your own PDFs: hold multi-turn conversations over them,
with answers that fall back to the course material only when your documents
don't cover the question.

Everything runs on your own machine: the model (Ollama), the vector store
(Chroma) and the web app. **No API keys, no accounts, no cloud services.**

---

## What you need

| | Requirement |
|---|---|
| OS | Windows, Linux or macOS |
| Python | 3.13 (installed for you by `uv`) |
| Ollama | <https://ollama.com/download> |
| GPU | Recommended: ~8 GB VRAM (NVIDIA, e.g. RTX 4060). Works on CPU, just slower. |
| Disk | ~10 GB (Ollama models ~5 GB, Python packages incl. PyTorch ~3 GB, vector store ~150 MB) |

## Setup

### 1. Get the code
```bash
git clone https://github.com/braaaeeedyn/bearlm.git
cd bearlm
```

### 2. Install `uv` (Python package manager)
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Install Ollama and pull the two models
Install Ollama from <https://ollama.com/download> and make sure it is running
(`ollama list` should work). Then:
```bash
ollama pull nomic-embed-text   # embeddings
ollama pull llama3.1:8b        # answer model (~5 GB; needs ~8 GB VRAM or RAM)
```
Under ~8 GB of VRAM, use a smaller model instead: `ollama pull qwen2.5:3b`, then
create a `.env` file containing `OLLAMA_CHAT_MODEL=qwen2.5:3b`.

### 4. Install the Python dependencies
```bash
uv sync
```
On Windows and Linux this installs a CUDA build of PyTorch for the reranker; it
falls back to the CPU automatically when there is no NVIDIA GPU.

### 5. Build the course search index
The repo ships the course material as **text chunks**
(`data/corpus_chunks.jsonl.gz`, ~10.4k chunks). This step embeds them on your
machine into a local Chroma store in `./db`:
```bash
uv run python -m bearlm.local_store
```
It takes a few minutes and prints the stored count at the end (expect
`vector_count = 10394`, `vector_dim = 768`).

### 6. Run it
```bash
uv run uvicorn bearlm.api:app
```
Open <http://localhost:8000>. The first question takes longer while the models
load. Other pages: `/projects` (your own PDFs) and `/analytics` (usage stats).

No `.env` file is needed: every setting has a working default. To change one,
copy `.env.example` to `.env` and edit it (see [Configuration](#configuration)).

---

## About the course PDFs

The original course PDFs and notebooks are **not included** (they belong to their
courses). Only the extracted text chunks are, which is all BearLM needs to answer
and cite. Citations still name the course and page, but clicking one opens the
PDF only if you have that file locally under `data/raw/<course>/`.

**Adding your own course files** (optional): put `.pdf`, `.md` or `.ipynb` files
in `data/raw/<course>/` (the folder name becomes the course label), then rebuild
the index from them:
```bash
uv run python -m bearlm.local_store --raw      # re-chunk data/raw (replaces the shipped chunks)
uv run python -m bearlm.local_store --export   # optional: write the index's chunks to data/corpus_chunks.jsonl.gz
uv run python -m bearlm.local_store --verify   # print the stored count + dimension
```
`--raw` rebuilds the whole index from `data/raw` only, so include every course
you want to keep. For your own documents, Projects are usually simpler.

---

## Using it

**Ask page** (`/`): questions about the course material. Answers cite their
sources inline, e.g. `(cs188 p.80)`, and list them as links under the answer.

**Projects** (`/projects`): a named collection of your PDFs.
- Add PDFs from the sidebar; each is embedded into the project's own collection.
- Hold multiple persistent chats per project. Follow-ups ("what about its top
  speed?") use the conversation so far.
- Answers come from your PDFs when they cover the question, and fall back to the
  course material otherwise. You can limit or turn off that fallback per project.
- **Reset context** clears a project's chats but keeps its PDFs.

**Chat with a PDF**: upload one PDF for a throwaway session answered strictly
from that document.

**How it avoids making things up:**
- The prompt allows answers only from the retrieved context, and asks for an
  exact refusal ("I don't know based on the …") when nothing relevant is found.
- For projects and uploads, a reranker scores each retrieved chunk, and chunks
  below `RERANK_MIN_SCORE` are dropped; if none are left, the answer is refused.
  This gate is deliberately loose (some off-topic questions pass it), so the
  strict prompt and citations are the main safeguard.
- Inline citations that name a source the model wasn't given are removed after
  generation, so every remaining citation points at a chunk that was actually used.

Your data stays local: projects in `data/projects/`, uploads in `data/uploads/`,
the index in `db/`, usage logs in `data/analytics.db` (all gitignored).

---

## Troubleshooting

- **"Can't reach Ollama … Start Ollama"** (HTTP 503): Ollama is not running. Start it (`ollama serve` or the Ollama app) and ask again.
- **"… rejected the request"** naming a model: that model isn't pulled. Run the `ollama pull` command from step 3.
- **"… may be out of GPU memory"** (HTTP 503): Ollama ran out of VRAM. BearLM already retried once. Ask again; if it keeps happening, set `OLLAMA_NUM_CTX=6144` and/or `RERANK_DEVICE=cpu` in `.env`, use a smaller `OLLAMA_CHAT_MODEL`, and don't run the server and an eval at the same time.
- **"… did not answer within 120 s"** (HTTP 503): Ollama stalled; see `OLLAMA_TIMEOUT_S`. Ask again.
- **Answers say "I don't know" for everything**: the index is empty. Run step 5 and check that `--verify` prints `vector_count = 10394`.
- **Clicking a citation shows "This course file isn't on this machine"**: expected; the course PDFs aren't published. See [About the course PDFs](#about-the-course-pdfs).

---

## Configuration

Every setting has a default in `src/bearlm/config.py` and can be overridden in
`.env` (template: `.env.example`).

| Variable | Default | What it does |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server. `127.0.0.1`, not `localhost`: on Windows `localhost` adds ~2 s per connection. |
| `OLLAMA_CHAT_MODEL` | `llama3.1:8b` | Answer model (`qwen2.5:3b` for <8 GB VRAM). |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model (768-dim). Changing it requires rebuilding the index (step 5). |
| `OLLAMA_NUM_CTX` | `8192` | Answer model context window. |
| `OLLAMA_NUM_PREDICT` | `1024` | Max tokens per answer; a cut-off answer ends with "_(Answer cut off at the length limit.)_". |
| `OLLAMA_TIMEOUT_S` | `120` | Max wait (seconds) for Ollama to connect or send the next streamed chunk; a stall timeout, not a total time limit. |
| `CHROMA_DIR` | `./db` | On-disk Chroma store. |
| `CHROMA_COLLECTION` | `bearlm_chunks` | Collection holding the course chunks. |
| `RETRIEVAL_K` | `4` | Chunks given to the model when reranking is off. |
| `RERANK_ENABLED` | `true` | Rerank course retrieval: fetch `RERANK_CANDIDATES`, keep `RERANK_TOP_N`. |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder reranker (downloaded on first use, ~1 GB). |
| `RERANK_CANDIDATES` | `8` | Candidates the reranker scores (8 was ~2x faster than 15 at equal recall). |
| `RERANK_TOP_N` | `4` | Chunks kept after reranking. |
| `RERANK_DEVICE` | `auto` | `auto` (CUDA if available, else CPU), `cpu` or `cuda`. On CUDA it loads in fp16 and falls back to CPU if loading fails. |
| `RERANK_MIN_SCORE` | `0.02` | Confidence gate for project/upload answers. |
| `PROJECT_FALLBACK_TO_CORPUS` | `true` | Let project answers fall back to the course material. |
| `PROJECT_MEMORY_SIZE` | `40` | Per-project memory of chunks that answered earlier questions (`0` disables). |
| `PROJECT_MEMORY_MERGE` | `8` | Remembered chunks added to each project question's rerank pool. |
| `HYBRID_SEARCH_ENABLED` | `true` | Combine keyword (BM25) and vector search. |
| `QUERY_REWRITE_ENABLED` | `false` | Rewrite each question with the model before searching (one extra model call). |
| `PROJECTS_DIR`, `UPLOADS_DIR` | `./data/projects`, `./data/uploads` | Where projects and uploads are stored. |

---

## Evaluation

The eval lives in `eval/` and runs against your local index (step 5 first).
The question set (`eval/questions_generated.jsonl`) was generated from real
chunks, so each question's source chunk is its exact gold label.

```bash
uv run python eval/retrieval_eval.py           # recall@k, precision@k, MRR: dense vs hybrid vs hybrid+rerank
uv run python eval/answer_eval.py --limit 10   # answers: refusal, citation correctness, faithfulness,
                                               #   completeness (local model as judge; --no-judge to skip), latency
uv run python eval/report.py [--json]          # summary of both
uv run python eval/generate_questions.py --per-course 5   # regenerate the question set
```

Latest results (50 questions, RTX 4060): hybrid+rerank recall@5 **0.86**, MRR
**0.84**; faithfulness **0.84**, completeness **0.87**, citation correctness
**0.81**; median answer **~3.8 s**; cost **$0** per query. Faithfulness and
completeness are graded by the same 8B model, so treat them as rough estimates.

Tests (offline, no Ollama needed): `uv run --with pytest python -m pytest -q tests`

---

## Project layout

```
src/bearlm/
  config.py          # all settings + defaults
  doc_processor.py   # extract PDF / Markdown / notebook text from data/raw
  text_splitter.py   # token-based chunking
  local_store.py     # Ollama embeddings + Chroma; build the index (shipped chunks or data/raw)
  search.py          # hybrid BM25 + vector search
  rerank.py          # bge-reranker cross-encoder (CUDA or CPU)
  rag.py             # retrieval + cited answers; grounded answers for projects/uploads
  projects.py        # projects, chats, uploads, per-project memory
  analytics.py       # local SQLite usage log + practice questions
  api.py             # FastAPI app: pages + JSON endpoints
web/                 # frontend (Ask, Projects, Analytics pages)
eval/                # retrieval + answer eval, report, question generation
data/corpus_chunks.jsonl.gz   # the course text chunks (public); PDFs are not included
tests/               # offline pytest suite
```

**Docs:** `PROJECT_HANDOFF.md` describes how the app works now;
`PROGRESS.md` is the full history of what changed and why.
