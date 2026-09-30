"""Central config for BearLM.

Loads secrets from `.env` and defines the constants the rest of the pipeline
shares (model names, chunk sizes, DB/collection names). Import from here instead
of scattering magic strings across modules.

--------------------------------------------------------------------------
SUGGESTIONS — you write the code, these comments tell you what each line does.
--------------------------------------------------------------------------
"""

# TODO(you): imports
#   - `os` (to read environment variables after they're loaded)
#   - `from pathlib import Path` (nice for building the data/raw path)
#   - `from dotenv import load_dotenv`  (this is the `python-dotenv` package)
import os 
from pathlib import Path
from dotenv import load_dotenv


# TODO(you): call load_dotenv() ONCE at import time so os.environ is populated.
#   Hint: `load_dotenv()` with no args walks up from the cwd looking for `.env`.
load_dotenv()

# --- Backend selection -------------------------------------------------------
# BACKEND picks the whole stack: "openai" = OpenAI embeddings/chat + MongoDB
# Atlas Vector Search (the original); "local" = Ollama embeddings/chat + Chroma
# (the local migration). Everything downstream branches on this so the two can
# be compared side by side without deleting either path.
BACKEND = os.environ.get("BACKEND", "openai").strip().lower()
if BACKEND not in ("openai", "local"):
    raise ValueError(f"BACKEND must be 'openai' or 'local', got {BACKEND!r}")

# --- Secrets (read from .env — never hard-code the real values here) ---------
# OpenAI/Atlas secrets are only required when BACKEND == "openai", so read them
# lazily (.get, not [...]) — a pure-local install shouldn't need them at all.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MONGODB_URI = os.environ.get("MONGODB_URI")
if BACKEND == "openai":
    missing = [k for k in ("OPENAI_API_KEY", "MONGODB_URI") if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"BACKEND=openai requires {', '.join(missing)} in the environment (.env)."
        )


# --- Paths -------------------------------------------------------------------
# TODO(you): RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
#   `parents[2]` climbs: config.py -> bearlm -> src -> project root.
#   RAW_DIR is where the 6 course folders live.
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


# --- Embedding model ---------------------------------------------------------
# TODO(you): EMBEDDING_MODEL = "text-embedding-3-small"
# TODO(you): EMBEDDING_DIM   = 1536   # the vector length this model returns;
#            you'll need this exact number when you create the Atlas Vector index.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


# --- Chunking knobs ----------------------------------------------------------
# TODO(you): CHUNK_SIZE_TOKENS    = 500
# TODO(you): CHUNK_OVERLAP_TOKENS = 50
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50 


# --- MongoDB names -----------------------------------------------------------
# TODO(you): DB_NAME          = "bearlm"
# TODO(you): CHUNKS_COLL      = "chunks"
# TODO(you): QUERY_LOGS_COLL  = "query_logs"
DB_NAME = "bearlm"
CHUNKS_COLL = "chunks"
QUERY_LOGS_COLL = "query_logs"


# =============================================================================
# LOCAL BACKEND (Ollama + Chroma) — used when BACKEND == "local"
# =============================================================================
# Ollama server + models. `qwen2.5:3b` is the fallback chat model for machines
# under ~8 GB VRAM; override with OLLAMA_CHAT_MODEL in .env if you pulled a
# different one. `nomic-embed-text` produces 768-dim embeddings.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.1:8b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
LOCAL_EMBEDDING_DIM = 768  # nomic-embed-text output width; verified after re-embed

# Chroma persistent store (created on first ingest under the project root).
CHROMA_DIR = os.environ.get("CHROMA_DIR", str(Path(__file__).resolve().parents[2] / "db"))
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "bearlm_chunks")

# Retrieval knobs shared by the local chain.
RETRIEVAL_K = int(os.environ.get("RETRIEVAL_K", "4"))          # chunks fed to the LLM
RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "false").strip().lower() in ("1", "true", "yes")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_CANDIDATES = int(os.environ.get("RERANK_CANDIDATES", "15"))  # retrieved before rerank
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", "4"))            # kept after rerank
# bge-reranker relevance in [0,1] below which a chunk is "not about this
# question" — the confidence gate for grounded project/upload answers (metric-
# independent, so it works regardless of Chroma's distance function). Measured:
# off-topic pairs score ~0.000, while genuine content matches on a single
# uploaded doc score ~0.04-0.05 (and much higher on the big corpus). 0.02 keeps
# real matches while filtering pure noise. Summary/meta questions ("what is in
# this pdf") score ~0.0 regardless and are handled by a separate meta bypass in
# rag.answer_grounded rather than by this threshold. Tune against eval/.
RERANK_MIN_SCORE = float(os.environ.get("RERANK_MIN_SCORE", "0.02"))

# Token-based chunking (shared by the RecursiveCharacterTextSplitter length fn).
# tiktoken's cl100k_base is a reasonable, model-agnostic token proxy.
TOKENIZER_ENCODING = os.environ.get("TOKENIZER_ENCODING", "cl100k_base")

# --- Projects & ad-hoc uploads ----------------------------------------------
# Each project is a folder of PDFs plus its own Chroma collection. Uploads are
# ephemeral single-PDF collections scoped to one session.
PROJECTS_DIR = Path(os.environ.get("PROJECTS_DIR", str(Path(__file__).resolve().parents[2] / "data" / "projects")))
UPLOADS_DIR = Path(os.environ.get("UPLOADS_DIR", str(Path(__file__).resolve().parents[2] / "data" / "uploads")))
PROJECT_COLLECTION_PREFIX = "project_"
UPLOAD_COLLECTION_PREFIX = "upload_"

# Project-first retrieval: consult project PDFs, and fall back to the main course
# corpus only when the project doesn't strongly cover the question.
PROJECT_FALLBACK_TO_CORPUS = os.environ.get("PROJECT_FALLBACK_TO_CORPUS", "true").strip().lower() in ("1", "true", "yes")

# --- Anti-hallucination: retrieval confidence threshold ----------------------
# Chroma relevance scores are normalized to [0, 1] (higher = more relevant). A
# retrieved chunk below MIN_RELEVANCE is treated as "not really about this
# question" and dropped; if nothing clears the bar the answer is refused rather
# than guessed. This value is embedding-model-specific and worth tuning against
# eval/ — start conservative and adjust.
MIN_RELEVANCE = float(os.environ.get("MIN_RELEVANCE", "0.30"))
GROUNDED_REFUSAL = "I don't know based on the provided materials."

# --- Hybrid search (BM25 keyword + dense vector, fused with RRF) --------------
# Dense vectors catch paraphrases; BM25 catches exact terms/identifiers. Fusing
# both (reciprocal rank fusion) gives a better candidate pool than either alone.
HYBRID_SEARCH_ENABLED = os.environ.get("HYBRID_SEARCH_ENABLED", "true").strip().lower() in ("1", "true", "yes")
RRF_K = int(os.environ.get("RRF_K", "60"))  # reciprocal-rank-fusion constant

# --- Query rewriting (LLM turns a question into a better retrieval query) -----
# Off by default: it adds one LLM call (~seconds locally) per query. Enable per
# request from the UI, or globally here.
QUERY_REWRITE_ENABLED = os.environ.get("QUERY_REWRITE_ENABLED", "false").strip().lower() in ("1", "true", "yes")

# --- Analytics (local backend) -----------------------------------------------
# Query logs go to a local SQLite file (no Atlas). Powers the Analytics page,
# the "most-asked area" insight, and auto-generated practice questions.
ANALYTICS_DB = Path(os.environ.get("ANALYTICS_DB", str(Path(__file__).resolve().parents[2] / "data" / "analytics.db")))