"""Central config: BACKEND flag, secrets, and shared constants."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parents[2]

# "local" = Ollama + Chroma; "openai" = OpenAI + MongoDB Atlas.
BACKEND = os.environ.get("BACKEND", "openai").strip().lower()
if BACKEND not in ("openai", "local"):
    raise ValueError(f"BACKEND must be 'openai' or 'local', got {BACKEND!r}")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MONGODB_URI = os.environ.get("MONGODB_URI")
if BACKEND == "openai":
    missing = [k for k in ("OPENAI_API_KEY", "MONGODB_URI") if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"BACKEND=openai requires {', '.join(missing)} in .env.")

RAW_DIR = _ROOT / "data" / "raw"

# --- OpenAI backend ---
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
DB_NAME = "bearlm"
CHUNKS_COLL = "chunks"
QUERY_LOGS_COLL = "query_logs"

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

# --- Local backend (Ollama + Chroma) ---
# localhost costs ~2 s per new connection on Windows (IPv6 fallback)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.1:8b")  # qwen2.5:3b if <8GB VRAM
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
# Longest legit answer seen was 815 tokens; 1024 adds ~25% headroom and bounds a runaway to ~31 s.
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "1024"))
OLLAMA_TIMEOUT_S = float(os.environ.get("OLLAMA_TIMEOUT_S", "120"))  # httpx stall timeout (connect / next chunk)
LOCAL_EMBEDDING_DIM = 768

CHROMA_DIR = os.environ.get("CHROMA_DIR", str(_ROOT / "db"))
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "bearlm_chunks")
TOKENIZER_ENCODING = os.environ.get("TOKENIZER_ENCODING", "cl100k_base")

# --- Retrieval + reranking ---
RETRIEVAL_K = int(os.environ.get("RETRIEVAL_K", "4"))
RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "true").strip().lower() in ("1", "true", "yes")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_CANDIDATES = int(os.environ.get("RERANK_CANDIDATES", "8"))
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", "4"))
# "auto" = cuda if torch sees a GPU, else cpu; or force "cpu" / "cuda".
RERANK_DEVICE = os.environ.get("RERANK_DEVICE", "auto").strip().lower()
# Confidence gate (projects/uploads only): real matches mostly score >= 0.02, but off-topic
# probes are not reliably below it (5/10 >= 0.02, max 0.68); kept as is by decision.
RERANK_MIN_SCORE = float(os.environ.get("RERANK_MIN_SCORE", "0.02"))

# --- Hybrid search (BM25 + dense, fused with reciprocal rank fusion) ---
HYBRID_SEARCH_ENABLED = os.environ.get("HYBRID_SEARCH_ENABLED", "true").strip().lower() in ("1", "true", "yes")
RRF_K = int(os.environ.get("RRF_K", "60"))

# --- Query rewriting (one extra LLM call per query; off by default) ---
QUERY_REWRITE_ENABLED = os.environ.get("QUERY_REWRITE_ENABLED", "false").strip().lower() in ("1", "true", "yes")

# --- Projects, uploads, fallback ---
PROJECTS_DIR = Path(os.environ.get("PROJECTS_DIR", str(_ROOT / "data" / "projects")))
UPLOADS_DIR = Path(os.environ.get("UPLOADS_DIR", str(_ROOT / "data" / "uploads")))
PROJECT_COLLECTION_PREFIX = "project_"
UPLOAD_COLLECTION_PREFIX = "upload_"
PROJECT_FALLBACK_TO_CORPUS = os.environ.get("PROJECT_FALLBACK_TO_CORPUS", "true").strip().lower() in ("1", "true", "yes")
# Per-project retrieval memory: max chunk ids stored (LRU; 0 disables) and how
# many of the most recent are merged into one query's rerank pool.
PROJECT_MEMORY_SIZE = int(os.environ.get("PROJECT_MEMORY_SIZE", "40"))
PROJECT_MEMORY_MERGE = int(os.environ.get("PROJECT_MEMORY_MERGE", "8"))
# Near-repeat reuse: a project question within this cosine of an earlier one
# reuses that question's retrieval (and gate outcome). <= 0 disables (the default:
# follow-ups embed too close to the previous turn, eval/speed_experiments.json).
NEAR_REPEAT_THRESHOLD = float(os.environ.get("NEAR_REPEAT_THRESHOLD", "0"))
MIN_RELEVANCE = float(os.environ.get("MIN_RELEVANCE", "0.30"))
GROUNDED_REFUSAL = "I don't know based on the provided materials."

# --- Analytics (local SQLite) ---
ANALYTICS_DB = Path(os.environ.get("ANALYTICS_DB", str(_ROOT / "data" / "analytics.db")))
