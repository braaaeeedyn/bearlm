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
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.1:8b")  # qwen2.5:3b if <8GB VRAM
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
LOCAL_EMBEDDING_DIM = 768

CHROMA_DIR = os.environ.get("CHROMA_DIR", str(_ROOT / "db"))
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "bearlm_chunks")
TOKENIZER_ENCODING = os.environ.get("TOKENIZER_ENCODING", "cl100k_base")

# --- Retrieval + reranking ---
RETRIEVAL_K = int(os.environ.get("RETRIEVAL_K", "4"))
RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "true").strip().lower() in ("1", "true", "yes")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_CANDIDATES = int(os.environ.get("RERANK_CANDIDATES", "15"))
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", "4"))
# Confidence gate: bge relevance below this is dropped. Off-topic scores ~0.0,
# real matches >= ~0.04, so 0.02 keeps real matches and filters noise.
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
MIN_RELEVANCE = float(os.environ.get("MIN_RELEVANCE", "0.30"))
GROUNDED_REFUSAL = "I don't know based on the provided materials."

# --- Analytics (local SQLite) ---
ANALYTICS_DB = Path(os.environ.get("ANALYTICS_DB", str(_ROOT / "data" / "analytics.db")))
