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

# --- Secrets (read from .env — never hard-code the real values here) ---------
# TODO(you): OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
# TODO(you): MONGODB_URI    = os.environ["MONGODB_URI"]
#   Using os.environ[...] (not .get) means the program fails loudly if a key is
#   missing, which is what you want for required secrets.
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MONGODB_URI = os.environ["MONGODB_URI"]


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