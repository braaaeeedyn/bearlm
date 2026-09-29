"""Stage 4 — Ingestion pipeline.

Walk data/raw/, extract text, chunk it, embed each chunk, and insert the chunks
into MongoDB Atlas (`bearlm.chunks`). After this runs, you create the Vector
Search index on the `embedding` field in the Atlas UI (that step is manual).

The corpus mixes two formats, so extraction is split by type:
  - PDF folders:      cs61a, cs61b, cs70, cs188   (one .pdf each)
  - Markdown folders: data8/chapters, data100/content   (many .md files)

Run it (once you've filled it in) with:
    uv run python -m bearlm.doc_processor

--------------------------------------------------------------------------
SUGGESTIONS ONLY. Each function has a signature + a step-by-step recipe in
comments. You write the actual lines. `raise NotImplementedError` marks the
spots to replace. Do them roughly top-to-bottom.
--------------------------------------------------------------------------
"""

# TODO(you): imports you'll likely need
#   - `from pathlib import Path`
#   - `from dataclasses import dataclass`            (a tidy way to model a Chunk)
#   - `import pdfplumber`  OR  `import fitz`  (fitz = pymupdf; faster, either works)
#   - `from openai import OpenAI`
#   - `from . import config, db`
import json
from pathlib import Path
from dataclasses import dataclass
import pdfplumber
from openai import OpenAI
from . import config, db


# A single unit we store. Keeping it a dataclass makes the metadata explicit and
# stops you from forgetting a field when you build the Mongo document.
# TODO(you): turn this into a real dataclass, e.g.:
#   @dataclass
#   class Chunk:
#       text: str
#       course: str       # "cs61a", "data100", ...
#       filename: str     # source file name
#       section: str      # page number (PDF) or section/heading (markdown)
@dataclass
class Chunk:
    text: str
    course: str
    filename: str
    section: str


# ---------------------------------------------------------------------------
# 1. EXTRACTION  — raw file  ->  text (+ where it came from)
# ---------------------------------------------------------------------------

def extract_pdf(path):  # (Path) -> list[tuple[str, str]]
    """Return a list of (page_text, section_label) for one PDF.

    Why per-page? So you can record which page a chunk came from (good citations).

    Recipe (pdfplumber version):
      1. out = []
      2. with pdfplumber.open(path) as pdf:
             for i, page in enumerate(pdf.pages, start=1):
                 text = page.extract_text() or ""      # None if page has no text
                 if text.strip():
                     out.append((text, f"p.{i}"))
      3. return out

    (pymupdf version: `doc = fitz.open(path)`; iterate `doc`; `page.get_text()`.)
    """
    output = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                output.append((text, f"p.{i}"))
    return output

def extract_markdown(path):  # (Path) -> list[tuple[str, str]]
    """Return a list of (text, section_label) for one .md / MyST file.

    Simplest correct version: treat the whole file as one blob and let the
    chunker split it later.
      1. text = path.read_text(encoding="utf-8", errors="ignore")
      2. return [(text, path.stem)] if text.strip() else []

    Nicer later: split on markdown headings (`#`, `##`) and use the heading as the
    section label so citations point at a named section. Start simple; upgrade if
    you have time.
    """
    text = path.read_text(encoding="utf-8", errors = "ignore")
    return [(text, path.stem)] if text.strip() else []


def extract_notebook(path):  # (Path) -> list[tuple[str, str]]
    """Return a list of (text, section_label) for one Jupyter notebook (.ipynb).

    Data 8 and Data 100 are Jupyter Book repos whose real chapters live in
    notebooks, not markdown — so this is where most of that content comes from.

    A notebook is JSON with a top-level "cells" list. Each cell has a
    "cell_type" ("markdown" or "code") and a "source" (a list of line strings,
    or occasionally a single string). We keep markdown cells (explanations) and
    code cells (the pandas/numpy examples that are often the actual answer), and
    skip cell outputs (long dataframes / repr noise). The whole notebook is
    returned as one blob labeled by its filename; the chunker splits it later.
    """
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except (json.JSONDecodeError, ValueError):
        return []
    parts = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") not in ("markdown", "code"):
            continue
        src = cell.get("source", "")
        text = "".join(src) if isinstance(src, list) else src
        if text.strip():
            parts.append(text)
    blob = "\n\n".join(parts)
    return [(blob, path.stem)] if blob.strip() else []


# ---------------------------------------------------------------------------
# 2. CHUNKING  — long text  ->  ~500-token windows with ~50-token overlap
# ---------------------------------------------------------------------------

def chunk_text(text):  # (str) -> list[str]
    """Split one long string into overlapping chunks.

    Real token counting needs `tiktoken`, which isn't in the deps. A solid
    approximation: chunk by WORDS, using ~0.75 words/token, so 500 tokens ≈ 375
    words and 50-token overlap ≈ 38 words. (Feel free to just use 400/50 words.)

    Recipe:
      1. words = text.split()
      2. size    = int(config.CHUNK_SIZE_TOKENS * 0.75)
         overlap = int(config.CHUNK_OVERLAP_TOKENS * 0.75)
         step    = size - overlap            # how far the window advances
      3. loop `start` from 0 to len(words) in steps of `step`:
             piece = " ".join(words[start:start + size])
             if piece.strip(): append to a results list
      4. return the results

    Watch out: make sure `step > 0`, or you'll loop forever.
    """
    words = text.split()
    size = int(config.CHUNK_SIZE_TOKENS * 0.75)
    overlap = int(config.CHUNK_OVERLAP_TOKENS * 0.75)
    step = max(size - overlap, 1)
    results = []
    for start in range(0, len(words), step):
        piece = " ".join(words[start:start + size])
        if piece.strip():
            results.append(piece)
    return results
    


# ---------------------------------------------------------------------------
# 3. EMBEDDING  — text  ->  vector[1536]  (OpenAI)
# ---------------------------------------------------------------------------

_client = None


def _get_client():
    """Construct the OpenAI client once, lazily (so local-only installs that lack
    OPENAI_API_KEY can still import this module for its extraction helpers)."""
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def embed_batch(texts):  # (list[str]) -> list[list[float]]
    """Embed a batch of chunk texts in one API call.

    Batching matters: one request of 100 texts is far cheaper/faster than 100
    requests. The embeddings endpoint accepts a list as `input`.

    Recipe:
      1. client = OpenAI()          # reads OPENAI_API_KEY from the env automatically
      2. resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
      3. return [d.embedding for d in resp.data]   # order matches `texts`

    Tip: create the client ONCE (pass it in, or module-level) rather than per call.
    Keep batches ~50–100 chunks to stay under token limits.
    """
    response = _get_client().embeddings.create(model = config.EMBEDDING_MODEL, input = texts)
    return [d.embedding for d in response.data]


# ---------------------------------------------------------------------------
# 4. ORCHESTRATION  — tie 1→2→3 together, per course folder
# ---------------------------------------------------------------------------

def iter_course_dirs():  # () -> Iterator[Path]
    """Yield each course folder under data/raw/ (cs61a, data100, ...).

    Recipe:
      for p in sorted(config.RAW_DIR.iterdir()):
          if p.is_dir():
              yield p
    """
    for p in sorted(config.RAW_DIR.iterdir()):
        if p.is_dir():
            yield p


def build_chunks_for_course(course_dir, chunker=chunk_text):  # (Path, callable) -> list[Chunk]
    """Extract + chunk every file in one course folder into Chunk objects.

    Recipe:
      1. course = course_dir.name           # this becomes the `course` metadata
      2. chunks = []
      3. for path in course_dir.rglob("*"):      # rglob = recurse into subfolders
             if path.suffix.lower() == ".pdf":
                 pieces = extract_pdf(path)
             elif path.suffix.lower() in {".md", ".myst"}:
                 pieces = extract_markdown(path)
             else:
                 continue                        # skip images, .git, etc.
             for raw_text, section in pieces:
                 for piece in chunk_text(raw_text):
                     chunks.append(Chunk(piece, course, path.name, section))
      4. return chunks

    Note: data8/data100 are cloned repos with their own .git — rglob("*.md") or the
    suffix filter above naturally skips the git internals.
    """
    course = course_dir.name
    chunks = []
    for path in course_dir.rglob("*"):
        if path.suffix.lower() == ".pdf":
            pieces = extract_pdf(path)
        elif path.suffix.lower() in {".md", ".myst"}:
            pieces = extract_markdown(path)
        elif path.suffix.lower() == ".ipynb":
            pieces = extract_notebook(path)
        else:
            continue
        for raw_text, section in pieces:
            for piece in chunker(raw_text):
                chunks.append(Chunk(piece, course, path.name, section))
    return chunks


def ingest():  # () -> None
    """Full run: every course -> chunks -> embeddings -> MongoDB.

    Recipe:
      1. coll = db.get_chunks_collection()
      2. (optional, makes re-runs safe) coll.delete_many({}) so you don't insert
         duplicates every time you run this while developing.
      3. for course_dir in iter_course_dirs():
             chunks = build_chunks_for_course(course_dir)
             then walk `chunks` in batches of ~100:
                 vectors = embed_batch([c.text for c in batch])
                 docs = [
                     {"text": c.text, "embedding": v, "course": c.course,
                      "filename": c.filename, "section": c.section}
                     for c, v in zip(batch, vectors)
                 ]
                 coll.insert_many(docs)
             print progress per course (counts) so a long run isn't silent.
      4. print a final total.

    After this succeeds: go to Atlas UI -> create a Vector Search index on
    `embedding`, dimensions = config.EMBEDDING_DIM (1536), similarity = cosine.
    """
    coll = db.get_chunks_collection()
    coll.delete_many({})
    total = 0
    for course_dir in iter_course_dirs():
        chunks = build_chunks_for_course(course_dir)
        course_count = 0
        for i in range(0, len(chunks), 100):
            batch = chunks[i:i+100]
            vectors = embed_batch([c.text for c in batch])
            docs = [{"text": c.text, "embedding": v, "course": c.course, "filename": c.filename, "section": c.section} for c, v in zip(batch, vectors)]
            coll.insert_many(docs)
            course_count += len(docs)
        print(f"{course_dir.name}: inserted {course_count} chunks")
        total += course_count
    print(f"Total: inserted {total} chunks")

if __name__ == "__main__":
    # TODO(you): a good first smoke test before the full ingest:
    #   from . import db
    #   print("Mongo reachable:", db.ping())
    # then, when ready:
    #   ingest()
    from . import db
    print("Mongo reachable:", db.ping())
    ingest()
