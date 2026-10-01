"""Ingestion for the openai backend: extract, chunk, embed, insert into Atlas.

Run: uv run python -m bearlm.doc_processor
"""

import json
from dataclasses import dataclass

import pdfplumber
from openai import OpenAI

from . import config, db


@dataclass
class Chunk:
    text: str
    course: str
    filename: str
    section: str


def extract_pdf(path):
    """One PDF -> list of (page_text, "p.N"). Per-page so citations can point at a page."""
    output = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                output.append((text, f"p.{i}"))
    return output


def extract_markdown(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [(text, path.stem)] if text.strip() else []


def extract_notebook(path):
    """One .ipynb -> its markdown+code cells as one blob (cell outputs are skipped)."""
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


def chunk_text(text):
    """Word-window chunker (~0.75 words/token) with overlap. Legacy; see text_splitter."""
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


_client = None


def _get_client():
    # Lazy so local-only installs without OPENAI_API_KEY can still import the extractors.
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def embed_batch(texts):
    response = _get_client().embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in response.data]


def iter_course_dirs():
    for p in sorted(config.RAW_DIR.iterdir()):
        if p.is_dir():
            yield p


def build_chunks_for_course(course_dir, chunker=chunk_text):
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


def ingest():
    coll = db.get_chunks_collection()
    coll.delete_many({})  # wipe so re-runs don't duplicate
    total = 0
    for course_dir in iter_course_dirs():
        chunks = build_chunks_for_course(course_dir)
        course_count = 0
        for i in range(0, len(chunks), 100):
            batch = chunks[i:i + 100]
            vectors = embed_batch([c.text for c in batch])
            docs = [
                {"text": c.text, "embedding": v, "course": c.course,
                 "filename": c.filename, "section": c.section}
                for c, v in zip(batch, vectors)
            ]
            coll.insert_many(docs)
            course_count += len(docs)
        print(f"{course_dir.name}: inserted {course_count} chunks")
        total += course_count
    print(f"Total: inserted {total} chunks")


if __name__ == "__main__":
    print("Mongo reachable:", db.ping())
    ingest()
