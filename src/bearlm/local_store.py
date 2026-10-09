"""Vector store + embeddings: Ollama (nomic-embed-text, 768-dim) + on-disk Chroma.

Build the course corpus store one of two ways:
    uv run python -m bearlm.local_store            # from the shipped chunks (data/corpus_chunks.jsonl.gz)
    uv run python -m bearlm.local_store --raw      # re-chunk your own files in data/raw/<course>/
    uv run python -m bearlm.local_store --export   # write the store's chunks to the export file
    uv run python -m bearlm.local_store --verify   # count + dim only
"""

from __future__ import annotations

import functools
import gzip
import json
import shutil
import sys
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from . import config, doc_processor, text_splitter


@functools.lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    """The local embedding function (nomic-embed-text via Ollama)."""
    return OllamaEmbeddings(
        model=config.OLLAMA_EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )


def get_vectorstore(collection_name: str | None = None) -> Chroma:
    """Return a persistent Chroma collection, wired to the local embedder.

    Defaults to the main course corpus; pass a name to address a project or
    upload collection (all live under the same CHROMA_DIR).
    """
    return Chroma(
        collection_name=collection_name or config.CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=config.CHROMA_DIR,
    )


def ingest_pdf_into_collection(pdf_path, collection_name: str, course_label: str,
                               batch_size: int = 32) -> int:
    """Extract, token-split, embed and add one PDF to a Chroma collection.

    Used by projects and ad-hoc uploads. `course_label` is stored as the chunk's
    course (shown in citations); `source` is the PDF filename so citations can
    deep-link to it. Returns the number of chunks added.
    """
    from pathlib import Path as _Path

    pdf_path = _Path(pdf_path)
    vs = get_vectorstore(collection_name)
    pieces = doc_processor.extract_pdf(pdf_path)  # [(page_text, "p.N"), ...]

    texts, metadatas, ids = [], [], []
    n = 0
    for raw_text, section in pieces:
        for piece in text_splitter.split_text(raw_text):
            # IDs are namespaced by source filename so adding a second PDF to a
            # project can't collide with / overwrite an existing PDF's chunks.
            cid = f"{collection_name}:{pdf_path.name}:{n}"
            texts.append(piece)
            metadatas.append({
                "course": course_label,
                "section": section,
                "source": pdf_path.name,
                "chunk_id": cid,
            })
            ids.append(cid)
            n += 1

    for i in range(0, len(texts), batch_size):
        _add_batch_with_retry(
            vs, texts[i : i + batch_size], ids[i : i + batch_size], metadatas[i : i + batch_size]
        )
    return n


def _reset_store() -> None:
    """Delete the on-disk Chroma directory so re-ingest starts clean (no dupes)."""
    get_embeddings.cache_clear()
    p = Path(config.CHROMA_DIR)
    if p.exists():
        shutil.rmtree(p)


def _add_batch_with_retry(vs, texts, ids, metadatas, retries: int = 6) -> None:
    """Embed+add one batch, retrying transient Ollama/socket failures with backoff.

    On Windows a large ingest can exhaust ephemeral sockets (WSAENOBUFS:
    "system lacked sufficient buffer space"). That is transient — pausing lets
    the TIME_WAIT sockets drain — so we back off and retry rather than aborting
    the whole run and losing progress.
    """
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            vs.add_texts(texts=texts, ids=ids, metadatas=metadatas)
            return
        except Exception as e:
            if attempt == retries:
                raise
            print(f"    batch failed (attempt {attempt}/{retries}): {type(e).__name__}; "
                  f"backing off {delay:.0f}s and retrying")
            time.sleep(delay)
            delay = min(delay * 2, 30.0)


def _fill_store(rows, batch_size: int = 32, pause: float = 0.15) -> int:
    """Replace the corpus store with (id, text, metadata) rows, embedding locally.

    Small batches, a brief pause and per-batch retry so a large corpus doesn't
    exhaust Windows sockets mid-run.
    """
    _reset_store()
    vs = get_vectorstore()
    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        _add_batch_with_retry(vs, texts=[r[1] for r in batch], ids=[r[0] for r in batch],
                              metadatas=[r[2] for r in batch])
        total += len(batch)
        if pause:
            time.sleep(pause)
        if total % 1000 < batch_size:
            print(f"  embedded {total}/{len(rows)}")
    print(f"[ingest] total embedded: {total}")
    return total


def ingest_raw(**kw) -> int:
    """Re-chunk data/raw/<course>/ (private course files) and rebuild the store."""
    rows = []
    for course_dir in doc_processor.iter_course_dirs():
        chunks = doc_processor.build_chunks_for_course(course_dir)
        # Stable, human-readable ids so eval gold_chunk_ids can reference them.
        for i, c in enumerate(chunks):
            cid = f"{c.course}:{i}"
            rows.append((cid, c.text, {"course": c.course, "section": c.section,
                                       "source": c.filename, "chunk_id": cid}))
        print(f"  {course_dir.name}: {len(chunks)} chunks")
    return _fill_store(rows, **kw)


def ingest_export(path=None, **kw) -> int:
    """Rebuild the store from the public chunk export (no course files needed)."""
    with gzip.open(path or config.CHUNKS_EXPORT, "rt", encoding="utf-8") as f:
        rows = [(r["id"], r["text"], r["metadata"]) for r in map(json.loads, f)]
    print(f"[ingest] {len(rows)} chunks from {path or config.CHUNKS_EXPORT}")
    return _fill_store(rows, **kw)


def export_chunks(path=None) -> int:
    """Write the corpus store's chunk text + metadata (no embeddings, no PDFs) as gzipped JSONL."""
    got = get_vectorstore()._collection.get(include=["documents", "metadatas"])
    rows = sorted(zip(got["ids"], got["documents"], got["metadatas"]))
    # mtime=0 keeps the file byte-identical across re-exports of the same store.
    with open(path or config.CHUNKS_EXPORT, "wb") as raw, \
            gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as g:
        for cid, text, meta in rows:
            g.write((json.dumps({"id": cid, "text": text, "metadata": meta}, ensure_ascii=False) + "\n").encode())
    print(f"[export] {len(rows)} chunks -> {path or config.CHUNKS_EXPORT}")
    return len(rows)


def drop_collection(collection_name: str) -> None:
    """Delete a Chroma collection (used when deleting a project or upload)."""
    try:
        get_vectorstore(collection_name).delete_collection()
    except Exception:
        pass


def collection_count(collection_name: str) -> int:
    """Number of vectors in a collection (0 if it does not exist)."""
    try:
        return get_vectorstore(collection_name)._collection.count()
    except Exception:
        return 0


def get_docs_by_ids(collection_name: str, ids: list[str]) -> list:
    """Fetch chunks by id as Documents; missing ids/collections are skipped ([] on error)."""
    from langchain_core.documents import Document

    if not ids:
        return []
    try:
        got = get_vectorstore(collection_name)._collection.get(
            ids=list(ids), include=["documents", "metadatas"])
    except Exception:
        return []
    return [Document(page_content=t or "", metadata=m or {})
            for t, m in zip(got["documents"], got["metadatas"])]


def delete_pdf_from_collection(collection_name: str, filename: str) -> None:
    """Delete all chunks that came from one PDF (matched by `source` metadata)."""
    try:
        get_vectorstore(collection_name)._collection.delete(where={"source": filename})
    except Exception:
        pass


def distinct_courses() -> list[str]:
    """Sorted list of distinct course labels present in the Chroma store."""
    col = get_vectorstore()._collection
    if col.count() == 0:
        return []
    got = col.get(include=["metadatas"])
    return sorted({(m or {}).get("course") for m in got["metadatas"] if m and m.get("course")})


def verify() -> None:
    """Report the stored vector count and embedding dimension."""
    vs = get_vectorstore()
    col = vs._collection  # chromadb Collection
    count = col.count()
    dim = None
    if count:
        got = col.get(limit=1, include=["embeddings"])
        embs = got.get("embeddings")
        if embs is not None and len(embs):
            dim = len(embs[0])
    print(f"[verify] Chroma collection '{config.CHROMA_COLLECTION}' @ {config.CHROMA_DIR}")
    print(f"[verify] vector_count = {count}")
    print(f"[verify] vector_dim   = {dim}  (expected {config.EMBEDDING_DIM})")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--export" in args:
        export_chunks()
    elif "--verify" not in args:
        ingest_raw() if "--raw" in args else ingest_export()
    verify()
