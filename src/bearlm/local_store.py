"""Local vector store + embeddings (Ollama + Chroma) — BACKEND == "local".

Phase 2 of the migration. Mirrors what `doc_processor` + `db` do for the OpenAI/
Atlas backend, but with:
  - embeddings:   OllamaEmbeddings(model="nomic-embed-text")   -> 768-dim
  - vector store: Chroma(persist_directory=config.CHROMA_DIR)

The corpus is re-extracted from ``data/raw/`` (fully local, no Atlas needed) and
re-embedded. Two chunkers are available:
  - default: the token-based RecursiveCharacterTextSplitter (Phase 2 spec).
  - --legacy-splitter: the original word-window chunker, which reproduces the
    exact 2,241-chunk corpus for a like-for-like count comparison.

Run:
    BACKEND=local uv run python -m bearlm.local_store            # token splitter
    BACKEND=local uv run python -m bearlm.local_store --legacy-splitter
    BACKEND=local uv run python -m bearlm.local_store --verify   # count + dim only
"""

from __future__ import annotations

import functools
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


def get_retriever(k: int | None = None, collection_name: str | None = None):
    """LangChain retriever over a Chroma collection (k defaults to config.RETRIEVAL_K)."""
    return get_vectorstore(collection_name).as_retriever(
        search_kwargs={"k": k or config.RETRIEVAL_K}
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
            texts.append(piece)
            metadatas.append({
                "course": course_label,
                "section": section,
                "source": pdf_path.name,
                "chunk_id": f"{collection_name}:{n}",
            })
            ids.append(f"{collection_name}:{n}")
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


def ingest_local(use_legacy_splitter: bool = False, batch_size: int = 32,
                 pause: float = 0.15) -> int:
    """Re-extract data/raw, embed with Ollama, and store in Chroma. Returns count.

    Uses small batches, a brief pause between them, and per-batch retry/backoff
    so a large corpus doesn't exhaust Windows sockets mid-run.
    """
    chunker = doc_processor.chunk_text if use_legacy_splitter else text_splitter.split_text
    which = "legacy word-window" if use_legacy_splitter else "token RecursiveCharacterTextSplitter"
    print(f"[ingest_local] splitter = {which} | batch_size = {batch_size}")

    _reset_store()
    vs = get_vectorstore()

    total = 0
    for course_dir in doc_processor.iter_course_dirs():
        chunks = doc_processor.build_chunks_for_course(course_dir, chunker=chunker)
        # Stable, human-readable ids so eval gold_chunk_ids can reference them.
        ids = [f"{c.course}:{i}" for i, c in enumerate(chunks)]
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            _add_batch_with_retry(
                vs,
                texts=[c.text for c in batch],
                ids=batch_ids,
                metadatas=[
                    {
                        "course": c.course,
                        "section": c.section,
                        "source": c.filename,
                        "chunk_id": cid,
                    }
                    for c, cid in zip(batch, batch_ids)
                ],
            )
            if pause:
                time.sleep(pause)
        print(f"  {course_dir.name}: embedded {len(chunks)} chunks")
        total += len(chunks)

    print(f"[ingest_local] total embedded: {total}")
    return total


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
    print(f"[verify] vector_dim   = {dim}  (expected {config.LOCAL_EMBEDDING_DIM})")


if __name__ == "__main__":
    if config.BACKEND != "local":
        print("WARNING: BACKEND != 'local'. Run with `BACKEND=local` to use Ollama/Chroma.")
    args = sys.argv[1:]
    if "--verify" in args:
        verify()
    else:
        ingest_local(use_legacy_splitter="--legacy-splitter" in args)
        verify()
