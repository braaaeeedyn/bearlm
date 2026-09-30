"""Hybrid retrieval: BM25 (keyword) + dense vectors, fused with RRF.

Dense embeddings catch paraphrase/semantic matches; BM25 catches exact terms,
identifiers and rare words the embedding smooths over. Reciprocal Rank Fusion
combines the two rankings into one candidate pool, which the reranker then
scores. Enabled via config.HYBRID_SEARCH_ENABLED.

The BM25 index for a collection is built lazily from its stored documents and
cached in memory, keyed by the collection's vector count so it rebuilds after a
project gains or loses a PDF.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document

from . import config

_bm25_cache: dict[str, tuple[int, object, list[Document]]] = {}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _get_bm25(collection_name: str):
    """Return (BM25Okapi | None, docs) for a collection, cached by vector count."""
    from . import local_store

    col = local_store.get_vectorstore(collection_name)._collection
    count = col.count()
    cached = _bm25_cache.get(collection_name)
    if cached and cached[0] == count:
        return cached[1], cached[2]
    if count == 0:
        _bm25_cache[collection_name] = (0, None, [])
        return None, []

    from rank_bm25 import BM25Okapi

    got = col.get(include=["documents", "metadatas"])
    texts = got.get("documents") or []
    metas = got.get("metadatas") or []
    docs = [Document(page_content=t, metadata=(m or {})) for t, m in zip(texts, metas)]
    bm25 = BM25Okapi([_tokenize(t) for t in texts])
    _bm25_cache[collection_name] = (count, bm25, docs)
    return bm25, docs


def _doc_key(d: Document):
    m = d.metadata or {}
    return m.get("chunk_id") or (m.get("source"), m.get("section"), (d.page_content or "")[:80])


def hybrid_candidates(query: str, collection_name: str, pool: int) -> list[Document]:
    """Fuse dense + BM25 rankings via RRF; return up to `pool` documents."""
    from . import local_store

    vs = local_store.get_vectorstore(collection_name)
    dense = vs.similarity_search(query, k=pool)

    sparse: list[Document] = []
    bm25, docs = _get_bm25(collection_name)
    if bm25 is not None and docs:
        scores = bm25.get_scores(_tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        sparse = [docs[i] for i in ranked_idx[:pool] if scores[i] > 0]

    fused: dict = {}
    seen: dict = {}
    for lst in (dense, sparse):
        for rank, d in enumerate(lst):
            k = _doc_key(d)
            fused[k] = fused.get(k, 0.0) + 1.0 / (config.RRF_K + rank + 1)
            seen.setdefault(k, d)

    order = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [seen[k] for k, _ in order[:pool]]


def candidates(query: str, collection_name: str, pool: int) -> list[Document]:
    """Candidate pool for retrieval: hybrid when enabled, else dense-only."""
    if config.HYBRID_SEARCH_ENABLED:
        return hybrid_candidates(query, collection_name, pool)
    from . import local_store

    return local_store.get_vectorstore(collection_name).similarity_search(query, k=pool)


def invalidate(collection_name: str | None = None) -> None:
    """Drop cached BM25 index (all, or one collection)."""
    if collection_name is None:
        _bm25_cache.clear()
    else:
        _bm25_cache.pop(collection_name, None)
