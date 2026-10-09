"""Cross-encoder reranking (Phase 4) — BACKEND == "local".

Retrieve a wide candidate set (config.RERANK_CANDIDATES, default 8), score each
(query, passage) pair with a CrossEncoder (BAAI/bge-reranker-base), and keep the
top N (config.RERANK_TOP_N, default 4). Toggled by config.RERANK_ENABLED so the
pipeline can be measured with and without it.

The model is a sentence-transformers CrossEncoder, downloaded from Hugging Face
on first use (independent of Ollama).
"""

from __future__ import annotations

import functools
from typing import Any

from . import config


@functools.lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder  # lazy: heavy import (torch)

    return CrossEncoder(config.RERANK_MODEL)


def _text_of(doc: Any) -> str:
    """Accept both LangChain Documents and plain dicts/strings."""
    if hasattr(doc, "page_content"):
        return doc.page_content
    if isinstance(doc, dict):
        return doc.get("text", "")
    return str(doc)


def rerank_scored(query: str, docs: list[Any]) -> list[tuple[Any, float]]:
    """Return every doc paired with its cross-encoder relevance logit, best first."""
    if not docs:
        return []
    pairs = [(query, _text_of(d)) for d in docs]
    scores = _cross_encoder().predict(pairs)
    return sorted(
        ((d, float(s)) for d, s in zip(docs, scores)),
        key=lambda ds: ds[1],
        reverse=True,
    )


def rerank(query: str, docs: list[Any], top_n: int | None = None) -> list[Any]:
    """Return the top-N docs by cross-encoder relevance to the query.

    Order is preserved for ties in score order returned by the model. If the
    candidate list is already small, this simply reorders it.
    """
    top_n = top_n or config.RERANK_TOP_N
    return [d for d, _ in rerank_scored(query, docs)[:top_n]]
