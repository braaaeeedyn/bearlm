"""Cross-encoder reranking (bge-reranker-base).

Retrieve a wide candidate set (config.RERANK_CANDIDATES, default 8), score each
(query, passage) pair with a CrossEncoder (BAAI/bge-reranker-base), and keep the
top N (config.RERANK_TOP_N, default 4). Toggled by config.RERANK_ENABLED so the
pipeline can be measured with and without it.

The model is a sentence-transformers CrossEncoder, downloaded from Hugging Face
on first use (independent of Ollama). It runs on config.RERANK_DEVICE: "auto"
picks cuda when torch sees a GPU, else cpu. On cuda it loads in fp16 (~0.55 GB
VRAM next to the Ollama model) and falls back to cpu if loading fails (e.g. OOM).
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from . import config

log = logging.getLogger(__name__)


def _device() -> str:
    if config.RERANK_DEVICE != "auto":
        return config.RERANK_DEVICE
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@functools.lru_cache(maxsize=1)
def _cross_encoder():
    import torch  # lazy: heavy import
    from sentence_transformers import CrossEncoder

    dev = _device()
    if dev.startswith("cuda"):
        try:
            return CrossEncoder(config.RERANK_MODEL, device=dev,
                                model_kwargs={"torch_dtype": torch.float16})
        except RuntimeError as e:  # includes torch.OutOfMemoryError
            log.warning("reranker failed to load on %s (%s); using cpu", dev, e)
            dev = "cpu"
    return CrossEncoder(config.RERANK_MODEL, device=dev)


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
