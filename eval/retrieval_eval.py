"""Retrieval quality: recall@1/3/5 and MRR, with and without the reranker.

Runs against the LOCAL backend (Ollama embeddings + Chroma). For each question
it retrieves a candidate pool from Chroma, then scores two orderings:
  - "vector"  : Chroma similarity order (no reranker)
  - "rerank"  : same candidates reordered by the CrossEncoder
A gold hit is decided by eval/_common.is_gold_hit (id-level if gold_chunk_ids is
set, else by course+section).

Run:
    BACKEND=local uv run python eval/retrieval_eval.py
    BACKEND=local uv run python eval/retrieval_eval.py --out eval/out/retrieval.json

Writes a JSON metrics file (default eval/out/retrieval.json) that report.py reads.
Does NOT invent numbers — if Chroma is empty it tells you to ingest first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow `import _common`
import _common  # noqa: E402

from bearlm import config, local_store  # noqa: E402

KS = (1, 3, 5)


def _first_hit_rank(ordered_docs, q) -> int | None:
    for rank, d in enumerate(ordered_docs, 1):
        if _common.is_gold_hit(d, q):
            return rank
    return None


def _summarize(ranks: list[int | None]) -> dict:
    n = len(ranks)
    out = {f"recall@{k}": sum(1 for r in ranks if r and r <= k) / n for k in KS}
    out["mrr"] = sum((1.0 / r) if r else 0.0 for r in ranks) / n
    out["n"] = n
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_common.EVAL_DIR / "out" / "retrieval.json"))
    ap.add_argument("--pool", type=int, default=max(config.RERANK_CANDIDATES, 5))
    args = ap.parse_args()

    if config.BACKEND != "local":
        sys.exit("Run with BACKEND=local (retrieval eval targets Ollama+Chroma).")

    vs = local_store.get_vectorstore()
    if vs._collection.count() == 0:
        sys.exit("Chroma is empty. Run `BACKEND=local uv run python -m bearlm.local_store` first.")

    from bearlm import rerank

    questions = _common.load_questions()
    retriever = local_store.get_retriever(args.pool)

    vector_ranks, rerank_ranks = [], []
    for q in questions:
        candidates = retriever.invoke(q["question"])
        vector_ranks.append(_first_hit_rank(candidates, q))
        reranked = rerank.rerank(q["question"], candidates, top_n=len(candidates))
        rerank_ranks.append(_first_hit_rank(reranked, q))

    metrics = {
        "backend": "local",
        "embed_model": config.OLLAMA_EMBED_MODEL,
        "pool": args.pool,
        "vector": _summarize(vector_ranks),
        "rerank": _summarize(rerank_ranks),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Questions: {len(questions)}   candidate pool: {args.pool}")
    print(f"{'metric':10} {'vector':>10} {'rerank':>10}")
    for k in KS:
        m = f"recall@{k}"
        print(f"{m:10} {metrics['vector'][m]:10.3f} {metrics['rerank'][m]:10.3f}")
    print(f"{'mrr':10} {metrics['vector']['mrr']:10.3f} {metrics['rerank']['mrr']:10.3f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
