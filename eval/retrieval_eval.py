"""Retrieval quality: recall@1/3/5 and MRR across an ablation of techniques.

Runs against the LOCAL backend (Ollama embeddings + Chroma). For each question it
scores three retrieval configurations on the same candidate pool:
  - "dense"          : vector similarity only
  - "hybrid"         : BM25 + dense fused with reciprocal rank fusion
  - "hybrid+rerank"  : the hybrid pool reordered by the bge cross-encoder
A gold hit is decided by eval/_common.is_gold_hit — id-level when gold_chunk_ids
is set (as in the generated eval set), else by course+section.

Run:
    BACKEND=local uv run python eval/retrieval_eval.py --questions eval/questions_generated.jsonl

Writes eval/out/retrieval.json (read by report.py). Invents no numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

from bearlm import config, local_store  # noqa: E402

KS = (1, 3, 5)
CONFIGS = ("dense", "hybrid", "hybrid+rerank")


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
    default_q = _common.EVAL_DIR / "questions_generated.jsonl"
    ap.add_argument("--questions", default=str(default_q if default_q.exists()
                                               else _common.EVAL_DIR / "questions.jsonl"))
    ap.add_argument("--out", default=str(_common.EVAL_DIR / "out" / "retrieval.json"))
    ap.add_argument("--pool", type=int, default=max(config.RERANK_CANDIDATES, 5))
    args = ap.parse_args()

    if config.BACKEND != "local":
        sys.exit("Run with BACKEND=local (retrieval eval targets Ollama+Chroma).")
    vs = local_store.get_vectorstore()
    if vs._collection.count() == 0:
        sys.exit("Chroma is empty. Run `BACKEND=local uv run python -m bearlm.local_store` first.")

    from bearlm import rerank, search

    questions = _common.load_questions(args.questions)
    if not questions:
        sys.exit(f"No questions in {args.questions}.")

    ranks = {c: [] for c in CONFIGS}
    for q in questions:
        query = q["question"]
        dense = vs.similarity_search(query, k=args.pool)
        hybrid = search.hybrid_candidates(query, config.CHROMA_COLLECTION, args.pool)
        reranked = rerank.rerank(query, hybrid, top_n=len(hybrid))
        ranks["dense"].append(_first_hit_rank(dense, q))
        ranks["hybrid"].append(_first_hit_rank(hybrid, q))
        ranks["hybrid+rerank"].append(_first_hit_rank(reranked, q))

    metrics = {
        "backend": "local",
        "embed_model": config.OLLAMA_EMBED_MODEL,
        "questions": args.questions,
        "n": len(questions),
        "pool": args.pool,
        **{c: _summarize(ranks[c]) for c in CONFIGS},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Questions: {len(questions)}   candidate pool: {args.pool}\n")
    header = f"{'metric':12}" + "".join(f"{c:>16}" for c in CONFIGS)
    print(header)
    print("-" * len(header))
    for k in KS:
        m = f"recall@{k}"
        print(f"{m:12}" + "".join(f"{metrics[c][m]:>16.3f}" for c in CONFIGS))
    print(f"{'mrr':12}" + "".join(f"{metrics[c]['mrr']:>16.3f}" for c in CONFIGS))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
