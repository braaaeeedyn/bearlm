"""Retrieval quality: recall@1/3/5, precision@1/3/5 and MRR across an ablation of techniques.

Runs against the LOCAL backend (Ollama embeddings + Chroma). For each question it
scores three retrieval configurations on the same candidate pool:
  - "dense"          : vector similarity only
  - "hybrid"         : BM25 + dense fused with reciprocal rank fusion
  - "hybrid+rerank"  : the hybrid pool reordered by the bge cross-encoder
A gold hit is decided by eval/_common.is_gold_hit — id-level when gold_chunk_ids
is set (as in the generated eval set), else by course+section.
recall@k is id-level (fraction of gold chunk ids in the top k, _common.recall_at_k).
precision@k is page-level: a top-k chunk counts as relevant if it is a gold hit or
shares the gold (course, section) (_common.precision_at_k).

Run:
    BACKEND=local uv run python eval/retrieval_eval.py --questions eval/questions_generated.jsonl

Writes eval/out/retrieval.json (read by report.py). Invents no numbers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

from bearlm import config, local_store  # noqa: E402

KS = (1, 3, 5)
CONFIGS = ("dense", "hybrid", "hybrid+rerank")
# Known-hard in-scope questions: (question, regex a correct top chunk should match).
PROBES = [("What is a B+ tree?", r"B\+\s*-?\s*tree")]


def _first_hit_rank(ordered_docs, q) -> int | None:
    for rank, d in enumerate(ordered_docs, 1):
        if _common.is_gold_hit(d, q):
            return rank
    return None


def _summarize(runs: list[tuple[list, dict]]) -> dict:
    """runs: (ordered docs, question) per question."""
    n = len(runs)
    out = {f"recall@{k}": sum(_common.recall_at_k(d, q, k) for d, q in runs) / n for k in KS}
    out.update({f"precision@{k}": sum(_common.precision_at_k(d, q, k) for d, q in runs) / n
                for k in KS})
    ranks = [_first_hit_rank(d, q) for d, q in runs]
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

    def top(scored):
        return scored[0][1] if scored else float("-inf")

    runs = {c: [] for c in CONFIGS}
    in_scope_tops = []
    for q in questions:
        query = q["question"]
        dense = vs.similarity_search(query, k=args.pool)
        hybrid = search.hybrid_candidates(query, config.CHROMA_COLLECTION, args.pool)
        scored = rerank.rerank_scored(query, hybrid)  # same order as rerank.rerank, plus scores
        in_scope_tops.append(top(scored))
        runs["dense"].append((dense, q))
        runs["hybrid"].append((hybrid, q))
        runs["hybrid+rerank"].append(([d for d, _ in scored], q))

    def rerank_hybrid(query):
        return rerank.rerank_scored(query, search.hybrid_candidates(query, config.CHROMA_COLLECTION, args.pool))

    off_tops = [top(rerank_hybrid(query)) for query in _common.OFF_TOPIC]
    probes = []
    for query, pattern in PROBES:
        scored = rerank_hybrid(query)
        probes.append({"question": query, "top_score": top(scored),
                       "term_hit": any(re.search(pattern, _common.doc_text(d), re.I)
                                       for d, _ in scored[:config.RERANK_TOP_N])})

    metrics = {
        "backend": "local",
        "embed_model": config.OLLAMA_EMBED_MODEL,
        "questions": args.questions,
        "n": len(questions),
        "pool": args.pool,
        **{c: _summarize(runs[c]) for c in CONFIGS},
        "gate": {**_common.gate_stats(in_scope_tops, off_tops, config.RERANK_MIN_SCORE),
                 "off_topic_top_scores": off_tops, "probes": probes},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Questions: {len(questions)}   candidate pool: {args.pool}\n")
    header = f"{'metric':12}" + "".join(f"{c:>16}" for c in CONFIGS)
    print(header)
    print("-" * len(header))
    for m in [f"{name}@{k}" for name in ("recall", "precision") for k in KS]:
        print(f"{m:12}" + "".join(f"{metrics[c][m]:>16.3f}" for c in CONFIGS))
    print(f"{'mrr':12}" + "".join(f"{metrics[c]['mrr']:>16.3f}" for c in CONFIGS))
    g = metrics["gate"]
    print(f"\ngate (min_score {g['min_score']}): in-scope below {g['in_scope_below_gate']}/"
          f"{g['n_in_scope']}, off-topic at/above {g['off_topic_at_or_above_gate']}/{g['n_off_topic']}")
    for p in probes:
        print(f"probe {p['question']!r}: top score {p['top_score']:.3f}, term hit {p['term_hit']}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
