"""Summary report: recall@5, MRR, median/p95 latency, cost per query.

Reads the artifacts produced by the other two scripts:
  - eval/out/retrieval.json   (from retrieval_eval.py)  -> recall@5, MRR
  - eval/out/answers.csv      (from answer_eval.py)      -> latency per backend

Prints one table. Missing inputs show as "-" placeholders rather than fabricated
numbers. Local cost is $0.00 (self-hosted). The OpenAI cost is an ESTIMATE from
the assumptions below — edit them to match your real token usage; it is a
formula, not a measured value.

Run:
    uv run python eval/report.py
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"

# --- OpenAI cost estimate assumptions (EDIT THESE) ---------------------------
# Prices in USD per 1M tokens; token counts are per-query assumptions.
PRICES = {
    "text-embedding-3-small": 0.02,   # embeddings
    "gpt-4o-mini-input": 0.15,
    "gpt-4o-mini-output": 0.60,
}
ASSUMED = {
    "embed_tokens": 20,        # question length embedded per query
    "prompt_tokens": 2500,     # ~5 chunks of context + system + question
    "output_tokens": 300,      # generated answer
}


def openai_cost_per_query() -> float:
    return (
        ASSUMED["embed_tokens"] / 1e6 * PRICES["text-embedding-3-small"]
        + ASSUMED["prompt_tokens"] / 1e6 * PRICES["gpt-4o-mini-input"]
        + ASSUMED["output_tokens"] / 1e6 * PRICES["gpt-4o-mini-output"]
    )


def load_retrieval() -> dict | None:
    p = OUT / "retrieval.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_latencies() -> dict[str, list[float]]:
    p = OUT / "answers.csv"
    lat: dict[str, list[float]] = {}
    if not p.exists():
        return lat
    with p.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                lat.setdefault(row["backend"], []).append(float(row["latency_s"]))
            except (KeyError, ValueError):
                continue
    return lat


def _p95(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    idx = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
    return s[idx]


def fmt(x, spec="{:.3f}") -> str:
    return spec.format(x) if isinstance(x, (int, float)) and x == x else "-"


RETR_CONFIGS = ("dense", "hybrid", "hybrid+rerank")


def main() -> None:
    retr = load_retrieval()
    lat = load_latencies()

    print("=" * 76)
    print("BearLM eval summary")
    print("=" * 76)

    # --- Retrieval ablation (recall@1/3/5 + MRR) ---
    print("Retrieval  (local: nomic-embed-text + Chroma)")
    if retr and all(c in retr for c in RETR_CONFIGS):
        n = retr.get("n", "?")
        header = f"  {'metric':12}" + "".join(f"{c:>16}" for c in RETR_CONFIGS)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for k in (1, 3, 5):
            m = f"recall@{k}"
            print(f"  {m:12}" + "".join(f"{fmt(retr[c].get(m)):>16}" for c in RETR_CONFIGS))
        print(f"  {'mrr':12}" + "".join(f"{fmt(retr[c].get('mrr')):>16}" for c in RETR_CONFIGS))
        print(f"  (n={n} questions, pool={retr.get('pool')})")
    else:
        print("  no retrieval.json — run: BACKEND=local uv run python eval/retrieval_eval.py")

    # --- Latency + cost (from answer_eval CSV) ---
    print("\nAnswering  (median / p95 latency, cost per query)")
    hdr = f"  {'backend':20}{'med(s)':>9}{'p95(s)':>9}{'$/query':>12}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    oai = openai_cost_per_query()
    for backend, cost in (("local", "$0.0000"), ("openai", "$" + format(oai, ".4f"))):
        ls = lat.get(backend, [])
        med = statistics.median(ls) if ls else None
        p95 = _p95(ls) if ls else None
        note = " (est.)" if backend == "openai" else ""
        print(f"  {backend:20}{fmt(med,'{:.2f}'):>9}{fmt(p95,'{:.2f}'):>9}{cost:>12}{note}")
    if not lat:
        print("  no answers.csv — run: uv run python eval/answer_eval.py")
    print("-" * 76)
    print("openai $/query is an ESTIMATE (edit PRICES/ASSUMED in report.py).")


if __name__ == "__main__":
    main()
