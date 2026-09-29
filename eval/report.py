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


def main() -> None:
    retr = load_retrieval()
    lat = load_latencies()

    rows = []
    # Local backend row (recall/mrr from retrieval.json; latency from CSV).
    def metrics_for(backend: str, retr_key: str | None):
        r5 = mrr = None
        if retr and retr_key and retr_key in retr:
            r5, mrr = retr[retr_key].get("recall@5"), retr[retr_key].get("mrr")
        ls = lat.get(backend, [])
        med = statistics.median(ls) if ls else None
        p95 = _p95(ls) if ls else None
        return r5, mrr, med, p95

    r5_v, mrr_v, med_l, p95_l = metrics_for("local", "vector")
    r5_r, mrr_r, _, _ = metrics_for("local", "rerank")
    _, _, med_o, p95_o = metrics_for("openai", None)

    oai = openai_cost_per_query()

    print("=" * 74)
    print("BearLM eval summary")
    print("=" * 74)
    header = f"{'config':22}{'recall@5':>10}{'MRR':>8}{'med(s)':>9}{'p95(s)':>9}{'$/query':>12}"
    print(header)
    print("-" * 74)
    print(f"{'local (vector)':22}{fmt(r5_v):>10}{fmt(mrr_v):>8}{fmt(med_l,'{:.2f}'):>9}{fmt(p95_l,'{:.2f}'):>9}{'$0.0000':>12}")
    print(f"{'local (+reranker)':22}{fmt(r5_r):>10}{fmt(mrr_r):>8}{fmt(med_l,'{:.2f}'):>9}{fmt(p95_l,'{:.2f}'):>9}{'$0.0000':>12}")
    print(f"{'openai (baseline)':22}{'-':>10}{'-':>8}{fmt(med_o,'{:.2f}'):>9}{fmt(p95_o,'{:.2f}'):>9}{'$'+format(oai,'.4f'):>12}")
    print("-" * 74)
    print("openai $/query is an ESTIMATE (edit PRICES/ASSUMED in report.py).")
    if not retr:
        print("note: eval/out/retrieval.json missing -> run retrieval_eval.py.")
    if not lat:
        print("note: eval/out/answers.csv missing -> run answer_eval.py.")


if __name__ == "__main__":
    main()
