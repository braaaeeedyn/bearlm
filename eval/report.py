"""Summary report: retrieval (recall@k, precision@k, MRR), answer quality, latency, cost.

Reads the artifacts produced by the other two scripts (from --out-dir, default eval/out):
  - retrieval.json   (from retrieval_eval.py)  -> recall@1/3/5 (id-level),
                                                  precision@1/3/5 (page-level), MRR
  - answers.csv      (from answer_eval.py)     -> refusal rate, faithfulness,
                                                  completeness, citation correctness,
                                                  latency and token counts per backend

Missing inputs show as "-" placeholders rather than fabricated numbers.
faithfulness/completeness come from the local llama3.1:8b judge: noisy estimates,
not ground truth. Local cost is $0.00 in API spend (self-hosted; hardware and
electricity not modelled). The OpenAI cost is an ESTIMATE: list prices below times
the mean measured (tiktoken) token counts from answers.csv, falling back to the
ASSUMED counts when no measurements exist.

Run:
    uv run python eval/report.py [--out-dir eval/out]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"

# --- OpenAI cost estimate assumptions (EDIT THESE) ---------------------------
# Prices in USD per 1M tokens; token counts are per-query fallbacks when
# answers.csv has no measured prompt/output tokens.
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


def openai_cost_per_query(prompt_tokens: float | None = None,
                          output_tokens: float | None = None) -> float:
    return (
        ASSUMED["embed_tokens"] / 1e6 * PRICES["text-embedding-3-small"]
        + (prompt_tokens or ASSUMED["prompt_tokens"]) / 1e6 * PRICES["gpt-4o-mini-input"]
        + (output_tokens or ASSUMED["output_tokens"]) / 1e6 * PRICES["gpt-4o-mini-output"]
    )


def load_retrieval(out_dir: Path = OUT) -> dict | None:
    p = out_dir / "retrieval.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_answers(out_dir: Path = OUT) -> dict[str, list[dict]]:
    """answers.csv rows grouped by backend."""
    p = out_dir / "answers.csv"
    rows: dict[str, list[dict]] = {}
    if not p.exists():
        return rows
    with p.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.setdefault(row.get("backend", "?"), []).append(row)
    return rows


def _floats(rows: list[dict], col: str) -> list[float]:
    out = []
    for row in rows:
        try:
            out.append(float(row[col]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def load_latencies(out_dir: Path = OUT) -> dict[str, list[float]]:
    return {b: _floats(rows, "latency_s") for b, rows in load_answers(out_dir).items()}


def _mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    out_dir = Path(ap.parse_args().out_dir)
    retr = load_retrieval(out_dir)
    answers = load_answers(out_dir)
    lat = {b: _floats(rows, "latency_s") for b, rows in answers.items()}

    print("=" * 76)
    print("BearLM eval summary")
    print("=" * 76)

    # --- Retrieval ablation (recall@1/3/5 + precision@1/3/5 + MRR) ---
    print("Retrieval  (local: nomic-embed-text + Chroma)")
    if retr and all(c in retr for c in RETR_CONFIGS):
        n = retr.get("n", "?")
        header = f"  {'metric':12}" + "".join(f"{c:>16}" for c in RETR_CONFIGS)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for m in [f"{name}@{k}" for name in ("recall", "precision") for k in (1, 3, 5)]:
            print(f"  {m:12}" + "".join(f"{fmt(retr[c].get(m)):>16}" for c in RETR_CONFIGS))
        print(f"  {'mrr':12}" + "".join(f"{fmt(retr[c].get('mrr')):>16}" for c in RETR_CONFIGS))
        print(f"  (n={n} questions, pool={retr.get('pool')}; recall id-level, precision page-level)")
    else:
        print("  no retrieval.json — run: BACKEND=local uv run python eval/retrieval_eval.py")

    # --- Answer quality (from answer_eval CSV) ---
    print("\nAnswer quality  (means skip blanks)")
    hdr = (f"  {'backend':10}{'n':>5}{'refusal':>9}{'faithfulness*':>15}"
           f"{'completeness*':>15}{'citation':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for backend in ("local", "openai"):
        rows = answers.get(backend, [])
        refused = _floats(rows, "refused")
        print(f"  {backend:10}{len(rows):>5}{fmt(_mean(refused)):>9}"
              f"{fmt(_mean(_floats(rows, 'faithfulness'))):>15}"
              f"{fmt(_mean(_floats(rows, 'completeness'))):>15}"
              f"{fmt(_mean(_floats(rows, 'citation_correctness'))):>10}")
    print("  * (local LLM judge) llama3.1:8b estimates, noisy; not ground truth.")

    # --- Latency + cost (from answer_eval CSV) ---
    print("\nAnswering  (median / p95 latency, cost per query)")
    hdr = f"  {'backend':20}{'med(s)':>9}{'p95(s)':>9}{'$/query':>12}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    measured = [r for rows in answers.values() for r in rows]
    oai = openai_cost_per_query(_mean(_floats(measured, "prompt_tokens")),
                                _mean(_floats(measured, "output_tokens")))
    for backend, cost in (("local", "$0.0000"), ("openai", "$" + format(oai, ".4f"))):
        ls = lat.get(backend, [])
        med = statistics.median(ls) if ls else None
        p95 = _p95(ls) if ls else None
        note = " (est.)" if backend == "openai" else ""
        print(f"  {backend:20}{fmt(med,'{:.2f}'):>9}{fmt(p95,'{:.2f}'):>9}{cost:>12}{note}")
    if not answers:
        print("  no answers.csv — run: uv run python eval/answer_eval.py")
    print("-" * 76)
    print("openai $/query is an ESTIMATE: list PRICES x mean measured tokens "
          "(ASSUMED in report.py when unmeasured).")


if __name__ == "__main__":
    main()
