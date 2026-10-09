"""Summary report: retrieval (recall@k, precision@k, MRR), answer quality, latency, cost.

Reads the artifacts produced by the other two scripts (from --out-dir, default eval/out):
  - retrieval.json   (from retrieval_eval.py)  -> recall@1/3/5 (id-level),
                                                  precision@1/3/5 (page-level), MRR
  - answers.csv      (from answer_eval.py)     -> refusal rate, faithfulness,
                                                  completeness, citation correctness,
                                                  latency, stage timings and token counts

Missing inputs show as "-" placeholders rather than fabricated numbers.
faithfulness/completeness come from the local llama3.1:8b judge: noisy estimates,
not ground truth. Cost per query is $0.00 in API spend (self-hosted; hardware and
electricity not modelled).

Run:
    uv run python eval/report.py [--out-dir eval/out] [--json]
--json prints the same summary (plus stage timings, generate tokens/s, uncited rate and
the answers.meta.json sidecar as run_meta) as one JSON object.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import _common  # eval/ is on sys.path when run as a script

OUT = Path(__file__).resolve().parent / "out"

def load_retrieval(out_dir: Path = OUT) -> dict | None:
    p = out_dir / "retrieval.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_answers(out_dir: Path = OUT) -> list[dict]:
    """answers.csv rows ([] if missing)."""
    p = out_dir / "answers.csv"
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _floats(rows: list[dict], col: str) -> list[float]:
    out = []
    for row in rows:
        try:
            out.append(float(row[col]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


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
STAGE_COLS = ("rewrite_s", "search_s", "rerank_s", "generate_s")


def _med_p95(xs: list[float]) -> dict:
    return {"median": statistics.median(xs) if xs else None, "p95": _p95(xs) if xs else None}


def _tok_rates(rows: list[dict]) -> list[float]:
    """Per-row output_tokens / generate_s over rows with generate_s > 0."""
    out = []
    for row in rows:
        try:
            tok, gen = float(row["output_tokens"]), float(row["generate_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if gen > 0:
            out.append(tok / gen)
    return out


def _known_courses(rows: list[dict]) -> set[str]:
    """Courses present in these rows' context (citations column) or gold labels, so
    non-digit course names like genML count as citations in uncited_rate."""
    known = {e.split()[0] for r in rows for e in (r.get("citations") or "").split(";") if e.strip()}
    return known | {r["gold_course"] for r in rows if r.get("gold_course")}


def summary_json(out_dir: Path) -> dict:
    """The summary as one JSON-able dict; None for anything not measured."""
    rows = load_answers(out_dir)
    answers = None
    if rows:
        answered = [r for r in rows if r.get("refused") == "0"
                    and not (r.get("answer") or "").startswith("[ERROR]")]
        known = _known_courses(rows)
        answers = {
            "n": len(rows),
            "refusal_rate": _mean(_floats(rows, "refused")),
            "faithfulness": _mean(_floats(rows, "faithfulness")),
            "completeness": _mean(_floats(rows, "completeness")),
            "citation_correctness": _mean(_floats(rows, "citation_correctness")),
            "uncited_rate": (sum(1 for r in answered if not _common.cited_refs(r.get("answer"), known))
                             / len(answered)) if answered else None,
            "latency_s": _med_p95(_floats(rows, "latency_s")),
            "stages_s": {k: _med_p95(_floats(rows, k)) for k in STAGE_COLS},
            "generate_tok_per_s": _med_p95(_tok_rates(rows)),
            "prompt_tokens_mean": _mean(_floats(rows, "prompt_tokens")),
            "output_tokens_mean": _mean(_floats(rows, "output_tokens")),
            "cost_per_query_usd": 0.0,
        }
    meta = out_dir / "answers.meta.json"
    run_meta = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else None
    return {"retrieval": load_retrieval(out_dir), "answers": answers, "run_meta": run_meta}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--json", action="store_true", help="print the summary as one JSON object")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    if args.json:
        print(json.dumps(summary_json(out_dir)))
        return
    retr = load_retrieval(out_dir)
    a = summary_json(out_dir)["answers"]

    print("=" * 76)
    print("BearLM eval summary")
    print("=" * 76)

    # --- Retrieval ablation (recall@1/3/5 + precision@1/3/5 + MRR) ---
    print("Retrieval  (nomic-embed-text + Chroma)")
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
        print("  no retrieval.json — run: uv run python eval/retrieval_eval.py")

    if not a:
        print("\nno answers.csv — run: uv run python eval/answer_eval.py")
        return
    print("\nAnswer quality  (means skip blanks)")
    for label, key in (("n", "n"), ("refusal", "refusal_rate"), ("faithfulness*", "faithfulness"),
                       ("completeness*", "completeness"), ("citation", "citation_correctness"),
                       ("uncited", "uncited_rate")):
        print(f"  {label:16}{fmt(a[key]) if key != 'n' else a['n']:>10}")
    print("  * local llama3.1:8b judge estimates, noisy; not ground truth.")
    lat = a["latency_s"]
    print(f"\nAnswering  latency median {fmt(lat['median'], '{:.2f}')} s, "
          f"p95 {fmt(lat['p95'], '{:.2f}')} s, $/query $0.0000 (self-hosted)")
    print("-" * 76)


if __name__ == "__main__":
    main()
