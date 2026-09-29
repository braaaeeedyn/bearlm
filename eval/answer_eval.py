"""End-to-end answer eval: run a backend over the questions, log for grading.

For each question it records: backend, question, answer, citations, and latency.
Rows are appended to a CSV with blank grading columns you fill in by hand
(grade = correct | partial | wrong | hallucinated).

Run one backend (reads BACKEND from the env):
    BACKEND=local  uv run python eval/answer_eval.py
    BACKEND=openai uv run python eval/answer_eval.py

Or run BOTH on the same questions in one go (spawns a subprocess per backend so
each imports cleanly with its own BACKEND):
    uv run python eval/answer_eval.py --both

Output CSV: eval/out/answers.csv  (override with --out). Same file is appended,
so delete it for a clean run. report.py reads latency from this file.

Does NOT grade for you and does NOT invent numbers.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

FIELDS = [
    "backend", "question", "gold_course", "gold_section",
    "answer", "citations", "latency_s",
    "grade",  # <- fill by hand: correct | partial | wrong | hallucinated
    "notes",
]


def run_one_backend(out_path: Path) -> None:
    from bearlm import config, rag  # imported here so BACKEND is already set

    questions = _common.load_questions()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for q in questions:
            t0 = time.perf_counter()
            try:
                res = rag.answer(q["question"])
                answer_text, cites = res.text, res.citations
            except Exception as e:  # log the failure rather than aborting the run
                answer_text, cites = f"[ERROR] {type(e).__name__}: {e}", []
            latency = time.perf_counter() - t0
            w.writerow({
                "backend": config.BACKEND,
                "question": q["question"],
                "gold_course": q.get("gold_course", ""),
                "gold_section": q.get("gold_section", ""),
                "answer": answer_text,
                "citations": "; ".join(f"{c['course']} {c['section']}" for c in cites),
                "latency_s": f"{latency:.3f}",
                "grade": "",
                "notes": "",
            })
            print(f"[{config.BACKEND}] {latency:6.2f}s  {q['question'][:60]}")
    print(f"Appended {config.BACKEND} rows to {out_path}")


def run_both(out_path: Path) -> None:
    for backend in ("local", "openai"):
        print(f"\n=== spawning backend={backend} ===")
        env = {**os.environ, "BACKEND": backend}
        subprocess.run(
            [sys.executable, __file__, "--out", str(out_path)],
            env=env, check=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_common.EVAL_DIR / "out" / "answers.csv"))
    ap.add_argument("--both", action="store_true", help="run local and openai backends")
    args = ap.parse_args()
    out_path = Path(args.out)
    if args.both:
        run_both(out_path)
    else:
        run_one_backend(out_path)


if __name__ == "__main__":
    main()
