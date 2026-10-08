import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import answer_eval

ROOT = Path(__file__).resolve().parent.parent


def test_report_prints_new_metrics(tmp_path):
    metrics = {f"{m}@{k}": 0.25 for m in ("recall", "precision") for k in (1, 3, 5)}
    retr = {"n": 2, "pool": 15, **{c: {**metrics, "mrr": 0.6, "n": 2}
                                   for c in ("dense", "hybrid", "hybrid+rerank")}}
    (tmp_path / "retrieval.json").write_text(json.dumps(retr), encoding="utf-8")
    with (tmp_path / "answers.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=answer_eval.FIELDS)
        w.writeheader()
        w.writerow({"backend": "local", "question": "q1", "answer": "a (cs61a p.1)",
                    "latency_s": "4.2", "refused": "0", "citation_correctness": "1.000",
                    "faithfulness": "0.900", "completeness": "0.800",
                    "prompt_tokens": "1200", "output_tokens": "150"})
        w.writerow({"backend": "local", "question": "q2",
                    "answer": "I don't know based on the course materials.",
                    "latency_s": "2.0", "refused": "1", "citation_correctness": "",
                    "faithfulness": "", "completeness": "",
                    "prompt_tokens": "900", "output_tokens": "10"})

    res = subprocess.run([sys.executable, "eval/report.py", "--out-dir", str(tmp_path)],
                         cwd=ROOT, env={**os.environ, "BACKEND": "local"},
                         capture_output=True, text=True, encoding="utf-8")
    assert res.returncode == 0, res.stderr
    for label in ("recall@1", "precision@1", "refusal", "faithfulness", "completeness",
                  "citation", "p95", "$/query"):
        assert label in res.stdout, label
    assert "0.500" in res.stdout  # refusal rate 1/2
