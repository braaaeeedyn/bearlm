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


def test_report_json_metrics(tmp_path):
    retr = {"n": 2, "pool": 8, "hybrid+rerank": {"recall@5": 0.5, "mrr": 0.75},
            "gate": {"off_topic_at_or_above_gate": 1}}
    (tmp_path / "retrieval.json").write_text(json.dumps(retr), encoding="utf-8")
    base = {"backend": "local", "citation_correctness": "", "faithfulness": "", "completeness": "",
            "rewrite_s": "0.000"}
    rows = [
        {**base, "question": "q1", "answer": "a (cs61a p.1)", "latency_s": "4.0", "refused": "0",
         "citation_correctness": "1.000", "faithfulness": "0.900", "completeness": "0.800",
         "prompt_tokens": "1200", "output_tokens": "150",
         "search_s": "0.200", "rerank_s": "1.000", "generate_s": "2.000"},
        {**base, "question": "q2", "answer": "an uncited answer", "latency_s": "6.0", "refused": "0",
         "citation_correctness": "0.000", "faithfulness": "0.700", "completeness": "0.600",
         "prompt_tokens": "1000", "output_tokens": "100",
         "search_s": "0.400", "rerank_s": "2.000", "generate_s": "3.000"},
        {**base, "question": "q3", "answer": "I don't know based on the course materials.",
         "latency_s": "1.0", "refused": "1", "prompt_tokens": "800", "output_tokens": "10",
         "search_s": "0.300", "rerank_s": "1.500", "generate_s": "0.000"},
        {**base, "question": "q4", "answer": "I don't know based on the course materials.",
         "latency_s": "1.0", "refused": "1", "prompt_tokens": "800", "output_tokens": "10",
         "search_s": "0.100", "rerank_s": "0.500", "generate_s": "0.000"},
    ]
    with (tmp_path / "answers.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=answer_eval.FIELDS)
        w.writeheader()
        w.writerows(rows)

    res = subprocess.run([sys.executable, "eval/report.py", "--out-dir", str(tmp_path), "--json"],
                         cwd=ROOT, env={**os.environ, "BACKEND": "local"},
                         capture_output=True, text=True, encoding="utf-8")
    assert res.returncode == 0, res.stderr
    data = json.loads(res.stdout)
    assert data["retrieval"] == retr
    local = data["answers"]["local"]
    assert local["n"] == 4 and local["refusal_rate"] == 0.5
    assert local["uncited_rate"] == 0.5  # 1 of the 2 answered rows has no (course section) cite
    assert local["stages_s"]["search_s"]["median"] == 0.25
    assert local["stages_s"]["rerank_s"]["median"] == 1.25
    assert local["stages_s"]["generate_s"]["median"] == 1.0
    assert local["stages_s"]["rewrite_s"]["median"] == 0.0
    assert local["latency_s"]["median"] == 2.5
    assert local["cost_per_query_usd"] == 0.0
    assert local["openai_cost_per_query_usd_est"] > 0


def test_report_json_run_meta_and_tok_rate(tmp_path):
    base = {"backend": "local", "refused": "0", "latency_s": "3.0", "answer": "a"}
    rows = [{**base, "question": "q1", "output_tokens": "100", "generate_s": "2.000"},   # 50 tok/s
            {**base, "question": "q2", "output_tokens": "120", "generate_s": "3.000"},   # 40 tok/s
            {**base, "question": "q3", "output_tokens": "90", "generate_s": "1.000"},    # 90 tok/s
            {**base, "question": "q4", "output_tokens": "10", "generate_s": "0.000"}]    # skipped
    with (tmp_path / "answers.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=answer_eval.FIELDS)
        w.writeheader()
        w.writerows(rows)
    meta = {"chroma_dir": "C:/db", "ollama_ps": [{"name": "llama3.1:8b", "size": 5, "size_vram": 5}]}
    (tmp_path / "answers.meta.json").write_text(json.dumps(meta), encoding="utf-8")

    res = subprocess.run([sys.executable, "eval/report.py", "--out-dir", str(tmp_path), "--json"],
                         cwd=ROOT, env={**os.environ, "BACKEND": "local"},
                         capture_output=True, text=True, encoding="utf-8")
    assert res.returncode == 0, res.stderr
    data = json.loads(res.stdout)
    assert data["run_meta"] == meta
    assert data["answers"]["local"]["generate_tok_per_s"]["median"] == 50.0
