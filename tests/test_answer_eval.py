import csv
import json

import pytest
from langchain_core.documents import Document

import answer_eval
from bearlm import rag


def test_run_eval_writes_metric_columns(monkeypatch, tmp_path):
    docs = [Document(page_content="A hash join builds a hash table.",
                     metadata={"course": "data101", "section": "p.3", "chunk_id": "data101:1"})]
    monkeypatch.setattr(rag, "answer", lambda q, *a, **k: rag.Answer(
        "It builds a hash table (data101 p.3).",
        [{"course": "data101", "section": "p.3", "source": "x.pdf"}], docs=docs))
    monkeypatch.setattr(answer_eval, "judge_answer",
                        lambda *a, **k: {"faithfulness": 0.9, "completeness": 0.7})

    qpath = tmp_path / "q.jsonl"
    qpath.write_text("\n".join(json.dumps({"question": f"q{i}", "gold_course": "data101",
                                           "gold_section": "p.3"}) for i in range(2)),
                     encoding="utf-8")
    out = tmp_path / "answers.csv"
    answer_eval.run_eval(out, str(qpath))

    with out.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == answer_eval.FIELDS
        rows = list(reader)
    assert len(rows) == 2
    for row in rows:
        assert row["refused"] == "0"
        assert float(row["citation_correctness"]) == 1.0
        assert float(row["faithfulness"]) == 0.9
        assert int(row["prompt_tokens"]) > 0
        assert float(row["latency_s"]) >= 0

    old = tmp_path / "old.csv"
    old.write_text("backend,question,gold_course,gold_section,answer,citations,latency_s,grade,notes\n",
                   encoding="utf-8")
    before = old.read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        answer_eval.run_eval(old, str(qpath))
    assert old.read_text(encoding="utf-8") == before


def test_answer_eval_writes_stage_timings(monkeypatch, tmp_path):
    timings = {"rewrite_s": 0.0, "search_s": 0.25, "rerank_s": 1.5, "generate_s": 6.125, "total_s": 7.875}
    monkeypatch.setattr(rag, "answer", lambda q, *a, **k: rag.Answer("ok", [], docs=[], timings=timings))
    qpath = tmp_path / "q.jsonl"
    qpath.write_text(json.dumps({"question": "q0"}), encoding="utf-8")
    out = tmp_path / "answers.csv"
    answer_eval.run_eval(out, str(qpath), judge=False)

    with out.open(encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert answer_eval.FIELDS.index("generate_s") < answer_eval.FIELDS.index("grade")
    assert answer_eval.FIELDS.index("rewrite_s") > answer_eval.FIELDS.index("output_tokens")
    for k in ("rewrite_s", "search_s", "rerank_s", "generate_s"):
        assert float(row[k]) == timings[k], k


def test_answer_eval_writes_run_meta_and_context_ids(monkeypatch, tmp_path):
    docs = [Document(page_content=f"t{i}", metadata={"course": "cs188", "section": f"p.{i}",
                                                      "chunk_id": f"cs188:{i}"}) for i in range(2)]
    monkeypatch.setattr(rag, "answer", lambda q, *a, **k: rag.Answer("ok (cs188 p.0)", [], docs=docs))
    ps = [{"name": "llama3.1:8b", "size": 10, "size_vram": 10}]
    monkeypatch.setattr(answer_eval, "_ollama_ps", lambda: ps)
    monkeypatch.setattr(answer_eval, "_nvidia_smi", lambda: (100, 8000))
    qpath = tmp_path / "q.jsonl"
    qpath.write_text(json.dumps({"question": "q0"}), encoding="utf-8")
    out = tmp_path / "answers.csv"
    answer_eval.run_eval(out, str(qpath), judge=False)

    with out.open(encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["context_ids"] == "cs188:0;cs188:1"
    meta = json.loads((tmp_path / "answers.meta.json").read_text(encoding="utf-8"))
    for key in ("chroma_dir", "collection_count", "questions_sha256", "ollama_ps", "gpu"):
        assert key in meta, key
    assert meta["config"]["RERANK_CANDIDATES"] > 0
    assert meta["ollama_ps"] == ps and meta["n_rows"] == 1
    assert meta["gpu"]["nvidia_smi_used_mb"] == 100

    def boom():
        raise OSError("down")

    monkeypatch.setattr(answer_eval, "_ollama_ps", boom)
    out.unlink()
    answer_eval.run_eval(out, str(qpath), judge=False)
    meta = json.loads((tmp_path / "answers.meta.json").read_text(encoding="utf-8"))
    assert meta["ollama_ps"] is None and "gpu" in meta


def test_answer_eval_flushes_rows_and_writes_meta_on_error(monkeypatch, tmp_path):
    monkeypatch.setattr(answer_eval, "judge_answer",
                        lambda *a, **k: {"faithfulness": 0.9, "completeness": 0.7})
    qpath = tmp_path / "q.jsonl"
    qpath.write_text("\n".join(json.dumps({"question": f"q{i}"}) for i in range(3)), encoding="utf-8")

    def answer_raising(exc):
        def fake(q, *a, **k):
            if q == "q1":
                raise exc
            return rag.Answer("ok", [], docs=[])
        return fake

    out = tmp_path / "answers.csv"
    monkeypatch.setattr(rag, "answer", answer_raising(TimeoutError("timed out")))
    answer_eval.run_eval(out, str(qpath))
    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert rows[1]["answer"].startswith("[ERROR] TimeoutError")
    assert rows[1]["faithfulness"] == "" and rows[1]["citation_correctness"] == ""
    assert (tmp_path / "answers.meta.json").exists()

    out2 = tmp_path / "killed.csv"
    monkeypatch.setattr(rag, "answer", answer_raising(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        answer_eval.run_eval(out2, str(qpath))
    with out2.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["question"] for r in rows] == ["q0"]
    meta = json.loads((tmp_path / "killed.meta.json").read_text(encoding="utf-8"))
    assert meta["n_rows"] == 1
