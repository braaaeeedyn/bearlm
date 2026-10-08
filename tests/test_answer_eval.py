import csv
import json

import pytest
from langchain_core.documents import Document

import answer_eval
from bearlm import rag


def test_run_one_backend_writes_metric_columns(monkeypatch, tmp_path):
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
    answer_eval.run_one_backend(out, str(qpath))

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
        answer_eval.run_one_backend(old, str(qpath))
    assert old.read_text(encoding="utf-8") == before
