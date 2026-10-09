from langchain_core.documents import Document

import _common
from answer_eval import parse_judge


def _d(course, section, cid):
    return {"course": course, "section": section, "chunk_id": cid}


def test_recall_at_k():
    q = {"gold_course": "cs61a", "gold_section": "p.1", "gold_chunk_ids": ["cs61a:3"]}
    docs = [_d("cs70", "p.9", "cs70:1"), _d("cs70", "p.9", "cs70:2"), _d("cs61a", "p.1", "cs61a:3")]
    assert _common.recall_at_k(docs, q, 1) == 0.0
    assert _common.recall_at_k(docs, q, 3) == 1.0

    q2 = {"gold_chunk_ids": ["cs61a:3", "cs61a:99"]}
    assert _common.recall_at_k(docs, q2, 3) == 0.5

    q3 = {"gold_course": "cs61a", "gold_section": "p.1"}
    docs3 = [Document(page_content="x", metadata={"course": "cs70", "section": "p.9"}),
             Document(page_content="y", metadata={"course": "cs61a", "section": "p.1"})]
    assert _common.recall_at_k(docs3, q3, 1) == 0.0
    assert _common.recall_at_k(docs3, q3, 2) == 1.0


def test_precision_at_k():
    q = {"gold_course": "cs61a", "gold_section": "p.1", "gold_chunk_ids": ["cs61a:3"]}
    docs = [_d("cs61a", "p.1", "cs61a:4"), _d("cs70", "p.9", "cs70:1"),
            _d("cs61a", "p.1", "cs61a:5"), _d("cs70", "p.2", "cs70:2")]
    assert _common.precision_at_k(docs, q, 4) == 0.5
    assert _common.precision_at_k([_d("cs61a", "p.7", "cs61a:3")], q, 1) == 1.0


def test_cited_refs_and_citation_correctness():
    text = "X (cs61a p.111). Y (cs70 p.164; cs61a p. 404). (see below)"
    docs = [Document(page_content="a", metadata={"course": "cs61a", "section": "p.111"}),
            Document(page_content="b", metadata={"course": "cs61a", "section": "p.404"})]
    assert _common.cited_refs(text) == [("cs61a", "p.111"), ("cs70", "p.164"), ("cs61a", "p.404")]
    assert abs(_common.citation_correctness(text, docs) - 2 / 3) < 1e-9
    assert _common.citation_correctness("An answer with no citations.", docs) == 0.0
    assert _common.citation_correctness("I don't know based on the course materials.", docs) is None


def test_citation_correctness_counts_genml():
    docs = [Document(page_content="a", metadata={"course": "genML", "section": "05.01-A"})]
    assert _common.citation_correctness("X (genML 05.01-A).", docs) == 1.0


def test_parse_judge():
    none = {"faithfulness": None, "completeness": None}
    assert parse_judge('Sure! {"faithfulness": 0.8, "completeness": 0.5} hope that helps') == {
        "faithfulness": 0.8, "completeness": 0.5}
    assert parse_judge('{"faithfulness": 1.7, "completeness": -2}') == {
        "faithfulness": 1.0, "completeness": 0.0}
    assert parse_judge("no json here") == none
    assert parse_judge("{not json}") == none
    assert parse_judge('{"faithfulness": "high"}') == none
    assert parse_judge(None) == none


def test_gate_stats():
    got = _common.gate_stats([0.9, 0.02, 0.01, 0.5], [0.0, 0.02, 0.3], 0.02)
    assert got == {"min_score": 0.02, "n_in_scope": 4, "in_scope_below_gate": 1,
                   "n_off_topic": 3, "off_topic_at_or_above_gate": 2}  # 0.02 == gate counts as >=
    assert _common.gate_stats([], [], 0.5)["off_topic_at_or_above_gate"] == 0
