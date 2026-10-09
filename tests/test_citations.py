import re

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import _common
from bearlm import config, rag

DOCS = [Document(page_content=f"text {i}", metadata={"course": c, "section": s, "chunk_id": f"{c}:{i}"})
        for i, (c, s) in enumerate([("cs188", "p.89"), ("cs188", "p.80"),
                                    ("genML", "05.01-What-Is-Machine-Learning")])]
CONTEXT_REFS = {_common._norm_ref(d.metadata["course"], d.metadata["section"]) for d in DOCS}
COURSES = {"cs188", "genML", "cs61b"}


@pytest.mark.parametrize("raw,want", [
    ("A [cs188 p.89].", "A (cs188 p.89)."),
    ("A [CS188 p.89; genML 05.01-What-Is-Machine-Learning].",
     "A (cs188 p.89; genML 05.01-What-Is-Machine-Learning)."),
    ("A (Source: cs188 p.89).", "A (cs188 p.89)."),
    ("A (cs188 p. 89).", "A (cs188 p.89)."),
    ("A [cs188 p.89, cs188 p.80] and (cs188 p.80).", "A (cs188 p.89; cs188 p.80) and (cs188 p.80)."),
])
def test_normalize_bracket_citations(raw, want):
    assert rag._normalize_citations(raw, DOCS) == want


def test_normalize_never_adds_refs_outside_context():
    answers = ["X [cs188 p.89].", "X [cs61b p.12].", "X [cs188 p.89; cs61b p.12].",
               "X (cs61b p.12) and [genML 05.01-What-Is-Machine-Learning].", "No cites here.",
               "X (sources: cs188 p.80, cs188 p.89)."]
    for text in answers:
        out = rag._normalize_citations(text, DOCS)
        added = set(_common.cited_refs(out, COURSES)) - set(_common.cited_refs(text, COURSES))
        assert added <= CONTEXT_REFS, (text, out)
    for text in ("X [cs61b p.12].", "X [cs188 p.89; cs61b p.12].", "X (cs188 p.89; cs61b p.12)."):
        assert rag._normalize_citations(text, DOCS) == text


@pytest.mark.parametrize("text", [
    "See [cs188 p.89](http://x) and [text](http://x).", "The interval [0, 1].", "Math $[a, b]$ here.",
    "Some words (e.g., the count).", "A pair (s,a).", "Code: arr[i] = 1.", "Q(s, a) and (x_1 x_2).",
])
def test_normalize_leaves_non_citations(text):
    assert rag._normalize_citations(text, DOCS) == text


def _grounded_system(monkeypatch):
    captured = []

    def fake_llm(msgs):
        captured.append(msgs)
        return AIMessage(content="ok")

    monkeypatch.setattr(rag, "_top_docs", lambda *a, **k: DOCS[1:2])
    monkeypatch.setattr(rag, "_get_llm", lambda: RunnableLambda(fake_llm))
    rag.answer_grounded("What is X?", rewrite=False)
    return "\n".join(str(m.content) for m in captured[0])


def test_context_and_prompts_use_paren_format(monkeypatch):
    ctx = rag._format_docs(DOCS[1:2])
    assert "(cs188 p.80)" in ctx and "[cs188 p.80]" not in ctx
    for prompt in (rag._STRICT_SYSTEM, _grounded_system(monkeypatch)):
        low = prompt.lower()
        assert "parenthes" in low and "(cs188 p.80)" in prompt
        assert re.search(r"never cite [^.]*not in the context", low)


def test_answer_paths_normalize(monkeypatch):
    reply = {"text": "X [cs188 p.80]."}
    monkeypatch.setattr(rag, "_get_llm", lambda: RunnableLambda(lambda _: AIMessage(content=reply["text"])))
    monkeypatch.setattr(rag, "_retrieve_local", lambda *a, **k: DOCS[1:2])
    monkeypatch.setattr(rag, "_top_docs", lambda *a, **k: DOCS[1:2])

    assert "(cs188 p.80)" in rag._answer_local("What is X?", rewrite=False).text
    assert "(cs188 p.80)" in rag.answer_grounded("What is X?", rewrite=False).text

    for refusal in ("I don't know based on the course materials.", config.GROUNDED_REFUSAL):
        reply["text"] = refusal
        assert rag._answer_local("What is X?", rewrite=False).text == refusal
        assert rag.answer_grounded("What is X?", rewrite=False).text == refusal
