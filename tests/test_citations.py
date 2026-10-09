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


_NON_CITATIONS = [
    "See [cs188 p.89](http://x) and [text](http://x).", "The interval [0, 1].", "Math $[a, b]$ here.",
    "Some words (e.g., the count).", "A pair (s,a).", "Code: arr[i] = 1.", "Q(s, a) and (x_1 x_2).",
    "(see cs61a p.3 for details)", "(cs61a lecture)", "(2003)", "O(n) time (n = 5)",
]


@pytest.mark.parametrize("raw,want", [
    ("X (cs188 p.90).", "X."),
    ("The count is 3 (cs188 p.80) and Y (data8 p.3).", "The count is 3 (cs188 p.80) and Y."),
    ("X (cs188 p.89; cs61b p.12).", "X (cs188 p.89)."),
    ("X [cs61b p.12].", "X."),
    ("X (cs188 p.80; genML 05.01-What-Is-Machine-Learning).", None),
] + [(t, None) for t in _NON_CITATIONS])
def test_drop_outside_citations(raw, want):
    out = rag._drop_outside_citations(raw, DOCS)
    assert out == (raw if want is None else want)
    raw_refs = set(_common.cited_refs(raw, COURSES))
    assert set(_common.cited_refs(out, COURSES)) <= CONTEXT_REFS | raw_refs
    if all(re.search(r"\d", sec) for _, sec in raw_refs - CONTEXT_REFS):
        assert set(_common.cited_refs(out, COURSES)) <= CONTEXT_REFS


def test_answer_paths_drop_outside_cites(monkeypatch):
    monkeypatch.setattr(rag, "_get_llm",
                        lambda: RunnableLambda(lambda _: AIMessage(content="X [cs188 p.80]. Y (data8 p.3).")))
    monkeypatch.setattr(rag, "_retrieve_local", lambda *a, **k: DOCS[1:2])
    monkeypatch.setattr(rag, "_top_docs", lambda *a, **k: DOCS[1:2])

    assert rag._answer_local("What is X?", rewrite=False).text == "X (cs188 p.80). Y."
    assert rag.answer_grounded("What is X?", rewrite=False).text == "X (cs188 p.80). Y."


PROBE_CTX = [Document(page_content="text", metadata={"course": c, "section": s, "chunk_id": f"{c}:1"})
             for c, s in [("cs188", "p.80"), ("data8", "p.3")]]


@pytest.mark.parametrize("s", [
    "The root is x = 5 (x2 = 5).", "Use the (L1 norm 3) penalty.", "Reject when (H0 p < 0.05).",
    "The F test (df1 = 3, df2 = 10) applies.", "Fit is good (R2 = 0.87).", "Weights shrink (L2 penalty 0.1).",
    "The terms (x1 + x2 = 1) sum to one.", "Gradient step (w1 = w0 - 0.1).", "Variance (sigma2 = 4) is known.",
    "Take (k1 = 2; k2 = 3) as constants.", "Recall (p1 = 0.3, p2 = 0.7).", "A 3x3 matrix (A3 is 3 by 3).",
    "Use log base 2 (log2 8 = 3).", "The norm (L1 = 3).", "Mod arithmetic (Z7 has 7 elements).",
    "Run (T1 > T2 by 3s).", "Hash (h1 = k mod 7).",
])
def test_drop_outside_keeps_math(s):
    assert rag._drop_outside_citations(s, PROBE_CTX) == s


@pytest.mark.parametrize("raw,want", [
    ("Keep this (cs188 p.80).", "Keep this (cs188 p.80)."),
    ("Drop this (cs188 p.999).", "Drop this."),
    ("Drop this (cs70 note 12).", "Drop this."),
    ("Mixed (cs188 p.80; cs70 note 2).", "Mixed (cs188 p.80)."),
])
def test_drop_outside_real_cites(raw, want):
    assert rag._drop_outside_citations(raw, PROBE_CTX) == want
