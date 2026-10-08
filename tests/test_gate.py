import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from bearlm import config, rag, rerank, search


def _patch_scores(monkeypatch, scores):
    docs = [Document(page_content=f"chunk {i}", metadata={"course": "cs61a", "section": f"p.{i}"})
            for i in range(len(scores))]
    monkeypatch.setattr(search, "candidates", lambda *a, **k: docs)
    monkeypatch.setattr(rerank, "rerank_scored", lambda q, ds: list(zip(ds, scores)))


# Top bge scores measured live for on-topic / off-topic probe questions.
@pytest.mark.parametrize("scores", [[0.4975, 0.0015], [0.797, 0.05], [0.99, 0.98]])
def test_gate_accepts_on_topic(monkeypatch, scores):
    _patch_scores(monkeypatch, scores)
    assert len(rag._top_docs("q", "c", gated=True)) >= 1


@pytest.mark.parametrize("scores", [[0.0194, 0.0158, 0.009], [0.001, 0.0]])
def test_gate_rejects_off_topic(monkeypatch, scores):
    _patch_scores(monkeypatch, scores)
    assert rag._top_docs("q", "c", gated=True) == []


def test_answer_grounded_refuses_when_gate_empty(monkeypatch):
    monkeypatch.setattr(rag, "_top_docs", lambda *a, **k: [])

    def no_llm():
        raise AssertionError("LLM must not be called")

    monkeypatch.setattr(rag, "_get_llm", no_llm)
    res = rag.answer_grounded("What is the capital of France?", rewrite=False)
    assert res.text == config.GROUNDED_REFUSAL
    assert res.source == "none"


def test_prompts_keep_refusal_contract(monkeypatch):
    assert "ONLY" in rag._STRICT_SYSTEM
    assert "I don't know based on the course materials." in rag._STRICT_SYSTEM
    assert "nothing relevant" in rag._STRICT_SYSTEM

    captured = []

    def fake_llm(msgs):
        captured.append(msgs)
        return AIMessage(content="ok")

    doc = Document(page_content="hash join text", metadata={"course": "data101", "section": "p.3"})
    monkeypatch.setattr(rag, "_top_docs", lambda *a, **k: [doc])
    monkeypatch.setattr(rag, "_get_llm", lambda: RunnableLambda(fake_llm))
    res = rag.answer_grounded("What is a hash join?", rewrite=False)
    assert res.text == "ok"
    assert res.docs == [doc]
    sent = "\n".join(str(m.content) for m in captured[0])
    assert config.GROUNDED_REFUSAL in sent
    assert "ONLY" in sent
    assert "nothing relevant" in sent
