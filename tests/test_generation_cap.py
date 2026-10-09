import re
import sys
import types
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from bearlm import config, rag

DOC = Document(page_content="text", metadata={"course": "cs188", "section": "p.80", "chunk_id": "cs188:1"})


def test_get_llm_caps_output_and_times_out(monkeypatch):
    seen = {}
    monkeypatch.setitem(sys.modules, "langchain_ollama",
                        types.SimpleNamespace(ChatOllama=lambda **kw: seen.update(kw)))
    rag._get_llm()
    assert seen["num_predict"] == config.OLLAMA_NUM_PREDICT
    assert seen["client_kwargs"]["timeout"] == config.OLLAMA_TIMEOUT_S
    src = Path(config.__file__).read_text(encoding="utf-8")
    assert re.search(r'OLLAMA_NUM_PREDICT",\s*"1024"', src)


def test_truncated_answers_show_notice(monkeypatch):
    reply = {"done_reason": "length"}
    monkeypatch.setattr(rag, "_get_llm", lambda: RunnableLambda(lambda _: AIMessage(
        content="X [cs188 p.80].", response_metadata=dict(reply))))
    monkeypatch.setattr(rag, "_retrieve_local", lambda *a, **k: [DOC])
    monkeypatch.setattr(rag, "_top_docs", lambda *a, **k: [DOC])

    for fn in (rag._answer_local, rag.answer_grounded):
        text = fn("What is X?", rewrite=False).text
        assert text.endswith(rag.TRUNCATED_NOTE) and "(cs188 p.80)" in text

    reply["done_reason"] = "stop"
    for fn in (rag._answer_local, rag.answer_grounded):
        assert rag.TRUNCATED_NOTE not in fn("What is X?", rewrite=False).text
