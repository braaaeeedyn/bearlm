"""HTTP-level API tests (TestClient). Retrieval, rerank, the LLM and Chroma are mocked."""

import threading
from pathlib import Path

import httpx
import ollama
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from bearlm import api, config, local_store, projects, rag, search

ROOT = Path(__file__).resolve().parent.parent
DOC = Document(page_content="Value iteration repeats Bellman updates.",
               metadata={"course": "cs188", "section": "p.80", "source": "188.pdf", "chunk_id": "cs188:1"})


@pytest.fixture
def llm(monkeypatch, tmp_path):
    """Script the LLM: each call pops the next item (an exception is raised, a str answered);
    the last item repeats. Returns the state dict with the call count."""
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(config, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "HYBRID_SEARCH_ENABLED", False)
    monkeypatch.setattr(rag, "_retrieve_local", lambda q, k=None, timings=None: [DOC])
    monkeypatch.setattr(rag, "_top_docs", lambda *a, **k: [DOC])
    monkeypatch.setattr(api, "_log_query", lambda *a, **k: None)
    monkeypatch.setattr(local_store, "collection_count", lambda name: 0)
    monkeypatch.setattr(local_store, "drop_collection", lambda name: None)
    state = {"calls": 0, "script": ["ok (cs188 p.80)"]}

    def call(_inp):
        state["calls"] += 1
        item = state["script"][min(state["calls"], len(state["script"])) - 1]
        if isinstance(item, BaseException):
            raise item
        return AIMessage(content=item)

    monkeypatch.setattr(rag, "_get_llm", lambda: RunnableLambda(call))
    return state


def _ask(client, url, q="What is value iteration?"):
    return client.post(url, json={"question": q, "rewrite": False})


def test_ollama_5xx_retried_once_then_503(llm):
    client = TestClient(api.app)
    for url in ("/api/ask", "/api/uploads/x/ask"):
        llm["calls"], llm["script"] = 0, [ollama.ResponseError("CUDA error: out of memory", 500)]
        r = _ask(client, url)
        assert r.status_code == 503, url
        assert "Ollama" in r.json()["detail"]
        assert "Traceback" not in r.json()["detail"]
        assert llm["calls"] == 2, url


def test_retry_recovers_after_one_failure(llm):
    client = TestClient(api.app)
    for url in ("/api/ask", "/api/uploads/x/ask"):
        llm["calls"] = 0
        llm["script"] = [ollama.ResponseError("stream broke", -1), "Value iteration (cs188 p.80)."]
        r = _ask(client, url)
        assert r.status_code == 200, url
        assert r.json()["answer"] == "Value iteration (cs188 p.80)."
        assert llm["calls"] == 2, url


def test_no_retry_on_4xx_connection_or_timeout(llm):
    client = TestClient(api.app)
    for exc in (ollama.ResponseError("model not found", 404), ConnectionError(), httpx.ReadTimeout("x")):
        llm["calls"], llm["script"] = 0, [exc]
        r = _ask(client, "/api/ask")
        assert r.status_code == 503, exc
        assert "Ollama" in r.json()["detail"]
        assert llm["calls"] == 1, exc


def test_retrieval_failure_returns_503(llm, monkeypatch):
    def down(*a, **k):
        raise ConnectionError("Failed to connect to Ollama")

    monkeypatch.setattr(rag, "_retrieve_local", down)
    r = _ask(TestClient(api.app), "/api/ask")
    assert r.status_code == 503
    assert "Ollama" in r.json()["detail"]
    assert llm["calls"] == 0


def test_failed_chat_turn_not_persisted(llm):
    llm["script"] = [ollama.ResponseError("CUDA error: out of memory", 500)]
    client = TestClient(api.app)
    slug = client.post("/api/projects", json={"name": "Fail"}).json()["slug"]
    chat = client.post(f"/api/projects/{slug}/chats", json={}).json()["id"]
    r = _ask(client, f"/api/projects/{slug}/chats/{chat}/ask")
    assert r.status_code == 503
    assert "Ollama" in r.json()["detail"]
    assert client.get(f"/api/projects/{slug}/chats/{chat}").json()["messages"] == []


def test_ask_page_reads_error_detail():
    js = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    assert ".detail" in js and "resp.json()" in js
    assert "Couldn't reach BearLM. Is the server running?" in js


def test_lifespan_prewarms_bm25(monkeypatch):
    built = threading.Event()
    monkeypatch.setattr(config, "HYBRID_SEARCH_ENABLED", True)
    monkeypatch.setattr(search, "_get_bm25", lambda name: built.set())
    with TestClient(api.app):
        assert built.wait(2)


def test_endpoint_smoke(llm):
    client = TestClient(api.app)
    for page in ("/", "/projects", "/analytics"):
        assert client.get(page).status_code == 200, page
    r = client.post("/api/ask", json={"question": "   "})
    assert r.status_code == 200 and r.json()["answer"] == "Please enter a question."

    r = client.post("/api/projects", json={"name": "Smoke"})
    assert r.status_code == 200
    slug = r.json()["slug"]
    assert client.post("/api/projects", json={"name": "Smoke"}).status_code == 409
    assert slug in [p["slug"] for p in client.get("/api/projects").json()["projects"]]
    assert client.get(f"/api/projects/{slug}").status_code == 200
    assert client.get("/api/projects/nope").status_code == 404

    chat = client.post(f"/api/projects/{slug}/chats", json={}).json()["id"]
    assert client.get(f"/api/projects/{slug}/chats/{chat}").status_code == 200
    assert client.post(f"/api/projects/{slug}/reset").status_code == 200
    assert client.delete(f"/api/projects/{slug}/chats/{chat}").status_code == 200
    assert client.delete(f"/api/projects/{slug}").status_code == 200
    assert client.get(f"/api/projects/{slug}").status_code == 404

    assert client.post("/api/uploads", files={"file": ("a.txt", b"hi", "text/plain")}).status_code == 400
    assert client.post("/api/uploads", files={"file": ("a.pdf", b"", "application/pdf")}).status_code == 400
    assert client.get("/source/cs61a/..%2F..%2Fpyproject.toml").status_code == 404
