import itertools
import sys
import types

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from bearlm import config, local_store, projects, rag, rerank, search

STAGES = ("rewrite_s", "search_s", "rerank_s", "generate_s", "total_s")


def _doc(cid, course="p1"):
    return Document(page_content=f"text {cid}",
                    metadata={"course": course, "section": "p.1", "source": "a.pdf", "chunk_id": cid})


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path / "projects")
    ticks = itertools.count()
    monkeypatch.setattr(projects, "_now", lambda: f"2026-01-01T00:00:{next(ticks):06d}")
    state = {"cands": {}}
    monkeypatch.setattr(search, "candidates", lambda q, coll, *a, **k: list(state["cands"].get(coll, [])))
    monkeypatch.setattr(rerank, "rerank_scored", lambda q, ds: [(d, 0.5) for d in ds])
    monkeypatch.setattr(local_store, "collection_count", lambda name: 1)
    monkeypatch.setattr(local_store, "get_docs_by_ids", lambda coll, ids: [])
    monkeypatch.setattr(rag, "_get_llm", lambda: RunnableLambda(lambda msgs: AIMessage(content="answer")))
    return state


def test_timings_populated_local(env, monkeypatch):
    monkeypatch.setattr(config, "RERANK_ENABLED", True)
    env["cands"][config.CHROMA_COLLECTION] = [_doc("cs61a:1", course="cs61a")]
    res = rag.answer("What is a hash join?", rewrite=False)
    assert set(STAGES) <= set(res.timings)
    assert all(res.timings[k] >= 0 for k in STAGES)
    assert res.timings["rewrite_s"] == 0.0
    parts = sum(res.timings[k] for k in STAGES if k != "total_s")
    assert res.timings["total_s"] >= parts - 1e-6


def test_timings_populated_grounded(env):
    slug = projects.create_project("A")["slug"]
    pc = projects.project_collection(slug)
    env["cands"][pc] = [_doc(f"{pc}:a.pdf:0"), _doc(f"{pc}:a.pdf:1")]
    res = rag.answer_grounded("What is a hash join?", primary_collection=pc, rewrite=False,
                              memory=projects.memory_ids(slug))
    assert set(STAGES) <= set(res.timings)
    assert all(v >= 0 for v in res.timings.values())
    assert res.timings["total_s"] >= sum(v for k, v in res.timings.items() if k != "total_s") - 1e-6

    refused = rag.answer_grounded("What is the capital of France?", primary_collection="empty",
                                  rewrite=False, allow_fallback=False)
    assert refused.text == config.GROUNDED_REFUSAL
    assert "total_s" in refused.timings


def test_rerank_config_honoured(monkeypatch):
    pools = []
    monkeypatch.setattr(config, "RERANK_CANDIDATES", 7)
    monkeypatch.setattr(config, "RERANK_ENABLED", True)
    monkeypatch.setattr(search, "candidates", lambda q, coll, pool, *a: pools.append(pool) or [_doc("x")])
    monkeypatch.setattr(rerank, "rerank_scored", lambda q, ds: [(d, 0.9) for d in ds])
    rag._top_docs("q", "c", gated=True)
    rag._retrieve_local("q")
    assert pools == [7, 7]

    built = []
    fake = types.ModuleType("sentence_transformers")
    fake.CrossEncoder = lambda name, **kw: built.append(name) or object()
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    monkeypatch.setattr(config, "RERANK_MODEL", "some/other-reranker")
    rerank._cross_encoder.cache_clear()
    try:
        rerank._cross_encoder()
        assert built == ["some/other-reranker"]
    finally:
        rerank._cross_encoder.cache_clear()
