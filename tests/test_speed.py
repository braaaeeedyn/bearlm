import itertools
import json
import sys
import types
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from bearlm import api, config, local_store, projects, rag, rerank, search

STAGES = ("rewrite_s", "search_s", "rerank_s", "generate_s", "total_s")


def _doc(cid, course="p1"):
    return Document(page_content=f"text {cid}",
                    metadata={"course": course, "section": "p.1", "source": "a.pdf", "chunk_id": cid})


class FakeEmb:
    """embed_query maps known strings to fixed vectors (unknown -> orthogonal axis)."""

    def __init__(self, vecs):
        self.vecs, self.calls = vecs, []

    def embed_query(self, text):
        self.calls.append(text)
        return self.vecs.get(text, [0.0, 0.0, 1.0])


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(config, "NEAR_REPEAT_THRESHOLD", 0.9)
    monkeypatch.setattr(projects, "_reuse", {})
    ticks = itertools.count()
    monkeypatch.setattr(projects, "_now", lambda: f"2026-01-01T00:00:{next(ticks):06d}")
    state = {"cands": {}, "scores": {}, "searches": 0, "reranks": 0, "llm_calls": 0}

    def cands(q, coll, *a, **k):
        state["searches"] += 1
        return list(state["cands"].get(coll, []))

    def scored(q, ds):
        state["reranks"] += 1
        return sorted(((d, state["scores"].get(d.metadata["chunk_id"], 0.5)) for d in ds),
                      key=lambda x: x[1], reverse=True)

    monkeypatch.setattr(search, "candidates", cands)
    monkeypatch.setattr(rerank, "rerank_scored", scored)
    monkeypatch.setattr(local_store, "collection_count", lambda name: 1)
    monkeypatch.setattr(local_store, "get_docs_by_ids", lambda coll, ids: [])

    def llm(msgs):
        state["llm_calls"] += 1
        return AIMessage(content="answer")

    monkeypatch.setattr(rag, "_get_llm", lambda: RunnableLambda(llm))
    emb = FakeEmb({
        "What is a hash join?": [1.0, 0.0, 0.0],
        "what is a hash join": [0.99, 0.05, 0.0],
        "What is a B+ tree?": [0.0, 1.0, 0.0],
        "What is the capital of France?": [0.6, 0.0, 0.8],
        "capital of France?": [0.62, 0.0, 0.78],
    })
    monkeypatch.setattr(local_store, "get_embeddings", lambda: emb)
    state["emb"] = emb
    return state


def _new(name):
    slug = projects.create_project(name)["slug"]
    return slug, projects.create_chat(slug)["id"]


def _ask(slug, q):
    pc = projects.project_collection(slug)
    return rag.answer_grounded(q, primary_collection=pc, rewrite=False,
                               memory=projects.memory_ids(slug), reuse=projects.reuse_cache(slug))


def _seed(env, slug):
    pc = projects.project_collection(slug)
    env["cands"][pc] = [_doc(f"{pc}:a.pdf:0"), _doc(f"{pc}:a.pdf:1")]
    return pc


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
    slug, _ = _new("A")
    _seed(env, slug)
    res = _ask(slug, "What is a hash join?")
    assert set(STAGES) | {"reuse_s"} <= set(res.timings)
    assert all(v >= 0 for v in res.timings.values())
    assert res.timings["total_s"] >= sum(v for k, v in res.timings.items() if k != "total_s") - 1e-6

    refused = rag.answer_grounded("What is the capital of France?", primary_collection="empty",
                                  rewrite=False, allow_fallback=False)
    assert refused.text == config.GROUNDED_REFUSAL
    assert "total_s" in refused.timings


def test_reuse_hits_above_threshold(env):
    slug, _ = _new("A")
    _seed(env, slug)
    first = _ask(slug, "What is a hash join?")
    s, r = env["searches"], env["reranks"]
    second = _ask(slug, "what is a hash join")
    assert (env["searches"], env["reranks"]) == (s, r)
    assert second.reused and not first.reused
    assert second.docs == first.docs and second.source == first.source == "project"


def test_reuse_misses_below_threshold(env):
    slug, _ = _new("A")
    _seed(env, slug)
    _ask(slug, "What is a hash join?")
    s = env["searches"]
    res = _ask(slug, "What is a B+ tree?")
    assert not res.reused and env["searches"] > s


def test_reuse_per_project(env):
    a, _ = _new("A")
    b, _ = _new("B")
    _seed(env, a)
    _seed(env, b)
    _ask(a, "What is a hash join?")
    s = env["searches"]
    res = _ask(b, "What is a hash join?")
    assert not res.reused and env["searches"] > s


def _assert_cleared(env, slug):
    s = env["searches"]
    res = _ask(slug, "What is a hash join?")
    assert not res.reused and env["searches"] > s


def test_reset_clears_reuse(env):
    slug, chat = _new("A")
    _seed(env, slug)
    api.ask_chat(slug, chat, api.AskRequest(question="What is a hash join?", rewrite=False))
    assert len(projects.reuse_cache(slug)) == 1
    projects.clear_chats(slug)
    _assert_cleared(env, slug)


def test_remove_pdf_clears_reuse(env, monkeypatch):
    monkeypatch.setattr(local_store, "delete_pdf_from_collection", lambda coll, fn: None)
    slug, _ = _new("A")
    _seed(env, slug)
    _ask(slug, "What is a hash join?")
    projects.remove_pdf(slug, "a.pdf")
    _assert_cleared(env, slug)


def test_project_changes_clear_reuse(env, monkeypatch):
    monkeypatch.setattr(local_store, "ingest_pdf_into_collection", lambda *a, **k: 3)
    slug, _ = _new("A")
    _seed(env, slug)
    _ask(slug, "What is a hash join?")
    projects.add_pdf_to_project(slug, b"%PDF-1.4", "b.pdf")
    _assert_cleared(env, slug)
    projects.set_corpus_scope(slug, ["data101"])
    _assert_cleared(env, slug)


def test_reuse_keeps_refusal(env):
    slug, _ = _new("A")
    pc = _seed(env, slug)
    for d in env["cands"][pc]:
        env["scores"][d.metadata["chunk_id"]] = 0.0
    env["cands"][config.CHROMA_COLLECTION] = [_doc("cs61a:1", course="cs61a")]
    env["scores"]["cs61a:1"] = 0.0
    first = _ask(slug, "What is the capital of France?")
    assert first.text == config.GROUNDED_REFUSAL
    env["scores"].clear()  # a fresh run would now answer; the reused one must not
    s = env["searches"]
    second = _ask(slug, "capital of France?")
    assert second.reused and second.text == config.GROUNDED_REFUSAL and second.source == "none"
    assert env["searches"] == s and env["llm_calls"] == 0


def test_reuse_keeps_answer(env):
    slug, _ = _new("A")
    _seed(env, slug)
    env["cands"][config.CHROMA_COLLECTION] = [_doc("cs61a:1", course="cs61a")]
    first = _ask(slug, "What is a hash join?")
    r = env["reranks"]
    second = _ask(slug, "what is a hash join")
    assert second.reused and second.text == "answer"
    assert second.docs == first.docs and second.source == first.source
    assert second.citations == first.citations
    assert env["reranks"] == r and env["llm_calls"] == 2


def test_meta_question_bypasses_reuse(env):
    slug, _ = _new("A")
    _seed(env, slug)
    _ask(slug, "Summarize this pdf")
    _ask(slug, "Summarize this pdf")
    assert env["emb"].calls == []
    assert len(projects.reuse_cache(slug)) == 0


def test_reuse_off_when_threshold_zero(env, monkeypatch):
    monkeypatch.setattr(config, "NEAR_REPEAT_THRESHOLD", 0.0)
    slug, _ = _new("A")
    _seed(env, slug)
    _ask(slug, "What is a hash join?")
    res = _ask(slug, "What is a hash join?")
    assert not res.reused and env["emb"].calls == []


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


def test_experiment_results_schema():
    data = json.loads((Path(__file__).resolve().parent.parent / "eval" / "speed_experiments.json")
                      .read_text(encoding="utf-8"))
    assert isinstance(data["machine"]["torch_cuda"], bool)
    assert isinstance(data["machine"]["rerank_device"], str)

    sb = data["stage_breakdown"]
    assert isinstance(sb["n"], int) and sb["n"] > 0
    for k in ("median_s", "p95_s"):
        assert set(STAGES) <= set(sb[k])
        assert all(isinstance(sb[k][s], (int, float)) for s in STAGES)

    nr = data["near_repeat"]
    assert nr["decision"] in ("keep", "reject")
    for k in ("criteria", "metrics_at_threshold", "sweep", "reason"):
        assert k in nr
    assert isinstance(nr["sweep"], list) and nr["sweep"]
    if nr["decision"] == "keep":
        assert isinstance(nr["threshold"], float)
        assert config.NEAR_REPEAT_THRESHOLD == nr["threshold"]
    else:
        assert nr["threshold"] is None or isinstance(nr["threshold"], float)
        assert config.NEAR_REPEAT_THRESHOLD == 0

    cr = data["cheaper_rerank"]
    assert cr["decision"] in ("keep", "reject")
    for k in ("criteria", "answer_quality", "reason"):
        assert k in cr
    assert cr["configs"]
    for row in cr["configs"]:
        for k in ("id", "model", "candidates", "median_search_s", "median_rerank_s",
                  "recall@5", "precision@5", "mrr"):
            assert k in row
    if cr["decision"] == "keep":
        chosen = cr["chosen"]
        assert {"model", "candidates", "min_score"} <= set(chosen)
        assert config.RERANK_MODEL == chosen["model"]
        assert config.RERANK_CANDIDATES == chosen["candidates"]
        assert config.RERANK_MIN_SCORE == chosen["min_score"]
    else:
        assert cr["chosen"] is None
        assert config.RERANK_MODEL == "BAAI/bge-reranker-base"
        assert config.RERANK_CANDIDATES == 15
