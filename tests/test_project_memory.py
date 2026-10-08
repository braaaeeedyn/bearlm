import itertools

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from bearlm import api, config, local_store, projects, rag, rerank, search


def _doc(cid, course="p1", text=None):
    return Document(page_content=text or f"text {cid}",
                    metadata={"course": course, "section": "p.1", "source": "a.pdf", "chunk_id": cid})


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Temp projects dir, mocked retrieval/rerank/LLM, recorded memory fetches."""
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path / "projects")
    ticks = itertools.count()
    monkeypatch.setattr(projects, "_now", lambda: f"2026-01-01T00:00:{next(ticks):06d}")
    state = {"cands": {}, "scores": {}, "store": {}, "fetches": [], "reranked": [], "llm_calls": 0}

    monkeypatch.setattr(search, "candidates", lambda q, coll, *a, **k: list(state["cands"].get(coll, [])))
    monkeypatch.setattr(rerank, "rerank_scored", lambda q, ds: (
        state["reranked"].append(list(ds)),
        sorted(((d, state["scores"].get(d.metadata["chunk_id"], 0.5)) for d in ds),
               key=lambda x: x[1], reverse=True))[1])
    monkeypatch.setattr(local_store, "collection_count", lambda name: 1)

    def fetch(coll, ids):
        state["fetches"].append((coll, list(ids)))
        return [state["store"][i] for i in ids if i in state["store"]]

    monkeypatch.setattr(local_store, "get_docs_by_ids", fetch)

    def llm(msgs):
        state["llm_calls"] += 1
        return AIMessage(content="answer")

    monkeypatch.setattr(rag, "_get_llm", lambda: RunnableLambda(llm))
    return state


def _new(name):
    slug = projects.create_project(name)["slug"]
    return slug, projects.create_chat(slug)["id"]


def _ask(slug, chat, q):
    return api.ask_chat(slug, chat, api.AskRequest(question=q, rewrite=False))


def _mem(slug):
    return projects.get_project(slug).get("memory", {})


def test_answer_records_chunk_ids(env):
    slug, chat = _new("A")
    pc = projects.project_collection(slug)
    env["cands"][pc] = [_doc(f"{pc}:a.pdf:0"), _doc(f"{pc}:a.pdf:1")]
    res = _ask(slug, chat, "What is a hash join?")
    assert res.source == "project"
    assert set(_mem(slug)) == {pc}
    assert set(_mem(slug)[pc]) == {f"{pc}:a.pdf:0", f"{pc}:a.pdf:1"}

    # Corpus fallback: the project has nothing above the gate.
    for d in env["cands"][pc]:
        env["scores"][d.metadata["chunk_id"]] = 0.0
    env["cands"][config.CHROMA_COLLECTION] = [_doc("data101:5", course="data101")]
    res = _ask(slug, chat, "What is a B+ tree?")
    assert res.source == "corpus"
    assert set(_mem(slug)[config.CHROMA_COLLECTION]) == {"data101:5"}


def test_cached_chunks_join_candidate_pool(env):
    slug, chat = _new("A")
    pc = projects.project_collection(slug)
    x = _doc(f"{pc}:a.pdf:9")
    env["store"][x.metadata["chunk_id"]] = x
    projects.remember(slug, pc, [x, _doc(f"{pc}:a.pdf:0")])
    env["cands"][pc] = [_doc(f"{pc}:a.pdf:0"), _doc(f"{pc}:a.pdf:1")]  # search misses X this time
    env["scores"][x.metadata["chunk_id"]] = 0.9

    res = rag.answer_grounded("follow-up", primary_collection=pc, rewrite=False,
                              memory=projects.memory_ids(slug))
    pool = [d.metadata["chunk_id"] for d in env["reranked"][-1]]
    assert pool.count(x.metadata["chunk_id"]) == 1
    assert pool.count(f"{pc}:a.pdf:0") == 1  # deduped against search results
    assert x in res.docs


def test_off_topic_with_warm_cache_refused(env):
    slug, chat = _new("A")
    pc = projects.project_collection(slug)
    x = _doc(f"{pc}:a.pdf:9")
    env["store"][x.metadata["chunk_id"]] = x
    projects.remember(slug, pc, [x])
    before = _mem(slug)
    env["cands"][pc] = [_doc(f"{pc}:a.pdf:1")]
    env["cands"][config.CHROMA_COLLECTION] = [_doc("cs61a:1", course="cs61a")]
    env["scores"].update({x.metadata["chunk_id"]: 0.001, f"{pc}:a.pdf:1": 0.0, "cs61a:1": 0.0})

    res = _ask(slug, chat, "What is the capital of France?")
    assert res.answer == config.GROUNDED_REFUSAL
    assert env["llm_calls"] == 0
    assert any(x in pool for pool in env["reranked"])  # cached chunk was scored, not trusted
    assert _mem(slug) == before


def test_meta_question_ignores_memory(env):
    slug, chat = _new("A")
    pc = projects.project_collection(slug)
    projects.remember(slug, pc, [_doc(f"{pc}:a.pdf:9")])
    before = _mem(slug)
    env["cands"][pc] = [_doc(f"{pc}:a.pdf:0")]
    res = _ask(slug, chat, "Summarize this pdf")
    assert res.source == "project"
    assert env["fetches"] == []
    assert _mem(slug) == before


def test_projects_isolated(env):
    a, _ = _new("A")
    b, chat_b = _new("B")
    pa, pb = projects.project_collection(a), projects.project_collection(b)
    projects.remember(a, pa, [_doc(f"{pa}:a.pdf:0")])
    projects.remember(a, config.CHROMA_COLLECTION, [_doc("data101:1", course="data101")])
    assert projects.memory_ids(b) == {}

    env["cands"][pb] = [_doc(f"{pb}:b.pdf:0")]
    _ask(b, chat_b, "What is a join?")
    _ask(b, chat_b, "And a hash join?")
    asked = [i for _, ids in env["fetches"] for i in ids]
    assert f"{pa}:a.pdf:0" not in asked and "data101:1" not in asked


def test_memory_capped_lru(env, monkeypatch):
    slug, _ = _new("A")
    pc = projects.project_collection(slug)
    monkeypatch.setattr(config, "PROJECT_MEMORY_SIZE", 3)
    monkeypatch.setattr(config, "PROJECT_MEMORY_MERGE", 2)
    for i in range(5):
        projects.remember(slug, pc, [_doc(f"c{i}")])
    projects.remember(slug, pc, [_doc("c2")])  # touching c2 makes it most recent
    assert set(_mem(slug)[pc]) == {"c2", "c3", "c4"}
    assert projects.memory_ids(slug)[pc] == ["c2", "c4"]

    other, _ = _new("B")
    monkeypatch.setattr(config, "PROJECT_MEMORY_SIZE", 0)
    projects.remember(other, projects.project_collection(other), [_doc("z")])
    assert _mem(other) == {}
    assert projects.memory_ids(slug) == {}


def test_reset_clears_memory(env):
    slug, _ = _new("A")
    projects.remember(slug, projects.project_collection(slug), [_doc("c0")])
    assert projects.memory_ids(slug)
    api.reset_project_context(slug)
    assert projects.memory_ids(slug) == {}
    assert _mem(slug) == {}


def test_stale_ids_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path / "projects")
    slug = projects.create_project("A")["slug"]
    pc = projects.project_collection(slug)
    keep, gone = f"{pc}:a.pdf:0", f"{pc}:a.pdf:1"
    other = f"{pc}:b.pdf:0"
    local_store.get_vectorstore(pc)._collection.add(
        ids=[keep, gone, other], documents=["kept", "gone", "other"],
        metadatas=[{"source": "a.pdf", "chunk_id": keep}, {"source": "a.pdf", "chunk_id": gone},
                   {"source": "b.pdf", "chunk_id": other}],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    local_store.get_vectorstore(pc)._collection.delete(ids=[gone])

    got = local_store.get_docs_by_ids(pc, [keep, gone, "never-existed"])
    assert [d.metadata["chunk_id"] for d in got] == [keep]
    assert got[0].page_content == "kept"
    assert local_store.get_docs_by_ids("no_such_collection", [keep]) == []
    assert local_store.get_docs_by_ids(pc, []) == []

    projects.remember(slug, pc, [_doc(keep), _doc(other)])
    projects.remove_pdf(slug, "a.pdf")
    assert set(_mem(slug)[pc]) == {other}


def test_corpus_memory_respects_course_scope(env):
    slug, chat = _new("A")
    projects.set_corpus_scope(slug, ["data101"])
    ok, bad = _doc("data101:7", course="data101"), _doc("cs61a:7", course="cs61a")
    env["store"].update({"data101:7": ok, "cs61a:7": bad})
    projects.remember(slug, config.CHROMA_COLLECTION, [ok, bad])
    env["cands"][config.CHROMA_COLLECTION] = [_doc("data101:1", course="data101")]

    res = _ask(slug, chat, "What is a B+ tree?")
    assert res.source == "corpus"
    pool = [d.metadata["chunk_id"] for p in env["reranked"] for d in p]
    assert "data101:7" in pool
    assert "cs61a:7" not in pool
