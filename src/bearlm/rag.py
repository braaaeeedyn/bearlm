"""Retrieval + RAG core: retrieve, dispatch by backend, and generate cited answers.

Also holds query rewriting, the prompt optimizer, and the grounded answer path
(confidence gate + project-first fallback) used by projects and uploads.
"""

import re
import time
from dataclasses import dataclass, field

from openai import OpenAI

from . import config, db
from .doc_processor import embed_batch

VECTOR_INDEX = "vector_index"
TOP_K = 5
NUM_CANDIDATES = 100
CHAT_MODEL = "gpt-4o-mini"


@dataclass
class Answer:
    text: str
    citations: list[dict]
    source: str = "corpus"  # "project" | "upload" | "corpus" | "none"
    rewritten_query: str | None = None
    docs: list = field(default_factory=list, repr=False)  # exact context used (for eval)
    timings: dict = field(default_factory=dict)  # per-stage seconds (local paths only)
    reused: bool = False  # retrieval reused from a near-repeat question


def _tick(timings, key, t0):
    """Add the seconds since t0 to timings[key] (no-op when timings is None)."""
    if timings is not None:
        timings[key] = timings.get(key, 0.0) + time.perf_counter() - t0


def retrieve(question, k=None):
    """Backend-dispatching retrieval."""
    if config.BACKEND == "local":
        return _retrieve_local(question, k)
    return _retrieve_openai(question, k)


def _retrieve_openai(question, k=None):
    """Embed the question and $vectorSearch Atlas for the top-k chunks."""
    k = k or TOP_K
    qvec = embed_batch([question])[0]
    coll = db.get_chunks_collection()
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX,
                "path": "embedding",
                "queryVector": qvec,
                "numCandidates": NUM_CANDIDATES,
                "limit": k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "course": 1,
                "section": 1,
                "filename": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(coll.aggregate(pipeline))

def build_prompt(question, chunks):
    """Chat messages for the openai backend: system rules + tagged context + question."""
    context = "\n\n".join(f"[{c['course']} {c['section']}] {c['text']}" for c in chunks)
    system = ("You are BearLM, a study assistant for UC Berkeley CS/DS courses. "
              "Answer ONLY using the provided context. If the answer isn't in the "
              "context, say you do not know. Cite sources as (course section). "
              "Write all mathematical notation in LaTeX: wrap inline math in "
              "single dollar signs ($...$) and standalone equations in double "
              "dollar signs ($$...$$)."
              )
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return [ {"role": "system", "content": system}, {"role": "user", "content": user},]


_client = OpenAI() if config.BACKEND == "openai" else None

_CITE_RULE = (
    "Cite sources inline as (course section), copying a source's header, e.g. "
    "(cs188 p.80), after the claims it supports; separate several with \"; \". "
    "Use parentheses, never brackets, and never cite a source that is not in the context. "
)

_STRICT_SYSTEM = (
    "You are BearLM, a study assistant for UC Berkeley CS/DS courses. "
    "Answer the question using ONLY the provided context. "
    "If the context is relevant but incomplete, answer with what it supports and "
    "say briefly what is missing. Only if the context contains nothing relevant "
    "to the question, reply with exactly: "
    "\"I don't know based on the course materials.\" "
    + _CITE_RULE
    + "Format for a Markdown renderer. Use fenced code blocks ONLY for actual code, "
    "tagged with the language (```python, ```sql). Write math in LaTeX (inline "
    "$...$, display $$...$$). Present tabular or relational data as a Markdown "
    "table written directly in the response — never wrap tables or the whole "
    "answer in a code fence."
)


def _format_docs(docs):
    """Render retrieved Documents as context blocks headed by their citable form."""
    out = []
    for d in docs:
        m = d.metadata or {}
        out.append(f"Source ({m.get('course', '?')} {m.get('section', '?')}):\n{d.page_content}")
    return "\n\n".join(out)


# A [..] group (not a markdown link) or a (..) group on one line, up to ~120 chars.
_CITE_GROUP_RE = re.compile(r"\[([^\[\]\n]{1,120})\](?!\()|\(([^()\n]{1,120})\)")
_CITE_LABEL_RE = re.compile(r"^\s*(?:sources?|course section)\s*:\s*", re.IGNORECASE)


def _normalize_citations(text, docs):
    """Rewrite citation markers that name only context chunks into canonical
    "(course section; course section)". Any group with a part that is not a context
    ref stays byte-identical, so this never adds or invents a citation."""
    refs = {}
    for d in docs:
        m = d.metadata or {}
        if m.get("course") and m.get("section"):
            sec = str(m["section"])
            refs[(str(m["course"]).lower(), re.sub(r"\s+", "", sec).lower())] = f"{m['course']} {sec}"
    if not refs or not text:
        return text

    def repl(match):
        body = _CITE_LABEL_RE.sub("", match.group(1) if match.group(1) is not None else match.group(2))
        out = []
        for part in re.split(r"[;,]", body):
            toks = part.split()
            key = (toks[0].lower(), "".join(toks[1:]).lower()) if len(toks) >= 2 else None
            if key not in refs:
                return match.group(0)
            out.append(refs[key])
        return "(" + "; ".join(out) + ")"

    return _CITE_GROUP_RE.sub(repl, text)


def _get_llm():
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=config.OLLAMA_CHAT_MODEL,
        temperature=0,
        num_ctx=config.OLLAMA_NUM_CTX,
        num_predict=config.OLLAMA_NUM_PREDICT,
        base_url=config.OLLAMA_BASE_URL,
        client_kwargs={"timeout": config.OLLAMA_TIMEOUT_S},
    )


TRUNCATED_NOTE = "\n\n_(Answer cut off at the length limit.)_"


def _finish(msg, docs):
    """Normalize cites in a generated AIMessage; flag output cut off by num_predict."""
    text = msg.content
    if "I don't know based on the" not in text:
        text = _normalize_citations(text, docs)
    if (msg.response_metadata or {}).get("done_reason") == "length":
        text += TRUNCATED_NOTE
    return text


def _retrieve_local(question, k=None, timings=None):
    """Retrieve (hybrid BM25+dense when enabled, optionally reranked) for a question."""
    from . import search

    k = k or config.RETRIEVAL_K
    t0 = time.perf_counter()
    if config.RERANK_ENABLED:
        from . import rerank

        cands = search.candidates(question, config.CHROMA_COLLECTION, config.RERANK_CANDIDATES)
        _tick(timings, "search_s", t0)
        t0 = time.perf_counter()
        docs = rerank.rerank(question, cands, config.RERANK_TOP_N)
        _tick(timings, "rerank_s", t0)
        return docs
    docs = search.candidates(question, config.CHROMA_COLLECTION, k)
    _tick(timings, "search_s", t0)
    return docs


def build_chain(k=None):
    """LCEL chain (retriever | prompt | llm | StrOutputParser) mapping question -> answer text."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough

    prompt = ChatPromptTemplate.from_messages(
        [("system", _STRICT_SYSTEM), ("human", "Context:\n{context}\n\nQuestion: {question}")]
    )
    retrieve_docs = RunnableLambda(lambda q: _retrieve_local(q, k))
    return (
        {"context": retrieve_docs | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | _get_llm()
        | StrOutputParser()
    )


def _answer_local(question, k=None, rewrite=None):  # (str, int|None, bool|None) -> Answer
    from langchain_core.prompts import ChatPromptTemplate

    start = time.perf_counter()
    timings = {"rewrite_s": 0.0, "search_s": 0.0, "rerank_s": 0.0, "generate_s": 0.0}
    if rewrite is None:
        rewrite = config.QUERY_REWRITE_ENABLED
    rq = rewrite_query(question) if rewrite else question
    if rewrite:
        _tick(timings, "rewrite_s", start)
    rewritten = rq if (rewrite and rq != question) else None
    docs = _retrieve_local(rq, k, timings)
    if not docs:
        _tick(timings, "total_s", start)
        return Answer("I don't know based on the course materials.", [], rewritten_query=rewritten,
                      timings=timings)

    prompt = ChatPromptTemplate.from_messages(
        [("system", _STRICT_SYSTEM), ("human", "Context:\n{context}\n\nQuestion: {question}")]
    )
    chain = prompt | _get_llm()
    t0 = time.perf_counter()
    msg = chain.invoke({"context": _format_docs(docs), "question": question})
    _tick(timings, "generate_s", t0)
    text = _finish(msg, docs)

    seen, citations = set(), []
    for d in docs:
        m = d.metadata or {}
        key = (m.get("course"), m.get("section"))
        if key not in seen:
            seen.add(key)
            citations.append({
                "course": m.get("course"),
                "section": m.get("section"),
                "source": m.get("source"),
            })
    _tick(timings, "total_s", start)
    return Answer(text, citations, rewritten_query=rewritten, docs=docs, timings=timings)


def _dedupe_citations(docs):
    seen, citations = set(), []
    for d in docs:
        m = d.metadata or {}
        key = (m.get("course"), m.get("section"), m.get("source"))
        if key not in seen:
            seen.add(key)
            citations.append({
                "course": m.get("course"),
                "section": m.get("section"),
                "source": m.get("source"),
            })
    return citations


# ---------------------------------------------------------------------------
# GROUNDED ANSWERS with a confidence gate (projects + uploads, anti-hallucination)
# ---------------------------------------------------------------------------

# Meta/summary questions ("what is in this pdf", "summarize this") are topically
# orthogonal to any single passage, so the cross-encoder scores them ~0 and the
# confidence gate would wrongly refuse. Detect them and answer from the doc's top
# chunks directly (the strict prompt still prevents hallucination).
_META_RE = re.compile(
    r"\b(summari[sz]e|summary|overview|tl;?dr|abstract|the gist|"
    r"what(?:'?s| is| are)?\s+(?:in|inside|contained in)\b|"
    r"what(?:'?s| is)\s+this\b|"
    r"what(?:'?s| is| does)\s+(?:this|the)\s+(?:document|pdf|file|paper|doc)\b|"
    r"main\s+(?:points|ideas|takeaways|topics|themes)|"
    r"key\s+(?:points|takeaways|ideas|topics|themes)|"
    r"tell me about\s+(?:this|the)|give (?:me )?(?:a|an)\s+(?:summary|overview))",
    re.IGNORECASE,
)


def _is_meta_question(question: str) -> bool:
    return bool(_META_RE.search(question or ""))


def _top_docs(question, collection_name, gated=True, allowed_courses=None, extra_ids=None,
              timings=None):
    """Vector-retrieve a pool from one collection and rerank it.

    gated=True keeps only chunks clearing config.RERANK_MIN_SCORE (the confidence
    gate → empty list means "not covered", so callers refuse or fall back).
    gated=False skips the threshold and just returns the reranked top chunks (for
    meta/summary questions, where topical scores are uninformative).
    extra_ids (gated only): remembered chunk ids added to the pool; they are
    re-scored by the reranker and must clear the gate like any other chunk.
    """
    from . import rerank, search

    t0 = time.perf_counter()
    cands = search.candidates(question, collection_name, config.RERANK_CANDIDATES, allowed_courses)
    if gated and extra_ids:
        from . import local_store

        # Corpus ids may point at new text after a re-embed; harmless, the reranker scores the text.
        cands, seen = list(cands), {search._doc_key(d) for d in cands}
        for d in local_store.get_docs_by_ids(collection_name, extra_ids):
            if allowed_courses and (d.metadata or {}).get("course") not in allowed_courses:
                continue
            if search._doc_key(d) not in seen:
                seen.add(search._doc_key(d))
                cands.append(d)
    _tick(timings, "search_s", t0)
    if not cands:
        return []
    t0 = time.perf_counter()
    scored = rerank.rerank_scored(question, cands)
    _tick(timings, "rerank_s", t0)
    if gated:
        scored = [(d, s) for d, s in scored if s >= config.RERANK_MIN_SCORE]
    return [d for d, _ in scored[: config.RERANK_TOP_N]]


def _retrieval_query(question, history):
    """Retrieval query for a (possibly follow-up) turn: prepend the previous user
    turn so short follow-ups like "explain more" still retrieve the right chunks."""
    prev = None
    for h in reversed(history or []):
        if h.get("role") == "user":
            prev = h.get("content")
            break
    return f"{prev} {question}".strip() if prev else question


def rewrite_query(question, history=None):
    """LLM-rewrite a (possibly conversational) question into a standalone, keyword-
    rich retrieval query. Resolves pronouns from history. Falls back to the raw
    question on any error. Used when query rewriting is enabled."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import StrOutputParser

    convo = ""
    if history:
        turns = [f"{h.get('role')}: {h.get('content', '')}" for h in history[-4:]]
        convo = "Conversation so far:\n" + "\n".join(turns) + "\n\n"
    system = (
        "You rewrite a user's question into a single concise, keyword-rich search "
        "query for retrieving relevant document passages. Resolve pronouns and "
        "references using the conversation. Output ONLY the query text — no quotes, "
        "no preamble, one line."
    )
    try:
        out = (_get_llm() | StrOutputParser()).invoke(
            [SystemMessage(content=system),
             HumanMessage(content=f"{convo}Question: {question}\n\nSearch query:")]
        )
        line = out.strip().strip('"').splitlines()[0].strip() if out else ""
        return line or question
    except Exception:
        return question


def optimize_prompt(prompt):
    """Return an improved, prompt-engineered version of the user's prompt (for the
    suggest-then-confirm UI). Does not answer the question or invent facts."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import StrOutputParser

    system = (
        "You are a prompt engineer for a grounded document Q&A assistant. Rewrite "
        "the user's prompt to be clearer and more effective: make the ask specific "
        "and unambiguous, state the desired form and level of detail, and stay "
        "faithful to the original intent. Do NOT answer the question, add facts, or "
        "include commentary. Output ONLY the improved prompt."
    )
    try:
        out = (_get_llm() | StrOutputParser()).invoke(
            [SystemMessage(content=system), HumanMessage(content=prompt)]
        )
        return (out or "").strip() or prompt
    except Exception:
        return prompt


def _make_retrieval_query(question, history, rewrite):
    """Pick the retrieval query: LLM rewrite if enabled, else prev-turn prepend."""
    if rewrite is None:
        rewrite = config.QUERY_REWRITE_ENABLED
    return rewrite_query(question, history) if rewrite else _retrieval_query(question, history)


def answer_grounded(question, primary_collection=None, primary_label="project",
                    allow_fallback=None, history=None, rewrite=None, corpus_courses=None, memory=None,
                    reuse=None):
    """Grounded answer with a confidence gate, project-first fallback, and
    optional multi-turn conversation history.

    - Try the primary collection (a project or an upload). If confident chunks
      exist, answer from them (source = primary_label).
    - Else, if fallback is allowed, try the main course corpus.
    - Else refuse with config.GROUNDED_REFUSAL rather than guess.
    `history` is a list of prior {role, content} turns for conversational memory.
    `memory` is {collection: [chunk ids]} merged into the gated rerank pools.
    `reuse` is a project's near-repeat cache of (query vector, docs, source): a
    non-meta question within config.NEAR_REPEAT_THRESHOLD cosine of a cached one
    reuses that entry's gated outcome (answer or refusal) and skips retrieval.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    if allow_fallback is None:
        allow_fallback = config.PROJECT_FALLBACK_TO_CORPUS

    from . import local_store

    start = time.perf_counter()
    timings = {"rewrite_s": 0.0, "reuse_s": 0.0, "search_s": 0.0, "rerank_s": 0.0, "generate_s": 0.0}
    if rewrite is None:
        rewrite = config.QUERY_REWRITE_ENABLED
    if rewrite:
        rq = rewrite_query(question, history)
        rewritten = rq if rq != question else None
        _tick(timings, "rewrite_s", start)
    else:
        rq = _retrieval_query(question, history)
        rewritten = None
    docs, used = [], "none"
    meta = _is_meta_question(question)

    vec, hit = None, None
    if reuse is not None and config.NEAR_REPEAT_THRESHOLD > 0 and not meta:
        t0 = time.perf_counter()
        vec = local_store.get_embeddings().embed_query(rq)
        best = max(reuse, key=lambda e: _cosine(vec, e[0]), default=None)
        if best is not None and _cosine(vec, best[0]) >= config.NEAR_REPEAT_THRESHOLD:
            hit = best
        _tick(timings, "reuse_s", t0)

    if hit is not None:
        docs, used = list(hit[1]), hit[2]
    elif primary_collection:
        if meta and local_store.collection_count(primary_collection) > 0:
            # "Summarize / what's in this doc" — answer from the doc's top chunks
            # without the confidence gate (which would wrongly refuse these).
            docs = _top_docs(rq, primary_collection, gated=False, timings=timings)
            used = primary_label if docs else "none"
        else:
            docs = _top_docs(rq, primary_collection, gated=True,
                             extra_ids=(memory or {}).get(primary_collection), timings=timings)
            if docs:
                used = primary_label

    # Fall back to the course corpus for uncovered questions — but NOT for a meta
    # question about "this" doc, and NOT if the project disabled corpus fallback
    # (corpus_courses == [] means "project docs only"). corpus_courses may also
    # restrict fallback to a subset of subjects.
    corpus_disabled = corpus_courses is not None and len(corpus_courses) == 0
    if (hit is None and not docs and (allow_fallback or not primary_collection)
            and not (meta and primary_collection) and not corpus_disabled):
        docs = _top_docs(rq, config.CHROMA_COLLECTION, gated=True, allowed_courses=corpus_courses,
                         extra_ids=(memory or {}).get(config.CHROMA_COLLECTION), timings=timings)
        if docs:
            used = "corpus"

    if vec is not None and hit is None:
        reuse.append((vec, list(docs), used))  # refusals too (docs=[], used="none")
    reused = hit is not None

    if not docs:
        _tick(timings, "total_s", start)
        return Answer(config.GROUNDED_REFUSAL, [], source="none", rewritten_query=rewritten,
                      timings=timings, reused=reused)

    system = (
        "You are BearLM, a grounded study assistant having a conversation. Answer "
        "the latest question using ONLY the provided context (plus the conversation "
        "so far for reference). If the context is relevant but incomplete, answer "
        "with what it supports and say briefly what is missing. Only if the context "
        "contains nothing relevant to the question, reply with exactly: "
        f"\"{config.GROUNDED_REFUSAL}\" " + _CITE_RULE
        + "Format for a Markdown renderer: fenced code blocks ONLY "
        "for actual code (```python, ```sql); math in LaTeX (inline $...$, display "
        "$$...$$); tabular or relational data as a Markdown table written directly, "
        "never wrapped in a code fence."
    )
    msgs = [SystemMessage(content=system)]
    for h in (history or [])[-6:]:  # last few turns for context
        content = h.get("content", "")
        msgs.append(HumanMessage(content=content) if h.get("role") == "user"
                    else AIMessage(content=content))
    msgs.append(HumanMessage(content=f"Context:\n{_format_docs(docs)}\n\nQuestion: {question}"))

    t0 = time.perf_counter()
    msg = _get_llm().invoke(msgs)
    _tick(timings, "generate_s", t0)
    text = _finish(msg, docs)
    _tick(timings, "total_s", start)
    return Answer(text, _dedupe_citations(docs), source=used, rewritten_query=rewritten, docs=docs,
                  timings=timings, reused=reused)


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = sum(x * x for x in a) ** 0.5, sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def answer(question, k=None, rewrite=None):
    """Backend-dispatching entry point the UI/eval calls."""
    if config.BACKEND == "local":
        return _answer_local(question, k, rewrite)
    return _answer_openai(question, k)


def _answer_openai(question, k=None):
    chunks = retrieve(question, k)
    if not chunks:
        return Answer("I couldn't find anything relevant.", [])
    messages = build_prompt(question, chunks)
    response = _client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    text = response.choices[0].message.content
    seen, citations = set(), []
    for chunk in chunks:
        key = (chunk["course"], chunk["section"])
        if key not in seen:
            seen.add(key)
            citations.append({
                "course": chunk["course"],
                "section": chunk["section"],
                "source": chunk.get("filename"),
            })
    return Answer(text, citations, docs=chunks)


if __name__ == "__main__":
    q = "What is the difference between a list and a tuple in Python?"
    result = answer(q)
    print(result.text)
    print("\nSources:")
    for c in result.citations:
        print(f"  - {c['course']} {c['section']}")
