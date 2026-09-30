"""Stage 5 — Retrieval + RAG core.

Turn a user question into a cited answer:
    question -> embed -> $vectorSearch over bearlm.chunks -> top-k chunks
             -> build a prompt (system rules + chunks + question)
             -> chat model -> answer + citations (course + section)

Run the smoke test (once you've filled it in) with:
    uv run python -m bearlm.rag

--------------------------------------------------------------------------
SUGGESTIONS ONLY. Each function has a signature + a step-by-step recipe in
comments. You write the actual lines. `raise NotImplementedError` marks the
spots to replace. Do them roughly top-to-bottom.

Prereqs (already done): ingestion ran (bearlm.chunks is full) and the Atlas
Vector Search index named `vector_index` is ACTIVE on the `embedding` field.
--------------------------------------------------------------------------
"""

# TODO(you): imports you'll likely need
#   - `from dataclasses import dataclass`          (a tidy way to model an Answer)
#   - `from openai import OpenAI`
#   - `from . import config, db`
#   - reuse `embed_batch` from doc_processor so you don't duplicate embedding code:
#       `from .doc_processor import embed_batch`
from dataclasses import dataclass
from openai import OpenAI
from . import config, db
from .doc_processor import embed_batch


# ---------------------------------------------------------------------------
# Config knobs for retrieval (tweak later if answers are thin or noisy)
# ---------------------------------------------------------------------------
# TODO(you): pick sensible defaults, e.g.
#   VECTOR_INDEX   = "vector_index"   # must match the Atlas index name exactly
#   TOP_K          = 5                # how many chunks to retrieve
#   NUM_CANDIDATES = 100              # $vectorSearch scans this many, returns TOP_K
#   CHAT_MODEL     = "gpt-4o-mini"    # cheap + good enough for grounded Q&A
VECTOR_INDEX = "vector_index"
TOP_K = 5
NUM_CANDIDATES = 100
CHAT_MODEL = "gpt-4o-mini"

# A tidy return type so callers (the Streamlit UI in Stage 6) get answer + sources.
# TODO(you): turn this into a real dataclass, e.g.
#   @dataclass
#   class Answer:
#       text: str                 # the model's written answer
#       citations: list[dict]     # e.g. [{"course": "cs61a", "section": "p.12"}]
@dataclass
class Answer:
    text: str
    citations: list[dict]
    source: str = "corpus"  # which collection answered: "project" | "upload" | "corpus" | "none"
    rewritten_query: str | None = None  # the LLM-rewritten retrieval query, if one was used

# ---------------------------------------------------------------------------
# 1. RETRIEVAL  — question  ->  top-k most similar chunks
# ---------------------------------------------------------------------------

def retrieve(question, k=None):  # (str, int|None) -> list[dict]
    """Backend-dispatching retrieval. See _retrieve_openai / _retrieve_local."""
    if config.BACKEND == "local":
        return _retrieve_local(question, k)
    return _retrieve_openai(question, k)


def _retrieve_openai(question, k=None):  # (str, int|None) -> list[dict]
    """Embed the question and return the top-k most similar chunk documents.

    Recipe:
      1. k = k or TOP_K
      2. qvec = embed_batch([question])[0]      # reuse the ingestion embedder
      3. coll = db.get_chunks_collection()
      4. build the aggregation pipeline:
           pipeline = [
             {"$vectorSearch": {
                 "index": VECTOR_INDEX,
                 "path": "embedding",
                 "queryVector": qvec,
                 "numCandidates": NUM_CANDIDATES,
                 "limit": k,
             }},
             {"$project": {
                 "_id": 0,
                 "text": 1, "course": 1, "section": 1, "filename": 1,
                 "score": {"$meta": "vectorSearchScore"},
             }},
           ]
      5. return list(coll.aggregate(pipeline))

    Watch out: `index` must equal the Atlas index name (`vector_index`), NOT the
    collection name. If you get an empty list, the index probably isn't ACTIVE yet.
    """
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




# ---------------------------------------------------------------------------
# 2. PROMPT  — retrieved chunks + question  ->  messages for the chat model
# ---------------------------------------------------------------------------

def build_prompt(question, chunks):  # (str, list[dict]) -> list[dict]
    """Assemble the chat `messages` list: a system rule + context + the question.

    Why label each chunk with its course/section? So the model can point at
    sources in its wording, and so YOU can attach structured citations after.

    Recipe:
      1. Build one context string from the chunks, each block tagged, e.g.:
           context = "\n\n".join(
               f"[{c['course']} {c['section']}] {c['text']}" for c in chunks
           )
      2. system = (
             "You are BearLM, a study assistant for UC Berkeley CS/DS courses. "
             "Answer ONLY using the provided context. If the answer isn't in the "
             "context, say you don't know. Cite sources as (course section)."
         )
      3. user = f"Context:\n{context}\n\nQuestion: {question}"
      4. return [
             {"role": "system", "content": system},
             {"role": "user", "content": user},
         ]

    Later upgrade: cap the context length so very long chunk sets don't blow the
    model's context window (e.g. keep only the first N chars per chunk).
    """
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


# ---------------------------------------------------------------------------
# 3. ANSWER  — tie retrieval + prompt + chat model together
# ---------------------------------------------------------------------------

_client = OpenAI() if config.BACKEND == "openai" else None


# ---------------------------------------------------------------------------
# LOCAL BACKEND (Ollama + Chroma + optional reranker) — Phases 3 & 4
# ---------------------------------------------------------------------------

# Strict grounding prompt: answer only from context, exact refusal string, and
# keep the LaTeX instruction so the web UI's KaTeX rendering still works.
_STRICT_SYSTEM = (
    "You are BearLM, a study assistant for UC Berkeley CS/DS courses. "
    "Answer the question using ONLY the provided context. "
    "If the answer is not contained in the context, reply with exactly: "
    "\"I don't know based on the course materials.\" "
    "Cite sources inline as (course section). "
    "Format for a Markdown renderer. Use fenced code blocks ONLY for actual code, "
    "tagged with the language (```python, ```sql). Write math in LaTeX (inline "
    "$...$, display $$...$$). Present tabular or relational data as a Markdown "
    "table written directly in the response — never wrap tables or the whole "
    "answer in a code fence."
)


def _format_docs(docs):
    """Render retrieved LangChain Documents into the tagged context block."""
    out = []
    for d in docs:
        m = d.metadata or {}
        out.append(f"[{m.get('course', '?')} {m.get('section', '?')}] {d.page_content}")
    return "\n\n".join(out)


def _get_llm():
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=config.OLLAMA_CHAT_MODEL,
        temperature=0,
        num_ctx=config.OLLAMA_NUM_CTX,
        base_url=config.OLLAMA_BASE_URL,
    )


def _retrieve_local(question, k=None):
    """Retrieve (hybrid BM25+dense when enabled, optionally reranked) for a question."""
    from . import search

    k = k or config.RETRIEVAL_K
    if config.RERANK_ENABLED:
        from . import rerank

        cands = search.candidates(question, config.CHROMA_COLLECTION, config.RERANK_CANDIDATES)
        return rerank.rerank(question, cands, config.RERANK_TOP_N)
    return search.candidates(question, config.CHROMA_COLLECTION, k)


def build_chain(k=None):
    """The Phase 3 LCEL chain: retriever | prompt | llm | StrOutputParser().

    Returns a runnable that maps a question string -> answer string. Reranking,
    when enabled, is folded into the retrieval step. Citations are added by
    `answer()`, which needs the documents themselves.
    """
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
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    if rewrite is None:
        rewrite = config.QUERY_REWRITE_ENABLED
    rq = rewrite_query(question) if rewrite else question
    rewritten = rq if (rewrite and rq != question) else None
    docs = _retrieve_local(rq, k)
    if not docs:
        return Answer("I don't know based on the course materials.", [], rewritten_query=rewritten)

    prompt = ChatPromptTemplate.from_messages(
        [("system", _STRICT_SYSTEM), ("human", "Context:\n{context}\n\nQuestion: {question}")]
    )
    chain = prompt | _get_llm() | StrOutputParser()
    text = chain.invoke({"context": _format_docs(docs), "question": question})

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
    return Answer(text, citations, rewritten_query=rewritten)


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

import re

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


def _top_docs(question, collection_name, gated=True, allowed_courses=None):
    """Vector-retrieve a pool from one collection and rerank it.

    gated=True keeps only chunks clearing config.RERANK_MIN_SCORE (the confidence
    gate → empty list means "not covered", so callers refuse or fall back).
    gated=False skips the threshold and just returns the reranked top chunks (for
    meta/summary questions, where topical scores are uninformative).
    """
    from . import rerank, search

    cands = search.candidates(question, collection_name, config.RERANK_CANDIDATES, allowed_courses)
    if not cands:
        return []
    scored = rerank.rerank_scored(question, cands)
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
                    allow_fallback=None, history=None, rewrite=None, corpus_courses=None):
    """Grounded answer with a confidence gate, project-first fallback, and
    optional multi-turn conversation history.

    - Try the primary collection (a project or an upload). If confident chunks
      exist, answer from them (source = primary_label).
    - Else, if fallback is allowed, try the main course corpus.
    - Else refuse with config.GROUNDED_REFUSAL rather than guess.
    `history` is a list of prior {role, content} turns for conversational memory.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.output_parsers import StrOutputParser

    if allow_fallback is None:
        allow_fallback = config.PROJECT_FALLBACK_TO_CORPUS

    from . import local_store

    if rewrite is None:
        rewrite = config.QUERY_REWRITE_ENABLED
    if rewrite:
        rq = rewrite_query(question, history)
        rewritten = rq if rq != question else None
    else:
        rq = _retrieval_query(question, history)
        rewritten = None
    docs, used = [], "none"
    meta = _is_meta_question(question)

    if primary_collection:
        if meta and local_store.collection_count(primary_collection) > 0:
            # "Summarize / what's in this doc" — answer from the doc's top chunks
            # without the confidence gate (which would wrongly refuse these).
            docs = _top_docs(rq, primary_collection, gated=False)
            used = primary_label if docs else "none"
        else:
            docs = _top_docs(rq, primary_collection, gated=True)
            if docs:
                used = primary_label

    # Fall back to the course corpus for uncovered questions — but NOT for a meta
    # question about "this" doc, and NOT if the project disabled corpus fallback
    # (corpus_courses == [] means "project docs only"). corpus_courses may also
    # restrict fallback to a subset of subjects.
    corpus_disabled = corpus_courses is not None and len(corpus_courses) == 0
    if (not docs and (allow_fallback or not primary_collection)
            and not (meta and primary_collection) and not corpus_disabled):
        docs = _top_docs(rq, config.CHROMA_COLLECTION, gated=True, allowed_courses=corpus_courses)
        if docs:
            used = "corpus"

    if not docs:
        return Answer(config.GROUNDED_REFUSAL, [], source="none", rewritten_query=rewritten)

    system = (
        "You are BearLM, a grounded study assistant having a conversation. Answer "
        "the latest question using ONLY the provided context (plus the conversation "
        "so far for reference). If the answer is not contained in the context, reply "
        f"with exactly: \"{config.GROUNDED_REFUSAL}\" Cite sources inline as "
        "(course section). Format for a Markdown renderer: fenced code blocks ONLY "
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

    text = (_get_llm() | StrOutputParser()).invoke(msgs)
    return Answer(text, _dedupe_citations(docs), source=used, rewritten_query=rewritten)


def answer(question, k=None, rewrite=None):  # (str, int|None, bool|None) -> Answer
    """Backend-dispatching entry point the UI/eval call."""
    if config.BACKEND == "local":
        return _answer_local(question, k, rewrite)
    return _answer_openai(question, k)


def _answer_openai(question, k=None):  # (str, int|None) -> Answer
    """The public entry point the UI will call: question in, cited Answer out.

    Recipe:
      1. chunks = retrieve(question, k)
      2. if not chunks: return Answer("I couldn't find anything relevant.", [])
      3. messages = build_prompt(question, chunks)
      4. resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
         text = resp.choices[0].message.content
      5. Build citations from the chunks you used (dedupe course+section), e.g.:
           seen, citations = set(), []
           for c in chunks:
               key = (c["course"], c["section"])
               if key not in seen:
                   seen.add(key)
                   citations.append({"course": c["course"], "section": c["section"]})
      6. return Answer(text, citations)

    Tip: create the OpenAI client once (reuse `_client`) rather than per call.
    """
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
    return Answer(text, citations)


if __name__ == "__main__":
    # TODO(you): smoke test once retrieve()/build_prompt()/answer() are written.
    #   q = "What is the difference between a list and a tuple in Python?"
    #   result = answer(q)
    #   print(result.text)
    #   print("\nSources:")
    #   for c in result.citations:
    #       print(f"  - {c['course']} {c['section']}")
    q = "What is the difference between a list and a tuple in Python?"
    result = answer(q)
    print(result.text)
    print("\nSources:")
    for c in result.citations:
        print(f"  - {c['course']} {c['section']}")
