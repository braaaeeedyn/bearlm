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
    "Write mathematical notation in LaTeX: inline math in $...$ and display "
    "equations in $$...$$."
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
    """Retrieve (and optionally rerank) Chroma documents for a question."""
    from . import local_store

    if config.RERANK_ENABLED:
        from . import rerank

        candidates = local_store.get_retriever(config.RERANK_CANDIDATES).invoke(question)
        return rerank.rerank(question, candidates, config.RERANK_TOP_N)
    return local_store.get_retriever(k or config.RETRIEVAL_K).invoke(question)


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


def _answer_local(question, k=None):  # (str, int|None) -> Answer
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    docs = _retrieve_local(question, k)
    if not docs:
        return Answer("I don't know based on the course materials.", [])

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
    return Answer(text, citations)


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

def _confident_docs(question, collection_name):
    """Vector-retrieve a pool from one collection, rerank, and keep only chunks
    whose cross-encoder relevance clears config.RERANK_MIN_SCORE.

    The reranker score is the confidence gate: metric-independent and calibrated,
    so a weak/irrelevant collection yields an empty list (→ refuse or fall back)
    instead of forcing a plausible-sounding but ungrounded answer.
    """
    from . import local_store, rerank

    vs = local_store.get_vectorstore(collection_name)
    candidates = vs.similarity_search(question, k=config.RERANK_CANDIDATES)
    if not candidates:
        return []
    scored = rerank.rerank_scored(question, candidates)
    strong = [d for d, s in scored if s >= config.RERANK_MIN_SCORE]
    return strong[: config.RERANK_TOP_N]


def answer_grounded(question, primary_collection=None, primary_label="project",
                    allow_fallback=None):
    """Grounded answer with a confidence gate and project-first fallback.

    - Try the primary collection (a project or an upload). If confident chunks
      exist, answer from them (source = primary_label).
    - Else, if fallback is allowed, try the main course corpus.
    - Else refuse with config.GROUNDED_REFUSAL rather than guess.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    if allow_fallback is None:
        allow_fallback = config.PROJECT_FALLBACK_TO_CORPUS

    docs, used = [], "none"
    if primary_collection:
        docs = _confident_docs(question, primary_collection)
        if docs:
            used = primary_label
    if not docs and (allow_fallback or not primary_collection):
        docs = _confident_docs(question, config.CHROMA_COLLECTION)
        if docs:
            used = "corpus"

    if not docs:
        return Answer(config.GROUNDED_REFUSAL, [], source="none")

    system = (
        "You are BearLM, a grounded study assistant. Answer the question using "
        "ONLY the provided context. If the answer is not fully contained in the "
        f"context, reply with exactly: \"{config.GROUNDED_REFUSAL}\" "
        "Cite sources inline as (course section). Write math in LaTeX: inline "
        "in $...$ and display in $$...$$."
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "Context:\n{context}\n\nQuestion: {question}")]
    )
    text = (prompt | _get_llm() | StrOutputParser()).invoke(
        {"context": _format_docs(docs), "question": question}
    )
    return Answer(text, _dedupe_citations(docs), source=used)


def answer(question, k=None):  # (str, int|None) -> Answer
    """Backend-dispatching entry point the UI/eval call."""
    if config.BACKEND == "local":
        return _answer_local(question, k)
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
