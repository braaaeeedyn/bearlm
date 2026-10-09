"""Retrieval + RAG core: retrieve and generate cited answers.

Also holds query rewriting, the prompt optimizer, and the grounded answer path
(confidence gate + project-first fallback) used by projects and uploads.
"""

import re
import time
from dataclasses import dataclass, field

from . import config


@dataclass
class Answer:
    text: str
    citations: list[dict]
    source: str = "corpus"  # "project" | "upload" | "corpus" | "none"
    rewritten_query: str | None = None
    docs: list = field(default_factory=list, repr=False)  # exact context used (for eval)
    timings: dict = field(default_factory=dict)  # per-stage seconds


def _tick(timings, key, t0):
    """Add the seconds since t0 to timings[key] (no-op when timings is None)."""
    if timings is not None:
        timings[key] = timings.get(key, 0.0) + time.perf_counter() - t0


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


# The same groups as _CITE_GROUP_RE, plus the spaces/tabs before them (removed along with a dropped group).
_CITE_GROUP_WS_RE = re.compile(r"([ \t]*)(?:" + _CITE_GROUP_RE.pattern + ")")
_CITE_COURSE_RE = re.compile(r"(?:cs|data)\d+[A-Za-z]*", re.IGNORECASE)


def _drop_outside_citations(text, docs):
    """Remove cite parts that name no context chunk. A group is rewritten only when every
    part is cite-like ("<course> <section with a digit>"); otherwise it stays byte-identical.
    Only ever removes refs, never adds one, and never touches a markdown link."""
    # ponytail: course = cs/data code (corpus prefixes) or a context course, so math like (R2 = 0.87)
    # stays. Limits: a fabricated cite to another prefix (stat134) or a non-digit course (genML, a
    # project slug) outside the context is kept; a prose group like (cs61b has 3 projects) whose
    # course is outside the context is still deleted. Upgrade: structured citations or a corpus course list.
    refs = {}
    for d in docs:
        m = d.metadata or {}
        if m.get("course") and m.get("section"):
            sec = str(m["section"])
            refs[(str(m["course"]).lower(), re.sub(r"\s+", "", sec).lower())] = f"{m['course']} {sec}"
    courses = {c for c, _ in refs}
    if not text:
        return text

    def repl(match):
        body = _CITE_LABEL_RE.sub("", match.group(2) if match.group(2) is not None else match.group(3))
        keys = []
        for part in re.split(r"[;,]", body):
            toks = part.split()
            if (len(toks) < 2 or not (_CITE_COURSE_RE.fullmatch(toks[0]) or toks[0].lower() in courses)
                    or not re.search(r"\d", "".join(toks[1:]))):
                return match.group(0)
            keys.append((toks[0].lower(), "".join(toks[1:]).lower()))
        kept = [refs[k] for k in keys if k in refs]
        if len(kept) == len(keys):
            return match.group(0)
        return match.group(1) + "(" + "; ".join(kept) + ")" if kept else ""

    return _CITE_GROUP_WS_RE.sub(repl, text)


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


class LLMUnavailable(RuntimeError):
    """Answer generation failed in Ollama; the message is safe to show the user."""


def ollama_error_message(exc):
    """User-facing message for an Ollama failure (generation, embedding or a stall)."""
    import httpx
    import ollama

    url = config.OLLAMA_BASE_URL
    if isinstance(exc, ConnectionError):
        return f"Can't reach Ollama at {url}. Start Ollama (`ollama serve`) and try again."
    if isinstance(exc, httpx.TimeoutException):
        return (f"Ollama at {url} did not answer within {config.OLLAMA_TIMEOUT_S:g} s. "
                "Try again, or see README \"Troubleshooting\".")
    if isinstance(exc, ollama.ResponseError) and 0 <= exc.status_code < 500:
        return (f"Ollama at {url} rejected the request ({exc.status_code}: {exc.error}). "
                f"Check that {config.OLLAMA_CHAT_MODEL} is pulled (`ollama pull {config.OLLAMA_CHAT_MODEL}`).")
    return (f"Ollama at {url} failed to generate an answer; it may be out of GPU memory. "
            "Try again, or see README \"Troubleshooting\".")


def _generate(runnable, inp):
    """invoke() an answer-generation runnable, mapping Ollama failures to LLMUnavailable."""
    import httpx
    import ollama

    for attempt in (1, 2):
        try:
            return runnable.invoke(inp)
        except ollama.ResponseError as e:
            # One retry for 5xx / mid-stream errors: run 5 hit a transient CUDA OOM while llama-server reloaded.
            if attempt == 1 and (e.status_code >= 500 or e.status_code == -1):
                continue
            raise LLMUnavailable(ollama_error_message(e)) from e
        except (ConnectionError, httpx.TimeoutException) as e:
            raise LLMUnavailable(ollama_error_message(e)) from e


TRUNCATED_NOTE = "\n\n_(Answer cut off at the length limit.)_"


def _finish(msg, docs):
    """Normalize cites in a generated AIMessage; flag output cut off by num_predict."""
    text = msg.content
    if "I don't know based on the" not in text:
        text = _normalize_citations(text, docs)
        text = _drop_outside_citations(text, docs)
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
    msg = _generate(chain, {"context": _format_docs(docs), "question": question})
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
                    allow_fallback=None, history=None, rewrite=None, corpus_courses=None, memory=None):
    """Grounded answer with a confidence gate, project-first fallback, and
    optional multi-turn conversation history.

    - Try the primary collection (a project or an upload). If confident chunks
      exist, answer from them (source = primary_label).
    - Else, if fallback is allowed, try the main course corpus.
    - Else refuse with config.GROUNDED_REFUSAL rather than guess.
    `history` is a list of prior {role, content} turns for conversational memory.
    `memory` is {collection: [chunk ids]} merged into the gated rerank pools.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    if allow_fallback is None:
        allow_fallback = config.PROJECT_FALLBACK_TO_CORPUS

    from . import local_store

    start = time.perf_counter()
    timings = {"rewrite_s": 0.0, "search_s": 0.0, "rerank_s": 0.0, "generate_s": 0.0}
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

    if primary_collection:
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
    if (not docs and (allow_fallback or not primary_collection)
            and not (meta and primary_collection) and not corpus_disabled):
        docs = _top_docs(rq, config.CHROMA_COLLECTION, gated=True, allowed_courses=corpus_courses,
                         extra_ids=(memory or {}).get(config.CHROMA_COLLECTION), timings=timings)
        if docs:
            used = "corpus"

    if not docs:
        _tick(timings, "total_s", start)
        return Answer(config.GROUNDED_REFUSAL, [], source="none", rewritten_query=rewritten,
                      timings=timings)

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
    msg = _generate(_get_llm(), msgs)
    _tick(timings, "generate_s", t0)
    text = _finish(msg, docs)
    _tick(timings, "total_s", start)
    return Answer(text, _dedupe_citations(docs), source=used, rewritten_query=rewritten, docs=docs,
                  timings=timings)


def answer(question, k=None, rewrite=None):
    """Corpus Q&A entry point the UI/eval calls."""
    return _answer_local(question, k, rewrite)


if __name__ == "__main__":
    q = "What is the difference between a list and a tuple in Python?"
    result = answer(q)
    print(result.text)
    print("\nSources:")
    for c in result.citations:
        print(f"  - {c['course']} {c['section']}")
