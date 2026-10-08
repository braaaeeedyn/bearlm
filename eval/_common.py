"""Shared helpers for the eval scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent


def load_questions(path: str | Path | None = None) -> list[dict]:
    """Load eval/questions.jsonl, skipping blank lines and '#' comments."""
    path = Path(path) if path else EVAL_DIR / "questions.jsonl"
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            rows.append(json.loads(s))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{i}: invalid JSON: {e}") from e
    return rows


def doc_meta(doc) -> dict:
    """Metadata dict from a LangChain Document (or passthrough dict)."""
    if hasattr(doc, "metadata"):
        return doc.metadata or {}
    return doc if isinstance(doc, dict) else {}


def is_gold_hit(doc, q: dict) -> bool:
    """True if a retrieved doc matches the question's gold reference.

    Prefers gold_chunk_ids (id-level) when provided; otherwise matches on
    (gold_course, gold_section).
    """
    m = doc_meta(doc)
    gold_ids = q.get("gold_chunk_ids") or []
    if gold_ids:
        return m.get("chunk_id") in set(gold_ids)
    return (
        m.get("course") == q.get("gold_course")
        and str(m.get("section")) == str(q.get("gold_section"))
    )


def is_relevant(doc, q: dict) -> bool:
    """Page-level relevance (used by precision@k): a gold hit, or the doc's
    (course, section) equals the gold (course, section)."""
    if is_gold_hit(doc, q):
        return True
    m = doc_meta(doc)
    return (
        m.get("course") == q.get("gold_course")
        and str(m.get("section")) == str(q.get("gold_section"))
    )


def recall_at_k(docs, q: dict, k: int) -> float:
    """Id-level recall@k: fraction of gold_chunk_ids in the top-k chunk_ids.
    Without gold ids: 1.0 if any top-k doc is a gold hit (course+section), else 0.0."""
    top = docs[:k]
    gold_ids = q.get("gold_chunk_ids") or []
    if gold_ids:
        got = {doc_meta(d).get("chunk_id") for d in top}
        return sum(1 for g in set(gold_ids) if g in got) / len(set(gold_ids))
    return 1.0 if any(is_gold_hit(d, q) for d in top) else 0.0


def precision_at_k(docs, q: dict, k: int) -> float:
    """Page-level precision@k: relevant docs (see is_relevant) in the top k, over k.
    Id-level precision would be capped at 1/k with one gold chunk per question."""
    return sum(1 for d in docs[:k] if is_relevant(d, q)) / k


def doc_text(doc) -> str:
    if hasattr(doc, "page_content"):
        return doc.page_content or ""
    return doc.get("text", "") if isinstance(doc, dict) else ""


def is_refusal(text: str) -> bool:
    return "I don't know based on the" in (text or "")


_COURSE_RE = re.compile(r"[A-Za-z]+\d+[A-Za-z]*")


def _norm_ref(course, section) -> tuple[str, str]:
    return str(course).lower(), re.sub(r"\s+", "", str(section)).lower()


def cited_refs(text: str, known_courses=()) -> list[tuple[str, str]]:
    """Inline citations like (cs61a p.111; cs70 p. 4) as normalized (course, section).

    ponytail: heuristic parse of every (...) group; a 2-word parenthetical whose first
    word looks like a course code (e.g. "(cs61a lecture)") can miscount. Upgrade to
    structured citations if this matters.
    """
    known = {c.lower() for c in known_courses}
    refs = []
    for group in re.findall(r"\(([^()]*)\)", text or ""):
        for part in re.split(r"[;,]", group):
            # "<course> <section>"; a section may carry one space after a dot ("p. 404")
            m = re.fullmatch(r"(\S+)\s+(\S+\.\s\S+|\S+)", part.strip())
            if not m:
                continue
            course, section = m.groups()
            if _COURSE_RE.fullmatch(course) or course.lower() in known:
                refs.append(_norm_ref(course, section))
    return refs


def citation_correctness(text: str, docs) -> float | None:
    """Share of inline citations that match a retrieved doc's (course, section).
    None for refusals; 0.0 for an answer with no inline citations."""
    if is_refusal(text):
        return None
    metas = [doc_meta(d) for d in docs]
    refs = cited_refs(text, {m.get("course") for m in metas if m.get("course")})
    if not refs:
        return 0.0
    have = {_norm_ref(m.get("course"), m.get("section")) for m in metas}
    return sum(1 for r in refs if r in have) / len(refs)
