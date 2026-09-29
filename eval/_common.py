"""Shared helpers for the eval scripts."""

from __future__ import annotations

import json
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
