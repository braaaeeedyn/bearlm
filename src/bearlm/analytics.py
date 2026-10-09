"""Local usage analytics (SQLite).

Logs each corpus question, aggregates it for the Analytics page, surfaces the
**most-asked area** (the course students query most — i.e. where they need the
most help), and generates **grounded practice questions** from that area's
material.
"""

from __future__ import annotations

import datetime
import json
import random
import re
import sqlite3
from collections import Counter

from . import config


def _conn() -> sqlite3.Connection:
    config.ANALYTICS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.ANALYTICS_DB))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS queries(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            question TEXT NOT NULL,
            source TEXT,
            courses TEXT,   -- JSON list of cited courses
            sections TEXT   -- JSON list of "course|section"
        )"""
    )
    return conn


def log_query(question: str, source: str | None, citations: list[dict]) -> None:
    """Record one question. Best-effort — never breaks a response."""
    courses = sorted({c.get("course") for c in citations if c.get("course")})
    sections = sorted(
        {f"{c.get('course')}|{c.get('section')}" for c in citations
         if c.get("course") and c.get("section")}
    )
    try:
        conn = _conn()
        conn.execute(
            "INSERT INTO queries(ts, question, source, courses, sections) VALUES(?,?,?,?,?)",
            (datetime.datetime.now(datetime.timezone.utc).isoformat(), question,
             source or "", json.dumps(courses), json.dumps(sections)),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _rows():
    try:
        conn = _conn()
        rows = conn.execute(
            "SELECT ts, question, source, courses, sections FROM queries"
        ).fetchall()
        conn.close()
        return rows
    except Exception:
        return []


def stats() -> dict:
    """Aggregate the query log for the Analytics page."""
    rows = _rows()
    total = len(rows)
    courses_c, units_c, days_c = Counter(), Counter(), Counter()
    uncited = 0
    unanswered = []
    for ts, q, _source, courses_json, sections_json in rows:
        courses = json.loads(courses_json or "[]")
        sections = json.loads(sections_json or "[]")
        if not courses:
            uncited += 1
            unanswered.append((ts, q))
        courses_c.update(courses)
        units_c.update(sections)
        day = (ts or "")[:10]
        if day:
            days_c[day] += 1

    by_course = [{"course": c, "count": n} for c, n in courses_c.most_common()]
    by_unit = [
        {"course": s.split("|", 1)[0], "section": s.split("|", 1)[1], "count": n}
        for s, n in units_c.most_common(8) if "|" in s
    ]
    per_day = [{"date": d, "count": n} for d, n in sorted(days_c.items())]
    recent_unanswered = [q for _, q in sorted(unanswered, reverse=True)[:8]]

    top_area = None
    if by_course:
        answered = total - uncited
        top = by_course[0]
        top_units = [u for u in by_unit if u["course"] == top["course"]][:3]
        top_area = {
            "course": top["course"],
            "count": top["count"],
            "share": round(100 * top["count"] / answered) if answered else 0,
            "top_units": top_units,
        }

    return {
        "total": total, "uncited": uncited, "courses_covered": len(by_course),
        "by_course": by_course, "by_unit": by_unit, "per_day": per_day,
        "recent_unanswered": recent_unanswered, "top_area": top_area,
    }


def top_course() -> str | None:
    s = stats()
    return s["top_area"]["course"] if s.get("top_area") else None


def _parse_qa(text: str, n: int) -> list[dict]:
    """Parse 'Q1: ... / A1: ...' style output into {question, answer} dicts."""
    items, cur = [], None
    for raw in (text or "").splitlines():
        line = raw.strip()
        mq = re.match(r"^Q\d*\s*[\).:\-]\s*(.*)", line, re.I)
        ma = re.match(r"^A\d*\s*[\).:\-]\s*(.*)", line, re.I)
        if mq:
            if cur:
                items.append(cur)
            cur = {"question": mq.group(1).strip(), "answer": ""}
        elif ma and cur is not None:
            cur["answer"] = ma.group(1).strip()
        elif cur is not None and line:
            cur["answer"] = (cur["answer"] + " " + line).strip()
    if cur:
        items.append(cur)
    return [it for it in items if it["question"]][:n]


def generate_practice(course: str | None = None, n: int = 5) -> dict:
    """Generate n grounded practice questions from a course's material.

    Defaults to the most-asked course (the area needing the most help).
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import StrOutputParser

    from . import local_store, rag

    course = course or top_course()
    if not course:
        return {"area": None, "questions": [], "message": "No questions logged yet."}

    col = local_store.get_vectorstore()._collection
    got = col.get(where={"course": course}, limit=300, include=["documents"])
    docs = [d for d in (got.get("documents") or []) if d and d.strip()]
    if not docs:
        return {"area": course, "questions": [], "message": "No material found for this area."}

    def _clean(t: str) -> str:
        # Strip PDF ligature artifacts like "(cid:27)" that some math PDFs extract.
        return re.sub(r"[ \t]{2,}", " ", re.sub(r"\(cid:\d+\)", "", t)).strip()

    # Prefer the least-noisy excerpts (fewest extraction artifacts) for better
    # questions: rank originals by artifact density, take a pool, then clean.
    substantial = [d for d in docs if len(d) > 120] or docs
    substantial.sort(key=lambda d: d.count("(cid:") / max(len(d), 1))
    pool = substantial[: max(20, len(substantial) // 3)]
    sample = random.sample(pool, min(len(pool), 10))
    context = "\n\n---\n\n".join(_clean(d)[:800] for d in sample)
    system = (
        f"You are creating exam-style practice questions for a student studying "
        f"the course '{course}'. Using ONLY the excerpts provided, write exactly "
        f"{n} practice questions that test the key concepts, each with a concise, "
        "correct answer grounded in the material. Do not invent facts beyond the "
        "excerpts. Format EXACTLY as:\nQ1: <question>\nA1: <answer>\nQ2: ...\n"
        "Output only the questions and answers, nothing else."
    )
    try:
        text = (rag._get_llm() | StrOutputParser()).invoke(
            [SystemMessage(content=system),
             HumanMessage(content=f"Excerpts:\n{context}")]
        )
        questions = _parse_qa(text, n)
    except Exception as e:
        return {"area": course, "questions": [], "message": f"Generation failed: {e}"}

    return {"area": course, "questions": questions}
