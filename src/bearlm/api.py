"""BearLM HTTP API + static frontend host.

A thin FastAPI layer in front of the existing RAG engine (`rag.answer`). It
serves the hand-built frontend in `web/` and exposes one JSON endpoint the page
calls to ask a question. Every question is logged to `bearlm.query_logs` (same
data the Stage 7 analytics will read).

Run it:
    uv run uvicorn bearlm.api:app --reload
then open http://localhost:8000
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, rag

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(title="BearLM", docs_url="/api/docs")


class AskRequest(BaseModel):
    question: str


class Citation(BaseModel):
    course: str
    section: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]


def _log_query(question: str, citations: list[dict]) -> None:
    """Best-effort log of one question; never let logging break a response."""
    try:
        coll = db.get_query_logs_collection()
        coll.insert_one(
            {
                "question": question,
                "timestamp": datetime.now(timezone.utc),
                "courses": sorted({c["course"] for c in citations}),
            }
        )
    except Exception:
        pass


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    question = req.question.strip()
    if not question:
        return AskResponse(answer="Please enter a question.", citations=[])
    result = rag.answer(question)
    _log_query(question, result.citations)
    return AskResponse(
        answer=result.text,
        citations=[Citation(**c) for c in result.citations],
    )


@app.get("/api/courses")
def courses() -> dict:
    """Distinct course labels actually present in the corpus (for the footer)."""
    try:
        return {"courses": sorted(db.get_chunks_collection().distinct("course"))}
    except Exception:
        return {"courses": []}


@app.get("/api/stats")
def stats() -> dict:
    """Aggregate query_logs for the analytics view."""
    logs = db.get_query_logs_collection()
    total = logs.count_documents({})
    uncited = logs.count_documents({"courses": {"$size": 0}})

    by_course = [
        {"course": row["_id"], "count": row["count"]}
        for row in logs.aggregate(
            [
                {"$unwind": "$courses"},
                {"$group": {"_id": "$courses", "count": {"$sum": 1}}},
                {"$sort": {"count": -1, "_id": 1}},
            ]
        )
    ]
    per_day = [
        {"date": row["_id"], "count": row["count"]}
        for row in logs.aggregate(
            [
                {
                    "$group": {
                        "_id": {
                            "$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}
                        },
                        "count": {"$sum": 1},
                    }
                },
                {"$sort": {"_id": 1}},
            ]
        )
    ]
    recent_unanswered = [
        row["question"]
        for row in logs.find(
            {"courses": {"$size": 0}}, {"_id": 0, "question": 1}
        )
        .sort("timestamp", -1)
        .limit(8)
    ]

    return {
        "total": total,
        "uncited": uncited,
        "courses_covered": len(by_course),
        "by_course": by_course,
        "per_day": per_day,
        "recent_unanswered": recent_unanswered,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/analytics")
def analytics() -> FileResponse:
    return FileResponse(WEB_DIR / "analytics.html")


# All other static assets (styles.css, app.js, favicon, …).
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
