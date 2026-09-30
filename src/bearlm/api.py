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

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, db, rag

MAX_PDF_BYTES = 50 * 1024 * 1024  # reject uploads larger than 50 MB

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(title="BearLM", docs_url="/api/docs")


@app.middleware("http")
async def _no_cache_static(request, call_next):
    """Make the browser always revalidate static assets, so edits show up without
    a hard reload (avoids stale cached JS/CSS)."""
    resp = await call_next(request)
    if request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.on_event("startup")
def _prewarm_bm25() -> None:
    """Build the corpus BM25 index in the background so the first hybrid query
    isn't slow (the index over ~10k chunks takes a while to build once)."""
    if config.BACKEND != "local" or not config.HYBRID_SEARCH_ENABLED:
        return
    import threading

    from . import search

    threading.Thread(
        target=lambda: search._get_bm25(config.CHROMA_COLLECTION),
        daemon=True,
    ).start()


class AskRequest(BaseModel):
    question: str
    rewrite: bool | None = None  # None = use config default; True/False overrides per request


class OptimizeRequest(BaseModel):
    prompt: str


class OptimizeResponse(BaseModel):
    optimized: str


class Citation(BaseModel):
    course: str
    section: str
    source: str | None = None  # source filename (e.g. "61a_txtbk.pdf"), for deep-linking


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    source: str = "corpus"  # "project" | "upload" | "corpus" | "none"
    rewritten_query: str | None = None  # LLM-rewritten retrieval query, if used


class CreateProjectRequest(BaseModel):
    name: str


def _require_local() -> None:
    if config.BACKEND != "local":
        raise HTTPException(status_code=400, detail="Projects/uploads require BACKEND=local.")


async def _read_pdf(file: UploadFile) -> bytes:
    """Read an uploaded file, enforcing PDF type and a size cap."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")
    data = await file.read()
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds the 50 MB limit.")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    return data


def _log_query(question: str, citations: list[dict], source: str = "corpus") -> None:
    """Best-effort log of one question; never let logging break a response.

    Local backend logs to SQLite (analytics.py); OpenAI backend to Atlas.
    """
    try:
        if config.BACKEND == "local":
            from . import analytics

            analytics.log_query(question, source, citations)
            return
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
    result = rag.answer(question, rewrite=req.rewrite)
    _log_query(question, result.citations, source=result.source)
    return AskResponse(
        answer=result.text,
        citations=[Citation(**c) for c in result.citations],
        source=result.source,
        rewritten_query=result.rewritten_query,
    )


@app.post("/api/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest) -> OptimizeResponse:
    """Suggest a prompt-engineered rewrite of the user's prompt (local LLM)."""
    _require_local()
    prompt = req.prompt.strip()
    if not prompt:
        return OptimizeResponse(optimized="")
    return OptimizeResponse(optimized=rag.optimize_prompt(prompt))


# ---------------------------------------------------------------------------
# Projects (persistent, grounded project-first) — local backend only
# ---------------------------------------------------------------------------

@app.get("/api/projects")
def list_projects() -> dict:
    _require_local()
    from . import projects

    return {"projects": projects.list_projects()}


@app.post("/api/projects")
def create_project(req: CreateProjectRequest) -> dict:
    _require_local()
    from . import projects

    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Project name is required.")
    try:
        return projects.create_project(name)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/api/projects/{slug}")
def get_project(slug: str) -> dict:
    _require_local()
    from . import projects

    meta = projects.get_project(slug)
    if meta is None:
        raise HTTPException(status_code=404, detail="No such project.")
    return projects.public(meta)


@app.delete("/api/projects/{slug}")
def delete_project(slug: str) -> dict:
    _require_local()
    from . import projects

    if projects.get_project(slug) is None:
        raise HTTPException(status_code=404, detail="No such project.")
    projects.delete_project(slug)
    return {"deleted": slug}


@app.post("/api/projects/{slug}/pdfs")
async def add_project_pdf(slug: str, file: UploadFile = File(...)) -> dict:
    _require_local()
    from . import projects

    if projects.get_project(slug) is None:
        raise HTTPException(status_code=404, detail="No such project.")
    data = await _read_pdf(file)
    return projects.add_pdf_to_project(slug, data, file.filename)


@app.delete("/api/projects/{slug}/pdfs/{filename}")
def remove_project_pdf(slug: str, filename: str) -> dict:
    _require_local()
    from . import projects

    if projects.get_project(slug) is None:
        raise HTTPException(status_code=404, detail="No such project.")
    return projects.remove_pdf(slug, filename)


class CorpusScopeRequest(BaseModel):
    courses: list[str] | None = None  # None = all; [] = none; [list] = subset


@app.post("/api/projects/{slug}/corpus")
def set_corpus_scope(slug: str, req: CorpusScopeRequest) -> dict:
    """Set which course-corpus subjects this project may fall back to."""
    _require_local()
    from . import projects

    if projects.get_project(slug) is None:
        raise HTTPException(status_code=404, detail="No such project.")
    return projects.set_corpus_scope(slug, req.courses)


# --- chats within a project (multi-turn) ------------------------------------

class CreateChatRequest(BaseModel):
    title: str | None = None


@app.post("/api/projects/{slug}/chats")
def create_chat(slug: str, req: CreateChatRequest | None = None) -> dict:
    _require_local()
    from . import projects

    if projects.get_project(slug) is None:
        raise HTTPException(status_code=404, detail="No such project.")
    return projects.create_chat(slug, req.title if req else None)


@app.get("/api/projects/{slug}/chats/{chat_id}")
def get_chat(slug: str, chat_id: str) -> dict:
    _require_local()
    from . import projects

    chat = projects.get_chat(slug, chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="No such chat.")
    return chat


@app.delete("/api/projects/{slug}/chats/{chat_id}")
def delete_chat(slug: str, chat_id: str) -> dict:
    _require_local()
    from . import projects

    projects.delete_chat(slug, chat_id)
    return {"deleted": chat_id}


@app.post("/api/projects/{slug}/reset")
def reset_project_context(slug: str) -> dict:
    """Clear all of a project's chats (conversation context) but keep its PDFs."""
    _require_local()
    from . import projects

    if projects.get_project(slug) is None:
        raise HTTPException(status_code=404, detail="No such project.")
    projects.clear_chats(slug)
    return {"reset": slug}


@app.post("/api/projects/{slug}/chats/{chat_id}/ask", response_model=AskResponse)
def ask_chat(slug: str, chat_id: str, req: AskRequest) -> AskResponse:
    _require_local()
    from . import projects

    chat = projects.get_chat(slug, chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="No such chat.")
    question = req.question.strip()
    if not question:
        return AskResponse(answer="Please enter a question.", citations=[], source="none")

    project = projects.get_project(slug)
    history = chat.get("messages", [])
    result = rag.answer_grounded(
        question,
        primary_collection=projects.project_collection(slug),
        primary_label="project",
        history=history,
        rewrite=req.rewrite,
        corpus_courses=(project or {}).get("corpus_courses"),
    )
    # Persist both turns so the conversation survives reloads.
    projects.append_message(slug, chat_id, "user", question)
    projects.append_message(slug, chat_id, "assistant", result.text,
                            citations=result.citations, source=result.source)
    return AskResponse(
        answer=result.text,
        citations=[Citation(**c) for c in result.citations],
        source=result.source,
        rewritten_query=result.rewritten_query,
    )


# ---------------------------------------------------------------------------
# Ad-hoc uploads (ephemeral, strictly grounded to the one PDF)
# ---------------------------------------------------------------------------

@app.post("/api/uploads")
async def create_upload(file: UploadFile = File(...)) -> dict:
    _require_local()
    from . import projects

    data = await _read_pdf(file)
    return projects.create_upload(data, file.filename)


@app.post("/api/uploads/{uid}/ask", response_model=AskResponse)
def ask_upload(uid: str, req: AskRequest) -> AskResponse:
    _require_local()
    from . import projects

    question = req.question.strip()
    if not question:
        return AskResponse(answer="Please enter a question.", citations=[], source="none")
    # Uploads are strictly grounded to their own PDF — no corpus fallback.
    result = rag.answer_grounded(
        question,
        primary_collection=projects.upload_collection(uid),
        primary_label="upload",
        allow_fallback=False,
        rewrite=req.rewrite,
    )
    return AskResponse(
        answer=result.text,
        citations=[Citation(**c) for c in result.citations],
        source=result.source,
        rewritten_query=result.rewritten_query,
    )


@app.delete("/api/uploads/{uid}")
def delete_upload(uid: str) -> dict:
    _require_local()
    from . import projects

    projects.delete_upload(uid)
    return {"deleted": uid}


@app.get("/api/courses")
def courses() -> dict:
    """Distinct course labels actually present in the corpus (for the footer)."""
    try:
        if config.BACKEND == "local":
            from . import local_store

            return {"courses": local_store.distinct_courses()}
        return {"courses": sorted(db.get_chunks_collection().distinct("course"))}
    except Exception:
        return {"courses": []}


class PracticeRequest(BaseModel):
    course: str | None = None
    n: int = 5


@app.post("/api/practice")
def practice(req: PracticeRequest) -> dict:
    """Generate grounded practice questions for the most-asked area (or a given course)."""
    _require_local()
    from . import analytics

    n = max(1, min(req.n, 10))
    return analytics.generate_practice(course=req.course, n=n)


@app.get("/api/stats")
def stats() -> dict:
    """Aggregate query logs for the analytics view (SQLite locally, else Atlas)."""
    if config.BACKEND == "local":
        from . import analytics

        return analytics.stats()

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


def _serve_within(base: Path, *parts: str) -> FileResponse:
    """Serve a file under `base`, refusing any path that escapes it (traversal).

    Served inline (no `filename=`) so PDFs open in-tab and honor `#page=N`.
    """
    base = base.resolve()
    target = base.joinpath(*parts).resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="Source not found")
    return FileResponse(target)


@app.get("/source/{course}/{filename}")
def source_file(course: str, filename: str) -> FileResponse:
    """Serve a raw course textbook PDF so citations can deep-link to it."""
    return _serve_within(config.RAW_DIR, course, filename)


@app.get("/projects/{slug}/pdfs/{filename}")
def project_pdf(slug: str, filename: str) -> FileResponse:
    """Serve a project's PDF for citation deep-links."""
    return _serve_within(config.PROJECTS_DIR, slug, "pdfs", filename)


def _render_pdf_thumb(path: Path, width: int = 300) -> bytes:
    """Render a PDF's first page to a PNG (for the grid view)."""
    import pymupdf

    doc = pymupdf.open(path)
    try:
        page = doc[0]
        zoom = max(width / max(page.rect.width, 1), 0.1)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        return pix.tobytes("png")
    finally:
        doc.close()


@app.get("/projects/{slug}/pdfs/{filename}/thumb")
def project_pdf_thumb(slug: str, filename: str) -> Response:
    """First-page PNG thumbnail of a project PDF (grid view)."""
    base = config.PROJECTS_DIR.resolve()
    target = (base / slug / "pdfs" / filename).resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    try:
        png = _render_pdf_thumb(target)
    except Exception:
        raise HTTPException(status_code=422, detail="Could not render thumbnail")
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "max-age=3600"})


@app.get("/uploads/{uid}/{filename}")
def upload_pdf(uid: str, filename: str) -> FileResponse:
    """Serve an ad-hoc upload's PDF for citation deep-links."""
    return _serve_within(config.UPLOADS_DIR, uid, filename)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/projects")
def projects_page() -> FileResponse:
    return FileResponse(WEB_DIR / "projects.html")


@app.get("/analytics")
def analytics() -> FileResponse:
    return FileResponse(WEB_DIR / "analytics.html")


# All other static assets (styles.css, app.js, favicon, …).
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
