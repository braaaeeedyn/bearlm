"""Projects and ad-hoc uploads (local backend).

A **project** is a persistent folder of PDFs plus its own Chroma collection:
    data/projects/<slug>/
        project.json      # {name, slug, created, pdfs: [...], history: [...]}
        pdfs/<file>.pdf
Answers in a project are grounded project-first (see rag.answer_grounded), with
fallback to the main course corpus when the project doesn't cover the question.

An **upload** is an ephemeral single-PDF collection scoped to one session; it is
grounded strictly to that one PDF (no corpus fallback).

Everything here is local-only (Ollama embeddings + Chroma); nothing touches
OpenAI or Atlas.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import config, local_store


# --- naming ------------------------------------------------------------------

def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return s or "project"


def project_collection(slug: str) -> str:
    return f"{config.PROJECT_COLLECTION_PREFIX}{slug}"


def upload_collection(uid: str) -> str:
    return f"{config.UPLOAD_COLLECTION_PREFIX}{uid}"


def _project_dir(slug: str) -> Path:
    return config.PROJECTS_DIR / slug


def _meta_path(slug: str) -> Path:
    return _project_dir(slug) / "project.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- project CRUD ------------------------------------------------------------

def create_project(name: str) -> dict:
    """Create a project folder + metadata. Slugs must be unique."""
    slug = slugify(name)
    pdir = _project_dir(slug)
    if pdir.exists():
        raise ValueError(f"A project named '{slug}' already exists.")
    (pdir / "pdfs").mkdir(parents=True)
    meta = {"name": name, "slug": slug, "created": _now(), "pdfs": [], "history": []}
    _meta_path(slug).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return public(meta)


def get_project(slug: str) -> dict | None:
    p = _meta_path(slug)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def list_projects() -> list[dict]:
    if not config.PROJECTS_DIR.exists():
        return []
    out = []
    for d in config.PROJECTS_DIR.iterdir():
        if d.is_dir() and (d / "project.json").exists():
            out.append(public(json.loads((d / "project.json").read_text(encoding="utf-8"))))
    out.sort(key=lambda p: p.get("created") or "", reverse=True)  # newest first
    return out


def delete_project(slug: str) -> None:
    """Remove the project's folder and its Chroma collection."""
    local_store.drop_collection(project_collection(slug))
    shutil.rmtree(_project_dir(slug), ignore_errors=True)


def _save(meta: dict) -> None:
    _meta_path(meta["slug"]).write_text(json.dumps(meta, indent=2), encoding="utf-8")


# --- documents ---------------------------------------------------------------

def add_pdf_to_project(slug: str, data: bytes, filename: str) -> dict:
    """Save an uploaded PDF into the project and embed it into its collection."""
    meta = get_project(slug)
    if meta is None:
        raise ValueError(f"No such project: {slug}")
    safe = Path(filename).name
    if not safe.lower().endswith(".pdf"):
        raise ValueError("Only .pdf files are supported.")
    dest = _project_dir(slug) / "pdfs" / safe
    dest.write_bytes(data)

    chunks = local_store.ingest_pdf_into_collection(
        dest, project_collection(slug), course_label=slug
    )
    entry = {"filename": safe, "chunks": chunks, "added": _now()}
    meta["pdfs"] = [p for p in meta.get("pdfs", []) if p["filename"] != safe] + [entry]
    _save(meta)
    return entry


# --- chats (multi-turn conversations within a project) -----------------------

def _chats(meta: dict) -> list[dict]:
    return meta.setdefault("chats", [])


def _find_chat(meta: dict, chat_id: str) -> dict | None:
    for c in meta.get("chats", []):
        if c["id"] == chat_id:
            return c
    return None


def _chat_summary(c: dict) -> dict:
    """Lightweight chat info for the sidebar list (no full messages)."""
    return {
        "id": c["id"],
        "title": c.get("title") or "New chat",
        "created": c.get("created"),
        "updated": c.get("updated", c.get("created")),
        "message_count": len(c.get("messages", [])),
    }


def create_chat(slug: str, title: str | None = None) -> dict:
    meta = get_project(slug)
    if meta is None:
        raise ValueError(f"No such project: {slug}")
    chat = {
        "id": uuid.uuid4().hex[:12],
        "title": title or "New chat",
        "created": _now(),
        "updated": _now(),
        "messages": [],
    }
    _chats(meta).insert(0, chat)  # newest first
    _save(meta)
    return chat


def list_chats(slug: str) -> list[dict]:
    meta = get_project(slug) or {}
    chats = sorted(meta.get("chats", []), key=lambda c: c.get("updated") or "", reverse=True)
    return [_chat_summary(c) for c in chats]


def get_chat(slug: str, chat_id: str) -> dict | None:
    meta = get_project(slug)
    return _find_chat(meta, chat_id) if meta else None


def delete_chat(slug: str, chat_id: str) -> None:
    meta = get_project(slug)
    if meta is None:
        return
    meta["chats"] = [c for c in meta.get("chats", []) if c["id"] != chat_id]
    _save(meta)


def clear_chats(slug: str) -> None:
    """Reset a project's conversation context: delete all chats, keep the PDFs.

    Long conversation histories can drift and induce hallucination; clearing them
    lets the user start fresh while retaining the grounded reference documents.
    """
    meta = get_project(slug)
    if meta is None:
        return
    meta["chats"] = []
    _save(meta)


def remove_pdf(slug: str, filename: str) -> dict:
    """Remove one PDF from a project: delete its chunks, file, and metadata entry."""
    meta = get_project(slug)
    if meta is None:
        raise ValueError(f"No such project: {slug}")
    safe = Path(filename).name
    local_store.delete_pdf_from_collection(project_collection(slug), safe)
    try:
        (_project_dir(slug) / "pdfs" / safe).unlink()
    except FileNotFoundError:
        pass
    meta["pdfs"] = [p for p in meta.get("pdfs", []) if p["filename"] != safe]
    _save(meta)
    return {"removed": safe}


def append_message(slug: str, chat_id: str, role: str, content: str,
                   citations: list[dict] | None = None, source: str | None = None) -> dict | None:
    """Append one message to a chat; title the chat from its first user message."""
    meta = get_project(slug)
    if meta is None:
        return None
    chat = _find_chat(meta, chat_id)
    if chat is None:
        return None
    msg = {"role": role, "content": content, "at": _now()}
    if citations is not None:
        msg["citations"] = citations
    if source:
        msg["source"] = source
    chat.setdefault("messages", []).append(msg)
    chat["updated"] = _now()
    if role == "user" and (not chat.get("title") or chat["title"] == "New chat"):
        chat["title"] = content.strip()[:48] or "New chat"
    _save(meta)
    return msg


def public(meta: dict) -> dict:
    """Metadata safe to return over the API (adds live chunk count + chat summaries)."""
    slug = meta["slug"]
    chats = sorted(meta.get("chats", []), key=lambda c: c.get("updated") or "", reverse=True)
    return {
        "name": meta.get("name", slug),
        "slug": slug,
        "created": meta.get("created"),
        "pdfs": meta.get("pdfs", []),
        "chats": [_chat_summary(c) for c in chats],
        "chunks": local_store.collection_count(project_collection(slug)),
    }


# --- ad-hoc uploads (ephemeral, single PDF) ----------------------------------

def create_upload(data: bytes, filename: str) -> dict:
    """Embed one PDF into a throwaway collection; return its id + label."""
    safe = Path(filename).name
    if not safe.lower().endswith(".pdf"):
        raise ValueError("Only .pdf files are supported.")
    uid = uuid.uuid4().hex[:12]
    updir = config.UPLOADS_DIR / uid
    updir.mkdir(parents=True, exist_ok=True)
    dest = updir / safe
    dest.write_bytes(data)

    label = Path(safe).stem
    chunks = local_store.ingest_pdf_into_collection(dest, upload_collection(uid), course_label=label)
    return {"upload_id": uid, "filename": safe, "label": label, "chunks": chunks}


def delete_upload(uid: str) -> None:
    local_store.drop_collection(upload_collection(uid))
    shutil.rmtree(config.UPLOADS_DIR / uid, ignore_errors=True)
