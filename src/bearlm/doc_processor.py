"""Extract course files (PDF / Markdown / notebook) from data/raw into chunks."""

import json
from dataclasses import dataclass

import pdfplumber

from . import config, text_splitter


@dataclass
class Chunk:
    text: str
    course: str
    filename: str
    section: str


def extract_pdf(path):
    """One PDF -> list of (page_text, "p.N"). Per-page so citations can point at a page."""
    output = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                output.append((text, f"p.{i}"))
    return output


def extract_markdown(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [(text, path.stem)] if text.strip() else []


def extract_notebook(path):
    """One .ipynb -> its markdown+code cells as one blob (cell outputs are skipped)."""
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except (json.JSONDecodeError, ValueError):
        return []
    parts = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") not in ("markdown", "code"):
            continue
        src = cell.get("source", "")
        text = "".join(src) if isinstance(src, list) else src
        if text.strip():
            parts.append(text)
    blob = "\n\n".join(parts)
    return [(blob, path.stem)] if blob.strip() else []


def iter_course_dirs():
    for p in sorted(config.RAW_DIR.iterdir()):
        if p.is_dir():
            yield p


def build_chunks_for_course(course_dir, chunker=text_splitter.split_text):
    course = course_dir.name
    chunks = []
    for path in course_dir.rglob("*"):
        if path.suffix.lower() == ".pdf":
            pieces = extract_pdf(path)
        elif path.suffix.lower() in {".md", ".myst"}:
            pieces = extract_markdown(path)
        elif path.suffix.lower() == ".ipynb":
            pieces = extract_notebook(path)
        else:
            continue
        for raw_text, section in pieces:
            for piece in chunker(raw_text):
                chunks.append(Chunk(piece, course, path.name, section))
    return chunks
