"""Generate a retrieval eval set from real corpus chunks.

Each question is written *from* a specific chunk, so that chunk's
(course, section, chunk_id) is the verified gold — recall@k then measures whether
retrieval surfaces the exact source chunk. This is a synthetic-but-grounded
benchmark (the questions are LLM-written, the gold labels are exact).

Run:
    BACKEND=local uv run python eval/generate_questions.py --per-course 5 \
        --out eval/questions_generated.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bearlm import config, local_store  # noqa: E402


def _noise(text: str) -> float:
    dots = text.count(". . .") + text.count("....")
    cid = text.count("(cid:")
    return (dots * 5 + cid) / max(len(text), 1)


def _is_good(text: str) -> bool:
    if not text or len(text) < 320:
        return False
    letters = sum(c.isalpha() for c in text)
    if letters / len(text) < 0.55:      # skip tables/formulas/number-heavy chunks
        return False
    if _noise(text) > 0.02:             # skip TOC / heavy PDF artifacts
        return False
    return True


def _clean(text: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", re.sub(r"\(cid:\d+\)", "", text)).strip()


def _gen_question(llm, parser, course, text):
    from langchain_core.messages import HumanMessage, SystemMessage

    system = (
        "You write a single specific study question that the given passage fully "
        f"answers (course: {course}). The question must be answerable from the "
        "passage alone and mention enough specifics to be unambiguous. Output ONLY "
        "the question, one line, no preamble."
    )
    out = (llm | parser).invoke(
        [SystemMessage(content=system), HumanMessage(content=f"Passage:\n{_clean(text)[:1200]}")]
    )
    q = (out or "").strip().strip('"').splitlines()[0].strip()
    return q if q.endswith("?") and len(q) > 15 else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-course", type=int, default=5)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "questions_generated.jsonl"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if config.BACKEND != "local":
        sys.exit("Run with BACKEND=local.")
    random.seed(args.seed)

    from langchain_core.output_parsers import StrOutputParser

    from bearlm import rag

    llm, parser = rag._get_llm(), StrOutputParser()
    col = local_store.get_vectorstore()._collection
    courses = local_store.distinct_courses()
    print(f"courses: {courses}")

    rows, out_path = [], Path(args.out)
    for course in courses:
        got = col.get(where={"course": course}, include=["documents", "metadatas"])
        pool = [
            (i, d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])
            if _is_good(d or "")
        ]
        random.shuffle(pool)
        made = 0
        for cid, doc, meta in pool:
            if made >= args.per_course:
                break
            try:
                q = _gen_question(llm, parser, course, doc)
            except Exception:
                q = ""
            if not q:
                continue
            rows.append({
                "question": q,
                "gold_course": course,
                "gold_section": meta.get("section"),
                "gold_chunk_ids": [cid],
            })
            made += 1
            print(f"  [{course}] {q[:70]}")
        print(f"{course}: {made} questions ({len(pool)} good chunks available)")

    out_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"\nWrote {len(rows)} questions to {out_path}")


if __name__ == "__main__":
    main()
