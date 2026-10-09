"""End-to-end answer eval: run a backend over the questions, score and log answers.

For each question it records: backend, question, answer, citations, latency, and
automatic answer metrics computed on the exact context the answer was built from:
  - refused               : 1 if the answer is the "I don't know based on the ..." refusal
  - citation_correctness  : share of inline (course section) citations that match a
                            retrieved chunk (blank for refusals, 0 if uncited)
  - faithfulness, completeness : 0..1 from the LOCAL llama3.1:8b judge (judge_answer).
                            Noisy LLM estimates, not ground truth. Skipped for
                            refusals and with --no-judge.
  - prompt_tokens, output_tokens : tiktoken counts (report.py prices OpenAI from these)
  - rewrite_s, search_s, rerank_s, generate_s : per-stage seconds from Answer.timings
                            (blank when missing, e.g. the openai path, or on error)
The grade/notes columns stay blank for optional hand grading
(grade = correct | partial | wrong | hallucinated).

Run one backend (reads BACKEND from the env):
    BACKEND=local  uv run python eval/answer_eval.py
    BACKEND=openai uv run python eval/answer_eval.py

Or run BOTH on the same questions in one go (spawns a subprocess per backend so
each imports cleanly with its own BACKEND):
    uv run python eval/answer_eval.py --both

Output CSV: eval/out/answers.csv  (override with --out). Same file is appended,
so delete it for a clean run (an existing file with a different header is refused,
not appended to). report.py reads latency, tokens and metrics from this file.

Does NOT invent numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

STAGE_FIELDS = ("rewrite_s", "search_s", "rerank_s", "generate_s")  # from Answer.timings
FIELDS = [
    "backend", "question", "gold_course", "gold_section",
    "answer", "citations", "latency_s",
    "refused", "citation_correctness", "faithfulness", "completeness",
    "prompt_tokens", "output_tokens",
    *STAGE_FIELDS,
    "grade",  # <- fill by hand: correct | partial | wrong | hallucinated
    "notes",
]

_JUDGE_PROMPT = """You grade an answer from a course-notes assistant. Reply with ONLY JSON:
{{"faithfulness": <0..1>, "completeness": <0..1>}}
faithfulness = share of the answer's claims that are supported by the CONTEXT.
completeness = how fully the answer covers what the QUESTION needs, judged against the {ref_name}.

CONTEXT:
{contexts}

{ref_block}QUESTION: {question}

ANSWER: {answer}
"""


def parse_judge(text: str) -> dict:
    """First {...} in the judge output -> faithfulness/completeness clamped to [0,1];
    None for missing or invalid values. Never raises."""
    out = {"faithfulness": None, "completeness": None}
    m = re.search(r"\{.*?\}", text or "", re.DOTALL)
    try:
        data = json.loads(m.group(0)) if m else {}
    except ValueError:
        data = {}
    if not isinstance(data, dict):
        return out
    for key in out:
        try:
            out[key] = min(1.0, max(0.0, float(data.get(key))))
        except (TypeError, ValueError):
            pass
    return out


def judge_answer(question: str, answer: str, contexts: list[str], reference: str | None = None) -> dict:
    """One local-LLM call scoring faithfulness + completeness (noisy estimate)."""
    from langchain_core.output_parsers import StrOutputParser

    from bearlm import rag

    prompt = _JUDGE_PROMPT.format(
        contexts="\n\n".join(contexts),
        ref_block=f"REFERENCE:\n{reference}\n\n" if reference else "",
        ref_name="REFERENCE" if reference else "CONTEXT",
        question=question,
        answer=answer,
    )
    try:
        return parse_judge((rag._get_llm() | StrOutputParser()).invoke(prompt))
    except Exception:
        return parse_judge("")


def _gold_text(q: dict) -> str | None:
    """Gold chunk text from Chroma (Chroma ids == chunk_id), best-effort."""
    try:
        from bearlm import local_store

        got = local_store.get_vectorstore()._collection.get(ids=q["gold_chunk_ids"])
        return "\n\n".join(got["documents"]) or None
    except Exception:
        return None


def _fmt(x) -> str:
    return "" if x is None else f"{x:.3f}"


def run_one_backend(out_path: Path, questions_path: str | None = None, limit: int = 0,
                    judge: bool = True) -> None:
    import tiktoken

    from bearlm import config, rag  # imported here so BACKEND is already set

    enc = tiktoken.get_encoding(config.TOKENIZER_ENCODING)
    questions = _common.load_questions(questions_path)
    if limit:
        questions = questions[:limit]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_path.exists()
    if not new_file:
        with out_path.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != FIELDS:
            sys.exit(f"{out_path} has a different header (older format). "
                     "Delete it or pass --out <new file>.")
    with out_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for q in questions:
            t0 = time.perf_counter()
            try:
                res = rag.answer(q["question"])
                answer_text, cites, docs, timings = res.text, res.citations, res.docs, res.timings
            except Exception as e:  # log the failure rather than aborting the run
                answer_text, cites, docs, timings = f"[ERROR] {type(e).__name__}: {e}", [], [], {}
            latency = time.perf_counter() - t0
            contexts = [_common.doc_text(d) for d in docs]
            refused = _common.is_refusal(answer_text)
            scores = {"faithfulness": None, "completeness": None}
            if judge and not refused:
                ref = _gold_text(q) if q.get("gold_chunk_ids") else None
                scores = judge_answer(q["question"], answer_text, contexts, ref)
            w.writerow({
                "backend": config.BACKEND,
                "question": q["question"],
                "gold_course": q.get("gold_course", ""),
                "gold_section": q.get("gold_section", ""),
                "answer": answer_text,
                "citations": "; ".join(f"{c['course']} {c['section']}" for c in cites),
                "latency_s": f"{latency:.3f}",
                "refused": int(refused),
                "citation_correctness": _fmt(_common.citation_correctness(answer_text, docs)),
                "faithfulness": _fmt(scores["faithfulness"]),
                "completeness": _fmt(scores["completeness"]),
                "prompt_tokens": len(enc.encode(
                    rag._STRICT_SYSTEM + "\n\n".join(contexts) + q["question"])),
                "output_tokens": len(enc.encode(answer_text)),
                **{k: _fmt(timings.get(k)) for k in STAGE_FIELDS},
                "grade": "",
                "notes": "",
            })
            print(f"[{config.BACKEND}] {latency:6.2f}s  {q['question'][:60]}")
    print(f"Appended {config.BACKEND} rows to {out_path}")


def run_both(out_path: Path, extra: list[str]) -> None:
    for backend in ("local", "openai"):
        print(f"\n=== spawning backend={backend} ===")
        env = {**os.environ, "BACKEND": backend}
        subprocess.run(
            [sys.executable, __file__, "--out", str(out_path), *extra],
            env=env, check=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_common.EVAL_DIR / "out" / "answers.csv"))
    ap.add_argument("--both", action="store_true", help="run local and openai backends")
    default_q = _common.EVAL_DIR / "questions_generated.jsonl"
    ap.add_argument("--questions", default=str(default_q if default_q.exists()
                                               else _common.EVAL_DIR / "questions.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the local LLM faithfulness/completeness judge")
    args = ap.parse_args()
    out_path = Path(args.out)
    if args.both:
        extra = ["--questions", args.questions] + (["--limit", str(args.limit)] if args.limit else [])
        extra += ["--no-judge"] if args.no_judge else []
        run_both(out_path, extra)
    else:
        run_one_backend(out_path, args.questions, args.limit, judge=not args.no_judge)


if __name__ == "__main__":
    main()
