"""End-to-end answer eval: answer every question, score and log the answers.

For each question it records: question, answer, citations, latency, and
automatic answer metrics computed on the exact context the answer was built from:
  - refused               : 1 if the answer is the "I don't know based on the ..." refusal
  - citation_correctness  : share of inline (course section) citations that match a
                            retrieved chunk (blank for refusals and [ERROR] rows, 0 if uncited)
  - faithfulness, completeness : 0..1 from the LOCAL llama3.1:8b judge (judge_answer).
                            Noisy LLM estimates, not ground truth. Skipped for
                            refusals, [ERROR] rows and with --no-judge.
  - prompt_tokens, output_tokens : tiktoken counts
  - context_ids           : ;-joined chunk_id of each context doc (provenance)
  - rewrite_s, search_s, rerank_s, generate_s : per-stage seconds from Answer.timings
                            (blank on error)
Each run also (over)writes <out>.meta.json: Chroma dir + count, questions sha256,
config, judge flag, Ollama /api/ps VRAM split and GPU/torch state (best-effort).
The grade/notes columns stay blank for optional hand grading
(grade = correct | partial | wrong | hallucinated).

Run:
    uv run python eval/answer_eval.py [--limit N] [--no-judge]

Output CSV: eval/out/answers.csv  (override with --out). Same file is appended,
so delete it for a clean run (an existing file with a different header is refused,
not appended to). report.py reads latency, tokens and metrics from this file.

Does NOT invent numbers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

STAGE_FIELDS = ("rewrite_s", "search_s", "rerank_s", "generate_s")  # from Answer.timings
FIELDS = [
    "question", "gold_course", "gold_section",
    "answer", "citations", "latency_s",
    "refused", "citation_correctness", "faithfulness", "completeness",
    "prompt_tokens", "output_tokens",
    "context_ids",  # ;-joined chunk_id of each context doc (provenance)
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


def _ollama_ps():
    """Resident Ollama models with their VRAM split, or None on any error."""
    import urllib.request

    from bearlm import config

    try:
        with urllib.request.urlopen(f"{config.OLLAMA_BASE_URL}/api/ps", timeout=3) as r:
            models = json.load(r).get("models") or []
        return [{k: m.get(k) for k in ("name", "size", "size_vram")} for m in models]
    except Exception:
        return None


def _nvidia_smi():
    """(used_mb, total_mb) of GPU 0 from nvidia-smi, or (None, None)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        used, total = out.splitlines()[0].split(",")
        return int(used), int(total)
    except Exception:
        return None, None


def _gpu_info() -> dict:
    out = {"torch_version": None, "cuda_available": None, "rerank_device": None,
           "torch_cuda_max_reserved_mb": None}
    try:
        import torch

        out["torch_version"] = torch.__version__
        out["cuda_available"] = torch.cuda.is_available()
        if out["cuda_available"]:
            out["torch_cuda_max_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    except Exception:
        pass
    try:
        from bearlm import rerank

        if rerank._cross_encoder.cache_info().currsize:
            out["rerank_device"] = str(rerank._cross_encoder().device)
    except Exception:
        pass
    out["nvidia_smi_used_mb"], out["nvidia_smi_total_mb"] = _nvidia_smi()
    return out


_META_CONFIG = ("OLLAMA_BASE_URL", "OLLAMA_CHAT_MODEL", "OLLAMA_EMBED_MODEL",
                "OLLAMA_NUM_CTX", "RERANK_ENABLED", "RERANK_MODEL", "RERANK_CANDIDATES",
                "RERANK_TOP_N", "RERANK_MIN_SCORE", "HYBRID_SEARCH_ENABLED", "RRF_K",
                "QUERY_REWRITE_ENABLED", "RERANK_DEVICE")


def write_run_meta(out_path: Path, questions_path, n_rows: int, judge: bool) -> None:
    """Best-effort sidecar <out>.meta.json: which store, config and VRAM split produced
    the CSV. Never raises."""
    try:
        from bearlm import config

        meta = {"chroma_dir": str(Path(config.CHROMA_DIR).resolve()),
                "collection": config.CHROMA_COLLECTION, "collection_count": None,
                "questions_path": str(questions_path), "questions_sha256": None,
                "n_rows": n_rows,
                "config": {k: getattr(config, k) for k in _META_CONFIG if hasattr(config, k)},
                "judge": judge}
        try:
            from bearlm import local_store

            meta["collection_count"] = local_store.get_vectorstore()._collection.count()
        except Exception:
            pass
        try:
            qp = Path(questions_path) if questions_path else _common.EVAL_DIR / "questions.jsonl"
            meta["questions_sha256"] = hashlib.sha256(qp.read_bytes()).hexdigest()
        except Exception:
            pass
        try:
            meta["ollama_ps"] = _ollama_ps()
        except Exception:
            meta["ollama_ps"] = None
        meta["gpu"] = _gpu_info()
        out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"run meta not written: {type(e).__name__}: {e}")


def _fmt(x) -> str:
    return "" if x is None else f"{x:.3f}"


def run_eval(out_path: Path, questions_path: str | None = None, limit: int = 0,
                    judge: bool = True) -> None:
    import tiktoken

    from bearlm import config, rag

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
    n_rows = 0
    try:
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
                errored = answer_text.startswith("[ERROR]")  # infra failure (e.g. Ollama OOM): not scored
                scores = {"faithfulness": None, "completeness": None}
                if judge and not refused and not errored:
                    ref = _gold_text(q) if q.get("gold_chunk_ids") else None
                    scores = judge_answer(q["question"], answer_text, contexts, ref)
                w.writerow({
                    "question": q["question"],
                    "gold_course": q.get("gold_course", ""),
                    "gold_section": q.get("gold_section", ""),
                    "answer": answer_text,
                    "citations": "; ".join(f"{c['course']} {c['section']}" for c in cites),
                    "latency_s": f"{latency:.3f}",
                    "refused": int(refused),
                    "citation_correctness": "" if errored else _fmt(_common.citation_correctness(answer_text, docs)),
                    "faithfulness": _fmt(scores["faithfulness"]),
                    "completeness": _fmt(scores["completeness"]),
                    "prompt_tokens": len(enc.encode(
                        rag._STRICT_SYSTEM + "\n\n".join(contexts) + q["question"])),
                    "output_tokens": len(enc.encode(answer_text)),
                    "context_ids": ";".join(_common.doc_meta(d).get("chunk_id") or "" for d in docs),
                    **{k: _fmt(timings.get(k)) for k in STAGE_FIELDS},
                    "grade": "",
                    "notes": "",
                })
                f.flush()  # keep finished rows if the run is killed
                n_rows += 1
                print(f"{latency:6.2f}s  {q['question'][:60]}")
    finally:  # a killed or crashed run still records what produced its rows
        write_run_meta(out_path, questions_path, n_rows, judge)
    print(f"Appended {n_rows} rows to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_common.EVAL_DIR / "out" / "answers.csv"))
    default_q = _common.EVAL_DIR / "questions_generated.jsonl"
    ap.add_argument("--questions", default=str(default_q if default_q.exists()
                                               else _common.EVAL_DIR / "questions.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the local LLM faithfulness/completeness judge")
    args = ap.parse_args()
    run_eval(Path(args.out), args.questions, args.limit, judge=not args.no_judge)


if __name__ == "__main__":
    main()
