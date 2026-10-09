"""Speed experiments: per-stage timings, near-repeat reuse, cheaper reranking.

Each flag runs one live experiment (local backend) and merges its section into
eval/speed_experiments.json; --decide applies the keep/reject criteria below in
code and records the decisions. Run with BACKEND=local, e.g.:
    BACKEND=local uv run python eval/speed_experiments.py --stages 20
    BACKEND=local uv run python eval/speed_experiments.py --rerank-sweep
    BACKEND=local uv run python eval/speed_experiments.py --near-repeat
    BACKEND=local uv run python eval/speed_experiments.py --answer-csv A=eval/out/answers_A.csv E=...
    uv run python eval/speed_experiments.py --decide

Invents no numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

RESULTS = _common.EVAL_DIR / "speed_experiments.json"
PAIRS = _common.EVAL_DIR / "near_repeat_pairs.jsonl"
QUESTIONS = _common.EVAL_DIR / "questions_generated.jsonl"
STAGES = ("rewrite_s", "search_s", "rerank_s", "generate_s", "total_s")
BGE, MINILM = "BAAI/bge-reranker-base", "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_CONFIGS = [("A", BGE, 15), ("B", BGE, 10), ("C", BGE, 8), ("D", MINILM, 15), ("E", MINILM, 10)]
BGE_MIN_SCORE = 0.02
OFF_TOPIC = _common.OFF_TOPIC
NEAR_REPEAT_CRITERIA = {"false_reuse_max": 0, "hit_rate_min": 0.5, "gate_agreement_min": 1.0,
                        "mean_overlap_min": 0.5, "median_overhead_s_max": 0.2}
RERANK_CRITERIA = {"speedup_min": 0.30, "recall@5_drop_max": 0.02, "precision@5_drop_max": 0.03,
                   "mrr_drop_max": 0.03, "faithfulness_drop_max": 0.05,
                   "completeness_drop_max": 0.05, "citation_correctness_drop_max": 0.05,
                   "refusals_rise_max": 1}


def _load() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}


def _merge(section: str, value) -> None:
    data = _load()
    data[section] = {**data.get(section, {}), **value} if isinstance(value, dict) else value
    RESULTS.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(0.95 * (len(xs) - 1)))]


def _machine() -> None:
    import torch

    from bearlm import rerank

    _merge("machine", {"torch_cuda": torch.cuda.is_available(),
                       "rerank_device": str(rerank._cross_encoder().device)})


def run_stages(n: int) -> None:
    from bearlm import rag

    from bearlm import config

    _machine()
    rag.answer("warm-up query")  # first call loads models; not counted
    rows = []
    for q in _common.load_questions(QUESTIONS)[:n]:
        t = rag.answer(q["question"]).timings
        rows.append(t)
        print("  ".join(f"{k}={t.get(k, 0.0):.2f}" for k in STAGES))
    _merge("stage_breakdown", {
        "n": len(rows),
        "config": {"rerank_model": config.RERANK_MODEL, "rerank_candidates": config.RERANK_CANDIDATES,
                   "rerank_enabled": config.RERANK_ENABLED, "query_rewrite": config.QUERY_REWRITE_ENABLED},
        "ollama_base_url": config.OLLAMA_BASE_URL,
        "median_s": {k: statistics.median(r.get(k, 0.0) for r in rows) for k in STAGES},
        "p95_s": {k: _p95([r.get(k, 0.0) for r in rows]) for k in STAGES},
    })


def _use_reranker(model: str, candidates: int) -> None:
    from bearlm import config, rerank

    config.RERANK_MODEL, config.RERANK_CANDIDATES = model, candidates
    rerank._cross_encoder.cache_clear()


def run_rerank_sweep() -> None:
    from bearlm import config, rerank, search

    _machine()
    questions = _common.load_questions(QUESTIONS)
    orig, configs = (config.RERANK_MODEL, config.RERANK_CANDIDATES), []
    for cid, model, n in RERANK_CONFIGS:
        _use_reranker(model, n)
        rerank.rerank_scored("warm-up query", search.candidates("warm-up query", config.CHROMA_COLLECTION, n))
        s_times, r_times, runs, tops = [], [], [], []
        for q in questions:
            t0 = time.perf_counter()
            cands = search.candidates(q["question"], config.CHROMA_COLLECTION, n)
            t1 = time.perf_counter()
            scored = rerank.rerank_scored(q["question"], cands)
            t2 = time.perf_counter()
            s_times.append(t1 - t0)
            r_times.append(t2 - t1)
            runs.append(([d for d, _ in scored], q))
            tops.append(scored[0][1] if scored else float("-inf"))
        off = []
        for q in OFF_TOPIC:
            scored = rerank.rerank_scored(q, search.candidates(q, config.CHROMA_COLLECTION, n))
            off.append(scored[0][1] if scored else float("-inf"))
        row = {
            "id": cid, "model": model, "candidates": n,
            "median_search_s": statistics.median(s_times),
            "median_rerank_s": statistics.median(r_times),
            "median_retrieval_s": statistics.median(a + b for a, b in zip(s_times, r_times)),
            "recall@5": sum(_common.recall_at_k(d, q, 5) for d, q in runs) / len(runs),
            "precision@5": sum(_common.precision_at_k(d, q, 5) for d, q in runs) / len(runs),
            "mrr": sum(_first_rr(d, q) for d, q in runs) / len(runs),
            "top_scores_in_scope": tops, "top_scores_off_topic": off,
        }
        configs.append(row)
        print(f"{cid} {model} n={n}: search {row['median_search_s']:.3f}s rerank "
              f"{row['median_rerank_s']:.3f}s R@5 {row['recall@5']:.3f} P@5 "
              f"{row['precision@5']:.3f} MRR {row['mrr']:.3f}")
    _use_reranker(*orig)
    _merge("cheaper_rerank", {"configs": configs})


def _first_rr(docs, q) -> float:
    for rank, d in enumerate(docs, 1):
        if _common.is_gold_hit(d, q):
            return 1.0 / rank
    return 0.0


def run_near_repeat() -> None:
    from bearlm import config, local_store, rag

    pairs = _common.load_questions(PAIRS)
    emb = local_store.get_embeddings()
    rows, cache, overheads, fresh = [], [], [], []
    for p in pairs:
        va, vb = emb.embed_query(p["a"]), emb.embed_query(p["b"])
        cache = (cache + [va, vb])[-50:]
        t0 = time.perf_counter()
        v = emb.embed_query(p["b"])
        max(rag._cosine(v, e) for e in cache)  # full scan of a (capped) cache, as in answer_grounded
        overheads.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        da = rag._top_docs(p["a"], config.CHROMA_COLLECTION, gated=True)
        fresh.append(time.perf_counter() - t0)
        db = rag._top_docs(p["b"], config.CHROMA_COLLECTION, gated=True)
        ia = [d.metadata.get("chunk_id") for d in da]
        ib = {d.metadata.get("chunk_id") for d in db}
        overlap = (sum(1 for i in ia if i in ib) / len(ia)) if ia else (1.0 if not ib else 0.0)
        rows.append({**p, "sim": rag._cosine(va, vb), "answer_a": bool(da), "answer_b": bool(db),
                     "overlap": overlap})
        print(f"{p['label']:9} sim={rows[-1]['sim']:.4f} gate {bool(da)}/{bool(db)} overlap {overlap:.2f}")
    sweep = []
    for i in range(80, 100):
        t = i / 100
        sweep.append({"threshold": t, **_reuse_metrics(rows, t)})
    _merge("near_repeat", {"pairs": rows, "sweep": sweep,
                           "median_overhead_s": statistics.median(overheads),
                           "median_fresh_retrieval_s": statistics.median(fresh)})


def _reuse_metrics(rows, t) -> dict:
    same = [r for r in rows if r["label"] == "same"]
    diff = [r for r in rows if r["label"] == "different"]
    hits = [r for r in same if r["sim"] >= t]
    return {
        "hit_rate": len(hits) / len(same) if same else 0.0,
        "false_reuse": sum(1 for r in diff if r["sim"] >= t),
        "n_same": len(same), "n_different": len(diff), "hits": len(hits),
        "gate_agreement": (sum(r["answer_a"] == r["answer_b"] for r in hits) / len(hits)) if hits else None,
        "mean_overlap": statistics.mean(r["overlap"] for r in hits) if hits else None,
    }


def summarize_answers(specs: list[str]) -> None:
    out = {}
    for spec in specs:
        cid, path = spec.split("=", 1)
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        def mean(col):
            vals = [float(r[col]) for r in rows if r[col] != ""]
            return statistics.mean(vals) if vals else None

        out[cid] = {"n": len(rows), "csv": path, "mean_faithfulness": mean("faithfulness"),
                    "mean_completeness": mean("completeness"),
                    "mean_citation_correctness": mean("citation_correctness"),
                    "refusals": sum(int(r["refused"]) for r in rows),
                    "median_latency_s": statistics.median(float(r["latency_s"]) for r in rows)}
        print(cid, out[cid])
    _merge("cheaper_rerank", {"answer_quality": {**_load().get("cheaper_rerank", {}).get("answer_quality", {}), **out}})


def decide_near_repeat(nr: dict) -> dict:
    rows = nr["pairs"]
    max_diff = max(r["sim"] for r in rows if r["label"] == "different")
    t = next((s["threshold"] for s in nr["sweep"]
              if s["false_reuse"] == 0 and s["threshold"] >= round(max_diff + 0.01, 6)), None)
    c = NEAR_REPEAT_CRITERIA
    if t is None:
        return {"decision": "reject", "threshold": None, "criteria": c, "metrics_at_threshold": {},
                "reason": f"no swept threshold <= 0.99 clears max different-pair sim {max_diff:.4f} + 0.01"}
    m = {**_reuse_metrics(rows, t), "median_overhead_s": nr["median_overhead_s"],
         "max_different_sim": max_diff}
    fails = []
    if m["false_reuse"] > c["false_reuse_max"]:
        fails.append(f"false reuse {m['false_reuse']}")
    if m["hit_rate"] < c["hit_rate_min"]:
        fails.append(f"hit rate {m['hit_rate']:.2f} < {c['hit_rate_min']}")
    if m["gate_agreement"] is None or m["gate_agreement"] < c["gate_agreement_min"]:
        fails.append(f"gate agreement {m['gate_agreement']} < {c['gate_agreement_min']}")
    if m["mean_overlap"] is None or m["mean_overlap"] < c["mean_overlap_min"]:
        fails.append(f"mean overlap {m['mean_overlap']} < {c['mean_overlap_min']}")
    if m["median_overhead_s"] > c["median_overhead_s_max"]:
        fails.append(f"overhead {m['median_overhead_s']:.3f}s > {c['median_overhead_s_max']}s")
    return {"decision": "reject" if fails else "keep", "threshold": t, "criteria": c,
            "metrics_at_threshold": m,
            "reason": "; ".join(fails) if fails else f"all criteria met at t*={t}"}


def _gate_probe(row: dict, base: dict) -> dict:
    """Smallest min_score refusing every off-topic question, and whether it refuses at
    most 1 more in-scope question than bge at BGE_MIN_SCORE."""
    min_score = round(max(row["top_scores_off_topic"]) + 0.0001, 4)
    refused = sum(1 for s in row["top_scores_in_scope"] if s < min_score)
    bge_refused = sum(1 for s in base["top_scores_in_scope"] if s < BGE_MIN_SCORE)
    return {"min_score": min_score, "in_scope_refused": refused, "bge_in_scope_refused": bge_refused,
            "pass": refused <= bge_refused + 1}


def decide_rerank(cr: dict) -> dict:
    c = RERANK_CRITERIA
    rows = {r["id"]: r for r in cr["configs"]}
    base = rows["A"]
    checks, passing = {}, []
    for r in sorted(cr["configs"], key=lambda r: r["median_retrieval_s"]):
        if r["id"] == "A":
            continue
        fails = []
        speedup = 1 - r["median_retrieval_s"] / base["median_retrieval_s"]
        if speedup < c["speedup_min"]:
            fails.append(f"speedup {speedup:.2f} < {c['speedup_min']}")
        for k in ("recall@5", "precision@5", "mrr"):
            if round(base[k] - r[k], 6) > c[f"{k}_drop_max"]:  # round: 0.86-0.84 > 0.02 in floats
                fails.append(f"{k} drop {base[k] - r[k]:.3f}")
        gate = _gate_probe(r, base) if r["model"] != BGE else None
        if gate and not gate["pass"]:
            fails.append(f"gate probe fails ({gate['in_scope_refused']} in-scope refused "
                         f"vs bge {gate['bge_in_scope_refused']})")
        checks[r["id"]] = {"speedup": speedup, "gate_probe": gate, "retrieval_fails": fails}
        if not fails:
            passing.append(r)
    out = {"criteria": c, "retrieval_checks": checks}
    if not passing:
        return {**out, "decision": "reject", "chosen": None,
                "reason": "no config passes the retrieval/speed/gate criteria"}
    pick = passing[0]  # fastest passing retrieval; the only one answer-evaluated
    aq = cr.get("answer_quality", {})
    if "A" not in aq or pick["id"] not in aq:
        return {**out, "decision": "reject", "chosen": None,
                "reason": f"answer quality for A and {pick['id']} not measured"}
    a, p = aq["A"], aq[pick["id"]]
    fails = []
    for k, ck in (("mean_faithfulness", "faithfulness_drop_max"),
                  ("mean_completeness", "completeness_drop_max"),
                  ("mean_citation_correctness", "citation_correctness_drop_max")):
        if round((a[k] or 0) - (p[k] or 0), 6) > c[ck]:
            fails.append(f"{k} drop {(a[k] or 0) - (p[k] or 0):.3f}")
    if p["refusals"] - a["refusals"] > c["refusals_rise_max"]:
        fails.append(f"refusals {a['refusals']} -> {p['refusals']}")
    if fails:
        return {**out, "decision": "reject", "chosen": None,
                "reason": f"{pick['id']} passes retrieval but fails answer quality: " + "; ".join(fails)}
    gate = checks[pick["id"]]["gate_probe"]
    return {**out, "decision": "keep",
            "chosen": {"id": pick["id"], "model": pick["model"], "candidates": pick["candidates"],
                       "min_score": gate["min_score"] if gate else BGE_MIN_SCORE},
            "reason": f"{pick['id']} is the fastest config meeting every criterion"}


def decide() -> None:
    data = _load()
    nr = decide_near_repeat(data["near_repeat"])
    cr = decide_rerank(data["cheaper_rerank"])
    _merge("near_repeat", nr)
    _merge("cheaper_rerank", cr)
    print(f"near_repeat: {nr['decision']} (threshold {nr['threshold']}): {nr['reason']}")
    print(f"cheaper_rerank: {cr['decision']} (chosen {cr['chosen']}): {cr['reason']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=int, default=0, metavar="N")
    ap.add_argument("--rerank-sweep", action="store_true")
    ap.add_argument("--near-repeat", action="store_true")
    ap.add_argument("--answer-csv", nargs="+", metavar="ID=PATH")
    ap.add_argument("--decide", action="store_true")
    args = ap.parse_args()
    if args.stages:
        run_stages(args.stages)
    if args.rerank_sweep:
        run_rerank_sweep()
    if args.near_repeat:
        run_near_repeat()
    if args.answer_csv:
        summarize_answers(args.answer_csv)
    if args.decide:
        decide()


if __name__ == "__main__":
    main()
