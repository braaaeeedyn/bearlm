import json
import re
from pathlib import Path

import _common
import speed_experiments
from bearlm import config, rag

ROOT = Path(__file__).resolve().parent.parent
HISTORY_TYPES = ("history-followup", "history-sibling-followup", "history-same")
REVIEW_PAIRS = [
    ("What are the advantages of the Naive Bayes classifier?", "What are its disadvantages?"),
    ("What are the advantages of using a hash join?", "What about the disadvantages?"),
    ("What is a variational autoencoder?", "How is it trained?"),
    ("What is L1 regularization?", "And L2?"),
]
DISJOINT_PREVS = ("What is the time complexity of the find operation in the disjoint set data structure?",
                  "What is the time complexity of find in a disjoint set?")


def _pairs():
    return _common.load_questions(speed_experiments.PAIRS)


def _results():
    return json.loads(speed_experiments.RESULTS.read_text(encoding="utf-8"))


def _read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def _threshold_md():
    t = config.NEAR_REPEAT_THRESHOLD
    return "0" if t == 0 else f"{t:g}"


def test_pairs_include_history_followups():
    rows = _pairs()
    orig = [r for r in rows if r.get("type") not in HISTORY_TYPES]
    assert sum(r["label"] == "same" for r in orig) == 22
    assert sum(r["label"] == "different" for r in orig) == 24

    def built(t, label):
        """Rows of type t whose b (and a) are built like rag._retrieval_query: prev + " " + follow-up."""
        return [r for r in rows if r.get("type") == t and r["label"] == label
                and r["b"] == r["prev"] + " " + r["followup"]
                and r["a"] == (r["prev"] if t == "history-followup" else r["prev"] + " " + r["followup_a"])]

    fu = built("history-followup", "different")
    sib = built("history-sibling-followup", "different")
    same = built("history-same", "same")
    assert len(fu) >= 12 and len(sib) >= 3 and len(same) >= 5
    assert len(fu) + len(sib) + len(same) == len(rows) - len(orig)  # every history row is well-formed
    assert all(rag._retrieval_query(r["followup"], [{"role": "user", "content": r["prev"]}]) == r["b"]
               for r in fu + sib + same)

    got = {(r["prev"], r["followup"]) for r in fu}
    for pair in REVIEW_PAIRS:
        assert pair in got, pair
    assert sum((p, "What about union?") in got for p in DISJOINT_PREVS) == 1
    assert any(r["followup_a"] == "What are its advantages?" and r["followup"] == "What are its disadvantages?"
               and "Naive Bayes" in r["prev"] for r in sib)


def test_near_repeat_decision_reproducible():
    nr = _results()["near_repeat"]
    assert {(p["a"], p["b"], p["label"]) for p in nr["pairs"]} == \
        {(p["a"], p["b"], p["label"]) for p in _pairs()}
    d = speed_experiments.decide_near_repeat(nr)
    assert (d["decision"], d["threshold"]) == (nr["decision"], nr["threshold"])
    assert speed_experiments.NEAR_REPEAT_CRITERIA == {
        "false_reuse_max": 0, "hit_rate_min": 0.5, "gate_agreement_min": 1.0,
        "mean_overlap_min": 0.5, "median_overhead_s_max": 0.2}


def test_stage_breakdown_at_current_defaults():
    sb = _results()["stage_breakdown"]
    assert sb["config"] == {"rerank_model": config.RERANK_MODEL,
                            "rerank_candidates": config.RERANK_CANDIDATES,
                            "rerank_enabled": config.RERANK_ENABLED,
                            "query_rewrite": config.QUERY_REWRITE_ENABLED}
    assert sb["n"] >= 20


def _row(text, var):
    return next(line for line in text.splitlines() if line.startswith(f"| `{var}` |"))


def test_readme_speed_claim():
    readme = _read("README.md")
    assert "~3x" not in readme
    assert "~2x" in _row(readme, "RERANK_CANDIDATES")
    assert _row(readme, "NEAR_REPEAT_THRESHOLD").split("|")[2].strip() == f"`{_threshold_md()}`"


def test_env_example_matches_defaults():
    env = _read(".env.example")
    values = dict(re.findall(r"^([A-Z_]+)=(\S*)", env, re.M))
    assert int(values["RERANK_CANDIDATES"]) == config.RERANK_CANDIDATES
    assert "retrieve 15" not in env
    if "NEAR_REPEAT_THRESHOLD" in values:
        assert float(values["NEAR_REPEAT_THRESHOLD"]) == config.NEAR_REPEAT_THRESHOLD


def test_running_doc_seeded():
    doc = _read("PROGRESS.md")
    assert doc.startswith("# BearLM — Running log")
    entries = re.split(r"^## ", doc, flags=re.M)[1:]
    assert len(entries) >= 6
    for e in entries:
        for label in ("**What changed:**", "**Why:**", "**Measured:**", "**Decision:**"):
            assert label in e, (e[:60], label)
    assert "3656e35" in doc and "748f5e7" in doc
    assert sum("near-repeat" in e.lower() for e in entries) >= 2
    assert "START HERE" not in doc and "Stage 8" not in doc


def test_current_doc_matches_config():
    doc = _read("PROJECT_HANDOFF.md")
    for bad in ("Streamlit", "Change history", "off-topic ≈ 0.0", "still manual", "15→rerank", "median ~6 s"):
        assert bad not in doc, bad
    assert "PROGRESS.md" in doc
    line = next(ln for ln in doc.splitlines() if "Near-repeat" in ln)
    if config.NEAR_REPEAT_THRESHOLD == 0:
        assert "off by default" in line and "0.96" not in line
    else:
        assert str(config.NEAR_REPEAT_THRESHOLD) in line
    s7 = re.search(r"^## 7\..*?(?=^## )", doc, re.M | re.S).group(0)
    assert "NEAR_REPEAT_THRESHOLD" in s7


def test_ollama_base_url_loopback():
    url = "http://127.0.0.1:11434"
    assert config.OLLAMA_BASE_URL == url
    values = dict(re.findall(r"^([A-Z_]+)=(\S*)", _read(".env.example"), re.M))
    assert values["OLLAMA_BASE_URL"] == url
    sb = _results()["stage_breakdown"]
    assert sb["ollama_base_url"] == url
    assert sb["median_s"]["search_s"] < 0.5


def test_docs_explain_localhost_penalty():
    handoff = _read("PROJECT_HANDOFF.md")
    assert "contention" not in handoff and "127.0.0.1" in handoff
    entries = re.split(r"^## ", _read("PROGRESS.md"), flags=re.M)[1:]
    assert len(entries) >= 7
    assert "127.0.0.1" in entries[-1] and "generate" in entries[-1]


def test_docs_quote_reproducible_quality():
    handoff = _read("PROJECT_HANDOFF.md")
    last = re.split(r"^## ", _read("PROGRESS.md"), flags=re.M)[-1]
    for doc in (handoff, last):
        assert "citation correctness 0.20" not in doc and "faithfulness 0.84" not in doc
        assert "0.18" in doc
    assert "5.3 s" in handoff and "5.4 s" not in handoff


def _last_entry():
    return re.split(r"^## ", _read("PROGRESS.md"), flags=re.M)[-1]


def test_run5_recorded():
    last = _last_entry()
    assert "Run 5" in last
    for label in ("**What changed:**", "**Why:**", "**Measured:**", "**Decision:**"):
        assert label in last, label
    assert "**Variance finding:**" in last and "0.8347" in last and "0.841" in last
    assert "B+" in last and "citation" in last.lower() and "gate" in last.lower()
    assert "RERANK_DEVICE" in last or "torch" in last


def test_gate_described_honestly():
    assert "Off-topic scores ~0.0" not in (ROOT / "src" / "bearlm" / "config.py").read_text(encoding="utf-8")
    handoff = _read("PROJECT_HANDOFF.md")
    assert "Gate recalibration" not in handoff
    s5 = re.search(r"^## 5\..*?(?=^## )", handoff, re.M | re.S).group(0)
    para = s5[s5.index("**Anti-hallucination**"):].split("\n\n")[0]
    assert "rag.answer" in para or "Ask page" in para
    assert re.search(r"not gated|no (confidence )?gate", para)


def test_gpu_decision_consistent():
    handoff = _read("PROJECT_HANDOFF.md")
    if "download.pytorch.org" in _read("pyproject.toml"):
        assert "+cu130" in _read("uv.lock")
        assert "installed torch is a CPU build" not in handoff
        assert "RERANK_DEVICE" in handoff
    else:
        assert "revert" in _last_entry().lower()
        assert "reranker runs on CPU" in handoff


def test_output_cap_documented():
    handoff = _read("PROJECT_HANDOFF.md")
    assert "OLLAMA_NUM_PREDICT" in _read("README.md") and "OLLAMA_NUM_PREDICT" in handoff
    assert "No output-token cap" not in handoff
    last = _last_entry()
    assert "num_predict" in last and "1024" in last


def test_revision2_env_and_timeout_docs():
    values = dict(re.findall(r"^([A-Z_]+)=(\S*)", _read(".env.example"), re.M))
    assert int(values["OLLAMA_NUM_PREDICT"]) == config.OLLAMA_NUM_PREDICT
    assert float(values["OLLAMA_TIMEOUT_S"]) == config.OLLAMA_TIMEOUT_S
    assert values["RERANK_DEVICE"] == config.RERANK_DEVICE
    handoff = _read("PROJECT_HANDOFF.md")
    for doc in (_read("README.md"), handoff):
        assert "per-request timeout" not in doc.lower()
    assert "slightly overstates" not in handoff
    assert "0.19" in handoff and "0.19" in _last_entry()
