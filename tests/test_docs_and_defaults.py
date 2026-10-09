import gzip
import json
import re
from pathlib import Path

from bearlm import config

ROOT = Path(__file__).resolve().parent.parent


def _read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def _env_values():
    return dict(re.findall(r"^([A-Z_]+)=(\S*)", _read(".env.example"), re.M))


def test_env_example_matches_defaults():
    values = _env_values()
    for key, val in values.items():
        default = getattr(config, key)
        if isinstance(default, bool):
            assert (val.lower() in ("1", "true", "yes")) == default, key
        elif isinstance(default, (int, float)):
            assert type(default)(val) == default, key
        elif isinstance(default, str) and key != "CHROMA_DIR":
            assert val == default, key


def test_every_env_setting_documented_in_readme():
    readme = _read("README.md")
    for key in _env_values():
        assert f"`{key}`" in readme, key


def test_ollama_base_url_loopback():
    # localhost costs ~2 s per new connection on Windows (IPv6 fallback).
    assert config.OLLAMA_BASE_URL == "http://127.0.0.1:11434"
    assert _env_values()["OLLAMA_BASE_URL"] == config.OLLAMA_BASE_URL


def test_no_cloud_backend_left():
    py = _read("pyproject.toml")
    for dep in ("openai", "pymongo", "ragas"):
        assert f'"{dep}' not in py, dep
    assert not hasattr(config, "BACKEND")
    assert "BACKEND" not in _read(".env.example")
    assert 'description = "Add your description here"' not in py


def test_chunk_export_ships_text_not_pdfs():
    with gzip.open(config.CHUNKS_EXPORT, "rt", encoding="utf-8") as f:
        first = json.loads(f.readline())
    assert set(first) == {"id", "text", "metadata"}
    assert {"course", "section", "source", "chunk_id"} <= set(first["metadata"])
    assert ":\\" not in first["metadata"]["source"] and not first["metadata"]["source"].startswith("/")
    gitignore = _read(".gitignore")
    assert "data/raw/" in gitignore and "db/" in gitignore


def test_docs_split_history_from_current():
    assert _read("PROGRESS.md").startswith("# BearLM — Running log")
    handoff = _read("PROJECT_HANDOFF.md")
    assert "PROGRESS.md" in handoff
    for gone in ("BACKEND=openai", "MongoDB Atlas", "NEAR_REPEAT_THRESHOLD", "--legacy-splitter", "ragas_eval"):
        assert gone not in handoff, gone
