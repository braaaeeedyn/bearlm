import sys
import types

import pytest

from bearlm import config, rerank


def _fake_torch(monkeypatch, cuda):
    torch = types.SimpleNamespace(float16="float16-dtype",
                                  cuda=types.SimpleNamespace(is_available=lambda: cuda))
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def _fake_cross_encoder(monkeypatch, fail_on_cuda=False):
    built = []

    class FakeCE:
        def __init__(self, model, device=None, model_kwargs=None):
            built.append({"model": model, "device": device, "model_kwargs": model_kwargs})
            if fail_on_cuda and device == "cuda":
                raise RuntimeError("CUDA out of memory")
            self.device = device

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(CrossEncoder=FakeCE))
    return built


@pytest.fixture(autouse=True)
def _clear_cache():
    rerank._cross_encoder.cache_clear()
    yield
    rerank._cross_encoder.cache_clear()


@pytest.mark.parametrize("setting,cuda,expected", [
    ("auto", True, "cuda"), ("auto", False, "cpu"), ("cpu", True, "cpu"), ("cuda", False, "cuda")])
def test_device_selection(monkeypatch, setting, cuda, expected):
    _fake_torch(monkeypatch, cuda)
    monkeypatch.setattr(config, "RERANK_DEVICE", setting)
    assert rerank._device() == expected


def test_cuda_uses_fp16_and_falls_back_to_cpu(monkeypatch):
    torch = _fake_torch(monkeypatch, True)
    monkeypatch.setattr(config, "RERANK_DEVICE", "auto")
    built = _fake_cross_encoder(monkeypatch)
    assert rerank._cross_encoder().device == "cuda"
    assert built[-1]["model_kwargs"] == {"torch_dtype": torch.float16}

    rerank._cross_encoder.cache_clear()
    built = _fake_cross_encoder(monkeypatch, fail_on_cuda=True)
    assert rerank._cross_encoder().device == "cpu"
    assert [b["device"] for b in built] == ["cuda", "cpu"]


def test_cross_encoder_does_not_mask_typeerror(monkeypatch):
    _fake_torch(monkeypatch, True)
    monkeypatch.setattr(config, "RERANK_DEVICE", "auto")

    def bad_ce(*a, **k):
        raise TypeError("unexpected keyword argument 'model_kwargs'")

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=bad_ce))
    with pytest.raises(TypeError):
        rerank._cross_encoder()
