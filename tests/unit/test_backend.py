from __future__ import annotations

import sys
import types

import torch

from MegaASR.runtime.backend import resolve_attn_backend


def _make_torch_cuda_available(monkeypatch, available: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)


def test_auto_prefers_flash_attention_2_when_available(monkeypatch) -> None:
    _make_torch_cuda_available(monkeypatch, True)
    monkeypatch.setitem(sys.modules, "flash_attn", types.SimpleNamespace())

    assert resolve_attn_backend() == "flash_attention_2"


def test_auto_falls_back_to_sdpa_when_flash_attn_missing(monkeypatch) -> None:
    _make_torch_cuda_available(monkeypatch, True)
    monkeypatch.delitem(sys.modules, "flash_attn", raising=False)

    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "flash_attn":
            raise ImportError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    assert resolve_attn_backend() == "sdpa"


def test_auto_falls_back_to_eager_when_no_sdpa(monkeypatch) -> None:
    _make_torch_cuda_available(monkeypatch, True)
    monkeypatch.delitem(sys.modules, "flash_attn", raising=False)

    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "flash_attn":
            raise ImportError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    # Remove scaled_dot_product_attention to simulate old PyTorch
    sdpa = getattr(torch.nn.functional, "scaled_dot_product_attention", None)
    if sdpa is not None:
        monkeypatch.delattr(torch.nn.functional, "scaled_dot_product_attention")

    assert resolve_attn_backend() == "eager"


def test_auto_skips_flash_attn_on_cpu(monkeypatch) -> None:
    _make_torch_cuda_available(monkeypatch, False)
    monkeypatch.setitem(sys.modules, "flash_attn", types.SimpleNamespace())

    # Even though flash_attn is "installed", auto should skip it on non-CUDA
    result = resolve_attn_backend()
    assert result in ("sdpa", "eager")


def test_requested_flash_attention_2_falls_back_to_sdpa(monkeypatch) -> None:
    _make_torch_cuda_available(monkeypatch, True)
    monkeypatch.setattr("MegaASR.runtime.backend.is_flash_attn_2_available", lambda: False)

    assert resolve_attn_backend("flash_attention_2") == "sdpa"


def test_requested_flash_attention_3_falls_back_to_flash_attention_2(monkeypatch) -> None:
    _make_torch_cuda_available(monkeypatch, True)
    monkeypatch.setattr("MegaASR.runtime.backend.is_flash_attn_3_available", lambda: False)
    monkeypatch.setattr("MegaASR.runtime.backend.is_flash_attn_2_available", lambda: True)

    assert resolve_attn_backend("flash_attention_3") == "flash_attention_2"


def test_requested_flash_attention_3_falls_back_to_sdpa(monkeypatch) -> None:
    _make_torch_cuda_available(monkeypatch, True)
    monkeypatch.setattr("MegaASR.runtime.backend.is_flash_attn_3_available", lambda: False)
    monkeypatch.setattr("MegaASR.runtime.backend.is_flash_attn_2_available", lambda: False)

    assert resolve_attn_backend("flash_attention_3") == "sdpa"


def test_requested_supported_non_auto_backend_is_returned(monkeypatch) -> None:
    _make_torch_cuda_available(monkeypatch, True)
    monkeypatch.delitem(sys.modules, "flash_attn", raising=False)

    assert resolve_attn_backend("sdpa") == "sdpa"
    assert resolve_attn_backend("eager") == "eager"
