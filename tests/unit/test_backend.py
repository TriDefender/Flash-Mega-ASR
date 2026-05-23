from __future__ import annotations

import sys
import types

from MegaASR.runtime.backend import resolve_attn_backend


def test_auto_prefers_flash_attention_2_when_available(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "flash_attn", types.SimpleNamespace())

    assert resolve_attn_backend() == "flash_attention_2"


def test_auto_falls_back_to_sdpa_when_flash_attn_missing(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "flash_attn", raising=False)

    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "flash_attn":
            raise ImportError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    assert resolve_attn_backend() == "sdpa"


def test_requested_flash_attention_2_falls_back_to_sdpa(monkeypatch) -> None:
    monkeypatch.setattr("MegaASR.runtime.backend.is_flash_attn_2_available", lambda: False)

    assert resolve_attn_backend("flash_attention_2") == "sdpa"


def test_requested_flash_attention_3_falls_back_to_flash_attention_2(monkeypatch) -> None:
    monkeypatch.setattr("MegaASR.runtime.backend.is_flash_attn_3_available", lambda: False)
    monkeypatch.setattr("MegaASR.runtime.backend.is_flash_attn_2_available", lambda: True)

    assert resolve_attn_backend("flash_attention_3") == "flash_attention_2"


def test_requested_flash_attention_3_falls_back_to_sdpa(monkeypatch) -> None:
    monkeypatch.setattr("MegaASR.runtime.backend.is_flash_attn_3_available", lambda: False)
    monkeypatch.setattr("MegaASR.runtime.backend.is_flash_attn_2_available", lambda: False)

    assert resolve_attn_backend("flash_attention_3") == "sdpa"


def test_requested_supported_non_auto_backend_is_returned(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "flash_attn", raising=False)

    assert resolve_attn_backend("sdpa") == "sdpa"
    assert resolve_attn_backend("eager") == "eager"
