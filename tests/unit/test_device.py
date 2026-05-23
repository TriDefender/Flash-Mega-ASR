from __future__ import annotations

import importlib
import sys
import types


class FakeCuda:
    def __init__(self) -> None:
        self.available = False
        self.count = 0
        self.capabilities: dict[int, tuple[int, int]] = {}
        self.names: dict[int, str] = {}
        self.properties: dict[int, object] = {}

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self.count

    def get_device_capability(self, index: int) -> tuple[int, int]:
        value = self.capabilities.get(index)
        if isinstance(value, Exception):
            raise value
        if value is None:
            raise RuntimeError("missing capability")
        return value

    def get_device_name(self, index: int) -> str:
        return self.names[index]

    def get_device_properties(self, index: int) -> object:
        return self.properties[index]


def load_device_module():
    fake_torch = types.ModuleType("torch")
    setattr(fake_torch, "bfloat16", object())
    setattr(fake_torch, "float16", object())
    setattr(fake_torch, "float32", object())
    setattr(fake_torch, "cuda", FakeCuda())
    setattr(fake_torch, "backends", types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)))

    sys.modules["torch"] = fake_torch
    sys.modules.pop("MegaASR.runtime.device", None)

    module = importlib.import_module("MegaASR.runtime.device")
    return module, fake_torch


def test_resolve_device_defaults_to_cuda() -> None:
    module, fake_torch = load_device_module()
    fake_torch.cuda.available = True

    assert module.resolve_device() == "cuda:0"


def test_resolve_device_defaults_to_mps() -> None:
    module, fake_torch = load_device_module()
    fake_torch.backends.mps = types.SimpleNamespace(is_available=lambda: True)

    assert module.resolve_device() == "mps"


def test_resolve_device_defaults_to_cpu() -> None:
    module, _ = load_device_module()

    assert module.resolve_device() == "cpu"


def test_resolve_device_maps_numeric_cuda_id() -> None:
    module, fake_torch = load_device_module()
    fake_torch.cuda.available = True

    assert module.resolve_device("1") == "cuda:1"


def test_resolve_device_falls_back_to_cpu_when_cuda_unavailable() -> None:
    module, _ = load_device_module()

    assert module.resolve_device("cuda:0") == "cpu"
    assert module.resolve_device("0") == "cpu"


def test_resolve_dtype_prefers_bfloat16_on_ampere_plus() -> None:
    module, fake_torch = load_device_module()
    fake_torch.cuda.capabilities[0] = (8, 0)

    assert module.resolve_dtype("cuda:0") is fake_torch.bfloat16


def test_resolve_dtype_uses_float16_on_older_cuda() -> None:
    module, fake_torch = load_device_module()
    fake_torch.cuda.capabilities[0] = (7, 5)

    assert module.resolve_dtype("cuda:0") is fake_torch.float16


def test_resolve_dtype_uses_float16_on_mps() -> None:
    module, fake_torch = load_device_module()

    assert module.resolve_dtype("mps") is fake_torch.float16


def test_resolve_dtype_uses_float32_on_cpu() -> None:
    module, fake_torch = load_device_module()

    assert module.resolve_dtype("cpu") is fake_torch.float32


def test_resolve_dtype_falls_back_to_float16_when_capability_lookup_fails() -> None:
    module, fake_torch = load_device_module()
    fake_torch.cuda.capabilities[0] = RuntimeError("boom")

    assert module.resolve_dtype("cuda:0") is fake_torch.float16


def test_get_device_info_with_no_cuda() -> None:
    module, fake_torch = load_device_module()
    fake_torch.backends.mps = types.SimpleNamespace(is_available=lambda: True)

    info = module.get_device_info()

    assert info == {
        "cuda_available": False,
        "cuda_device_count": 0,
        "mps_available": True,
    }


def test_get_device_info_with_cuda_devices() -> None:
    module, fake_torch = load_device_module()
    fake_torch.cuda.available = True
    fake_torch.cuda.count = 2
    fake_torch.cuda.capabilities = {0: (8, 0), 1: (8, 1)}
    fake_torch.cuda.names = {0: "GPU-0", 1: "GPU-1"}
    fake_torch.cuda.properties = {
        0: types.SimpleNamespace(total_mem=10_000_000_000),
        1: types.SimpleNamespace(total_memory=20_000_000_000),
    }

    info = module.get_device_info()

    assert info["cuda_available"] is True
    assert info["cuda_device_count"] == 2
    assert info["mps_available"] is False
    assert info["cuda:0"] == {"name": "GPU-0", "capability": "8.0", "memory_gb": 10.0}
    assert info["cuda:1"] == {"name": "GPU-1", "capability": "8.1", "memory_gb": 20.0}
