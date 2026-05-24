from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import torch


def load_router_module():
    module_path = Path(__file__).resolve().parents[2] / "src" / "MegaASR" / "model" / "router.py"

    megaasr_pkg = sys.modules.setdefault("MegaASR", ModuleType("MegaASR"))
    megaasr_pkg.__path__ = []
    model_pkg = sys.modules.setdefault("MegaASR.model", ModuleType("MegaASR.model"))
    model_pkg.__path__ = []
    utils_pkg = sys.modules.setdefault("MegaASR.model.utils", ModuleType("MegaASR.model.utils"))
    utils_pkg.__path__ = []

    audio_quality_module = ModuleType("MegaASR.model.utils.audio_quality")
    setattr(audio_quality_module, "LogMelSpectrogram", object)
    setattr(audio_quality_module, "create_audio_quality_model", lambda config: None)

    sys.modules["MegaASR.model.utils.audio_quality"] = audio_quality_module
    soundfile_module = ModuleType("soundfile")
    setattr(soundfile_module, "read", None)
    sys.modules.setdefault("soundfile", soundfile_module)
    safetensors_module = ModuleType("safetensors")
    setattr(safetensors_module, "safe_open", None)
    sys.modules.setdefault("safetensors", safetensors_module)
    safetensors_torch_module = ModuleType("safetensors.torch")
    setattr(safetensors_torch_module, "load_file", None)
    sys.modules.setdefault("safetensors.torch", safetensors_torch_module)

    scipy_module = sys.modules.setdefault("scipy", ModuleType("scipy"))
    scipy_module.__path__ = []
    signal_module = ModuleType("scipy.signal")
    setattr(signal_module, "resample_poly", lambda audio, up, down: audio)
    sys.modules["scipy.signal"] = signal_module

    spec = spec_from_file_location("MegaASR.model.router_for_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


router = load_router_module()
AudioQualityRouter = router.AudioQualityRouter


def make_router(*, sample_rate=16000, device="cpu"):
    instance = object.__new__(AudioQualityRouter)
    instance.sample_rate = sample_rate
    instance.device = device
    instance.threshold = 0.5
    return instance


def test_load_audio_returns_correct_shape(monkeypatch):
    mono_signal = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    stereo_signal = np.stack([mono_signal, mono_signal], axis=1)

    monkeypatch.setattr(router.sf, "read", lambda *args, **kwargs: (stereo_signal, 16000))

    loaded = make_router()._load_audio("dummy.wav")

    assert tuple(loaded.shape) == (1, mono_signal.shape[0])
    assert loaded.dtype == getattr(torch, "float32")


def test_load_audio_resamples_when_sr_mismatch(monkeypatch):
    mono_signal = np.array([0.2, -0.1, 0.4], dtype=np.float32)
    stereo_signal = np.stack([mono_signal, mono_signal], axis=1)
    calls = []

    def fake_resample(audio, up, down):
        calls.append((audio.copy(), up, down))
        return audio

    monkeypatch.setattr(router.sf, "read", lambda *args, **kwargs: (stereo_signal, 44100))
    monkeypatch.setattr(router, "resample_poly", fake_resample)

    make_router()._load_audio("dummy.wav")

    assert len(calls) == 1


def test_load_audio_no_resample_when_16k(monkeypatch):
    mono_signal = np.array([0.2, -0.1, 0.4], dtype=np.float32)
    stereo_signal = np.stack([mono_signal, mono_signal], axis=1)
    called = False

    def fake_resample(audio, up, down):
        nonlocal called
        called = True
        return audio

    monkeypatch.setattr(router.sf, "read", lambda *args, **kwargs: (stereo_signal, 16000))
    monkeypatch.setattr(router, "resample_poly", fake_resample)

    make_router()._load_audio("dummy.wav")

    assert called is False
