from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import torch


torch_float32 = getattr(torch, "float32")
torch_ones = getattr(torch, "ones")
torch_softmax = getattr(torch, "softmax")
torch_tensor = getattr(torch, "tensor")
torch_is_inference_mode_enabled = getattr(torch, "is_inference_mode_enabled")


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

    try:
        import_module("soundfile")
    except ImportError:
        soundfile_module = ModuleType("soundfile")
        setattr(soundfile_module, "read", None)
        sys.modules.setdefault("soundfile", soundfile_module)

    try:
        import_module("safetensors")
    except ImportError:
        safetensors_module = ModuleType("safetensors")
        setattr(safetensors_module, "safe_open", None)
        sys.modules.setdefault("safetensors", safetensors_module)

    try:
        import_module("safetensors.torch")
    except ImportError:
        safetensors_torch_module = ModuleType("safetensors.torch")
        setattr(safetensors_torch_module, "load_file", None)
        sys.modules.setdefault("safetensors.torch", safetensors_torch_module)

    try:
        import_module("scipy.signal")
    except ImportError:
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
    assert loaded.dtype == torch_float32


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


def test_infer_uses_inference_mode():
    class DummyMelExtractor:
        def __call__(self, waveform):
            assert torch_is_inference_mode_enabled() is True
            return torch_ones((1, 2, 3), dtype=torch_float32)

    class DummyModel:
        def __call__(self, mel, mask=None):
            assert torch_is_inference_mode_enabled() is True
            assert tuple(mel.shape) == (1, 3, 2)
            assert mask is None
            return torch_tensor([[0.0, 1.0]], dtype=torch_float32)

    router_instance = make_router()
    router_instance.mel_extractor = DummyMelExtractor()
    router_instance.model = DummyModel()

    result = router_instance.infer(torch_tensor([0.1, -0.2, 0.3], dtype=torch_float32))

    assert result["label"] == 1


def test_batch_infer_uses_inference_mode():
    class DummyMelExtractor:
        def __call__(self, waveform):
            assert torch_is_inference_mode_enabled() is True
            return torch_ones((1, 2, waveform.shape[-1]), dtype=torch_float32)

    class DummyModel:
        def __call__(self, mel, mask=None):
            assert torch_is_inference_mode_enabled() is True
            assert tuple(mel.shape) == (2, 4, 2)
            assert mask is not None
            assert tuple(mask.shape) == (2, 4)
            return torch_tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch_float32)

    router_instance = make_router()
    router_instance.mel_extractor = DummyMelExtractor()
    router_instance.model = DummyModel()
    router_instance._load_audio_batch = lambda audio_paths: [
        torch_tensor([[0.1, 0.2]], dtype=torch_float32),
        torch_tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch_float32),
    ]

    results = router_instance.batch_infer(["a.wav", "b.wav"])

    assert [result["label"] for result in results] == [0, 1]


def test_batch_infer_vectorized_matches_threshold():
    class DummyMelExtractor:
        def __call__(self, waveform):
            return torch_ones((1, 2, waveform.shape[-1]), dtype=torch_float32)

    logits = torch_tensor(
        [
            [2.0, 0.0],
            [0.0, 0.0],
            [0.0, 2.0],
            [-1.0, 1.0],
        ],
        dtype=torch_float32,
    )

    class DummyModel:
        def __call__(self, mel, mask=None):
            assert tuple(mel.shape) == (4, 4, 2)
            assert mask is not None
            assert tuple(mask.shape) == (4, 4)
            return logits

    router_instance = make_router()
    router_instance.threshold = 0.5
    router_instance.mel_extractor = DummyMelExtractor()
    router_instance.model = DummyModel()
    router_instance._load_audio_batch = lambda audio_paths: [
        torch_tensor([[0.1]], dtype=torch_float32),
        torch_tensor([[0.1, 0.2]], dtype=torch_float32),
        torch_tensor([[0.1, 0.2, 0.3]], dtype=torch_float32),
        torch_tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch_float32),
    ]

    results = router_instance.batch_infer(["a.wav", "b.wav", "c.wav", "d.wav"])
    expected_probs = torch_softmax(logits, dim=-1)[:, 1].tolist()

    assert len(results) == 4
    assert [result["is_degraded"] for result in results] == [False, True, True, True]
    assert [result["label"] for result in results] == [0, 1, 1, 1]
    assert [result["degraded_prob"] for result in results] == expected_probs


def test_batch_infer_empty_input():
    router_instance = make_router()
    router_instance.mel_extractor = object()
    router_instance.model = object()

    assert router_instance.batch_infer([]) == []


def test_batch_infer_all_degraded():
    class DummyMelExtractor:
        def __call__(self, waveform):
            return torch_ones((1, 2, waveform.shape[-1]), dtype=torch_float32)

    class DummyModel:
        def __call__(self, mel, mask=None):
            return torch_tensor(
                [[0.0, 3.0], [0.0, 2.0], [-1.0, 2.0]],
                dtype=torch_float32,
            )

    router_instance = make_router()
    router_instance.threshold = 0.5
    router_instance.mel_extractor = DummyMelExtractor()
    router_instance.model = DummyModel()
    router_instance._load_audio_batch = lambda audio_paths: [
        torch_tensor([[0.1]], dtype=torch_float32),
        torch_tensor([[0.1, 0.2]], dtype=torch_float32),
        torch_tensor([[0.1, 0.2, 0.3]], dtype=torch_float32),
    ]

    results = router_instance.batch_infer(["a.wav", "b.wav", "c.wav"])

    assert [result["is_degraded"] for result in results] == [True, True, True]
    assert [result["label"] for result in results] == [1, 1, 1]


def test_batch_infer_none_degraded():
    class DummyMelExtractor:
        def __call__(self, waveform):
            return torch_ones((1, 2, waveform.shape[-1]), dtype=torch_float32)

    class DummyModel:
        def __call__(self, mel, mask=None):
            return torch_tensor(
                [[3.0, 0.0], [2.0, 0.0], [1.0, -1.0]],
                dtype=torch_float32,
            )

    router_instance = make_router()
    router_instance.threshold = 0.5
    router_instance.mel_extractor = DummyMelExtractor()
    router_instance.model = DummyModel()
    router_instance._load_audio_batch = lambda audio_paths: [
        torch_tensor([[0.1]], dtype=torch_float32),
        torch_tensor([[0.1, 0.2]], dtype=torch_float32),
        torch_tensor([[0.1, 0.2, 0.3]], dtype=torch_float32),
    ]

    results = router_instance.batch_infer(["a.wav", "b.wav", "c.wav"])

    assert [result["is_degraded"] for result in results] == [False, False, False]
    assert [result["label"] for result in results] == [0, 0, 0]
