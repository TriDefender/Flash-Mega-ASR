from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

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
        import_module("torchaudio")
    except ImportError:
        torchaudio_module = ModuleType("torchaudio")
        setattr(
            torchaudio_module,
            "transforms",
            SimpleNamespace(
                Resample=lambda orig_freq, new_freq: SimpleNamespace(
                    to=lambda device: (lambda waveform: waveform)
                )
            ),
        )
        sys.modules.setdefault("torchaudio", torchaudio_module)

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
    instance._resamplers = {}
    return instance


class DummyBatchMelExtractor:
    def __init__(self, *, n_mels=2, hop_length=1):
        self.n_mels = n_mels
        self.mel_transform = SimpleNamespace(hop_length=hop_length)

    def __call__(self, waveform):
        batch_size = waveform.shape[0]
        time_steps = waveform.shape[-1] + 1
        return torch_ones((batch_size, 1, self.n_mels, time_steps), dtype=torch_float32)


def test_load_audio_returns_correct_shape(monkeypatch):
    mono_signal = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    stereo_signal = np.stack([mono_signal, mono_signal], axis=1)

    monkeypatch.setattr(router.sf, "read", lambda *args, **kwargs: (stereo_signal, 16000))

    loaded = make_router()._load_audio("dummy.wav")

    assert tuple(loaded.shape) == (1, mono_signal.shape[0])
    assert loaded.dtype == torch_float32


def test_load_audio_caches_resampler(monkeypatch):
    mono_signal = np.array([0.2, -0.1, 0.4], dtype=np.float32)
    stereo_signal = np.stack([mono_signal, mono_signal], axis=1)
    init_calls = []
    apply_calls = []

    class FakeResample:
        def __init__(self, orig_freq, new_freq):
            init_calls.append((orig_freq, new_freq))

        def to(self, device):
            self.device = device
            return self

        def __call__(self, waveform):
            apply_calls.append((self.device, tuple(waveform.shape)))
            return waveform

    monkeypatch.setattr(router.sf, "read", lambda *args, **kwargs: (stereo_signal, 8000))
    monkeypatch.setattr(router.torchaudio.transforms, "Resample", FakeResample)

    router_instance = make_router()
    router_instance._load_audio("first.wav")
    router_instance._load_audio("second.wav")

    assert init_calls == [(8000, 16000)]
    assert len(apply_calls) == 2
    assert set(router_instance._resamplers) == {8000}


def test_load_audio_no_resample_when_16k(monkeypatch):
    mono_signal = np.array([0.2, -0.1, 0.4], dtype=np.float32)
    stereo_signal = np.stack([mono_signal, mono_signal], axis=1)
    called = False

    class FakeResample:
        def __init__(self, orig_freq, new_freq):
            nonlocal called
            called = True

        def to(self, device):
            return self

        def __call__(self, waveform):
            nonlocal called
            called = True
            return waveform

    monkeypatch.setattr(router.sf, "read", lambda *args, **kwargs: (stereo_signal, 16000))
    monkeypatch.setattr(router.torchaudio.transforms, "Resample", FakeResample)

    router_instance = make_router()
    router_instance._load_audio("dummy.wav")

    assert called is False
    assert router_instance._resamplers == {}


def test_load_audio_different_sample_rates(monkeypatch):
    mono_signal = np.array([0.2, -0.1, 0.4], dtype=np.float32)
    stereo_signal = np.stack([mono_signal, mono_signal], axis=1)
    sample_rates = iter([8000, 44100])
    init_calls = []

    class FakeResample:
        def __init__(self, orig_freq, new_freq):
            init_calls.append((orig_freq, new_freq))

        def to(self, device):
            return self

        def __call__(self, waveform):
            return waveform

    monkeypatch.setattr(router.sf, "read", lambda *args, **kwargs: (stereo_signal, next(sample_rates)))
    monkeypatch.setattr(router.torchaudio.transforms, "Resample", FakeResample)

    router_instance = make_router()
    router_instance._load_audio("8k.wav")
    router_instance._load_audio("44k.wav")

    assert init_calls == [(8000, 16000), (44100, 16000)]
    assert set(router_instance._resamplers) == {8000, 44100}


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
    class DummyMelExtractor(DummyBatchMelExtractor):
        def __call__(self, waveform):
            assert torch_is_inference_mode_enabled() is True
            return super().__call__(waveform)

    class DummyModel:
        def __call__(self, mel, mask=None):
            assert torch_is_inference_mode_enabled() is True
            assert tuple(mel.shape) == (2, 5, 2)
            assert mask is not None
            assert tuple(mask.shape) == (2, 5)
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
            assert tuple(mel.shape) == (4, 5, 2)
            assert mask is not None
            assert tuple(mask.shape) == (4, 5)
            return logits

    router_instance = make_router()
    router_instance.threshold = 0.5
    router_instance.mel_extractor = DummyBatchMelExtractor()
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
    class DummyModel:
        def __call__(self, mel, mask=None):
            return torch_tensor(
                [[0.0, 3.0], [0.0, 2.0], [-1.0, 2.0]],
                dtype=torch_float32,
            )

    router_instance = make_router()
    router_instance.threshold = 0.5
    router_instance.mel_extractor = DummyBatchMelExtractor()
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
    class DummyModel:
        def __call__(self, mel, mask=None):
            return torch_tensor(
                [[3.0, 0.0], [2.0, 0.0], [1.0, -1.0]],
                dtype=torch_float32,
            )

    router_instance = make_router()
    router_instance.threshold = 0.5
    router_instance.mel_extractor = DummyBatchMelExtractor()
    router_instance.model = DummyModel()
    router_instance._load_audio_batch = lambda audio_paths: [
        torch_tensor([[0.1]], dtype=torch_float32),
        torch_tensor([[0.1, 0.2]], dtype=torch_float32),
        torch_tensor([[0.1, 0.2, 0.3]], dtype=torch_float32),
    ]

    results = router_instance.batch_infer(["a.wav", "b.wav", "c.wav"])

    assert [result["is_degraded"] for result in results] == [False, False, False]
    assert [result["label"] for result in results] == [0, 0, 0]


def test_batch_infer_calls_mel_extractor_once():
    call_shapes = []

    class DummyMelExtractor(DummyBatchMelExtractor):
        def __call__(self, waveform):
            call_shapes.append(tuple(waveform.shape))
            return super().__call__(waveform)

    class DummyModel:
        def __call__(self, mel, mask=None):
            assert tuple(mel.shape) == (3, 5, 2)
            assert mask is not None
            assert tuple(mask.shape) == (3, 5)
            return torch_tensor(
                [[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]],
                dtype=torch_float32,
            )

    router_instance = make_router()
    router_instance.mel_extractor = DummyMelExtractor()
    router_instance.model = DummyModel()
    router_instance._load_audio_batch = lambda audio_paths: [
        torch_tensor([[0.1, 0.2]], dtype=torch_float32),
        torch_tensor([[0.1, 0.2, 0.3]], dtype=torch_float32),
        torch_tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch_float32),
    ]

    router_instance.batch_infer(["a.wav", "b.wav", "c.wav"])

    assert call_shapes == [(3, 1, 4)]


def test_batch_infer_mixed_lengths():
    expected_mask_lengths = [2, 4, 6]

    class DummyMelExtractor(DummyBatchMelExtractor):
        def __init__(self):
            super().__init__(hop_length=2)

    class DummyModel:
        def __call__(self, mel, mask=None):
            assert tuple(mel.shape) == (3, 11, 2)
            assert mask is not None
            assert mask.sum(dim=1).tolist() == expected_mask_lengths
            return torch_tensor(
                [[2.0, 0.0], [0.0, 2.0], [-1.0, 1.0]],
                dtype=torch_float32,
            )

    router_instance = make_router()
    router_instance.mel_extractor = DummyMelExtractor()
    router_instance.model = DummyModel()
    router_instance._load_audio_batch = lambda audio_paths: [
        torch_tensor([[0.1, 0.2]], dtype=torch_float32),
        torch_tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=torch_float32),
        torch_tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]], dtype=torch_float32),
    ]

    results = router_instance.batch_infer(["short.wav", "medium.wav", "long.wav"])

    assert len(results) == 3
    assert [result["label"] for result in results] == [0, 1, 1]
    assert [result["is_degraded"] for result in results] == [False, True, True]


def test_batch_infer_single_item_fallback():
    class DummyMelExtractor(DummyBatchMelExtractor):
        pass

    class DummyModel:
        def __call__(self, mel, mask=None):
            assert tuple(mel.shape) == (1, 4, 2)
            assert mask is not None
            assert tuple(mask.shape) == (1, 4)
            assert mask[0].tolist() == [True, True, True, True]
            return torch_tensor([[0.0, 2.0]], dtype=torch_float32)

    router_instance = make_router()
    router_instance.mel_extractor = DummyMelExtractor()
    router_instance.model = DummyModel()
    router_instance._load_audio_batch = lambda audio_paths: [
        torch_tensor([[0.1, 0.2, 0.3]], dtype=torch_float32),
    ]

    results = router_instance.batch_infer(["only.wav"])

    assert len(results) == 1
    assert results[0]["label"] == 1
