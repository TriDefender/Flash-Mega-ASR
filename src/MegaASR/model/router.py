from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, cast

import soundfile as sf  # pyright: ignore[reportMissingImports]
import torch
import torch.nn.functional as F
import torchaudio  # pyright: ignore[reportMissingImports]
from safetensors.torch import load_file as safe_load_file
from safetensors import safe_open

from .utils.audio_quality import LogMelSpectrogram, create_audio_quality_model

class AudioQualityRouter:
    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str],
        *,
        device: str | None = None,
        threshold: float = 0.5,
        sample_rate: int = 16000,
    ) -> None:
        self.checkpoint_path = str(Path(checkpoint_path).expanduser())
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.sample_rate = sample_rate
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

        self.model, self.mel_extractor = self._load_model()

    def _load_model(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        checkpoint_path = Path(self.checkpoint_path)
        if checkpoint_path.suffix == ".safetensors":
            with safe_open(str(checkpoint_path), framework="pt", device="cpu") as f:
                metadata = f.metadata()
            checkpoint_config = json.loads(metadata.get("config", "{}"))
            config = checkpoint_config.get("model", {})
            state_dict = safe_load_file(str(checkpoint_path), device=self.device)
        elif checkpoint_path.suffix in (".pt", ".pth", ".bin"):
            raise ValueError(
                f"Non-safetensors checkpoint '{checkpoint_path}' is not supported for security. "
                "Convert to safetensors first: `safetensors.torch.save_file(state_dict, path)`"
            )
        else:
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path.suffix}. Use .safetensors")

        model = create_audio_quality_model(config)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        if self.device == "cuda" and hasattr(torch, "compile"):
            try:
                model = cast(torch.nn.Module, torch.compile(model, mode="reduce-overhead"))
            except Exception:
                pass

        mel_extractor = LogMelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=config.get("n_mels", 80),
        ).to(self.device)
        mel_extractor.eval()

        return model, mel_extractor

    def _load_audio(self, audio_path: str | os.PathLike[str]) -> torch.Tensor:
        audio_np, sr = sf.read(str(audio_path), always_2d=True)
        audio_np = audio_np.mean(axis=1)
        waveform = getattr(torch, "as_tensor")(audio_np, dtype=getattr(torch, "float32")).unsqueeze(0)

        if sr != self.sample_rate:
            if sr not in self._resamplers:
                self._resamplers[sr] = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=self.sample_rate,
                ).to(self.device)
            waveform = self._resamplers[sr](waveform.to(self.device))

        return waveform.to(self.device)

    def _load_audio_batch(
        self, audio_paths: list[str | os.PathLike[str]]
    ) -> list[torch.Tensor]:
        """Load and resample multiple audio files to waveforms."""
        return [self._load_audio(p) for p in audio_paths]

    @torch.inference_mode()
    def infer(self, audio_path: str | os.PathLike[str] | torch.Tensor) -> dict[str, Any]:
        if isinstance(audio_path, torch.Tensor):
            waveform = audio_path.to(self.device)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.ndim != 2:
                raise ValueError("waveform tensor must have shape [time] or [batch, time]")
        else:
            waveform = self._load_audio(audio_path)

        mel = self.mel_extractor(waveform)
        mel = mel.squeeze(0).transpose(0, 1).unsqueeze(0)

        logits = self.model(mel, mask=None)
        probs = F.softmax(logits, dim=-1)
        degraded_prob = float(probs[0, 1].item())
        is_degraded = degraded_prob >= self.threshold

        return {
            "is_degraded": is_degraded,
            "degraded_prob": degraded_prob,
            "label": int(is_degraded),
        }

    @torch.inference_mode()
    def batch_infer(
        self, audio_paths: list[str | os.PathLike[str]]
    ) -> list[dict[str, Any]]:
        """Batch inference for multiple audio files.

        Loads all audio, computes mel features, pads to common length,
        and runs a single forward pass through the model.
        """
        if not audio_paths:
            return []

        waveforms = self._load_audio_batch(audio_paths)
        waveform_lengths = [waveform.shape[-1] for waveform in waveforms]
        max_wave_len = max(waveform_lengths)

        batch_size = len(waveforms)
        padded_waveforms = waveforms[0].new_zeros((batch_size, 1, max_wave_len))
        for i, waveform in enumerate(waveforms):
            padded_waveforms[i, :, : waveform.shape[-1]] = waveform

        mels_batch = self.mel_extractor(padded_waveforms)
        if mels_batch.ndim == 4:
            mels_batch = mels_batch.squeeze(1)

        hop_length = int(getattr(self.mel_extractor.mel_transform, "hop_length", 160))
        mel_lengths = [(length // hop_length) + 1 for length in waveform_lengths]
        max_mel_t = mels_batch.shape[-1]

        masks = padded_waveforms.new_ones((batch_size, max_mel_t)).bool()
        for i, mel_len in enumerate(mel_lengths):
            if mel_len < max_mel_t:
                masks[i, mel_len:] = False

        mels_batch = mels_batch.transpose(1, 2)

        logits = self.model(mels_batch, mask=masks)
        probs = F.softmax(logits, dim=-1)

        degraded_probs = probs[:, 1]
        is_degraded_mask = degraded_probs >= self.threshold
        degraded_probs_list = degraded_probs.cpu().tolist()

        results = []
        for i in range(batch_size):
            is_degraded = bool(is_degraded_mask[i].item())
            degraded_prob = float(degraded_probs_list[i])
            results.append(
                {
                    "is_degraded": is_degraded,
                    "degraded_prob": degraded_prob,
                    "label": int(is_degraded),
                }
            )

        return results

    def predict(self, audio_path: str | os.PathLike[str]) -> tuple[bool, float]:
        result = self.infer(audio_path)
        return result["is_degraded"], result["degraded_prob"]

    def batch_predict(
        self, audio_paths: list[str | os.PathLike[str]]
    ) -> list[tuple[bool, float]]:
        """Batch predict - returns list of (is_degraded, degraded_prob) tuples."""
        results = self.batch_infer(audio_paths)
        return [(r["is_degraded"], r["degraded_prob"]) for r in results]


def get_router(*args: Any, **kwargs: Any) -> AudioQualityRouter:
    return AudioQualityRouter(*args, **kwargs)
