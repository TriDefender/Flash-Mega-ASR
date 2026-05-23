from __future__ import annotations

import math
import os
import json
from pathlib import Path
from typing import Any

import soundfile as sf  # pyright: ignore[reportMissingImports]
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as safe_load_file
from safetensors import safe_open
from scipy.signal import resample_poly

from .utils.audio_quality import LogMelSpectrogram, create_audio_quality_model

class AudioQualityRouter:
    DEFAULT_CHECKPOINT = "ckpt/Mega-ASR/audio_quality_router/best_acc_model.safetensors"

    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str] | None = None,
        *,
        device: str | None = None,
        threshold: float = 0.5,
        sample_rate: int = 16000,
    ) -> None:
        self.checkpoint_path = str(
            Path(checkpoint_path or self.DEFAULT_CHECKPOINT).expanduser()
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.sample_rate = sample_rate

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

        mel_extractor = LogMelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=config.get("n_mels", 80),
        ).to(self.device)
        mel_extractor.eval()

        return model, mel_extractor

    def _load_audio(self, audio_path: str | os.PathLike[str]) -> torch.Tensor:
        audio_np, sr = sf.read(str(audio_path), always_2d=True)
        audio_np = audio_np.mean(axis=1)

        if sr != self.sample_rate:
            gcd = math.gcd(sr, self.sample_rate)
            audio_np = resample_poly(
                audio_np,
                self.sample_rate // gcd,
                sr // gcd,
            )

        waveform = torch.Tensor(audio_np).float().unsqueeze(0)

        return waveform.to(self.device)

    def _load_audio_batch(
        self, audio_paths: list[str | os.PathLike[str]]
    ) -> list[torch.Tensor]:
        """Load and resample multiple audio files to waveforms."""
        return [self._load_audio(p) for p in audio_paths]

    @torch.no_grad()
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

    @torch.no_grad()
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
        mels = [self.mel_extractor(w) for w in waveforms]

        max_t = max(m.shape[-1] for m in mels)

        batch_size = len(mels)
        n_mels = mels[0].shape[1]
        padded = mels[0].new_zeros((batch_size, n_mels, max_t))
        masks = padded.new_ones((batch_size, max_t)).bool()

        for i, mel in enumerate(mels):
            mel_no_batch = mel.squeeze(0)
            t = mel_no_batch.shape[-1]
            padded[i, :, :t] = mel_no_batch[:, :t]
            if t < max_t:
                masks[i, t:] = False

        mels_batch = padded.transpose(1, 2)

        logits = self.model(mels_batch, mask=masks)
        probs = F.softmax(logits, dim=-1)

        results = []
        for i in range(batch_size):
            degraded_prob = float(probs[i, 1].item())
            is_degraded = degraded_prob >= self.threshold
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
