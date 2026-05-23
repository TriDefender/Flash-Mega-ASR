from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import Any


class Qwen3ASR:
    NAME = "Qwen3-ASR-1.7B"
    HF_REPO_ID = "Qwen/Qwen3-ASR-1.7B"
    DEFAULT_MODEL_DIR = "ckpt/Mega-ASR/Qwen3-ASR-1.7B"

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        *,
        repo_id: str | None = None,
        device_map: str | None = None,
        dtype: Any | None = None,
        attn_implementation: str | None = None,
        backend: str = "auto",
        max_inference_batch_size: int = 32,
        max_new_tokens: int = 2048,
        download_kwargs: dict[str, Any] | None = None,
        **model_kwargs: Any,
    ) -> None:
        resolve_attn_backend = import_module("MegaASR.runtime.backend").resolve_attn_backend
        device_runtime = import_module("MegaASR.runtime.device")
        resolve_device = device_runtime.resolve_device
        resolve_dtype = device_runtime.resolve_dtype
        Qwen3ASRModel = import_module("qwen_asr").Qwen3ASRModel

        repo_id = repo_id or self.HF_REPO_ID
        self.model_path = str(Path(model_path or self.DEFAULT_MODEL_DIR).expanduser())
        if not self._has_local_model(self.model_path):
            self.model_path = self.download_model(
                self.model_path,
                repo_id=repo_id,
                **(download_kwargs or {}),
            )

        if device_map is None:
            device_map = resolve_device()
        if dtype is None:
            dtype = resolve_dtype(device_map)
        if attn_implementation is None:
            attn_implementation = resolve_attn_backend(backend)

        self.backend = backend
        self.device_map = device_map
        self.dtype = dtype
        self.attn_implementation = attn_implementation

        self.model = Qwen3ASRModel.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
            **model_kwargs,
        )

    @staticmethod
    def _has_local_model(model_path: str | os.PathLike[str]) -> bool:
        path = Path(model_path).expanduser()
        return path.is_dir() and (path / "config.json").is_file()

    @staticmethod
    def download_model(
        model_path: str | os.PathLike[str],
        *,
        repo_id: str,
        **snapshot_kwargs: Any,
    ) -> str:
        from huggingface_hub import snapshot_download

        local_dir = Path(model_path).expanduser()
        local_dir.mkdir(parents=True, exist_ok=True)

        return snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            **snapshot_kwargs,
        )

    def infer(
        self,
        audio: Any,
        *,
        language: str | None = None,
        return_objects: bool = False,
        return_time_stamps: bool = False,
        **transcribe_kwargs: Any,
    ) -> str | list[str] | Any:
        input_was_list = isinstance(audio, (list, tuple))
        if isinstance(audio, os.PathLike):
            audio = str(audio)
        elif isinstance(audio, (list, tuple)):
            audio = [str(item) if isinstance(item, os.PathLike) else item for item in audio]

        results = self.model.transcribe(
            audio=audio,
            language=language,
            return_time_stamps=return_time_stamps,
            **transcribe_kwargs,
        )

        if return_objects:
            # Return raw ASRTranscription objects (with .text, .chunks etc.)
            if not input_was_list and isinstance(results, list) and len(results) == 1:
                return results[0]
            return results

        # Normalize to list of strings
        if isinstance(results, list):
            texts = [str(getattr(result, "text", result)).strip() for result in results]
        else:
            texts = [str(getattr(results, "text", results)).strip()]

        # Return a single string for single audio input, list for batch input
        if not input_was_list and len(texts) == 1:
            return texts[0]
        return texts


def get_mega_asr(*args: Any, **kwargs: Any) -> Qwen3ASR:
    return Qwen3ASR(*args, **kwargs)
