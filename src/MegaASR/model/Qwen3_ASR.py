from __future__ import annotations

import os
from importlib import import_module
from typing import Any


class Qwen3ASR:
    NAME = "Qwen3-ASR-1.7B"
    HF_REPO_ID = "Qwen/Qwen3-ASR-1.7B"

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
        **model_kwargs: Any,
    ) -> None:
        resolve_attn_backend = import_module("MegaASR.runtime.backend").resolve_attn_backend
        device_runtime = import_module("MegaASR.runtime.device")
        resolve_device = device_runtime.resolve_device
        resolve_dtype = device_runtime.resolve_dtype
        Qwen3ASRModel = import_module("qwen_asr").Qwen3ASRModel

        # model_path can be a local directory OR a HuggingFace repo ID.
        # Default to the HF repo ID so from_pretrained uses the HF cache.
        self.model_path = str(model_path) if model_path else (repo_id or self.HF_REPO_ID)

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
