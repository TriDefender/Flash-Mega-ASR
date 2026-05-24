from __future__ import annotations

import os
from typing import Any

import torch

from .Qwen3_ASR import Qwen3ASR
from .hub import download_all_assets, resolve_sources
from .router import AudioQualityRouter
from .utils.lora_switch import LoRADeltaSwitch

class MegaASR:
    NAME = "Mega-ASR"

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        *,
        lora_dir: str | os.PathLike[str] | None = None,
        router_checkpoint: str | os.PathLike[str] | None = None,
        ckpt_dir: str | os.PathLike[str] | None = None,
        routing_enabled: bool = True,
        quality_threshold: float = 0.5,
        device_map: str | None = None,
        quality_device: str | None = None,
        max_inference_batch_size: int = 32,
        max_new_tokens: int = 256,
        keep_delta_on_gpu: bool = True,
        **model_kwargs: Any,
    ) -> None:
        # Resolve all asset sources: explicit paths > ckpt_dir > HF Hub
        sources = resolve_sources(
            model_path=model_path,
            lora_dir=lora_dir,
            router_checkpoint=router_checkpoint,
            ckpt_dir=ckpt_dir,
            routing_enabled=routing_enabled,
        )

        self.model_path = sources["model_path"]
        self.lora_dir = sources["lora_dir"]
        self.router_checkpoint = sources["router_checkpoint"]
        self.routing_enabled = routing_enabled

        self.stats = {"total": 0, "use_base": 0, "use_lora": 0}
        self.switch_times: list[dict[str, float | str]] = []

        self.router = None
        if self.routing_enabled and self.router_checkpoint:
            self.router = AudioQualityRouter(
                checkpoint_path=self.router_checkpoint,
                device=quality_device,
                threshold=quality_threshold,
            )

        self.asr = Qwen3ASR(
            model_path=self.model_path,
            device_map=device_map,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
            **model_kwargs,
        )

        self.lora_switch = LoRADeltaSwitch(keep_delta_on_gpu=keep_delta_on_gpu)
        self._load_loras()
        self._set_lora(True)

    @classmethod
    def download(cls, target_dir: str | os.PathLike[str] | None = None) -> str:
        """Download all Mega-ASR assets to a local directory."""
        return download_all_assets(target_dir)

    def _load_loras(self) -> None:
        if not self.lora_dir:
            raise RuntimeError("LoRA adapter directory is required but was not resolved")
        self.lora_switch.add_adapter(
            parent_module=self.asr.model.model,
            adapter_dir=self.lora_dir,
            name="mega_asr_merged_adapter",
        )

    def _set_lora(self, active: bool) -> None:
        elapsed = self.lora_switch.set_active(active)
        if elapsed > 0:
            direction = "base_to_lora" if active else "lora_to_base"
            self.switch_times.append({"direction": direction, "time": elapsed})

    @staticmethod
    def _unwrap_audio(audio: Any) -> Any:
        if isinstance(audio, (list, tuple)) and len(audio) == 1:
            return audio[0]
        return audio

    def _route(self, audio: Any) -> tuple[bool, float | None, str]:
        if self.routing_enabled and self.router is not None:
            is_degraded, degraded_prob = self.router.predict(audio)
            return is_degraded, degraded_prob, "router"

        return True, None, "default"

    def infer(
        self,
        audio: Any,
        *,
        language: str | None = None,
        return_objects: bool = False,
        return_route: bool = False,
        return_time_stamps: bool = False,
        **transcribe_kwargs: Any,
    ) -> Any:
        audio = self._unwrap_audio(audio)
        use_lora, degraded_prob, route_source = self._route(audio)

        self._set_lora(use_lora)
        result = self.asr.infer(
            audio,
            language=language,
            return_objects=return_objects,
            return_time_stamps=return_time_stamps,
            **transcribe_kwargs,
        )

        self.stats["total"] += 1
        if use_lora:
            self.stats["use_lora"] += 1
        else:
            self.stats["use_base"] += 1

        if return_route:
            return {
                "text": result,
                "use_lora": use_lora,
                "degraded_prob": degraded_prob,
                "route_source": route_source,
                "backend": getattr(self.asr, "attn_implementation", None),
                "device": getattr(self.asr, "device_map", None),
            }

        return result

    def infer_with_lora(self, audio: Any, **kwargs: Any) -> Any:
        self._set_lora(True)
        return self.asr.infer(self._unwrap_audio(audio), **kwargs)

    def infer_without_lora(self, audio: Any, **kwargs: Any) -> Any:
        self._set_lora(False)
        return self.asr.infer(self._unwrap_audio(audio), **kwargs)

    @torch.no_grad()
    def batch_infer(self, audios: list[Any], **kwargs: Any) -> list[Any]:
        """True batched inference with grouped LoRA routing.

        Routes all audios in batch, groups by route decision (base vs lora),
        then runs one batched transcribe() call per group.
        """
        audio_paths = [self._unwrap_audio(audio) for audio in audios]

        if (
            self.router is not None
            and self.routing_enabled
            and hasattr(self.router, "batch_predict")
        ):
            route_results = self.router.batch_predict(audio_paths)
            routes = [(is_degraded, degraded_prob, "router") for is_degraded, degraded_prob in route_results]
        else:
            routes = [self._route(audio) for audio in audio_paths]

        base_indices = [idx for idx, route in enumerate(routes) if not route[0]]
        lora_indices = [idx for idx, route in enumerate(routes) if route[0]]

        results: list[Any] = [None] * len(audio_paths)
        groups = [("lora", lora_indices), ("base", base_indices)]
        if not self.lora_switch.active:
            groups = [("base", base_indices), ("lora", lora_indices)]

        for mode, indices in groups:
            if not indices:
                continue

            use_lora = mode == "lora"
            self._set_lora(use_lora)

            group_audios = [audio_paths[idx] for idx in indices]

            if len(group_audios) == 1:
                batch_results = [self.asr.infer(group_audios[0], **kwargs)]
            else:
                batch_results = self.asr.infer(group_audios, **kwargs)
                if not isinstance(batch_results, list):
                    batch_results = [batch_results]

            for result_idx, audio_idx in enumerate(indices):
                results[audio_idx] = batch_results[result_idx]

            if use_lora:
                self.stats["use_lora"] += len(indices)
            else:
                self.stats["use_base"] += len(indices)

        self.stats["total"] += len(audio_paths)
        return results


def get_mega_asr(*args: Any, **kwargs: Any) -> MegaASR:
    return MegaASR(*args, **kwargs)


def get_Mega_ASR(*args: Any, **kwargs: Any) -> MegaASR:
    return get_mega_asr(*args, **kwargs)
