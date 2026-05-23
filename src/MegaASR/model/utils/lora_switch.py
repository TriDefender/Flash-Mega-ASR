from __future__ import annotations

import json
import os
import time
import warnings
from typing import Any

import torch
from safetensors.torch import load_file as safe_load_file


class LoRADeltaSwitch:
    def __init__(self, keep_delta_on_gpu: bool = True) -> None:
        self.keep_delta_on_gpu = keep_delta_on_gpu
        self.items: list[dict[str, Any]] = []
        self.active = False

    def _load_adapter_state(self, adapter_dir: str | os.PathLike[str]) -> dict[str, torch.Tensor]:
        adapter_dir = str(adapter_dir)
        safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        bin_path = os.path.join(adapter_dir, "adapter_model.bin")

        if os.path.exists(safetensors_path):
            return safe_load_file(safetensors_path)
        return torch.load(bin_path, map_location="cpu")

    def _load_adapter_config(self, adapter_dir: str | os.PathLike[str]) -> dict[str, Any]:
        config_path = os.path.join(str(adapter_dir), "adapter_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_adapter_blocks(self, adapter_dir: str | os.PathLike[str]) -> dict[str, Any]:
        blocks_path = os.path.join(str(adapter_dir), "mega_lora_blocks.json")
        if not os.path.exists(blocks_path):
            return {}

        with open(blocks_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _normalize_module_name(name: str) -> str:
        for prefix in ("base_model.model.",):
            if name.startswith(prefix):
                name = name[len(prefix) :]

        if name.startswith("thinker.layers."):
            name = name.replace("thinker.layers.", "thinker.model.layers.", 1)

        return name

    @staticmethod
    def _module_name_candidates(name: str) -> list[str]:
        candidates = [name]

        if name.startswith("model."):
            candidates.append(name[len("model.") :])

        if name.startswith("thinker.layers."):
            candidates.append(name.replace("thinker.layers.", "thinker.model.layers.", 1))

        if name.startswith("thinker.model."):
            candidates.append(name.replace("thinker.model.", "thinker.", 1))

        return list(dict.fromkeys(candidates))

    @staticmethod
    def _raw_module_name(key: str, marker: str) -> str:
        name = key.split(marker)[0]
        for prefix in ("base_model.model.", "model."):
            if name.startswith(prefix):
                return name[len(prefix) :]
        return name

    def _split_lora_key(self, key: str) -> tuple[str | None, str | None, str | None]:
        raw_key = key
        key = self._normalize_module_name(key)

        for marker in (".lora_A.", ".lora_B."):
            if marker in key:
                module_name = key.split(marker)[0]
                raw_module_name = self._raw_module_name(raw_key, marker)
                kind = "A" if marker == ".lora_A." else "B"
                return module_name, raw_module_name, kind

        return None, None, None

    def add_adapter(
        self,
        parent_module: torch.nn.Module,
        adapter_dir: str | os.PathLike[str],
        name: str,
        strip_prefixes: list[str] | None = None,
    ) -> None:
        config = self._load_adapter_config(adapter_dir)
        state = self._load_adapter_state(adapter_dir)
        blocks = self._load_adapter_blocks(adapter_dir)

        lora_alpha = config.get("lora_alpha", 1)
        rank = config.get("r")
        alpha_pattern = config.get("alpha_pattern") or {}
        rank_pattern = config.get("rank_pattern") or {}
        fan_in_fan_out = bool(config.get("fan_in_fan_out", False))

        module_dict = dict(parent_module.named_modules())
        grouped: dict[str, dict[str, torch.Tensor]] = {}

        for key, tensor in state.items():
            module_name, raw_module_name, kind = self._split_lora_key(key)
            if module_name is None or raw_module_name is None or kind is None:
                continue

            if strip_prefixes:
                for prefix in strip_prefixes:
                    if module_name.startswith(prefix):
                        module_name = module_name[len(prefix) :]
                    if raw_module_name.startswith(prefix):
                        raw_module_name = raw_module_name[len(prefix) :]

            matched_name = None
            for candidate in self._module_name_candidates(module_name):
                if candidate in module_dict:
                    matched_name = candidate
                    break

            target_name = matched_name or module_name
            group_key = f"{target_name}\0{raw_module_name}"
            item = grouped.setdefault(
                group_key,
                {
                    "target_module_name": target_name,
                    "raw_module_name": raw_module_name,
                },
            )
            item[kind] = tensor.cpu()

        loaded = 0
        missing = []

        for pair in grouped.values():
            if "A" not in pair or "B" not in pair:
                continue
            module_name = pair["target_module_name"]
            raw_module_name = pair["raw_module_name"]
            if module_name not in module_dict:
                missing.append(module_name)
                continue

            module = module_dict[module_name]
            if not hasattr(module, "weight"):
                missing.append(module_name)
                continue

            weight = module.weight
            a_matrix = pair["A"].to(device=weight.device, dtype=torch.float32)
            b_matrix = pair["B"].to(device=weight.device, dtype=torch.float32)
            module_blocks = blocks.get(raw_module_name) or blocks.get(module_name)

            if module_blocks:
                deltas = []
                for block in module_blocks:
                    start = int(block["start"])
                    end = int(block["end"])
                    block_rank = int(block.get("rank", end - start))
                    block_alpha = int(block.get("alpha", block_rank))
                    delta = torch.matmul(b_matrix[:, start:end], a_matrix[start:end])
                    delta = delta * (float(block_alpha) / float(block_rank))
                    if fan_in_fan_out:
                        delta = delta.T
                    deltas.append(delta)
            else:
                adapter_rank = rank_pattern.get(raw_module_name, rank_pattern.get(module_name, rank))
                if adapter_rank is None:
                    adapter_rank = a_matrix.shape[0]
                adapter_alpha = alpha_pattern.get(
                    raw_module_name,
                    alpha_pattern.get(module_name, lora_alpha),
                )
                scaling = float(adapter_alpha) / float(adapter_rank)
                delta = torch.matmul(b_matrix, a_matrix) * scaling
                if fan_in_fan_out:
                    delta = delta.T
                deltas = [delta]

            for delta in deltas:
                if delta.shape != weight.shape:
                    try:
                        delta = delta.reshape(weight.shape)
                    except Exception:
                        missing.append(
                            f"{module_name}: delta shape {tuple(delta.shape)} != "
                            f"weight shape {tuple(weight.shape)}"
                        )
                        continue

                delta = delta.to(dtype=weight.dtype)
                if self.keep_delta_on_gpu:
                    delta = delta.to(device=weight.device)
                else:
                    delta = delta.cpu()

                self.items.append(
                    {
                        "name": name,
                        "module_name": module_name,
                        "weight": weight,
                        "delta": delta,
                    }
                )
                loaded += 1

        if missing:
            warnings.warn(
                f"LoRA adapter {name} loaded {loaded} deltas, "
                f"missing {len(missing)} modules. Examples: {missing[:5]}",
                stacklevel=2,
            )

    @torch.no_grad()
    def set_active(self, active: bool) -> float:
        if self.active == active:
            return 0.0

        start = time.perf_counter()
        sign = 1.0 if active else -1.0

        for item in self.items:
            weight = item["weight"]
            delta = item["delta"]
            if delta.device != weight.device:
                delta = delta.to(device=weight.device)
            weight.data.add_(delta, alpha=sign)

        self.active = active
        return time.perf_counter() - start
