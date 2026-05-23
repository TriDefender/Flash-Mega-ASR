# coding=utf-8
import os
from typing import Optional

import torch
from safetensors.torch import load_file as safe_load_file
from transformers import Trainer


class MegaASRTrainer(Trainer):
    """Trainer for Mega-ASR LoRA SFT."""

    def __init__(self, *args, processor=None, base_model_path: str = "",
                 merged_from_lora_path: str = "", lr_encoder: float = 1e-5,
                 lr_aligner: float = 1e-5, lr_llm: float = 1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.base_model_path = base_model_path
        self.merged_from_lora_path = merged_from_lora_path
        self.lr_encoder = lr_encoder
        self.lr_aligner = lr_aligner
        self.lr_llm = lr_llm

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        dtype = getattr(self.model, "dtype", None)
        if dtype is None:
            return inputs
        for k, v in list(inputs.items()):
            if torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.to(dtype=dtype)
        return inputs

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.thinker.save_pretrained(output_dir, safe_serialization=True)

        if self.processor is not None:
            self.processor.save_pretrained(output_dir)
        self._write_text(output_dir, "base_model.txt", self.base_model_path)
        self._write_text(output_dir, "merged_from_lora.txt", self.merged_from_lora_path)

        for name in ["model.safetensors", "pytorch_model.bin",
                     "model.safetensors.index.json", "pytorch_model.bin.index.json"]:
            path = os.path.join(output_dir, name)
            if os.path.exists(path):
                os.remove(path)

    @staticmethod
    def _write_text(output_dir: str, name: str, text: str):
        if text:
            with open(os.path.join(output_dir, name), "w", encoding="utf-8") as f:
                f.write(text + "\n")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        model = model or self.model
        adapter_path = os.path.join(resume_from_checkpoint, "adapter_model.safetensors")
        if os.path.isfile(adapter_path):
            model.thinker.load_state_dict(safe_load_file(adapter_path), strict=False)
            return
        return super()._load_from_checkpoint(resume_from_checkpoint, model=model)

    @staticmethod
    def _group_name(name: str) -> str:
        if "lora_" not in name:
            return "other"
        if any(x in name for x in ["audio_tower.conv_out", "audio_tower.proj1", "audio_tower.proj2"]):
            return "aligner"
        if "audio_tower.layers." in name:
            return "encoder"
        if "model.layers." in name and "audio_tower.layers." not in name:
            return "llm"
        return "other"

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        groups = {"encoder": [], "aligner": [], "llm": [], "other": []}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                groups[self._group_name(name)].append(param)

        lrs = {"encoder": self.lr_encoder, "aligner": self.lr_aligner,
               "llm": self.lr_llm, "other": self.args.learning_rate}
        optim_groups = [
            {"params": params, "lr": lrs[name], "weight_decay": self.args.weight_decay}
            for name, params in groups.items() if params
        ]

        if self.args.process_index == 0:
            for name, params in groups.items():
                print(f"[optimizer] {name:7s}: {sum(p.numel() for p in params)} params")

        self.optimizer = torch.optim.AdamW(
            optim_groups,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        return self.optimizer
