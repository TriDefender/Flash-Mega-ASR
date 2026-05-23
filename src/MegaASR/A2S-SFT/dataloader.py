# coding=utf-8
from dataclasses import dataclass
from typing import Any, Dict, List

import librosa
import torch
from datasets import load_dataset


def read_audio(path: str, sr: int = 16000):
    return librosa.load(path, sr=sr, mono=True)[0]


def audio_messages(prompt: str):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": None}]},
    ]


@dataclass
class Qwen3ASRCollator:
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [x.get("prompt", "") for x in features]
        targets = [x["text"] for x in features]
        audios = [read_audio(x["audio"], self.sampling_rate) for x in features]

        prefixes = [
            self.processor.apply_chat_template(
                [audio_messages(p)],
                add_generation_prompt=True,
                tokenize=False,
            )[0]
            for p in prompts
        ]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [p + t + eos for p, t in zip(prefixes, targets)]

        batch = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_batch = self.processor(
            text=prefixes,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        labels = batch["input_ids"].clone()
        prefix_lens = prefix_batch["attention_mask"].sum(dim=1)
        full_lens = batch["attention_mask"].sum(dim=1)

        seq_len = labels.size(1)
        padding_side = getattr(self.processor.tokenizer, "padding_side", "right")

        for i, prefix_len in enumerate(prefix_lens):
            start = seq_len - int(full_lens[i]) if padding_side == "left" else 0
            labels[i, start:start + int(prefix_len)] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        batch["labels"] = labels
        return batch


def build_datasets(train_file: str, eval_file: str = ""):
    files = {"train": train_file}
    if eval_file:
        files["validation"] = eval_file
    return load_dataset("json", data_files=files)
