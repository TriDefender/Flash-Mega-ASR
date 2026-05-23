#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert LibriSpeech metadata JSONL to Mega-ASR SFT JSONL.

Input example:
{
  "index": 0,
  "audio_path": ".../xxx.flac",
  "answer": "THE TRANSCRIPT TEXT",
  "subset": "test_clean",
  "task_type": "understanding"
}

Output example:
{
  "audio": ".../xxx.flac",
  "text": "language English<asr_text>THE TRANSCRIPT TEXT",
  "prompt": ""
}
"""

import argparse
import json
from pathlib import Path


def normalize_text(text: str, case: str) -> str:
    text = " ".join(str(text).strip().split())

    if case == "lower":
        return text.lower()
    if case == "upper":
        return text.upper()
    if case == "none":
        return text

    raise ValueError(f"Unknown case mode: {case}")


def convert_one(item: dict, text_case: str = "none", language: str = "English") -> dict:
    audio = item.get("audio_path") or item.get("audio")
    answer = item.get("answer") or item.get("text")

    if not audio:
        raise ValueError(f"Missing audio_path/audio in item: {item}")
    if answer is None:
        raise ValueError(f"Missing answer/text in item: {item}")

    answer = normalize_text(answer, text_case)

    return {
        "audio": audio,
        "text": f"language {language}<asr_text>{answer}",
        "prompt": "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument(
        "--text_case",
        type=str,
        default="none",
        choices=["none", "lower", "upper"],
        help="LibriSpeech transcripts are usually uppercase. Use none to preserve original text.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            out = convert_one(item, text_case=args.text_case, language=args.language)
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1

    print(f"[done] converted {count} samples")
    print(f"[output] {output_path}")


if __name__ == "__main__":
    main()
