"""Standalone inference script for Mega-ASR.

Usage:
    python infer.py --audio audio.wav
    python infer.py --audio audio.wav --ckpt_dir ckpt/Mega-ASR
    python infer.py --audio audio.wav --no-routing
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mega-ASR inference")
    parser.add_argument(
        "--audio",
        required=True,
        help="audio file path",
    )
    parser.add_argument(
        "--ckpt_dir",
        default=str(ROOT_DIR / "ckpt" / "Mega-ASR"),
        help="Mega-ASR checkpoint root dir",
    )
    parser.add_argument(
        "--routing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable or disable the audio-quality router",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="router threshold")
    parser.add_argument("--device_map", default=None, help="transformers device_map")
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from MegaASR.model.megaASR import MegaASR

    ckpt_dir = Path(args.ckpt_dir)
    model = MegaASR(
        model_path=ckpt_dir / "Qwen3-ASR-1.7B",
        lora_dir=ckpt_dir / "mega-asr-merged",
        router_checkpoint=ckpt_dir / "audio_quality_router" / "best_acc_model.safetensors",
        routing_enabled=args.routing,
        quality_threshold=args.threshold,
        device_map=args.device_map,
    )
    result = model.infer(args.audio, return_route=True)
    print(result)


if __name__ == "__main__":
    main()
