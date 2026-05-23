import sys
sys.path.append("src")

import argparse
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_AUDIO = ROOT_DIR / "assets/example/F01_22GC010K_STR.wav"
DEFAULT_CKPT_DIR = ROOT_DIR / "ckpt/Mega-ASR"
DEFAULT_ROUTING = True
DEFAULT_THRESHOLD = 0.5


def str2bool(value):
    if isinstance(value, bool):
        return value
    return value.lower() in ("1", "true", "yes", "y")


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def parse_args():
    parser = argparse.ArgumentParser(description="Mega-ASR inference")
    parser.add_argument("--audio", default=DEFAULT_AUDIO, help="audio file path")
    parser.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR, help="Mega-ASR ckpt root")
    parser.add_argument("--routing", type=str2bool, default=DEFAULT_ROUTING, help="enable router")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="router threshold")
    parser.add_argument("--device_map", default=None, help="transformers device_map")
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1")
    parser.add_argument("--keep_delta_on_gpu", type=str2bool, default=True, help="keep LoRA deltas on GPU")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from MegaASR.model.megaASR import MegaASR

    audio = resolve_path(args.audio)
    ckpt_dir = resolve_path(args.ckpt_dir)
    model = MegaASR(
        model_path=ckpt_dir / "Qwen3-ASR-1.7B",
        lora_dir=ckpt_dir / "mega-asr-merged",
        router_checkpoint=ckpt_dir / "audio_quality_router/best_acc_model.safetensors",
        routing_enabled=args.routing,
        quality_threshold=args.threshold,
        device_map=args.device_map,
        keep_delta_on_gpu=args.keep_delta_on_gpu,
    )
    result = model.infer(audio, return_route=True)
    print(result)

if __name__ == "__main__":
    main()
