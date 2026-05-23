"""Flash-Mega-ASR CLI - Fast batched ASR with flash-attention and LoRA routing."""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

logger = logging.getLogger(__name__)

ATTN_CHOICES = ["auto", "flash_attention_2", "sdpa", "eager"]
DTYPE_CHOICES = ["auto", "bfloat16", "float16", "float32"]


def str2bool(value: str | bool) -> bool:
    """Convert a CLI bool-like value to bool."""
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="flash-mega-asr",
        description="Flash-attention optimized Mega-ASR for robust speech recognition",
    )

    parser.add_argument("--file-name", type=str, help="Single audio file path or URL")
    parser.add_argument("--files", nargs="+", type=str, help="Multiple audio files for batch processing")
    parser.add_argument("--batch-size", type=int, default=24, help="Batch size for grouped inference (default: 24)")
    parser.add_argument("--device", type=str, default=None, help='Device: "0" for CUDA:0, "mps", "cpu" (default: auto)')
    parser.add_argument(
        "--attn",
        type=str,
        default="auto",
        choices=ATTN_CHOICES,
        help="Attention backend (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=DTYPE_CHOICES,
        help="Data type (default: auto)",
    )
    parser.add_argument(
        "--routing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable LoRA routing (default: enabled)",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Router threshold for LoRA activation (default: 0.5)")
    parser.add_argument("--timestamps", action="store_true", help="Enable timestamp output")
    parser.add_argument("--model-path", type=str, default=None, help="Base model path")
    parser.add_argument("--lora-dir", type=str, default=None, help="LoRA adapter directory")
    parser.add_argument("--router-checkpoint", type=str, default=None, help="Router checkpoint path")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Mega-ASR checkpoint root dir (convenience)")
    parser.add_argument("--transcript-path", type=str, default="output.json", help="Output JSON path (default: output.json)")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens (default: 256)")
    parser.add_argument("--backend-report", action="store_true", help="Print resolved backend/device/dtype info")
    parser.add_argument("--language", type=str, default="auto", help="Language hint (default: auto-detect)")

    args = parser.parse_args(argv)

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0.0 and 1.0")
    if args.file_name and args.files:
        parser.error("use either --file-name or --files, not both")
    if not args.file_name and not args.files and not args.backend_report:
        parser.error("--file-name or --files is required")

    return args


def resolve_ckpt_paths(args: argparse.Namespace) -> dict[str, str | None]:
    """Resolve checkpoint paths from --ckpt-dir or individual flags."""
    if args.ckpt_dir:
        ckpt_dir = Path(args.ckpt_dir).expanduser()
        return {
            "model_path": args.model_path or str(ckpt_dir / "Qwen3-ASR-1.7B"),
            "lora_dir": args.lora_dir or str(ckpt_dir / "mega-asr-merged"),
            "router_checkpoint": (
                args.router_checkpoint
                or str(ckpt_dir / "audio_quality_router" / "best_acc_model.safetensors")
                if args.routing
                else None
            ),
        }

    from MegaASR.model.megaASR import MegaASR

    return {
        "model_path": args.model_path or MegaASR.DEFAULT_MODEL_DIR,
        "lora_dir": args.lora_dir or MegaASR.DEFAULT_LORA_DIR,
        "router_checkpoint": (args.router_checkpoint or MegaASR.DEFAULT_ROUTER_CHECKPOINT) if args.routing else None,
    }


def resolve_dtype_arg(dtype_name: str, device: str) -> Any:
    """Resolve the dtype argument to a torch dtype."""
    if dtype_name != "auto":
        import torch

        return getattr(torch, dtype_name)

    from MegaASR.runtime.device import resolve_dtype

    return resolve_dtype(device)


def normalize_language(language: str | None) -> str | None:
    """Normalize auto-detect hints to None for model calls."""
    if language is None:
        return None

    normalized = language.strip().lower()
    if normalized in {"", "auto", "auto-detect", "autodetect"}:
        return None
    return language


def _coerce_chunk_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, tuple):
        return [_coerce_chunk_value(item) for item in value]
    if isinstance(value, list):
        return [_coerce_chunk_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _coerce_chunk_value(item) for key, item in value.items()}
    return value


def _extract_text_and_chunks(result: Any) -> tuple[str, list[dict[str, Any]] | None]:
    if isinstance(result, dict):
        text = str(result.get("text", "")).strip()
        chunks = result.get("chunks")
    else:
        text = str(getattr(result, "text", result)).strip()
        chunks = getattr(result, "chunks", None)

    normalized_chunks: list[dict[str, Any]] | None = None
    if chunks is not None:
        normalized_chunks = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                normalized_chunks.append({str(key): _coerce_chunk_value(value) for key, value in chunk.items()})
            elif hasattr(chunk, "__dict__"):
                normalized_chunks.append(
                    {str(key): _coerce_chunk_value(value) for key, value in vars(chunk).items() if not key.startswith("_")}
                )
            else:
                normalized_chunks.append({"value": _coerce_chunk_value(chunk)})

    return text, normalized_chunks


def build_single_result(audio_input: str, raw_result: Any, include_route: bool) -> dict[str, Any]:
    """Normalize a single inference result into JSON-serializable output."""
    payload = raw_result if isinstance(raw_result, dict) else {"text": raw_result}
    text, chunks = _extract_text_and_chunks(payload)

    result: dict[str, Any] = {
        "file": audio_input,
        "text": text,
    }
    if chunks is not None:
        result["chunks"] = chunks
    if include_route:
        for key in ("use_lora", "degraded_prob", "route_source", "backend", "device"):
            if key in payload:
                result[key] = _coerce_chunk_value(payload[key])
    return result


def build_batch_results(audio_inputs: list[str], raw_results: list[Any]) -> list[dict[str, Any]]:
    """Normalize batched inference results into JSON-serializable output."""
    results: list[dict[str, Any]] = []
    for audio_input, raw_result in zip(audio_inputs, raw_results, strict=True):
        text, chunks = _extract_text_and_chunks(raw_result)
        item: dict[str, Any] = {
            "file": audio_input,
            "text": text,
        }
        if chunks is not None:
            item["chunks"] = chunks
        results.append(item)
    return results


def build_metadata(args: argparse.Namespace, *, backend: str, device: str, dtype: Any, elapsed: float) -> dict[str, Any]:
    """Build output metadata."""
    return {
        "backend": backend,
        "device": device,
        "dtype": str(dtype),
        "routing_enabled": args.routing,
        "batch_size": args.batch_size,
        "threshold": args.threshold,
        "language": args.language,
        "timestamps": args.timestamps,
        "elapsed_s": round(elapsed, 4),
    }


def dump_output(output: dict[str, Any], transcript_path: str) -> None:
    """Persist JSON output to disk."""
    output_path = Path(transcript_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def report_backend(args: argparse.Namespace, *, backend: str, device: str, dtype: Any) -> None:
    """Print resolved runtime information."""
    from MegaASR.runtime.device import get_device_info

    info = get_device_info()
    print(f"Backend:  {backend}")
    print(f"Device:   {device}")
    print(f"Dtype:    {dtype}")
    print(f"Routing:  {args.routing}")
    print(f"Device info: {json.dumps(info, indent=2)}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)

    from MegaASR.model.megaASR import MegaASR
    from MegaASR.runtime.backend import resolve_attn_backend
    from MegaASR.runtime.device import resolve_device

    paths = resolve_ckpt_paths(args)
    device = resolve_device(args.device)
    dtype = resolve_dtype_arg(args.dtype, device)
    backend = resolve_attn_backend(args.attn)

    if args.backend_report:
        report_backend(args, backend=backend, device=device, dtype=dtype)
        if not args.file_name and not args.files:
            return

    language = normalize_language(args.language)
    audio_inputs = args.files or [args.file_name]
    return_objects = args.timestamps

    model = MegaASR(
        model_path=paths["model_path"],
        lora_dir=paths["lora_dir"],
        router_checkpoint=paths["router_checkpoint"],
        routing_enabled=args.routing,
        quality_threshold=args.threshold,
        device_map=device,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        attn_implementation=backend,
        dtype=dtype,
    )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(style="yellow1", pulse_style="white"),
        TimeElapsedColumn(),
    ) as progress:
        progress.add_task("[yellow]Transcribing...", total=None)
        start = time.perf_counter()

        if len(audio_inputs) == 1:
            raw_result = model.infer(
                audio_inputs[0],
                language=language,
                return_objects=return_objects,
                return_route=True,
            )
            results = [build_single_result(audio_inputs[0], raw_result, include_route=True)]
        else:
            raw_results = model.batch_infer(
                audio_inputs,
                language=language,
                return_objects=return_objects,
            )
            results = build_batch_results(audio_inputs, raw_results)

        elapsed = time.perf_counter() - start

    output = {
        "results": results,
        "metadata": build_metadata(args, backend=backend, device=device, dtype=dtype, elapsed=elapsed),
    }
    dump_output(output, args.transcript_path)

    print(f"Output saved to {args.transcript_path}")
    if results:
        preview = results[0].get("text", "")[:200]
        print(f"Transcription: {preview}...")


if __name__ == "__main__":
    main()
