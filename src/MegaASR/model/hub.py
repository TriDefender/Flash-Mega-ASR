"""Centralized HuggingFace Hub resolver for Flash-Mega-ASR.

When no local checkpoint paths are provided, resolves all model assets
from the HuggingFace cache via ``snapshot_download``.  On first call the
assets are downloaded; subsequent calls hit the local HF cache only.

Resolution precedence:
  explicit path argument  >  ``ckpt_dir`` derived path  >  HF Hub default
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---- upstream repo IDs -------------------------------------------------- #
_BASE_MODEL_REPO = "Qwen/Qwen3-ASR-1.7B"
_MEGA_ASR_REPO = "zhifeixie/Mega-ASR"

# patterns that snapshot_download should fetch from the Mega-ASR repo
# (excludes the large base model -- that comes from its own repo)
_MEGA_ASR_PATTERNS = [
    "mega-asr-merged/*",
    "audio_quality_router/*",
]


def _is_repo_id(value: str) -> bool:
    """Check if a string looks like an HF repo ID (org/name) rather than a local path."""
    if "\\" in value:
        return False
    if "/" in value and not Path(value).expanduser().is_dir():
        return True
    return False


def _normalize_model_source(model_path: str | os.PathLike[str]) -> str:
    """Normalize local model paths without corrupting Hugging Face repo IDs."""
    value = os.fspath(model_path)
    if _is_repo_id(value):
        return value
    return str(Path(value).expanduser())


def download_all_assets(
    target_dir: str | os.PathLike[str] | None = None,
    *,
    base_model_repo: str = _BASE_MODEL_REPO,
    mega_asr_repo: str = _MEGA_ASR_REPO,
) -> str:
    """Download the base Qwen model plus Mega-ASR extras into the local layout."""
    from huggingface_hub import snapshot_download

    local_dir = Path(target_dir).expanduser() if target_dir else Path("ckpt/Mega-ASR")
    snapshot_download(
        repo_id=base_model_repo,
        repo_type="model",
        local_dir=str(local_dir / "Qwen3-ASR-1.7B"),
        local_dir_use_symlinks=False,
    )
    snapshot_download(
        repo_id=mega_asr_repo,
        repo_type="model",
        allow_patterns=_MEGA_ASR_PATTERNS,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    return str(local_dir)


def _resolve_mega_asr_snapshot(
    *,
    repo_id: str = _MEGA_ASR_REPO,
    revision: str | None = None,
    **snapshot_kwargs: Any,
) -> str:
    """Download (or retrieve from cache) the Mega-ASR extras repo.

    Returns the local cache directory path.
    """
    from huggingface_hub import snapshot_download

    # allow_patterns uses glob-style matching; try both directory spellings
    # the upstream repo uses "mega-asr-merged" but be safe
    try:
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=_MEGA_ASR_PATTERNS,
            revision=revision,
            **snapshot_kwargs,
        )
    except Exception:
        # Fallback: download without patterns (full repo) if filtered fails
        logger.warning(
            "Filtered snapshot_download failed, falling back to full repo download"
        )
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            **snapshot_kwargs,
        )

    return str(path)


def resolve_sources(
    *,
    model_path: str | os.PathLike[str] | None = None,
    lora_dir: str | os.PathLike[str] | None = None,
    router_checkpoint: str | os.PathLike[str] | None = None,
    ckpt_dir: str | os.PathLike[str] | None = None,
    routing_enabled: bool = True,
) -> dict[str, str | None]:
    """Resolve all asset sources.

    Returns a dict with keys ``model_path``, ``lora_dir``,
    ``router_checkpoint`` pointing to either local paths or HF-cache paths.
    """
    # 1. Explicit ckpt_dir overrides -- derive local paths from it
    if ckpt_dir is not None:
        ckpt = Path(ckpt_dir).expanduser()
        return {
            "model_path": str(model_path or ckpt / "Qwen3-ASR-1.7B"),
            "lora_dir": str(lora_dir or ckpt / "mega-asr-merged"),
            "router_checkpoint": (
                str(router_checkpoint or ckpt / "audio_quality_router" / "best_acc_model.safetensors")
                if routing_enabled
                else None
            ),
        }

    # 2. All explicit paths given -- pass through (do NOT pass HF repo IDs through Path)
    if model_path and lora_dir and (router_checkpoint is not None or not routing_enabled):
        return {
            "model_path": _normalize_model_source(model_path),
            "lora_dir": str(Path(lora_dir).expanduser()),
            "router_checkpoint": str(Path(router_checkpoint).expanduser()) if routing_enabled and router_checkpoint else None,
        }

    # 3. Partial paths + HF Hub for the rest
    snapshot_path: str | None = None

    if lora_dir is None or (routing_enabled and router_checkpoint is None):
        logger.info("Resolving Mega-ASR extras from HuggingFace Hub …")
        snapshot_path = _resolve_mega_asr_snapshot()

    resolved_lora_dir = str(Path(lora_dir).expanduser()) if lora_dir else None
    resolved_router = str(Path(router_checkpoint).expanduser()) if router_checkpoint else None

    if resolved_lora_dir is None and snapshot_path:
        resolved_lora_dir = str(Path(snapshot_path) / "mega-asr-merged")

    if routing_enabled and resolved_router is None and snapshot_path:
        resolved_router = str(Path(snapshot_path) / "audio_quality_router" / "best_acc_model.safetensors")

    # Base model: explicit local path > HF repo ID (from_pretrained handles cache)
    if model_path is not None:
        resolved_model = _normalize_model_source(model_path)
    else:
        # Default: use HF repo ID directly (do NOT pass through Path on Windows)
        resolved_model = _BASE_MODEL_REPO

    return {
        "model_path": resolved_model,
        "lora_dir": resolved_lora_dir,
        "router_checkpoint": resolved_router,
    }
