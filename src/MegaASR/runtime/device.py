"""Device and precision policy for Flash-Mega-ASR."""
from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

BFLOAT16 = getattr(torch, "bfloat16")
FLOAT16 = getattr(torch, "float16")
FLOAT32 = getattr(torch, "float32")


def resolve_device(device_id: str | None = None) -> str:
    """Resolve the compute device.

    Args:
        device_id: "0", "1", etc. for CUDA, "mps" for Apple Silicon, None for auto.

    Returns:
        Device string like "cuda:0", "mps", or "cpu"
    """
    if device_id is None or device_id == "":
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device_id == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        logger.warning("MPS requested but not available, falling back to CPU")
        return "cpu"

    if device_id.isdigit() or device_id.startswith("cuda:"):
        if torch.cuda.is_available():
            return f"cuda:{device_id}" if device_id.isdigit() else device_id
        logger.warning(f"CUDA device {device_id} requested but CUDA not available, falling back to CPU")
        return "cpu"

    return device_id


def resolve_dtype(device: str | None = None) -> Any:
    """Resolve the optimal dtype for the given device.

    Args:
        device: Device string. None for auto-detection.

    Returns:
        torch.bfloat16 on CUDA 8+, torch.float16 on other CUDA, torch.float32 on CPU
    """
    if device is None:
        device = resolve_device()

    if device.startswith("cuda"):
        try:
            device_idx = int(device.split(":")[-1]) if ":" in device else 0
            cap = torch.cuda.get_device_capability(device_idx)
            if cap[0] >= 8:
                logger.info(f"Using bfloat16 on {device} (compute capability {cap[0]}.{cap[1]})")
                return BFLOAT16
        except (RuntimeError, ValueError):
            pass
        logger.info(f"Using float16 on {device}")
        return FLOAT16

    if device == "mps":
        return FLOAT16

    logger.info("Using float32 on CPU")
    return FLOAT32


def get_device_info() -> dict:
    """Get comprehensive device information for diagnostics."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            props = torch.cuda.get_device_properties(i)
            total_memory = getattr(props, "total_mem", None)
            if total_memory is None:
                total_memory = props.total_memory
            info[f"cuda:{i}"] = {
                "name": torch.cuda.get_device_name(i),
                "capability": f"{cap[0]}.{cap[1]}",
                "memory_gb": round(total_memory / 1e9, 1),
            }
    return info
