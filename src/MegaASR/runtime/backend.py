"""Attention backend resolver for Flash-Mega-ASR.

Resolution order in 'auto' mode (device-aware):
  CUDA: flash_attention_2 -> sdpa -> eager
  CPU / MPS: sdpa -> eager
"""
from __future__ import annotations

import importlib
import logging
from typing import Literal

import torch

logger = logging.getLogger(__name__)

BackendName = Literal["auto", "flash_attention_2", "flash_attention_3", "sdpa", "eager"]


def is_flash_attn_2_available() -> bool:
    """Check if flash-attn 2 package is installed and CUDA is available."""
    if not torch.cuda.is_available():
        return False
    try:
        importlib.import_module("flash_attn")
    except ImportError:
        return False
    return True


def is_flash_attn_3_available() -> bool:
    """Check if flash-attn 3 kernels are available (Hopper+)."""
    if not torch.cuda.is_available():
        return False
    try:
        flash_attn = importlib.import_module("flash_attn")
    except ImportError:
        return False
    return hasattr(flash_attn, "flash_attn_with_kvcache")


def is_sdpa_available() -> bool:
    """Check if PyTorch SDPA is available (torch >= 2.0)."""
    return hasattr(torch.nn.functional, "scaled_dot_product_attention")


def resolve_attn_backend(requested: BackendName = "auto") -> str:
    """Resolve the attention backend to use.

    Args:
        requested: Backend name. "auto" tries FA2 -> SDPA -> eager (device-aware).

    Returns:
        One of: "flash_attention_2", "flash_attention_3", "sdpa", "eager"
    """
    if requested != "auto":
        if requested == "flash_attention_2" and not is_flash_attn_2_available():
            logger.warning("flash_attention_2 requested but not available, falling back to sdpa")
            return "sdpa" if is_sdpa_available() else "eager"
        if requested == "flash_attention_3" and not is_flash_attn_3_available():
            logger.warning("flash_attention_3 requested but not available, falling back to flash_attention_2")
            if is_flash_attn_2_available():
                return "flash_attention_2"
            return "sdpa" if is_sdpa_available() else "eager"
        if requested == "sdpa" and not is_sdpa_available():
            logger.warning("sdpa requested but not available, falling back to eager")
            return "eager"
        return requested

    # Auto mode: device-aware fallback chain
    if is_flash_attn_2_available():
        logger.info("Auto-resolved attention backend: flash_attention_2")
        return "flash_attention_2"

    if is_sdpa_available():
        logger.info("Auto-resolved attention backend: sdpa (flash-attn not available)")
        return "sdpa"

    logger.info("Auto-resolved attention backend: eager (neither flash-attn nor sdpa available)")
    return "eager"
