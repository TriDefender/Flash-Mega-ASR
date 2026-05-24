"""Flash-Mega-ASR: flash-attention optimized batched inference with LoRA routing."""

from MegaASR.model.megaASR import MegaASR
from MegaASR.runtime.results import BatchTranscriptionResult, TranscriptionResult

__all__ = [
    "MegaASR",
    "TranscriptionResult",
    "BatchTranscriptionResult",
]
