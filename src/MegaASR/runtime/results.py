"""Result formatting utilities for Flash-Mega-ASR."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TranscriptionResult:
    """Single transcription result with metadata."""

    text: str
    chunks: list[dict[str, Any]] = field(default_factory=list)
    use_lora: bool | None = None
    degraded_prob: float | None = None
    route_source: str = "default"
    backend: str = ""
    dtype: str = ""
    device: str = ""
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)


@dataclass
class BatchTranscriptionResult:
    """Batch transcription results."""

    results: list[TranscriptionResult]
    total_elapsed_s: float = 0.0
    batch_size: int = 0
    backend: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [result.to_dict() for result in self.results],
            "total_elapsed_s": self.total_elapsed_s,
            "batch_size": self.batch_size,
            "backend": self.backend,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)
