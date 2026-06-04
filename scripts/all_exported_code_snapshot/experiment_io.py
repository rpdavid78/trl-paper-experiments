"""Utilities for TRL ICLR experimental runs.

These helpers are intentionally small and dependency-light. They are meant to be
imported by the CIFAR scripts without changing the model code.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import ContextDecorator
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def to_jsonable(x: Any) -> Any:
    """Convert common scientific Python/PyTorch objects to JSON-safe values."""
    if is_dataclass(x):
        return to_jsonable(asdict(x))
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    if hasattr(x, "tolist"):
        try:
            return x.tolist()
        except Exception:
            pass
    return x


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")


class StageTimer(ContextDecorator):
    """Measure wall-clock time and peak CUDA memory for a code block."""

    def __init__(self, name: str, out: Optional[Dict[str, Dict[str, float]]] = None):
        self.name = name
        self.out = out if out is not None else {}
        self.start = 0.0
        self.elapsed_sec = 0.0
        self.peak_vram_gb = 0.0

    def __enter__(self):
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed_sec = time.perf_counter() - self.start
        if torch is not None and torch.cuda.is_available():
            self.peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        self.out[self.name] = {
            "wall_sec": float(self.elapsed_sec),
            "peak_vram_gb": float(self.peak_vram_gb),
        }
        return False


def flatten_timings(prefix: str, timings: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    row = {}
    for stage, vals in timings.items():
        safe_stage = stage.replace(" ", "_").replace("/", "_")
        for key, value in vals.items():
            row[f"{prefix}_{safe_stage}_{key}"] = value
    return row
