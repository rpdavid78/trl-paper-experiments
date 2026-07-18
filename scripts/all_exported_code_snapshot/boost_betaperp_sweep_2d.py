"""Compatibility import for the historical standalone sweep location.

The executable implementation is integrated with the canonical CIFAR-100
runner so it can reuse the exact TRL spine, projected prior, sampler, FixBN
path, and metric functions. Run it through the CLI documented in the top-level
README instead of copying experimental hooks into another file.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cifar100_all_methods_iclr import boost_betaperp_sweep_2d  # noqa: E402,F401

__all__ = ["boost_betaperp_sweep_2d"]


if __name__ == "__main__":
    raise SystemExit(
        "Run: python scripts/cifar100_all_methods_iclr.py --methods trl "
        "--run-boost-betaperp-sweep"
    )
