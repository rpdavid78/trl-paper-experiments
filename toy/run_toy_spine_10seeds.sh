#!/usr/bin/env bash
set -euo pipefail
GPU="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/../results_toy_spine"
CUDA_VISIBLE_DEVICES="$GPU" python "$SCRIPT_DIR/toy_spine_single_vs_full.py" \
  --tasks sine moons \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --out-dir "$SCRIPT_DIR/../results_toy_spine" \
  2>&1 | tee "$SCRIPT_DIR/../results_toy_spine/toy_spine_3seeds.log"
