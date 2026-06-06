#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/rerun_toy_tables.py"

GPU_SINE="${GPU_SINE:-1}"
GPU_MOONS="${GPU_MOONS:-0}"

run_table3_sine() {
  CUDA_VISIBLE_DEVICES="$GPU_SINE" python "$SCRIPT" \
    --task sine \
    --out-dir results_sine_noise015_30seeds_final \
    --seeds $(seq 0 29) \
    --sine-noise 0.15 \
    2>&1 | tee results_sine_noise015_30seeds_final.log
}

run_table4_twomoons() {
  CUDA_VISIBLE_DEVICES="$GPU_MOONS" python "$SCRIPT" \
    --task two_moons \
    --out-dir results_twomoons_noise03_500_1000_h16_10seeds_final \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --samples 250 \
    --moons-noise 0.30 \
    --moons-epochs 3000 \
    --moons-n-train 500 \
    --moons-n-test 1000 \
    --moons-hidden 16 \
    --moons-trl-steps 50 \
    --moons-trl-step-size 0.08 \
    --moons-trl-perp-scale 0.05 \
    --moons-trl-k 30 \
    2>&1 | tee results_twomoons_noise03_500_1000_h16_10seeds_final.log
}

run_table5_spine() {
  CUDA_VISIBLE_DEVICES="$GPU_MOONS" python "$SCRIPT" \
    --task all \
    --out-dir results_table5_spine_isolation_final_10seeds \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --samples 250 \
    --sine-noise 0.15 \
    --moons-noise 0.30 \
    --moons-epochs 3000 \
    --moons-n-train 500 \
    --moons-n-test 1000 \
    --moons-hidden 16 \
    --moons-trl-steps 50 \
    --moons-trl-step-size 0.08 \
    --moons-trl-perp-scale 0.05 \
    --moons-trl-k 30 \
    2>&1 | tee results_table5_spine_isolation_final_10seeds.log
}

case "$MODE" in
  table3|sine) run_table3_sine ;;
  table4|twomoons|two_moons) run_table4_twomoons ;;
  table5|spine) run_table5_spine ;;
  all) run_table3_sine; run_table4_twomoons; run_table5_spine ;;
  *) echo "Unknown mode: $MODE"; echo "Use: table3, table4, table5, all"; exit 1 ;;
esac
