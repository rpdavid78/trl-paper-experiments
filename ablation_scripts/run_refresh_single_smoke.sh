#!/usr/bin/env bash
set -euo pipefail
SEED=${1:-0}
GPU=${2:-0}
cd .
export PYTHONPATH=ablation_scripts:.:./scripts:.:${PYTHONPATH:-}
CUDA_VISIBLE_DEVICES=${GPU} python ablation_scripts/trl_refresh_single_ablation_cifar100.py \
  --seed ${SEED} \
  --ckpt-dir checkpoints/checkpoints_c100_seed${SEED} \
  --results results/trl_refresh_single_ablation_smoke_seed${SEED}.jsonl \
  --tube-scale 4.0 \
  --n-samples 3 \
  --fixbn-batches 2 \
  --hvp-batches 1 \
  --fresh-max-points 3 \
  --modes single fresh
