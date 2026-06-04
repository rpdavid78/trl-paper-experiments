#!/usr/bin/env bash
set -euo pipefail
SEED=${1:-0}
GPU=${2:-0}
cd /home/rpdavid/projects/trl_iclr_code/trl_iclr_code
export PYTHONPATH=/mnt/hd2/rpdavid/trl_extra_ablation_update/code:/mnt/hd2/rpdavid/trl_export/code:/home/rpdavid/projects/trl_iclr_code/trl_iclr_code/scripts:/home/rpdavid/projects/trl_iclr_code/trl_iclr_code:${PYTHONPATH:-}
CUDA_VISIBLE_DEVICES=${GPU} python /mnt/hd2/rpdavid/trl_extra_ablation_update/code/trl_refresh_single_ablation_cifar100.py \
  --seed ${SEED} \
  --ckpt-dir /mnt/hd2/rpdavid/trl_checkpoints/checkpoints_c100_seed${SEED} \
  --results /mnt/hd2/rpdavid/trl_results/trl_refresh_single_ablation_smoke_seed${SEED}.jsonl \
  --tube-scale 4.0 \
  --n-samples 3 \
  --fixbn-batches 2 \
  --hvp-batches 1 \
  --fresh-max-points 3 \
  --modes single fresh
