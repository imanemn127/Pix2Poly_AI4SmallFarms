#!/bin/bash
# Convenience script to launch training.
# Run from the root of the cloned repository:
#   bash run_train.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

# Change to the directory containing this script (repo root)
cd "$(dirname "$0")"

python train.py \
  experiment=p2p_ai4smallfarms \
  2>&1 | tee train_ai4smallfarms.log
