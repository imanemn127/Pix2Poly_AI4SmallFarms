#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

cd /mnt/DATA/IMANE/AI4SmallFarms

/mnt/DATA/IMANE/p3/bin/python train.py \
  experiment=p2p_ai4smallfarms \
  2>&1 | tee /mnt/DATA/IMANE/AI4SmallFarms/train_full_ai4smallfarms.log
