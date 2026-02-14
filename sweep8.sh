#!/usr/bin/env bash
set -e

run_one () {
  echo "=============================="
  echo "TRAIN: $*"
  python -m src.train "$@"
  latest=$(ls -td runs/* | head -n 1)
  echo "EVAL: $latest"
  python -m src.eval --run_dir "$latest"
}

# 8 lightweight demo trials
run_one --patch 4 --mask_ratio 0.4 --mask_mode random --lambda_insul 0.02 --beta_l1 0.1 --dist_gamma 0.5 --epochs 30 --seed 101
run_one --patch 4 --mask_ratio 0.4 --mask_mode dist --dist_k 2.0 --lambda_insul 0.02 --beta_l1 0.1 --dist_gamma 0.5 --epochs 30 --seed 102
run_one --patch 4 --mask_ratio 0.4 --mask_mode dist --dist_k 3.0 --lambda_insul 0.02 --beta_l1 0.1 --dist_gamma 0.5 --epochs 30 --seed 103
run_one --patch 4 --mask_ratio 0.4 --mask_mode mixed --mixed_prob 0.7 --dist_k 3.0 --lambda_insul 0.02 --beta_l1 0.1 --dist_gamma 0.5 --epochs 30 --seed 104
run_one --patch 4 --mask_ratio 0.4 --mask_mode mixed --mixed_prob 0.7 --dist_k 3.0 --lambda_insul 0.01 --beta_l1 0.1 --dist_gamma 0.5 --epochs 30 --seed 105
run_one --patch 4 --mask_ratio 0.4 --mask_mode mixed --mixed_prob 0.7 --dist_k 3.0 --lambda_insul 0.05 --beta_l1 0.1 --dist_gamma 0.5 --epochs 30 --seed 106
run_one --patch 4 --mask_ratio 0.4 --mask_mode mixed --mixed_prob 0.7 --dist_k 3.0 --lambda_insul 0.02 --beta_l1 0.0 --dist_gamma 0.5 --epochs 30 --seed 107
run_one --patch 4 --mask_ratio 0.4 --mask_mode mixed --mixed_prob 0.7 --dist_k 3.0 --lambda_insul 0.02 --beta_l1 0.2 --dist_gamma 0.5 --epochs 30 --seed 108

echo "All 8 demo trials finished."
