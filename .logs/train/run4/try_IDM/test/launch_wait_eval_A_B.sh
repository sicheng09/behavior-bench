#!/usr/bin/env bash
# Arm wait-and-eval for try_IDM C-A / C-B (mb65k) after training finishes.
set -euo pipefail

REPO_ROOT="${HOME}/wsc/behavior-bench"
TEST_DIR="$REPO_ROOT/.logs/train/run4/try_IDM/test"
cd "$REPO_ROOT"

chmod +x \
  "$TEST_DIR/wait_and_eval_cons_mix.sh" \
  "$TEST_DIR/run_one_policy_eval.sh" \
  "$TEST_DIR/resolve_final_ego_checkpoint.sh"

# From try_IDM/train launch (2026-07-18)
WID_A="${WID_A:-stq4hogz}"
WID_B="${WID_B:-plvd8yph}"

tmux kill-session -t wait-eval-tryIDM-A-mb65k 2>/dev/null || true
tmux kill-session -t wait-eval-tryIDM-B-mb65k 2>/dev/null || true

tmux new-session -d -s wait-eval-tryIDM-A-mb65k "bash -lc '
cd \$HOME/wsc/behavior-bench
export TRAIN_SESSION=cons-mix-A-mb65k-1b-gpu0
export WANDB_RUN_ID=$WID_A
export TRAIN_DIR_NAME=cons_mix_A_steer0333_mb65k_1b_tryIDM
export EVAL_GPU_IDM=2
export EVAL_GPU_PPO=3
export MIN_EPOCH=953
bash .logs/train/run4/try_IDM/test/wait_and_eval_cons_mix.sh
'"

tmux new-session -d -s wait-eval-tryIDM-B-mb65k "bash -lc '
cd \$HOME/wsc/behavior-bench
export TRAIN_SESSION=cons-mix-B-mb65k-1b-gpu1
export WANDB_RUN_ID=$WID_B
export TRAIN_DIR_NAME=cons_mix_B_shape_mb65k_1b_tryIDM
export EVAL_GPU_IDM=4
export EVAL_GPU_PPO=5
export MIN_EPOCH=953
bash .logs/train/run4/try_IDM/test/wait_and_eval_cons_mix.sh
'"

{
  echo "==== try_IDM wait-and-eval armed ===="
  echo "date=$(date -Iseconds)"
  echo "A: train=cons-mix-A-mb65k-1b-gpu0 wandb=$WID_A -> eval GPU 2=IDM, 3=PPO"
  echo "B: train=cons-mix-B-mb65k-1b-gpu1 wandb=$WID_B -> eval GPU 4=IDM, 5=PPO"
  echo "watch: wait-eval-tryIDM-A-mb65k / wait-eval-tryIDM-B-mb65k"
  echo "results: $TEST_DIR/cons_mix_*_tryIDM/Drive_Recurrent/"
  echo "====================================="
} | tee "$TEST_DIR/launch_wait_eval_A_B_$(date +%Y%m%d_%H%M%S).md"

sleep 2
tmux ls | rg 'wait-eval-tryIDM|cons-mix-.*mb65k' || true
