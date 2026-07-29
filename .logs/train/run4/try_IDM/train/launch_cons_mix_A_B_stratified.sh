#!/usr/bin/env bash
# Launch try_IDM ConservativeMix A+B with stratified H0-aligned ego geometry.
set -euo pipefail

REPO_ROOT="${HOME}/wsc/behavior-bench"
TRAIN_DIR="$REPO_ROOT/.logs/train/run4/try_IDM/train"
TEST_DIR="$REPO_ROOT/.logs/train/run4/try_IDM/test"
cd "$REPO_ROOT"

chmod +x \
  "$TRAIN_DIR/start_cons_mix_A_stratified_1b.sh" \
  "$TRAIN_DIR/start_cons_mix_B_stratified_1b.sh" \
  "$TEST_DIR/wait_and_eval_cons_mix.sh" \
  "$TEST_DIR/run_one_policy_eval.sh" \
  "$TEST_DIR/resolve_final_ego_checkpoint.sh"

echo "==== Launching stratified A (GPU0) + B (GPU1) ===="
bash "$TRAIN_DIR/start_cons_mix_A_stratified_1b.sh"
bash "$TRAIN_DIR/start_cons_mix_B_stratified_1b.sh"

wait_wandb_id() {
  local pattern="$1"
  local wid=""
  for _ in $(seq 1 90); do
    local log
    log="$(ls -t "$TRAIN_DIR"/${pattern}_*.log 2>/dev/null | head -1 || true)"
    if [ -n "$log" ]; then
      wid="$(rg -o 'runs/[a-z0-9]{8}' "$log" 2>/dev/null | head -1 | cut -d/ -f2 || true)"
      if [ -n "$wid" ]; then
        echo "$wid"
        return 0
      fi
    fi
    sleep 10
  done
  return 1
}

echo "Waiting for W&B run ids..."
WID_A="$(wait_wandb_id cons_mix_A_steer0333_stratified_1b_tryIDM)"
WID_B="$(wait_wandb_id cons_mix_B_shape_stratified_1b_tryIDM)"
echo "A WANDB_RUN_ID=$WID_A"
echo "B WANDB_RUN_ID=$WID_B"

tmux kill-session -t wait-eval-tryIDM-A-strat 2>/dev/null || true
tmux kill-session -t wait-eval-tryIDM-B-strat 2>/dev/null || true

tmux new-session -d -s wait-eval-tryIDM-A-strat "bash -lc '
cd \$HOME/wsc/behavior-bench
export TRAIN_SESSION=cons-mix-A-strat-1b-gpu0
export WANDB_RUN_ID=$WID_A
export TRAIN_DIR_NAME=cons_mix_A_steer0333_stratified_1b_tryIDM
export EVAL_GPU_IDM=2
export EVAL_GPU_PPO=3
export MIN_EPOCH=953
bash .logs/train/run4/try_IDM/test/wait_and_eval_cons_mix.sh
'"

tmux new-session -d -s wait-eval-tryIDM-B-strat "bash -lc '
cd \$HOME/wsc/behavior-bench
export TRAIN_SESSION=cons-mix-B-strat-1b-gpu1
export WANDB_RUN_ID=$WID_B
export TRAIN_DIR_NAME=cons_mix_B_shape_stratified_1b_tryIDM
export EVAL_GPU_IDM=4
export EVAL_GPU_PPO=5
export MIN_EPOCH=953
bash .logs/train/run4/try_IDM/test/wait_and_eval_cons_mix.sh
'"

{
  echo "==== try_IDM cons_mix A/B stratified launch summary ===="
  echo "date=$(date -Iseconds)"
  echo "mode: mix_ppo_sampling=stratified policy_mb=32768,32768 update_steps=16,16"
  echo "A: tmux=cons-mix-A-strat-1b-gpu0 wandb=$WID_A eval GPU 2=IDM 3=PPO"
  echo "B: tmux=cons-mix-B-strat-1b-gpu1 wandb=$WID_B eval GPU 4=IDM 5=PPO"
  echo "watch: wait-eval-tryIDM-A-strat / wait-eval-tryIDM-B-strat"
  echo "train logs: $TRAIN_DIR/cons_mix_*_stratified_1b_tryIDM_*.log"
  echo "eval root:  $TEST_DIR/cons_mix_*_stratified_1b_tryIDM/"
  echo "doc:        $TRAIN_DIR/stratified_mix_training.md"
  echo "========================================"
} | tee "$TRAIN_DIR/launch_cons_mix_A_B_stratified_$(date +%Y%m%d_%H%M%S).md"

sleep 3
tmux ls | rg 'cons-mix-.*strat|wait-eval-tryIDM-.*strat' || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | head -4
