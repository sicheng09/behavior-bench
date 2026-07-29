#!/usr/bin/env bash
# Launch ConservativeMix A+B r2 training, then arm wait-and-eval (pufferinter vs IDM/PPO).
set -euo pipefail

REPO_ROOT="${HOME}/wsc/behavior-bench"
cd "$REPO_ROOT"

RUN_DIR="$REPO_ROOT/.logs/train/run4/try/run"
TEST_DIR="$REPO_ROOT/.logs/train/run4/try/test"

chmod +x \
  "$RUN_DIR/start_cons_mix_A_batch2x_1b_r2.sh" \
  "$RUN_DIR/start_cons_mix_B_batch2x_1b_r2.sh" \
  "$TEST_DIR/wait_and_eval_cons_mix.sh" \
  "$TEST_DIR/run_one_policy_eval.sh" \
  "$TEST_DIR/resolve_final_ego_checkpoint.sh"

echo "==== Launching A r2 (GPU0) + B r2 (GPU1) ===="
bash "$RUN_DIR/start_cons_mix_A_batch2x_1b_r2.sh"
bash "$RUN_DIR/start_cons_mix_B_batch2x_1b_r2.sh"

wait_wandb_id() {
  local pattern="$1"
  local wid=""
  for _ in $(seq 1 90); do
    local log
    log="$(ls -t "$RUN_DIR"/${pattern}_*.log 2>/dev/null | head -1 || true)"
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
WID_A="$(wait_wandb_id cons_mix_A_steer0333_batch2x_1b_run4_r2)"
WID_B="$(wait_wandb_id cons_mix_B_shape_batch2x_1b_run4_r2)"
echo "A r2 WANDB_RUN_ID=$WID_A"
echo "B r2 WANDB_RUN_ID=$WID_B"

tmux kill-session -t wait-eval-cons-mix-A-r2 2>/dev/null || true
tmux kill-session -t wait-eval-cons-mix-B-r2 2>/dev/null || true

tmux new-session -d -s wait-eval-cons-mix-A-r2 "bash -lc '
cd \$HOME/wsc/behavior-bench
export TRAIN_SESSION=cons-mix-A-batch2x-1b-r2-gpu0
export WANDB_RUN_ID=$WID_A
export TRAIN_DIR_NAME=cons_mix_A_steer0333_batch2x_1b_run4_r2
export EVAL_GPU_IDM=2
export EVAL_GPU_PPO=3
export MIN_EPOCH=953
bash .logs/train/run4/try/test/wait_and_eval_cons_mix.sh
'"

tmux new-session -d -s wait-eval-cons-mix-B-r2 "bash -lc '
cd \$HOME/wsc/behavior-bench
export TRAIN_SESSION=cons-mix-B-batch2x-1b-r2-gpu1
export WANDB_RUN_ID=$WID_B
export TRAIN_DIR_NAME=cons_mix_B_shape_batch2x_1b_run4_r2
export EVAL_GPU_IDM=4
export EVAL_GPU_PPO=5
export MIN_EPOCH=953
bash .logs/train/run4/try/test/wait_and_eval_cons_mix.sh
'"

{
  echo "==== cons_mix A/B r2 launch summary ===="
  echo "date=$(date -Iseconds)"
  echo "A train tmux: cons-mix-A-batch2x-1b-r2-gpu0  wandb=$WID_A"
  echo "B train tmux: cons-mix-B-batch2x-1b-r2-gpu1  wandb=$WID_B"
  echo "A wait-eval:  wait-eval-cons-mix-A-r2  (eval GPU 2=IDM, 3=PPO)"
  echo "B wait-eval:  wait-eval-cons-mix-B-r2  (eval GPU 4=IDM, 5=PPO)"
  echo "train logs:   $RUN_DIR/cons_mix_*_run4_r2_*.log"
  echo "eval root:    $TEST_DIR/cons_mix_*_run4_r2/"
  echo "========================================"
} | tee "$RUN_DIR/launch_cons_mix_A_B_r2_$(date +%Y%m%d_%H%M%S).md"

sleep 2
tmux ls | rg 'cons-mix|wait-eval-cons-mix' || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | head -4
