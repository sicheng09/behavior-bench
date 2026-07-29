#!/usr/bin/env bash
# Wait for a ConservativeMix training tmux session to finish, then eval ego
# (policy 0, Drive+Recurrent) vs IDM and vs same-weight PPO on pufferinter.
#
# Required env:
#   TRAIN_SESSION   e.g. cons-mix-A-batch2x-1b-gpu0
#   WANDB_RUN_ID    e.g. 6qdti7k0
#   TRAIN_DIR_NAME  e.g. cons_mix_A_steer0333_batch2x_1b_run4
#
# Optional:
#   EVAL_GPU_IDM / EVAL_GPU_PPO (default 2 / 3)
#   MIN_EPOCH (default 953)
#   POLICY_DIR_NAME (default Drive_Recurrent)
set -euo pipefail

cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

REPO_ROOT="$(pwd)"
TEST_ROOT="$REPO_ROOT/.logs/train/run4/try/test"
mkdir -p "$TEST_ROOT"

TRAIN_SESSION="${TRAIN_SESSION:?TRAIN_SESSION required}"
WANDB_RUN_ID="${WANDB_RUN_ID:?WANDB_RUN_ID required}"
TRAIN_DIR_NAME="${TRAIN_DIR_NAME:?TRAIN_DIR_NAME required}"
POLICY_DIR_NAME="${POLICY_DIR_NAME:-Drive_Recurrent}"
EVAL_GPU_IDM="${EVAL_GPU_IDM:-2}"
EVAL_GPU_PPO="${EVAL_GPU_PPO:-3}"
MIN_EPOCH="${MIN_EPOCH:-953}"
ENV_NAME="${ENV_NAME:-puffer_drive_conservative_mix}"

EXP_DIR="$REPO_ROOT/experiments/${ENV_NAME}_${WANDB_RUN_ID}"
WATCH_LOG="$TEST_ROOT/wait_and_eval_${TRAIN_DIR_NAME}_$(date +%Y%m%d_%H%M%S).log"
EVAL_SCRIPT="$TEST_ROOT/run_one_policy_eval.sh"
RESOLVE_SCRIPT="$TEST_ROOT/resolve_final_ego_checkpoint.sh"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCH_LOG"
}

wait_for_session() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    log "Waiting for training session: $session"
    sleep 60
  done
}

gpu_has_compute_process() {
  local gpu="$1"
  nvidia-smi | awk -v gpu="$gpu" '
    $1 == "|" && $2 == gpu && $4 == "N/A" && $5 == "N/A" && $6 ~ /^[0-9]+$/ && $8 == "C" { found=1 }
    END { exit found ? 0 : 1 }
  '
}

wait_for_gpu_free() {
  local gpu="$1"
  while gpu_has_compute_process "$gpu"; do
    log "Waiting for GPU $gpu to be free"
    sleep 60
  done
}

launch_eval() {
  local traffic="$1"
  local gpu="$2"
  local ts session
  ts="$(date +%Y%m%d_%H%M%S)"
  session="eval-${TRAIN_DIR_NAME}-vs-${traffic}-gpu${gpu}-${ts}"
  # tmux session names cannot contain '.' easily; sanitize
  session="$(echo "$session" | tr '.' '_' | cut -c1-80)"

  wait_for_gpu_free "$gpu"
  tmux new-session -d -s "$session" \
    "bash -lc 'MAP_IDS=all $EVAL_SCRIPT $TRAIN_DIR_NAME $POLICY_DIR_NAME $WEIGHTS Drive Recurrent 256 256 $gpu $traffic'"
  log "Launched eval: $session on GPU $gpu (traffic=$traffic)"
  echo "$session"
}

log "Watch log: $WATCH_LOG"
log "Train session: $TRAIN_SESSION"
log "WANDB_RUN_ID: $WANDB_RUN_ID"
log "Expected experiment dir: $EXP_DIR"
log "Results: $TEST_ROOT/$TRAIN_DIR_NAME/$POLICY_DIR_NAME/"
log "Eval GPUs: IDM=$EVAL_GPU_IDM PPO=$EVAL_GPU_PPO MIN_EPOCH=$MIN_EPOCH"

wait_for_session "$TRAIN_SESSION"
log "Training session ended."

log "Resolving final ego checkpoint (min_epoch=$MIN_EPOCH)..."
while [ ! -d "$EXP_DIR" ]; do
  log "Waiting for experiment dir: $EXP_DIR"
  sleep 60
done

WEIGHTS="$("$RESOLVE_SCRIPT" "$EXP_DIR" "$MIN_EPOCH")"
log "Using checkpoint: $WEIGHTS"

# vs IDM first (main metric), then vs PPO (generalization gate).
# Launch both in parallel on separate GPUs when free.
IDM_SESSION="$(launch_eval idm "$EVAL_GPU_IDM")"
PPO_SESSION="$(launch_eval ppo "$EVAL_GPU_PPO")"

log "Waiting for eval sessions to finish..."
while tmux has-session -t "$IDM_SESSION" 2>/dev/null || tmux has-session -t "$PPO_SESSION" 2>/dev/null; do
  log "Eval still running: idm=$IDM_SESSION ppo=$PPO_SESSION"
  sleep 120
done

log "All evals finished for $TRAIN_DIR_NAME"
log "Results under: $TEST_ROOT/$TRAIN_DIR_NAME/$POLICY_DIR_NAME/"
