#!/usr/bin/env bash
# Wait for a run_repaet mix training tmux to finish, then eval Drive+Recurrent
# vs IDM on pufferinter using the *same GPU* as training (EVAL_GPU).
#
# Required env:
#   TRAIN_SESSION   e.g. mixmoe-mb98k-1500m-gpu1
#   WANDB_RUN_ID    e.g. ncn8qxko
#   TRAIN_DIR_NAME  result subdirectory under TEST_ROOT
#   TRAIN_RUN_NAME  --train.name used by puffer (for PID match)
#   TEST_ROOT       e.g. $REPO/.logs/train/run_repaet/trainMOE/test_IDM
#   EVAL_GPU        same GPU as training
#
# Optional:
#   POLICY_IDX (default 0; Perception high Drive = 2)
#   MIN_EPOCH (default 953)
#   POLICY_DIR_NAME (default Drive_Recurrent)
#   ENV_NAME (default puffer_drive)
set -euo pipefail

cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

REPO_ROOT="$(pwd)"
HELPERS="$REPO_ROOT/.logs/train/run_repaet/test_helpers"

TRAIN_SESSION="${TRAIN_SESSION:?TRAIN_SESSION required}"
WANDB_RUN_ID="${WANDB_RUN_ID:?WANDB_RUN_ID required}"
TRAIN_DIR_NAME="${TRAIN_DIR_NAME:?TRAIN_DIR_NAME required}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:?TRAIN_RUN_NAME required}"
TEST_ROOT="${TEST_ROOT:?TEST_ROOT required}"
EVAL_GPU="${EVAL_GPU:?EVAL_GPU required}"

POLICY_IDX="${POLICY_IDX:-0}"
POLICY_DIR_NAME="${POLICY_DIR_NAME:-Drive_Recurrent}"
MIN_EPOCH="${MIN_EPOCH:-953}"
ENV_NAME="${ENV_NAME:-puffer_drive}"
POLICY_CLASS="${POLICY_CLASS:-Drive}"
RNN_NAME="${RNN_NAME:-Recurrent}"
RNN_INPUT="${RNN_INPUT:-256}"
RNN_HIDDEN="${RNN_HIDDEN:-256}"

mkdir -p "$TEST_ROOT"
EXP_DIR="$REPO_ROOT/experiments/${ENV_NAME}_${WANDB_RUN_ID}"
WATCH_LOG="$TEST_ROOT/wait_and_eval_${TRAIN_DIR_NAME}_$(date +%Y%m%d_%H%M%S).log"
EVAL_SCRIPT="$HELPERS/run_one_policy_eval.sh"
RESOLVE_SCRIPT="$HELPERS/resolve_final_policy_checkpoint.sh"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCH_LOG"
}

train_pid_running() {
  pgrep -f "puffer train .*--train.name ${TRAIN_RUN_NAME}" >/dev/null 2>&1
}

wait_for_training_done() {
  local saw_alive=0
  while true; do
    local tmux_alive=0
    local proc_alive=0
    if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
      tmux_alive=1
      saw_alive=1
    fi
    if train_pid_running; then
      proc_alive=1
      saw_alive=1
    fi
    if (( tmux_alive == 0 && proc_alive == 0 )); then
      if (( saw_alive == 1 )); then
        return 0
      fi
      log "Neither tmux nor train PID seen yet for $TRAIN_SESSION / $TRAIN_RUN_NAME; waiting"
      sleep 60
      continue
    fi
    log "Waiting for training: tmux=$tmux_alive proc=$proc_alive session=$TRAIN_SESSION"
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
    sleep 30
  done
}

launch_eval_idm() {
  local gpu="$1"
  local ts session
  ts="$(date +%Y%m%d_%H%M%S)"
  session="eval-${TRAIN_DIR_NAME}-vs-idm-gpu${gpu}-${ts}"
  session="$(echo "$session" | tr '.' '_' | cut -c1-80)"

  wait_for_gpu_free "$gpu"
  tmux new-session -d -s "$session" \
    "bash -lc 'export TEST_ROOT=$TEST_ROOT; MAP_IDS=all $EVAL_SCRIPT $TRAIN_DIR_NAME $POLICY_DIR_NAME $WEIGHTS $POLICY_CLASS $RNN_NAME $RNN_INPUT $RNN_HIDDEN $gpu idm'"
  log "Launched eval: $session on GPU $gpu (traffic=idm)" >&2
  printf '%s\n' "$session"
}

log "Watch log: $WATCH_LOG"
log "Train session: $TRAIN_SESSION"
log "WANDB_RUN_ID: $WANDB_RUN_ID"
log "TRAIN_RUN_NAME: $TRAIN_RUN_NAME"
log "Expected experiment dir: $EXP_DIR"
log "Policy slot: $POLICY_IDX ($POLICY_CLASS+$RNN_NAME)"
log "Results: $TEST_ROOT/$TRAIN_DIR_NAME/$POLICY_DIR_NAME/"
log "Eval GPU (reuse train GPU): $EVAL_GPU MIN_EPOCH=$MIN_EPOCH"

wait_for_training_done
log "Training finished (tmux gone + no train PID)."

log "Resolving final policy_${POLICY_IDX} checkpoint (min_epoch=$MIN_EPOCH)..."
while [ ! -d "$EXP_DIR" ]; do
  log "Waiting for experiment dir: $EXP_DIR"
  sleep 60
done

WEIGHTS="$("$RESOLVE_SCRIPT" "$EXP_DIR" "$MIN_EPOCH" "$POLICY_IDX")"
log "Using checkpoint: $WEIGHTS"

IDM_SESSION="$(launch_eval_idm "$EVAL_GPU")"

log "Waiting for IDM eval session to finish..."
while tmux has-session -t "$IDM_SESSION" 2>/dev/null; do
  log "Eval still running: $IDM_SESSION"
  sleep 120
done

log "IDM eval finished for $TRAIN_DIR_NAME"
log "Results under: $TEST_ROOT/$TRAIN_DIR_NAME/$POLICY_DIR_NAME/"
