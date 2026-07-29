#!/usr/bin/env bash
# Wait for training, resolve ckpt, then launch TRAFFIC_GPU_MAP in parallel.
#
# Required:
#   TRAIN_SESSION TRAIN_RUN_NAME LOG_DIR RUN_NAME_PREFIX RESULT_ROOT
#   TRAFFIC_GPU_MAP  e.g. "idm:2 expert:4 smart:5"
# Optional: POLICY_IDX MIN_EPOCH
set -euo pipefail

cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

REPO_ROOT="$(pwd)"
HELPERS="$REPO_ROOT/.logs/train/run_repaet/test_helpers"
RESOLVE_SCRIPT="$HELPERS/resolve_final_policy_checkpoint.sh"
LAUNCH_SCRIPT="$HELPERS/launch_parallel_traffics.sh"

TRAIN_SESSION="${TRAIN_SESSION:?}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:?}"
LOG_DIR="${LOG_DIR:?}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:?}"
RESULT_ROOT="${RESULT_ROOT:?}"
TRAFFIC_GPU_MAP="${TRAFFIC_GPU_MAP:?}"

POLICY_IDX="${POLICY_IDX:-0}"
MIN_EPOCH="${MIN_EPOCH:-953}"
ENV_NAME="${ENV_NAME:-puffer_drive}"

mkdir -p "$RESULT_ROOT" "$LOG_DIR"
WATCH_LOG="$RESULT_ROOT/wait_and_eval_parallel_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCH_LOG"; }

train_pid_running() {
  pgrep -f "puffer train .*--train.name ${TRAIN_RUN_NAME}" >/dev/null 2>&1
}

wait_for_training_done() {
  local saw_alive=0
  while true; do
    local tmux_alive=0 proc_alive=0
    tmux has-session -t "$TRAIN_SESSION" 2>/dev/null && tmux_alive=1 && saw_alive=1
    train_pid_running && proc_alive=1 && saw_alive=1
    if (( tmux_alive == 0 && proc_alive == 0 )); then
      (( saw_alive == 1 )) && return 0
      log "Neither tmux nor train PID seen yet; waiting"
      sleep 60
      continue
    fi
    log "Waiting for training: tmux=$tmux_alive proc=$proc_alive"
    sleep 60
  done
}

newest_train_log() {
  ls -1t "$LOG_DIR"/${RUN_NAME_PREFIX}_*.log 2>/dev/null | head -1 || true
}

discover_wandb_id() {
  local logf="$1" wid=""
  wid="$(rg -o 'wandb.ai/[^/]+/[^/]+/runs/([a-z0-9]+)' -r '$1' "$logf" | tail -1 || true)"
  [ -z "$wid" ] && wid="$(rg -o 'run-[0-9]+_[0-9]+-([a-z0-9]+)' -r '$1' "$logf" | tail -1 || true)"
  printf '%s' "$wid"
}

log "Watch log: $WATCH_LOG"
log "Train=$TRAIN_SESSION run=$TRAIN_RUN_NAME map=[$TRAFFIC_GPU_MAP]"

wait_for_training_done
log "Training finished."

TRAIN_LOG="$(newest_train_log)"
[ -n "$TRAIN_LOG" ] && [ -f "$TRAIN_LOG" ] || { log "ERROR: no train log"; exit 1; }
WANDB_RUN_ID="$(discover_wandb_id "$TRAIN_LOG")"
[ -n "$WANDB_RUN_ID" ] || { log "ERROR: no wandb id"; exit 1; }
echo "$WANDB_RUN_ID" >"$RESULT_ROOT/wandb_run_id.txt"
log "W&B=$WANDB_RUN_ID"

EXP_DIR="$REPO_ROOT/experiments/${ENV_NAME}_${WANDB_RUN_ID}"
while [ ! -d "$EXP_DIR" ]; do log "Waiting for $EXP_DIR"; sleep 30; done

export STABLE_SECONDS=1
WEIGHTS="$("$RESOLVE_SCRIPT" "$EXP_DIR" "$MIN_EPOCH" "$POLICY_IDX")"
echo "$WEIGHTS" >"$RESULT_ROOT/checkpoint.txt"
log "Checkpoint: $WEIGHTS"

export TRAIN_RUN_NAME RESULT_ROOT WEIGHTS TRAFFIC_GPU_MAP
bash "$LAUNCH_SCRIPT"
log "All parallel evals launched."
