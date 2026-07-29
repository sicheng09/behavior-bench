#!/usr/bin/env bash
# Wait for a mix training tmux to finish, discover W&B run id from the train log,
# then sequentially eval Drive+Recurrent vs TRAFFICS on the same GPU.
#
# Required env:
#   TRAIN_SESSION / TRAIN_RUN_NAME / EVAL_GPU
#   LOG_DIR          directory containing train logs for this run
#   RUN_NAME_PREFIX  used to pick newest ${RUN_NAME_PREFIX}_*.log
#   RESULT_ROOT      e.g. .../repeat/drivelite_mixA_stratified
#                    → writes test_IDM|test_Expert|test_Smart under RESULT_ROOT
#
# Optional:
#   POLICY_IDX (default 0; Perception high = 2)
#   MIN_EPOCH (default 953)
#   TRAFFICS (default "idm expert smart")
#   ENV_NAME (default puffer_drive)
set -euo pipefail

cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

REPO_ROOT="$(pwd)"
HELPERS="$REPO_ROOT/.logs/train/run_repaet/test_helpers"

TRAIN_SESSION="${TRAIN_SESSION:?}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:?}"
EVAL_GPU="${EVAL_GPU:?}"
LOG_DIR="${LOG_DIR:?}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:?}"
RESULT_ROOT="${RESULT_ROOT:?}"

POLICY_IDX="${POLICY_IDX:-0}"
POLICY_DIR_NAME="${POLICY_DIR_NAME:-Drive_Recurrent}"
MIN_EPOCH="${MIN_EPOCH:-953}"
ENV_NAME="${ENV_NAME:-puffer_drive}"
POLICY_CLASS="${POLICY_CLASS:-Drive}"
RNN_NAME="${RNN_NAME:-Recurrent}"
RNN_INPUT="${RNN_INPUT:-256}"
RNN_HIDDEN="${RNN_HIDDEN:-256}"
TRAFFICS="${TRAFFICS:-idm expert smart}"

EVAL_SCRIPT="$HELPERS/run_one_policy_eval.sh"
RESOLVE_SCRIPT="$HELPERS/resolve_final_policy_checkpoint.sh"
mkdir -p "$RESULT_ROOT" "$LOG_DIR"
WATCH_LOG="$RESULT_ROOT/wait_and_eval_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCH_LOG"
}

train_pid_running() {
  pgrep -f "puffer train .*--train.name ${TRAIN_RUN_NAME}" >/dev/null 2>&1
}

wait_for_training_done() {
  local saw_alive=0
  while true; do
    local tmux_alive=0 proc_alive=0
    if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
      tmux_alive=1; saw_alive=1
    fi
    if train_pid_running; then
      proc_alive=1; saw_alive=1
    fi
    if (( tmux_alive == 0 && proc_alive == 0 )); then
      if (( saw_alive == 1 )); then
        return 0
      fi
      log "Neither tmux nor train PID seen yet; waiting ($TRAIN_SESSION / $TRAIN_RUN_NAME)"
      sleep 60
      continue
    fi
    log "Waiting for training: tmux=$tmux_alive proc=$proc_alive session=$TRAIN_SESSION"
    sleep 60
  done
}

newest_train_log() {
  ls -1t "$LOG_DIR"/${RUN_NAME_PREFIX}_*.log 2>/dev/null | head -1 || true
}

discover_wandb_id() {
  local logf="$1" wid=""
  wid="$(rg -o 'wandb.ai/[^/]+/[^/]+/runs/([a-z0-9]+)' -r '$1' "$logf" | tail -1 || true)"
  if [ -z "$wid" ]; then
    wid="$(rg -o 'run-[0-9]+_[0-9]+-([a-z0-9]+)' -r '$1' "$logf" | tail -1 || true)"
  fi
  if [ -z "$wid" ]; then
    wid="$(rg -o 'Synced [^\n]*\(([a-z0-9]+)\)' -r '$1' "$logf" | tail -1 || true)"
  fi
  printf '%s' "$wid"
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

traffic_test_root() {
  local t="$1"
  case "$t" in
    idm) echo "$RESULT_ROOT/test_IDM" ;;
    expert) echo "$RESULT_ROOT/test_Expert" ;;
    smart) echo "$RESULT_ROOT/test_Smart" ;;
    *) echo "$RESULT_ROOT/test_${t}" ;;
  esac
}

run_one_traffic() {
  local traffic="$1" gpu="$2" weights="$3"
  local test_root session ts
  test_root="$(traffic_test_root "$traffic")"
  mkdir -p "$test_root"
  ts="$(date +%Y%m%d_%H%M%S)"
  session="eval-${TRAIN_RUN_NAME}-vs-${traffic}-gpu${gpu}-${ts}"
  session="$(echo "$session" | tr '.' '_' | cut -c1-80)"

  wait_for_gpu_free "$gpu"
  tmux new-session -d -s "$session" "bash -lc '
export TEST_ROOT=$test_root MAP_IDS=all SMART_WEIGHTS=$REPO_ROOT/weights/SMART_epoch_030.pt
$EVAL_SCRIPT $TRAIN_RUN_NAME $POLICY_DIR_NAME $weights $POLICY_CLASS $RNN_NAME $RNN_INPUT $RNN_HIDDEN $gpu $traffic
echo EXIT_CODE=\$?
'"
  log "Launched $session (traffic=$traffic) → $test_root"
  while tmux has-session -t "$session" 2>/dev/null; do
    log "Eval still running: $session"
    sleep 120
  done
  log "Finished traffic=$traffic for $TRAIN_RUN_NAME"
}

log "Watch log: $WATCH_LOG"
log "Train session=$TRAIN_SESSION run_name=$TRAIN_RUN_NAME gpu=$EVAL_GPU"
log "LOG_DIR=$LOG_DIR prefix=$RUN_NAME_PREFIX"
log "RESULT_ROOT=$RESULT_ROOT traffics=[$TRAFFICS] policy_idx=$POLICY_IDX"

wait_for_training_done
log "Training finished."

TRAIN_LOG="$(newest_train_log)"
if [ -z "$TRAIN_LOG" ] || [ ! -f "$TRAIN_LOG" ]; then
  log "ERROR: no train log matching ${LOG_DIR}/${RUN_NAME_PREFIX}_*.log"
  exit 1
fi
log "Train log: $TRAIN_LOG"

WANDB_RUN_ID="$(discover_wandb_id "$TRAIN_LOG")"
if [ -z "$WANDB_RUN_ID" ]; then
  log "ERROR: could not discover W&B run id from $TRAIN_LOG"
  exit 1
fi
log "Discovered W&B run: $WANDB_RUN_ID"
echo "$WANDB_RUN_ID" >"$RESULT_ROOT/wandb_run_id.txt"

EXP_DIR="$REPO_ROOT/experiments/${ENV_NAME}_${WANDB_RUN_ID}"
while [ ! -d "$EXP_DIR" ]; do
  log "Waiting for experiment dir: $EXP_DIR"
  sleep 30
done

export STABLE_SECONDS=1
WEIGHTS="$("$RESOLVE_SCRIPT" "$EXP_DIR" "$MIN_EPOCH" "$POLICY_IDX")"
log "Checkpoint: $WEIGHTS"
echo "$WEIGHTS" >"$RESULT_ROOT/checkpoint.txt"

for traffic in $TRAFFICS; do
  run_one_traffic "$traffic" "$EVAL_GPU" "$WEIGHTS"
done

log "All traffics done for $TRAIN_RUN_NAME"
log "Results under: $RESULT_ROOT/test_{IDM,Expert,Smart}/"
