#!/usr/bin/env bash
# Wait for homogeneous DriveConditioned 2B training, then eval three creward
# profiles (aggr / normal / caut) vs IDM on pufferinter.
#
# Outputs under .logs/train/run4/DriveConditioned/tset/2B/
# Mirrors tset/500M/wait_and_eval_creward_vs_idm.sh (log->stderr fix included).
set -euo pipefail

cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

REPO_ROOT="$(pwd)"
TEST_ROOT="$REPO_ROOT/.logs/train/run4/DriveConditioned/tset/2B"
mkdir -p "$TEST_ROOT"

TRAIN_SESSION="${TRAIN_SESSION:-drive-cond-homo-2b-gpu6}"
WANDB_RUN_ID="${WANDB_RUN_ID:-dfkvxczz}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-homogeneous_drive_conditioned_2b_batch1x_run4}"
# 2e9 / 524288 ≈ 3814.7 → final ckpt epoch 3814+
MIN_EPOCH="${MIN_EPOCH:-3814}"
ENV_NAME="${ENV_NAME:-puffer_drive}"

# GPUs for the three profiles (avoid train GPU 6)
GPU_AGGR="${GPU_AGGR:-2}"
GPU_NORMAL="${GPU_NORMAL:-3}"
GPU_CAUT="${GPU_CAUT:-7}"

EXP_DIR="$REPO_ROOT/experiments/${ENV_NAME}_${WANDB_RUN_ID}"
WATCH_LOG="$TEST_ROOT/wait_and_eval_creward_vs_idm_$(date +%Y%m%d_%H%M%S).log"
EVAL_SCRIPT="$TEST_ROOT/run_conditioned_vs_idm.sh"

log() {
  # Always write to the watch log; use stderr so command substitutions
  # like WEIGHTS="$(resolve_checkpoint ...)" are not polluted.
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCH_LOG" >&2
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
      log "Neither tmux nor train PID seen yet; waiting"
      sleep 60
      continue
    fi
    log "Waiting for training: tmux=$tmux_alive proc=$proc_alive session=$TRAIN_SESSION"
    sleep 60
  done
}

resolve_checkpoint() {
  local min_epoch="$1"
  local cand=""
  local ep=""
  local stable_since=""
  local last_ep=""
  local now
  local start_ts
  start_ts="$(date +%s)"

  while true; do
    now="$(date +%s)"
    if (( now - start_ts > 14400 )); then
      log "ERROR: timed out waiting for checkpoint epoch>=$min_epoch in $EXP_DIR"
      exit 1
    fi

    cand="$(ls -1 "$EXP_DIR"/model_puffer_drive_*.pt 2>/dev/null | sort | tail -1 || true)"
    if [ -z "$cand" ] || [ ! -f "$cand" ]; then
      log "No checkpoint yet in $EXP_DIR"
      sleep 30
      continue
    fi

    ep="$(basename "$cand" | sed -n 's/.*_\([0-9][0-9]*\)\.pt$/\1/p' | sed 's/^0*//')"
    if [ -z "$ep" ]; then
      sleep 10
      continue
    fi
    if (( ep < min_epoch )); then
      log "Latest ckpt epoch=$ep < $min_epoch ($cand)"
      sleep 30
      continue
    fi

    if [ "$ep" != "$last_ep" ]; then
      last_ep="$ep"
      stable_since="$now"
      log "Saw target ckpt epoch=$ep; waiting for file stability"
      sleep 10
      continue
    fi

    if (( now - stable_since >= 20 )); then
      echo "$cand"
      return 0
    fi
    sleep 5
  done
}

launch_one() {
  local profile="$1"
  local gpu="$2"
  local weights="$3"
  local ts session
  ts="$(date +%Y%m%d_%H%M%S)"
  session="eval-drivecond-2b-${profile}-vs-idm-gpu${gpu}-${ts}"

  log "Launching $profile vs idm on GPU $gpu (tmux=$session)"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" "bash -lc '
set -euo pipefail
MAP_IDS=all \"$EVAL_SCRIPT\" \"$profile\" \"$weights\" \"$gpu\"
'"
  echo "$session"
}

{
  log "==== wait_and_eval DriveConditioned 2B creward profiles vs IDM ===="
  log "TRAIN_SESSION=$TRAIN_SESSION"
  log "WANDB_RUN_ID=$WANDB_RUN_ID"
  log "TRAIN_RUN_NAME=$TRAIN_RUN_NAME"
  log "EXP_DIR=$EXP_DIR"
  log "MIN_EPOCH=$MIN_EPOCH"
  log "GPUs: aggr=$GPU_AGGR normal=$GPU_NORMAL caut=$GPU_CAUT"
  log "OUTPUT_ROOT=$TEST_ROOT"
}

wait_for_training_done
log "Training finished."

WEIGHTS="$(resolve_checkpoint "$MIN_EPOCH")"
log "Using weights: $WEIGHTS"

# sanity: path must be a real file and not contain log noise
if [ ! -f "$WEIGHTS" ]; then
  log "ERROR: resolved weights path is not a file: [$WEIGHTS]"
  exit 1
fi

S1="$(launch_one conditioned_aggr "$GPU_AGGR" "$WEIGHTS")"
S2="$(launch_one conditioned_normal "$GPU_NORMAL" "$WEIGHTS")"
S3="$(launch_one conditioned_caut "$GPU_CAUT" "$WEIGHTS")"

log "Launched sessions: $S1 | $S2 | $S3"
log "Done launching. Watch logs under $TEST_ROOT/<profile>_vs_idm/"
