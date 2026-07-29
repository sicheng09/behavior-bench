#!/usr/bin/env bash
# Wait for frozen-policy_0 try_Multi training, then eval vs IDM + vs PPO (clean logs).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/wsc/behavior-bench}"
cd "$REPO_ROOT"

TRAIN_SESSION="${TRAIN_SESSION:-try-multi-ego-scratch-frozenB0-gpu1}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-ego_scratch_frozen_B_ego0_mb65k_1b_tryMulti}"
MIN_EPOCH="${MIN_EPOCH:-953}"
EVAL_GPU_IDM="${EVAL_GPU_IDM:-2}"
EVAL_GPU_PPO="${EVAL_GPU_PPO:-4}"
TEST_ROOT="$REPO_ROOT/.logs/train/run4/try_Multi/test"
EVAL_SCRIPT="$TEST_ROOT/run_one_policy_eval.sh"
RESOLVE_SCRIPT="$REPO_ROOT/.logs/train/run4/try_IDM/test/resolve_final_ego_checkpoint.sh"
WATCH_LOG="$TEST_ROOT/wait_and_eval_${TRAIN_RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCH_LOG"; }

# Discover wandb id from latest train log header / experiments dir growth
find_exp_dir() {
  # Prefer matching RUN_NAME in recent wandb or newest conservative_mix dir after start
  local log
  log="$(ls -t "$REPO_ROOT/.logs/train/run4/try_Multi/train/${TRAIN_RUN_NAME}"_*.log 2>/dev/null | head -1 || true)"
  if [[ -n "$log" ]]; then
    local wid
    wid="$(rg -o 'runs/[a-z0-9]+' "$log" | head -1 | sed 's|runs/||' || true)"
    if [[ -n "$wid" && -d "$REPO_ROOT/experiments/puffer_drive_conservative_mix_${wid}" ]]; then
      echo "$REPO_ROOT/experiments/puffer_drive_conservative_mix_${wid}"
      return
    fi
  fi
  # fallback: newest mix experiment modified recently
  ls -td "$REPO_ROOT"/experiments/puffer_drive_conservative_mix_*/ 2>/dev/null | head -1 | sed 's|/$||'
}

wait_training() {
  while true; do
    local tmux_alive=0 proc_alive=0
    if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then tmux_alive=1; fi
    if pgrep -f "train.name ${TRAIN_RUN_NAME}|--train.name ${TRAIN_RUN_NAME}|train.name.${TRAIN_RUN_NAME}" >/dev/null 2>&1 \
       || pgrep -f "puffer train.*${TRAIN_RUN_NAME}" >/dev/null 2>&1; then
      proc_alive=1
    fi
    # also match by run name substring in cmdline
    if pgrep -af 'puffer train' | rg -q "$TRAIN_RUN_NAME"; then proc_alive=1; fi

    if (( tmux_alive == 0 && proc_alive == 0 )); then
      # Ensure we actually saw training at least once via log
      if ls "$REPO_ROOT/.logs/train/run4/try_Multi/train/${TRAIN_RUN_NAME}"_*.log >/dev/null 2>&1; then
        log "Training finished (tmux gone + no train PID)."
        return
      fi
      log "Waiting for train log to appear..."
    else
      log "Waiting for training: tmux=$tmux_alive proc=$proc_alive session=$TRAIN_SESSION"
    fi
    sleep 60
  done
}

launch_eval() {
  local traffic="$1" gpu="$2"
  local ts session
  ts="$(date +%Y%m%d_%H%M%S)"
  session="eval-${TRAIN_RUN_NAME}-vs-${traffic}-gpu${gpu}-${ts}"
  session="$(echo "$session" | tr '.' '_' | cut -c1-80)"
  tmux new-session -d -s "$session" \
    "bash -lc 'MAP_IDS=all $EVAL_SCRIPT $TRAIN_RUN_NAME Drive_Recurrent $WEIGHTS Drive Recurrent 256 256 $gpu $traffic; echo EXIT=\$?'"
  log "Launched eval: $session on GPU $gpu (traffic=$traffic)" >&2
  printf '%s\n' "$session"
}

mkdir -p "$TEST_ROOT"
log "Watch log: $WATCH_LOG"
log "Train session: $TRAIN_SESSION"
log "Eval GPUs: IDM=$EVAL_GPU_IDM PPO=$EVAL_GPU_PPO MIN_EPOCH=$MIN_EPOCH"

wait_training

EXP_DIR="$(find_exp_dir)"
log "Experiment dir: $EXP_DIR"
while [[ ! -d "$EXP_DIR" ]]; do
  log "Waiting for experiment dir..."
  sleep 30
  EXP_DIR="$(find_exp_dir)"
done

WEIGHTS="$("$RESOLVE_SCRIPT" "$EXP_DIR" "$MIN_EPOCH")"
log "Using checkpoint: $WEIGHTS"

IDM_SESSION="$(launch_eval idm "$EVAL_GPU_IDM")"
PPO_SESSION="$(launch_eval ppo "$EVAL_GPU_PPO")"

log "Waiting for eval sessions..."
while tmux has-session -t "$IDM_SESSION" 2>/dev/null || tmux has-session -t "$PPO_SESSION" 2>/dev/null; do
  log "Eval still running: idm=$IDM_SESSION ppo=$PPO_SESSION"
  sleep 120
done

log "All evals finished for $TRAIN_RUN_NAME"
log "Results under: $TEST_ROOT/$TRAIN_RUN_NAME/Drive_Recurrent/"
