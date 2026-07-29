#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

REPO_ROOT="$(pwd)"
VAL_ROOT="$REPO_ROOT/.logs/val/run2"
RESULTS_ROOT="$VAL_ROOT/results1000"
mkdir -p "$RESULTS_ROOT"

WATCH_LOG="$VAL_ROOT/wait_and_eval_run2_1000_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCH_LOG"
}

wait_for_file() {
  local path="$1"
  while [ ! -f "$path" ]; do
    log "Waiting for checkpoint: $path"
    sleep 300
  done
}

wait_for_eval_sessions() {
  local prefix="$1"
  while tmux list-sessions 2>/dev/null | grep -q "$prefix"; do
    log "Waiting for eval sessions matching: $prefix"
    sleep 300
  done
}

launch_eval() {
  local session_name="$1"
  local train_dir="$2"
  local policy_dir="$3"
  local weights="$4"
  local policy_class="$5"
  local rnn_name="$6"
  local rnn_input="$7"
  local rnn_hidden="$8"
  local gpu="$9"

  tmux new-session -d -s "$session_name" \
    "bash -lc 'MAP_IDS=all .logs/val/run2/run_one_policy_eval_1000.sh $train_dir $policy_dir $weights $policy_class $rnn_name $rnn_input $rnn_hidden $gpu'"
  log "Launched eval: $session_name -> $train_dir/$policy_dir on GPU $gpu"
}

LOW_CKPT="$REPO_ROOT/experiments/puffer_drive_r8jh6epn/model_puffer_drive_001000.pt"
MID_CKPT="$REPO_ROOT/experiments/puffer_drive_877i3fj2/model_puffer_drive_001000.pt"
HIGH_CKPT="$REPO_ROOT/experiments/puffer_drive_zegl2eqd/model_puffer_drive_001000.pt"
MIX_ROOT="$REPO_ROOT/experiments/puffer_drive_d2spiva4"
MIX_LOW_CKPT="$MIX_ROOT/model_policy_0_puffer_drive_001000.pt"
MIX_MID_CKPT="$MIX_ROOT/model_policy_1_puffer_drive_001000.pt"
MIX_HIGH_CKPT="$MIX_ROOT/model_policy_2_puffer_drive_001000.pt"

log "Waiting for run2 001000 checkpoints."
wait_for_file "$LOW_CKPT"
wait_for_file "$MID_CKPT"
wait_for_file "$HIGH_CKPT"
wait_for_file "$MIX_LOW_CKPT"
wait_for_file "$MIX_MID_CKPT"
wait_for_file "$MIX_HIGH_CKPT"

TS="$(date +%Y%m%d_%H%M%S)"
log "Launching homogeneous 001000 evals."
launch_eval "eval1000-run2-homo-low-gpu4-$TS" \
  "homogeneous_perception_low_2b_batch1x_run2" \
  "DrivePerceptionLow_Recurrent" \
  "$LOW_CKPT" "DrivePerceptionLow" "Recurrent" "256" "256" "4"

launch_eval "eval1000-run2-homo-mid-gpu5-$TS" \
  "homogeneous_perception_mid_2b_batch1x_run2" \
  "DrivePerceptionMid_Recurrent" \
  "$MID_CKPT" "DrivePerceptionMid" "Recurrent" "256" "256" "5"

launch_eval "eval1000-run2-homo-high-gpu6-$TS" \
  "homogeneous_perception_high_drive_2b_batch1x_run2" \
  "Drive_Recurrent" \
  "$HIGH_CKPT" "Drive" "Recurrent" "256" "256" "6"

wait_for_eval_sessions "eval1000-run2-homo"

TS="$(date +%Y%m%d_%H%M%S)"
log "Launching mixed 001000 evals."
launch_eval "eval1000-run2-mix-p0-low-gpu4-$TS" \
  "perception_mixed_low_mid_high_6b_batch3x_run2" \
  "p0_DrivePerceptionLow_Recurrent" \
  "$MIX_LOW_CKPT" "DrivePerceptionLow" "Recurrent" "256" "256" "4"

launch_eval "eval1000-run2-mix-p1-mid-gpu5-$TS" \
  "perception_mixed_low_mid_high_6b_batch3x_run2" \
  "p1_DrivePerceptionMid_Recurrent" \
  "$MIX_MID_CKPT" "DrivePerceptionMid" "Recurrent" "256" "256" "5"

launch_eval "eval1000-run2-mix-p2-high-gpu6-$TS" \
  "perception_mixed_low_mid_high_6b_batch3x_run2" \
  "p2_Drive_Recurrent" \
  "$MIX_HIGH_CKPT" "Drive" "Recurrent" "256" "256" "6"

log "Run2 001000 eval sessions launched."
