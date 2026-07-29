#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

REPO_ROOT="$(pwd)"
VAL_ROOT="$REPO_ROOT/.logs/val/run2"
RESULTS_ROOT="$VAL_ROOT/results"
mkdir -p "$RESULTS_ROOT"

WATCH_LOG="$VAL_ROOT/wait_and_eval_run2_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCH_LOG"
}

wait_for_session() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    log "Waiting for training session: $session"
    sleep 300
  done
}

wait_for_file() {
  local path="$1"
  while [ ! -f "$path" ]; do
    log "Waiting for checkpoint: $path"
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
    "bash -lc 'MAP_IDS=all .logs/val/run2/run_one_policy_eval.sh $train_dir $policy_dir $weights $policy_class $rnn_name $rnn_input $rnn_hidden $gpu'"
  log "Launched eval: $session_name -> $train_dir/$policy_dir on GPU $gpu"
}

LOW_SESSION="perception-low-homogeneous-run2-gpu0-20260713_204738"
MID_SESSION="perception-mid-homogeneous-run2-gpu1-20260713_204750"
HIGH_SESSION="perception-high-homogeneous-run2-gpu2-20260713_204808"
MIX_SESSION="perception-mixed-low-mid-high-run2-gpu3-20260713_204818"

log "Waiting for run2 trainings to finish."
wait_for_session "$LOW_SESSION"
wait_for_session "$MID_SESSION"
wait_for_session "$HIGH_SESSION"
wait_for_session "$MIX_SESSION"

LOW_CKPT="$REPO_ROOT/experiments/puffer_drive_r8jh6epn/model_puffer_drive_003815.pt"
MID_CKPT="$REPO_ROOT/experiments/puffer_drive_877i3fj2/model_puffer_drive_003815.pt"
HIGH_CKPT="$REPO_ROOT/experiments/puffer_drive_zegl2eqd/model_puffer_drive_003815.pt"
MIX_ROOT="$REPO_ROOT/experiments/puffer_drive_d2spiva4"
MIX_LOW_CKPT="$MIX_ROOT/model_policy_0_puffer_drive_003815.pt"
MIX_MID_CKPT="$MIX_ROOT/model_policy_1_puffer_drive_003815.pt"
MIX_HIGH_CKPT="$MIX_ROOT/model_policy_2_puffer_drive_003815.pt"

wait_for_file "$LOW_CKPT"
wait_for_file "$MID_CKPT"
wait_for_file "$HIGH_CKPT"
wait_for_file "$MIX_LOW_CKPT"
wait_for_file "$MIX_MID_CKPT"
wait_for_file "$MIX_HIGH_CKPT"

log "All checkpoints are ready. Launching pufferinter vs IDM evals."

TS="$(date +%Y%m%d_%H%M%S)"
launch_eval "eval-run2-homo-low-gpu4-$TS" \
  "homogeneous_perception_low_2b_batch1x_run2" \
  "DrivePerceptionLow_Recurrent" \
  "$LOW_CKPT" "DrivePerceptionLow" "Recurrent" "256" "256" "4"

launch_eval "eval-run2-homo-mid-gpu5-$TS" \
  "homogeneous_perception_mid_2b_batch1x_run2" \
  "DrivePerceptionMid_Recurrent" \
  "$MID_CKPT" "DrivePerceptionMid" "Recurrent" "256" "256" "5"

launch_eval "eval-run2-homo-high-gpu6-$TS" \
  "homogeneous_perception_high_drive_2b_batch1x_run2" \
  "Drive_Recurrent" \
  "$HIGH_CKPT" "Drive" "Recurrent" "256" "256" "6"

launch_eval "eval-run2-mix-p0-low-gpu4-$TS" \
  "perception_mixed_low_mid_high_6b_batch3x_run2" \
  "p0_DrivePerceptionLow_Recurrent" \
  "$MIX_LOW_CKPT" "DrivePerceptionLow" "Recurrent" "256" "256" "4"

launch_eval "eval-run2-mix-p1-mid-gpu5-$TS" \
  "perception_mixed_low_mid_high_6b_batch3x_run2" \
  "p1_DrivePerceptionMid_Recurrent" \
  "$MIX_MID_CKPT" "DrivePerceptionMid" "Recurrent" "256" "256" "5"

launch_eval "eval-run2-mix-p2-high-gpu6-$TS" \
  "perception_mixed_low_mid_high_6b_batch3x_run2" \
  "p2_Drive_Recurrent" \
  "$MIX_HIGH_CKPT" "Drive" "Recurrent" "256" "256" "6"

log "Run2 eval sessions launched."
