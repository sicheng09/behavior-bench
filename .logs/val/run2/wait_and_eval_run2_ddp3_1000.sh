#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

REPO_ROOT="$(pwd)"
VAL_ROOT="$REPO_ROOT/.logs/val/run2"
mkdir -p "$VAL_ROOT/results1000"

WATCH_LOG="$VAL_ROOT/wait_and_eval_run2_ddp3_1000_$(date +%Y%m%d_%H%M%S).log"

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

gpu_has_compute_process() {
  local gpu="$1"
  nvidia-smi | awk -v gpu="$gpu" '
    $1 == "|" && $2 == gpu && $4 == "N/A" && $5 == "N/A" && $6 ~ /^[0-9]+$/ && $8 == "C" { found=1 }
    END { exit found ? 0 : 1 }
  '
}

wait_for_gpus_free() {
  local gpus=("$@")
  while true; do
    local busy=()
    for gpu in "${gpus[@]}"; do
      if gpu_has_compute_process "$gpu"; then
        busy+=("$gpu")
      fi
    done
    if [ "${#busy[@]}" -eq 0 ]; then
      return 0
    fi
    log "Waiting for GPUs to be free: ${busy[*]}"
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

RUN_ID="gob17caq"
CKPT_ROOT="$REPO_ROOT/experiments/puffer_drive_${RUN_ID}"
P0="$CKPT_ROOT/model_policy_0_puffer_drive_001000.pt"
P1="$CKPT_ROOT/model_policy_1_puffer_drive_001000.pt"
P2="$CKPT_ROOT/model_policy_2_puffer_drive_001000.pt"

log "Waiting for DDP3 run2 001000 checkpoints."
wait_for_file "$P0"
wait_for_file "$P1"
wait_for_file "$P2"

log "DDP3 001000 checkpoints are ready."
wait_for_gpus_free 4 5 6

TS="$(date +%Y%m%d_%H%M%S)"
launch_eval "eval1000-run2-ddp3-p0-low-gpu4-$TS" \
  "perception_mixed_low_mid_high_ddp3_global6b_run2" \
  "p0_DrivePerceptionLow_Recurrent" \
  "$P0" "DrivePerceptionLow" "Recurrent" "256" "256" "4"

launch_eval "eval1000-run2-ddp3-p1-mid-gpu5-$TS" \
  "perception_mixed_low_mid_high_ddp3_global6b_run2" \
  "p1_DrivePerceptionMid_Recurrent" \
  "$P1" "DrivePerceptionMid" "Recurrent" "256" "256" "5"

launch_eval "eval1000-run2-ddp3-p2-high-gpu6-$TS" \
  "perception_mixed_low_mid_high_ddp3_global6b_run2" \
  "p2_Drive_Recurrent" \
  "$P2" "Drive" "Recurrent" "256" "256" "6"

log "DDP3 001000 eval sessions launched."
