#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

REPO_ROOT="$(pwd)"
VAL_ROOT="$REPO_ROOT/.logs/val/run1/results/500M"
WATCH_LOG="$VAL_ROOT/wait_and_eval_mixed_b_500m_$(date +%Y%m%d_%H%M%S).log"

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

MIX_B_SESSION="run1-500m-mixed-b-gpu5-20260714_120324"
MIX_B_ROOT="$REPO_ROOT/experiments/puffer_drive_s0cc66y6"
P0="$MIX_B_ROOT/model_policy_0_puffer_drive_000954.pt"
P1="$MIX_B_ROOT/model_policy_1_puffer_drive_000954.pt"
P2="$MIX_B_ROOT/model_policy_2_puffer_drive_000954.pt"

wait_for_session "$MIX_B_SESSION"
wait_for_file "$P0"
wait_for_file "$P1"
wait_for_file "$P2"
wait_for_gpus_free 4 6 7

TS="$(date +%Y%m%d_%H%M%S)"
tmux new-session -d -s "eval500m-mix-b-p0-gpu4-$TS" \
  "bash -lc 'MAP_IDS=all .logs/val/run1/results/500M/run_one_policy_eval_500m.sh mix_drive_lstm_drivelite_lstm224_drive_nolstm_1500m_batch3x_run1_500m p0_Drive_Recurrent $P0 Drive Recurrent 256 256 4'"
tmux new-session -d -s "eval500m-mix-b-p1-gpu6-$TS" \
  "bash -lc 'MAP_IDS=all .logs/val/run1/results/500M/run_one_policy_eval_500m.sh mix_drive_lstm_drivelite_lstm224_drive_nolstm_1500m_batch3x_run1_500m p1_DriveLite_Recurrent224 $P1 DriveLite Recurrent 224 224 6'"
tmux new-session -d -s "eval500m-mix-b-p2-gpu7-$TS" \
  "bash -lc 'MAP_IDS=all .logs/val/run1/results/500M/run_one_policy_eval_500m.sh mix_drive_lstm_drivelite_lstm224_drive_nolstm_1500m_batch3x_run1_500m p2_Drive_NoLSTM $P2 Drive None 256 256 7'"

log "Launched Mixed B 500M evals."
