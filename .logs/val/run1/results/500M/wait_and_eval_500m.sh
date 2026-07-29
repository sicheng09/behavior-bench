#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

REPO_ROOT="$(pwd)"
VAL_ROOT="$REPO_ROOT/.logs/val/run1/results/500M"
mkdir -p "$VAL_ROOT"

WATCH_LOG="$VAL_ROOT/wait_and_eval_500m_$(date +%Y%m%d_%H%M%S).log"

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
    "bash -lc 'MAP_IDS=all .logs/val/run1/results/500M/run_one_policy_eval_500m.sh $train_dir $policy_dir $weights $policy_class $rnn_name $rnn_input $rnn_hidden $gpu'"
  log "Launched eval: $session_name -> $train_dir/$policy_dir on GPU $gpu"
}

SESS_DRIVE_LSTM="run1-500m-drive-lstm-gpu6-20260714_120854"
SESS_DRIVE_FF="run1-500m-drive-nolstm-gpu0-20260714_120054"
SESS_LITE_FF="run1-500m-drivelite-nolstm-gpu1-20260714_120104"
SESS_LITE_LSTM="run1-500m-drivelite-lstm224-gpu2-20260714_120117"
SESS_MIX_A="run1-500m-mixed-a-gpu4-20260714_120313"
SESS_MIX_B="run1-500m-mixed-b-gpu5-20260714_120324"

DRIVE_LSTM="$REPO_ROOT/experiments/puffer_drive_fdfw3v5e/model_puffer_drive_000954.pt"
DRIVE_FF="$REPO_ROOT/experiments/puffer_drive_7y88i67e/model_puffer_drive_000954.pt"
LITE_FF="$REPO_ROOT/experiments/puffer_drive_o2smiumc/model_puffer_drive_000954.pt"
LITE_LSTM="$REPO_ROOT/experiments/puffer_drive_vtfdqz3s/model_puffer_drive_000954.pt"
MIX_A_ROOT="$REPO_ROOT/experiments/puffer_drive_unvigwxk"
MIX_A_P0="$MIX_A_ROOT/model_policy_0_puffer_drive_000954.pt"
MIX_A_P1="$MIX_A_ROOT/model_policy_1_puffer_drive_000954.pt"
MIX_A_P2="$MIX_A_ROOT/model_policy_2_puffer_drive_000954.pt"
MIX_B_ROOT="$REPO_ROOT/experiments/puffer_drive_s0cc66y6"
MIX_B_P0="$MIX_B_ROOT/model_policy_0_puffer_drive_000954.pt"
MIX_B_P1="$MIX_B_ROOT/model_policy_1_puffer_drive_000954.pt"
MIX_B_P2="$MIX_B_ROOT/model_policy_2_puffer_drive_000954.pt"

log "Waiting for run1 500M homogeneous trainings to finish."
wait_for_session "$SESS_DRIVE_LSTM"
wait_for_session "$SESS_DRIVE_FF"
wait_for_session "$SESS_LITE_FF"
wait_for_session "$SESS_LITE_LSTM"

wait_for_file "$DRIVE_LSTM"
wait_for_file "$DRIVE_FF"
wait_for_file "$LITE_FF"
wait_for_file "$LITE_LSTM"

log "Homogeneous 500M checkpoints are ready."
wait_for_gpus_free 0 1 2 3

TS="$(date +%Y%m%d_%H%M%S)"
launch_eval "eval500m-drive-lstm-gpu0-$TS" \
  "homogeneous_drive_lstm_500m_batch1x_run1" \
  "Drive_Recurrent" \
  "$DRIVE_LSTM" "Drive" "Recurrent" "256" "256" "0"

launch_eval "eval500m-drive-nolstm-gpu1-$TS" \
  "homogeneous_drive_nolstm_500m_batch1x_run1" \
  "Drive_NoLSTM" \
  "$DRIVE_FF" "Drive" "None" "256" "256" "1"

launch_eval "eval500m-drivelite-nolstm-gpu2-$TS" \
  "homogeneous_drivelite_nolstm_500m_batch1x_run1" \
  "DriveLite_NoLSTM" \
  "$LITE_FF" "DriveLite" "None" "224" "224" "2"

launch_eval "eval500m-drivelite-lstm224-gpu3-$TS" \
  "homogeneous_drivelite_lstm224_500m_batch1x_run1" \
  "DriveLite_Recurrent224" \
  "$LITE_LSTM" "DriveLite" "Recurrent" "224" "224" "3"

wait_for_eval_sessions "eval500m-"
log "Homogeneous 500M evals finished."

log "Waiting for run1 500M mixed trainings to finish."
wait_for_session "$SESS_MIX_A"
wait_for_session "$SESS_MIX_B"

wait_for_file "$MIX_A_P0"
wait_for_file "$MIX_A_P1"
wait_for_file "$MIX_A_P2"
wait_for_file "$MIX_B_P0"
wait_for_file "$MIX_B_P1"
wait_for_file "$MIX_B_P2"

log "Mixed 500M checkpoints are ready."
wait_for_gpus_free 0 1 2 3 4 5

TS="$(date +%Y%m%d_%H%M%S)"
launch_eval "eval500m-mix-a-p0-gpu0-$TS" \
  "mix_intelligence_3level_1500m_batch3x_run1_500m" \
  "p0_Drive_Recurrent" \
  "$MIX_A_P0" "Drive" "Recurrent" "256" "256" "0"

launch_eval "eval500m-mix-a-p1-gpu1-$TS" \
  "mix_intelligence_3level_1500m_batch3x_run1_500m" \
  "p1_Drive_NoLSTM" \
  "$MIX_A_P1" "Drive" "None" "256" "256" "1"

launch_eval "eval500m-mix-a-p2-gpu2-$TS" \
  "mix_intelligence_3level_1500m_batch3x_run1_500m" \
  "p2_DriveLite_NoLSTM" \
  "$MIX_A_P2" "DriveLite" "None" "224" "224" "2"

launch_eval "eval500m-mix-b-p0-gpu3-$TS" \
  "mix_drive_lstm_drivelite_lstm224_drive_nolstm_1500m_batch3x_run1_500m" \
  "p0_Drive_Recurrent" \
  "$MIX_B_P0" "Drive" "Recurrent" "256" "256" "3"

launch_eval "eval500m-mix-b-p1-gpu4-$TS" \
  "mix_drive_lstm_drivelite_lstm224_drive_nolstm_1500m_batch3x_run1_500m" \
  "p1_DriveLite_Recurrent224" \
  "$MIX_B_P1" "DriveLite" "Recurrent" "224" "224" "4"

launch_eval "eval500m-mix-b-p2-gpu5-$TS" \
  "mix_drive_lstm_drivelite_lstm224_drive_nolstm_1500m_batch3x_run1_500m" \
  "p2_Drive_NoLSTM" \
  "$MIX_B_P2" "Drive" "None" "256" "256" "5"

log "Mixed 500M evals launched."
