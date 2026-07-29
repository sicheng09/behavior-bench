#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

REPO_ROOT="$(pwd)"
VAL_ROOT="$REPO_ROOT/.logs/val/run1/results/500M"
WATCH_LOG="$VAL_ROOT/wait_and_eval_mixed_b_repeat_p2_500m_$(date +%Y%m%d_%H%M%S).log"

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

pick_free_gpu() {
  for gpu in 1 2 3 4 5 6 7 0; do
    if ! gpu_has_compute_process "$gpu"; then
      echo "$gpu"
      return 0
    fi
  done
  return 1
}

SESSION="run1-500m-mixed-b-repeat-gpu0-20260714_165152"
RUN_ID="REPLACE_RUN_ID"

log "Waiting for Mixed B 500M repeat session."
wait_for_session "$SESSION"

if [ "$RUN_ID" = "REPLACE_RUN_ID" ]; then
  RUN_ID="$(grep -m1 'wandb: setting up run' .logs/train/run1/500M/mix_drive_lstm_drivelite_lstm224_drive_nolstm_1500m_batch3x_run1_500m_repeat_*.log | awk '{print $NF}')"
fi

CKPT="$REPO_ROOT/experiments/puffer_drive_${RUN_ID}/model_policy_2_puffer_drive_000954.pt"
wait_for_file "$CKPT"

while ! GPU_ID="$(pick_free_gpu)"; do
  log "Waiting for any GPU to be free."
  sleep 300
done

log "Launching Mixed B repeat p2 eval on GPU $GPU_ID."
tmux new-session -d -s "eval500m-mix-b-repeat-p2-gpu${GPU_ID}-$(date +%Y%m%d_%H%M%S)" \
  "bash -lc 'MAP_IDS=all .logs/val/run1/results/500M/run_one_policy_eval_500m.sh mix_drive_lstm_drivelite_lstm224_drive_nolstm_1500m_batch3x_run1_500m_repeat p2_Drive_NoLSTM $CKPT Drive None 256 256 $GPU_ID'"
