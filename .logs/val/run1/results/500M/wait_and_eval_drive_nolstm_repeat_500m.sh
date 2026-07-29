#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

REPO_ROOT="$(pwd)"
VAL_ROOT="$REPO_ROOT/.logs/val/run1/results/500M"
WATCH_LOG="$VAL_ROOT/wait_and_eval_drive_nolstm_repeat_500m_$(date +%Y%m%d_%H%M%S).log"

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
  for gpu in 2 3 4 5 6 7 1 0; do
    if ! gpu_has_compute_process "$gpu"; then
      echo "$gpu"
      return 0
    fi
  done
  return 1
}

SESSION="run1-500m-drive-nolstm-repeat-gpu1-20260714_170451"
RUN_ID="REPLACE_RUN_ID"

wait_for_session "$SESSION"

if [ "$RUN_ID" = "REPLACE_RUN_ID" ]; then
  RUN_ID="$(grep -m1 'wandb: setting up run' .logs/train/run1/500M/homogeneous_drive_nolstm_500m_batch1x_run1_repeat_*.log | awk '{print $NF}')"
fi

CKPT="$REPO_ROOT/experiments/puffer_drive_${RUN_ID}/model_puffer_drive_000954.pt"
wait_for_file "$CKPT"

while ! GPU_ID="$(pick_free_gpu)"; do
  log "Waiting for any GPU to be free."
  sleep 300
done

log "Launching Drive_NoLSTM repeat eval on GPU $GPU_ID."
tmux new-session -d -s "eval500m-drive-nolstm-repeat-gpu${GPU_ID}-$(date +%Y%m%d_%H%M%S)" \
  "bash -lc 'MAP_IDS=all .logs/val/run1/results/500M/run_one_policy_eval_500m.sh homogeneous_drive_nolstm_500m_batch1x_run1_repeat Drive_NoLSTM $CKPT Drive None 256 256 $GPU_ID'"
