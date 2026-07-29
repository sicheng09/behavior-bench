#!/usr/bin/env bash
# Wait for pure PPO+IDM50 single-GPU 500M training, then eval vs IDM.
set -euo pipefail

cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

REPO_ROOT="$(pwd)"
VAL_ROOT="$REPO_ROOT/.logs/val/run3"
RESULTS_ROOT="$VAL_ROOT/results"
mkdir -p "$RESULTS_ROOT"

TAG="${TAG:-ppo_idm50_500m}"
WATCH_LOG="$VAL_ROOT/wait_and_eval_${TAG}_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCH_LOG"
}

wait_for_session() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    log "Waiting for training session: $session"
    sleep 60
  done
}

wait_for_file() {
  local path="$1"
  while [ ! -f "$path" ]; do
    log "Waiting for checkpoint: $path"
    sleep 60
  done
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
    sleep 60
  done
}

TRAIN_SESSION="${TRAIN_SESSION:-ppo-idm50-drive-lstm-500m-gpu2}"
WANDB_RUN_ID="${WANDB_RUN_ID:?Set WANDB_RUN_ID to the training W&B run id}"
EXP_DIR="$REPO_ROOT/experiments/puffer_drive_${WANDB_RUN_ID}"
TRAIN_DIR_NAME="${TRAIN_DIR_NAME:-ppo_idm50_drive_lstm_500m_run3}"
POLICY_DIR_NAME="${POLICY_DIR_NAME:-Drive_Recurrent}"
EVAL_GPU="${EVAL_GPU:-2}"
EVAL_SESSION_PREFIX="${EVAL_SESSION_PREFIX:-eval-run3-ppo-idm50-vs-idm}"
MIN_EPOCH="${MIN_EPOCH:-954}"
RESOLVE_CKPT="$VAL_ROOT/resolve_final_single_policy_checkpoint.sh"

log "Watch log: $WATCH_LOG"
log "Tag: $TAG"
log "Train session: $TRAIN_SESSION"
log "Expected experiment dir: $EXP_DIR"
log "Min checkpoint epoch: $MIN_EPOCH (model_puffer_drive_*.pt)"
log "Eval: Drive/Recurrent vs IDM on pufferinter -> ${RESULTS_ROOT}/${TRAIN_DIR_NAME}/${POLICY_DIR_NAME}"

wait_for_session "$TRAIN_SESSION"
log "Training session ended."

log "Waiting for experiment dir and FINAL checkpoint (epoch>=$MIN_EPOCH)."
while [ ! -d "$EXP_DIR" ]; do
  log "Waiting for experiment dir: $EXP_DIR"
  sleep 60
done

WEIGHTS="$("$RESOLVE_CKPT" "$EXP_DIR" "$MIN_EPOCH")"
if [ -z "$WEIGHTS" ] || [ ! -f "$WEIGHTS" ]; then
  log "ERROR: failed to resolve final checkpoint under $EXP_DIR"
  exit 1
fi
wait_for_file "$WEIGHTS"
log "Using FINAL checkpoint: $WEIGHTS"

wait_for_gpu_free "$EVAL_GPU"

TS="$(date +%Y%m%d_%H%M%S)"
EVAL_SESSION="${EVAL_SESSION_PREFIX}-gpu${EVAL_GPU}-${TS}"
tmux new-session -d -s "$EVAL_SESSION" \
  "bash -lc 'MAP_IDS=all $REPO_ROOT/.logs/val/run3/run_one_policy_eval.sh $TRAIN_DIR_NAME $POLICY_DIR_NAME $WEIGHTS Drive Recurrent 256 256 $EVAL_GPU'"
log "Launched eval: $EVAL_SESSION on GPU $EVAL_GPU"
log "Wrapper/results: $RESULTS_ROOT/$TRAIN_DIR_NAME/$POLICY_DIR_NAME/"
log "Done launching. Attach with: tmux attach -t $EVAL_SESSION"
