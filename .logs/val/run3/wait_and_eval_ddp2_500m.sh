#!/usr/bin/env bash
# Wait for adversarial DDP2 500M baseline training to finish, then eval
# ego (policy 0) vs IDM on pufferinter. Results under .logs/val/run3/results/.
set -euo pipefail

cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

REPO_ROOT="$(pwd)"
VAL_ROOT="$REPO_ROOT/.logs/val/run3"
RESULTS_ROOT="$VAL_ROOT/results"
mkdir -p "$RESULTS_ROOT"

TAG="${TAG:-ddp2_500m}"
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

resolve_latest_policy0() {
  local exp_dir="$1"
  ls -1 "$exp_dir"/model_policy_0_puffer_drive_adversarial_*.pt 2>/dev/null | sort | tail -1
}

TRAIN_SESSION="${TRAIN_SESSION:-adv-mix-drive-lstm-ddp2-500m-gpu34}"
WANDB_RUN_ID="${WANDB_RUN_ID:-wpj0xhn8}"
EXP_DIR="$REPO_ROOT/experiments/puffer_drive_adversarial_${WANDB_RUN_ID}"
TRAIN_DIR_NAME="${TRAIN_DIR_NAME:-adv_mix_ego_opponent_drive_lstm_ddp2_500m_run3}"
POLICY_DIR_NAME="${POLICY_DIR_NAME:-p0_ego_Drive_Recurrent}"
EVAL_GPU="${EVAL_GPU:-3}"
EVAL_SESSION_PREFIX="${EVAL_SESSION_PREFIX:-eval-run3-adv500m-p0-ego-vs-idm}"

log "Watch log: $WATCH_LOG"
log "Tag: $TAG"
log "Train session: $TRAIN_SESSION"
log "Expected experiment dir: $EXP_DIR"
log "Eval: ego policy0 vs IDM on pufferinter -> ${RESULTS_ROOT}/${TRAIN_DIR_NAME}/${POLICY_DIR_NAME}"

wait_for_session "$TRAIN_SESSION"
log "Training session ended."

# Final checkpoint is written on done_training (~epoch 953/954 for 500M).
log "Waiting for experiment dir and ego checkpoint."
while [ ! -d "$EXP_DIR" ]; do
  log "Waiting for experiment dir: $EXP_DIR"
  sleep 60
done

WEIGHTS=""
for _ in $(seq 1 120); do
  WEIGHTS="$(resolve_latest_policy0 "$EXP_DIR" || true)"
  if [ -n "$WEIGHTS" ]; then
    break
  fi
  log "Waiting for model_policy_0_*.pt under $EXP_DIR"
  sleep 60
done

if [ -z "$WEIGHTS" ]; then
  log "ERROR: no ego checkpoint found under $EXP_DIR"
  exit 1
fi

wait_for_file "$WEIGHTS"
log "Using checkpoint: $WEIGHTS"

wait_for_gpu_free "$EVAL_GPU"

TS="$(date +%Y%m%d_%H%M%S)"
EVAL_SESSION="${EVAL_SESSION_PREFIX}-gpu${EVAL_GPU}-${TS}"
tmux new-session -d -s "$EVAL_SESSION" \
  "bash -lc 'MAP_IDS=all $REPO_ROOT/.logs/val/run3/run_one_policy_eval.sh $TRAIN_DIR_NAME $POLICY_DIR_NAME $WEIGHTS Drive Recurrent 256 256 $EVAL_GPU'"
log "Launched eval: $EVAL_SESSION on GPU $EVAL_GPU"
log "Wrapper/results: $RESULTS_ROOT/$TRAIN_DIR_NAME/$POLICY_DIR_NAME/"
log "Done launching. Attach with: tmux attach -t $EVAL_SESSION"
