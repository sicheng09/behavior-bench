#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

REPO_ROOT="$(pwd)"
LOG_DIR="$REPO_ROOT/.logs/train/run2/500M"
mkdir -p "$LOG_DIR"
WATCH_LOG="$LOG_DIR/wait_and_start_ddp3_500m_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCH_LOG"
}

wait_for_session() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    log "Waiting for session: $session"
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

WAIT_SESSIONS=(
  "run2-500m-perception-low-gpu1-20260714_183232"
  "run2-500m-perception-mid-gpu3-20260714_183248"
  "run2-500m-perception-high-gpu4-20260714_183300"
)

for session in "${WAIT_SESSIONS[@]}"; do
  wait_for_session "$session"
done

wait_for_gpus_free 1 3 4

SESSION="run2-500m-perception-mixed-ddp3-gpu134-$(date +%Y%m%d_%H%M%S)"
tmux new-session -d -s "$SESSION" "bash -lc '
set -euo pipefail
cd \"$HOME/wsc/behavior-bench\"
source \"$HOME/miniconda3/etc/profile.d/conda.sh\"
conda activate behavior-bench
export REPO_ROOT=\"\$(pwd)\"
export DRIVE_BINARIES_DATA_ROOT=\"\$REPO_ROOT/resources/drive/binaries\"
export LOG_DIR=\"\$REPO_ROOT/.logs/train/run2/500M\"
export CUDA_VISIBLE_DEVICES=\"1,3,4\"
export NPROC_PER_NODE=\"3\"
export RUN_NAME=\"perception_mixed_low_mid_high_ddp3_global1500m_run2_500m\"
export TRAIN_LOG=\"\$LOG_DIR/\${RUN_NAME}_\$(date +%Y%m%d_%H%M%S).log\"
echo \"TRAIN_LOG=\$TRAIN_LOG\"
echo \"GPU_IDS=\$CUDA_VISIBLE_DEVICES\"
echo \"PER_RANK_TOTAL=500M PER_RANK_BATCH=524288 GLOBAL~=1.5B\"
torchrun --standalone --nnodes=1 --nproc-per-node=\"\$NPROC_PER_NODE\" \
  -m pufferlib.pufferl train puffer_drive \
  --config pufferlib/config/ocean/drive.ini \
  --train.name \"\$RUN_NAME\" \
  --train.batch-size 524288 \
  --train.total-timesteps 500000000 \
  --env.num-maps 10000 \
  --env.mix-traffic False \
  --env.ppo-fraction 1.0 \
  --env.idm-fraction 0.0 \
  --env.expert-fraction 0.0 \
  --train.mix-ppo True \
  --train.mix-ppo-policy-mix \"low:1,mid:1,high:1\" \
  --train.mix-ppo-policy-names \"DrivePerceptionLow,DrivePerceptionMid,Drive\" \
  --train.mix-ppo-rnn-names \"Recurrent,Recurrent,Recurrent\" \
  --train.mix-ppo-policy-paths \",,\" \
  --train.mix-ppo-policy-trainable \"True,True,True\" \
  --eval.split validation \
  --eval.num-maps 20 \
  --eval.wosac-realism-eval False \
  --eval.human-replay-eval False \
  --wandb \
  --wandb-project behavior-bench \
  --wandb-group perception-500m-run2
'"

log "Started DDP3 500M session: $SESSION"
