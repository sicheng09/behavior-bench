#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

REPO_ROOT="$(pwd)"
LOG_DIR="$REPO_ROOT/.logs/train/run1/500M"
mkdir -p "$LOG_DIR"

WATCH_LOG="$LOG_DIR/wait_and_start_mixed_500m_$(date +%Y%m%d_%H%M%S).log"

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

start_mixed_a() {
  local session="run1-500m-mixed-a-gpu0-$(date +%Y%m%d_%H%M%S)"
  tmux new-session -d -s "$session" "bash -lc '
set -euo pipefail
cd \"$HOME/wsc/behavior-bench\"
source \"$HOME/miniconda3/etc/profile.d/conda.sh\"
conda activate behavior-bench
export REPO_ROOT=\"\$(pwd)\"
export DRIVE_BINARIES_DATA_ROOT=\"\$REPO_ROOT/resources/drive/binaries\"
export LOG_DIR=\"\$REPO_ROOT/.logs/train/run1/500M\"
export CUDA_VISIBLE_DEVICES=0
export RUN_NAME=\"mix_intelligence_3level_1500m_batch3x_run1_500m\"
export TRAIN_LOG=\"\$LOG_DIR/\${RUN_NAME}_\$(date +%Y%m%d_%H%M%S).log\"
echo \"TRAIN_LOG=\$TRAIN_LOG\"
echo \"POLICIES=Drive,Drive,DriveLite RNN=Recurrent,None,None TOTAL=1.5B BATCH=1572864\"
puffer train puffer_drive \
  --config pufferlib/config/ocean/drive.ini \
  --train.name \"\$RUN_NAME\" \
  --train.batch-size 1572864 \
  --train.total-timesteps 1500000000 \
  --env.num-maps 10000 \
  --env.mix-traffic False \
  --env.ppo-fraction 1.0 \
  --env.idm-fraction 0.0 \
  --env.expert-fraction 0.0 \
  --train.mix-ppo True \
  --train.mix-ppo-policy-mix \"l3:1,l2:1,l1:1\" \
  --train.mix-ppo-policy-names \"Drive,Drive,DriveLite\" \
  --train.mix-ppo-rnn-names \"Recurrent,None,None\" \
  --train.mix-ppo-policy-paths \",,\" \
  --train.mix-ppo-policy-trainable \"True,True,True\" \
  --eval.split validation \
  --eval.num-maps 20 \
  --eval.wosac-realism-eval False \
  --eval.human-replay-eval False \
  --wandb \
  --wandb-project behavior-bench \
  --wandb-group drive-lite-500m-run1 \
  2>&1 | tee -a \"\$TRAIN_LOG\"
'"
  log "Started Mixed A session: $session"
}

start_mixed_b() {
  local session="run1-500m-mixed-b-gpu1-$(date +%Y%m%d_%H%M%S)"
  tmux new-session -d -s "$session" "bash -lc '
set -euo pipefail
cd \"$HOME/wsc/behavior-bench\"
source \"$HOME/miniconda3/etc/profile.d/conda.sh\"
conda activate behavior-bench
export REPO_ROOT=\"\$(pwd)\"
export DRIVE_BINARIES_DATA_ROOT=\"\$REPO_ROOT/resources/drive/binaries\"
export LOG_DIR=\"\$REPO_ROOT/.logs/train/run1/500M\"
export CUDA_VISIBLE_DEVICES=1
export RUN_NAME=\"mix_drive_lstm_drivelite_lstm224_drive_nolstm_1500m_batch3x_run1_500m\"
export TRAIN_LOG=\"\$LOG_DIR/\${RUN_NAME}_\$(date +%Y%m%d_%H%M%S).log\"
echo \"TRAIN_LOG=\$TRAIN_LOG\"
echo \"POLICIES=Drive,DriveLite,Drive RNN=Recurrent,Recurrent,None TOTAL=1.5B BATCH=1572864\"
puffer train puffer_drive \
  --config pufferlib/config/ocean/drive.ini \
  --train.name \"\$RUN_NAME\" \
  --train.batch-size 1572864 \
  --train.total-timesteps 1500000000 \
  --env.num-maps 10000 \
  --env.mix-traffic False \
  --env.ppo-fraction 1.0 \
  --env.idm-fraction 0.0 \
  --env.expert-fraction 0.0 \
  --train.mix-ppo True \
  --train.mix-ppo-policy-mix \"drive_lstm:1,drivelite_lstm:1,drive_ff:1\" \
  --train.mix-ppo-policy-names \"Drive,DriveLite,Drive\" \
  --train.mix-ppo-rnn-names \"Recurrent,Recurrent,None\" \
  --train.mix-ppo-rnn-input-sizes \"256,224,0\" \
  --train.mix-ppo-rnn-hidden-sizes \"256,224,0\" \
  --train.mix-ppo-policy-paths \",,\" \
  --train.mix-ppo-policy-trainable \"True,True,True\" \
  --eval.split validation \
  --eval.num-maps 20 \
  --eval.wosac-realism-eval False \
  --eval.human-replay-eval False \
  --wandb \
  --wandb-project behavior-bench \
  --wandb-group drive-lite-500m-run1 \
  2>&1 | tee -a \"\$TRAIN_LOG\"
'"
  log "Started Mixed B session: $session"
}

HOMO_SESSIONS=(
  "run1-500m-drive-nolstm-gpu0-20260714_120054"
  "run1-500m-drivelite-nolstm-gpu1-20260714_120104"
  "run1-500m-drivelite-lstm224-gpu2-20260714_120117"
)

log "Waiting for 500M homogeneous runs to finish."
for session in "${HOMO_SESSIONS[@]}"; do
  wait_for_session "$session"
done

log "Homogeneous 500M runs finished. Waiting for GPU0/GPU1."
wait_for_gpus_free 0 1
start_mixed_a
start_mixed_b
