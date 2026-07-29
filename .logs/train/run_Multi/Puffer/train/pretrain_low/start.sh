#!/usr/bin/env bash
# Hybrid pretrain · Low = HybridDriveLiteLow (60°/50m, no LSTM) · 500M
# Launch style aligned with run_repaet/repeat (tmux + tee; avoid script(1)).
set -euo pipefail
cd "$HOME/wsc/behavior-bench"

GPU_IDS="${GPU_IDS:-1}"
RUN_NAME="${RUN_NAME:-hybrid_pretrain_low}"
TMUX_SESSION="${TMUX_SESSION:-multi-pretrain-low-gpu${GPU_IDS}}"
RESULT_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$RESULT_ROOT"
DATA_DIR="${DATA_DIR:-$RESULT_ROOT}"

tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
tmux new-session -d -s "$TMUX_SESSION" "bash -lc '
set -euo pipefail
cd \$HOME/wsc/behavior-bench
source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"
conda activate behavior-bench
export REPO_ROOT=\"\$(pwd)\"
export DRIVE_BINARIES_DATA_ROOT=\"\${DRIVE_BINARIES_DATA_ROOT:-\$REPO_ROOT/pufferlib/resources/drive/binaries}\"
export CUDA_VISIBLE_DEVICES=${GPU_IDS}
# Match run_repaet/repeat: tee captures dashboard frames (mild ANSI).
# Do NOT use script(1) — that dumps full-screen redraws and floods ESC codes.
export PYTHONUNBUFFERED=1
export TRAIN_LOG=\"${LOG_DIR}/${RUN_NAME}_\$(date +%Y%m%d_%H%M%S).log\"
mkdir -p \"${LOG_DIR}\" \"${DATA_DIR}\"
{
  echo \"==== run_Multi · Hybrid pretrain LOW 500M ====\"
  echo \"date=\$(date -Iseconds)\"
  echo \"TMUX_SESSION=${TMUX_SESSION}\"
  echo \"GPU_IDS=${GPU_IDS}\"
  echo \"RUN_NAME=${RUN_NAME}\"
  echo \"policy=HybridDriveLiteLow  rnn=None\"
  echo \"perception=60deg / 50m\"
  echo \"policy_size input=16 hidden=80\"
  echo \"env.num-maps=10000  total_timesteps=500000000\"
  echo \"DATA_DIR=${DATA_DIR}\"
  echo \"DRIVE_BINARIES_DATA_ROOT=\$DRIVE_BINARIES_DATA_ROOT\"
  echo \"================================================================\"
} | tee \"\$TRAIN_LOG\"
puffer train puffer_drive \
  --config pufferlib/config/ocean/drive.ini \
  --train.name \"${RUN_NAME}\" \
  --train.data-dir \"${DATA_DIR}\" \
  --policy-name HybridDriveLiteLow \
  --rnn-name None \
  --policy.input-size 16 \
  --policy.hidden-size 80 \
  --env.num-maps 10000 \
  --train.total-timesteps 500000000 \
  --eval.wosac-realism-eval False \
  --eval.human-replay-eval False \
  --wandb \
  --wandb-project behavior-bench \
  --wandb-group run_Multi-hybrid-pretrain \
  2>&1 | tee -a \"\$TRAIN_LOG\"
'"
echo "Started $TMUX_SESSION GPU=$GPU_IDS RUN_NAME=$RUN_NAME"
echo "$TMUX_SESSION" >"$RESULT_ROOT/tmux_session.txt"
echo "$RUN_NAME" >"$RESULT_ROOT/train_run_name.txt"
