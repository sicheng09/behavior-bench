#!/usr/bin/env bash
# Repeat · MIXMOE Drive/DriveMoE/DriveMoE3 stratified (same recipe, new seed/run).
set -euo pipefail
cd "$HOME/wsc/behavior-bench"

GPU_IDS="${GPU_IDS:-3}"
RUN_NAME="mix_moe_drive_moe_moe3_stratified_1500m_run_repaet_repeat"
TMUX_SESSION="${TMUX_SESSION:-repeat-mixmoe-strat-gpu${GPU_IDS}}"
RESULT_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$RESULT_ROOT"

tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
tmux new-session -d -s "$TMUX_SESSION" "bash -lc '
set -euo pipefail
cd \$HOME/wsc/behavior-bench
source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"
conda activate behavior-bench
export REPO_ROOT=\"\$(pwd)\"
export DRIVE_BINARIES_DATA_ROOT=\"\$REPO_ROOT/resources/drive/binaries\"
export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export TRAIN_LOG=\"${LOG_DIR}/${RUN_NAME}_\$(date +%Y%m%d_%H%M%S).log\"
mkdir -p \"${LOG_DIR}\"
{
  echo \"==== REPEAT · MIXMOE stratified 1500M ====\"
  echo \"date=\$(date -Iseconds)\"
  echo \"ref_first=o4trtoxc (SMART Coll 4.24% / −37.6%; shared was worse)\"
  echo \"RUN_NAME=${RUN_NAME}\"
  echo \"mix=drive:1,moe:1,moe3:1\"
  echo \"batch=1572864 steps=1500000000 stratified mb32768×16\"
  echo \"========================================================\"
} | tee \"\$TRAIN_LOG\"
puffer train puffer_drive \
  --config pufferlib/config/ocean/drive.ini \
  --train.name \"${RUN_NAME}\" \
  --train.batch-size 1572864 \
  --train.total-timesteps 1500000000 \
  --train.minibatch-size 32768 \
  --train.max-minibatch-size 32768 \
  --train.bptt-horizon 32 \
  --train.update-epochs 1 \
  --env.num-maps 10000 \
  --env.mix-traffic False \
  --env.ppo-fraction 1.0 \
  --env.idm-fraction 0.0 \
  --env.expert-fraction 0.0 \
  --train.mix-ppo True \
  --train.mix-ppo-policy-mix \"drive:1,moe:1,moe3:1\" \
  --train.mix-ppo-policy-names \"Drive,DriveMoE,DriveMoE3\" \
  --train.mix-ppo-rnn-names \"Recurrent,Recurrent,Recurrent\" \
  --train.mix-ppo-policy-paths \",,\" \
  --train.mix-ppo-policy-trainable \"True,True,True\" \
  --train.mix-ppo-sampling stratified \
  --train.mix-ppo-policy-minibatch-sizes \"32768,32768,32768\" \
  --train.mix-ppo-policy-update-steps \"16,16,16\" \
  --eval.split validation \
  --eval.num-maps 20 \
  --eval.wosac-realism-eval False \
  --eval.human-replay-eval False \
  --wandb \
  --wandb-project behavior-bench \
  --wandb-group run_repaet-repeat \
  2>&1 | tee -a \"\$TRAIN_LOG\"
'"
echo "Started $TMUX_SESSION GPU=$GPU_IDS RUN_NAME=$RUN_NAME"
echo "$TMUX_SESSION" >"$RESULT_ROOT/tmux_session.txt"
echo "$RUN_NAME" >"$RESULT_ROOT/train_run_name.txt"
