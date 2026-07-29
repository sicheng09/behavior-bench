#!/usr/bin/env bash
# Perception mixed low/mid/high · 500M/policy H0-aligned (shared or stratified).
# Usage: SAMPLING=shared|stratified GPU_IDS=0 bash start_perception_mix_1500m.sh
set -euo pipefail

cd /home/fanyuqi/wsc/behavior-bench

SAMPLING="${SAMPLING:-shared}"
GPU_IDS="${GPU_IDS:-0}"

MIX_LABEL="low:1,mid:1,high:1"
POLICY_NAMES="DrivePerceptionLow,DrivePerceptionMid,Drive"
RNN_NAMES="Recurrent,Recurrent,Recurrent"

case "$SAMPLING" in
  shared)
    RUN_TAG="mb98k_shared_1500m"
    MINIBATCH_SIZE="98304"
    EXTRA_MIX_ARGS_STR="--train.mix-ppo-sampling shared"
    GEO_NOTE="shared global mb=98304 (=3×32768); ~32768/policy/step expect"
    ;;
  stratified)
    RUN_TAG="stratified_1500m"
    MINIBATCH_SIZE="32768"
    EXTRA_MIX_ARGS_STR="--train.mix-ppo-sampling stratified --train.mix-ppo-policy-minibatch-sizes 32768,32768,32768 --train.mix-ppo-policy-update-steps 16,16,16"
    GEO_NOTE="stratified policy_mb=32768×3 update_steps=16×3; pool~524288/policy"
    ;;
  *)
    echo "SAMPLING must be shared or stratified, got: $SAMPLING" >&2
    exit 1
    ;;
esac

export TMUX_SESSION="${TMUX_SESSION:-perception-${RUN_TAG}-gpu${GPU_IDS}}"
RUN_NAME="perception_mix_low_mid_high_${RUN_TAG}_run_repaet"

tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
tmux new-session -d -s "$TMUX_SESSION" "bash -lc '
set -euo pipefail

cd /home/fanyuqi/wsc/behavior-bench
source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"
conda activate behavior-bench

export REPO_ROOT=\"\$(pwd)\"
export DRIVE_BINARIES_DATA_ROOT=\"\$REPO_ROOT/resources/drive/binaries\"
export LOG_DIR=\"\$REPO_ROOT/.logs/train/run_repaet/trainPerception\"
mkdir -p \"\$LOG_DIR\"

export CUDA_VISIBLE_DEVICES=\"${GPU_IDS}\"
export RUN_NAME=\"${RUN_NAME}\"
export CONFIG_PATH=\"pufferlib/config/ocean/drive.ini\"
export ENV_NAME=\"puffer_drive\"
export ENV_NUM_MAPS=\"10000\"
export TRAIN_BATCH_SIZE=\"1572864\"
export TOTAL_TIMESTEPS=\"1500000000\"
export MINIBATCH_SIZE=\"${MINIBATCH_SIZE}\"
export BPTT_HORIZON=\"32\"
export UPDATE_EPOCHS=\"1\"
export WANDB_PROJECT=\"behavior-bench\"
export WANDB_GROUP=\"run_repaet-perception\"
export TRAIN_LOG=\"\$LOG_DIR/\${RUN_NAME}_\$(date +%Y%m%d_%H%M%S).log\"

{
  echo \"==== run_repaet/trainPerception · Perception mix (${SAMPLING}) 1500M ====\"
  echo \"date=\$(date -Iseconds)\"
  echo \"TMUX_SESSION=${TMUX_SESSION}\"
  echo \"TRAIN_LOG=\$TRAIN_LOG\"
  echo \"GPU_IDS=${GPU_IDS}\"
  echo \"RUN_NAME=\$RUN_NAME\"
  echo \"SAMPLING=${SAMPLING}\"
  echo \"mix=${MIX_LABEL}\"
  echo \"policies=${POLICY_NAMES}\"
  echo \"rnn_names=${RNN_NAMES}\"
  echo \"TRAIN_BATCH_SIZE=\$TRAIN_BATCH_SIZE TOTAL_TIMESTEPS=\$TOTAL_TIMESTEPS\"
  echo \"outer_epochs ~= \$((TOTAL_TIMESTEPS / TRAIN_BATCH_SIZE))\"
  echo \"MINIBATCH_SIZE=\$MINIBATCH_SIZE\"
  echo \"geometry: ${GEO_NOTE}\"
  echo \"ref_old=.logs/try/Perception (batch×3 only; minibatch not scaled)\"
  echo \"ref_h0=.logs/train/run4/problem homogeneous 500M\"
  echo \"========================================================================\"
} | tee \"\$TRAIN_LOG\"

puffer train \"\$ENV_NAME\" \
  --config \"\$CONFIG_PATH\" \
  --train.name \"\$RUN_NAME\" \
  --train.batch-size \"\$TRAIN_BATCH_SIZE\" \
  --train.total-timesteps \"\$TOTAL_TIMESTEPS\" \
  --train.minibatch-size \"\$MINIBATCH_SIZE\" \
  --train.max-minibatch-size \"\$MINIBATCH_SIZE\" \
  --train.bptt-horizon \"\$BPTT_HORIZON\" \
  --train.update-epochs \"\$UPDATE_EPOCHS\" \
  --env.num-maps \"\$ENV_NUM_MAPS\" \
  --env.mix-traffic False \
  --env.ppo-fraction 1.0 \
  --env.idm-fraction 0.0 \
  --env.expert-fraction 0.0 \
  --train.mix-ppo True \
  --train.mix-ppo-policy-mix \"${MIX_LABEL}\" \
  --train.mix-ppo-policy-names \"${POLICY_NAMES}\" \
  --train.mix-ppo-rnn-names \"${RNN_NAMES}\" \
  --train.mix-ppo-policy-paths \",,\" \
  --train.mix-ppo-policy-trainable \"True,True,True\" \
  ${EXTRA_MIX_ARGS_STR} \
  --eval.split validation \
  --eval.num-maps 20 \
  --eval.wosac-realism-eval False \
  --eval.human-replay-eval False \
  --wandb \
  --wandb-project \"\$WANDB_PROJECT\" \
  --wandb-group \"\$WANDB_GROUP\" \
  2>&1 | tee -a \"\$TRAIN_LOG\"
'"

echo "Started tmux session: $TMUX_SESSION"
echo "  SAMPLING=$SAMPLING GPU=$GPU_IDS"
echo "  Attach: tmux attach -t $TMUX_SESSION"
echo "  Logs:   /home/fanyuqi/wsc/behavior-bench/.logs/train/run_repaet/trainPerception/"
