#!/usr/bin/env bash
# Control: pure PPO (Drive/Recurrent) + 50% IDM traffic. No adversarial opponent.
# Single-GPU 500M to match homogeneous Drive+LSTM batch1x budget
# (batch=524288, total_timesteps=500M, one update stream).
# Logs under .logs/train/run3.
set -euo pipefail

MAIN_ROOT="${HOME}/wsc/behavior-bench"
LOG_DIR="${MAIN_ROOT}/.logs/train/run3"
mkdir -p "${LOG_DIR}"

export TMUX_SESSION="${TMUX_SESSION:-ppo-idm50-drive-lstm-500m-gpu2}"

tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
tmux new-session -d -s "${TMUX_SESSION}" "bash -lc '
set -euo pipefail

source \"${HOME}/miniconda3/etc/profile.d/conda.sh\"
conda activate behavior-bench

export MAIN_ROOT=\"${MAIN_ROOT}\"
export LOG_DIR=\"${LOG_DIR}\"
cd \"\${MAIN_ROOT}\"

export DRIVE_BINARIES_DATA_ROOT=\"\${MAIN_ROOT}/resources/drive/binaries\"
export PUFFER_DISABLE_VIDEO=\"1\"

export GPU_IDS=\"2\"
export RUN_NAME=\"ppo_idm50_drive_lstm_500m_run3\"
export CONFIG_PATH=\"pufferlib/config/ocean/drive.ini\"
export ENV_NUM_MAPS=\"10000\"

export TRAIN_BATCH_SIZE=\"524288\"
export TRAIN_MINIBATCH_SIZE=\"32768\"
export TRAIN_MAX_MINIBATCH_SIZE=\"32768\"
export TOTAL_TIMESTEPS=\"500000000\"

export ENABLE_MIX_TRAFFIC=\"True\"
export PPO_FRACTION=\"0.5\"
export IDM_FRACTION=\"0.5\"
export EXPERT_FRACTION=\"0.0\"

export EVAL_SPLIT=\"validation\"
export EVAL_NUM_MAPS=\"20\"
export ENABLE_WOSAC_REALISM=\"False\"
export ENABLE_HUMAN_REPLAY=\"False\"

export WANDB_PROJECT=\"behavior-bench\"
export WANDB_GROUP=\"ppo-idm50-run3\"
export CUDA_VISIBLE_DEVICES=\"\${GPU_IDS}\"
export TRAIN_LOG=\"\${LOG_DIR}/\${RUN_NAME}_\$(date +%Y%m%d_%H%M%S).log\"

{
  echo \"==== pure PPO + IDM50 single-GPU 500M launch ====\"
  echo \"date=\$(date -Is)\"
  echo \"TMUX_SESSION=${TMUX_SESSION}\"
  echo \"TRAIN_LOG=\${TRAIN_LOG}\"
  echo \"GPU_IDS=\${GPU_IDS}\"
  echo \"RUN_NAME=\${RUN_NAME}\"
  echo \"CONFIG_PATH=\${CONFIG_PATH}\"
  echo \"ENV_NUM_MAPS=\${ENV_NUM_MAPS}\"
  echo \"ENABLE_MIX_TRAFFIC=\${ENABLE_MIX_TRAFFIC}\"
  echo \"PPO_FRACTION=\${PPO_FRACTION}\"
  echo \"IDM_FRACTION=\${IDM_FRACTION}\"
  echo \"TRAIN_BATCH_SIZE=\${TRAIN_BATCH_SIZE}\"
  echo \"TOTAL_TIMESTEPS=\${TOTAL_TIMESTEPS}\"
  echo \"mix_ppo=False (single Drive/Recurrent)\"
  echo \"per_rank_updates    ~= \$((TOTAL_TIMESTEPS / TRAIN_BATCH_SIZE))\"
  echo \"NOTE: single GPU — global batch = TRAIN_BATCH_SIZE (matches homo 500M)\"
  echo \"==========================================\"
} | tee -a \"\${TRAIN_LOG}\"

python -m pufferlib.pufferl train puffer_drive \
  --config \"\${CONFIG_PATH}\" \
  --train.name \"\${RUN_NAME}\" \
  --train.batch-size \"\${TRAIN_BATCH_SIZE}\" \
  --train.minibatch-size \"\${TRAIN_MINIBATCH_SIZE}\" \
  --train.max-minibatch-size \"\${TRAIN_MAX_MINIBATCH_SIZE}\" \
  --train.total-timesteps \"\${TOTAL_TIMESTEPS}\" \
  --train.mix-ppo False \
  --env.num-maps \"\${ENV_NUM_MAPS}\" \
  --env.mix-traffic \"\${ENABLE_MIX_TRAFFIC}\" \
  --env.ppo-fraction \"\${PPO_FRACTION}\" \
  --env.idm-fraction \"\${IDM_FRACTION}\" \
  --env.expert-fraction \"\${EXPERT_FRACTION}\" \
  --eval.split \"\${EVAL_SPLIT}\" \
  --eval.num-maps \"\${EVAL_NUM_MAPS}\" \
  --eval.wosac-realism-eval \"\${ENABLE_WOSAC_REALISM}\" \
  --eval.human-replay-eval \"\${ENABLE_HUMAN_REPLAY}\" \
  --wandb \
  --wandb-project \"\${WANDB_PROJECT}\" \
  --wandb-group \"\${WANDB_GROUP}\" \
  2>&1 | tee -a \"\${TRAIN_LOG}\"
'"

echo "Started tmux session: ${TMUX_SESSION}"
echo "Logs: ${LOG_DIR}/"
echo "Attach: tmux attach -t ${TMUX_SESSION}"
