#!/usr/bin/env bash
# Ablation A: DDP2 500M with lower opponent ego_cost / higher fault.
# Baseline: start_adversarial_mix_ddp2_500m.sh (ego_cost=0.20, fault=1.00).
# This: opponent_mix_low_ego_cost.yaml (ego_cost=0.10, fault=1.50).
set -euo pipefail

MAIN_ROOT="${HOME}/wsc/behavior-bench"
LOG_DIR="${MAIN_ROOT}/.logs/train/run3"
mkdir -p "${LOG_DIR}"

export TMUX_SESSION="${TMUX_SESSION:-adv-mix-drive-lstm-ddp2-500m-low-ego-cost-gpu34}"

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

export GPU_IDS=\"3,4\"
export NPROC_PER_NODE=\"2\"
export RUN_NAME=\"adv_mix_ego_opponent_drive_lstm_ddp2_500m_low_ego_cost_run3\"
export CONFIG_PATH=\"pufferlib/config/ocean/drive_adversarial.ini\"
export ADV_YAML=\"\${MAIN_ROOT}/pufferlib/config/adversarial/opponent_mix_low_ego_cost.yaml\"
export ENV_NUM_MAPS=\"10000\"

export TRAIN_BATCH_SIZE=\"524288\"
export TRAIN_MINIBATCH_SIZE=\"32768\"
export TRAIN_MAX_MINIBATCH_SIZE=\"32768\"
export TOTAL_TIMESTEPS=\"500000000\"

export MIX_PPO_POLICY_MIX=\"ego:0.5, primary_opponent:0.5\"
export MIX_PPO_POLICY_NAMES=\"Drive,Drive\"
export MIX_PPO_RNN_NAMES=\"Recurrent,Recurrent\"
export MIX_PPO_POLICY_PATHS=\",\"
export MIX_PPO_POLICY_TRAINABLE=\"True,True\"

export ENABLE_MIX_TRAFFIC=\"False\"
export PPO_FRACTION=\"1.0\"
export IDM_FRACTION=\"0.0\"
export EXPERT_FRACTION=\"0.0\"

export EVAL_SPLIT=\"validation\"
export EVAL_NUM_MAPS=\"20\"
export ENABLE_WOSAC_REALISM=\"False\"
export ENABLE_HUMAN_REPLAY=\"False\"

export WANDB_PROJECT=\"behavior-bench\"
export WANDB_GROUP=\"adversarial-mix-run3-ddp\"
export CUDA_VISIBLE_DEVICES=\"\${GPU_IDS}\"
export TRAIN_LOG=\"\${LOG_DIR}/\${RUN_NAME}_\$(date +%Y%m%d_%H%M%S).log\"

{
  echo \"==== adversarial mix DDP2 500M low_ego_cost launch ====\"
  echo \"date=\$(date -Is)\"
  echo \"TMUX_SESSION=${TMUX_SESSION}\"
  echo \"TRAIN_LOG=\${TRAIN_LOG}\"
  echo \"GPU_IDS=\${GPU_IDS}\"
  echo \"NPROC_PER_NODE=\${NPROC_PER_NODE}\"
  echo \"RUN_NAME=\${RUN_NAME}\"
  echo \"CONFIG_PATH=\${CONFIG_PATH}\"
  echo \"ADV_YAML=\${ADV_YAML}\"
  echo \"ENV_NUM_MAPS=\${ENV_NUM_MAPS}\"
  echo \"TRAIN_BATCH_SIZE=\${TRAIN_BATCH_SIZE}\"
  echo \"TRAIN_MINIBATCH_SIZE=\${TRAIN_MINIBATCH_SIZE}\"
  echo \"TOTAL_TIMESTEPS=\${TOTAL_TIMESTEPS}\"
  echo \"MIX_PPO_POLICY_MIX=\${MIX_PPO_POLICY_MIX}\"
  echo \"global_batch/update ~= \$((TRAIN_BATCH_SIZE * NPROC_PER_NODE))\"
  echo \"global_agent_steps  ~= \$((TOTAL_TIMESTEPS * NPROC_PER_NODE))\"
  echo \"per_rank_updates    ~= \$((TOTAL_TIMESTEPS / TRAIN_BATCH_SIZE))\"
  echo \"==========================================\"
} | tee -a \"\${TRAIN_LOG}\"

torchrun --standalone --nnodes=1 --nproc-per-node=\"\${NPROC_PER_NODE}\" \
  -m pufferlib.pufferl train puffer_drive_adversarial \
  --config \"\${CONFIG_PATH}\" \
  --train.name \"\${RUN_NAME}\" \
  --train.batch-size \"\${TRAIN_BATCH_SIZE}\" \
  --train.minibatch-size \"\${TRAIN_MINIBATCH_SIZE}\" \
  --train.max-minibatch-size \"\${TRAIN_MAX_MINIBATCH_SIZE}\" \
  --train.total-timesteps \"\${TOTAL_TIMESTEPS}\" \
  --train.mix-ppo True \
  --train.mix-ppo-policy-mix \"\${MIX_PPO_POLICY_MIX}\" \
  --train.mix-ppo-policy-names \"\${MIX_PPO_POLICY_NAMES}\" \
  --train.mix-ppo-rnn-names \"\${MIX_PPO_RNN_NAMES}\" \
  --train.mix-ppo-policy-paths \"\${MIX_PPO_POLICY_PATHS}\" \
  --train.mix-ppo-policy-trainable \"\${MIX_PPO_POLICY_TRAINABLE}\" \
  --env.num-maps \"\${ENV_NUM_MAPS}\" \
  --env.adversarial-config-path \"\${ADV_YAML}\" \
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
