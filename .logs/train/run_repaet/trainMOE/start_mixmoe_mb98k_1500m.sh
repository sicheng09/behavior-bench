#!/usr/bin/env bash
# MIXMOE 1:1:1 (Drive, DriveMoE, DriveMoE3) with H0-aligned optimizer geometry.
# Per-policy budget matches run4 homogeneous 500M:
#   batch×3, steps×3, minibatch×3 → 16 opt steps/epoch, ~32768 samples/policy/step.
# See ../README.md and run4/try_IDM/train/README_optimizer_alignment.md
set -euo pipefail

cd /home/fanyuqi/wsc/behavior-bench

export TMUX_SESSION="${TMUX_SESSION:-mixmoe-mb98k-1500m-gpu1}"

tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
tmux new-session -d -s "$TMUX_SESSION" "bash -lc '
set -euo pipefail

cd /home/fanyuqi/wsc/behavior-bench
source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"
conda activate behavior-bench

export REPO_ROOT=\"\$(pwd)\"
export DRIVE_BINARIES_DATA_ROOT=\"\$REPO_ROOT/resources/drive/binaries\"
export LOG_DIR=\"\$REPO_ROOT/.logs/train/run_repaet/trainMOE\"
mkdir -p \"\$LOG_DIR\"

# ===== Match H0 per-policy: 524288/update, 500M steps, 16 opt steps/epoch, ~32768/step =====
export GPU_IDS=\"${GPU_IDS:-1}\"
export RUN_NAME=\"mix_moe_drive_moe_moe3_mb98k_1500m_run_repaet\"
export CONFIG_PATH=\"pufferlib/config/ocean/drive.ini\"
export ENV_NAME=\"puffer_drive\"
export ENV_NUM_MAPS=\"10000\"
export TRAIN_BATCH_SIZE=\"1572864\"
export TOTAL_TIMESTEPS=\"1500000000\"
export MINIBATCH_SIZE=\"98304\"
export BPTT_HORIZON=\"32\"
export UPDATE_EPOCHS=\"1\"

export ENABLE_MIX_TRAFFIC=\"False\"
export PPO_FRACTION=\"1.0\"
export IDM_FRACTION=\"0.0\"
export EXPERT_FRACTION=\"0.0\"

export EVAL_SPLIT=\"validation\"
export EVAL_NUM_MAPS=\"20\"
export ENABLE_WOSAC_REALISM=\"False\"
export ENABLE_HUMAN_REPLAY=\"False\"

export WANDB_PROJECT=\"behavior-bench\"
export WANDB_GROUP=\"run_repaet-mixmoe\"

export CUDA_VISIBLE_DEVICES=\"\$GPU_IDS\"
export TRAIN_LOG=\"\$LOG_DIR/\${RUN_NAME}_\$(date +%Y%m%d_%H%M%S).log\"

{
  echo \"==== run_repaet · MIXMOE mb98k 1500M (H0-aligned optimizer, 500M/policy) ====\"
  echo \"date=\$(date -Iseconds)\"
  echo \"TMUX_SESSION=$TMUX_SESSION\"
  echo \"TRAIN_LOG=\$TRAIN_LOG\"
  echo \"GPU_IDS=\$GPU_IDS\"
  echo \"RUN_NAME=\$RUN_NAME\"
  echo \"ENV_NAME=\$ENV_NAME\"
  echo \"CONFIG_PATH=\$CONFIG_PATH\"
  echo \"mix=drive:1,moe:1,moe3:1\"
  echo \"policies=Drive,DriveMoE,DriveMoE3 + Recurrent×3\"
  echo \"ENV_NUM_MAPS=\$ENV_NUM_MAPS\"
  echo \"mix_traffic=\$ENABLE_MIX_TRAFFIC\"
  echo \"TRAIN_BATCH_SIZE=\$TRAIN_BATCH_SIZE  # 3×524288\"
  echo \"TOTAL_TIMESTEPS=\$TOTAL_TIMESTEPS    # 3×500M\"
  echo \"outer_epochs ~= \$((TOTAL_TIMESTEPS / TRAIN_BATCH_SIZE))\"
  echo \"MINIBATCH_SIZE=\$MINIBATCH_SIZE      # 3×32768\"
  echo \"opt_steps_per_epoch ~= \$((TRAIN_BATCH_SIZE / MINIBATCH_SIZE))  # expect 16\"
  echo \"per_policy_samples_per_step ~= \$((MINIBATCH_SIZE / 3))         # expect 32768\"
  echo \"BPTT=\$BPTT_HORIZON UPDATE_EPOCHS=\$UPDATE_EPOCHS\"
  echo \"ref_h0=.logs/train/run4/problem homogeneous_drive_lstm_500m_batch1x\"
  echo \"ref_old_mixmoe3x=.logs/try/MOE (batch×3 only; minibatch not scaled)\"
  echo \"ref_align=.logs/train/run4/try_IDM/train/README_optimizer_alignment.md\"
  echo \"===========================================================================\"
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
  --env.mix-traffic \"\$ENABLE_MIX_TRAFFIC\" \
  --env.ppo-fraction \"\$PPO_FRACTION\" \
  --env.idm-fraction \"\$IDM_FRACTION\" \
  --env.expert-fraction \"\$EXPERT_FRACTION\" \
  --train.mix-ppo True \
  --train.mix-ppo-policy-mix \"drive:1,moe:1,moe3:1\" \
  --train.mix-ppo-policy-names \"Drive,DriveMoE,DriveMoE3\" \
  --train.mix-ppo-rnn-names \"Recurrent,Recurrent,Recurrent\" \
  --train.mix-ppo-policy-paths \",,\" \
  --train.mix-ppo-policy-trainable \"True,True,True\" \
  --eval.split \"\$EVAL_SPLIT\" \
  --eval.num-maps \"\$EVAL_NUM_MAPS\" \
  --eval.wosac-realism-eval \"\$ENABLE_WOSAC_REALISM\" \
  --eval.human-replay-eval \"\$ENABLE_HUMAN_REPLAY\" \
  --wandb \
  --wandb-project \"\$WANDB_PROJECT\" \
  --wandb-group \"\$WANDB_GROUP\" \
  2>&1 | tee -a \"\$TRAIN_LOG\"
'"

echo "Started tmux session: $TMUX_SESSION"
echo "Attach: tmux attach -t $TMUX_SESSION"
echo "Logs:   /home/fanyuqi/wsc/behavior-bench/.logs/train/run_repaet/trainMOE/"
