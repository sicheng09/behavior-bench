#!/usr/bin/env bash
# try_Multi: ego from scratch + frozen ConservativeMix-B partner (policy_1).
#
# Ego optimizer geometry matches H0 / try_IDM mb65k alignment:
#   batch=1048576, total=1B, minibatch=65536, mix 50:50 shared
#   → ego ≈524288 env steps / outer epoch, ≈16 opt.step, ≈32768 ego samples/step
# Partner: load plvd8yph policy_1 @ epoch 954, trainable=False (no opt.step).
# See ../README.md and try_IDM/train/README_optimizer_alignment.md
set -euo pipefail

cd /home/fanyuqi/wsc/behavior-bench

export TMUX_SESSION="${TMUX_SESSION:-try-multi-ego-scratch-frozenB-gpu1}"

tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
tmux new-session -d -s "$TMUX_SESSION" "bash -lc '
set -euo pipefail

cd /home/fanyuqi/wsc/behavior-bench
source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"
conda activate behavior-bench

export REPO_ROOT=\"\$(pwd)\"
export DRIVE_BINARIES_DATA_ROOT=\"\$REPO_ROOT/resources/drive/binaries\"
export LOG_DIR=\"\$REPO_ROOT/.logs/train/run4/try_Multi/train\"
mkdir -p \"\$LOG_DIR\"

# ===== Match H0 ego: 524288/update, 500M steps, 16 opt steps/epoch, ~32768 ego/step =====
export GPU_IDS=\"${GPU_IDS:-1}\"
export RUN_NAME=\"ego_scratch_frozen_B_partner_mb65k_1b_tryMulti\"
export CONFIG_PATH=\"pufferlib/config/ocean/drive_conservative_mix.ini\"
export ENV_NAME=\"puffer_drive_conservative_mix\"
export ENV_NUM_MAPS=\"10000\"
export TRAIN_BATCH_SIZE=\"1048576\"
export TOTAL_TIMESTEPS=\"1000000000\"
export MINIBATCH_SIZE=\"65536\"
export BPTT_HORIZON=\"32\"
export UPDATE_EPOCHS=\"1\"

export ENABLE_MIX_TRAFFIC=\"False\"
export PPO_FRACTION=\"1.0\"
export IDM_FRACTION=\"0.0\"
export EXPERT_FRACTION=\"0.0\"

# Env role: partner still uses shaping reward channel (not optimized; frozen).
export PARTNER_MODE=\"reward_shaping\"
export PARTNER_MAX_ABS_STEER=\"0.333\"
export PARTNER_TARGET_HEADWAY=\"1.8\"

# Frozen partner snapshot from C-B mb65k (plvd8yph) policy_1 @ 000954
export FROZEN_PARTNER_PATH=\"\$REPO_ROOT/.logs/train/run4/try_Multi/weights/plvd8yph_policy_1_000954.pt\"

export EVAL_SPLIT=\"validation\"
export EVAL_NUM_MAPS=\"20\"
export ENABLE_WOSAC_REALISM=\"False\"
export ENABLE_HUMAN_REPLAY=\"False\"

export WANDB_PROJECT=\"behavior-bench\"
export WANDB_GROUP=\"run4-try-Multi-frozen-partner\"

export CUDA_VISIBLE_DEVICES=\"\$GPU_IDS\"
export TRAIN_LOG=\"\$LOG_DIR/\${RUN_NAME}_\$(date +%Y%m%d_%H%M%S).log\"

{
  echo \"==== try_Multi · ego scratch + frozen C-B partner (mb65k; H0-aligned ego) ====\"
  echo \"date=\$(date -Iseconds)\"
  echo \"TMUX_SESSION=$TMUX_SESSION\"
  echo \"TRAIN_LOG=\$TRAIN_LOG\"
  echo \"GPU_IDS=\$GPU_IDS\"
  echo \"RUN_NAME=\$RUN_NAME\"
  echo \"ENV_NAME=\$ENV_NAME\"
  echo \"TRAIN_BATCH_SIZE=\$TRAIN_BATCH_SIZE\"
  echo \"TOTAL_TIMESTEPS=\$TOTAL_TIMESTEPS\"
  echo \"outer_epochs ~= \$((TOTAL_TIMESTEPS / TRAIN_BATCH_SIZE))\"
  echo \"opt_steps_per_epoch ~= \$((TRAIN_BATCH_SIZE / MINIBATCH_SIZE))  # expect 16\"
  echo \"MINIBATCH_SIZE=\$MINIBATCH_SIZE  # ego≈32768/step under 50:50\"
  echo \"BPTT=\$BPTT_HORIZON UPDATE_EPOCHS=\$UPDATE_EPOCHS\"
  echo \"partner_mode=\$PARTNER_MODE headway=\$PARTNER_TARGET_HEADWAY\"
  echo \"mix_ppo_policy_trainable=True,False\"
  echo \"FROZEN_PARTNER_PATH=\$FROZEN_PARTNER_PATH\"
  echo \"source_run=plvd8yph (C-B mb65k policy_1 @ 000954)\"
  echo \"alignment=try_IDM/train/README_optimizer_alignment.md\"
  echo \"=============================================================\"
  ls -la \"\$FROZEN_PARTNER_PATH\"
} | tee \"\$TRAIN_LOG\"

test -f \"\$FROZEN_PARTNER_PATH\"

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
  --env.partner-mode \"\$PARTNER_MODE\" \
  --env.partner-max-abs-steer \"\$PARTNER_MAX_ABS_STEER\" \
  --env.partner-target-headway \"\$PARTNER_TARGET_HEADWAY\" \
  --train.mix-ppo True \
  --train.mix-ppo-sampling shared \
  --train.mix-ppo-policy-mix \"ego:0.5,partner:0.5\" \
  --train.mix-ppo-policy-names \"Drive,Drive\" \
  --train.mix-ppo-rnn-names \"Recurrent,Recurrent\" \
  --train.mix-ppo-policy-paths \",\$FROZEN_PARTNER_PATH\" \
  --train.mix-ppo-policy-trainable \"True,False\" \
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
echo "Logs:   /home/fanyuqi/wsc/behavior-bench/.logs/train/run4/try_Multi/train/"
