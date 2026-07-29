#!/usr/bin/env bash
# try_Multi: ego from scratch + frozen C-B mb65k *ego* (plvd8yph policy_0) as partner.
#
# Same H0-aligned geometry as frozen-policy_1 run (shared / mb65k / 1B).
# Diff: freeze policy_0 (the Coll-2.55% ego) instead of policy_1.
set -euo pipefail

cd /home/fanyuqi/wsc/behavior-bench

export TMUX_SESSION="${TMUX_SESSION:-try-multi-ego-scratch-frozenB0-gpu1}"

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

export GPU_IDS=\"${GPU_IDS:-1}\"
export RUN_NAME=\"ego_scratch_frozen_B_ego0_mb65k_1b_tryMulti\"
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

export PARTNER_MODE=\"reward_shaping\"
export PARTNER_MAX_ABS_STEER=\"0.333\"
export PARTNER_TARGET_HEADWAY=\"1.8\"

# Frozen partner = C-B mb65k ego (policy_0 @ 000954), not policy_1
export FROZEN_PARTNER_PATH=\"\$REPO_ROOT/.logs/train/run4/try_Multi/weights/plvd8yph_policy_0_000954.pt\"

export EVAL_SPLIT=\"validation\"
export EVAL_NUM_MAPS=\"20\"
export ENABLE_WOSAC_REALISM=\"False\"
export ENABLE_HUMAN_REPLAY=\"False\"

export WANDB_PROJECT=\"behavior-bench\"
export WANDB_GROUP=\"run4-try-Multi-frozen-partner\"

export CUDA_VISIBLE_DEVICES=\"\$GPU_IDS\"
export TRAIN_LOG=\"\$LOG_DIR/\${RUN_NAME}_\$(date +%Y%m%d_%H%M%S).log\"

{
  echo \"==== try_Multi · ego scratch + frozen C-B ego(policy_0) as partner (mb65k) ====\"
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
  echo \"MINIBATCH_SIZE=\$MINIBATCH_SIZE\"
  echo \"BPTT=\$BPTT_HORIZON UPDATE_EPOCHS=\$UPDATE_EPOCHS\"
  echo \"partner_mode=\$PARTNER_MODE headway=\$PARTNER_TARGET_HEADWAY\"
  echo \"mix_ppo_sampling=shared\"
  echo \"mix_ppo_policy_trainable=True,False\"
  echo \"FROZEN_PARTNER_PATH=\$FROZEN_PARTNER_PATH\"
  echo \"source_run=plvd8yph (C-B mb65k policy_0 @ 000954; vs-IDM Coll 2.55%)\"
  echo \"contrast=previous try_Multi froze policy_1\"
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
