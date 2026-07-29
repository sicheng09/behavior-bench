#!/usr/bin/env bash
# ConservativeMix Phase B · repeat (r2) to reduce seed/run variance.
# Same recipe as start_cons_mix_B_batch2x_1b.sh; new RUN_NAME / tmux / W&B run.
set -euo pipefail

cd /home/fanyuqi/wsc/behavior-bench

export TMUX_SESSION="${TMUX_SESSION:-cons-mix-B-batch2x-1b-r2-gpu1}"

tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
tmux new-session -d -s "$TMUX_SESSION" "bash -lc '
set -euo pipefail

cd /home/fanyuqi/wsc/behavior-bench
source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"
conda activate behavior-bench

export REPO_ROOT=\"\$(pwd)\"
export DRIVE_BINARIES_DATA_ROOT=\"\$REPO_ROOT/resources/drive/binaries\"
export LOG_DIR=\"\$REPO_ROOT/.logs/train/run4/try/run\"
mkdir -p \"\$LOG_DIR\"

export GPU_IDS=\"1\"
export RUN_NAME=\"cons_mix_B_shape_batch2x_1b_run4_r2\"
export CONFIG_PATH=\"pufferlib/config/ocean/drive_conservative_mix.ini\"
export ENV_NAME=\"puffer_drive_conservative_mix\"
export ENV_NUM_MAPS=\"10000\"
export TRAIN_BATCH_SIZE=\"1048576\"
export TOTAL_TIMESTEPS=\"1000000000\"
export MINIBATCH_SIZE=\"32768\"
export BPTT_HORIZON=\"32\"
export UPDATE_EPOCHS=\"1\"

export ENABLE_MIX_TRAFFIC=\"False\"
export PPO_FRACTION=\"1.0\"
export IDM_FRACTION=\"0.0\"
export EXPERT_FRACTION=\"0.0\"

export PARTNER_MODE=\"reward_shaping\"
export PARTNER_MAX_ABS_STEER=\"0.333\"
export PARTNER_TARGET_HEADWAY=\"1.8\"

export EVAL_SPLIT=\"validation\"
export EVAL_NUM_MAPS=\"20\"
export ENABLE_WOSAC_REALISM=\"False\"
export ENABLE_HUMAN_REPLAY=\"False\"

export WANDB_PROJECT=\"behavior-bench\"
export WANDB_GROUP=\"run4-try-cons-mix\"

export CUDA_VISIBLE_DEVICES=\"\$GPU_IDS\"
export TRAIN_LOG=\"\$LOG_DIR/\${RUN_NAME}_\$(date +%Y%m%d_%H%M%S).log\"

{
  echo \"==== ConservativeMix Phase B r2 (repeat; reward_shaping) ====\"
  echo \"date=\$(date -Iseconds)\"
  echo \"TMUX_SESSION=$TMUX_SESSION\"
  echo \"TRAIN_LOG=\$TRAIN_LOG\"
  echo \"GPU_IDS=\$GPU_IDS\"
  echo \"RUN_NAME=\$RUN_NAME\"
  echo \"repeat_of=cons_mix_B_shape_batch2x_1b_run4 (l8z87oaa)\"
  echo \"ENV_NAME=\$ENV_NAME\"
  echo \"TRAIN_BATCH_SIZE=\$TRAIN_BATCH_SIZE\"
  echo \"TOTAL_TIMESTEPS=\$TOTAL_TIMESTEPS\"
  echo \"updates ~= \$((TOTAL_TIMESTEPS / TRAIN_BATCH_SIZE))\"
  echo \"partner_mode=\$PARTNER_MODE headway=\$PARTNER_TARGET_HEADWAY\"
  echo \"policies=Drive,Drive (no steer mask; shaping only)\"
  echo \"=============================================================\"
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
  --env.partner-mode \"\$PARTNER_MODE\" \
  --env.partner-max-abs-steer \"\$PARTNER_MAX_ABS_STEER\" \
  --env.partner-target-headway \"\$PARTNER_TARGET_HEADWAY\" \
  --train.mix-ppo True \
  --train.mix-ppo-policy-mix \"ego:0.5,partner:0.5\" \
  --train.mix-ppo-policy-names \"Drive,Drive\" \
  --train.mix-ppo-rnn-names \"Recurrent,Recurrent\" \
  --train.mix-ppo-policy-paths \",\" \
  --train.mix-ppo-policy-trainable \"True,True\" \
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
echo "Logs:   /home/fanyuqi/wsc/behavior-bench/.logs/train/run4/try/run/"
