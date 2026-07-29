#!/usr/bin/env bash
# Homogeneous DriveConditioned + reward_conditioning=1, 500M batch1x.
# Matches H0 schedule (num_maps=10000, batch=524288, steps=500M, Recurrent 256).
# Config base: drive_gigaflow_conditioning_classic.ini (classic dynamics).
set -euo pipefail

cd /home/fanyuqi/wsc/behavior-bench

export TMUX_SESSION="${TMUX_SESSION:-drive-cond-homo-500m-gpu6}"

tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
tmux new-session -d -s "$TMUX_SESSION" "bash -lc '
set -euo pipefail

cd /home/fanyuqi/wsc/behavior-bench
source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"
conda activate behavior-bench

export REPO_ROOT=\"\$(pwd)\"
export DRIVE_BINARIES_DATA_ROOT=\"\$REPO_ROOT/resources/drive/binaries\"
export LOG_DIR=\"\$REPO_ROOT/.logs/train/run4/DriveConditioned\"
mkdir -p \"\$LOG_DIR\"

export GPU_IDS=\"6\"
export RUN_NAME=\"homogeneous_drive_conditioned_500m_batch1x_run4\"
export CONFIG_PATH=\"pufferlib/config/ocean/drive_gigaflow_conditioning_classic.ini\"
export ENV_NAME=\"puffer_drive\"
export ENV_NUM_MAPS=\"10000\"
export TRAIN_BATCH_SIZE=\"524288\"
export TOTAL_TIMESTEPS=\"500000000\"
export MINIBATCH_SIZE=\"32768\"
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
export WANDB_GROUP=\"run4-drive-conditioned\"

export CUDA_VISIBLE_DEVICES=\"\$GPU_IDS\"
export TRAIN_LOG=\"\$LOG_DIR/\${RUN_NAME}_\$(date +%Y%m%d_%H%M%S).log\"

{
  echo \"==== Homogeneous DriveConditioned 500M batch1x (run4) ====\"
  echo \"date=\$(date -Iseconds)\"
  echo \"TMUX_SESSION=$TMUX_SESSION\"
  echo \"TRAIN_LOG=\$TRAIN_LOG\"
  echo \"GPU_IDS=\$GPU_IDS\"
  echo \"RUN_NAME=\$RUN_NAME\"
  echo \"ENV_NAME=\$ENV_NAME\"
  echo \"CONFIG_PATH=\$CONFIG_PATH\"
  echo \"policy=DriveConditioned rnn=Recurrent\"
  echo \"reward_conditioning=1 (from ini)\"
  echo \"ENV_NUM_MAPS=\$ENV_NUM_MAPS\"
  echo \"mix_traffic=\$ENABLE_MIX_TRAFFIC ppo=\$PPO_FRACTION idm=\$IDM_FRACTION expert=\$EXPERT_FRACTION\"
  echo \"TRAIN_BATCH_SIZE=\$TRAIN_BATCH_SIZE\"
  echo \"TOTAL_TIMESTEPS=\$TOTAL_TIMESTEPS\"
  echo \"updates ~= \$((TOTAL_TIMESTEPS / TRAIN_BATCH_SIZE))\"
  echo \"MINIBATCH_SIZE=\$MINIBATCH_SIZE BPTT=\$BPTT_HORIZON UPDATE_EPOCHS=\$UPDATE_EPOCHS\"
  echo \"matched_schedule=run4/problem homogeneous_drive_lstm_500m_batch1x\"
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
  --env.reward-conditioning 1 \
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
echo "Logs:   /home/fanyuqi/wsc/behavior-bench/.logs/train/run4/DriveConditioned/"
