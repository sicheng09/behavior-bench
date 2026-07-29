#!/usr/bin/env bash
# ConservativeMix Phase A: ego budget matched to homogeneous Drive+LSTM 500M batch1x.
# See launch_cons_mix_A_batch2x_1b.md for the 2x batch / 2x steps rationale.
set -euo pipefail

cd /home/fanyuqi/wsc/behavior-bench

export TMUX_SESSION="${TMUX_SESSION:-cons-mix-A-batch2x-1b-gpu0}"

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

# ===== Match homogeneous run4 500M batch1x for ego (policy 0) =====
# homo: batch=524288, steps=500M, updates≈953
# mix 50:50 → total batch=2x, total steps=2x so ego gets ~524288/update and ~500M steps
export GPU_IDS=\"0\"
export RUN_NAME=\"cons_mix_A_steer0333_batch2x_1b_run4\"
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

export PARTNER_MODE=\"action_constraint\"
export PARTNER_MAX_ABS_STEER=\"0.333\"

export EVAL_SPLIT=\"validation\"
export EVAL_NUM_MAPS=\"20\"
export ENABLE_WOSAC_REALISM=\"False\"
export ENABLE_HUMAN_REPLAY=\"False\"

export WANDB_PROJECT=\"behavior-bench\"
export WANDB_GROUP=\"run4-try-cons-mix\"

export CUDA_VISIBLE_DEVICES=\"\$GPU_IDS\"
export TRAIN_LOG=\"\$LOG_DIR/\${RUN_NAME}_\$(date +%Y%m%d_%H%M%S).log\"

{
  echo \"==== ConservativeMix Phase A (ego-matched to homo 500M batch1x) ====\"
  echo \"date=\$(date -Iseconds)\"
  echo \"TMUX_SESSION=$TMUX_SESSION\"
  echo \"TRAIN_LOG=\$TRAIN_LOG\"
  echo \"GPU_IDS=\$GPU_IDS\"
  echo \"RUN_NAME=\$RUN_NAME\"
  echo \"ENV_NAME=\$ENV_NAME\"
  echo \"CONFIG_PATH=\$CONFIG_PATH\"
  echo \"ENV_NUM_MAPS=\$ENV_NUM_MAPS\"
  echo \"TRAIN_BATCH_SIZE=\$TRAIN_BATCH_SIZE  # 2×524288; ego≈524288/update\"
  echo \"TOTAL_TIMESTEPS=\$TOTAL_TIMESTEPS    # 2×500M; ego≈500M steps\"
  echo \"updates ~= \$((TOTAL_TIMESTEPS / TRAIN_BATCH_SIZE))\"
  echo \"MINIBATCH_SIZE=\$MINIBATCH_SIZE BPTT=\$BPTT_HORIZON UPDATE_EPOCHS=\$UPDATE_EPOCHS\"
  echo \"partner_mode=\$PARTNER_MODE max_abs_steer=\$PARTNER_MAX_ABS_STEER\"
  echo \"mix_traffic=\$ENABLE_MIX_TRAFFIC idm_fraction=\$IDM_FRACTION\"
  echo \"matched_from=run4/problem homogeneous_drive_lstm_500m_batch1x\"
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
  --train.mix-ppo True \
  --train.mix-ppo-policy-mix \"ego:0.5,partner:0.5\" \
  --train.mix-ppo-policy-names \"Drive,DriveSteerConstrained\" \
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
echo "Doc:    /home/fanyuqi/wsc/behavior-bench/.logs/train/run4/try/run/launch_cons_mix_A_batch2x_1b.md"
