#!/usr/bin/env bash
# Wait for weak-goal 500M ablation, then ego vs IDM on pufferinter.
set -euo pipefail

export TAG="ddp2_500m_weak_goal"
export TRAIN_SESSION="adv-mix-drive-lstm-ddp2-500m-weak-goal-gpu56"
export WANDB_RUN_ID="jbx6brjj"
export TRAIN_DIR_NAME="adv_mix_ego_opponent_drive_lstm_ddp2_500m_weak_goal_run3"
export POLICY_DIR_NAME="p0_ego_Drive_Recurrent"
export EVAL_GPU="${EVAL_GPU:-5}"
export EVAL_SESSION_PREFIX="eval-run3-adv500m-weak-goal-p0-ego-vs-idm"

exec "$(dirname "$0")/wait_and_eval_ddp2_500m.sh"
