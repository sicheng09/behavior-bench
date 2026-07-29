#!/usr/bin/env bash
# Arm wait-and-eval for Perception mix → policy_2 Drive+LSTM vs IDM on train GPUs.
set -euo pipefail

REPO_ROOT="${HOME}/wsc/behavior-bench"
HELPERS="$REPO_ROOT/.logs/train/run_repaet/test_helpers"
TEST_ROOT="$REPO_ROOT/.logs/train/run_repaet/trainPerception/test_IDM"
WAIT="$HELPERS/wait_and_eval_drive_idm.sh"
cd "$REPO_ROOT"
chmod +x "$HELPERS"/*.sh "$TEST_ROOT/launch_wait_eval.sh"

arm() {
  local watch="$1" session="$2" wid="$3" dir="$4" run="$5" gpu="$6"
  tmux kill-session -t "$watch" 2>/dev/null || true
  tmux new-session -d -s "$watch" "bash -lc '
cd \$HOME/wsc/behavior-bench
export TRAIN_SESSION=$session
export WANDB_RUN_ID=$wid
export TRAIN_DIR_NAME=$dir
export TRAIN_RUN_NAME=$run
export TEST_ROOT=$TEST_ROOT
export EVAL_GPU=$gpu
export POLICY_IDX=2
export MIN_EPOCH=953
bash $WAIT
'"
  echo "armed $watch (train=$session wandb=$wid gpu=$gpu policy=2)"
}

arm wait-eval-perc-shared \
  perception-mb98k_shared_1500m-gpu0 yj5u9i7h \
  perception_mix_low_mid_high_mb98k_shared_1500m_run_repaet \
  perception_mix_low_mid_high_mb98k_shared_1500m_run_repaet 0

arm wait-eval-perc-strat \
  perception-stratified_1500m-gpu7 yov2516v \
  perception_mix_low_mid_high_stratified_1500m_run_repaet \
  perception_mix_low_mid_high_stratified_1500m_run_repaet 7

{
  echo "==== run_repaet/trainPerception test_IDM wait-and-eval armed ===="
  echo "date=$(date -Iseconds)"
  echo "NOTE: evaluates policy_2 = Drive+Recurrent (high perception slot)"
  echo "shared yj5u9i7h GPU0 | stratified yov2516v GPU7"
  echo "results: $TEST_ROOT/<train_dir>/Drive_Recurrent/"
  echo "==============================================================="
} | tee "$TEST_ROOT/launch_wait_eval_$(date +%Y%m%d_%H%M%S).md"

sleep 1
tmux ls | rg 'wait-eval-perc-|perception-' || true
