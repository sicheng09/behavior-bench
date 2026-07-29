#!/usr/bin/env bash
# Arm wait-and-eval for MIXMOE shared/stratified → Drive+LSTM vs IDM on train GPUs.
set -euo pipefail

REPO_ROOT="${HOME}/wsc/behavior-bench"
HELPERS="$REPO_ROOT/.logs/train/run_repaet/test_helpers"
TEST_ROOT="$REPO_ROOT/.logs/train/run_repaet/trainMOE/test_IDM"
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
export POLICY_IDX=0
export MIN_EPOCH=953
bash $WAIT
'"
  echo "armed $watch (train=$session wandb=$wid gpu=$gpu)"
}

arm wait-eval-mixmoe-shared \
  mixmoe-mb98k-1500m-gpu1 ncn8qxko \
  mix_moe_drive_moe_moe3_mb98k_1500m_run_repaet \
  mix_moe_drive_moe_moe3_mb98k_1500m_run_repaet 1

arm wait-eval-mixmoe-strat \
  mixmoe-strat-1500m-gpu2 o4trtoxc \
  mix_moe_drive_moe_moe3_stratified_1500m_run_repaet \
  mix_moe_drive_moe_moe3_stratified_1500m_run_repaet 2

{
  echo "==== run_repaet/trainMOE test_IDM wait-and-eval armed ===="
  echo "date=$(date -Iseconds)"
  echo "shared: ncn8qxko GPU1 | stratified: o4trtoxc GPU2"
  echo "results: $TEST_ROOT/<train_dir>/Drive_Recurrent/"
  echo "======================================================="
} | tee "$TEST_ROOT/launch_wait_eval_$(date +%Y%m%d_%H%M%S).md"

sleep 1
tmux ls | rg 'wait-eval-mixmoe|mixmoe-' || true
