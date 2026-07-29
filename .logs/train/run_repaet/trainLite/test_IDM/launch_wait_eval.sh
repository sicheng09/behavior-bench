#!/usr/bin/env bash
# Arm wait-and-eval for DriveLite Mixed A/B × shared/stratified → Drive+LSTM vs IDM.
set -euo pipefail

REPO_ROOT="${HOME}/wsc/behavior-bench"
HELPERS="$REPO_ROOT/.logs/train/run_repaet/test_helpers"
TEST_ROOT="$REPO_ROOT/.logs/train/run_repaet/trainLite/test_IDM"
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

arm wait-eval-dl-A-shared \
  drivelite-mixA_mb98k_shared_1500m-gpu3 uhnxy2lw \
  drivelite_mixA_mb98k_shared_1500m_run_repaet \
  drivelite_mixA_mb98k_shared_1500m_run_repaet 3

arm wait-eval-dl-A-strat \
  drivelite-mixA_stratified_1500m-gpu4 r0jht7bf \
  drivelite_mixA_stratified_1500m_run_repaet \
  drivelite_mixA_stratified_1500m_run_repaet 4

arm wait-eval-dl-B-shared \
  drivelite-mixB_mb98k_shared_1500m-gpu5 d9fq08yq \
  drivelite_mixB_mb98k_shared_1500m_run_repaet \
  drivelite_mixB_mb98k_shared_1500m_run_repaet 5

arm wait-eval-dl-B-strat \
  drivelite-mixB_stratified_1500m-gpu6 qgytyxt4 \
  drivelite_mixB_stratified_1500m_run_repaet \
  drivelite_mixB_stratified_1500m_run_repaet 6

{
  echo "==== run_repaet/trainLite test_IDM wait-and-eval armed ===="
  echo "date=$(date -Iseconds)"
  echo "A shared uhnxy2lw GPU3 | A strat r0jht7bf GPU4"
  echo "B shared d9fq08yq GPU5 | B strat qgytyxt4 GPU6"
  echo "results: $TEST_ROOT/<train_dir>/Drive_Recurrent/"
  echo "=========================================================="
} | tee "$TEST_ROOT/launch_wait_eval_$(date +%Y%m%d_%H%M%S).md"

sleep 1
tmux ls | rg 'wait-eval-dl-|drivelite-' || true
