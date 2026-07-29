#!/usr/bin/env bash
# Launch DriveLite Mixed A/B × shared/stratified → Drive+LSTM vs Expert.
set -euo pipefail

REPO_ROOT="${HOME}/wsc/behavior-bench"
HELPERS="$REPO_ROOT/.logs/train/run_repaet/test_helpers"
TEST_ROOT="$REPO_ROOT/.logs/train/run_repaet/trainLite/test_Expert"
EVAL="$HELPERS/run_one_policy_eval.sh"
RESOLVE="$HELPERS/resolve_final_policy_checkpoint.sh"
cd "$REPO_ROOT"
mkdir -p "$TEST_ROOT"
chmod +x "$HELPERS"/*.sh "$TEST_ROOT/launch_eval.sh"
export STABLE_SECONDS=1

launch() {
  local wid="$1" dir="$2" gpu="$3"
  local exp="$REPO_ROOT/experiments/puffer_drive_${wid}"
  local weights
  weights="$("$RESOLVE" "$exp" 953 0)"
  local ts session
  ts="$(date +%Y%m%d_%H%M%S)"
  session="eval-${dir}-vs-expert-gpu${gpu}-${ts}"
  session="$(echo "$session" | tr '.' '_' | cut -c1-80)"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" "bash -lc '
export TEST_ROOT=$TEST_ROOT MAP_IDS=all
$EVAL $dir Drive_Recurrent $weights Drive Recurrent 256 256 $gpu expert
echo EXIT_CODE=\$?
'"
  echo "launched $session (wandb=$wid gpu=$gpu weights=$weights)"
}

launch uhnxy2lw drivelite_mixA_mb98k_shared_1500m_run_repaet 3
launch r0jht7bf drivelite_mixA_stratified_1500m_run_repaet 4
launch d9fq08yq drivelite_mixB_mb98k_shared_1500m_run_repaet 5
launch qgytyxt4 drivelite_mixB_stratified_1500m_run_repaet 6

{
  echo "==== run_repaet/trainLite test_Expert launched ===="
  echo "date=$(date -Iseconds)"
  echo "A shared uhnxy2lw GPU3 | A strat r0jht7bf GPU4"
  echo "B shared d9fq08yq GPU5 | B strat qgytyxt4 GPU6"
  echo "results: $TEST_ROOT/<train_dir>/Drive_Recurrent/"
  echo "=================================================="
} | tee "$TEST_ROOT/launch_eval_$(date +%Y%m%d_%H%M%S).md"

sleep 1
tmux ls | rg 'vs-expert|drivelite' || true
