#!/usr/bin/env bash
# Launch MIXMOE shared/stratified → Drive+LSTM vs Expert (train GPU map).
set -euo pipefail

REPO_ROOT="${HOME}/wsc/behavior-bench"
HELPERS="$REPO_ROOT/.logs/train/run_repaet/test_helpers"
TEST_ROOT="$REPO_ROOT/.logs/train/run_repaet/trainMOE/test_Expert"
EVAL="$HELPERS/run_one_policy_eval.sh"
RESOLVE="$HELPERS/resolve_final_policy_checkpoint.sh"
cd "$REPO_ROOT"
mkdir -p "$TEST_ROOT"
chmod +x "$HELPERS"/*.sh "$TEST_ROOT/launch_eval.sh"
export STABLE_SECONDS=1

launch() {
  local watch="$1" wid="$2" dir="$3" gpu="$4"
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

launch mixmoe-shared ncn8qxko mix_moe_drive_moe_moe3_mb98k_1500m_run_repaet 1
launch mixmoe-strat  o4trtoxc mix_moe_drive_moe_moe3_stratified_1500m_run_repaet 2

{
  echo "==== run_repaet/trainMOE test_Expert launched ===="
  echo "date=$(date -Iseconds)"
  echo "shared: ncn8qxko GPU1 | stratified: o4trtoxc GPU2"
  echo "results: $TEST_ROOT/<train_dir>/Drive_Recurrent/"
  echo "================================================="
} | tee "$TEST_ROOT/launch_eval_$(date +%Y%m%d_%H%M%S).md"

sleep 1
tmux ls | rg 'vs-expert|mixmoe' || true
