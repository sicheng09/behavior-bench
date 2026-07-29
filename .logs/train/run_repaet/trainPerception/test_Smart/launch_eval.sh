#!/usr/bin/env bash
# Launch Perception mix → policy_2 Drive+LSTM vs SMART (train GPU map).
set -euo pipefail

REPO_ROOT="${HOME}/wsc/behavior-bench"
HELPERS="$REPO_ROOT/.logs/train/run_repaet/test_helpers"
TEST_ROOT="$REPO_ROOT/.logs/train/run_repaet/trainPerception/test_Smart"
EVAL="$HELPERS/run_one_policy_eval.sh"
RESOLVE="$HELPERS/resolve_final_policy_checkpoint.sh"
cd "$REPO_ROOT"
mkdir -p "$TEST_ROOT"
chmod +x "$HELPERS"/*.sh "$TEST_ROOT/launch_eval.sh"
export STABLE_SECONDS=1
export SMART_WEIGHTS="${SMART_WEIGHTS:-$REPO_ROOT/weights/SMART_epoch_030.pt}"

launch() {
  local wid="$1" dir="$2" gpu="$3"
  local exp="$REPO_ROOT/experiments/puffer_drive_${wid}"
  local weights
  weights="$("$RESOLVE" "$exp" 953 2)"
  local ts session
  ts="$(date +%Y%m%d_%H%M%S)"
  session="eval-${dir}-vs-smart-gpu${gpu}-${ts}"
  session="$(echo "$session" | tr '.' '_' | cut -c1-80)"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" "bash -lc '
export TEST_ROOT=$TEST_ROOT MAP_IDS=all SMART_WEIGHTS=$SMART_WEIGHTS
$EVAL $dir Drive_Recurrent $weights Drive Recurrent 256 256 $gpu smart
echo EXIT_CODE=\$?
'"
  echo "launched $session (wandb=$wid gpu=$gpu policy=2 weights=$weights)"
}

launch yj5u9i7h perception_mix_low_mid_high_mb98k_shared_1500m_run_repaet 0
launch yov2516v perception_mix_low_mid_high_stratified_1500m_run_repaet 7

{
  echo "==== run_repaet/trainPerception test_Smart launched ===="
  echo "date=$(date -Iseconds)"
  echo "smart_weights=$SMART_WEIGHTS"
  echo "NOTE: evaluates policy_2 = Drive+Recurrent (high perception slot)"
  echo "shared yj5u9i7h GPU0 | stratified yov2516v GPU7"
  echo "results: $TEST_ROOT/<train_dir>/Drive_Recurrent/"
  echo "======================================================="
} | tee "$TEST_ROOT/launch_eval_$(date +%Y%m%d_%H%M%S).md"

sleep 1
tmux ls | rg 'vs-smart|perception' || true
