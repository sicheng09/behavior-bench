#!/usr/bin/env bash
# Launch 3 selected repeat trains + multitraffic wait-eval watchers.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/wsc/behavior-bench"
HELPERS="$REPO/.logs/train/run_repaet/test_helpers"
WAIT="$HELPERS/wait_and_eval_multitraffic.sh"

chmod +x "$HELPERS"/*.sh \
  "$ROOT"/drivelite_mixA_stratified/start.sh \
  "$ROOT"/perception_stratified/start.sh \
  "$ROOT"/mixmoe_stratified/start.sh \
  "$ROOT/launch_selected_repeats.sh"

arm_watch() {
  local watch="$1" session="$2" run_name="$3" result_root="$4" gpu="$5" policy_idx="$6"
  tmux kill-session -t "$watch" 2>/dev/null || true
  tmux new-session -d -s "$watch" "bash -lc '
export TRAIN_SESSION=$session
export TRAIN_RUN_NAME=$run_name
export RUN_NAME_PREFIX=$run_name
export LOG_DIR=$result_root
export RESULT_ROOT=$result_root
export EVAL_GPU=$gpu
export POLICY_IDX=$policy_idx
export TRAFFICS=\"idm expert smart\"
export MIN_EPOCH=953
bash $WAIT
'"
  echo "armed watcher $watch (train=$session gpu=$gpu policy=$policy_idx)"
}

echo "==== launching repeat trains ===="
GPU_IDS=0 bash "$ROOT/drivelite_mixA_stratified/start.sh"
GPU_IDS=1 bash "$ROOT/perception_stratified/start.sh"
GPU_IDS=3 bash "$ROOT/mixmoe_stratified/start.sh"

sleep 3

echo "==== arming wait-eval (idm→expert→smart) ===="
arm_watch wait-repeat-dlA-strat \
  repeat-dlA-strat-gpu0 \
  drivelite_mixA_stratified_1500m_run_repaet_repeat \
  "$ROOT/drivelite_mixA_stratified" 0 0

arm_watch wait-repeat-perc-strat \
  repeat-perc-strat-gpu1 \
  perception_mix_low_mid_high_stratified_1500m_run_repaet_repeat \
  "$ROOT/perception_stratified" 1 2

arm_watch wait-repeat-mixmoe-strat \
  repeat-mixmoe-strat-gpu3 \
  mix_moe_drive_moe_moe3_stratified_1500m_run_repaet_repeat \
  "$ROOT/mixmoe_stratified" 3 0

{
  echo "==== run_repaet/repeat launched ===="
  echo "date=$(date -Iseconds)"
  echo "1) DriveLite A strat  GPU0  ref r0jht7bf (IDM/SMART ~2.7–2.9%)"
  echo "2) Perception strat   GPU1  ref yov2516v (IDM 2.72%)  eval policy_2"
  echo "3) MIXMOE strat       GPU3  ref o4trtoxc (SMART −37.6%)"
  echo "post-train: idm → expert → smart on same GPU"
  echo "wandb group: run_repaet-repeat"
  echo "==================================="
} | tee "$ROOT/launch_$(date +%Y%m%d_%H%M%S).md"

sleep 2
tmux ls | rg 'repeat-' || true
echo
ps aux | rg 'puffer train .*run_repaet_repeat' | rg -v rg || true
