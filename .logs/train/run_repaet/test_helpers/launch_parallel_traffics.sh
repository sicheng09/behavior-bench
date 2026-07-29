#!/usr/bin/env bash
# Launch idm/expert/smart evals in parallel on given GPUs.
#
# Env:
#   TRAIN_RUN_NAME  RESULT_ROOT  WEIGHTS
#   POLICY_IDX (default 0) — only used for logging; WEIGHTS already resolved
#   TRAFFIC_GPU_MAP  e.g. "idm:0 expert:2 smart:4"
# Optional:
#   POLICY_DIR_NAME POLICY_CLASS RNN_NAME RNN_INPUT RNN_HIDDEN
set -euo pipefail

cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

REPO_ROOT="$(pwd)"
HELPERS="$REPO_ROOT/.logs/train/run_repaet/test_helpers"
EVAL_SCRIPT="$HELPERS/run_one_policy_eval.sh"

TRAIN_RUN_NAME="${TRAIN_RUN_NAME:?}"
RESULT_ROOT="${RESULT_ROOT:?}"
WEIGHTS="${WEIGHTS:?}"
TRAFFIC_GPU_MAP="${TRAFFIC_GPU_MAP:?}"

POLICY_DIR_NAME="${POLICY_DIR_NAME:-Drive_Recurrent}"
POLICY_CLASS="${POLICY_CLASS:-Drive}"
RNN_NAME="${RNN_NAME:-Recurrent}"
RNN_INPUT="${RNN_INPUT:-256}"
RNN_HIDDEN="${RNN_HIDDEN:-256}"

traffic_test_root() {
  case "$1" in
    idm) echo "$RESULT_ROOT/test_IDM" ;;
    expert) echo "$RESULT_ROOT/test_Expert" ;;
    smart) echo "$RESULT_ROOT/test_Smart" ;;
    *) echo "$RESULT_ROOT/test_$1" ;;
  esac
}

mkdir -p "$RESULT_ROOT"
LAUNCH_LOG="$RESULT_ROOT/parallel_eval_launch_$(date +%Y%m%d_%H%M%S).log"
{
  echo "==== parallel traffic evals ===="
  echo "date=$(date -Iseconds)"
  echo "TRAIN_RUN_NAME=$TRAIN_RUN_NAME"
  echo "WEIGHTS=$WEIGHTS"
  echo "TRAFFIC_GPU_MAP=$TRAFFIC_GPU_MAP"
  echo "RESULT_ROOT=$RESULT_ROOT"
} | tee "$LAUNCH_LOG"

gpu_busy() {
  local gpu="$1"
  nvidia-smi | awk -v gpu="$gpu" '
    $1 == "|" && $2 == gpu && $4 == "N/A" && $5 == "N/A" && $6 ~ /^[0-9]+$/ && $8 == "C" { found=1 }
    END { exit found ? 0 : 1 }
  '
}

for pair in $TRAFFIC_GPU_MAP; do
  traffic="${pair%%:*}"
  gpu="${pair##*:}"
  test_root="$(traffic_test_root "$traffic")"
  mkdir -p "$test_root"
  # skip if an eval for this traffic already running for this run
  if tmux ls 2>/dev/null | rg -q "eval-${TRAIN_RUN_NAME}-vs-${traffic}-"; then
    echo "SKIP $traffic — already have a tmux eval session" | tee -a "$LAUNCH_LOG"
    continue
  fi
  # also skip if result summary already exists
  if find "$test_root/$TRAIN_RUN_NAME" -name 'summary.json' 2>/dev/null | head -1 | grep -q .; then
    echo "SKIP $traffic — summary.json already present" | tee -a "$LAUNCH_LOG"
    continue
  fi
  while gpu_busy "$gpu"; do
    echo "Waiting for GPU $gpu free before $traffic..." | tee -a "$LAUNCH_LOG"
    sleep 60
  done
  ts="$(date +%Y%m%d_%H%M%S)"
  session="eval-${TRAIN_RUN_NAME}-vs-${traffic}-gpu${gpu}-${ts}"
  session="$(echo "$session" | tr '.' '_' | cut -c1-80)"
  tmux new-session -d -s "$session" "bash -lc '
export TEST_ROOT=$test_root MAP_IDS=all SMART_WEIGHTS=$REPO_ROOT/weights/SMART_epoch_030.pt
$EVAL_SCRIPT $TRAIN_RUN_NAME $POLICY_DIR_NAME $WEIGHTS $POLICY_CLASS $RNN_NAME $RNN_INPUT $RNN_HIDDEN $gpu $traffic
echo EXIT_CODE=\$?
'"
  echo "LAUNCHED $session → $test_root" | tee -a "$LAUNCH_LOG"
  sleep 5
done

echo "done. tmux sessions:" | tee -a "$LAUNCH_LOG"
tmux ls 2>/dev/null | rg "eval-${TRAIN_RUN_NAME}-vs-" | tee -a "$LAUNCH_LOG" || true
