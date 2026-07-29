#!/usr/bin/env bash
# H0 homogeneous Drive+LSTM ego vs SMART traffic (pufferinter).
# Mirrors test_IDM / test_Expert layout under problem/test_Smart/.
#
# Usage:
#   run_one_policy_eval.sh <train_dir_name> <policy_dir_name> <ego_weights> \
#       <policy_class> <rnn_name> <rnn_input> <rnn_hidden> <gpu_id> [smart_weights]
set -euo pipefail

if [ "$#" -lt 8 ] || [ "$#" -gt 9 ]; then
  echo "Usage: $0 <train_dir_name> <policy_dir_name> <ego_weights> <policy_class> <rnn_name> <rnn_input> <rnn_hidden> <gpu_id> [smart_weights]" >&2
  exit 2
fi

TRAIN_DIR_NAME="$1"
POLICY_DIR_NAME="$2"
WEIGHTS_PATH="$3"
POLICY_CLASS="$4"
RNN_NAME="$5"
RNN_INPUT="$6"
RNN_HIDDEN="$7"
GPU_ID="$8"
SMART_WEIGHTS="${9:-}"
TRAFFIC_TYPE="smart"

cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

export REPO_ROOT="$(pwd)"
export DRIVE_BINARIES_DATA_ROOT="$REPO_ROOT/data/eval_splits"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export EVAL_SPLIT="pufferinter"
export MAP_IDS="${MAP_IDS:-all}"

# Clean wrapper logs (avoid Rich SGR noise in IDE)
export TERM=dumb
export NO_COLOR=1
export FORCE_COLOR=0
unset COLORTERM || true

if [ -z "$SMART_WEIGHTS" ]; then
  SMART_WEIGHTS="$REPO_ROOT/weights/SMART_epoch_030.pt"
fi

OUTPUT_ROOT="$REPO_ROOT/.logs/train/run4/problem/test_Smart/$TRAIN_DIR_NAME/$POLICY_DIR_NAME"
mkdir -p "$OUTPUT_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
WRAPPER_LOG="$OUTPUT_ROOT/eval_${TRAIN_DIR_NAME}_${POLICY_DIR_NAME}_${POLICY_CLASS}_vs_${TRAFFIC_TYPE}_gpu${GPU_ID}_${TS}.log"

strip_ansi() {
  sed -u -e 's/\x1b\[[0-9;?]*[A-Za-z]//g' -e 's/\x1b[()].//g' -e 's/\r//g'
}

{
  echo "train_dir_name=$TRAIN_DIR_NAME"
  echo "policy_dir_name=$POLICY_DIR_NAME"
  echo "weights_path=$WEIGHTS_PATH"
  echo "policy_class=$POLICY_CLASS"
  echo "rnn_name=$RNN_NAME"
  echo "rnn_input=$RNN_INPUT"
  echo "rnn_hidden=$RNN_HIDDEN"
  echo "gpu_id=$GPU_ID"
  echo "split=$EVAL_SPLIT"
  echo "traffic=$TRAFFIC_TYPE"
  echo "smart_weights=$SMART_WEIGHTS"
  echo "map_ids=$MAP_IDS"
  echo "output_root=$OUTPUT_ROOT"
  echo "log_sanitized=1"
  echo "protocol=same as test_IDM (pufferinter, map-ids=all); ego=ppo traffic=smart"
  echo

  test -f "$WEIGHTS_PATH"
  test -f "$SMART_WEIGHTS"

  python pufferlib/ocean/benchmark/eval.py \
    --output-dir "$OUTPUT_ROOT" \
    --planner.type ppo \
    --planner.ppo.weights-path "$WEIGHTS_PATH" \
    --planner.ppo.device cuda \
    --planner.ppo.policy-class-name "$POLICY_CLASS" \
    --planner.ppo.input-size 64 \
    --planner.ppo.hidden-size 256 \
    --planner.ppo.rnn-name "$RNN_NAME" \
    --planner.ppo.rnn-input-size "$RNN_INPUT" \
    --planner.ppo.rnn-hidden-size "$RNN_HIDDEN" \
    --planner.ppo.reward-conditioning False \
    --traffic.type smart \
    --traffic.smart.weights-path "$SMART_WEIGHTS" \
    --traffic.smart.device cuda \
    --traffic.smart.temperature 1.0 \
    --traffic.smart.greedy True \
    --traffic.smart.repredict-interval 5 \
    --eval.split "$EVAL_SPLIT" \
    --eval.viz False \
    --eval.planner-viz False \
    --map-ids "$MAP_IDS"
} 2>&1 | strip_ansi | tee -a "$WRAPPER_LOG"
