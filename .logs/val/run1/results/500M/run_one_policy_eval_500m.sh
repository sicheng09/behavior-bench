#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 8 ]; then
  echo "Usage: $0 <train_dir_name> <policy_dir_name> <weights_path> <policy_class> <rnn_name> <rnn_input> <rnn_hidden> <gpu_id>" >&2
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

cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

export REPO_ROOT="$(pwd)"
export DRIVE_BINARIES_DATA_ROOT="$REPO_ROOT/data/eval_splits"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export EVAL_SPLIT="pufferinter"
export TRAFFIC_TYPE="idm"
export MAP_IDS="${MAP_IDS:-all}"

OUTPUT_ROOT="$REPO_ROOT/.logs/val/run1/results/500M/$TRAIN_DIR_NAME/$POLICY_DIR_NAME"
mkdir -p "$OUTPUT_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
WRAPPER_LOG="$OUTPUT_ROOT/eval_${TRAIN_DIR_NAME}_${POLICY_DIR_NAME}_${POLICY_CLASS}_vs_idm_gpu${GPU_ID}_${TS}.log"

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
  echo "map_ids=$MAP_IDS"
  echo "output_root=$OUTPUT_ROOT"
  echo

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
    --traffic.type "$TRAFFIC_TYPE" \
    --eval.split "$EVAL_SPLIT" \
    --eval.viz False \
    --eval.planner-viz False \
    --map-ids "$MAP_IDS"
} 2>&1 | tee -a "$WRAPPER_LOG"
