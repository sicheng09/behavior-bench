#!/usr/bin/env bash
# PDM ego vs IDM traffic on pufferinter (default evaluation.ini PDM params).
#
# Usage:
#   run_pdm_vs_idm.sh [gpu_id]
set -euo pipefail

GPU_ID="${1:-5}"
TAG="${TAG:-pdm_default_vs_idm}"

cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

export REPO_ROOT="$(pwd)"
export DRIVE_BINARIES_DATA_ROOT="$REPO_ROOT/data/eval_splits"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export EVAL_SPLIT="pufferinter"
export MAP_IDS="${MAP_IDS:-all}"

OUTPUT_ROOT="$REPO_ROOT/.logs/train/run4/try_PDM/$TAG"
mkdir -p "$OUTPUT_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
WRAPPER_LOG="$OUTPUT_ROOT/eval_${TAG}_gpu${GPU_ID}_${TS}.log"

{
  echo "tag=$TAG"
  echo "planner=pdm"
  echo "traffic=idm"
  echo "gpu_id=$GPU_ID"
  echo "split=$EVAL_SPLIT"
  echo "map_ids=$MAP_IDS"
  echo "output_root=$OUTPUT_ROOT"
  echo "pdm_params=evaluation.ini defaults (horizon=40, max_velocity=25, ...)"
  echo

  python pufferlib/ocean/benchmark/eval.py \
    --output-dir "$OUTPUT_ROOT" \
    --planner.type pdm \
    --traffic.type idm \
    --eval.split "$EVAL_SPLIT" \
    --eval.viz False \
    --eval.planner-viz False \
    --map-ids "$MAP_IDS"
} 2>&1 | tee -a "$WRAPPER_LOG"
