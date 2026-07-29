#!/usr/bin/env bash
# PDM ego vs Expert traffic (WOMD trajectory replay) on pufferinter.
# Same PDM defaults as run_pdm_vs_idm.sh / evaluation.ini.
#
# Usage:
#   run_pdm_vs_expert.sh [gpu_id]
set -euo pipefail

GPU_ID="${1:-5}"
TAG="${TAG:-pdm_default_vs_expert}"

cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

export REPO_ROOT="$(pwd)"
export DRIVE_BINARIES_DATA_ROOT="$REPO_ROOT/data/eval_splits"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export EVAL_SPLIT="pufferinter"
export MAP_IDS="${MAP_IDS:-all}"

# Clean wrapper logs
export TERM=dumb
export NO_COLOR=1
export FORCE_COLOR=0
unset COLORTERM || true

OUTPUT_ROOT="$REPO_ROOT/.logs/train/run4/try_PDM/$TAG"
mkdir -p "$OUTPUT_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
WRAPPER_LOG="$OUTPUT_ROOT/eval_${TAG}_gpu${GPU_ID}_${TS}.log"

strip_ansi() {
  sed -u -e 's/\x1b\[[0-9;?]*[A-Za-z]//g' -e 's/\x1b[()].//g' -e 's/\r//g'
}

{
  echo "tag=$TAG"
  echo "planner=pdm"
  echo "traffic=expert"
  echo "gpu_id=$GPU_ID"
  echo "split=$EVAL_SPLIT"
  echo "map_ids=$MAP_IDS"
  echo "output_root=$OUTPUT_ROOT"
  echo "pdm_params=evaluation.ini defaults (same as pdm_default_vs_idm)"
  echo "protocol=same as pdm_default_vs_idm; traffic=expert (WOMD replay)"
  echo "log_sanitized=1"
  echo

  python pufferlib/ocean/benchmark/eval.py \
    --output-dir "$OUTPUT_ROOT" \
    --planner.type pdm \
    --traffic.type expert \
    --eval.split "$EVAL_SPLIT" \
    --eval.viz False \
    --eval.planner-viz False \
    --map-ids "$MAP_IDS"
} 2>&1 | strip_ansi | tee -a "$WRAPPER_LOG"
