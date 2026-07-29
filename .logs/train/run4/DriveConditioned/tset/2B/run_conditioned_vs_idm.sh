#!/usr/bin/env bash
# DriveConditioned ego (one creward profile) vs IDM traffic on pufferinter.
# Outputs under tset/2B/<profile>_vs_idm/
#
# Usage:
#   run_conditioned_vs_idm.sh <profile> <weights_path> <gpu_id>
# profile: conditioned_aggr | conditioned_normal | conditioned_caut
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <conditioned_aggr|conditioned_normal|conditioned_caut> <weights_path> <gpu_id>" >&2
  exit 2
fi

PROFILE="$1"
WEIGHTS_PATH="$2"
GPU_ID="$3"

case "$PROFILE" in
  conditioned_aggr|conditioned_normal|conditioned_caut) ;;
  *)
    echo "ERROR: unknown profile '$PROFILE'" >&2
    exit 2
    ;;
esac

# CLI section uses hyphens: conditioned_aggr -> conditioned-aggr
PROFILE_CLI="${PROFILE//_/-}"

cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

export REPO_ROOT="$(pwd)"
export DRIVE_BINARIES_DATA_ROOT="$REPO_ROOT/data/eval_splits"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export EVAL_SPLIT="pufferinter"
export MAP_IDS="${MAP_IDS:-all}"

OUTPUT_ROOT="$REPO_ROOT/.logs/train/run4/DriveConditioned/tset/2B/${PROFILE}_vs_idm"
mkdir -p "$OUTPUT_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
WRAPPER_LOG="$OUTPUT_ROOT/eval_${PROFILE}_vs_idm_gpu${GPU_ID}_${TS}.log"

{
  echo "profile=$PROFILE"
  echo "weights_path=$WEIGHTS_PATH"
  echo "planner=$PROFILE"
  echo "traffic=idm"
  echo "gpu_id=$GPU_ID"
  echo "split=$EVAL_SPLIT"
  echo "map_ids=$MAP_IDS"
  echo "output_root=$OUTPUT_ROOT"
  echo "train_tag=2B"
  echo "creward=evaluation.ini defaults for [$PROFILE]"
  echo

  python pufferlib/ocean/benchmark/eval.py \
    --output-dir "$OUTPUT_ROOT" \
    --planner.type "$PROFILE" \
    --planner.${PROFILE_CLI}.weights-path "$WEIGHTS_PATH" \
    --planner.${PROFILE_CLI}.device cuda \
    --traffic.type idm \
    --eval.split "$EVAL_SPLIT" \
    --eval.viz False \
    --eval.planner-viz False \
    --map-ids "$MAP_IDS"
} 2>&1 | tee -a "$WRAPPER_LOG"
