#!/usr/bin/env bash
# SMART ego pufferinter benchmark (same protocol as try_IDM / try_Multi).
#
# Usage:
#   run_smart_ego_eval.sh <run_dir_name> <smart_weights> <gpu_id> <traffic_type> [ppo_traffic_weights]
#
# traffic_type: idm | ppo
#   - idm: default IDM traffic (v0=15, T=1.5, ...)
#   - ppo: traffic uses PPO; default traffic weights = H0 run4_r2 if arg5 omitted
set -euo pipefail

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo "Usage: $0 <run_dir_name> <smart_weights> <gpu_id> <traffic_type> [ppo_traffic_weights]" >&2
  exit 2
fi

RUN_DIR_NAME="$1"
SMART_WEIGHTS="$2"
GPU_ID="$3"
TRAFFIC_TYPE="$4"
PPO_TRAFFIC_WEIGHTS="${5:-}"

cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

export REPO_ROOT="$(pwd)"
export DRIVE_BINARIES_DATA_ROOT="$REPO_ROOT/data/eval_splits"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export EVAL_SPLIT="pufferinter"
export MAP_IDS="${MAP_IDS:-all}"

# Clean wrapper logs (match try_Multi sanitization)
export TERM=dumb
export NO_COLOR=1
export FORCE_COLOR=0
unset COLORTERM || true

POLICY_DIR_NAME="SMART"
OUTPUT_ROOT="$REPO_ROOT/.logs/train/run4/Smart/test/$RUN_DIR_NAME/$POLICY_DIR_NAME"
mkdir -p "$OUTPUT_ROOT"

if [ "$TRAFFIC_TYPE" = "ppo" ] && [ -z "$PPO_TRAFFIC_WEIGHTS" ]; then
  PPO_TRAFFIC_WEIGHTS="$REPO_ROOT/experiments/puffer_drive_u9ymfcqr/model_puffer_drive_000954.pt"
fi

TS="$(date +%Y%m%d_%H%M%S)"
WRAPPER_LOG="$OUTPUT_ROOT/eval_${RUN_DIR_NAME}_${POLICY_DIR_NAME}_vs_${TRAFFIC_TYPE}_gpu${GPU_ID}_${TS}.log"

strip_ansi() {
  sed -u -e 's/\x1b\[[0-9;?]*[A-Za-z]//g' -e 's/\x1b[()].//g' -e 's/\r//g'
}

{
  echo "run_dir_name=$RUN_DIR_NAME"
  echo "policy_dir_name=$POLICY_DIR_NAME"
  echo "planner=smart"
  echo "smart_weights=$SMART_WEIGHTS"
  echo "gpu_id=$GPU_ID"
  echo "split=$EVAL_SPLIT"
  echo "traffic=$TRAFFIC_TYPE"
  if [ "$TRAFFIC_TYPE" = "ppo" ]; then
    echo "ppo_traffic_weights=$PPO_TRAFFIC_WEIGHTS"
    echo "ppo_traffic_note=H0 run4_r2 (u9ymfcqr) unless overridden"
  fi
  echo "map_ids=$MAP_IDS"
  echo "output_root=$OUTPUT_ROOT"
  echo "log_sanitized=1 (TERM=dumb + strip_ansi)"
  echo "protocol=same as try_IDM/try_Multi (pufferinter, map-ids=all)"
  echo

  test -f "$SMART_WEIGHTS"
  if [ "$TRAFFIC_TYPE" = "ppo" ]; then
    test -f "$PPO_TRAFFIC_WEIGHTS"
  fi

  EXTRA_TRAFFIC_ARGS=()
  if [ "$TRAFFIC_TYPE" = "ppo" ]; then
    EXTRA_TRAFFIC_ARGS+=(
      --traffic.ppo.weights-path "$PPO_TRAFFIC_WEIGHTS"
      --traffic.ppo.device cuda
      --traffic.ppo.policy-class-name Drive
      --traffic.ppo.input-size 64
      --traffic.ppo.hidden-size 256
      --traffic.ppo.rnn-name Recurrent
      --traffic.ppo.rnn-input-size 256
      --traffic.ppo.rnn-hidden-size 256
      --traffic.ppo.reward-conditioning False
    )
  fi

  python pufferlib/ocean/benchmark/eval.py \
    --output-dir "$OUTPUT_ROOT" \
    --planner.type smart \
    --planner.smart.weights-path "$SMART_WEIGHTS" \
    --planner.smart.device cuda \
    --planner.smart.temperature 1.0 \
    --planner.smart.greedy True \
    --planner.smart.repredict-interval 5 \
    --traffic.type "$TRAFFIC_TYPE" \
    "${EXTRA_TRAFFIC_ARGS[@]}" \
    --eval.split "$EVAL_SPLIT" \
    --eval.viz False \
    --eval.planner-viz False \
    --map-ids "$MAP_IDS"
} 2>&1 | strip_ansi | tee -a "$WRAPPER_LOG"
