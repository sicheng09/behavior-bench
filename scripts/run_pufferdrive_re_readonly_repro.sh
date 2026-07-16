#!/usr/bin/env bash
set -euo pipefail

source /home/fanyuqi/miniconda3/etc/profile.d/conda.sh
conda activate Pufferdrive

WORKSPACE="/home/fanyuqi/wsc/behavior-bench"
TARGET="/home/fanyuqi/pufferdrive_re"
HARNESS="${WORKSPACE}/scripts/repro_pufferdrive_re_readonly.py"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE}/.logs/repro/pufferdrive_re_500m}"
TRAIN_DIR="${TRAIN_DIR:-/data2/puffer/data/GPUDrive_medium/binaries/training}"
VAL_DIR="${VAL_DIR:-/data2/puffer/data/GPUDrive_medium/binaries/validation}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,3,5,6,7}"
NPROC="${NPROC:-6}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${TARGET}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/target_state"

snapshot_target() {
  local label="$1"
  git -C "${TARGET}" status --short > "${OUTPUT_ROOT}/target_state/${label}_git_status.txt"
  git -C "${TARGET}" diff --stat > "${OUTPUT_ROOT}/target_state/${label}_git_diff_stat.txt"
  git -C "${TARGET}" diff --cached --stat > "${OUTPUT_ROOT}/target_state/${label}_git_diff_cached_stat.txt"
}

run_diagnose() {
  snapshot_target before_diagnose
  python "${HARNESS}" --output-root "${OUTPUT_ROOT}" diagnose \
    --map-dir "${TRAIN_DIR}" \
    --num-maps 10000 \
    --seed 42 \
    --device cuda \
    2>&1 | tee "${OUTPUT_ROOT}/logs/diagnose.log"
  snapshot_target after_diagnose
}

run_smoke() {
  snapshot_target before_smoke
  for variant in h-legacy h-fixed-init m-legacy; do
    torchrun --standalone --nnodes=1 --nproc-per-node="${NPROC}" \
      "${HARNESS}" --output-root "${OUTPUT_ROOT}/smoke" train \
      --variant "${variant}" \
      --map-dir "${TRAIN_DIR}" \
      --num-maps 10000 \
      --total-env-steps 3145728 \
      --seed 42 \
      --checkpoint-interval 999999 \
      2>&1 | tee "${OUTPUT_ROOT}/logs/smoke_${variant}.log"
    local checkpoint
    checkpoint="$(checkpoint_for_at "${OUTPUT_ROOT}/smoke" "${variant}")"
    python "${HARNESS}" --output-root "${OUTPUT_ROOT}/smoke/eval" eval \
      --variant "${variant}" \
      --checkpoint "${checkpoint}" \
      --training-dir "${TRAIN_DIR}" \
      --validation-dir "${VAL_DIR}" \
      --rollouts 2 \
      --device cuda \
      2>&1 | tee "${OUTPUT_ROOT}/logs/smoke_eval_${variant}.log"
  done
  snapshot_target after_smoke
}

run_train() {
  local variant="$1"
  snapshot_target "before_train_${variant}"
  torchrun --standalone --nnodes=1 --nproc-per-node="${NPROC}" \
    "${HARNESS}" --output-root "${OUTPUT_ROOT}" train \
    --variant "${variant}" \
    --map-dir "${TRAIN_DIR}" \
    --num-maps 10000 \
    --total-env-steps 500000000 \
    --seed 42 \
    --checkpoint-interval 999999 \
    2>&1 | tee "${OUTPUT_ROOT}/logs/train_${variant}.log"
  snapshot_target "after_train_${variant}"
}

checkpoint_for_at() {
  local root="$1"
  local variant="$2"
  python - "${root}/runs/${variant}/summary.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["final_checkpoint"])
PY
}

checkpoint_for() {
  local variant="$1"
  checkpoint_for_at "${OUTPUT_ROOT}" "${variant}"
}

run_eval() {
  local variant="$1"
  local checkpoint
  checkpoint="$(checkpoint_for "${variant}")"
  snapshot_target "before_eval_${variant}"
  python "${HARNESS}" --output-root "${OUTPUT_ROOT}/eval" eval \
    --variant "${variant}" \
    --checkpoint "${checkpoint}" \
    --training-dir "${TRAIN_DIR}" \
    --validation-dir "${VAL_DIR}" \
    --rollouts 32 \
    --device cuda \
    2>&1 | tee "${OUTPUT_ROOT}/logs/eval_${variant}.log"
  snapshot_target "after_eval_${variant}"
}

case "${1:-}" in
  diagnose)
    run_diagnose
    ;;
  smoke)
    run_smoke
    ;;
  train)
    run_train "${2:?variant required}"
    ;;
  eval)
    run_eval "${2:?variant required}"
    ;;
  all)
    run_diagnose
    run_smoke
    run_train h-legacy
    run_train h-fixed-init
    run_train m-legacy
    run_eval h-legacy
    run_eval h-fixed-init
    run_eval m-legacy
    ;;
  *)
    echo "Usage: $0 {diagnose|smoke|train VARIANT|eval VARIANT|all}" >&2
    exit 2
    ;;
esac
