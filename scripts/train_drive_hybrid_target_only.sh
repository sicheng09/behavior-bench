#!/usr/bin/env bash
# Opt-in target-only mixed-intelligence training.
# Existing puffer train puffer_drive commands are unaffected.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/fanyuqi/wsc/behavior-bench}"
LOW_WEIGHTS="${LOW_WEIGHTS:?Set LOW_WEIGHTS to a HybridDriveLiteLow/DriveLite checkpoint}"
MID_WEIGHTS="${MID_WEIGHTS:?Set MID_WEIGHTS to a HybridDriveMid/Drive checkpoint}"
HIGH_WEIGHTS="${HIGH_WEIGHTS:?Set HIGH_WEIGHTS to a HybridDriveMoE3High/DriveMoE3 checkpoint}"
RUN_NAME="${RUN_NAME:-drive_hybrid_original_o70_l10_m10_h10}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/.logs/train/run_Multi/Puffer}"
export DRIVE_BINARIES_DATA_ROOT="${DRIVE_BINARIES_DATA_ROOT:-$REPO_ROOT/pufferlib/resources/drive/binaries}"

if [[ ! -f "$LOW_WEIGHTS" ]]; then
  echo "LOW_WEIGHTS does not exist: $LOW_WEIGHTS" >&2
  exit 2
fi
if [[ ! -f "$MID_WEIGHTS" ]]; then
  echo "MID_WEIGHTS does not exist: $MID_WEIGHTS" >&2
  exit 2
fi
if [[ ! -f "$HIGH_WEIGHTS" ]]; then
  echo "HIGH_WEIGHTS does not exist: $HIGH_WEIGHTS" >&2
  exit 2
fi

cd "$REPO_ROOT"
mkdir -p "$LOG_DIR"

# Original is the only trainable policy. Low/Mid/High are frozen partners. The
# policy mix is inside the PPO-controlled population; IDM/Expert traffic is
# still controlled by the existing environment-level mix_traffic options.
puffer train puffer_drive_hybrid \
  --config pufferlib/config/ocean/drive_hybrid.ini \
  --train.name "$RUN_NAME" \
  --train.data-dir "$LOG_DIR" \
  --train.mix-ppo-policy-paths ",$LOW_WEIGHTS,$MID_WEIGHTS,$HIGH_WEIGHTS" \
  "$@"
