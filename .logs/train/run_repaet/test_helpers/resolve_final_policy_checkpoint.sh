#!/usr/bin/env bash
# Resolve final mix_ppo checkpoint for a given policy slot under experiments/puffer_drive_*.
#
# Usage: resolve_final_policy_checkpoint.sh <exp_dir> [min_epoch] [policy_idx]
set -euo pipefail

EXP_DIR="${1:?exp_dir required}"
MIN_EPOCH="${2:-953}"
POLICY_IDX="${3:-0}"
STABLE_SECONDS="${STABLE_SECONDS:-20}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-7200}"

epoch_of() {
  local f="$1"
  basename "$f" | sed -n 's/.*_\([0-9][0-9]*\)\.pt$/\1/p' | sed 's/^0*//'
}

latest() {
  local cand
  cand="$(ls -1 "$EXP_DIR"/model_policy_${POLICY_IDX}_*.pt 2>/dev/null | sort | tail -1 || true)"
  if [ -n "$cand" ]; then
    echo "$cand"
    return
  fi
  # Homogeneous fallback (rare for mix runs)
  if [ "$POLICY_IDX" = "0" ]; then
    ls -1 "$EXP_DIR"/model_puffer_drive_*.pt 2>/dev/null | sort | tail -1 || true
  fi
}

start_ts="$(date +%s)"
last_epoch=""
stable_since=""

while true; do
  now="$(date +%s)"
  if (( now - start_ts > MAX_WAIT_SECONDS )); then
    echo "ERROR: timed out waiting for policy_${POLICY_IDX} checkpoint in $EXP_DIR (min_epoch=$MIN_EPOCH)" >&2
    exit 1
  fi

  cand="$(latest)"
  if [ -z "$cand" ] || [ ! -f "$cand" ]; then
    sleep 10
    continue
  fi

  sz="$(stat -c%s "$cand" 2>/dev/null || echo 0)"
  if [ "$sz" -lt 1000 ]; then
    sleep 5
    continue
  fi

  ep="$(epoch_of "$cand")"
  if [ -z "$ep" ] || (( ep < MIN_EPOCH )); then
    sleep 10
    continue
  fi

  if [ "$ep" != "$last_epoch" ]; then
    last_epoch="$ep"
    stable_since="$now"
    sleep 5
    continue
  fi

  if (( now - stable_since >= STABLE_SECONDS )); then
    echo "$cand"
    exit 0
  fi
  sleep 5
done
