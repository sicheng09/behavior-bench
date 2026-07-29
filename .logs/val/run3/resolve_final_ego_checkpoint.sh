#!/usr/bin/env bash
# Resolve final ego (policy 0) checkpoint under an experiment dir.
# Avoids picking mid-training weights when the session has just ended.
#
# Usage:
#   resolve_final_ego_checkpoint.sh <exp_dir> [min_epoch]
# Env:
#   STABLE_SECONDS  (default 20) — re-check until newest epoch stops changing
#   MAX_WAIT_SECONDS (default 7200)
set -euo pipefail

EXP_DIR="${1:?exp_dir required}"
MIN_EPOCH="${2:-954}"
STABLE_SECONDS="${STABLE_SECONDS:-20}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-7200}"

epoch_of() {
  local f="$1"
  basename "$f" | sed -n 's/.*_\([0-9][0-9]*\)\.pt$/\1/p' | sed 's/^0*//'
}

latest_policy0() {
  ls -1 "$EXP_DIR"/model_policy_0_puffer_drive_adversarial_*.pt 2>/dev/null | sort | tail -1 || true
}

start_ts="$(date +%s)"
last_epoch=""
stable_since=""

while true; do
  now="$(date +%s)"
  if (( now - start_ts > MAX_WAIT_SECONDS )); then
    echo "ERROR: timed out waiting for final ego checkpoint in $EXP_DIR (min_epoch=$MIN_EPOCH)" >&2
    exit 1
  fi

  cand="$(latest_policy0)"
  if [ -z "$cand" ] || [ ! -f "$cand" ]; then
    sleep 10
    continue
  fi

  # skip zero-size / still-writing files
  sz="$(stat -c%s "$cand" 2>/dev/null || echo 0)"
  if [ "$sz" -lt 1000 ]; then
    sleep 5
    continue
  fi

  ep="$(epoch_of "$cand")"
  if [ -z "$ep" ]; then
    sleep 5
    continue
  fi
  if (( ep < MIN_EPOCH )); then
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
