#!/usr/bin/env bash
# Serial H0 PPO ego vs PDM traffic — one run at a time to avoid OOM.
#
# Root cause (2026-07-21/22): kernel oom-killer SIGKILL'd three parallel
# eval.py. MultiAgentPDMPlanner builds N_traffic × 15 DriveBatch copies per
# map; single-process anon-rss reached ~300GB / ~490GB / ~977GB. Parallel ×3
# exhausted the ~1TiB host (no swap).
#
# Usage:
#   bash run_h0_vs_pdm_serial.sh [gpu_id]
set -euo pipefail

GPU_ID="${1:-2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HOME/wsc/behavior-bench" && pwd)"
ONE="$SCRIPT_DIR/run_one_policy_eval.sh"
LOG_DIR="$SCRIPT_DIR"
SERIAL_LOG="$LOG_DIR/serial_h0_vs_pdm_$(date +%Y%m%d_%H%M%S).log"

mem_gb() {
  awk '/MemAvailable/ {printf "%.1f", $2/1024/1024}' /proc/meminfo
}

rss_gb() {
  local pid="$1"
  if [ -r "/proc/$pid/status" ]; then
    awk '/VmRSS/ {printf "%.1f", $2/1024/1024}' "/proc/$pid/status"
  else
    echo "?"
  fi
}

{
  echo "==== serial H0 vs PDM relaunch ===="
  echo "date=$(date -Iseconds)"
  echo "gpu_id=$GPU_ID"
  echo "reason=prior parallel runs OOM-killed (dmesg pid 3462511/3462520/3462654)"
  echo "strategy=one eval at a time; abort if MemAvailable < 80GiB before start"
  echo "host_mem_available_gib=$(mem_gb)"
  echo

  declare -a JOBS=(
    "homogeneous_drive_lstm_500m_batch1x_run1|$ROOT/experiments/puffer_drive_fdfw3v5e/model_puffer_drive_000954.pt"
    "homogeneous_drive_lstm_500m_batch1x_run4_r1|$ROOT/experiments/puffer_drive_1ytcyfuk/model_puffer_drive_000954.pt"
    "homogeneous_drive_lstm_500m_batch1x_run4_r2|$ROOT/experiments/puffer_drive_u9ymfcqr/model_puffer_drive_000954.pt"
  )

  for job in "${JOBS[@]}"; do
    IFS='|' read -r RUN_NAME WEIGHTS <<<"$job"
    avail="$(mem_gb)"
    echo "---- start $RUN_NAME ----"
    echo "time=$(date -Iseconds) mem_available_gib=$avail"
    if awk -v a="$avail" 'BEGIN{exit !(a<80)}'; then
      echo "ERROR: MemAvailable ${avail}GiB < 80GiB; refusing to start (OOM risk)"
      exit 3
    fi

    # Run in background so we can sample RSS; wait for completion
    bash "$ONE" \
      "$RUN_NAME" Drive_Recurrent \
      "$WEIGHTS" \
      Drive Recurrent 256 256 "$GPU_ID" &
    pid=$!
    echo "pid=$pid"

    # Light RSS monitor (PDM can grow for hours)
    while kill -0 "$pid" 2>/dev/null; do
      sleep 60
      if kill -0 "$pid" 2>/dev/null; then
        echo "$(date +%H:%M:%S) monitor $RUN_NAME pid=$pid rss_gib=$(rss_gb "$pid") avail_gib=$(mem_gb)"
      fi
    done
    status=0
    wait "$pid" || status=$?
    echo "---- done $RUN_NAME exit=$status time=$(date -Iseconds) avail_gib=$(mem_gb) ----"
    echo
    if [ "$status" -ne 0 ]; then
      echo "ERROR: $RUN_NAME failed (exit=$status); stopping serial chain"
      exit "$status"
    fi
    # confirm summary exists
    if ! ls "$LOG_DIR/$RUN_NAME/Drive_Recurrent"/*/summary.csv >/dev/null 2>&1; then
      echo "WARNING: no summary.csv yet under $RUN_NAME (check newest eval dir)"
    fi
    # brief cool-down for allocator / page reclaim
    sleep 15
  done

  echo "==== all three H0 vs PDM finished OK ===="
  echo "date=$(date -Iseconds)"
} 2>&1 | tee -a "$SERIAL_LOG"
