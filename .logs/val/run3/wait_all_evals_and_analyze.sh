#!/usr/bin/env bash
# Wait until all pending run3 train→eval pipelines finish, then analyze.
# Assumes per-run wait_and_eval_*.sh watchers are already launched for trains.
set -euo pipefail

cd "$HOME/wsc/behavior-bench"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

REPO_ROOT="$(pwd)"
VAL_ROOT="$REPO_ROOT/.logs/val/run3"
RESULTS="$VAL_ROOT/results"
LOG="$VAL_ROOT/wait_all_and_analyze_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$RESULTS"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# Paths that must eventually contain a summary.csv (dir or nested run dir).
REQUIRED=(
  "$RESULTS/adv_mix_ego_opponent_drive_lstm_ddp2_batch2x_4b_run3/p0_ego_Drive_Recurrent"
  "$RESULTS/adv_mix_ego_opponent_drive_lstm_ddp2_500m_low_ego_cost_run3/p0_ego_Drive_Recurrent"
  "$RESULTS/adv_mix_ego_opponent_drive_lstm_ddp2_500m_high_normality_run3/p0_ego_Drive_Recurrent"
  "$RESULTS/adv_mix_ego_opponent_drive_lstm_ddp2_500m_conservative_run3/p0_ego_Drive_Recurrent"
)

# Also keep baseline/weak_goal present (already done; just verify).
BASELINE_OK=(
  "$RESULTS/adv_mix_ego_opponent_drive_lstm_ddp2_500m_run3/p0_ego_Drive_Recurrent"
  "$RESULTS/adv_mix_ego_opponent_drive_lstm_ddp2_500m_weak_goal_run3/p0_ego_Drive_Recurrent"
)

has_summary() {
  local root="$1"
  if [ -f "$root/summary.csv" ]; then
    return 0
  fi
  local f
  f="$(ls -1 "$root"/*/summary.csv 2>/dev/null | sort | tail -1 || true)"
  [ -n "$f" ]
}

log "Watch log: $LOG"
log "Waiting for train auto-eval watchers + in-flight 4B eval to produce summaries."

for root in "${BASELINE_OK[@]}"; do
  if has_summary "$root"; then
    log "OK baseline present: $root"
  else
    log "WARN missing baseline summary: $root"
  fi
done

# Training sessions that should finish (watchers will launch evals).
TRAIN_SESSIONS=(
  adv-mix-drive-lstm-ddp2-500m-low-ego-cost-gpu34
  adv-mix-drive-lstm-ddp2-500m-high-normality-gpu56
  adv-mix-drive-lstm-ddp2-500m-conservative-gpu12
)
WAITERS=(
  wait-eval-low-ego-cost
  wait-eval-high-normality
  wait-eval-conservative
)

still_pending() {
  local root
  for root in "${REQUIRED[@]}"; do
    if ! has_summary "$root"; then
      return 0
    fi
  done
  return 1
}

while still_pending; do
  pending=()
  for root in "${REQUIRED[@]}"; do
    if ! has_summary "$root"; then
      pending+=("$(basename "$(dirname "$root")")")
    fi
  done
  trains_alive=()
  for s in "${TRAIN_SESSIONS[@]}"; do
    if tmux has-session -t "$s" 2>/dev/null; then
      trains_alive+=("$s")
    fi
  done
  waiters_alive=()
  for s in "${WAITERS[@]}"; do
    if tmux has-session -t "$s" 2>/dev/null; then
      waiters_alive+=("$s")
    fi
  done
  evals_alive="$(tmux ls 2>/dev/null | rg -c 'eval-run3' || true)"
  log "Pending summaries: ${pending[*]:-none} | trains=${#trains_alive[@]} waiters=${#waiters_alive[@]} eval_tmux=${evals_alive:-0}"
  # If a train died and waiter died without summary, surface error after grace.
  for i in "${!TRAIN_SESSIONS[@]}"; do
    ts="${TRAIN_SESSIONS[$i]}"
    ws="${WAITERS[$i]}"
    root="${REQUIRED[$((i + 1))]}"  # REQUIRED[0] is 4b; 1..3 map to trains
    if ! tmux has-session -t "$ts" 2>/dev/null \
      && ! tmux has-session -t "$ws" 2>/dev/null \
      && ! has_summary "$root"; then
      # give a bit more time for eval session to appear
      sleep 30
      if ! has_summary "$root" && ! tmux ls 2>/dev/null | rg -q "eval-run3.*$(basename "$(dirname "$root")" | tr '_' '-')"; then
        # check any eval for this train dir
        if ! ls "$root"/eval_*.log >/dev/null 2>&1 && ! has_summary "$root"; then
          log "WARN: train+waiter gone but no summary yet for $root — will keep waiting (eval may still be running under another name)"
        fi
      fi
    fi
  done
  sleep 120
done

log "All required summaries found. Running analysis."
python3 "$VAL_ROOT/analyze_run3_results.py" | tee -a "$LOG"
log "Analysis complete. Report: $RESULTS/run3_comparison_report.md"
