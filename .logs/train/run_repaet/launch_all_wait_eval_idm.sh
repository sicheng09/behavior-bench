#!/usr/bin/env bash
# Arm all run_repaet post-train IDM evals (Drive+LSTM on train GPUs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
bash "$ROOT/trainMOE/test_IDM/launch_wait_eval.sh"
bash "$ROOT/trainLite/test_IDM/launch_wait_eval.sh"
bash "$ROOT/trainPerception/test_IDM/launch_wait_eval.sh"
echo "==== all wait-eval watchers ===="
tmux ls | rg 'wait-eval-' || true
