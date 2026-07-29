#!/usr/bin/env bash
# Launch all run_repaet vs-SMART evals in parallel (Drive+LSTM, train GPU map).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
bash "$ROOT/trainMOE/test_Smart/launch_eval.sh"
bash "$ROOT/trainLite/test_Smart/launch_eval.sh"
bash "$ROOT/trainPerception/test_Smart/launch_eval.sh"
echo "==== all vs-smart eval sessions ===="
tmux ls | rg 'vs-smart' || true
