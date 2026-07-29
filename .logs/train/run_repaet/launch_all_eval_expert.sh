#!/usr/bin/env bash
# Launch all run_repaet vs-Expert evals in parallel (Drive+LSTM, train GPU map).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
bash "$ROOT/trainMOE/test_Expert/launch_eval.sh"
bash "$ROOT/trainLite/test_Expert/launch_eval.sh"
bash "$ROOT/trainPerception/test_Expert/launch_eval.sh"
echo "==== all vs-expert eval sessions ===="
tmux ls | rg 'vs-expert' || true
