#!/usr/bin/env bash
# Launch Perception mix × shared/stratified on free GPUs 0 and 7.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

SAMPLING=shared     GPU_IDS=0 bash "$ROOT/start_perception_mix_1500m.sh"
SAMPLING=stratified GPU_IDS=7 bash "$ROOT/start_perception_mix_1500m.sh"

echo "==== perception mix runs requested ===="
tmux ls 2>/dev/null | rg 'perception-' || true
