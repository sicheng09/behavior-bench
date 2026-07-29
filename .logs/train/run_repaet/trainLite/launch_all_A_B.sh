#!/usr/bin/env bash
# Launch DriveLite Mixed A/B × shared/stratified on GPUs 3,4,5,6.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

VARIANT=A SAMPLING=shared     GPU_IDS=3 bash "$ROOT/start_drivelite_mix_1500m.sh"
VARIANT=A SAMPLING=stratified GPU_IDS=4 bash "$ROOT/start_drivelite_mix_1500m.sh"
VARIANT=B SAMPLING=shared     GPU_IDS=5 bash "$ROOT/start_drivelite_mix_1500m.sh"
VARIANT=B SAMPLING=stratified GPU_IDS=6 bash "$ROOT/start_drivelite_mix_1500m.sh"

echo "==== all four DriveLite mix runs requested ===="
tmux ls 2>/dev/null | rg 'drivelite-' || true
