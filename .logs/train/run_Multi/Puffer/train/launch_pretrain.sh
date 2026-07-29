#!/usr/bin/env bash
# Launch Low/Mid/High hybrid pretrains in run_repaet/repeat style (tmux + tee).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
chmod +x \
  "$ROOT/pretrain_low/start.sh" \
  "$ROOT/pretrain_mid/start.sh" \
  "$ROOT/pretrain_high/start.sh" \
  "$ROOT/launch_pretrain.sh"

echo "==== launching run_Multi hybrid pretrains ===="
GPU_IDS=1 bash "$ROOT/pretrain_low/start.sh"
GPU_IDS=2 bash "$ROOT/pretrain_mid/start.sh"
GPU_IDS=3 bash "$ROOT/pretrain_high/start.sh"

{
  echo "==== run_Multi/Puffer/train pretrain launched ===="
  echo "date=$(date -Iseconds)"
  echo "1) Low  HybridDriveLiteLow   GPU1  (60deg/50m, no LSTM)"
  echo "2) Mid  HybridDriveMid       GPU2  (120deg/50m + Recurrent)"
  echo "3) High HybridDriveMoE3High  GPU3  (global + Recurrent)"
  echo "format: tmux + tee  →  <run>/<run_name>_YYYYMMDD_HHMMSS.log"
  echo "wandb group: run_Multi-hybrid-pretrain"
  echo "================================================="
} | tee "$ROOT/launch_$(date +%Y%m%d_%H%M%S).md"

sleep 2
tmux ls | rg 'multi-pretrain-' || true
echo
ps aux | rg 'puffer train .*hybrid_pretrain_' | rg -v rg | head -10 || true
