#!/usr/bin/env bash
# Launch try_IDM ConservativeMix A (GPU0) + B (GPU1) with H0-aligned minibatch.
set -euo pipefail

REPO_ROOT="${HOME}/wsc/behavior-bench"
TRAIN_DIR="$REPO_ROOT/.logs/train/run4/try_IDM/train"
cd "$REPO_ROOT"

chmod +x \
  "$TRAIN_DIR/start_cons_mix_A_mb65k_1b.sh" \
  "$TRAIN_DIR/start_cons_mix_B_mb65k_1b.sh"

echo "==== Launching try_IDM A (GPU0) + B (GPU1), minibatch=65536 ===="
bash "$TRAIN_DIR/start_cons_mix_A_mb65k_1b.sh"
bash "$TRAIN_DIR/start_cons_mix_B_mb65k_1b.sh"

{
  echo "==== try_IDM cons_mix A/B mb65k launch summary ===="
  echo "date=$(date -Iseconds)"
  echo "A tmux: cons-mix-A-mb65k-1b-gpu0"
  echo "B tmux: cons-mix-B-mb65k-1b-gpu1"
  echo "doc:    $TRAIN_DIR/README_optimizer_alignment.md"
  echo "logs:   $TRAIN_DIR/cons_mix_*_tryIDM_*.log"
  echo "========================================"
} | tee "$TRAIN_DIR/launch_cons_mix_A_B_mb65k_$(date +%Y%m%d_%H%M%S).md"

sleep 3
tmux ls | rg 'cons-mix-.*mb65k' || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | head -4
