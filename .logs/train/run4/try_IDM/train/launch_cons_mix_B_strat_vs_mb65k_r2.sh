#!/usr/bin/env bash
# Head-to-head C-B retrain: stratified (new) vs shared mb65k (old), same wall-clock.
# Purpose: exclude single-run noise / method confounding after stratified B IDM
# Coll drifted far from mb65k B (4.92% vs 2.55%).
set -euo pipefail

REPO_ROOT="${HOME}/wsc/behavior-bench"
TRAIN_DIR="$REPO_ROOT/.logs/train/run4/try_IDM/train"
TEST_DIR="$REPO_ROOT/.logs/train/run4/try_IDM/test"
cd "$REPO_ROOT"

chmod +x "$TEST_DIR/wait_and_eval_cons_mix.sh" "$TEST_DIR/run_one_policy_eval.sh" \
  "$TEST_DIR/resolve_final_ego_checkpoint.sh"

# Stop auto-decide so it does not also launch a stratified r2.
tmux kill-session -t auto-decide-strat-B 2>/dev/null || true

launch_train() {
  local session="$1" gpu="$2" run_name="$3" mode="$4"  # mode: strat|mb65k
  tmux kill-session -t "$session" 2>/dev/null || true

  local mb mb_max sampling strat_flags
  if [ "$mode" = "strat" ]; then
    mb=32768
    mb_max=32768
    sampling="stratified"
    strat_flags='--train.mix-ppo-sampling stratified --train.mix-ppo-policy-minibatch-sizes "32768,32768" --train.mix-ppo-policy-update-steps "16,16"'
  else
    mb=65536
    mb_max=65536
    sampling="shared"
    strat_flags=""
  fi

  tmux new-session -d -s "$session" "bash -lc '
set -euo pipefail
cd $REPO_ROOT
source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"
conda activate behavior-bench
export REPO_ROOT=\"\$(pwd)\"
export DRIVE_BINARIES_DATA_ROOT=\"\$REPO_ROOT/resources/drive/binaries\"
export LOG_DIR=\"\$REPO_ROOT/.logs/train/run4/try_IDM/train\"
mkdir -p \"\$LOG_DIR\"
export CUDA_VISIBLE_DEVICES=$gpu
export RUN_NAME=$run_name
export TRAIN_LOG=\"\$LOG_DIR/\${RUN_NAME}_\$(date +%Y%m%d_%H%M%S).log\"
{
  echo \"==== try_IDM · C-B head-to-head r2 ($mode) ====\"
  echo \"date=\$(date -Iseconds)\"
  echo \"TMUX_SESSION=$session GPU=$gpu\"
  echo \"RUN_NAME=\$RUN_NAME\"
  echo \"mix_ppo_sampling=$sampling minibatch=$mb\"
  echo \"batch=1048576 steps=1B partner=reward_shaping headway=1.8\"
  echo \"================================================\"
} | tee \"\$TRAIN_LOG\"
puffer train puffer_drive_conservative_mix \
  --config pufferlib/config/ocean/drive_conservative_mix.ini \
  --train.name \"\$RUN_NAME\" \
  --train.batch-size 1048576 \
  --train.total-timesteps 1000000000 \
  --train.minibatch-size $mb \
  --train.max-minibatch-size $mb_max \
  --train.bptt-horizon 32 \
  --train.update-epochs 1 \
  --env.num-maps 10000 \
  --env.mix-traffic False \
  --env.partner-mode reward_shaping \
  --env.partner-max-abs-steer 0.333 \
  --env.partner-target-headway 1.8 \
  --env.partner-w-center 0.05 \
  --env.partner-w-align 0.05 \
  --env.partner-w-steer 0.05 \
  --env.partner-w-gap 0.10 \
  --train.mix-ppo True \
  --train.mix-ppo-policy-mix \"ego:0.5,partner:0.5\" \
  --train.mix-ppo-policy-names \"Drive,Drive\" \
  --train.mix-ppo-rnn-names \"Recurrent,Recurrent\" \
  --train.mix-ppo-policy-paths \",\" \
  --train.mix-ppo-policy-trainable \"True,True\" \
  $strat_flags \
  --eval.split validation \
  --eval.num-maps 20 \
  --eval.wosac-realism-eval False \
  --eval.human-replay-eval False \
  --wandb \
  --wandb-project behavior-bench \
  --wandb-group run4-try-idm-B-r2-head2head \
  2>&1 | tee -a \"\$TRAIN_LOG\"
'"
  echo "Started train: $session ($mode) GPU=$gpu RUN=$run_name"
}

wait_wandb_id() {
  local pattern="$1"
  local wid="" log
  for _ in $(seq 1 90); do
    log="$(ls -t "$TRAIN_DIR"/${pattern}_*.log 2>/dev/null | head -1 || true)"
    if [ -n "$log" ]; then
      wid="$(rg -o 'runs/[a-z0-9]{8}' "$log" 2>/dev/null | head -1 | cut -d/ -f2 || true)"
      if [ -n "$wid" ]; then
        echo "$wid"
        return 0
      fi
    fi
    sleep 10
  done
  return 1
}

arm_wait_eval() {
  local wait_sess="$1" train_sess="$2" wid="$3" dir_name="$4" gpu_idm="$5" gpu_ppo="$6"
  tmux kill-session -t "$wait_sess" 2>/dev/null || true
  tmux new-session -d -s "$wait_sess" "bash -lc '
cd \$HOME/wsc/behavior-bench
export TRAIN_SESSION=$train_sess
export WANDB_RUN_ID=$wid
export TRAIN_DIR_NAME=$dir_name
export EVAL_GPU_IDM=$gpu_idm
export EVAL_GPU_PPO=$gpu_ppo
export MIN_EPOCH=953
bash .logs/train/run4/try_IDM/test/wait_and_eval_cons_mix.sh
'"
  echo "Armed wait-eval: $wait_sess (wandb=$wid eval GPU $gpu_idm/$gpu_ppo)"
}

echo "==== Launch C-B head-to-head: stratified (new) + mb65k shared (old) ===="

# Train: GPU0 stratified, GPU1 mb65k. Eval GPUs avoid current PPO on 3/5.
RUN_STRAT="cons_mix_B_shape_stratified_1b_tryIDM_r2"
RUN_MB65K="cons_mix_B_shape_mb65k_1b_tryIDM_r2"
SESS_STRAT="cons-mix-B-strat-r2-gpu0"
SESS_MB65K="cons-mix-B-mb65k-r2-gpu1"

launch_train "$SESS_STRAT" 0 "$RUN_STRAT" strat
launch_train "$SESS_MB65K" 1 "$RUN_MB65K" mb65k

echo "Waiting for W&B ids..."
WID_STRAT="$(wait_wandb_id "$RUN_STRAT")"
WID_MB65K="$(wait_wandb_id "$RUN_MB65K")"
echo "stratified r2 WANDB=$WID_STRAT"
echo "mb65k r2     WANDB=$WID_MB65K"

# Eval: 2/4 for stratified, 6/7 for mb65k (3/5 busy with ongoing strat-r1 PPO)
arm_wait_eval "wait-eval-tryIDM-B-strat-r2" "$SESS_STRAT" "$WID_STRAT" "$RUN_STRAT" 2 4
arm_wait_eval "wait-eval-tryIDM-B-mb65k-r2" "$SESS_MB65K" "$WID_MB65K" "$RUN_MB65K" 6 7

{
  echo "==== C-B head-to-head r2 launch summary ===="
  echo "date=$(date -Iseconds)"
  echo "killed: auto-decide-strat-B (manual head-to-head replaces it)"
  echo "NEW stratified: tmux=$SESS_STRAT wandb=$WID_STRAT run=$RUN_STRAT eval GPU 2/4"
  echo "OLD mb65k:      tmux=$SESS_MB65K wandb=$WID_MB65K run=$RUN_MB65K eval GPU 6/7"
  echo "group: run4-try-idm-B-r2-head2head"
  echo "note: ongoing strat-r1 PPO evals on GPU 3/5 untouched"
  echo "========================================"
} | tee "$TRAIN_DIR/launch_cons_mix_B_strat_vs_mb65k_r2_$(date +%Y%m%d_%H%M%S).md"

sleep 2
tmux ls | rg 'cons-mix-B-.*r2|wait-eval-tryIDM-B-.*r2' || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | head -2
