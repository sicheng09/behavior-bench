#!/usr/bin/env bash
# Wait for stratified C-B train+eval to finish, compare to mb65k B, and
# retrain once if metrics diverge "a lot" (criteria agreed 2026-07-19).
#
# Triggers (any one):
#   1) Gate flip vs mb65k PASS: IDM FAIL or PPO FAIL
#   2) |IDM Coll - 2.55pp| >= 1.5  OR  |IDM Fault - 1.87pp| >= 1.5
#   3) vs-PPO Coll > 1.68%
#
# Writes: .logs/train/run4/try_IDM/result/stratified_B_retrain_decision.md
set -euo pipefail

REPO_ROOT="${HOME}/wsc/behavior-bench"
TRAIN_DIR="$REPO_ROOT/.logs/train/run4/try_IDM/train"
TEST_DIR="$REPO_ROOT/.logs/train/run4/try_IDM/test"
RESULT_DIR="$REPO_ROOT/.logs/train/run4/try_IDM/result"
DECISION_MD="$RESULT_DIR/stratified_B_retrain_decision.md"
WATCH_LOG="$TRAIN_DIR/auto_decide_retrain_strat_B_$(date +%Y%m%d_%H%M%S).log"

TRAIN_DIR_NAME="cons_mix_B_shape_stratified_1b_tryIDM"
POLICY_DIR="Drive_Recurrent"
EVAL_ROOT="$TEST_DIR/$TRAIN_DIR_NAME/$POLICY_DIR"
WAIT_EVAL_SESSION="wait-eval-tryIDM-B-strat"
TRAIN_SESSION="cons-mix-B-strat-1b-gpu1"

# mb65k reference (percent points)
REF_COLL=2.55
REF_FAULT=1.87
GATE_COLL=4.90
GATE_FAULT=3.89
GATE_PPO=1.68
DELTA_PP=1.5

mkdir -p "$RESULT_DIR" "$TRAIN_DIR"
cd "$REPO_ROOT"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCH_LOG"
}

rate_from_summary() {
  local summary="$1"
  local key="$2"
  awk -F, -v k="$key" '$1 == k { printf "%.4f", $2; exit }' "$summary"
}

find_summary_for_traffic() {
  local traffic="$1"
  local d cfg t
  # Newest matching traffic.type first
  while IFS= read -r d; do
    cfg="$d/config.json"
    [ -f "$cfg" ] || continue
    [ -f "$d/summary.csv" ] || continue
    t="$(python3 -c "import json; print(json.load(open('$cfg'))['traffic']['type'])" 2>/dev/null || true)"
    if [ "$t" = "$traffic" ]; then
      echo "$d/summary.csv"
      return 0
    fi
  done < <(ls -1dt "$EVAL_ROOT"/20*/ 2>/dev/null || true)
  return 1
}

evals_ready() {
  # wait-eval session gone, train session gone, both summaries present
  if tmux has-session -t "$WAIT_EVAL_SESSION" 2>/dev/null; then
    return 1
  fi
  if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
    return 1
  fi
  if pgrep -f "puffer train .*--train.name ${TRAIN_DIR_NAME}" >/dev/null 2>&1; then
    return 1
  fi
  local idm_s ppo_s
  idm_s="$(find_summary_for_traffic idm || true)"
  ppo_s="$(find_summary_for_traffic ppo || true)"
  [ -n "$idm_s" ] && [ -n "$ppo_s" ]
}

log "Watch log: $WATCH_LOG"
log "Waiting for $TRAIN_DIR_NAME train+eval (session=$TRAIN_SESSION, wait=$WAIT_EVAL_SESSION)"

while ! evals_ready; do
  log "Not ready yet (tmux train/wait-eval or missing idm/ppo summary)"
  sleep 120
done

log "Eval artifacts ready. Parsing metrics..."

IDM_SUMMARY="$(find_summary_for_traffic idm)"
PPO_SUMMARY="$(find_summary_for_traffic ppo)"
IDM_COLL_R="$(rate_from_summary "$IDM_SUMMARY" collision_rate)"
IDM_FAULT_R="$(rate_from_summary "$IDM_SUMMARY" at_fault_collision_rate)"
PPO_COLL_R="$(rate_from_summary "$PPO_SUMMARY" collision_rate)"

IDM_COLL="$(python3 -c "print(round(float('$IDM_COLL_R')*100, 2))")"
IDM_FAULT="$(python3 -c "print(round(float('$IDM_FAULT_R')*100, 2))")"
PPO_COLL="$(python3 -c "print(round(float('$PPO_COLL_R')*100, 2))")"

log "stratified B: IDM Coll=${IDM_COLL}% Fault=${IDM_FAULT}% | PPO Coll=${PPO_COLL}%"
log "reference mb65k B: IDM Coll=${REF_COLL}% Fault=${REF_FAULT}%"

DECISION="$(python3 - <<PY
coll = float("$IDM_COLL")
fault = float("$IDM_FAULT")
ppo = float("$PPO_COLL")
ref_c, ref_f = float("$REF_COLL"), float("$REF_FAULT")
gate_c, gate_f, gate_ppo = float("$GATE_COLL"), float("$GATE_FAULT"), float("$GATE_PPO")
delta = float("$DELTA_PP")

idm_pass = (coll <= gate_c) or (fault <= gate_f)
ppo_pass = ppo <= gate_ppo
overall_pass = idm_pass and ppo_pass

reasons = []
# 1) gate flip from mb65k PASS
if not overall_pass:
    reasons.append(f"gate_flip: stratified overall FAIL (idm_pass={idm_pass}, ppo_pass={ppo_pass})")
# 2) large absolute delta vs mb65k
if abs(coll - ref_c) >= delta:
    reasons.append(f"|Coll {coll}-ref {ref_c}|={abs(coll-ref_c):.2f}pp >= {delta}")
if abs(fault - ref_f) >= delta:
    reasons.append(f"|Fault {fault}-ref {ref_f}|={abs(fault-ref_f):.2f}pp >= {delta}")
# 3) PPO collapse
if ppo > gate_ppo:
    reasons.append(f"ppo_coll {ppo}% > {gate_ppo}%")

retrain = bool(reasons)
print("RETRAIN" if retrain else "KEEP")
print("---")
if reasons:
    print("\n".join(reasons))
else:
    print("within tolerance; still gate-pass and |Δ| < 1.5pp on Coll/Fault")
print("---")
print(f"idm_pass={idm_pass} ppo_pass={ppo_pass} overall_pass={overall_pass}")
PY
)"

ACTION="$(echo "$DECISION" | head -1)"
REASONS="$(echo "$DECISION" | sed -n '/^---$/,/^---$/p' | sed '1d;$d')"
PASS_LINE="$(echo "$DECISION" | tail -1)"

{
  echo "# Stratified C-B vs mb65k · auto retrain decision"
  echo
  echo "> generated: $(date -Iseconds)"
  echo "> watcher: \`$WATCH_LOG\`"
  echo
  echo "## Metrics"
  echo
  echo "| Run | IDM Coll | IDM Fault | PPO Coll |"
  echo "| --- | ---: | ---: | ---: |"
  echo "| mb65k B (ref) | ${REF_COLL}% | ${REF_FAULT}% | 0.68% |"
  echo "| stratified B | ${IDM_COLL}% | ${IDM_FAULT}% | ${PPO_COLL}% |"
  echo
  echo "- IDM summary: \`$IDM_SUMMARY\`"
  echo "- PPO summary: \`$PPO_SUMMARY\`"
  echo "- $PASS_LINE"
  echo
  echo "## Decision: **${ACTION}**"
  echo
  echo '```'
  echo "$REASONS"
  echo '```'
  echo
  if [ "$ACTION" = "RETRAIN" ]; then
    echo "Action: launching stratified C-B r2 (same recipe, new run name) + wait-eval."
  else
    echo "Action: no retrain."
  fi
} | tee "$DECISION_MD"

log "Wrote decision: $DECISION_MD ($ACTION)"

# Sentinel for Cursor loop / agent wake
echo "AGENT_LOOP_WAKE_stratB_decide {\"prompt\":\"Read $DECISION_MD and stratified B eval metrics; if RETRAIN was launched confirm tmux/wait-eval; summarize for user. Decision=$ACTION\"}"

if [ "$ACTION" != "RETRAIN" ]; then
  log "KEEP — exiting without retrain."
  exit 0
fi

# ---- Retrain once (r2) ----
R2_RUN="cons_mix_B_shape_stratified_1b_tryIDM_r2"
R2_TRAIN_SESSION="cons-mix-B-strat-r2-gpu1"
R2_WAIT_SESSION="wait-eval-tryIDM-B-strat-r2"
R2_GPU_TRAIN=1
R2_EVAL_IDM=4
R2_EVAL_PPO=5

log "RETRAIN — starting $R2_RUN on GPU $R2_GPU_TRAIN"

tmux kill-session -t "$R2_TRAIN_SESSION" 2>/dev/null || true
tmux new-session -d -s "$R2_TRAIN_SESSION" "bash -lc '
set -euo pipefail
cd $REPO_ROOT
source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"
conda activate behavior-bench
export REPO_ROOT=\"\$(pwd)\"
export DRIVE_BINARIES_DATA_ROOT=\"\$REPO_ROOT/resources/drive/binaries\"
export LOG_DIR=\"\$REPO_ROOT/.logs/train/run4/try_IDM/train\"
mkdir -p \"\$LOG_DIR\"
export CUDA_VISIBLE_DEVICES=$R2_GPU_TRAIN
export RUN_NAME=$R2_RUN
export TRAIN_LOG=\"\$LOG_DIR/\${RUN_NAME}_\$(date +%Y%m%d_%H%M%S).log\"
{
  echo \"==== try_IDM · ConservativeMix B stratified r2 (auto-retrain) ====\"
  echo \"date=\$(date -Iseconds)\"
  echo \"reason: see $DECISION_MD\"
  echo \"RUN_NAME=\$RUN_NAME\"
  echo \"same recipe as stratified B: sampling=stratified mb=32768,32768 steps=16,16 batch=1048576 steps=1B\"
  echo \"================================================================\"
} | tee \"\$TRAIN_LOG\"
puffer train puffer_drive_conservative_mix \
  --config pufferlib/config/ocean/drive_conservative_mix.ini \
  --train.name \"\$RUN_NAME\" \
  --train.batch-size 1048576 \
  --train.total-timesteps 1000000000 \
  --train.minibatch-size 32768 \
  --train.max-minibatch-size 32768 \
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
  --train.mix-ppo-sampling stratified \
  --train.mix-ppo-policy-minibatch-sizes \"32768,32768\" \
  --train.mix-ppo-policy-update-steps \"16,16\" \
  --eval.split validation \
  --eval.num-maps 20 \
  --eval.wosac-realism-eval False \
  --eval.human-replay-eval False \
  --wandb \
  --wandb-project behavior-bench \
  --wandb-group run4-try-idm-stratified \
  2>&1 | tee -a \"\$TRAIN_LOG\"
'"

log "Waiting for W&B id for r2..."
WID=""
for _ in $(seq 1 90); do
  LOG="$(ls -t "$TRAIN_DIR"/${R2_RUN}_*.log 2>/dev/null | head -1 || true)"
  if [ -n "$LOG" ]; then
    WID="$(rg -o 'runs/[a-z0-9]{8}' "$LOG" 2>/dev/null | head -1 | cut -d/ -f2 || true)"
    if [ -n "$WID" ]; then
      break
    fi
  fi
  sleep 10
done
if [ -z "$WID" ]; then
  log "ERROR: could not find W&B id for r2; train may still be starting — check tmux $R2_TRAIN_SESSION"
  echo "AGENT_LOOP_WAKE_stratB_decide {\"prompt\":\"Auto-retrain launched but W&B id missing; check tmux $R2_TRAIN_SESSION and $DECISION_MD\"}"
  exit 1
fi
log "r2 WANDB_RUN_ID=$WID"

tmux kill-session -t "$R2_WAIT_SESSION" 2>/dev/null || true
tmux new-session -d -s "$R2_WAIT_SESSION" "bash -lc '
cd \$HOME/wsc/behavior-bench
export TRAIN_SESSION=$R2_TRAIN_SESSION
export WANDB_RUN_ID=$WID
export TRAIN_DIR_NAME=$R2_RUN
export EVAL_GPU_IDM=$R2_EVAL_IDM
export EVAL_GPU_PPO=$R2_EVAL_PPO
export MIN_EPOCH=953
bash .logs/train/run4/try_IDM/test/wait_and_eval_cons_mix.sh
'"

{
  echo
  echo "## Retrain launched"
  echo
  echo "| item | value |"
  echo "| --- | --- |"
  echo "| tmux train | \`$R2_TRAIN_SESSION\` |"
  echo "| tmux wait-eval | \`$R2_WAIT_SESSION\` |"
  echo "| RUN_NAME | \`$R2_RUN\` |"
  echo "| W&B | \`$WID\` |"
  echo "| eval GPUs | IDM=$R2_EVAL_IDM PPO=$R2_EVAL_PPO |"
} | tee -a "$DECISION_MD"

log "r2 train+wait-eval armed."
echo "AGENT_LOOP_WAKE_stratB_decide {\"prompt\":\"Stratified B triggered RETRAIN. Confirm tmux $R2_TRAIN_SESSION / $R2_WAIT_SESSION wandb=$WID; summarize decision from $DECISION_MD for user.\"}"
