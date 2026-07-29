# ConservativeMix Launch Recipes (run4 / try)

Ops-only launch and eval notes for `puffer_drive_conservative_mix`.  
Spec: `.logs/train/run4/try/2026-07-18-conservative-mixed-training-design.zh.md`  
Plan: `.logs/train/run4/try/2026-07-18-conservative-mixed-training-plan.md`

**Do not start 500M runs from this doc without explicit user OK.** Phase A code is launchable after Task 6; Phase B after Task 8.

---

## Prereqs

```bash
cd /home/fanyuqi/wsc/behavior-bench   # or conservative-mix worktree
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench
export REPO_ROOT="$(pwd)"
export DRIVE_BINARIES_DATA_ROOT="$REPO_ROOT/data/eval_splits"
```

Default ini: `pufferlib/config/ocean/drive_conservative_mix.ini` (mirror: `config/ocean/drive_conservative_mix.ini`).

| Key default | Value |
| --- | --- |
| `env.partner_mode` | `action_constraint` (Phase A) |
| `env.partner_max_abs_steer` | `0.333` |
| `env.idm_fraction` | `0.0` (no eval-IDM in training) |
| `train.mix_ppo` | `True` |
| `train.mix_ppo_policy_mix` | `ego:0.5, partner:0.5` |
| `train.mix_ppo_policy_names` | `Drive,DriveSteerConstrained` |
| `train.batch_size` | `524288` (batch1x, aligned with run4 H0) |

Override 500M budget on CLI (ini still lists 2B as template default):

```bash
--train.total-timesteps 500000000
```

---

## Training recipes

### Phase A main — C-A (`cons_mix_A_steer0333_500m`)

Steer logits mask on partner only; default `partner_max_abs_steer=0.333`.

```bash
puffer train puffer_drive_conservative_mix \
  --train.total-timesteps 500000000 \
  --tag cons_mix_A_steer0333_500m
```

### Ablation tight — C-A-tight (`cons_mix_A_steer0167_500m`)

Same as C-A but tighter lateral cap (±0.167 rad).

```bash
puffer train puffer_drive_conservative_mix \
  --env.partner-max-abs-steer 0.167 \
  --tag cons_mix_A_steer0167_500m
```

### Phase B — C-B reward shaping (`cons_mix_B_shape_500m`)

Partner-only shaping in `step`; unconstrained partner policy (both slots `Drive`).

```bash
puffer train puffer_drive_conservative_mix \
  --env.partner-mode reward_shaping \
  --train.mix-ppo-policy-names Drive,Drive \
  --tag cons_mix_B_shape_500m
```

Optional follow-ups (not in brief; same 500M budget):

```bash
# C-AB: constraint + shaping
puffer train puffer_drive_conservative_mix \
  --env.partner-mode both \
  --tag cons_mix_AB_both_500m

# Neg control only — weak story; uses existing mix_traffic, not ConservativeMix
puffer train puffer_drive \
  --env.mix-traffic True \
  --env.idm-fraction 0.5 \
  --tag neg_ppo_idm50_500m
```

### Default path unchanged (non-invasive gate)

Must behave identically to pre-pack baseline:

```bash
puffer train puffer_drive
```

Revert checklist: `.logs/train/run4/try/revert_checklist.md`

---

## Optional smoke (few steps / 1 map)

**Skipped in Task 9** — no GPU-free trivial path; do not substitute for unit tests.

If a GPU is free and user approves a short sanity check:

```bash
puffer train puffer_drive_conservative_mix \
  --train.total-timesteps 32768 \
  --env.num-maps 1 \
  --tag cons_mix_smoke_32k
```

Watch W&B / logs for: startup without import errors, `mix_ppo` two-policy routing, partner `large_steer_frac≈0` under C-A.

---

## Eval (unchanged run4 harness)

Primary metric split: **`pufferinter`**, **`--map-ids all`** (~589 valid maps).  
Use final ckpt `model_puffer_drive_conservative_mix_000954.pt` (or last step at 500M).

Output root convention (mirror run4 H0):

```text
.logs/train/run4/try/test/<run_tag>/Drive_Recurrent/
```

### vs IDM (main table)

```bash
export CUDA_VISIBLE_DEVICES=<gpu>
WEIGHTS="experiments/puffer_drive_conservative_mix_<wandb_run>/model_puffer_drive_conservative_mix_000954.pt"
OUT=".logs/train/run4/try/test/cons_mix_A_steer0333_500m/Drive_Recurrent"

python pufferlib/ocean/benchmark/eval.py \
  --output-dir "$OUT" \
  --planner.type ppo \
  --planner.ppo.weights-path "$WEIGHTS" \
  --planner.ppo.device cuda \
  --planner.ppo.policy-class-name Drive \
  --planner.ppo.input-size 64 \
  --planner.ppo.hidden-size 256 \
  --planner.ppo.rnn-name Recurrent \
  --planner.ppo.rnn-input-size 256 \
  --planner.ppo.rnn-hidden-size 256 \
  --planner.ppo.reward-conditioning False \
  --traffic.type idm \
  --eval.split pufferinter \
  --eval.viz False \
  --eval.planner-viz False \
  --map-ids all
```

Reference wrapper pattern: `.logs/val/run3/run_one_policy_eval.sh` (run4 logs use the same `eval.py` flags).

### vs PPO (generalization gate)

Same ckpt for ego **and** traffic (`traffic.type ppo`, same `weights-path`).

```bash
python pufferlib/ocean/benchmark/eval.py \
  --output-dir "$OUT" \
  --planner.type ppo \
  --planner.ppo.weights-path "$WEIGHTS" \
  --planner.ppo.device cuda \
  --planner.ppo.policy-class-name Drive \
  --planner.ppo.input-size 64 \
  --planner.ppo.hidden-size 256 \
  --planner.ppo.rnn-name Recurrent \
  --planner.ppo.rnn-input-size 256 \
  --planner.ppo.rnn-hidden-size 256 \
  --planner.ppo.reward-conditioning False \
  --traffic.type ppo \
  --traffic.ppo.weights-path "$WEIGHTS" \
  --traffic.ppo.device cuda \
  --traffic.ppo.policy-class-name Drive \
  --traffic.ppo.input-size 64 \
  --traffic.ppo.hidden-size 256 \
  --traffic.ppo.rnn-name Recurrent \
  --traffic.ppo.rnn-input-size 256 \
  --traffic.ppo.rnn-hidden-size 256 \
  --traffic.ppo.reward-conditioning False \
  --eval.split pufferinter \
  --eval.viz False \
  --eval.planner-viz False \
  --map-ids all
```

Report: Goal / Coll / At-fault / Offroad; decompose collisions **Lateral vs Front** from `collision_snapshots.json` (same as `.logs/train/run4/problem/homogeneous_drive_lstm_500m_results.md`).

---

## Success gate (vs H0)

Baseline H0: homogeneous `puffer_drive` run4 (e.g. run4_r2 · `u9ymfcqr`).

| Metric | H0 vs IDM (run4_r2) | Target |
| --- | ---: | --- |
| Collision | 5.77% | **≤ 4.90%** (−15% relative) |
| At-fault | 4.58% | **≤ 3.89%** (−15% relative) |
| vs PPO Collision | 0.68% | **≤ 1.68%** (+1 pp absolute cap) |

Pass if **either** vs-IDM Collision **or** At-fault meets −≥15% relative **and** vs-PPO Collision stays within +1 pp of H0.

H0 reference table: `.logs/train/run4/problem/homogeneous_drive_lstm_500m_results.md` §3.

---

## Experiment matrix (500M)

| ID | Command tag | Notes |
| --- | --- | --- |
| H0 | (existing run4) | Baseline — already logged |
| C-A | `cons_mix_A_steer0333_500m` | **Primary** — launch first |
| C-A-tight | `cons_mix_A_steer0167_500m` | Steer ablation |
| C-B | `cons_mix_B_shape_500m` | After A validated |
| C-AB | `cons_mix_AB_both_500m` | Optional combo |
| Neg | `neg_ppo_idm50_500m` | Negative control only |
