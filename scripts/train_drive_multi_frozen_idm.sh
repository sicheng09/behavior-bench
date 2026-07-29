#!/usr/bin/env bash
# Train one ego policy among two frozen neural traffic policies and IDM.
# Physical vehicle target: 50% ego, 25% frozen A, 12.5% frozen B, 12.5% IDM.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/fanyuqi/wsc/behavior-bench}"
FROZEN_A_PATH="${FROZEN_A_PATH:?Set FROZEN_A_PATH to a single-policy .pt checkpoint}"
FROZEN_B_PATH="${FROZEN_B_PATH:?Set FROZEN_B_PATH to a single-policy .pt checkpoint}"
RUN_NAME="${RUN_NAME:-drive_ego50_frozen25_frozen12p5_idm12p5}"

if [[ ! -f "$FROZEN_A_PATH" ]]; then
  echo "FROZEN_A_PATH does not exist: $FROZEN_A_PATH" >&2
  exit 2
fi
if [[ ! -f "$FROZEN_B_PATH" ]]; then
  echo "FROZEN_B_PATH does not exist: $FROZEN_B_PATH" >&2
  exit 2
fi

cd "$REPO_ROOT"
export DRIVE_BINARIES_DATA_ROOT="${DRIVE_BINARIES_DATA_ROOT:-$REPO_ROOT/resources/drive/binaries}"

# IDM is excluded from the PPO buffer.  The PPO-controlled 87.5% is split
# ego:frozen_a:frozen_b = 4:2:1, so ego owns 4/7 of the PPO rollout:
#   917504 * 4/7 = 524288 ego transitions per outer epoch.
# Scaling total timesteps by the same 7/4 keeps ~500M ego transitions and the
# same number of outer epochs as the homogeneous 500M baseline.
puffer train puffer_drive_conservative_mix \
  --config pufferlib/config/ocean/drive_conservative_mix.ini \
  --train.name "$RUN_NAME" \
  --train.batch-size 917504 \
  --train.total-timesteps 875000000 \
  --train.minibatch-size 32768 \
  --train.max-minibatch-size 32768 \
  --train.bptt-horizon 32 \
  --train.update-epochs 1 \
  --env.num-maps 10000 \
  --env.mix-traffic True \
  --env.ppo-fraction 0.875 \
  --env.idm-fraction 0.125 \
  --env.expert-fraction 0.0 \
  --env.idm-target-velocity 15.0 \
  --env.idm-random-velocity False \
  --env.partner-mode off \
  --train.mix-ppo True \
  --train.mix-ppo-policy-mix "ego:4,frozen_a:2,frozen_b:1" \
  --train.mix-ppo-policy-names "Drive,Drive,Drive" \
  --train.mix-ppo-rnn-names "Recurrent,Recurrent,Recurrent" \
  --train.mix-ppo-policy-paths ",$FROZEN_A_PATH,$FROZEN_B_PATH" \
  --train.mix-ppo-policy-trainable "True,False,False" \
  --train.mix-ppo-sampling stratified \
  --train.mix-ppo-policy-minibatch-sizes "32768,0,0" \
  --train.mix-ppo-policy-update-steps "16,0,0" \
  --wandb \
  --wandb-project behavior-bench \
  --wandb-group multi-frozen-idm
