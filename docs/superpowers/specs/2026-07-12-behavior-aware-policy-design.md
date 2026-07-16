# Behavior-Aware LSTM Policy Design

## Context

Current PPO policies perform well when evaluation traffic resembles the training population, but degrade when traffic changes to unseen IDM or PDM behavior. Collision and fault metrics are the main concern. The goal is to improve generalization to unseen traffic behavior without making invasive changes to the existing training and evaluation stack.

The repository already contains useful building blocks:

- `Drive` encodes a single observation and decodes action/value.
- `LSTMWrapper` adds temporal memory around `Drive.encode_observations()` and `Drive.decode_actions()`.
- `DriveLatentWorldModel` adds a latent transition auxiliary loss, but its prediction path does not directly influence action selection at evaluation time.
- Reward-conditioned policies can represent diverse low-level driving styles through creward inputs.
- Evaluation already supports IDM, PDM, PPO, SMART, and conditioned traffic controllers.

This design starts from a non-invasive new policy rather than modifying existing `Drive` or `DriveLatentWorldModel` behavior.

## Goal

Add a true LSTM-based behavior-aware policy that learns a traffic behavior belief from history, predicts next-step partner motion, and feeds that behavior representation into actor/value decisions.

The policy should improve robustness when test traffic comes from unseen IDM/PDM parameters or unseen policy IDs. It should not depend on test traffic having a known reward coefficient vector.

## Non-Goals

- Do not implement multi-step action reranking in the first version.
- Do not require standard reward coefficients for IDM/PDM traffic.
- Do not replace existing `Drive`, `DriveConditioned`, or `DriveLatentWorldModel` code paths.
- Do not add a second recurrent module until the shared-LSTM version has been tested.

## Existing LSTM Semantics

The original recurrent policy is:

```text
obs_t
  -> Drive.encode_observations(obs_t)
  -> LSTMWrapper LSTMCell
  -> Drive.decode_actions(h_t)
  -> action_t, value_t
```

`Drive` itself is feedforward. The wrapper owns temporal state through `state["lstm_h"]` and `state["lstm_c"]`. During evaluation, each step encodes the current observation, updates the LSTM cell, stores the new hidden state back into `state`, and decodes action/value from the LSTM output. During training, the wrapper iterates over the BPTT sequence and resets hidden state at episode boundaries.

The new policy keeps this separation: the policy owns observation encoders and heads; the wrapper owns recurrent sequence handling.

## Proposed Architecture

Add a new policy class, tentatively `DriveBehaviorAware`, and a new wrapper, tentatively `BehaviorAwareLSTMWrapper`.

High-level flow:

```text
obs_t
  -> ego / road / partner encoders
  -> scene_embedding_t
  -> shared LSTM core
  -> h_t
  -> behavior_latent_t = behavior_head(h_t)

actor/value input:
  concat(h_t, behavior_latent_t)

prediction input:
  concat(behavior_latent_t, current_partner_tokens_t)
```

`h_t` remains the general recurrent memory. It can contain ego history, route progress, road context, action inertia, and traffic context. `behavior_latent_t` is derived from `h_t` but is explicitly shaped by next-partner prediction. It is intended to represent traffic behavior belief: how nearby agents respond over time, not who they are by ID.

Actor/value receive both `h_t` and `behavior_latent_t`. This keeps the original policy capacity while making the behavior belief directly available for action selection.

## Observation Encoding

The first version should reuse the `Drive` observation split:

- ego block
- partner block
- road block

For decision features:

- encode ego with an MLP
- encode road objects with an MLP plus road-type one-hot, then max-pool
- encode partner objects with an MLP, then max-pool
- concatenate ego, road, and pooled partner features into `scene_embedding_t`

For prediction features:

- retain per-partner encoded tokens before max-pooling
- use the encoded partner tokens directly in the first version
- use these tokens with `behavior_latent_t` to predict next-step partner deltas

The first version should predict only the observed partner slots, not all scenario agents.

## Behavior Latent

`behavior_latent_t` is not identical to LSTM hidden state `h_t`.

- `h_t` is general recurrent memory optimized mainly through PPO.
- `behavior_latent_t = MLP(h_t)` is a constrained projection trained to support next partner motion prediction.

This creates a bottleneck for behavior-relevant information. The actor/value path uses it so that the predictive belief can influence decisions.

## Prediction Target

Use next-step partner motion prediction as the primary auxiliary task:

```text
target_t = partner_state_{t+1} - partner_state_t
prediction_t = f(behavior_latent_t, partner_token_t)
```

The first target should include compact continuous fields only:

- relative x delta
- relative y delta
- relative speed delta
- relative heading delta

This avoids predicting the full observation vector, road state, or ego state. It also avoids requiring reward labels for IDM/PDM.

Use a validity mask so padding partner slots do not contribute to the loss. A simple first mask can treat all-zero partner slots as invalid, matching the current padded observation convention.

## Training Loss

The PPO loss remains unchanged. The wrapper exposes `compute_auxiliary_loss()` like `LatentWorldModelWrapper`:

```text
loss = ppo_loss + prediction_loss_coef * masked_mse(prediction, target)
```

The auxiliary loss should be logged as:

- `behavior_prediction_loss`
- optionally `behavior_prediction_valid_fraction`

The prediction loss should be computed during BPTT training from stored `mb_obs[:, :-1]` and `mb_obs[:, 1:]`. It should not use future observations during evaluation.

## Evaluation Behavior

At evaluation time:

```text
obs_t
  -> scene_embedding_t
  -> LSTM update
  -> behavior_latent_t
  -> actor/value
```

The prediction head may still run for diagnostics, but action selection must only depend on current and past observations through LSTM state. No future observation is available.

## Non-Invasive Integration

Implementation should add new paths:

- `DriveBehaviorAware` in `pufferlib/ocean/torch.py`
- `BehaviorAwareLSTMWrapper`, either in `pufferlib/ocean/torch.py` near the policy or in `pufferlib/models.py` if it becomes generic enough
- planner support in `pufferlib/planning/policy.py` for evaluation checkpoints
- registry support for `behavior_aware`
- training config support through existing policy-name mechanisms where possible

Existing policy names and planner types should retain their current behavior.

## Training Distribution

The generalization objective requires traffic heterogeneity during training. The first training sweep should prioritize behavior diversity, not only network architecture diversity:

- reward-conditioned PPO traffic with multiple creward profiles
- IDM traffic with randomized target velocity, headway, minimum gap, acceleration, and deceleration where supported
- PDM traffic or PDM evaluation as a held-out generalization target
- PPO opponent snapshots if stable enough in the current training setup

The initial implementation can start with the traffic sources already easiest to configure, then expand after the policy path is verified.

## Ablation Plan

Evaluate at least:

- `Drive + Recurrent`: original baseline
- `BehaviorAware no-fuse`: prediction loss exists, but actor/value use only `h_t`
- `BehaviorAware fused`: actor/value use `concat(h_t, behavior_latent_t)`

The main comparison should be on unseen or held-out traffic:

- pufferinter vs IDM
- pufferinter vs PDM
- pufferrandom vs IDM
- pufferrandom vs PDM
- optionally vs PPO and conditioned traffic

Primary metrics:

- collision rate
- at-fault collision rate
- goal completion
- offroad rate

The desired outcome is not necessarily higher in-distribution PPO-vs-PPO performance. The desired outcome is smaller degradation when traffic changes to unseen IDM/PDM behavior.

## Open Implementation Choices

The first implementation should choose conservative defaults:

- `behavior_latent_dim`: 64 or 128
- `prediction_loss_coef`: start with 0.05 or 0.1 to avoid dominating PPO
- prediction target normalization: use raw normalized observation-space fields first
- prediction mask: all-zero partner slots are invalid
- fusion: concatenate `h_t` and `behavior_latent_t`, followed by a small MLP before actor/value

These should become config parameters after the first working version is validated.

## Risks

- The auxiliary loss can dominate PPO and harm driving performance if weighted too strongly.
- A weak prediction target may teach short-term kinematics but not strategic interaction.
- If training traffic diversity is insufficient, the latent may still overfit to seen behavior.
- If partner slot ordering changes abruptly, per-slot prediction may be noisy. The first version accepts this risk and uses one-step prediction only.

## Success Criteria

The design is successful if the fused behavior-aware policy shows smaller collision/fault degradation than `Drive + Recurrent` when evaluated against held-out IDM/PDM traffic, while keeping goal completion within an acceptable range.

