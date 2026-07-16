# Behavior-Aware LSTM Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a non-invasive `DriveBehaviorAware` policy that uses true LSTM memory, learns a behavior latent through next-partner prediction, and feeds that latent into actor/value decisions for better traffic generalization.

**Architecture:** Add a new feedforward policy class with Drive-style encoders plus behavior/prediction/fusion heads, and add a dedicated recurrent wrapper that mirrors `LSTMWrapper` while collecting masked next-partner prediction loss during BPTT. Training loads through the existing `policy_name`/`rnn_name` mechanism; evaluation loads through a new `behavior_aware` planner/registry path.

**Tech Stack:** Python, PyTorch, Gymnasium spaces, existing `pufferlib.ocean.torch`, `pufferlib.models.LSTMWrapper` conventions, `pytest`/`unittest`, config-driven planner registry.

---

## File Structure

- Modify `pufferlib/ocean/torch.py`: add `DriveBehaviorAware` and `BehaviorAwareRecurrent`.
- Modify `pufferlib/planning/policy.py`: add `BehaviorAwareConfig` and `BehaviorAwarePlanner`.
- Modify `pufferlib/planning/registry.py`: register `behavior_aware` and build its config.
- Modify `pufferlib/config/evaluation.ini`: add default `[planner.behavior_aware]` and `[traffic.behavior_aware]` sections.
- Create `tests/test_behavior_aware_policy.py`: unit tests for model shapes, prediction loss, mask handling, and no-fuse vs fused actor input.
- Modify `tests/test_eval_planners.py`: registry/config tests for the new planner type.

Do not modify existing `Drive`, `DriveConditioned`, or `DriveLatentWorldModel` behavior.

## Task 1: Policy Unit Tests

**Files:**
- Create: `tests/test_behavior_aware_policy.py`
- Modify: none
- Test: `tests/test_behavior_aware_policy.py`

- [ ] **Step 1: Write a dummy env builder and observation helper**

Add this file with the following imports and helpers:

```python
import unittest
from types import SimpleNamespace

import gymnasium
import numpy as np
import torch

import pufferlib
from pufferlib.ocean import torch as drive_torch


def make_dummy_env(action_type="discrete"):
    ego_dim = 7
    max_partner_objects = 3
    partner_features = 7
    max_road_objects = 4
    road_features = 7
    obs_dim = ego_dim + max_partner_objects * partner_features + max_road_objects * road_features
    if action_type == "discrete":
        action_space = gymnasium.spaces.MultiDiscrete([7 * 13])
    else:
        action_space = pufferlib.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    return SimpleNamespace(
        single_observation_space=gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        ),
        single_action_space=action_space,
        max_partner_objects=max_partner_objects,
        partner_features=partner_features,
        max_road_objects=max_road_objects,
        road_features=road_features,
        dynamics_model="classic",
    )


def make_obs(batch, env, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(batch, env.single_observation_space.shape[0])).astype(np.float32)
    ego_dim = 7
    partner_dim = env.max_partner_objects * env.partner_features
    road_start = ego_dim + partner_dim
    road = obs[:, road_start:].reshape(batch, env.max_road_objects, env.road_features)
    road[:, :, -1] = rng.integers(0, 7, size=(batch, env.max_road_objects))
    obs[:, road_start:] = road.reshape(batch, -1)
    return torch.from_numpy(obs)
```

- [ ] **Step 2: Write the failing eval-shape test**

Append this test class:

```python
class TestDriveBehaviorAware(unittest.TestCase):
    def test_forward_eval_returns_discrete_logits_and_value(self):
        env = make_dummy_env()
        policy = drive_torch.DriveBehaviorAware(
            env,
            input_size=16,
            hidden_size=32,
            behavior_latent_dim=8,
        )
        wrapper = drive_torch.BehaviorAwareRecurrent(
            env,
            policy,
            input_size=32,
            hidden_size=32,
        )
        obs = make_obs(5, env)
        state = {
            "lstm_h": torch.zeros(5, 32),
            "lstm_c": torch.zeros(5, 32),
        }

        logits, value = wrapper.forward_eval(obs, state)

        self.assertIsInstance(logits, tuple)
        self.assertEqual(logits[0].shape, (5, 91))
        self.assertEqual(value.shape, (5, 1))
        self.assertEqual(state["lstm_h"].shape, (5, 32))
        self.assertEqual(state["lstm_c"].shape, (5, 32))
```

- [ ] **Step 3: Write the failing auxiliary-loss test**

Append:

```python
    def test_training_forward_records_prediction_loss(self):
        env = make_dummy_env()
        policy = drive_torch.DriveBehaviorAware(
            env,
            input_size=16,
            hidden_size=32,
            behavior_latent_dim=8,
            prediction_loss_coef=0.1,
        )
        wrapper = drive_torch.BehaviorAwareRecurrent(
            env,
            policy,
            input_size=32,
            hidden_size=32,
        )
        obs = make_obs(4 * 6, env).reshape(4, 6, -1)
        actions = torch.zeros(4, 5, 1, dtype=torch.long)
        state = {"action": actions, "lstm_h": None, "lstm_c": None}
        trunc_or_term_before = torch.zeros(4, 5)

        logits, values = wrapper(obs[:, :-1], state, trunc_or_term_before)
        aux_loss, aux_logs = wrapper.compute_auxiliary_loss()

        self.assertEqual(logits[0].shape, (20, 91))
        self.assertEqual(values.shape, (4, 5))
        self.assertGreaterEqual(float(aux_loss), 0.0)
        self.assertIn("behavior_prediction_loss", aux_logs)
        self.assertIn("behavior_prediction_valid_fraction", aux_logs)
```

- [ ] **Step 4: Write the failing padding-mask test**

Append:

```python
    def test_prediction_loss_ignores_zero_partner_slots(self):
        env = make_dummy_env()
        policy = drive_torch.DriveBehaviorAware(
            env,
            input_size=16,
            hidden_size=32,
            behavior_latent_dim=8,
            prediction_loss_coef=0.1,
        )
        wrapper = drive_torch.BehaviorAwareRecurrent(
            env,
            policy,
            input_size=32,
            hidden_size=32,
        )
        obs = make_obs(2 * 4, env).reshape(2, 4, -1)
        ego_dim = 7
        partner_dim = env.max_partner_objects * env.partner_features
        obs[:, :, ego_dim : ego_dim + partner_dim] = 0.0
        actions = torch.zeros(2, 3, 1, dtype=torch.long)
        state = {"action": actions, "lstm_h": None, "lstm_c": None}
        trunc_or_term_before = torch.zeros(2, 3)

        wrapper(obs[:, :-1], state, trunc_or_term_before)
        aux_loss, aux_logs = wrapper.compute_auxiliary_loss()

        self.assertEqual(float(aux_loss), 0.0)
        self.assertEqual(aux_logs["behavior_prediction_valid_fraction"], 0.0)
```

- [ ] **Step 5: Write the failing fused/no-fuse shape test**

Append:

```python
    def test_no_fuse_mode_keeps_actor_input_on_lstm_hidden_only(self):
        env = make_dummy_env()
        fused = drive_torch.DriveBehaviorAware(
            env,
            input_size=16,
            hidden_size=32,
            behavior_latent_dim=8,
            fuse_behavior_latent=True,
        )
        no_fuse = drive_torch.DriveBehaviorAware(
            env,
            input_size=16,
            hidden_size=32,
            behavior_latent_dim=8,
            fuse_behavior_latent=False,
        )

        self.assertEqual(fused.policy_feature_dim, 40)
        self.assertEqual(no_fuse.policy_feature_dim, 32)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_behavior_aware_policy.py -q`

Expected: FAIL with an `AttributeError` indicating `DriveBehaviorAware` is not defined.

## Task 2: Implement `DriveBehaviorAware`

**Files:**
- Modify: `pufferlib/ocean/torch.py`
- Test: `tests/test_behavior_aware_policy.py`

- [ ] **Step 1: Add the policy class skeleton**

In `pufferlib/ocean/torch.py`, add the class after `DriveLatentWorldModel` or near other Drive policy classes:

```python
class DriveBehaviorAware(nn.Module):
    """Drive policy with a behavior latent shaped by partner prediction."""

    def __init__(
        self,
        env,
        input_size=64,
        hidden_size=256,
        behavior_latent_dim=64,
        prediction_loss_coef=0.1,
        fuse_behavior_latent=True,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.behavior_latent_dim = behavior_latent_dim
        self.prediction_loss_coef = float(prediction_loss_coef)
        self.fuse_behavior_latent = bool(fuse_behavior_latent)
        self.observation_size = env.single_observation_space.shape[0]
        self.max_partner_objects = env.max_partner_objects
        self.partner_features = env.partner_features
        self.max_road_objects = env.max_road_objects
        self.road_features = env.road_features
        self.road_features_after_onehot = env.road_features + 6
        self.ego_dim = 10 if env.dynamics_model == "jerk" else 7
        self.partner_delta_dim = 4

        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.ego_dim, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.road_features_after_onehot, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.partner_features, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.shared_embedding = nn.Sequential(
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(3 * input_size, hidden_size)),
        )
        self.behavior_head = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, behavior_latent_dim)),
            nn.LayerNorm(behavior_latent_dim),
            nn.GELU(),
        )
        self.policy_feature_dim = (
            hidden_size + behavior_latent_dim if self.fuse_behavior_latent else hidden_size
        )
        self.policy_fusion = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.policy_feature_dim, hidden_size)),
            nn.GELU(),
        )
        self.prediction_head = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(input_size + behavior_latent_dim, hidden_size)),
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, self.partner_delta_dim)),
        )

        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()
        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)
```

- [ ] **Step 2: Add observation parsing**

Inside `DriveBehaviorAware`, add:

```python
    def _parse_observations(self, observations):
        ego_dim = self.ego_dim
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features
        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim : ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim]
        partner_objects = partner_obs.view(-1, self.max_partner_objects, self.partner_features)
        road_objects = road_obs.view(-1, self.max_road_objects, self.road_features)
        return ego_obs, partner_objects, road_objects
```

- [ ] **Step 3: Add encoder that keeps partner tokens**

Inside `DriveBehaviorAware`, add:

```python
    def encode_observations(self, observations, state=None):
        ego_obs, partner_objects, road_objects = self._parse_observations(observations)
        road_continuous = road_objects[:, :, : self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_onehot = F.one_hot(road_categorical.long().clamp(0, 6), num_classes=7)
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)

        ego_features = self.ego_encoder(ego_obs)
        partner_tokens = self.partner_encoder(partner_objects)
        partner_features, _ = partner_tokens.max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)
        concat_features = torch.cat([ego_features, road_features, partner_features], dim=1)
        embedding = F.relu(self.shared_embedding(concat_features))

        if state is not None:
            state["partner_tokens"] = partner_tokens
            state["partner_objects"] = partner_objects
        return embedding
```

- [ ] **Step 4: Add behavior and action decoding**

Inside `DriveBehaviorAware`, add:

```python
    def behavior_from_hidden(self, hidden):
        return self.behavior_head(hidden)

    def policy_features(self, hidden, behavior_latent):
        if self.fuse_behavior_latent:
            fused = torch.cat([hidden, behavior_latent], dim=-1)
        else:
            fused = hidden
        return self.policy_fusion(fused)

    def decode_actions(self, hidden, behavior_latent=None):
        if behavior_latent is None:
            behavior_latent = self.behavior_from_hidden(hidden)
        policy_hidden = self.policy_features(hidden, behavior_latent)
        if self.is_continuous:
            parameters = self.actor(policy_hidden)
            loc, scale = torch.split(parameters, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale) + 1e-4
            action = torch.distributions.Normal(loc, std)
        else:
            action = self.actor(policy_hidden)
            action = torch.split(action, self.atn_dim, dim=1)
        value = self.value_fn(policy_hidden)
        return action, value
```

- [ ] **Step 5: Add prediction helper methods**

Inside `DriveBehaviorAware`, add:

```python
    def partner_valid_mask(self, partner_objects):
        return partner_objects.abs().sum(dim=-1) > 0

    def partner_prediction_targets(self, current_partner_objects, next_partner_objects):
        current = current_partner_objects[:, :, :4]
        nxt = next_partner_objects[:, :, :4]
        return nxt - current

    def predict_partner_delta(self, behavior_latent, partner_tokens):
        latent = behavior_latent.unsqueeze(1).expand(-1, partner_tokens.shape[1], -1)
        prediction_input = torch.cat([partner_tokens, latent], dim=-1)
        return self.prediction_head(prediction_input)

    def forward(self, observations, state=None):
        state = {} if state is None else state
        hidden = self.encode_observations(observations, state=state)
        behavior_latent = self.behavior_from_hidden(hidden)
        return self.decode_actions(hidden, behavior_latent)

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def forward_eval(self, x, state=None):
        return self.forward(x, state)
```

- [ ] **Step 6: Run the policy tests**

Run: `pytest tests/test_behavior_aware_policy.py -q`

Expected: still FAIL because `BehaviorAwareRecurrent` is not defined.

## Task 3: Implement `BehaviorAwareRecurrent`

**Files:**
- Modify: `pufferlib/ocean/torch.py`
- Test: `tests/test_behavior_aware_policy.py`

- [ ] **Step 1: Add wrapper class init and eval path**

Add after `DriveBehaviorAware`:

```python
class BehaviorAwareRecurrent(nn.Module):
    """LSTM wrapper that exposes behavior prediction auxiliary loss."""

    def __init__(self, env, policy, input_size=256, hidden_size=256, **kwargs):
        super().__init__()
        self.obs_shape = env.single_observation_space.shape
        self.policy = policy
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.is_continuous = self.policy.is_continuous
        self.atn_dim = self.policy.atn_dim
        self._prediction_loss = None
        self._prediction_valid_fraction = 0.0

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.cell = torch.nn.LSTMCell(input_size, hidden_size)
        self.cell.weight_ih = self.lstm.weight_ih_l0
        self.cell.weight_hh = self.lstm.weight_hh_l0
        self.cell.bias_ih = self.lstm.bias_ih_l0
        self.cell.bias_hh = self.lstm.bias_hh_l0

    def forward_eval(self, observations, state):
        step_state = {}
        encoded = self.policy.encode_observations(observations, state=step_state)
        h = state["lstm_h"]
        c = state["lstm_c"]
        lstm_state = (h, c) if h is not None else None
        hidden, c = self.cell(encoded, lstm_state)
        state["hidden"] = hidden
        state["lstm_h"] = hidden
        state["lstm_c"] = c
        behavior_latent = self.policy.behavior_from_hidden(hidden)
        state["behavior_latent"] = behavior_latent
        logits, values = self.policy.decode_actions(hidden, behavior_latent)
        return logits, values
```

- [ ] **Step 2: Add training shape validation and loop**

Inside `BehaviorAwareRecurrent`, add:

```python
    def forward(self, observations, state, trunc_or_term_before, episode_ended=False):
        device = observations.device
        x = observations
        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        if x_shape[-space_n:] != space_shape:
            raise ValueError("Invalid input tensor shape", x.shape)
        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError("Invalid input tensor shape", x.shape)

        values = torch.zeros(B, TT, device=device)
        logits = torch.zeros(B, TT, sum(self.policy.atn_dim), device=device)
        prediction_losses = []
        valid_fracs = []

        for t in range(TT):
            mask = trunc_or_term_before[:, t] == 1.0
            h = state["lstm_h"]
            c = state["lstm_c"]
            if h is not None:
                h = torch.where(mask.unsqueeze(-1), torch.zeros_like(h), h)
                c = torch.where(mask.unsqueeze(-1), torch.zeros_like(c), c)
                lstm_state = (h, c)
            else:
                lstm_state = None

            step_state = {}
            encoded = self.policy.encode_observations(x[:, t, :], state=step_state)
            hidden, c = self.cell(encoded, lstm_state)
            behavior_latent = self.policy.behavior_from_hidden(hidden)
            logits_t, values_t = self.policy.decode_actions(hidden, behavior_latent)
            logits[:, t, :] = logits_t[0] if isinstance(logits_t, tuple) else logits_t
            values[:, t] = values_t.flatten()

            if t + 1 < TT:
                loss_t, valid_frac_t = self._prediction_loss_for_step(
                    behavior_latent,
                    step_state["partner_tokens"],
                    step_state["partner_objects"],
                    x[:, t + 1, :],
                )
                prediction_losses.append(loss_t)
                valid_fracs.append(valid_frac_t)

            state["hidden"] = hidden
            state["lstm_h"] = hidden
            state["lstm_c"] = c

        if episode_ended:
            state["lstm_h"] = state["lstm_h"].detach()
            state["lstm_c"] = state["lstm_c"].detach()

        if prediction_losses:
            self._prediction_loss = torch.stack(prediction_losses).mean()
            self._prediction_valid_fraction = float(torch.stack(valid_fracs).mean().detach().cpu())
        else:
            self._prediction_loss = torch.tensor(0.0, device=device)
            self._prediction_valid_fraction = 0.0

        logits = logits.reshape(B * TT, sum(self.policy.atn_dim))
        return (logits,), values
```

- [ ] **Step 3: Add prediction-loss helper and auxiliary hook**

Inside `BehaviorAwareRecurrent`, add:

```python
    def _prediction_loss_for_step(
        self,
        behavior_latent,
        partner_tokens,
        current_partner_objects,
        next_observations,
    ):
        _, next_partner_objects, _ = self.policy._parse_observations(next_observations)
        valid = (
            self.policy.partner_valid_mask(current_partner_objects)
            & self.policy.partner_valid_mask(next_partner_objects)
        )
        valid_fraction = valid.float().mean()
        if not valid.any():
            zero = behavior_latent.sum() * 0.0
            return zero, valid_fraction
        prediction = self.policy.predict_partner_delta(behavior_latent, partner_tokens)
        target = self.policy.partner_prediction_targets(current_partner_objects, next_partner_objects)
        loss = F.mse_loss(prediction[valid], target[valid])
        return loss, valid_fraction

    def compute_auxiliary_loss(self):
        if self._prediction_loss is None:
            device = next(self.parameters()).device
            loss = torch.tensor(0.0, device=device)
        else:
            loss = self.policy.prediction_loss_coef * self._prediction_loss
        logs = {
            "behavior_prediction_loss": float(loss.detach().cpu()),
            "behavior_prediction_valid_fraction": self._prediction_valid_fraction,
        }
        self._prediction_loss = None
        self._prediction_valid_fraction = 0.0
        return loss, logs
```

- [ ] **Step 4: Run policy tests**

Run: `pytest tests/test_behavior_aware_policy.py -q`

Expected: PASS.

- [ ] **Step 5: Run a small training-loader smoke test**

Run:

```bash
python - <<'PY'
from pufferlib.pufferl import load_config
args = load_config("puffer_drive")
args["policy_name"] = "DriveBehaviorAware"
args["rnn_name"] = "BehaviorAwareRecurrent"
args["policy"]["input_size"] = 16
args["policy"]["hidden_size"] = 32
args["policy"]["behavior_latent_dim"] = 8
args["rnn"]["input_size"] = 32
args["rnn"]["hidden_size"] = 32
print(args["policy_name"], args["rnn_name"])
PY
```

Expected output includes: `DriveBehaviorAware BehaviorAwareRecurrent`.

## Task 4: Evaluation Planner Support

**Files:**
- Modify: `pufferlib/planning/policy.py`
- Test: `tests/test_eval_planners.py`

- [ ] **Step 1: Add behavior-aware config dataclass**

In `pufferlib/planning/policy.py`, near `WorldModelConfig`, add:

```python
@dataclass
class BehaviorAwareConfig:
    weights_path: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    input_size: int = 64
    hidden_size: int = 256
    behavior_latent_dim: int = 64
    prediction_loss_coef: float = 0.1
    fuse_behavior_latent: bool = True
    policy_action_type: str = "discrete"
    stochastic: bool = False
    temperature: float = 1.0
    accel_values: np.ndarray = None
    steer_values: np.ndarray = None

    def __post_init__(self):
        if self.accel_values is None:
            self.accel_values = np.array(
                [-4.0, -2.67, -1.33, 0.0, 1.33, 2.67, 4.0], dtype=np.float32
            )
        if self.steer_values is None:
            self.steer_values = np.linspace(-1.0, 1.0, 13, dtype=np.float32)
```

- [ ] **Step 2: Add planner class**

In `pufferlib/planning/policy.py`, after `WorldModelPlanner`, add a planner mirroring `WorldModelPlanner` but constructing `DriveBehaviorAware` and `BehaviorAwareRecurrent`:

```python
class BehaviorAwarePlanner(WorldModelPlanner):
    """Planner using DriveBehaviorAware with BehaviorAwareRecurrent."""

    def __init__(self, env, agent_idx: int, action_lb: np.ndarray, action_ub: np.ndarray, config: BehaviorAwareConfig):
        BasePlanner.__init__(
            self,
            horizon=1,
            action_dim=len(action_lb),
            action_lb=action_lb,
            action_ub=action_ub,
        )
        self.env = env
        self.agent_idx = agent_idx
        self.config = config
        self.device = torch.device(config.device)

        from pufferlib.ocean.torch import DriveBehaviorAware, BehaviorAwareRecurrent
        from pufferlib.ocean.drive.drive import Drive

        policy_env = Drive(
            episode_length=env.episode_length,
            action_type=config.policy_action_type,
            max_controlled_agents=1,
            split=env.split,
        )
        base_policy = DriveBehaviorAware(
            policy_env,
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            behavior_latent_dim=config.behavior_latent_dim,
            prediction_loss_coef=config.prediction_loss_coef,
            fuse_behavior_latent=config.fuse_behavior_latent,
        )
        self.policy = BehaviorAwareRecurrent(
            policy_env,
            base_policy,
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
        ).to(self.device)
        self.policy.eval()
        policy_env.close()

        self.lstm_h = None
        self.lstm_c = None
        self._obs_buffer = None
        accel_norm = config.accel_values / np.max(np.abs(config.accel_values))
        steer_norm = config.steer_values / np.max(np.abs(config.steer_values))
        self._accel_lut = torch.from_numpy(accel_norm).float().to(self.device)
        self._steer_lut = torch.from_numpy(steer_norm).float().to(self.device)
        self._num_steer = len(config.steer_values)

        if config.weights_path:
            self._load_weights(config.weights_path)
```

The class inherits `WorldModelPlanner.plan()`, `_load_weights()`, `_sample_action_gpu()`, and plotting no-op behavior.

- [ ] **Step 3: Run import smoke test**

Run:

```bash
python - <<'PY'
from pufferlib.planning.policy import BehaviorAwareConfig, BehaviorAwarePlanner
print(BehaviorAwareConfig.__name__, BehaviorAwarePlanner.__name__)
PY
```

Expected output: `BehaviorAwareConfig BehaviorAwarePlanner`.

## Task 5: Registry and Config Support

**Files:**
- Modify: `pufferlib/planning/registry.py`
- Modify: `pufferlib/config/evaluation.ini`
- Modify: `tests/test_eval_planners.py`

- [ ] **Step 1: Add planner type to registry sets and class lookup**

In `pufferlib/planning/registry.py`, include `behavior_aware` in `_CLASSIC_NEURAL_PLANNERS`:

```python
_CLASSIC_NEURAL_PLANNERS = {"ppo", "smart", "world_model", "hybrid", "behavior_aware"} | _CONDITIONED_VARIANTS
```

In `_get_planner_class`, add:

```python
    elif planner_type == "behavior_aware":
        from pufferlib.planning.policy import BehaviorAwarePlanner, BehaviorAwareConfig
        return BehaviorAwarePlanner, BehaviorAwareConfig
```

- [ ] **Step 2: Add config builder**

In `pufferlib/planning/registry.py`, add:

```python
def _build_behavior_aware_config(cfg: dict):
    from pufferlib.planning.policy import BehaviorAwareConfig
    return BehaviorAwareConfig(
        weights_path=str(cfg.get("weights_path", "")),
        device=str(cfg.get("device", "cuda")),
        input_size=int(cfg.get("input_size", 64)),
        hidden_size=int(cfg.get("hidden_size", 256)),
        behavior_latent_dim=int(cfg.get("behavior_latent_dim", 64)),
        prediction_loss_coef=float(cfg.get("prediction_loss_coef", 0.1)),
        fuse_behavior_latent=str(cfg.get("fuse_behavior_latent", "true")).lower() in ("true", "1", "yes"),
        policy_action_type=str(cfg.get("policy_action_type", "discrete")),
        stochastic=str(cfg.get("stochastic", "false")).lower() in ("true", "1", "yes"),
        temperature=float(cfg.get("temperature", 1.0)),
    )
```

- [ ] **Step 3: Use builder for ego and traffic**

In `create_ego_planner`, add before conditioned variants:

```python
    elif planner_type == "behavior_aware":
        ba_cfg = _build_behavior_aware_config(type_cfg)
        planner = cls(env=env, agent_idx=ego_agent_idx, action_lb=ac_lb, action_ub=ac_ub, config=ba_cfg)
```

In `create_traffic_controller`, add before conditioned variants:

```python
    elif traffic_type == "behavior_aware":
        ba_cfg = _build_behavior_aware_config(type_cfg)
        planner = cls(env=env, agent_idx=ego_agent_idx, action_lb=ac_lb, action_ub=ac_ub, config=ba_cfg)
```

- [ ] **Step 4: Add evaluation config sections**

In `pufferlib/config/evaluation.ini`, add after `[planner.ppo]`:

```ini
[planner.behavior_aware]
weights_path =
device = cuda
input_size = 64
hidden_size = 256
behavior_latent_dim = 64
prediction_loss_coef = 0.1
fuse_behavior_latent = true
policy_action_type = discrete
stochastic = false
temperature = 1.0
```

Add after `[traffic.ppo]`:

```ini
[traffic.behavior_aware]
weights_path =
device = cuda
input_size = 64
hidden_size = 256
behavior_latent_dim = 64
prediction_loss_coef = 0.1
fuse_behavior_latent = true
policy_action_type = discrete
stochastic = false
temperature = 1.0
```

- [ ] **Step 5: Add registry tests**

Append to `tests/test_eval_planners.py`:

```python
    def test_behavior_aware_config_builder(self):
        from pufferlib.planning.registry import _build_behavior_aware_config

        cfg = _build_behavior_aware_config({
            "weights_path": "behavior.pt",
            "device": "cpu",
            "input_size": 32,
            "hidden_size": 64,
            "behavior_latent_dim": 16,
            "prediction_loss_coef": 0.05,
            "fuse_behavior_latent": "false",
            "policy_action_type": "discrete",
        })

        self.assertEqual(cfg.weights_path, "behavior.pt")
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.input_size, 32)
        self.assertEqual(cfg.hidden_size, 64)
        self.assertEqual(cfg.behavior_latent_dim, 16)
        self.assertEqual(cfg.prediction_loss_coef, 0.05)
        self.assertFalse(cfg.fuse_behavior_latent)

    def test_load_eval_config_has_behavior_aware_sections(self):
        from pufferlib.planning.registry import load_eval_config

        config = load_eval_config(argv=[
            "--planner.type", "behavior_aware",
            "--planner.behavior_aware.device", "cpu",
            "--traffic.type", "behavior_aware",
            "--traffic.behavior_aware.device", "cpu",
        ])

        self.assertEqual(config["planner"]["type"], "behavior_aware")
        self.assertEqual(config["planner"]["behavior_aware"]["device"], "cpu")
        self.assertEqual(config["traffic"]["type"], "behavior_aware")
        self.assertEqual(config["traffic"]["behavior_aware"]["device"], "cpu")
```

- [ ] **Step 6: Run registry tests**

Run: `pytest tests/test_eval_planners.py -q`

Expected: PASS.

## Task 6: Training Config Smoke and Small Runtime Verification

**Files:**
- Modify: none unless previous tests reveal a bug
- Test: command-line smoke checks

- [ ] **Step 1: Verify policy and wrapper load from config**

Run:

```bash
python - <<'PY'
from pufferlib.pufferl import load_config
args = load_config("puffer_drive")
args["policy_name"] = "DriveBehaviorAware"
args["rnn_name"] = "BehaviorAwareRecurrent"
args["policy"]["input_size"] = 16
args["policy"]["hidden_size"] = 32
args["policy"]["behavior_latent_dim"] = 8
args["rnn"]["input_size"] = 32
args["rnn"]["hidden_size"] = 32
assert args["policy_name"] == "DriveBehaviorAware"
assert args["rnn_name"] == "BehaviorAwareRecurrent"
print("behavior-aware config ok")
PY
```

Expected output: `behavior-aware config ok`.

- [ ] **Step 2: Run focused tests together**

Run:

```bash
pytest tests/test_behavior_aware_policy.py tests/test_eval_planners.py -q
```

Expected: PASS.

- [ ] **Step 3: Run a tiny training command if data and C extension are available**

Run:

```bash
puffer train puffer_drive \
  --policy-name DriveBehaviorAware \
  --rnn-name BehaviorAwareRecurrent \
  --policy.input-size 16 \
  --policy.hidden-size 32 \
  --policy.behavior-latent-dim 8 \
  --policy.prediction-loss-coef 0.05 \
  --rnn.input-size 32 \
  --rnn.hidden-size 32 \
  --train.total-timesteps 32768 \
  --train.batch-size 1024 \
  --train.minibatch-size 256 \
  --train.bptt-horizon 8 \
  --vec.num-workers 1 \
  --vec.num-envs 1 \
  --env.num-agents 64 \
  --env.num-maps 10 \
  --wandb False
```

Expected: the run starts, completes at least one PPO train iteration, and logs `behavior_prediction_loss`.

If the command cannot run because `DRIVE_BINARIES_DATA_ROOT` or the C extension is missing, record the exact error and proceed with unit tests as the verification baseline.

## Task 7: Initial Ablation Commands

**Files:**
- Modify: none
- Test: evaluation commands after checkpoints exist

- [ ] **Step 1: Train fused behavior-aware policy**

Run a normal training job with:

```bash
puffer train puffer_drive \
  --policy-name DriveBehaviorAware \
  --rnn-name BehaviorAwareRecurrent \
  --policy.behavior-latent-dim 64 \
  --policy.prediction-loss-coef 0.05 \
  --policy.fuse-behavior-latent True \
  --env.mix-traffic True \
  --env.ppo-fraction 0.5 \
  --env.idm-fraction 0.5 \
  --env.idm-random-velocity True
```

Expected: checkpoint contains `policy.behavior_head`, `policy.prediction_head`, and `cell`/`lstm` recurrent weights.

- [ ] **Step 2: Train no-fuse ablation**

Run the same command with:

```bash
--policy.fuse-behavior-latent False
```

Expected: checkpoint trains with prediction loss but actor/value receive only `h_t`.

- [ ] **Step 3: Evaluate fused checkpoint vs IDM**

Run:

```bash
python pufferlib/ocean/benchmark/eval.py \
  --planner.type behavior_aware \
  --planner.behavior_aware.weights-path /path/to/fused_checkpoint.pt \
  --planner.behavior_aware.device cuda \
  --traffic.type idm \
  --eval.split pufferinter \
  --map-ids 0-100
```

Expected: final summary includes goal completion, collision rate, fault collision rate, and offroad rate.

- [ ] **Step 4: Evaluate fused checkpoint vs PDM**

Run:

```bash
python pufferlib/ocean/benchmark/eval.py \
  --planner.type behavior_aware \
  --planner.behavior_aware.weights-path /path/to/fused_checkpoint.pt \
  --planner.behavior_aware.device cuda \
  --traffic.type pdm \
  --eval.split pufferinter \
  --map-ids 0-100
```

Expected: final summary includes the same metrics against PDM traffic.

- [ ] **Step 5: Compare against Drive recurrent baseline**

Run the existing PPO planner evaluations using the baseline recurrent checkpoint:

```bash
python pufferlib/ocean/benchmark/eval.py \
  --planner.type ppo \
  --planner.ppo.weights-path /path/to/drive_recurrent_checkpoint.pt \
  --planner.ppo.policy-class-name Drive \
  --planner.ppo.device cuda \
  --traffic.type idm \
  --eval.split pufferinter \
  --map-ids 0-100
```

Expected: compare collision/fault/goal degradation between `Drive + Recurrent`, `BehaviorAware no-fuse`, and `BehaviorAware fused`.

## Self-Review

- Spec coverage: the plan adds the new policy, true LSTM wrapper, next-partner prediction loss, actor/value fusion, no-fuse ablation, registry support, and held-out IDM/PDM evaluation commands.
- Red-flag scan: no incomplete-work markers or unnamed files remain in the plan.
- Type consistency: `DriveBehaviorAware`, `BehaviorAwareRecurrent`, `BehaviorAwareConfig`, and `BehaviorAwarePlanner` names are consistent across tests, implementation, config, and registry tasks.
- Scope check: multi-step reranking and reward-coefficient inference are not included in this implementation plan.

