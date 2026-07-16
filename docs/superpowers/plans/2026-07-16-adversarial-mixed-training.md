# Adversarial Mixed Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a config-gated `AdversarialMixDrive` that reuses the existing `mix_ppo` global policy assignment while preserving Ego rewards and replacing only Primary Opponent rewards with a realistic, fault-constrained adversarial objective.

**Architecture:** Existing `mix_ppo` remains the sole policy router, BPTT owner, optimizer manager, logger, and checkpoint writer. A new Python `Drive` subclass interprets policy 0 as Ego and policy 1 as Primary Opponent, audits per-scene role composition using `agent_offsets`, and overwrites only Opponent entries in the reward buffer after `Drive.step()`. The first version reads existing observations, actions, reward-component buffers, and `get_global_agent_state()`; it does not modify `pufferl.py`, `drive.h`, or `binding.c`.

**Tech Stack:** Python 3.9+, NumPy, PyYAML, PufferLib/PuffeRL, C-backed PufferDrive environment, unittest/pytest.

## Global Constraints

- `puffer train puffer_drive` and existing `mix_ppo` configurations must retain identical behavior and no new per-step overhead.
- Ego is policy 0, uses `Drive + Recurrent`, and receives the original `Drive` reward unchanged.
- Primary Opponent is policy 1, initially also uses `Drive + Recurrent`, and receives only the asymmetric reward.
- Policy IDs remain fixed by the existing global `assign_policy_ids`; no episode-level owner switching is permitted.
- Reward output is finite and clipped to `[-1, 1]`; configured non-fault positive reward is capped at `0.25`; `w2 >= 4 * w1`.
- Unknown or unimplemented opponent strategies fail at startup; they never silently fall back.
- The first version must not modify `pufferlib/pufferl.py`, `pufferlib/ocean/drive/drive.h`, or `pufferlib/ocean/drive/binding.c`.
- Do **not** change existing validation metric definitions for Adversary training: leave `pufferlib/evaluation/collision_classifier.py`, `.logs/val` at-fault / PDM semantics, and related eval tests unchanged. Adversary fault logic lives only in `pufferlib/adversarial/reward.py`.
- Pairwise calculations are restricted to scene slices from `agent_offsets`; never construct a global `num_agents × num_agents` matrix.
- Local Git identity is configured (`SichengWang <sichengwang@buaa.edu.cn>`). Run commit steps with that identity; do not invent a different author.

---

## File Map

**Create**

- `pufferlib/adversarial/__init__.py` — public exports.
- `pufferlib/adversarial/config.py` — YAML schema, validation, and loading.
- `pufferlib/adversarial/registry.py` — explicit role/strategy registry.
- `pufferlib/adversarial/sampler.py` — future sampler protocol only; no first-version dynamic sampling.
- `pufferlib/adversarial/reward.py` — pure NumPy adversarial reward evaluator.
- `pufferlib/adversarial/env.py` — `AdversarialMixDrive`, reward routing, and scene audit.
- `pufferlib/config/adversarial/opponent_mix.yaml` — default asymmetric reward settings.
- `pufferlib/config/ocean/drive_adversarial.ini` — independent training profile.
- `tests/test_adversarial_config.py` — schema and registry tests.
- `tests/test_adversarial_reward.py` — deterministic synthetic reward tests.
- `tests/test_adversarial_assignment_audit.py` — role normalization and scene audit tests.
- `tests/test_adversarial_mix_env.py` — wrapper and small PPO integration tests.

**Modify**

- `setup.py` — declare `PyYAML>=6.0`.
- `pufferlib/ocean/environment.py` — register `puffer_drive_adversarial`.

**Explicitly unchanged**

- `pufferlib/pufferl.py`
- `pufferlib/policy_mix.py`
- `pufferlib/ocean/drive/drive.py`
- `pufferlib/ocean/drive/drive.h`
- `pufferlib/ocean/drive/binding.c`
- `pufferlib/evaluation/collision_classifier.py` and validation at-fault / PDM metric definitions

---

### Task 1: Configuration Schema and Strategy Registry

**Files:**
- Create: `pufferlib/adversarial/__init__.py`
- Create: `pufferlib/adversarial/config.py`
- Create: `pufferlib/adversarial/registry.py`
- Create: `pufferlib/adversarial/sampler.py`
- Modify: `setup.py:317-331`
- Test: `tests/test_adversarial_config.py`

**Interfaces:**
- Produces: `load_adversarial_config(path: str | Path) -> AdversarialConfig`
- Produces: `StrategyRegistry.resolve(name: str) -> StrategyDefinition`
- Produces: `OpponentSampler` protocol for later curriculum work
- Consumes: no feature-specific interfaces

- [ ] **Step 1: Write failing configuration and registry tests**

Create `tests/test_adversarial_config.py`:

```python
from pathlib import Path

import pytest
import yaml

from pufferlib.adversarial.config import load_adversarial_config
from pufferlib.adversarial.registry import DEFAULT_STRATEGY_REGISTRY


def _write_config(tmp_path: Path, overrides=None) -> Path:
    data = {
        "version": 1,
        "roles": {
            "ego": {"policy_index": 0, "strategy": "ego_drive_recurrent"},
            "primary_opponent": {
                "policy_index": 1,
                "strategy": "primary_opponent_drive_recurrent",
            },
        },
        "role_assignment": {
            "mode": "global_deficit",
            "warn_mixed_scene_rate_below": 0.8,
            "fail_mixed_scene_rate_below": None,
        },
        "reward": {
            "weights": {
                "ego_cost": 0.20,
                "fault": 1.0,
                "kinematics": 0.20,
                "normality": 0.02,
            },
            "limits": {
                "positive_reward_cap": 0.25,
                "kinematics_penalty_cap": 0.50,
                "max_invalid_reward_events": 10,
            },
            "thresholds": {
                "hard_brake_mps2": 3.0,
                "hard_steer_rad": 0.5,
                "ttc_seconds": 2.0,
                "safe_distance_m": 2.0,
                "max_accel_mps2": 4.0,
                "max_lateral_accel_mps2": 4.0,
                "max_jerk_mps3": 8.0,
                "max_steer_rate_radps": 1.0,
                "normal_speed_mps": 10.0,
                "fault_lookback_steps": 5,
            },
        },
    }
    if overrides:
        overrides(data)
    path = tmp_path / "adversarial.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_load_adversarial_config_accepts_safe_defaults(tmp_path):
    cfg = load_adversarial_config(_write_config(tmp_path))
    assert cfg.ego_policy_index == 0
    assert cfg.opponent_policy_index == 1
    assert cfg.role_assignment.mode == "global_deficit"
    assert cfg.reward.weights.fault == 1.0


def test_config_rejects_fault_weight_below_safety_ratio(tmp_path):
    path = _write_config(
        tmp_path,
        lambda data: data["reward"]["weights"].update(
            {"ego_cost": 0.4, "fault": 1.0}
        ),
    )
    with pytest.raises(ValueError, match="fault.*4"):
        load_adversarial_config(path)


def test_config_rejects_positive_cap_above_quarter(tmp_path):
    path = _write_config(
        tmp_path,
        lambda data: data["reward"]["limits"].update(
            {"positive_reward_cap": 0.3}
        ),
    )
    with pytest.raises(ValueError, match="positive_reward_cap"):
        load_adversarial_config(path)


def test_registry_resolves_only_implemented_v1_strategies():
    ego = DEFAULT_STRATEGY_REGISTRY.resolve("ego_drive_recurrent")
    opponent = DEFAULT_STRATEGY_REGISTRY.resolve(
        "primary_opponent_drive_recurrent"
    )
    assert ego.policy_index == 0
    assert ego.reward_mode == "base"
    assert opponent.policy_index == 1
    assert opponent.reward_mode == "adversarial"
    with pytest.raises(ValueError, match="Unknown or unimplemented"):
        DEFAULT_STRATEGY_REGISTRY.resolve("idm")
```

- [ ] **Step 2: Run tests and confirm the missing-module failure**

Run:

```bash
pytest -q tests/test_adversarial_config.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'pufferlib.adversarial'`.

- [ ] **Step 3: Declare PyYAML and implement the immutable config model**

Add `"PyYAML>=6.0"` to `setup.py` immediately after `"msgpack==1.1.2"`.

Create `pufferlib/adversarial/config.py` with these exact public types:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class RewardWeights:
    ego_cost: float
    fault: float
    kinematics: float
    normality: float


@dataclass(frozen=True)
class RewardLimits:
    positive_reward_cap: float
    kinematics_penalty_cap: float
    max_invalid_reward_events: int


@dataclass(frozen=True)
class RewardThresholds:
    hard_brake_mps2: float
    hard_steer_rad: float
    ttc_seconds: float
    safe_distance_m: float
    max_accel_mps2: float
    max_lateral_accel_mps2: float
    max_jerk_mps3: float
    max_steer_rate_radps: float
    normal_speed_mps: float
    fault_lookback_steps: int


@dataclass(frozen=True)
class RewardConfig:
    weights: RewardWeights
    limits: RewardLimits
    thresholds: RewardThresholds


@dataclass(frozen=True)
class RoleAssignmentConfig:
    mode: str
    warn_mixed_scene_rate_below: float
    fail_mixed_scene_rate_below: Optional[float]


@dataclass(frozen=True)
class AdversarialConfig:
    version: int
    ego_policy_index: int
    opponent_policy_index: int
    role_assignment: RoleAssignmentConfig
    reward: RewardConfig


def load_adversarial_config(path) -> AdversarialConfig:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        raise FileNotFoundError(f"Adversarial config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Adversarial config root must be a mapping")
    if raw.get("version") != 1:
        raise ValueError("Adversarial config version must be 1")

    roles = raw["roles"]
    assignment = RoleAssignmentConfig(**raw["role_assignment"])
    reward_raw = raw["reward"]
    reward = RewardConfig(
        weights=RewardWeights(**reward_raw["weights"]),
        limits=RewardLimits(**reward_raw["limits"]),
        thresholds=RewardThresholds(**reward_raw["thresholds"]),
    )
    config = AdversarialConfig(
        version=1,
        ego_policy_index=int(roles["ego"]["policy_index"]),
        opponent_policy_index=int(
            roles["primary_opponent"]["policy_index"]
        ),
        role_assignment=assignment,
        reward=reward,
    )
    _validate_config(config)
    return config


def _validate_config(config: AdversarialConfig) -> None:
    if config.ego_policy_index != 0 or config.opponent_policy_index != 1:
        raise ValueError("V1 requires Ego policy 0 and Primary Opponent policy 1")
    if config.role_assignment.mode != "global_deficit":
        raise ValueError("V1 role_assignment.mode must be global_deficit")
    if not 0 <= config.role_assignment.warn_mixed_scene_rate_below <= 1:
        raise ValueError("warn_mixed_scene_rate_below must be in [0, 1]")
    fail_rate = config.role_assignment.fail_mixed_scene_rate_below
    if fail_rate is not None and not 0 <= fail_rate <= 1:
        raise ValueError("fail_mixed_scene_rate_below must be null or in [0, 1]")
    weights = config.reward.weights
    if weights.fault < 4.0 * weights.ego_cost:
        raise ValueError("fault weight must be at least 4 times ego_cost")
    if config.reward.limits.positive_reward_cap > 0.25:
        raise ValueError("positive_reward_cap must be <= 0.25")
    if config.reward.limits.max_invalid_reward_events < 1:
        raise ValueError("max_invalid_reward_events must be positive")
    if config.reward.thresholds.fault_lookback_steps < 1:
        raise ValueError("fault_lookback_steps must be positive")
```

- [ ] **Step 4: Implement the explicit strategy registry and sampler protocol**

Create `pufferlib/adversarial/registry.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyDefinition:
    name: str
    policy_index: int
    reward_mode: str
    trainable: bool


class StrategyRegistry:
    def __init__(self):
        self._strategies = {}

    def register(self, definition: StrategyDefinition) -> None:
        if definition.name in self._strategies:
            raise ValueError(f"Strategy already registered: {definition.name}")
        self._strategies[definition.name] = definition

    def resolve(self, name: str) -> StrategyDefinition:
        try:
            return self._strategies[name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown or unimplemented adversarial strategy: {name}"
            ) from exc


DEFAULT_STRATEGY_REGISTRY = StrategyRegistry()
DEFAULT_STRATEGY_REGISTRY.register(
    StrategyDefinition("ego_drive_recurrent", 0, "base", True)
)
DEFAULT_STRATEGY_REGISTRY.register(
    StrategyDefinition(
        "primary_opponent_drive_recurrent", 1, "adversarial", True
    )
)
```

Extend `load_adversarial_config` to resolve both configured strategy names through `DEFAULT_STRATEGY_REGISTRY` and verify their policy indices match the YAML.

Create `pufferlib/adversarial/sampler.py`:

```python
from typing import Dict, Protocol


class OpponentSampler(Protocol):
    def sample(self, scene_context: Dict, history: Dict) -> str:
        """Return a registered behavior-strategy name for a future episode."""
        ...
```

Create `pufferlib/adversarial/__init__.py` exporting `AdversarialConfig`, `load_adversarial_config`, and `DEFAULT_STRATEGY_REGISTRY`.

- [ ] **Step 5: Run config tests**

Run:

```bash
pytest -q tests/test_adversarial_config.py
```

Expected: `4 passed`.

- [ ] **Step 6: Commit Task 1 if Git identity is available**

Run:

```bash
git var GIT_AUTHOR_IDENT &&
git add setup.py pufferlib/adversarial tests/test_adversarial_config.py &&
git commit -m "feat: add adversarial training configuration"
```

Expected: one commit. If `git var` fails, do not configure an identity; leave files uncommitted and report the blocker.

---

### Task 2: Pure NumPy Asymmetric Reward Evaluator

**Files:**
- Create: `pufferlib/adversarial/reward.py`
- Test: `tests/test_adversarial_reward.py`

**Interfaces:**
- Consumes: `AdversarialConfig.reward` from Task 1
- Produces: `StateFrame.from_mapping(mapping) -> StateFrame`
- Produces: `AsymmetricRewardEvaluator.reset(num_agents: int) -> None`
- Produces: `AsymmetricRewardEvaluator.evaluate(...) -> RewardEvaluation`

- [ ] **Step 1: Write synthetic reward tests before implementation**

Create `tests/test_adversarial_reward.py` with helpers that construct two-agent, one-scene states:

```python
import numpy as np

from pufferlib.adversarial.config import (
    RewardConfig,
    RewardLimits,
    RewardThresholds,
    RewardWeights,
)
from pufferlib.adversarial.reward import (
    AsymmetricRewardEvaluator,
    StateFrame,
)


def reward_config():
    return RewardConfig(
        weights=RewardWeights(0.20, 1.0, 0.20, 0.02),
        limits=RewardLimits(0.25, 0.50),
        thresholds=RewardThresholds(
            hard_brake_mps2=3.0,
            hard_steer_rad=0.5,
            ttc_seconds=2.0,
            safe_distance_m=2.0,
            max_accel_mps2=4.0,
            max_lateral_accel_mps2=4.0,
            max_jerk_mps3=8.0,
            max_steer_rate_radps=1.0,
            normal_speed_mps=10.0,
            fault_lookback_steps=5,
        ),
    )


def frame(x, y, heading, valid=None):
    x = np.asarray(x, dtype=np.float32)
    return StateFrame(
        x=x,
        y=np.asarray(y, dtype=np.float32),
        heading=np.asarray(heading, dtype=np.float32),
        length=np.full_like(x, 4.5),
        width=np.full_like(x, 2.0),
        valid=np.ones_like(x, dtype=bool) if valid is None else valid,
    )


def evaluate(pre, post, base, raw, actions=None, pre_speed=None, post_speed=None):
    evaluator = AsymmetricRewardEvaluator(reward_config(), dt=0.1)
    evaluator.reset(2)
    pre_obs = np.zeros((2, 23), dtype=np.float32)
    post_obs = np.zeros((2, 23), dtype=np.float32)
    if pre_speed is not None:
        pre_obs[:, 2] = np.asarray(pre_speed, dtype=np.float32) / 100.0
    if post_speed is not None:
        post_obs[:, 2] = np.asarray(post_speed, dtype=np.float32) / 100.0
    return evaluator.evaluate(
        base_rewards=np.asarray(base, dtype=np.float32),
        pre_state=pre,
        post_state=post,
        actions=np.zeros((2, 2), dtype=np.float32)
        if actions is None else np.asarray(actions, dtype=np.float32),
        pre_observations=pre_obs,
        post_observations=post_obs,
        raw_components=np.asarray(raw, dtype=np.float32),
        agent_offsets=np.asarray([0, 2], dtype=np.int32),
        role_ids=np.asarray([0, 1], dtype=np.int64),
        action_type="continuous",
    )


def test_ego_reward_is_bit_exact():
    pre = frame([0, 8], [0, 0], [0, 0])
    post = frame([1, 9], [0, 0], [0, 0])
    result = evaluate(pre, post, [0.1234567, 0.4], np.zeros((2, 11)))
    assert result.rewards[0] == np.float32(0.1234567)


def test_unrelated_ego_brake_does_not_reward_lateral_opponent():
    pre = frame([0, 0], [0, 10], [0, 0])
    post = frame([0.2, 0], [0, 10.5], [0, 0])
    result = evaluate(pre, post, [0, 0], np.zeros((2, 11)))
    assert result.components["ego_cost"][1] == 0


def test_closing_ttc_risk_rewards_opponent_below_positive_cap():
    pre = frame([0, 8], [0, 0], [0, 0])
    post = frame([1.0, 8.2], [0, 0], [0, 0])
    result = evaluate(pre, post, [0, 0], np.zeros((2, 11)))
    assert 0 < result.rewards[1] <= 0.25


def test_opponent_collision_is_conservatively_faulted():
    pre = frame([0, 4], [0, 2], [0, -1.57])
    post = frame([0.5, 4], [0, 1.5], [0, -1.57])
    raw = np.zeros((2, 11), dtype=np.float32)
    raw[1, 0] = 1.0
    result = evaluate(pre, post, [0, 0], raw)
    assert result.components["fault_penalty"][1] == 1.0
    assert result.rewards[1] == -1.0


def test_high_confidence_ego_rear_end_is_not_opponent_fault():
    pre = frame([0, 6], [0, 0], [0, 0])
    post = frame([1.2, 6.5], [0, 0], [0, 0])
    raw = np.zeros((2, 11), dtype=np.float32)
    raw[1, 0] = 1.0
    result = evaluate(pre, post, [0, 0], raw)
    assert result.components["fault_penalty"][1] == 0.0


def test_kinematic_violation_is_negative_and_finite():
    pre = frame([0, 8], [0, 0], [0, 0])
    post = frame([0, 10], [0, 0], [0, 0])
    result = evaluate(
        pre, post, [0, 0], np.zeros((2, 11)),
        pre_speed=[0, 0], post_speed=[0, 20],
    )
    assert result.components["kinematics_cost"][1] > 0
    assert -1 <= result.rewards[1] < 0
    assert np.isfinite(result.rewards).all()


def test_non_finite_base_reward_is_replaced_and_counted():
    pre = frame([0, 8], [0, 0], [0, 0])
    post = frame([0, 9], [0, 0], [0, 0])
    result = evaluate(pre, post, [0, np.nan], np.zeros((2, 11)))
    assert result.rewards[1] == -1.0
    assert result.metrics["adv/invalid_reward_events"] == 1
```

- [ ] **Step 2: Run the reward tests and verify missing symbols**

Run:

```bash
pytest -q tests/test_adversarial_reward.py
```

Expected: collection fails because `pufferlib.adversarial.reward` does not exist.

- [ ] **Step 3: Implement state and result types**

Create `pufferlib/adversarial/reward.py` with:

```python
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .config import RewardConfig


@dataclass(frozen=True)
class StateFrame:
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    length: np.ndarray
    width: np.ndarray
    valid: np.ndarray

    @classmethod
    def from_mapping(cls, state):
        x = np.asarray(state["x"], dtype=np.float32)
        y = np.asarray(state["y"], dtype=np.float32)
        return cls(
            x=x,
            y=y,
            heading=np.asarray(state["heading"], dtype=np.float32),
            length=np.asarray(state["length"], dtype=np.float32),
            width=np.asarray(state["width"], dtype=np.float32),
            valid=np.isfinite(x) & np.isfinite(y) & (x > -9000) & (y > -9000),
        )


@dataclass(frozen=True)
class RewardEvaluation:
    rewards: np.ndarray
    components: Dict[str, np.ndarray]
    metrics: Dict[str, float]
```

- [ ] **Step 4: Implement scene-local TTC, causality, fault, and kinematics**

Implement `AsymmetricRewardEvaluator` with this public skeleton:

```python
class AsymmetricRewardEvaluator:
    def __init__(self, config: RewardConfig, dt: float):
        self.config = config
        self.dt = float(dt)
        self._previous_accel = None

    def reset(self, num_agents: int) -> None:
        self._previous_accel = np.zeros(num_agents, dtype=np.float32)

    def evaluate(
        self,
        *,
        base_rewards,
        pre_state,
        post_state,
        actions,
        pre_observations,
        post_observations,
        raw_components,
        agent_offsets,
        role_ids,
        action_type,
    ) -> RewardEvaluation:
        base_rewards = np.asarray(base_rewards, dtype=np.float32)
        rewards = base_rewards.copy()
        role_ids = np.asarray(role_ids, dtype=np.int64)
        n = len(base_rewards)
        if self._previous_accel is None or len(self._previous_accel) != n:
            self.reset(n)

        ego_cost = np.zeros(n, dtype=np.float32)
        fault_penalty = np.zeros(n, dtype=np.float32)
        kinematics_cost = np.zeros(n, dtype=np.float32)
        normality = np.zeros(n, dtype=np.float32)

        pre_speed = np.abs(
            np.asarray(pre_observations, dtype=np.float32)[:, 2]
        ) * 100.0
        post_speed = np.abs(
            np.asarray(post_observations, dtype=np.float32)[:, 2]
        ) * 100.0
        accel = (post_speed - pre_speed) / self.dt
        jerk = (accel - self._previous_accel) / self.dt

        for scene_idx in range(len(agent_offsets) - 1):
            start = int(agent_offsets[scene_idx])
            stop = int(agent_offsets[scene_idx + 1])
            scene_roles = role_ids[start:stop]
            ego_indices = start + np.flatnonzero(scene_roles == 0)
            opponent_indices = start + np.flatnonzero(scene_roles == 1)
            for opponent_idx in opponent_indices:
                ego_cost[opponent_idx] = self._max_causal_ego_cost(
                    opponent_idx,
                    ego_indices,
                    pre_state,
                    post_state,
                    accel,
                )
                fault_penalty[opponent_idx] = self._fault_penalty(
                    opponent_idx,
                    ego_indices,
                    pre_state,
                    post_state,
                    accel,
                    actions,
                    raw_components,
                    action_type,
                )
                kinematics_cost[opponent_idx] = self._kinematics_cost(
                    opponent_idx,
                    pre_state,
                    post_state,
                    accel,
                    jerk,
                    actions,
                    raw_components,
                    action_type,
                )
                normality[opponent_idx] = np.clip(
                    post_speed[opponent_idx]
                    / self.config.thresholds.normal_speed_mps,
                    0.0,
                    1.0,
                )

        opponent_mask = role_ids == 1
        w = self.config.weights
        raw_adversarial = (
            w.ego_cost * ego_cost
            - w.fault * fault_penalty
            - w.kinematics * kinematics_cost
            + w.normality * normality
        )
        positive = np.minimum(
            np.maximum(raw_adversarial, 0.0),
            self.config.limits.positive_reward_cap,
        )
        negative = np.minimum(raw_adversarial, 0.0)
        adversarial = np.clip(positive + negative, -1.0, 1.0)
        adversarial[fault_penalty > 0] = -1.0
        rewards[opponent_mask] = adversarial[opponent_mask]
        invalid_reward_mask = ~np.isfinite(rewards)
        rewards = np.nan_to_num(rewards, nan=-1.0, posinf=-1.0, neginf=-1.0)
        self._previous_accel = accel.astype(np.float32, copy=True)

        components = {
            "ego_cost": ego_cost,
            "fault_penalty": fault_penalty,
            "kinematics_cost": kinematics_cost,
            "normality": normality,
        }
        metrics = self._aggregate_metrics(components, rewards, opponent_mask)
        metrics["adv/invalid_reward_events"] = float(
            invalid_reward_mask.sum()
        )
        return RewardEvaluation(rewards, components, metrics)
```

Implement private helpers with these rules:

- `_max_causal_ego_cost`: process only valid Ego pairs; transform relative position into Ego heading frame; require opponent longitudinal coordinate `> 0`, absolute lateral coordinate below `max(ego.width, opponent.width)`, and positive closing speed; use `max(0, (ttc_threshold - ttc) / ttc_threshold)` plus normalized Ego hard-brake magnitude; return the maximum pair cost clipped to `1`.
- `_fault_penalty`: if `raw_components[opponent_idx, 0] <= 0`, return `0`; otherwise return `0` only for the high-confidence rear-end case: Ego was behind Opponent in Opponent frame, Ego was closing, Opponent absolute acceleration stayed below `hard_brake_mps2`, and Opponent steering command stayed below `hard_steer_rad`; all ambiguous collisions return `1`.
- `_kinematics_cost`: sum normalized hinge violations for acceleration, jerk, heading-derived lateral acceleration, steering/steering-rate, offroad (`raw_components[:, 1]`), reverse (`[:, 9]`), and speed limit (`[:, 10]`); clip to `kinematics_penalty_cap / max(weights.kinematics, 1e-6)`.
- `_aggregate_metrics`: return scalar means/sums under the Opponent mask using keys `adv/cost_ego`, `adv/penalty_fault`, `adv/cost_kinematics`, `adv/normality`, `adv/reward_total`, and `adv/fault_collision_events`.
- Decode classic discrete actions with 7 acceleration bins `[-4, -2.67, -1.33, 0, 1.33, 2.67, 4]` and 13 steering bins from `-1` to `1`; continuous actions use their two columns directly.

- [ ] **Step 5: Run reward tests and refine only evidenced failures**

Run:

```bash
pytest -q tests/test_adversarial_reward.py
```

Expected: all tests pass. Do not loosen the conservative collision test to make ambiguous collisions positive.

- [ ] **Step 6: Commit Task 2 if Git identity is available**

```bash
git var GIT_AUTHOR_IDENT &&
git add pufferlib/adversarial/reward.py tests/test_adversarial_reward.py &&
git commit -m "feat: add asymmetric adversarial reward evaluator"
```

---

### Task 3: Scene Audit and Adversarial Environment Wrapper

**Files:**
- Create: `pufferlib/adversarial/env.py`
- Modify: `pufferlib/adversarial/__init__.py`
- Modify: `pufferlib/ocean/environment.py:182-186`
- Test: `tests/test_adversarial_assignment_audit.py`
- Test: `tests/test_adversarial_mix_env.py`

**Interfaces:**
- Consumes: `load_adversarial_config`, `AsymmetricRewardEvaluator`, existing `Drive.policy_log_ids`, `Drive.agent_offsets`
- Produces: `normalize_role_ids(policy_log_ids, num_agents) -> np.ndarray`
- Produces: `audit_scene_roles(agent_offsets, role_ids) -> Dict[str, float]`
- Produces: `AdversarialMixDrive(Drive)`

- [ ] **Step 1: Write role normalization and scene-audit tests**

Create `tests/test_adversarial_assignment_audit.py`:

```python
import numpy as np

from pufferlib.adversarial.env import audit_scene_roles, normalize_role_ids


def test_normalize_role_ids_repeats_existing_mix_pattern():
    ids = normalize_role_ids([0, 1], 5)
    assert ids.tolist() == [0, 1, 0, 1, 0]


def test_scene_audit_allows_natural_homogeneous_small_scenes():
    metrics = audit_scene_roles(
        np.asarray([0, 2, 3, 6], dtype=np.int32),
        np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
    )
    assert metrics["assignment/mixed_scene_rate"] == 2 / 3
    assert metrics["assignment/ego_only_scene_rate"] == 1 / 3
    assert metrics["assignment/opponent_only_scene_rate"] == 0


def test_scene_audit_rejects_invalid_offsets():
    with np.testing.assert_raises_regex(ValueError, "agent_offsets"):
        audit_scene_roles(
            np.asarray([0, 3, 2], dtype=np.int32),
            np.asarray([0, 1, 0], dtype=np.int64),
        )
```

- [ ] **Step 2: Write a wrapper-level reward-routing test with a fake base step**

In `tests/test_adversarial_mix_env.py`, create an instance through `__new__` so the test does not require C:

```python
import numpy as np

from pufferlib.adversarial.env import AdversarialMixDrive
from pufferlib.adversarial.reward import RewardEvaluation, StateFrame


class FakeEvaluator:
    def evaluate(self, **kwargs):
        rewards = kwargs["base_rewards"].copy()
        rewards[1::2] = 0.2
        return RewardEvaluation(
            rewards=rewards,
            components={},
            metrics={"adv/reward_total": 0.2},
        )

    def reset(self, num_agents):
        return None


def test_route_rewards_preserves_ego_and_replaces_opponent():
    env = AdversarialMixDrive.__new__(AdversarialMixDrive)
    env.num_agents = 4
    env._role_ids = np.asarray([0, 1, 0, 1])
    env.agent_offsets = np.asarray([0, 2, 4], dtype=np.int32)
    env._adversarial_evaluator = FakeEvaluator()
    env._action_type_flag = 1
    env.observations = np.zeros((4, 10), dtype=np.float32)
    env._pre_observations = np.zeros((4, 10), dtype=np.float32)
    env.reward_components_raw = np.zeros((4, 11), dtype=np.float32)
    env._pre_state = StateFrame(
        x=np.zeros(4), y=np.zeros(4), heading=np.zeros(4),
        length=np.ones(4), width=np.ones(4),
        valid=np.ones(4, dtype=bool),
    )
    env.rewards = np.asarray([0.1, -0.1, 0.3, -0.3], dtype=np.float32)
    env.get_global_agent_state = lambda: {
        "x": np.zeros(4), "y": np.zeros(4), "heading": np.zeros(4),
        "length": np.ones(4), "width": np.ones(4),
    }

    env._apply_adversarial_rewards(np.zeros((4, 2), dtype=np.float32))

    assert env.rewards.tolist() == [
        np.float32(0.1), np.float32(0.2),
        np.float32(0.3), np.float32(0.2),
    ]
```

- [ ] **Step 3: Run tests and verify missing wrapper helpers**

```bash
pytest -q \
  tests/test_adversarial_assignment_audit.py \
  tests/test_adversarial_mix_env.py
```

Expected: collection fails because `pufferlib.adversarial.env` does not exist.

- [ ] **Step 4: Implement role normalization and scene audit**

Create `pufferlib/adversarial/env.py` starting with:

```python
from pathlib import Path

import numpy as np

from pufferlib.ocean.drive.drive import Drive

from .config import load_adversarial_config
from .reward import AsymmetricRewardEvaluator, StateFrame


def normalize_role_ids(policy_log_ids, num_agents):
    if not policy_log_ids:
        raise ValueError("AdversarialMixDrive requires mix_ppo policy_log_ids")
    ids = np.asarray(policy_log_ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) == 0:
        raise ValueError("policy_log_ids must be a non-empty 1D sequence")
    repeats = (num_agents + len(ids) - 1) // len(ids)
    ids = np.tile(ids, repeats)[:num_agents]
    unknown = np.setdiff1d(np.unique(ids), np.asarray([0, 1]))
    if len(unknown):
        raise ValueError(f"V1 supports only policy ids 0 and 1, got {unknown}")
    return ids


def audit_scene_roles(agent_offsets, role_ids):
    offsets = np.asarray(agent_offsets, dtype=np.int64)
    role_ids = np.asarray(role_ids, dtype=np.int64)
    if (
        offsets.ndim != 1
        or len(offsets) < 2
        or offsets[0] != 0
        or offsets[-1] != len(role_ids)
        or np.any(np.diff(offsets) < 0)
    ):
        raise ValueError("agent_offsets must be monotonic and span role_ids")

    mixed = ego_only = opponent_only = 0
    size_buckets = {}
    for start, stop in zip(offsets[:-1], offsets[1:]):
        roles = role_ids[start:stop]
        has_ego = np.any(roles == 0)
        has_opponent = np.any(roles == 1)
        kind = "mixed" if has_ego and has_opponent else (
            "ego_only" if has_ego else "opponent_only"
        )
        mixed += kind == "mixed"
        ego_only += kind == "ego_only"
        opponent_only += kind == "opponent_only"
        size = int(stop - start)
        bucket = size_buckets.setdefault(
            size, {"total": 0, "mixed": 0}
        )
        bucket["total"] += 1
        bucket["mixed"] += kind == "mixed"

    total = max(len(offsets) - 1, 1)
    metrics = {
        "assignment/mixed_scene_rate": mixed / total,
        "assignment/ego_only_scene_rate": ego_only / total,
        "assignment/opponent_only_scene_rate": opponent_only / total,
    }
    for size, counts in size_buckets.items():
        metrics[f"assignment/size_{size}/mixed_rate"] = (
            counts["mixed"] / counts["total"]
        )
    return metrics
```

- [ ] **Step 5: Implement `AdversarialMixDrive` without changing base Drive**

Continue `pufferlib/adversarial/env.py`:

```python
class AdversarialMixDrive(Drive):
    def __init__(self, adversarial_config_path, **kwargs):
        self.adversarial_config_path = str(adversarial_config_path)
        self.adversarial_config = load_adversarial_config(
            self.adversarial_config_path
        )
        super().__init__(**kwargs)
        if self.policy_log_count != 2:
            raise ValueError("AdversarialMixDrive requires exactly 2 mix_ppo policies")
        self._role_ids = normalize_role_ids(
            self.policy_log_ids, self.num_agents
        )
        self._adversarial_evaluator = AsymmetricRewardEvaluator(
            self.adversarial_config.reward, dt=self.dt
        )
        self._assignment_metrics = audit_scene_roles(
            self.agent_offsets, self._role_ids
        )
        self._pending_adversarial_metrics = {}
        self._invalid_reward_events = 0
        self._pre_state = StateFrame.from_mapping(
            self.get_global_agent_state()
        )
        self._pre_observations = self.observations.copy()
        self._check_assignment_thresholds()

    def _check_assignment_thresholds(self):
        import warnings

        mixed_rate = self._assignment_metrics[
            "assignment/mixed_scene_rate"
        ]
        warn_below = (
            self.adversarial_config.role_assignment
            .warn_mixed_scene_rate_below
        )
        if mixed_rate < warn_below:
            warnings.warn(
                f"mixed_scene_rate {mixed_rate:.3f} is below "
                f"warning threshold {warn_below:.3f}",
                RuntimeWarning,
            )
        fail_below = (
            self.adversarial_config.role_assignment
            .fail_mixed_scene_rate_below
        )
        if fail_below is not None and mixed_rate < fail_below:
            raise ValueError(
                f"mixed_scene_rate {mixed_rate:.3f} is below "
                f"configured failure threshold {fail_below:.3f}"
            )

    def reset(self, seed=0):
        observations, info = super().reset(seed)
        self._adversarial_evaluator.reset(self.num_agents)
        self._pre_state = StateFrame.from_mapping(
            self.get_global_agent_state()
        )
        self._pre_observations = self.observations.copy()
        return observations, info

    def resample_maps(self):
        super().resample_maps()
        self._assignment_metrics = audit_scene_roles(
            self.agent_offsets, self._role_ids
        )
        self._check_assignment_thresholds()
        self._adversarial_evaluator.reset(self.num_agents)
        self._pre_state = StateFrame.from_mapping(
            self.get_global_agent_state()
        )
        self._pre_observations = self.observations.copy()

    def _apply_adversarial_rewards(self, actions):
        base_rewards = self.rewards.copy()
        post_state = StateFrame.from_mapping(
            self.get_global_agent_state()
        )
        result = self._adversarial_evaluator.evaluate(
            base_rewards=base_rewards,
            pre_state=self._pre_state,
            post_state=post_state,
            actions=actions,
            pre_observations=self._pre_observations,
            post_observations=self.observations,
            raw_components=self.reward_components_raw,
            agent_offsets=self.agent_offsets,
            role_ids=self._role_ids,
            action_type=(
                "discrete" if self._action_type_flag == 0 else "continuous"
            ),
        )
        self.rewards[:] = result.rewards
        self._pending_adversarial_metrics = result.metrics
        self._invalid_reward_events += int(
            result.metrics.get("adv/invalid_reward_events", 0)
        )
        if (
            self._invalid_reward_events
            > self.adversarial_config.reward.limits.max_invalid_reward_events
        ):
            raise FloatingPointError(
                "Adversarial reward exceeded max_invalid_reward_events"
            )
        self._pre_state = post_state
        self._pre_observations = self.observations.copy()

    def step(self, actions):
        will_resample = (
            self.tick > 0
            and self.resample_frequency > 0
            and self.tick % self.resample_frequency == 0
        )
        result = super().step(actions)
        if not will_resample:
            self._apply_adversarial_rewards(actions)
        observations, rewards, terminals, truncations, info = result
        if self.tick % self.report_interval == 0:
            metrics = {
                **self._assignment_metrics,
                **self._pending_adversarial_metrics,
            }
            info.append({"adversarial": metrics})
        return observations, rewards, terminals, truncations, info
```

After tests expose lifecycle edge cases, adjust only:

- the resample branch so stale base rewards are never transformed;
- invalid post-collision positions through `StateFrame.valid`;
- duplicate report dictionaries if `super().step()` already emitted logs.

Do not modify `Drive.step()`.

- [ ] **Step 6: Register the new environment lazily**

Modify `pufferlib/ocean/environment.py`:

```python
MAKE_FUNCTIONS = {
    "drive": "Drive",
    "drive_adversarial": lazy_import(
        "pufferlib.adversarial.env", "AdversarialMixDrive"
    ),
    "spaces": make_spaces,
    "multiagent": make_multiagent,
}
```

Export `AdversarialMixDrive` from `pufferlib/adversarial/__init__.py`.

- [ ] **Step 7: Run wrapper tests**

```bash
pytest -q \
  tests/test_adversarial_assignment_audit.py \
  tests/test_adversarial_mix_env.py
```

Expected: all tests pass without loading or rebuilding the C extension.

- [ ] **Step 8: Commit Task 3 if Git identity is available**

```bash
git var GIT_AUTHOR_IDENT &&
git add \
  pufferlib/adversarial/env.py \
  pufferlib/adversarial/__init__.py \
  pufferlib/ocean/environment.py \
  tests/test_adversarial_assignment_audit.py \
  tests/test_adversarial_mix_env.py &&
git commit -m "feat: add adversarial mixed drive wrapper"
```

---

### Task 4: Independent INI/YAML Training Configuration

**Files:**
- Create: `pufferlib/config/adversarial/opponent_mix.yaml`
- Create: `pufferlib/config/ocean/drive_adversarial.ini`
- Modify: `tests/test_adversarial_config.py`

**Interfaces:**
- Consumes: `AdversarialMixDrive.__init__(adversarial_config_path, **kwargs)`
- Produces: loadable environment name `puffer_drive_adversarial`

- [ ] **Step 1: Add a failing integration-level config test**

Append to `tests/test_adversarial_config.py`:

```python
from unittest.mock import patch

from pufferlib.pufferl import load_config


@patch("sys.argv", ["pufferl.py"])
def test_load_drive_adversarial_profile():
    args = load_config("puffer_drive_adversarial")
    assert args["env_name"] == "puffer_drive_adversarial"
    assert args["policy_name"] == "Drive"
    assert args["rnn_name"] == "Recurrent"
    assert args["train"]["mix_ppo"] is True
    assert args["train"]["mix_ppo_policy_mix"] == (
        "ego:0.5, primary_opponent:0.5"
    )
    assert args["train"]["mix_ppo_policy_names"] == "Drive,Drive"
    assert args["train"]["mix_ppo_rnn_names"] == "Recurrent,Recurrent"
    assert args["env"]["adversarial_config_path"].endswith(
        "opponent_mix.yaml"
    )
```

- [ ] **Step 2: Run the profile test and confirm no config is found**

```bash
pytest -q tests/test_adversarial_config.py::test_load_drive_adversarial_profile
```

Expected: FAIL with `No config for env_name puffer_drive_adversarial`.

- [ ] **Step 3: Create the default YAML**

Create `pufferlib/config/adversarial/opponent_mix.yaml`:

```yaml
version: 1

roles:
  ego:
    policy_index: 0
    strategy: ego_drive_recurrent
  primary_opponent:
    policy_index: 1
    strategy: primary_opponent_drive_recurrent

role_assignment:
  mode: global_deficit
  warn_mixed_scene_rate_below: 0.8
  fail_mixed_scene_rate_below: null

reward:
  weights:
    ego_cost: 0.20
    fault: 1.00
    kinematics: 0.20
    normality: 0.02
  limits:
    positive_reward_cap: 0.25
    kinematics_penalty_cap: 0.50
    max_invalid_reward_events: 10
  thresholds:
    hard_brake_mps2: 3.0
    hard_steer_rad: 0.5
    ttc_seconds: 2.0
    safe_distance_m: 2.0
    max_accel_mps2: 4.0
    max_lateral_accel_mps2: 4.0
    max_jerk_mps3: 8.0
    max_steer_rate_radps: 1.0
    normal_speed_mps: 10.0
    fault_lookback_steps: 5
```

- [ ] **Step 4: Create the independent INI profile**

Copy the complete current baseline first:

```bash
cp pufferlib/config/ocean/drive.ini \
  pufferlib/config/ocean/drive_adversarial.ini
```

Then make only these semantic changes:

```ini
[base]
package = ocean
env_name = puffer_drive_adversarial
policy_name = Drive
rnn_name = Recurrent
```

Add under `[env]`:

```ini
adversarial_config_path = pufferlib/config/adversarial/opponent_mix.yaml
```

Set under `[train]`:

```ini
mix_ppo = True
mix_ppo_policy_mix = ego:0.5, primary_opponent:0.5
mix_ppo_policy_names = Drive,Drive
mix_ppo_rnn_names = Recurrent,Recurrent
mix_ppo_rnn_input_sizes =
mix_ppo_rnn_hidden_sizes =
mix_ppo_policy_paths = ,
mix_ppo_policy_trainable = True,True
opponent_pool = False
compile = False
```

Do not change the copied baseline reward values: those remain the Ego reward and the C-computed base reward.

- [ ] **Step 5: Run configuration tests**

```bash
pytest -q tests/test_adversarial_config.py tests/test_drive_config.py
```

Expected: all tests pass, including all original `puffer_drive` assertions.

- [ ] **Step 6: Commit Task 4 if Git identity is available**

```bash
git var GIT_AUTHOR_IDENT &&
git add \
  pufferlib/config/adversarial/opponent_mix.yaml \
  pufferlib/config/ocean/drive_adversarial.ini \
  tests/test_adversarial_config.py &&
git commit -m "config: add adversarial mixed training profile"
```

---

### Task 5: End-to-End Joint PPO Verification

**Files:**
- Modify: `tests/test_adversarial_mix_env.py`
- Reference: `tests/test_mix_ppo.py:419-578`

**Interfaces:**
- Consumes: all Tasks 1–4
- Produces: evidence that existing mix_ppo trains both policies with role-specific rewards

- [ ] **Step 1: Add a one-update integration test**

Append a test patterned after `TestMixPPO.test_mix_ppo_runs_one_small_training_update`, but load `puffer_drive_adversarial`:

```python
import os
from unittest.mock import patch

import torch

from pufferlib.pufferl import (
    PuffeRL,
    load_config,
    load_env,
    load_mixed_policies,
    load_policy,
)


@patch("sys.argv", ["pufferl.py"])
def test_adversarial_mix_runs_one_joint_update():
    args = load_config("puffer_drive_adversarial")
    args["train"].update({
        "device": "cpu",
        "optimizer": "adam",
        "compile": False,
        "total_timesteps": 16,
        "batch_size": 16,
        "bptt_horizon": 4,
        "minibatch_size": 16,
        "max_minibatch_size": 16,
        "update_epochs": 1,
        "render": False,
        "checkpoint_interval": 999999,
    })
    args["vec"].update({
        "num_workers": 1,
        "num_envs": 1,
        "batch_size": 1,
    })
    args["env"].update({
        "num_agents": 4,
        "action_type": "discrete",
        "num_maps": 1,
        "init_mode": "create_all_valid",
        "control_mode": "control_agents",
        "episode_length": 2,
        "resample_frequency": 1000,
        "report_interval": 1,
    })
    args["policy"].update({"input_size": 32, "hidden_size": 32})
    args["rnn"].update({"input_size": 32, "hidden_size": 32})
    args["eval"] = {
        "wosac_realism_eval": False,
        "human_replay_eval": False,
    }

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    old_root = os.environ.get("DRIVE_BINARIES_DATA_ROOT")
    os.environ["DRIVE_BINARIES_DATA_ROOT"] = os.path.join(
        root, "resources", "drive", "binaries"
    )
    os.environ["PUFFER_DISABLE_VIDEO"] = "1"
    vecenv = trainer = None
    try:
        vecenv = load_env("puffer_drive_adversarial", args)
        base_policy = load_policy(
            args, vecenv, "puffer_drive_adversarial"
        )
        policies = load_mixed_policies(
            args,
            vecenv,
            "puffer_drive_adversarial",
            base_policy,
        )
        before = [
            {name: value.detach().clone()
             for name, value in policy.state_dict().items()}
            for policy in policies
        ]
        train_config = dict(
            **args["train"],
            env="puffer_drive_adversarial",
            eval=args["eval"],
        )
        trainer = PuffeRL(train_config, vecenv, policies, logger=None)
        trainer.evaluate()
        assert "adversarial/assignment/mixed_scene_rate" in trainer.stats
        assert "adversarial/adv/reward_total" in trainer.stats
        trainer.train()

        for policy_idx, policy in enumerate(trainer.uncompiled_policies):
            assert any(
                not torch.equal(before[policy_idx][name], value)
                for name, value in policy.state_dict().items()
            )
        assert trainer.mix_ppo is True
        assert len(trainer.policies) == 2
    finally:
        if trainer is not None:
            trainer.utilization.stop()
        if vecenv is not None:
            vecenv.close()
        if old_root is None:
            os.environ.pop("DRIVE_BINARIES_DATA_ROOT", None)
        else:
            os.environ["DRIVE_BINARIES_DATA_ROOT"] = old_root
```

- [ ] **Step 2: Run the integration test**

```bash
pytest -q \
  tests/test_adversarial_mix_env.py::test_adversarial_mix_runs_one_joint_update
```

Expected: PASS. If the resource split path differs, use the exact resource-root setup already used by `tests/test_mix_ppo.py`; do not skip the test.

- [ ] **Step 3: Run focused regression tests**

```bash
pytest -q \
  tests/test_adversarial_config.py \
  tests/test_adversarial_reward.py \
  tests/test_adversarial_assignment_audit.py \
  tests/test_adversarial_mix_env.py \
  tests/test_mix_ppo.py \
  tests/test_drive_config.py \
  tests/test_collision_classifier.py
```

Expected: all tests pass.

- [ ] **Step 4: Check edited-file diagnostics**

Read IDE diagnostics for:

```text
pufferlib/adversarial/
pufferlib/ocean/environment.py
tests/test_adversarial_*.py
```

Expected: no new errors.

- [ ] **Step 5: Measure wrapper overhead**

Run the existing simulator/performance harness once with `puffer_drive` and once with `puffer_drive_adversarial`, using identical agent count, map count, action type, and 1,000 warm steps plus at least 10,000 measured steps.

Record:

```text
baseline_sps
adversarial_sps
overhead = 1 - adversarial_sps / baseline_sps
```

Expected: overhead `<= 0.10`. If it exceeds 10%, profile Python scene loops and state copies before proposing any C changes.

- [ ] **Step 6: Commit Task 5 if Git identity is available**

```bash
git var GIT_AUTHOR_IDENT &&
git add tests/test_adversarial_mix_env.py &&
git commit -m "test: verify adversarial joint PPO training"
```

---

### Task 6: Documentation and Final Compatibility Check

**Files:**
- Modify: `TRAINING.md`
- Modify: `docs/superpowers/specs/2026-07-16-adversarial-mixed-training-design.zh.md` only if implementation evidence requires a factual correction

**Interfaces:**
- Consumes: verified commands and configuration from Tasks 1–5
- Produces: reproducible launch and evaluation instructions

- [ ] **Step 1: Document the training command and role semantics**

Add a section to `TRAINING.md` containing:

```markdown
## Adversarial Mixed Training

Train two `Drive + Recurrent` policies in the same global agent population:

```bash
puffer train puffer_drive_adversarial \
  --config pufferlib/config/ocean/drive_adversarial.ini
```

- policy 0 (`ego`) uses the original Drive reward.
- policy 1 (`primary_opponent`) uses the asymmetric adversarial reward.
- Existing `mix_ppo` performs global deficit-based policy assignment and owns
  BPTT routing, optimizers, checkpoints, and per-policy logs.
- Small scenes may naturally be Ego-only or Opponent-only; inspect
  `adversarial/assignment/*` metrics rather than forcing per-scene rounding.

Override the global ratio without changing code:

```bash
puffer train puffer_drive_adversarial \
  --config pufferlib/config/ocean/drive_adversarial.ini \
  --train.mix-ppo-policy-mix "ego:0.75,primary_opponent:0.25"
```

The original command remains unchanged:

```bash
puffer train puffer_drive
```
```

- [ ] **Step 2: Run the original default-path smoke tests once more**

```bash
pytest -q tests/test_mix_ppo.py tests/test_drive_config.py
```

Expected: all tests pass.

- [ ] **Step 3: Verify no forbidden core files changed**

```bash
git diff --name-only -- \
  pufferlib/pufferl.py \
  pufferlib/policy_mix.py \
  pufferlib/ocean/drive/drive.py \
  pufferlib/ocean/drive/drive.h \
  pufferlib/ocean/drive/binding.c
```

Expected: no changes introduced by this implementation. Pre-existing user changes may already appear; compare against the pre-implementation status snapshot and do not overwrite them.

- [ ] **Step 4: Commit documentation if Git identity is available**

```bash
git var GIT_AUTHOR_IDENT &&
git add TRAINING.md \
  docs/superpowers/specs/2026-07-16-adversarial-mixed-training-design.zh.md &&
git commit -m "docs: explain adversarial mixed training"
```

- [ ] **Step 5: Final verification summary**

Report:

- exact tests run and pass/fail counts;
- baseline and adversarial SPS plus measured overhead;
- files created/modified;
- confirmation that Ego reward equality tests pass;
- confirmation that both optimizers update;
- confirmation that original `puffer_drive` tests remain green;
- any skipped commit caused by missing Git identity.
