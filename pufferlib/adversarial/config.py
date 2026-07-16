from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional

import yaml

from pufferlib.adversarial.registry import DEFAULT_STRATEGY_REGISTRY


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


def load_adversarial_config(path: str | Path) -> AdversarialConfig:
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
    ego_role = roles["ego"]
    opponent_role = roles["primary_opponent"]
    ego_strategy = DEFAULT_STRATEGY_REGISTRY.resolve(ego_role["strategy"])
    opponent_strategy = DEFAULT_STRATEGY_REGISTRY.resolve(
        opponent_role["strategy"]
    )
    ego_policy_index = int(ego_role["policy_index"])
    opponent_policy_index = int(opponent_role["policy_index"])
    if ego_strategy.policy_index != ego_policy_index:
        raise ValueError("Ego strategy policy index does not match YAML")
    if opponent_strategy.policy_index != opponent_policy_index:
        raise ValueError("Primary Opponent strategy policy index does not match YAML")

    assignment = RoleAssignmentConfig(**raw["role_assignment"])
    reward_raw = raw["reward"]
    reward = RewardConfig(
        weights=RewardWeights(**reward_raw["weights"]),
        limits=RewardLimits(**reward_raw["limits"]),
        thresholds=RewardThresholds(**reward_raw["thresholds"]),
    )
    config = AdversarialConfig(
        version=1,
        ego_policy_index=ego_policy_index,
        opponent_policy_index=opponent_policy_index,
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
    _require_finite(
        "role_assignment",
        {
            "warn_mixed_scene_rate_below": (
                config.role_assignment.warn_mixed_scene_rate_below
            ),
            **(
                {}
                if config.role_assignment.fail_mixed_scene_rate_below is None
                else {
                    "fail_mixed_scene_rate_below": (
                        config.role_assignment.fail_mixed_scene_rate_below
                    )
                }
            ),
        },
    )
    if not 0 <= config.role_assignment.warn_mixed_scene_rate_below <= 1:
        raise ValueError("warn_mixed_scene_rate_below must be in [0, 1]")
    fail_rate = config.role_assignment.fail_mixed_scene_rate_below
    if fail_rate is not None and not 0 <= fail_rate <= 1:
        raise ValueError("fail_mixed_scene_rate_below must be null or in [0, 1]")
    weights = config.reward.weights
    _require_finite("reward.weights", vars(weights))
    _require_finite(
        "reward.limits",
        {
            "positive_reward_cap": config.reward.limits.positive_reward_cap,
            "kinematics_penalty_cap": (
                config.reward.limits.kinematics_penalty_cap
            ),
        },
    )
    _require_finite(
        "reward.thresholds",
        {
            name: value
            for name, value in vars(config.reward.thresholds).items()
            if name != "fault_lookback_steps"
        },
    )
    if weights.fault < 4.0 * weights.ego_cost:
        raise ValueError("fault weight must be at least 4 times ego_cost")
    if config.reward.limits.positive_reward_cap > 0.25:
        raise ValueError("positive_reward_cap must be <= 0.25")
    if config.reward.limits.max_invalid_reward_events < 1:
        raise ValueError("max_invalid_reward_events must be positive")
    if config.reward.thresholds.fault_lookback_steps < 1:
        raise ValueError("fault_lookback_steps must be positive")


def _require_finite(group: str, values: dict) -> None:
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{group}.{name} must be a finite number")
        if not math.isfinite(value):
            raise ValueError(f"{group}.{name} must be finite")
