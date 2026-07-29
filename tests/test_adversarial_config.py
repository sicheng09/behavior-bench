from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from pufferlib.adversarial.config import load_adversarial_config
from pufferlib.adversarial.registry import DEFAULT_STRATEGY_REGISTRY
from pufferlib.pufferl import load_config


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
    assert cfg.reward.weights.goal == 0.0


def test_load_adversarial_config_accepts_weak_goal_weight(tmp_path):
    path = _write_config(
        tmp_path,
        lambda data: data["reward"]["weights"].update({"goal": 0.15}),
    )
    cfg = load_adversarial_config(path)
    assert cfg.reward.weights.goal == 0.15


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


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("weights", "fault", float("nan")),
        ("weights", "ego_cost", float("inf")),
        ("limits", "positive_reward_cap", float("nan")),
    ],
)
def test_config_rejects_non_finite_reward_values(
    tmp_path, section, field, value
):
    path = _write_config(
        tmp_path,
        lambda data: data["reward"][section].update({field: value}),
    )
    with pytest.raises(ValueError, match="finite"):
        load_adversarial_config(path)


def test_loader_rejects_unknown_strategy(tmp_path):
    path = _write_config(
        tmp_path,
        lambda data: data["roles"]["primary_opponent"].update(
            {"strategy": "idm"}
        ),
    )
    with pytest.raises(ValueError, match="Unknown or unimplemented"):
        load_adversarial_config(path)


def test_config_rejects_non_positive_physical_threshold(tmp_path):
    path = _write_config(
        tmp_path,
        lambda data: data["reward"]["thresholds"].update(
            {"normal_speed_mps": 0.0}
        ),
    )
    with pytest.raises(ValueError, match="normal_speed_mps.*positive"):
        load_adversarial_config(path)


def test_config_rejects_negative_reward_weight(tmp_path):
    path = _write_config(
        tmp_path,
        lambda data: data["reward"]["weights"].update(
            {"normality": -0.1}
        ),
    )
    with pytest.raises(ValueError, match="normality.*non-negative"):
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
