from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pufferlib.ocean.environment import MAKE_FUNCTIONS
from pufferlib.pufferl import load_config

from pufferlib.conservative.env import (
    ConservativeMixDrive,
    audit_scene_roles,
    normalize_role_ids,
)
from pufferlib.conservative.reward import compute_lead_gap_and_speed
from pufferlib.ocean.drive.drive import Drive


def test_normalize_role_ids_tiles_and_maps_all_nonzero_policies_to_partner():
    ids = normalize_role_ids([0, 1], 5)
    assert list(ids) == [0, 1, 0, 1, 0]
    ids = normalize_role_ids([0, 1, 2, 0], 6, policy_log_count=3)
    assert list(ids) == [0, 1, 1, 0, 0, 1]
    with pytest.raises(ValueError):
        normalize_role_ids([0, -1], 4)
    with pytest.raises(ValueError):
        normalize_role_ids([0, 3], 4, policy_log_count=3)


def test_audit_scene_roles_mixed_rate():
    metrics = audit_scene_roles(
        agent_offsets=np.asarray([0, 2, 4]),
        role_ids=np.asarray([0, 1, 0, 1]),
    )
    assert metrics["assignment/mixed_scene_rate"] == pytest.approx(1.0)
    assert metrics["assignment/partner_only_scene_rate"] == pytest.approx(0.0)


def _drive_init_stub(self, **kwargs):
    self.num_agents = 4
    self.policy_log_ids = [0, 1, 0, 1]
    self.policy_log_count = 2
    self.agent_offsets = np.asarray([0, 2, 4], dtype=np.int32)
    self.tick = 0
    self.resample_frequency = 0
    self.report_interval = 1000
    self.rewards = np.asarray([0.1, -0.1, 0.3, -0.3], dtype=np.float32)
    # Zero lane components so large |steer| alone moves partner rewards.
    self.reward_components_raw = np.zeros((4, 8), dtype=np.float32)
    self.observations = np.zeros((4, 10), dtype=np.float32)
    self.terminals = np.zeros(4, dtype=np.float32)
    self.truncations = np.zeros(4, dtype=np.float32)


def _make_env(partner_mode, **kwargs):
    with patch(
        "pufferlib.conservative.env.Drive.__init__",
        _drive_init_stub,
    ):
        env = ConservativeMixDrive(partner_mode=partner_mode, **kwargs)
    return env


def _fake_drive_step(self, actions):
    return (
        self.observations,
        self.rewards,
        self.terminals,
        self.truncations,
        [],
    )


# Partner large |steer| (index 0 → -1.0); ego zero steer (6).
_PARTNER_STEER_ACTIONS = np.asarray([6, 0, 6, 0], dtype=np.int32)


def test_init_pops_conservative_kwargs_sets_attrs_and_registers():
    captured = {}

    def _capturing_stub(self, **kwargs):
        captured.update(kwargs)
        _drive_init_stub(self, **kwargs)

    with patch(
        "pufferlib.conservative.env.Drive.__init__",
        _capturing_stub,
    ):
        env = ConservativeMixDrive(
            partner_mode="action_constraint",
            partner_max_abs_steer=0.333,
        )
    assert not any(key.startswith("partner_") for key in captured)
    assert env.partner_mode == "action_constraint"
    assert env.partner_max_abs_steer == 0.333
    assert env.conservative_config.partner_mode == "action_constraint"
    import pufferlib.ocean.torch as drive_torch

    from pufferlib.conservative.policy import DriveSteerConstrained

    assert getattr(drive_torch, "DriveSteerConstrained") is DriveSteerConstrained


def test_init_rejects_too_few_policies():
    def _bad_count(self, **kwargs):
        self.num_agents = 4
        self.policy_log_ids = [0, 0, 0, 0]
        self.policy_log_count = 1
        self.agent_offsets = np.asarray([0, 2, 4], dtype=np.int32)

    with patch(
        "pufferlib.conservative.env.Drive.__init__",
        _bad_count,
    ):
        with pytest.raises(ValueError, match="at least 2"):
            ConservativeMixDrive(partner_mode="action_constraint")


def test_init_accepts_multiple_partner_policies_and_maps_roles():
    def _multi_policy(self, **kwargs):
        self.num_agents = 7
        self.policy_log_ids = [0, 0, 0, 0, 1, 1, 2]
        self.policy_log_count = 3
        self.agent_offsets = np.asarray([0, 7], dtype=np.int32)
        self.tick = 0
        self.resample_frequency = 0
        self.report_interval = 1000
        self.rewards = np.zeros(7, dtype=np.float32)
        self.reward_components_raw = np.zeros((7, 8), dtype=np.float32)
        self.observations = np.zeros((7, 10), dtype=np.float32)
        self.terminals = np.zeros(7, dtype=np.float32)
        self.truncations = np.zeros(7, dtype=np.float32)

    with patch("pufferlib.conservative.env.Drive.__init__", _multi_policy):
        env = ConservativeMixDrive(partner_mode="off")

    assert list(env._role_ids) == [0, 0, 0, 0, 1, 1, 1]
    assert env._assignment_metrics["assignment/mixed_scene_rate"] == 1.0


def test_init_rejects_unknown_partner_mode():
    with pytest.raises(ValueError, match="partner_mode"):
        ConservativeMixDrive(partner_mode="idm")


def test_step_shaping_disabled_preserves_drive_rewards():
    """action_constraint / off: step may override Drive.step but must not alter rewards."""
    for mode in ("action_constraint", "off"):
        env = _make_env(mode)
        base = env.rewards.copy()
        with patch.object(Drive, "step", _fake_drive_step):
            env.step(_PARTNER_STEER_ACTIONS)
        np.testing.assert_array_equal(env.rewards, base)


@pytest.mark.parametrize("partner_mode", ["reward_shaping", "both"])
def test_step_reward_shaping_changes_partner_preserves_ego(partner_mode):
    env = _make_env(partner_mode)
    base = env.rewards.copy()
    with patch.object(Drive, "step", _fake_drive_step):
        env.step(_PARTNER_STEER_ACTIONS)
    # Ego (policy 0) identical to Drive base rewards.
    assert env.rewards[0] == base[0]
    assert env.rewards[2] == base[2]
    assert env.rewards[0].tobytes() == base[0].tobytes()
    assert env.rewards[2].tobytes() == base[2].tobytes()
    # Partner (policy 1) rewritten by shaping.
    assert env.rewards[1] != base[1]
    assert env.rewards[3] != base[3]


def test_step_action_constraint_skips_shaping_evaluator():
    env = _make_env("action_constraint")
    assert not env.conservative_config.use_reward_shaping
    assert getattr(env, "_partner_shaping_evaluator", None) is None
    base = env.rewards.copy()
    mock_eval = MagicMock()
    env._partner_shaping_evaluator = mock_eval
    with patch.object(Drive, "step", _fake_drive_step):
        env.step(_PARTNER_STEER_ACTIONS)
    mock_eval.evaluate.assert_not_called()
    np.testing.assert_array_equal(env.rewards, base)


def test_compute_lead_gap_same_scene_forward_vehicle():
    # Two agents heading +x; agent 1 is 10m ahead of agent 0 (length 4 → bumper 6).
    x = np.asarray([0.0, 10.0, 100.0], dtype=np.float32)
    y = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    heading = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    length = np.asarray([4.0, 4.0, 4.0], dtype=np.float32)
    # Separate scenes: [0,1] and [2]
    offsets = np.asarray([0, 2, 3], dtype=np.int32)
    prev_x = np.asarray([-1.0, 9.0, 99.0], dtype=np.float32)
    prev_y = np.zeros(3, dtype=np.float32)
    gap, speed = compute_lead_gap_and_speed(
        x, y, heading, length, offsets, prev_x=prev_x, prev_y=prev_y, dt=0.1
    )
    assert gap[0] == pytest.approx(6.0)
    assert np.isinf(gap[1])  # no one ahead in scene
    assert np.isinf(gap[2])  # singleton scene
    assert speed[0] == pytest.approx(10.0)  # 1m / 0.1s


def test_conservative_mix_registered_lazily():
    assert "drive_conservative_mix" in MAKE_FUNCTIONS


@patch("sys.argv", ["pufferl.py"])
def test_load_conservative_mix_ini():
    args = load_config("puffer_drive_conservative_mix")
    assert args["env_name"] == "puffer_drive_conservative_mix"
    assert args["train"]["mix_ppo"] in (True, "True", "true", 1, "1")
    assert "DriveSteerConstrained" in args["train"]["mix_ppo_policy_names"]
    assert float(args["env"].get("idm_fraction", 0)) == 0.0
