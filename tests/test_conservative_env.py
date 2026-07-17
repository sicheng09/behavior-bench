from unittest.mock import patch

import numpy as np
import pytest

from pufferlib.ocean.environment import MAKE_FUNCTIONS
from pufferlib.pufferl import load_config

from pufferlib.conservative.env import (
    ConservativeMixDrive,
    audit_scene_roles,
    normalize_role_ids,
)
from pufferlib.ocean.drive.drive import Drive


def test_normalize_role_ids_tiles_and_rejects_unknown():
    ids = normalize_role_ids([0, 1], 5)
    assert list(ids) == [0, 1, 0, 1, 0]
    with pytest.raises(ValueError):
        normalize_role_ids([0, 2], 4)


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


def test_init_rejects_wrong_policy_count():
    def _bad_count(self, **kwargs):
        self.num_agents = 4
        self.policy_log_ids = [0, 0, 0, 0]
        self.policy_log_count = 1
        self.agent_offsets = np.asarray([0, 2, 4], dtype=np.int32)

    with patch(
        "pufferlib.conservative.env.Drive.__init__",
        _bad_count,
    ):
        with pytest.raises(ValueError, match="exactly 2"):
            ConservativeMixDrive(partner_mode="action_constraint")


def test_init_rejects_unknown_partner_mode():
    with pytest.raises(ValueError, match="partner_mode"):
        ConservativeMixDrive(partner_mode="idm")


def test_step_inherits_drive_without_override():
    assert ConservativeMixDrive.step is Drive.step


def test_conservative_mix_registered_lazily():
    assert "drive_conservative_mix" in MAKE_FUNCTIONS


@patch("sys.argv", ["pufferl.py"])
def test_load_conservative_mix_ini():
    args = load_config("puffer_drive_conservative_mix")
    assert args["env_name"] == "puffer_drive_conservative_mix"
    assert args["train"]["mix_ppo"] in (True, "True", "true", 1, "1")
    assert "DriveSteerConstrained" in args["train"]["mix_ppo_policy_names"]
    assert float(args["env"].get("idm_fraction", 0)) == 0.0
