import numpy as np
import pytest

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
    with pytest.raises(ValueError, match="agent_offsets"):
        audit_scene_roles(
            np.asarray([0, 3, 2], dtype=np.int32),
            np.asarray([0, 1, 0], dtype=np.int64),
        )
