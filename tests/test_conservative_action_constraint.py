import math

import numpy as np
import pytest
import torch

from pufferlib.conservative.action_constraint import (
    NUM_ACTIONS,
    NUM_STEER,
    STEERING_VALUES,
    legal_action_mask,
    legal_steer_indices,
    mask_discrete_logits,
)


def test_steering_table_matches_drive_h_center():
    assert NUM_STEER == 13
    assert NUM_ACTIONS == 91
    assert STEERING_VALUES[6] == pytest.approx(0.0)
    assert abs(STEERING_VALUES[5]) == pytest.approx(0.167, abs=1e-3)


def test_legal_steer_indices_default_333():
    idx = legal_steer_indices(0.333)
    assert idx == [4, 5, 6, 7, 8]


def test_legal_steer_indices_tight_167():
    idx = legal_steer_indices(0.167)
    assert idx == [5, 6, 7]


def test_legal_action_mask_blocks_large_steer_keeps_all_accel():
    mask = legal_action_mask(0.333)
    assert mask.shape == (91,)
    assert mask.dtype == bool
    # accel=3 (zero), steer=6 (center) -> action 3*13+6 = 45
    assert mask[45]
    # accel=3, steer=0 (|steer|=1.0) -> 3*13+0 = 39
    assert not mask[39]
    # every accel row has exactly 5 legal steers
    for a in range(7):
        assert int(mask[a * 13 : (a + 1) * 13].sum()) == 5


def test_mask_discrete_logits_tuple_and_tensor():
    logits = torch.zeros(2, 91)
    mask = legal_action_mask(0.333)
    out = mask_discrete_logits((logits,), mask)
    assert isinstance(out, tuple)
    assert torch.isneginf(out[0][0, 39])
    assert out[0][0, 45] == 0.0
    # already-sampled path must remain differentiable-safe: no NaN after softmax
    probs = torch.softmax(out[0], dim=-1)
    assert torch.isfinite(probs).all()
    assert probs[0, 39] == pytest.approx(0.0)
