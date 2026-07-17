import numpy as np
import pytest

from pufferlib.conservative.action_constraint import NUM_STEER, STEERING_VALUES
from pufferlib.conservative.config import ConservativePartnerConfig
from pufferlib.conservative.reward import (
    RC_L_ALIGN,
    RC_L_CENTER,
    PartnerShapingEvaluator,
)


def _cfg(**kwargs):
    return ConservativePartnerConfig(partner_mode="reward_shaping", **kwargs)


def _components(n, center=0.0, align=0.0):
    # Enough columns for RC_L_CENTER=7
    raw = np.zeros((n, 8), dtype=np.float32)
    raw[:, RC_L_CENTER] = center
    raw[:, RC_L_ALIGN] = align
    return raw


def _eval(
    *,
    base,
    roles,
    actions,
    components=None,
    lead_gap_m=None,
    speed_mps=None,
    config=None,
):
    evaluator = PartnerShapingEvaluator(config or _cfg())
    return evaluator.evaluate(
        base_rewards=np.asarray(base, dtype=np.float32),
        role_ids=np.asarray(roles, dtype=np.int64),
        actions=np.asarray(actions),
        reward_components_raw=components,
        lead_gap_m=None
        if lead_gap_m is None
        else np.asarray(lead_gap_m, dtype=np.float32),
        speed_mps=None
        if speed_mps is None
        else np.asarray(speed_mps, dtype=np.float32),
    )


def test_ego_rewards_unchanged_byte_identical():
    base = np.asarray([0.1234567, 0.4, -0.25], dtype=np.float32)
    # ego, partner, ego
    result = _eval(
        base=base,
        roles=[0, 1, 0],
        actions=[6, 0, 6],  # partner large |steer|
        components=_components(3, center=0.5, align=0.5),
        lead_gap_m=[np.inf, 10.0, np.inf],
        speed_mps=[10.0, 10.0, 10.0],
    )
    assert result.rewards[0] == base[0]
    assert result.rewards[2] == base[2]
    assert result.rewards[0].tobytes() == base[0].tobytes()
    assert result.rewards[2].tobytes() == base[2].tobytes()


def test_larger_abs_steer_lowers_partner_reward():
    base = np.asarray([0.0, 0.0], dtype=np.float32)
    roles = [1, 1]
    # steer index 6 → 0.0; steer index 0 → -1.0
    zero_steer = 6  # STEERING_VALUES[6] == 0
    large_steer = 0  # STEERING_VALUES[0] == -1.0
    assert STEERING_VALUES[zero_steer] == pytest.approx(0.0)
    assert abs(STEERING_VALUES[large_steer]) == pytest.approx(1.0)

    components = _components(2)
    lead = [np.inf, np.inf]
    speed = [10.0, 10.0]

    r_zero = _eval(
        base=base,
        roles=roles,
        actions=[zero_steer, zero_steer],
        components=components,
        lead_gap_m=lead,
        speed_mps=speed,
    )
    r_large = _eval(
        base=base,
        roles=roles,
        actions=[large_steer, large_steer],
        components=components,
        lead_gap_m=lead,
        speed_mps=speed,
    )
    assert r_large.rewards[0] < r_zero.rewards[0]
    assert r_large.rewards[1] < r_zero.rewards[1]


def test_headway_below_target_lowers_partner_reward():
    T = 1.8
    base = np.asarray([0.0, 0.0], dtype=np.float32)
    # Same speed; short gap → headway << T; long gap → headway >> T
    speed = [10.0, 10.0]
    short_gap = 1.0  # headway = 0.1s
    long_gap = 50.0  # headway = 5.0s
    assert short_gap / 10.0 < T
    assert long_gap / 10.0 > T

    r_short = _eval(
        base=base,
        roles=[1, 1],
        actions=[6, 6],
        components=_components(2),
        lead_gap_m=[short_gap, short_gap],
        speed_mps=speed,
        config=_cfg(partner_target_headway=T),
    )
    r_long = _eval(
        base=base,
        roles=[1, 1],
        actions=[6, 6],
        components=_components(2),
        lead_gap_m=[long_gap, long_gap],
        speed_mps=speed,
        config=_cfg(partner_target_headway=T),
    )
    assert r_short.rewards[0] < r_long.rewards[0]


def test_outputs_finite_and_clipped():
    base = np.asarray([0.9, 0.9, 0.9], dtype=np.float32)
    # Push shaping strongly positive/negative via huge components + steer
    components = _components(3, center=10.0, align=10.0)
    result = _eval(
        base=base,
        roles=[0, 1, 1],
        actions=[0, 0, NUM_STEER + 0],  # large |steer| via % NUM_STEER
        components=components,
        lead_gap_m=[np.inf, 0.01, 1000.0],
        speed_mps=[10.0, 10.0, 0.05],
        config=_cfg(w_center=1.0, w_align=1.0, w_steer=1.0, w_gap=1.0),
    )
    assert np.isfinite(result.rewards).all()
    assert (result.rewards >= -1.0).all()
    assert (result.rewards <= 1.0).all()
    for key in (
        "partner/steer_abs_mean",
        "partner/gap_shaping_mean",
        "partner/reward_mean",
    ):
        assert key in result.metrics
        assert np.isfinite(result.metrics[key])


def test_no_partners_returns_base_copy():
    base = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
    result = _eval(
        base=base,
        roles=[0, 0, 0],
        actions=[6, 6, 6],
        components=_components(3),
        lead_gap_m=[1.0, 1.0, 1.0],
        speed_mps=[10.0, 10.0, 10.0],
    )
    np.testing.assert_array_equal(result.rewards, base)
    assert result.rewards is not base
    assert result.rewards.tobytes() == base.tobytes()
