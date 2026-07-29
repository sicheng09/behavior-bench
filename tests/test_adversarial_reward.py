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


def reward_config(goal=0.0):
    return RewardConfig(
        weights=RewardWeights(0.20, 1.0, 0.20, 0.02, goal),
        limits=RewardLimits(0.25, 0.50, 10),
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


def evaluate(
    pre,
    post,
    base,
    raw,
    actions=None,
    pre_speed=None,
    post_speed=None,
    config=None,
):
    evaluator = AsymmetricRewardEvaluator(config or reward_config(), dt=0.1)
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
        if actions is None
        else np.asarray(actions, dtype=np.float32),
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
    result = evaluate(
        pre,
        post,
        [0, 0],
        np.zeros((2, 11)),
        pre_speed=[10, 5],
        post_speed=[5, 5],
    )
    assert result.components["ego_cost"][1] == 0


def test_closing_ttc_risk_rewards_opponent_below_positive_cap():
    pre = frame([0, 8], [0, 0], [0, 0])
    post = frame([1.0, 8.2], [0, 0], [0, 0])
    result = evaluate(
        pre,
        post,
        [0, 0],
        np.zeros((2, 11)),
        pre_speed=[10, 2],
        post_speed=[10, 2],
    )
    assert 0 < result.rewards[1] <= 0.25


def test_distant_front_opponent_does_not_receive_ego_brake_credit():
    pre = frame([0, 40], [0, 0], [0, 0])
    post = frame([0.5, 40.5], [0, 0], [0, 0])
    result = evaluate(
        pre,
        post,
        [0, 0],
        np.zeros((2, 11)),
        pre_speed=[10, 5],
        post_speed=[5, 5],
    )
    assert result.components["ego_cost"][1] == 0


def test_opponent_collision_is_conservatively_faulted():
    pre = frame([0, 4], [0, 2], [0, -1.57])
    post = frame([0.5, 4], [0, 1.5], [0, -1.57])
    raw = np.zeros((2, 11), dtype=np.float32)
    raw[1, 0] = 1.0
    result = evaluate(
        pre,
        post,
        [0, 0],
        raw,
        pre_speed=[5, 5],
        post_speed=[5, 5],
    )
    assert result.components["fault_penalty"][1] == 1.0
    assert result.rewards[1] == -1.0


def test_high_confidence_ego_rear_end_is_not_opponent_fault():
    pre = frame([0, 6], [0, 0], [0, 0])
    post = frame([1.2, 6.5], [0, 0], [0, 0])
    raw = np.zeros((2, 11), dtype=np.float32)
    raw[1, 0] = 1.0
    result = evaluate(
        pre,
        post,
        [0, 0],
        raw,
        pre_speed=[12, 5],
        post_speed=[12, 5],
    )
    assert result.components["fault_penalty"][1] == 0.0


def test_opponent_hard_brake_keeps_rear_end_fault():
    pre = frame([0, 6], [0, 0], [0, 0])
    post = frame([1.2, 6.5], [0, 0], [0, 0])
    raw = np.zeros((2, 11), dtype=np.float32)
    raw[1, 0] = 1.0
    result = evaluate(
        pre,
        post,
        [0, 0],
        raw,
        pre_speed=[12, 8],
        post_speed=[12, 4],
    )
    assert result.components["fault_penalty"][1] == 1.0


def test_kinematic_violation_is_negative_and_finite():
    pre = frame([0, 8], [0, 0], [0, 0])
    post = frame([0, 10], [0, 0], [0, 0])
    result = evaluate(
        pre,
        post,
        [0, 0],
        np.zeros((2, 11)),
        pre_speed=[0, 0],
        post_speed=[0, 20],
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


def test_weak_goal_event_adds_capped_positive_bonus():
    # Stationary distant opponent: no ego_cost / kinematics; only goal fires.
    pre = frame([0, 40], [0, 0], [0, 0])
    post = frame([0, 40], [0, 0], [0, 0])
    raw = np.zeros((2, 11), dtype=np.float32)
    raw[1, 2] = 1.0  # GOAL_COMPONENT
    result = evaluate(
        pre,
        post,
        [0, 0],
        raw,
        pre_speed=[0, 0],
        post_speed=[0, 0],
        config=reward_config(goal=0.15),
    )
    assert result.components["goal_event"][1] == 1.0
    assert abs(float(result.rewards[1]) - 0.15) < 1e-5
