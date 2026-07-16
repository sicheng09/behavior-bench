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
        x=np.zeros(4),
        y=np.zeros(4),
        heading=np.zeros(4),
        length=np.ones(4),
        width=np.ones(4),
        valid=np.ones(4, dtype=bool),
    )
    env.rewards = np.asarray([0.1, -0.1, 0.3, -0.3], dtype=np.float32)
    env.get_global_agent_state = lambda: {
        "x": np.zeros(4),
        "y": np.zeros(4),
        "heading": np.zeros(4),
        "length": np.ones(4),
        "width": np.ones(4),
    }
    env._pending_adversarial_metrics = {}
    env._invalid_reward_events = 0

    class _Limits:
        max_invalid_reward_events = 10

    class _Reward:
        limits = _Limits()

    class _Cfg:
        reward = _Reward()

    env.adversarial_config = _Cfg()

    env._apply_adversarial_rewards(np.zeros((4, 2), dtype=np.float32))

    assert env.rewards.tolist() == [
        np.float32(0.1),
        np.float32(0.2),
        np.float32(0.3),
        np.float32(0.2),
    ]
