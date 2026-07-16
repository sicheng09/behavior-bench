import os
from unittest.mock import patch

import numpy as np
import torch

from pufferlib.adversarial.env import AdversarialMixDrive
from pufferlib.adversarial.reward import RewardEvaluation, StateFrame
from pufferlib.pufferl import (
    PuffeRL,
    load_config,
    load_env,
    load_mixed_policies,
    load_policy,
)


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
    zeros = np.zeros(4, dtype=np.float32)
    ones = np.ones(4, dtype=np.float32)
    valid = np.ones(4, dtype=bool)
    env._pre_state = StateFrame(
        x=zeros.copy(),
        y=zeros.copy(),
        heading=zeros.copy(),
        length=ones.copy(),
        width=ones.copy(),
        valid=valid.copy(),
    )
    env._state_x = zeros.copy()
    env._state_y = zeros.copy()
    env._state_heading = zeros.copy()
    env._state_length = ones.copy()
    env._state_width = ones.copy()
    env._state_valid = valid.copy()
    env._pre_x = zeros.copy()
    env._pre_y = zeros.copy()
    env._pre_heading = zeros.copy()
    env._pre_length = ones.copy()
    env._pre_width = ones.copy()
    env._pre_valid = valid.copy()
    env._capture_post_state_frame = lambda: StateFrame(
        x=env._state_x,
        y=env._state_y,
        heading=env._state_heading,
        length=env._state_length,
        width=env._state_width,
        valid=env._state_valid,
    )
    env.rewards = np.asarray([0.1, -0.1, 0.3, -0.3], dtype=np.float32)
    env._pending_adversarial_metrics = {}
    env._invalid_reward_events = 0

    class _Limits:
        max_invalid_reward_events = 10

    class _Reward:
        limits = _Limits()

    class _Cfg:
        reward = _Reward()

    env.adversarial_config = _Cfg()

    env._apply_adversarial_rewards(
        np.zeros((4, 2), dtype=np.float32), collect_metrics=True
    )

    assert env.rewards.tolist() == [
        np.float32(0.1),
        np.float32(0.2),
        np.float32(0.3),
        np.float32(0.2),
    ]


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
            {
                name: value.detach().clone()
                for name, value in policy.state_dict().items()
            }
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

