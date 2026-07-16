import unittest
from types import SimpleNamespace

import gymnasium
import numpy as np
import torch

import pufferlib
from pufferlib.ocean import torch as drive_torch


def make_dummy_env(action_type="discrete"):
    ego_dim = 7
    max_partner_objects = 3
    partner_features = 8
    max_road_objects = 4
    road_features = 7
    obs_dim = ego_dim + max_partner_objects * partner_features + max_road_objects * road_features
    if action_type == "discrete":
        action_space = gymnasium.spaces.MultiDiscrete([7 * 13])
    else:
        action_space = pufferlib.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    return SimpleNamespace(
        single_observation_space=gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        ),
        single_action_space=action_space,
        max_partner_objects=max_partner_objects,
        partner_features=partner_features,
        max_road_objects=max_road_objects,
        road_features=road_features,
        dynamics_model="classic",
    )


def make_obs(batch, env, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(batch, env.single_observation_space.shape[0])).astype(np.float32)
    ego_dim = 7
    partner_dim = env.max_partner_objects * env.partner_features
    road_start = ego_dim + partner_dim
    road = obs[:, road_start:].reshape(batch, env.max_road_objects, env.road_features)
    road[:, :, -1] = rng.integers(0, 7, size=(batch, env.max_road_objects))
    obs[:, road_start:] = road.reshape(batch, -1)
    return torch.from_numpy(obs)


class TestDriveBehaviorAware(unittest.TestCase):
    def test_forward_eval_returns_discrete_logits_and_value(self):
        env = make_dummy_env()
        policy = drive_torch.DriveBehaviorAware(
            env,
            input_size=16,
            hidden_size=32,
            behavior_latent_dim=8,
        )
        wrapper = drive_torch.BehaviorAwareRecurrent(
            env,
            policy,
            input_size=32,
            hidden_size=32,
        )
        obs = make_obs(5, env)
        state = {
            "lstm_h": torch.zeros(5, 32),
            "lstm_c": torch.zeros(5, 32),
        }

        logits, value = wrapper.forward_eval(obs, state)

        self.assertIsInstance(logits, tuple)
        self.assertEqual(logits[0].shape, (5, 91))
        self.assertEqual(value.shape, (5, 1))
        self.assertEqual(state["lstm_h"].shape, (5, 32))
        self.assertEqual(state["lstm_c"].shape, (5, 32))

    def test_training_forward_records_prediction_loss(self):
        env = make_dummy_env()
        policy = drive_torch.DriveBehaviorAware(
            env,
            input_size=16,
            hidden_size=32,
            behavior_latent_dim=8,
            prediction_loss_coef=0.1,
        )
        wrapper = drive_torch.BehaviorAwareRecurrent(
            env,
            policy,
            input_size=32,
            hidden_size=32,
        )
        obs = make_obs(4 * 6, env).reshape(4, 6, -1)
        actions = torch.zeros(4, 5, 1, dtype=torch.long)
        state = {"action": actions, "lstm_h": None, "lstm_c": None}
        trunc_or_term_before = torch.zeros(4, 5)

        logits, values = wrapper(obs[:, :-1], state, trunc_or_term_before)
        aux_loss, aux_logs = wrapper.compute_auxiliary_loss()

        self.assertEqual(logits[0].shape, (20, 91))
        self.assertEqual(values.shape, (4, 5))
        self.assertGreaterEqual(float(aux_loss), 0.0)
        self.assertIn("behavior_prediction_loss", aux_logs)
        self.assertIn("behavior_prediction_valid_fraction", aux_logs)

    def test_prediction_loss_ignores_zero_partner_slots(self):
        env = make_dummy_env()
        policy = drive_torch.DriveBehaviorAware(
            env,
            input_size=16,
            hidden_size=32,
            behavior_latent_dim=8,
            prediction_loss_coef=0.1,
        )
        wrapper = drive_torch.BehaviorAwareRecurrent(
            env,
            policy,
            input_size=32,
            hidden_size=32,
        )
        obs = make_obs(2 * 4, env).reshape(2, 4, -1)
        ego_dim = 7
        partner_dim = env.max_partner_objects * env.partner_features
        obs[:, :, ego_dim : ego_dim + partner_dim] = 0.0
        actions = torch.zeros(2, 3, 1, dtype=torch.long)
        state = {"action": actions, "lstm_h": None, "lstm_c": None}
        trunc_or_term_before = torch.zeros(2, 3)

        wrapper(obs[:, :-1], state, trunc_or_term_before)
        aux_loss, aux_logs = wrapper.compute_auxiliary_loss()

        self.assertEqual(float(aux_loss), 0.0)
        self.assertEqual(aux_logs["behavior_prediction_valid_fraction"], 0.0)

    def test_prediction_loss_ignores_cross_episode_transitions(self):
        env = make_dummy_env()
        policy = drive_torch.DriveBehaviorAware(
            env,
            input_size=16,
            hidden_size=32,
            behavior_latent_dim=8,
            prediction_loss_coef=0.1,
        )
        wrapper = drive_torch.BehaviorAwareRecurrent(
            env,
            policy,
            input_size=32,
            hidden_size=32,
        )
        obs = make_obs(2, env).reshape(1, 2, -1)
        ego_dim = 7
        partner_dim = env.max_partner_objects * env.partner_features
        # Force both sides to look valid and very different. The transition
        # should still be ignored because t+1 starts a new episode.
        obs[:, :, ego_dim : ego_dim + partner_dim] = 0.0
        obs[0, 0, ego_dim : ego_dim + env.partner_features] = torch.tensor(
            [1.0, 1.0, 0.5, 0.7, 0.1, 0.2, 0.3, 1.0]
        )
        obs[0, 1, ego_dim : ego_dim + env.partner_features] = torch.tensor(
            [-5.0, 4.0, 0.5, 0.7, -0.4, 0.9, -0.2, 1.0]
        )
        actions = torch.zeros(1, 2, 1, dtype=torch.long)
        state = {"action": actions, "lstm_h": None, "lstm_c": None}
        trunc_or_term_before = torch.tensor([[0.0, 1.0]])

        wrapper(obs, state, trunc_or_term_before)
        aux_loss, aux_logs = wrapper.compute_auxiliary_loss()

        self.assertEqual(float(aux_loss), 0.0)
        self.assertEqual(aux_logs["behavior_prediction_valid_fraction"], 0.0)

    def test_prediction_targets_use_dynamic_partner_fields(self):
        env = make_dummy_env()
        policy = drive_torch.DriveBehaviorAware(
            env,
            input_size=16,
            hidden_size=32,
            behavior_latent_dim=8,
        )
        current = torch.zeros(1, 1, env.partner_features)
        next_partner = torch.zeros(1, 1, env.partner_features)
        current[0, 0] = torch.tensor([1.0, 2.0, 0.5, 0.7, 0.1, 0.2, 0.3, 1.0])
        next_partner[0, 0] = torch.tensor([1.5, 1.0, 0.5, 0.7, 0.4, -0.1, 0.9, 1.0])

        target = policy.partner_prediction_targets(current, next_partner)

        expected = torch.tensor([[[0.5, -1.0, 0.3, -0.3, 0.6]]])
        self.assertEqual(target.shape, (1, 1, 5))
        torch.testing.assert_close(target, expected)

    def test_no_fuse_mode_keeps_actor_input_on_lstm_hidden_only(self):
        env = make_dummy_env()
        fused = drive_torch.DriveBehaviorAware(
            env,
            input_size=16,
            hidden_size=32,
            behavior_latent_dim=8,
            fuse_behavior_latent=True,
        )
        no_fuse = drive_torch.DriveBehaviorAware(
            env,
            input_size=16,
            hidden_size=32,
            behavior_latent_dim=8,
            fuse_behavior_latent=False,
        )

        self.assertEqual(fused.policy_feature_dim, 40)
        self.assertEqual(no_fuse.policy_feature_dim, 32)


if __name__ == "__main__":
    unittest.main()
