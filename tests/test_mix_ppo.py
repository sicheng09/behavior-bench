import unittest
import os
import io
import sys
import tempfile
from contextlib import redirect_stdout
from types import SimpleNamespace

import gymnasium
import numpy as np
import torch

import pufferlib.models
from pufferlib.policy_mix import (
    assign_policy_ids,
    group_policy_losses,
    flatten_policy_stats,
    parse_policy_mix,
)
from pufferlib.pufferl import Profile, PuffeRL, load_config, load_env, load_mixed_policies, load_policy


class TestMixPPO(unittest.TestCase):
    def test_parse_policy_mix_accepts_named_fractions(self):
        names, fractions = parse_policy_mix("learner:0.5, transformer:0.25, gameformer:0.25")

        self.assertEqual(names, ["learner", "transformer", "gameformer"])
        self.assertEqual(fractions, [0.5, 0.25, 0.25])

    def test_assign_policy_ids_interleaves_by_fraction_deficit(self):
        policy_ids = assign_policy_ids(
            total_agents=8,
            fractions=[0.5, 0.25, 0.25],
            device="cpu",
        )

        self.assertEqual(policy_ids.dtype, torch.long)
        self.assertEqual(torch.bincount(policy_ids, minlength=3).tolist(), [4, 2, 2])
        self.assertEqual(policy_ids.tolist(), [0, 1, 2, 0, 0, 1, 2, 0])

    def test_group_policy_losses_removes_redundant_policy_prefixes(self):
        losses = {
            "policy_0/policy_loss": 0.1,
            "policy_0/value_loss": 0.2,
            "policy_1/policy_loss": 0.3,
        }

        grouped = group_policy_losses(losses)

        self.assertEqual(grouped, {
            0: {"policy_loss": 0.1, "value_loss": 0.2},
            1: {"policy_loss": 0.3},
        })

    def test_flatten_policy_stats_keeps_full_logging_keys(self):
        stats = {
            0: {"score": 0.9, "collision_rate": 0.1},
            1: {"score": 0.8, "collision_rate": 0.2},
        }

        flat = flatten_policy_stats(stats)

        self.assertEqual(flat["mix_ppo/policy_0/score"], 0.9)
        self.assertEqual(flat["mix_ppo/policy_1/collision_rate"], 0.2)

    def test_mix_ppo_dashboard_keeps_policy_columns_without_rich_color_codes(self):
        trainer = SimpleNamespace(
            config={"device": "cpu", "env": "puffer_drive", "total_timesteps": 1024},
            sps=128,
            global_step=256,
            profile=Profile(),
            mix_ppo=True,
            policies=[object(), object(), object()],
            losses={
                "policy_0/policy_loss": 0.1,
                "policy_1/policy_loss": 0.2,
                "policy_2/policy_loss": 0.3,
            },
            stats={
                "n": 40.0,
                "score": 0.5,
                "mix_ppo/policy_0/n": 20.0,
                "mix_ppo/policy_0/score": 0.6,
                "mix_ppo/policy_1/n": 10.0,
                "mix_ppo/policy_1/score": 0.7,
                "mix_ppo/policy_2/n": 10.0,
                "mix_ppo/policy_2/score": 0.8,
            },
            last_stats={},
            utilization=SimpleNamespace(
                cpu_util=[0.0],
                gpu_util=[0.0],
                cpu_mem=[0.0],
                gpu_mem=[0.0],
            ),
            model_size=1_900_000,
            epoch=1,
            uptime=3,
        )

        output = io.StringIO()
        with redirect_stdout(output):
            PuffeRL.print_dashboard(trainer, idx=[0])

        rendered = output.getvalue()
        self.assertIn("p0.Losses", rendered)
        self.assertIn("p1.Losses", rendered)
        self.assertIn("p2.Losses", rendered)
        self.assertIn("p0 Stats", rendered)
        self.assertIn("p1 Stats", rendered)
        self.assertIn("p2 Stats", rendered)
        self.assertIn("40.000", rendered)
        self.assertIn("20.000", rendered)
        self.assertIn("10.000", rendered)
        self.assertNotIn("\x1b[96m", rendered)
        self.assertNotIn("\x1b[1;36m", rendered)
        self.assertNotIn("\x1b[0m", rendered)

    def test_load_mixed_policies_allows_per_policy_rnn_names(self):
        driver_env = SimpleNamespace(
            single_observation_space=gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(7 + 3 * 8 + 4 * 7,),
                dtype=np.float32,
            ),
            single_action_space=gymnasium.spaces.MultiDiscrete([91]),
            max_partner_objects=3,
            partner_features=8,
            max_road_objects=4,
            road_features=7,
            dynamics_model="classic",
        )
        vecenv = SimpleNamespace(driver_env=driver_env)
        args = {
            "package": "ocean",
            "policy_name": "Drive",
            "rnn_name": "Recurrent",
            "policy": {"input_size": 16, "hidden_size": 32},
            "rnn": {"input_size": 32, "hidden_size": 32},
            "train": {
                "device": "cpu",
                "mix_ppo": True,
                "mix_ppo_policy_mix": "l3:1,l2:1,l1:1",
                "mix_ppo_policy_names": "Drive,Drive,Drive",
                "mix_ppo_rnn_names": "Recurrent,None,None",
                "mix_ppo_policy_paths": ",,",
            },
        }

        policies = load_mixed_policies(args, vecenv, "puffer_drive")

        self.assertEqual(len(policies), 3)
        self.assertIsInstance(policies[0], pufferlib.models.LSTMWrapper)
        self.assertNotIsInstance(policies[1], pufferlib.models.LSTMWrapper)
        self.assertNotIsInstance(policies[2], pufferlib.models.LSTMWrapper)

    def test_drive_lite_forward_eval_matches_drive_action_interface(self):
        from pufferlib.ocean.torch import Drive, DriveLite

        driver_env = SimpleNamespace(
            single_observation_space=gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(7 + 3 * 8 + 4 * 7,),
                dtype=np.float32,
            ),
            single_action_space=gymnasium.spaces.MultiDiscrete([91]),
            max_partner_objects=3,
            partner_features=8,
            max_road_objects=4,
            road_features=7,
            dynamics_model="classic",
        )
        policy = DriveLite(driver_env, input_size=16, hidden_size=32)
        obs = torch.zeros(5, driver_env.single_observation_space.shape[0])

        logits, value = policy.forward_eval(obs)

        self.assertIsInstance(logits, tuple)
        self.assertEqual(logits[0].shape, (5, 91))
        self.assertEqual(value.shape, (5, 1))
        self.assertEqual(policy.hidden_size, 28)

        drive_params = sum(p.numel() for p in Drive(driver_env, input_size=64, hidden_size=256).parameters())
        lite_params = sum(p.numel() for p in DriveLite(driver_env, input_size=64, hidden_size=256).parameters())
        self.assertLessEqual(lite_params, int(drive_params * 0.55))

    def test_drive_perception_low_masks_objects_outside_front_sector(self):
        from pufferlib.ocean.torch import DrivePerceptionLow

        driver_env = SimpleNamespace(
            single_observation_space=gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(7 + 3 * 8 + 4 * 7,),
                dtype=np.float32,
            ),
            single_action_space=gymnasium.spaces.MultiDiscrete([91]),
            max_partner_objects=3,
            partner_features=8,
            max_road_objects=4,
            road_features=7,
            dynamics_model="classic",
        )
        policy = DrivePerceptionLow(driver_env, input_size=16, hidden_size=32)
        obs = torch.zeros(1, driver_env.single_observation_space.shape[0])
        ego_dim = 7
        partner_start = ego_dim
        road_start = ego_dim + 3 * 8

        partners = torch.zeros(1, 3, 8)
        partners[0, 0, :2] = torch.tensor([0.5, 0.0])   # front, visible
        partners[0, 0, 2] = 1.0
        partners[0, 1, :2] = torch.tensor([0.0, 0.5])   # side, outside +/-30 deg
        partners[0, 1, 2] = 1.0
        partners[0, 2, :2] = torch.tensor([1.5, 0.0])   # beyond 50m normalized radius
        partners[0, 2, 2] = 1.0
        obs[:, partner_start:road_start] = partners.reshape(1, -1)

        roads = torch.zeros(1, 4, 7)
        roads[0, 0, :2] = torch.tensor([0.5, 0.0])      # front, visible
        roads[0, 0, -1] = 2.0
        roads[0, 1, :2] = torch.tensor([0.0, 0.5])      # side, masked
        roads[0, 1, -1] = 3.0
        obs[:, road_start:] = roads.reshape(1, -1)

        masked = policy.apply_perception_mask(obs)
        masked_partners = masked[:, partner_start:road_start].reshape(1, 3, 8)
        masked_roads = masked[:, road_start:].reshape(1, 4, 7)

        self.assertTrue(torch.equal(masked_partners[0, 0], partners[0, 0]))
        self.assertTrue(torch.equal(masked_partners[0, 1], torch.zeros(8)))
        self.assertTrue(torch.equal(masked_partners[0, 2], torch.zeros(8)))
        self.assertTrue(torch.equal(masked_roads[0, 0], roads[0, 0]))
        self.assertTrue(torch.equal(masked_roads[0, 1], torch.zeros(7)))

    def test_drive_perception_mid_keeps_wider_front_sector_than_low(self):
        from pufferlib.ocean.torch import DrivePerceptionLow, DrivePerceptionMid

        driver_env = SimpleNamespace(
            single_observation_space=gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(7 + 1 * 8 + 1 * 7,),
                dtype=np.float32,
            ),
            single_action_space=gymnasium.spaces.MultiDiscrete([91]),
            max_partner_objects=1,
            partner_features=8,
            max_road_objects=1,
            road_features=7,
            dynamics_model="classic",
        )
        obs = torch.zeros(1, driver_env.single_observation_space.shape[0])
        # ~45 degrees: outside low (+/-30), inside mid (+/-60).
        obs[:, 7:15] = torch.tensor([[0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        low_masked = DrivePerceptionLow(driver_env).apply_perception_mask(obs)
        mid_masked = DrivePerceptionMid(driver_env).apply_perception_mask(obs)

        self.assertTrue(torch.equal(low_masked[:, 7:15], torch.zeros(1, 8)))
        self.assertTrue(torch.equal(mid_masked[:, 7:15], obs[:, 7:15]))

    def test_mix_ppo_loads_perception_level_policies(self):
        from pufferlib.ocean.torch import DrivePerceptionLow, DrivePerceptionMid, Recurrent

        driver_env = SimpleNamespace(
            single_observation_space=gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(7 + 3 * 8 + 4 * 7,),
                dtype=np.float32,
            ),
            single_action_space=gymnasium.spaces.MultiDiscrete([91]),
            max_partner_objects=3,
            partner_features=8,
            max_road_objects=4,
            road_features=7,
            dynamics_model="classic",
        )
        vecenv = SimpleNamespace(driver_env=driver_env)
        args = {
            "package": "ocean",
            "policy_name": "Drive",
            "rnn_name": "Recurrent",
            "policy": {"input_size": 16, "hidden_size": 32},
            "rnn": {"input_size": 32, "hidden_size": 32},
            "train": {
                "device": "cpu",
                "mix_ppo": True,
                "mix_ppo_policy_mix": "low:1,mid:1,high:1",
                "mix_ppo_policy_names": "DrivePerceptionLow,DrivePerceptionMid,Drive",
                "mix_ppo_rnn_names": "Recurrent,Recurrent,Recurrent",
                "mix_ppo_policy_paths": ",,",
            },
        }

        policies = load_mixed_policies(args, vecenv, "puffer_drive")

        self.assertEqual(len(policies), 3)
        self.assertIsInstance(policies[0], Recurrent)
        self.assertIsInstance(policies[0].policy, DrivePerceptionLow)
        self.assertIsInstance(policies[1].policy, DrivePerceptionMid)

    def test_mix_ppo_checkpoint_saves_each_policy_separately(self):
        with tempfile.TemporaryDirectory() as data_dir:
            policies = [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]
            for policy_idx, policy in enumerate(policies):
                with torch.no_grad():
                    policy.weight.fill_(policy_idx + 1)
                    policy.bias.fill_(policy_idx + 10)

            trainer = SimpleNamespace(
                logger=SimpleNamespace(run_id="unit"),
                config={"data_dir": data_dir, "env": "puffer_drive"},
                epoch=1000,
                uncompiled_policy=policies[0],
                uncompiled_policies=policies,
                mix_ppo=True,
                policy_trainable=[True, True, False],
                agent_policy_ids=torch.tensor([[0, 1, 2, 0]], dtype=torch.long),
                optimizer=SimpleNamespace(state_dict=lambda: {"main": "optimizer"}),
                optimizers=[
                    SimpleNamespace(state_dict=lambda idx=idx: {"policy": idx})
                    for idx in range(3)
                ],
                global_step=1234,
            )

            model_path = PuffeRL.save_checkpoint(trainer)
            checkpoint_dir = os.path.dirname(model_path)
            policy_model_names = [
                f"model_policy_{idx}_puffer_drive_001000.pt"
                for idx in range(3)
            ]

            for policy_idx, model_name in enumerate(policy_model_names):
                policy_path = os.path.join(checkpoint_dir, model_name)
                self.assertTrue(os.path.exists(policy_path), model_name)
                saved = torch.load(policy_path, map_location="cpu")
                for key, tensor in policies[policy_idx].state_dict().items():
                    self.assertTrue(torch.equal(saved[key], tensor), key)

            trainer_state = torch.load(
                os.path.join(checkpoint_dir, "trainer_state.pt"),
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(trainer_state["mix_policy_model_names"], policy_model_names)

    def test_drive_moe_forward_eval_with_default_recurrent_wrapper(self):
        from pufferlib.ocean.torch import DriveMoE, DriveMoE3, Recurrent

        env_name = "puffer_drive"
        argv = sys.argv[:]
        try:
            sys.argv = [sys.argv[0]]
            args = load_config(env_name)
        finally:
            sys.argv = argv
        args["vec"].update({"num_workers": 1, "num_envs": 1, "batch_size": 1})
        args["env"].update(
            {
                "num_agents": 2,
                "action_type": "discrete",
                "num_maps": 1,
                "init_mode": "create_all_valid",
                "control_mode": "control_agents",
                "episode_length": 2,
                "resample_frequency": 1000,
            }
        )
        args["policy"].update({"input_size": 32, "hidden_size": 32})
        args["rnn"].update({"input_size": 32, "hidden_size": 32})
        old_data_root = os.environ.get("DRIVE_BINARIES_DATA_ROOT")
        os.environ["DRIVE_BINARIES_DATA_ROOT"] = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "resources",
            "drive",
            "binaries",
        )

        vecenv = None
        try:
            vecenv = load_env(env_name, args)
            policy = DriveMoE(vecenv.driver_env, input_size=32, hidden_size=32, num_experts=2)
            self.assertEqual(policy.num_experts, 2)

            obs = torch.zeros(2, vecenv.driver_env.single_observation_space.shape[0])
            hidden = policy.encode_observations(obs)
            self.assertEqual(hidden.shape, (2, 32))
            self.assertEqual(policy.last_router_weights.shape, (2, 2))
            self.assertTrue(torch.allclose(policy.last_router_weights.sum(dim=1), torch.ones(2)))

            recurrent = Recurrent(vecenv.driver_env, policy, input_size=32, hidden_size=32)
            state = {
                "lstm_h": torch.zeros(2, 32),
                "lstm_c": torch.zeros(2, 32),
            }
            logits, value = recurrent.forward_eval(obs, state)
            self.assertEqual(value.shape, (2, 1))
            self.assertEqual(sum(t.shape[1] for t in logits), sum(policy.atn_dim))

            policy3 = DriveMoE3(vecenv.driver_env, input_size=32, hidden_size=32)
            self.assertEqual(policy3.num_experts, 3)
            hidden3 = policy3.encode_observations(obs)
            self.assertEqual(hidden3.shape, (2, 32))
            self.assertEqual(policy3.last_router_weights.shape, (2, 3))
            self.assertTrue(torch.allclose(policy3.last_router_weights.sum(dim=1), torch.ones(2)))
        finally:
            if vecenv is not None:
                vecenv.close()
            if old_data_root is None:
                os.environ.pop("DRIVE_BINARIES_DATA_ROOT", None)
            else:
                os.environ["DRIVE_BINARIES_DATA_ROOT"] = old_data_root

    def test_mix_ppo_runs_one_small_training_update(self):
        env_name = "puffer_drive"
        argv = sys.argv[:]
        try:
            sys.argv = [sys.argv[0]]
            args = load_config(env_name)
        finally:
            sys.argv = argv

        args["train"].update(
            {
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
                "mix_ppo": True,
                "mix_ppo_policy_mix": "learner:0.5, clone:0.5",
                "mix_ppo_policy_names": "Drive,Drive",
            }
        )
        args["vec"].update({"num_workers": 1, "num_envs": 1, "batch_size": 1})
        args["env"].update(
            {
                "num_agents": 4,
                "action_type": "discrete",
                "num_maps": 1,
                "init_mode": "create_all_valid",
                "control_mode": "control_agents",
                "episode_length": 2,
                "resample_frequency": 1000,
            }
        )
        args["policy"].update({"input_size": 32, "hidden_size": 32})
        args["rnn"].update({"input_size": 32, "hidden_size": 32})
        args["eval"] = {"wosac_realism_eval": False, "human_replay_eval": False}
        old_data_root = os.environ.get("DRIVE_BINARIES_DATA_ROOT")
        os.environ["DRIVE_BINARIES_DATA_ROOT"] = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "resources",
            "drive",
            "binaries",
        )
        os.environ["PUFFER_DISABLE_VIDEO"] = "1"

        vecenv = None
        pufferl = None
        try:
            vecenv = load_env(env_name, args)
            policy = load_policy(args, vecenv, env_name)
            policies = load_mixed_policies(args, vecenv, env_name, policy)
            train_config = dict(**args["train"], env=env_name, eval=args["eval"])
            pufferl = PuffeRL(train_config, vecenv, policies, logger=None)

            pufferl.evaluate()
            self.assertIn("mix_ppo/policy_0/score", pufferl.stats)
            self.assertIn("mix_ppo/policy_1/score", pufferl.stats)
            pufferl.train()

            self.assertTrue(pufferl.mix_ppo)
            self.assertEqual(len(pufferl.policies), 2)
            self.assertEqual(torch.bincount(pufferl.segment_policy_ids, minlength=2).tolist(), [2, 2])
        finally:
            if pufferl is not None:
                pufferl.utilization.stop()
            if vecenv is not None:
                vecenv.close()
            if old_data_root is None:
                os.environ.pop("DRIVE_BINARIES_DATA_ROOT", None)
            else:
                os.environ["DRIVE_BINARIES_DATA_ROOT"] = old_data_root

    def test_mix_ppo_bootstraps_all_segments_when_batch_spans_multiple_rollouts(self):
        env_name = "puffer_drive"
        argv = sys.argv[:]
        try:
            sys.argv = [sys.argv[0]]
            args = load_config(env_name)
        finally:
            sys.argv = argv

        num_agents = 4
        bptt_horizon = 2
        rollout_repeats = 3
        args["train"].update(
            {
                "device": "cpu",
                "optimizer": "adam",
                "compile": False,
                "total_timesteps": num_agents * bptt_horizon * rollout_repeats,
                "batch_size": num_agents * bptt_horizon * rollout_repeats,
                "bptt_horizon": bptt_horizon,
                "minibatch_size": num_agents * bptt_horizon,
                "max_minibatch_size": num_agents * bptt_horizon,
                "update_epochs": 1,
                "render": False,
                "checkpoint_interval": 999999,
                "mix_ppo": True,
                "mix_ppo_policy_mix": "learner:0.5, clone:0.5",
                "mix_ppo_policy_names": "Drive,Drive",
            }
        )
        args["vec"].update({"num_workers": 1, "num_envs": 1, "batch_size": 1})
        args["env"].update(
            {
                "num_agents": num_agents,
                "action_type": "discrete",
                "num_maps": 1,
                "init_mode": "create_all_valid",
                "control_mode": "control_agents",
                "episode_length": 8,
                "resample_frequency": 1000,
            }
        )
        args["policy"].update({"input_size": 32, "hidden_size": 32})
        args["rnn"].update({"input_size": 32, "hidden_size": 32})
        args["eval"] = {"wosac_realism_eval": False, "human_replay_eval": False}
        old_data_root = os.environ.get("DRIVE_BINARIES_DATA_ROOT")
        os.environ["DRIVE_BINARIES_DATA_ROOT"] = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "resources",
            "drive",
            "binaries",
        )
        os.environ["PUFFER_DISABLE_VIDEO"] = "1"

        vecenv = None
        pufferl = None
        try:
            vecenv = load_env(env_name, args)
            policy = load_policy(args, vecenv, env_name)
            policies = load_mixed_policies(args, vecenv, env_name, policy)
            train_config = dict(**args["train"], env=env_name, eval=args["eval"])
            pufferl = PuffeRL(train_config, vecenv, policies, logger=None)

            pufferl.evaluate()

            bootstrap_values = pufferl.values[:, -1]
            for repeat_idx in range(rollout_repeats):
                rows = slice(repeat_idx * num_agents, (repeat_idx + 1) * num_agents)
                self.assertTrue(
                    torch.all(bootstrap_values[rows] != 0),
                    f"missing bootstrap values for rollout segment {repeat_idx}",
                )
        finally:
            if pufferl is not None:
                pufferl.utilization.stop()
            if vecenv is not None:
                vecenv.close()
            if old_data_root is None:
                os.environ.pop("DRIVE_BINARIES_DATA_ROOT", None)
            else:
                os.environ["DRIVE_BINARIES_DATA_ROOT"] = old_data_root

    def test_single_policy_bootstraps_all_segments_when_batch_spans_multiple_rollouts(self):
        env_name = "puffer_drive"
        argv = sys.argv[:]
        try:
            sys.argv = [sys.argv[0]]
            args = load_config(env_name)
        finally:
            sys.argv = argv

        num_agents = 4
        bptt_horizon = 2
        rollout_repeats = 3
        args["train"].update(
            {
                "device": "cpu",
                "optimizer": "adam",
                "compile": False,
                "total_timesteps": num_agents * bptt_horizon * rollout_repeats,
                "batch_size": num_agents * bptt_horizon * rollout_repeats,
                "bptt_horizon": bptt_horizon,
                "minibatch_size": num_agents * bptt_horizon,
                "max_minibatch_size": num_agents * bptt_horizon,
                "update_epochs": 1,
                "render": False,
                "checkpoint_interval": 999999,
                "mix_ppo": False,
            }
        )
        args["vec"].update({"num_workers": 1, "num_envs": 1, "batch_size": 1})
        args["env"].update(
            {
                "num_agents": num_agents,
                "action_type": "discrete",
                "num_maps": 1,
                "init_mode": "create_all_valid",
                "control_mode": "control_agents",
                "episode_length": 8,
                "resample_frequency": 1000,
            }
        )
        args["policy"].update({"input_size": 32, "hidden_size": 32})
        args["rnn"].update({"input_size": 32, "hidden_size": 32})
        args["eval"] = {"wosac_realism_eval": False, "human_replay_eval": False}
        old_data_root = os.environ.get("DRIVE_BINARIES_DATA_ROOT")
        os.environ["DRIVE_BINARIES_DATA_ROOT"] = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "resources",
            "drive",
            "binaries",
        )
        os.environ["PUFFER_DISABLE_VIDEO"] = "1"

        vecenv = None
        pufferl = None
        try:
            vecenv = load_env(env_name, args)
            policy = load_policy(args, vecenv, env_name)
            train_config = dict(**args["train"], env=env_name, eval=args["eval"])
            pufferl = PuffeRL(train_config, vecenv, policy, logger=None)

            pufferl.evaluate()

            bootstrap_values = pufferl.values[:, -1]
            for repeat_idx in range(rollout_repeats):
                rows = slice(repeat_idx * num_agents, (repeat_idx + 1) * num_agents)
                self.assertTrue(
                    torch.all(bootstrap_values[rows] != 0),
                    f"missing single-policy bootstrap values for rollout segment {repeat_idx}",
                )
        finally:
            if pufferl is not None:
                pufferl.utilization.stop()
            if vecenv is not None:
                vecenv.close()
            if old_data_root is None:
                os.environ.pop("DRIVE_BINARIES_DATA_ROOT", None)
            else:
                os.environ["DRIVE_BINARIES_DATA_ROOT"] = old_data_root


if __name__ == "__main__":
    unittest.main()
