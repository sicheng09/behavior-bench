import unittest
from types import SimpleNamespace

import gymnasium
import torch

from pufferlib.ocean.torch import (
    Drive,
    DriveMoE3,
    HybridDriveLiteLow,
    HybridDriveMid,
    HybridDriveMoE3High,
    HybridDriveOriginal,
)


def _driver_env():
    return SimpleNamespace(
        single_observation_space=gymnasium.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(7 + 8 + 7,), dtype=float
        ),
        single_action_space=gymnasium.spaces.MultiDiscrete([91]),
        max_partner_objects=1,
        partner_features=8,
        max_road_objects=1,
        road_features=7,
        dynamics_model="classic",
    )


class TestHybridPerception(unittest.TestCase):
    def test_low_mid_high_visibility_contract(self):
        env = _driver_env()
        obs = torch.zeros(1, 22)
        # 45 degrees and 0.5 normalized distance: outside Low, inside Mid/High.
        obs[:, 7:15] = torch.tensor([[0.5, 0.5, 1.0, 0, 0, 0, 0, 0]])

        low = HybridDriveLiteLow(env, input_size=16, hidden_size=80)
        mid = HybridDriveMid(env, input_size=16, hidden_size=80)
        high = HybridDriveMoE3High(env, input_size=16, hidden_size=80)

        self.assertTrue(torch.equal(low.apply_perception_mask(obs)[:, 7:15], torch.zeros(1, 8)))
        self.assertTrue(torch.equal(mid.apply_perception_mask(obs)[:, 7:15], obs[:, 7:15]))
        self.assertTrue(torch.equal(high.apply_perception_mask(obs), obs))

    def test_original_and_high_have_no_extra_parameters(self):
        env = _driver_env()
        original_base = Drive(env, input_size=16, hidden_size=80)
        original = HybridDriveOriginal(env, input_size=16, hidden_size=80)
        high_base = DriveMoE3(env, input_size=16, hidden_size=80)
        high = HybridDriveMoE3High(env, input_size=16, hidden_size=80)
        self.assertEqual(list(original_base.state_dict()), list(original.state_dict()))
        self.assertEqual(list(high_base.state_dict()), list(high.state_dict()))
        self.assertFalse(any(name.startswith("perception") for name, _ in original.named_parameters()))
        self.assertFalse(any(name.startswith("perception") for name, _ in high.named_parameters()))

    def test_hybrid_level_contract(self):
        env = _driver_env()
        low = HybridDriveLiteLow(env, input_size=16, hidden_size=80)
        mid = HybridDriveMid(env, input_size=16, hidden_size=80)
        high = HybridDriveMoE3High(env, input_size=16, hidden_size=80)
        original = HybridDriveOriginal(env, input_size=16, hidden_size=80)

        # DriveLite's hidden size is intentionally smaller and Low is intended
        # to run without the recurrent wrapper.
        self.assertEqual(low.hidden_size, 70)
        self.assertEqual(mid.hidden_size, 80)
        self.assertEqual(high.hidden_size, 80)
        self.assertEqual(original.hidden_size, 80)


if __name__ == "__main__":
    unittest.main()
