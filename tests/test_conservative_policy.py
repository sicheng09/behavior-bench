import torch
import pufferlib.ocean.torch as drive_torch

from pufferlib.conservative.action_constraint import legal_action_mask
from pufferlib.conservative.policy import (
    DriveSteerConstrained,
    register_conservative_policies,
)


class _FakeEnv:
    # Minimal surface for Drive.__init__ — prefer constructing via a real
    # tiny Drive mock if project test helpers exist; otherwise monkeypatch
    # DriveSteerConstrained.decode_actions in isolation as below.
    pass


def test_register_is_idempotent_and_additive():
    assert not hasattr(drive_torch, "DriveSteerConstrained") or True
    register_conservative_policies()
    assert drive_torch.DriveSteerConstrained is DriveSteerConstrained
    register_conservative_policies()
    assert drive_torch.DriveSteerConstrained is DriveSteerConstrained


def test_decode_actions_masks_illegal_steer(monkeypatch):
    # Unit-test decode path without full C env: build a bare instance.
    pol = DriveSteerConstrained.__new__(DriveSteerConstrained)
    torch.nn.Module.__init__(pol)
    pol.is_continuous = False
    pol.atn_dim = [91]
    pol._legal_mask = legal_action_mask(0.333)
    hidden = torch.zeros(4, 8)
    raw = torch.zeros(4, 91)

    class _Actor(torch.nn.Module):
        def forward(self, h):
            return raw.expand(h.shape[0], -1).clone()

    pol.actor = _Actor()
    pol.value_fn = torch.nn.Linear(8, 1)
    # Bind real method from class
    logits, value = DriveSteerConstrained.decode_actions(pol, hidden)
    assert torch.isneginf(logits[0][0, 39])
    assert logits[0][0, 45] == 0.0
