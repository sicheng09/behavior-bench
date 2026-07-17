import pufferlib.ocean.torch as drive_torch
from pufferlib.ocean.torch import Drive

from pufferlib.conservative.action_constraint import (
    legal_action_mask,
    mask_discrete_logits,
)


class DriveSteerConstrained(Drive):
    """Drive policy with illegal large-steer logits masked to -inf."""

    def __init__(self, env, input_size=64, hidden_size=256, **kwargs):
        super().__init__(env, input_size=input_size, hidden_size=hidden_size, **kwargs)
        max_abs = float(getattr(env, "partner_max_abs_steer", 0.333))
        self._legal_mask = legal_action_mask(max_abs)
        self.partner_max_abs_steer = max_abs

    def decode_actions(self, flat_hidden):
        action, value = super().decode_actions(flat_hidden)
        if self.is_continuous:
            return action, value
        return mask_discrete_logits(action, self._legal_mask), value


def register_conservative_policies():
    if getattr(drive_torch, "DriveSteerConstrained", None) is not DriveSteerConstrained:
        setattr(drive_torch, "DriveSteerConstrained", DriveSteerConstrained)
