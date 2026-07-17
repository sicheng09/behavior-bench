import numpy as np
import torch

STEERING_VALUES = (
    -1.000, -0.833, -0.667, -0.500, -0.333, -0.167, 0.000,
    0.167, 0.333, 0.500, 0.667, 0.833, 1.000,
)
ACCELERATION_VALUES = (
    -4.0000, -2.6670, -1.3330, -0.0000, 1.3330, 2.6670, 4.0000,
)
NUM_STEER = len(STEERING_VALUES)
NUM_ACCEL = len(ACCELERATION_VALUES)
NUM_ACTIONS = NUM_ACCEL * NUM_STEER


def legal_steer_indices(max_abs_steer: float) -> list:
    return [i for i, s in enumerate(STEERING_VALUES) if abs(s) <= max_abs_steer + 1e-6]


def legal_action_mask(max_abs_steer: float, num_actions: int = NUM_ACTIONS) -> np.ndarray:
    legal_steer = set(legal_steer_indices(max_abs_steer))
    mask = np.zeros(num_actions, dtype=bool)
    for a in range(NUM_ACCEL):
        for s in range(NUM_STEER):
            idx = a * NUM_STEER + s
            if idx < num_actions and s in legal_steer:
                mask[idx] = True
    if not mask.any():
        raise ValueError(f"no legal actions for max_abs_steer={max_abs_steer}")
    return mask


def mask_discrete_logits(logits, legal_mask):
    """Mask illegal discrete actions with -inf. Accepts tensor or 1-tuple of tensor."""
    mask_t = torch.as_tensor(legal_mask, device=None)
    if isinstance(logits, tuple):
        masked = []
        for chunk in logits:
            m = mask_t.to(device=chunk.device, dtype=torch.bool)
            if chunk.shape[-1] != m.numel():
                raise ValueError(
                    f"logit dim {chunk.shape[-1]} != mask {m.numel()}"
                )
            out = chunk.clone()
            out[..., ~m] = -float("inf")
            masked.append(out)
        return tuple(masked)
    m = mask_t.to(device=logits.device, dtype=torch.bool)
    out = logits.clone()
    out[..., ~m] = -float("inf")
    return out
