from dataclasses import dataclass

import numpy as np

from pufferlib.conservative.action_constraint import NUM_STEER, STEERING_VALUES
from pufferlib.conservative.config import ConservativePartnerConfig

# Drive.REWARD_COMPONENT_NAMES order (must match drive.py):
# collision, offroad, goal, jerk_legacy, velocity, comfort, l_align, l_center, ...
RC_L_ALIGN = 6
RC_L_CENTER = 7

_PARTNER_ROLE = 1


@dataclass
class PartnerShapingResult:
    rewards: np.ndarray
    metrics: dict


class PartnerShapingEvaluator:
    def __init__(self, config: ConservativePartnerConfig):
        self.config = config

    def evaluate(
        self,
        *,
        base_rewards: np.ndarray,
        role_ids: np.ndarray,
        actions: np.ndarray,
        reward_components_raw: np.ndarray | None = None,
        lead_gap_m: np.ndarray | None = None,
        speed_mps: np.ndarray | None = None,
    ) -> PartnerShapingResult:
        base = np.asarray(base_rewards, dtype=np.float32).reshape(-1)
        roles = np.asarray(role_ids).reshape(-1)
        acts = np.asarray(actions).reshape(-1)
        n = base.shape[0]
        if roles.shape[0] != n or acts.shape[0] != n:
            raise ValueError("base_rewards, role_ids, and actions must share length N")

        rewards = base.copy()
        partner_mask = roles == _PARTNER_ROLE
        partner_idx = np.flatnonzero(partner_mask)

        if partner_idx.size == 0:
            return PartnerShapingResult(
                rewards=rewards,
                metrics={
                    "partner/steer_abs_mean": 0.0,
                    "partner/gap_shaping_mean": 0.0,
                    "partner/reward_mean": 0.0,
                },
            )

        cfg = self.config
        T = float(cfg.partner_target_headway)

        steer_abs = np.empty(partner_idx.size, dtype=np.float64)
        gap_vals = np.empty(partner_idx.size, dtype=np.float64)

        for k, i in enumerate(partner_idx):
            steer = STEERING_VALUES[int(acts[i]) % NUM_STEER]
            steer_abs[k] = abs(steer)

            if reward_components_raw is not None:
                raw_i = reward_components_raw[i]
                r_center = float(raw_i[RC_L_CENTER]) if len(raw_i) > RC_L_CENTER else 0.0
                r_align = float(raw_i[RC_L_ALIGN]) if len(raw_i) > RC_L_ALIGN else 0.0
            else:
                r_center = 0.0
                r_align = 0.0

            if lead_gap_m is None or speed_mps is None:
                gap_shaping = 0.0
            else:
                gap = float(lead_gap_m[i])
                if not np.isfinite(gap):
                    gap_shaping = 0.0
                else:
                    speed = max(float(speed_mps[i]), 0.1)
                    headway = gap / speed
                    gap_shaping = float(np.clip((headway / T) - 1.0, -1.0, 1.0))

            gap_vals[k] = gap_shaping
            shaped = (
                float(base[i])
                + cfg.w_center * r_center
                + cfg.w_align * r_align
                - cfg.w_steer * steer_abs[k]
                + cfg.w_gap * gap_shaping
            )
            rewards[i] = np.float32(np.clip(shaped, -1.0, 1.0))

        metrics = {
            "partner/steer_abs_mean": float(steer_abs.mean()),
            "partner/gap_shaping_mean": float(gap_vals.mean()),
            "partner/reward_mean": float(rewards[partner_idx].mean()),
        }
        return PartnerShapingResult(rewards=rewards, metrics=metrics)
