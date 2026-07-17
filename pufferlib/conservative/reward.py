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


def compute_lead_gap_and_speed(
    x,
    y,
    heading,
    length,
    agent_offsets,
    *,
    prev_x=None,
    prev_y=None,
    dt=0.1,
    lateral_m=2.5,
):
    """Per-agent same-scene forward bumper gap (m) and speed (m/s).

    Lead = nearest valid vehicle ahead along heading with |lateral| < lateral_m.
    Gap is bumper-to-bumper (center distance minus half-lengths); no lead → +inf.
    Speed from finite difference when prev positions are finite; else 0.
    Scenes are sliced via agent_offsets (no global NxN).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    heading = np.asarray(heading, dtype=np.float64)
    length = np.asarray(length, dtype=np.float64)
    offsets = np.asarray(agent_offsets, dtype=np.int64)
    n = x.shape[0]
    lead_gap = np.full(n, np.inf, dtype=np.float32)
    speed = np.zeros(n, dtype=np.float32)

    if prev_x is not None and prev_y is not None and dt > 0:
        px = np.asarray(prev_x, dtype=np.float64)
        py = np.asarray(prev_y, dtype=np.float64)
        valid_prev = np.isfinite(px) & np.isfinite(py)
        dx = x - px
        dy = y - py
        speed_est = np.sqrt(dx * dx + dy * dy) / float(dt)
        speed = np.where(valid_prev, speed_est, 0.0).astype(np.float32)

    for start, stop in zip(offsets[:-1], offsets[1:]):
        idxs = range(int(start), int(stop))
        for i in idxs:
            if not (np.isfinite(x[i]) and np.isfinite(y[i])):
                continue
            cos_h = np.cos(heading[i])
            sin_h = np.sin(heading[i])
            best = np.inf
            for j in idxs:
                if j == i:
                    continue
                if not (np.isfinite(x[j]) and np.isfinite(y[j])):
                    continue
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                forward = dx * cos_h + dy * sin_h
                if forward <= 0:
                    continue
                lateral = abs(-dx * sin_h + dy * cos_h)
                if lateral > lateral_m:
                    continue
                bumper = forward - 0.5 * (length[i] + length[j])
                if bumper < best:
                    best = bumper
            if np.isfinite(best):
                lead_gap[i] = np.float32(max(best, 0.0))
    return lead_gap, speed


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
