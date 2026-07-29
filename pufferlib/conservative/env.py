import warnings

import numpy as np

from pufferlib.ocean.drive.drive import Drive

from pufferlib.conservative.config import parse_conservative_config
from pufferlib.conservative.policy import register_conservative_policies
from pufferlib.conservative.reward import (
    PartnerShapingEvaluator,
    compute_lead_gap_and_speed,
)

EGO = 0
PARTNER = 1

_CONSERVATIVE_KEYS = {
    "partner_mode",
    "partner_max_abs_steer",
    "partner_constrain_accel",
    "partner_target_headway",
    "partner_w_center",
    "partner_w_align",
    "partner_w_steer",
    "partner_w_gap",
    "warn_mixed_scene_rate_below",
    "fail_mixed_scene_rate_below",
}

# Drive control timestep (seconds); used for finite-difference speed.
_DT_S = 0.1
_LATERAL_LANE_M = 2.5


def normalize_role_ids(policy_log_ids, num_agents, policy_log_count=None):
    """Map one ego policy and any number of partner policies to two roles.

    ``mix_ppo`` keeps the original policy ids so each neural policy can be
    forwarded, sampled, and logged independently.  Conservative shaping only
    needs to distinguish the learner (policy 0) from surrounding neural
    traffic (every policy id > 0), so collapse the latter to PARTNER here.
    """
    if not policy_log_ids:
        raise ValueError("ConservativeMixDrive requires mix_ppo policy_log_ids")
    ids = np.asarray(policy_log_ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) == 0:
        raise ValueError("policy_log_ids must be a non-empty 1D sequence")
    if np.any(ids < 0):
        raise ValueError("policy_log_ids must be non-negative")
    if policy_log_count is not None:
        count = int(policy_log_count)
        if count < 2:
            raise ValueError(
                "ConservativeMixDrive requires at least 2 mix_ppo policies"
            )
        if np.any(ids >= count):
            raise ValueError(
                "policy_log_ids must be smaller than policy_log_count"
            )
    repeats = (num_agents + len(ids) - 1) // len(ids)
    ids = np.tile(ids, repeats)[:num_agents]
    return np.where(ids == EGO, EGO, PARTNER).astype(np.int64, copy=False)


def audit_scene_roles(agent_offsets, role_ids):
    offsets = np.asarray(agent_offsets, dtype=np.int64)
    role_ids = np.asarray(role_ids, dtype=np.int64)
    if (
        offsets.ndim != 1
        or len(offsets) < 2
        or offsets[0] != 0
        or offsets[-1] != len(role_ids)
        or np.any(np.diff(offsets) < 0)
    ):
        raise ValueError("agent_offsets must be monotonic and span role_ids")

    mixed = ego_only = partner_only = 0
    size_buckets = {}
    for start, stop in zip(offsets[:-1], offsets[1:]):
        roles = role_ids[start:stop]
        has_ego = np.any(roles == EGO)
        has_partner = np.any(roles == PARTNER)
        kind = (
            "mixed"
            if has_ego and has_partner
            else ("ego_only" if has_ego else "partner_only")
        )
        mixed += kind == "mixed"
        ego_only += kind == "ego_only"
        partner_only += kind == "partner_only"
        size = int(stop - start)
        bucket = size_buckets.setdefault(size, {"total": 0, "mixed": 0})
        bucket["total"] += 1
        bucket["mixed"] += kind == "mixed"

    total = max(len(offsets) - 1, 1)
    metrics = {
        "assignment/mixed_scene_rate": mixed / total,
        "assignment/ego_only_scene_rate": ego_only / total,
        "assignment/partner_only_scene_rate": partner_only / total,
    }
    for size, counts in size_buckets.items():
        metrics[f"assignment/size_{size}/mixed_rate"] = (
            counts["mixed"] / counts["total"]
        )
    return metrics


class ConservativeMixDrive(Drive):
    def __init__(self, **kwargs):
        raw = dict(kwargs)
        cons_kwargs = {k: raw.pop(k) for k in list(raw) if k in _CONSERVATIVE_KEYS}
        self.conservative_config = parse_conservative_config(cons_kwargs)
        self.partner_max_abs_steer = (
            self.conservative_config.partner_max_abs_steer
        )
        self.partner_mode = self.conservative_config.partner_mode
        super().__init__(**raw)
        if self.policy_log_count < 2:
            raise ValueError(
                "ConservativeMixDrive requires at least 2 mix_ppo policies"
            )
        register_conservative_policies()
        self._role_ids = normalize_role_ids(
            self.policy_log_ids,
            self.num_agents,
            self.policy_log_count,
        )
        self._assignment_metrics = audit_scene_roles(
            self.agent_offsets, self._role_ids
        )
        self._partner_shaping_evaluator = None
        self._pending_shaping_metrics = {}
        self._gap_x = self._gap_y = None
        self._gap_heading = self._gap_length = None
        self._gap_z = self._gap_id = self._gap_width = self._gap_type = None
        self._prev_x = self._prev_y = None
        if self.conservative_config.use_reward_shaping:
            self._partner_shaping_evaluator = PartnerShapingEvaluator(
                self.conservative_config
            )
            self._init_gap_state_buffers()
        self._check_assignment_thresholds()

    def _init_gap_state_buffers(self):
        n = self.num_agents
        self._gap_x = np.zeros(n, dtype=np.float32)
        self._gap_y = np.zeros(n, dtype=np.float32)
        self._gap_z = np.zeros(n, dtype=np.float32)
        self._gap_heading = np.zeros(n, dtype=np.float32)
        self._gap_id = np.zeros(n, dtype=np.int32)
        self._gap_length = np.zeros(n, dtype=np.float32)
        self._gap_width = np.zeros(n, dtype=np.float32)
        self._gap_type = np.zeros(n, dtype=np.int32)
        self._prev_x = np.full(n, np.nan, dtype=np.float32)
        self._prev_y = np.full(n, np.nan, dtype=np.float32)

    def _check_assignment_thresholds(self):
        mixed_rate = self._assignment_metrics["assignment/mixed_scene_rate"]
        warn_below = self.conservative_config.warn_mixed_scene_rate_below
        if mixed_rate < warn_below:
            warnings.warn(
                f"mixed_scene_rate {mixed_rate:.3f} is below "
                f"warning threshold {warn_below:.3f}",
                RuntimeWarning,
            )
        fail_below = self.conservative_config.fail_mixed_scene_rate_below
        if fail_below is not None and mixed_rate < fail_below:
            raise ValueError(
                f"mixed_scene_rate {mixed_rate:.3f} is below "
                f"configured failure threshold {fail_below:.3f}"
            )

    def resample_maps(self):
        super().resample_maps()
        self._assignment_metrics = audit_scene_roles(
            self.agent_offsets, self._role_ids
        )
        self._check_assignment_thresholds()
        if self._partner_shaping_evaluator is not None:
            if self.num_agents != len(self._gap_x):
                self._init_gap_state_buffers()
            else:
                self._prev_x.fill(np.nan)
                self._prev_y.fill(np.nan)

    def _lead_gap_and_speed(self):
        """Same-scene forward lead gap + finite-diff speed; None if unavailable."""
        if self._gap_x is None or not hasattr(self, "c_envs"):
            return None, None
        try:
            from pufferlib.ocean.drive import binding

            binding.vec_get_global_agent_state(
                self.c_envs,
                self._gap_x,
                self._gap_y,
                self._gap_z,
                self._gap_heading,
                self._gap_id,
                self._gap_length,
                self._gap_width,
                self._gap_type,
            )
        except Exception:
            return None, None
        lead_gap_m, speed_mps = compute_lead_gap_and_speed(
            self._gap_x,
            self._gap_y,
            self._gap_heading,
            self._gap_length,
            self.agent_offsets,
            prev_x=self._prev_x,
            prev_y=self._prev_y,
            dt=_DT_S,
            lateral_m=_LATERAL_LANE_M,
        )
        np.copyto(self._prev_x, self._gap_x)
        np.copyto(self._prev_y, self._gap_y)
        return lead_gap_m, speed_mps

    def _apply_partner_shaping(self, actions, collect_metrics):
        lead_gap_m, speed_mps = self._lead_gap_and_speed()
        result = self._partner_shaping_evaluator.evaluate(
            base_rewards=self.rewards,
            role_ids=self._role_ids,
            actions=actions,
            reward_components_raw=self.reward_components_raw,
            lead_gap_m=lead_gap_m,
            speed_mps=speed_mps,
        )
        self.rewards[:] = result.rewards
        if collect_metrics:
            self._pending_shaping_metrics = result.metrics

    def step(self, actions):
        will_resample = (
            self.tick > 0
            and self.resample_frequency > 0
            and self.tick % self.resample_frequency == 0
        )
        # Drive.step increments tick on the non-resample path; mirror that
        # prediction so metrics are collected only on report boundaries.
        next_tick = self.tick if will_resample else self.tick + 1
        collect_metrics = next_tick % self.report_interval == 0
        result = super().step(actions)
        if (
            not will_resample
            and self.conservative_config.use_reward_shaping
            and self._partner_shaping_evaluator is not None
        ):
            self._apply_partner_shaping(actions, collect_metrics)
        observations, rewards, terminals, truncations, info = result
        if (
            collect_metrics
            and not will_resample
            and self.conservative_config.use_reward_shaping
        ):
            metrics = {
                **self._assignment_metrics,
                **self._pending_shaping_metrics,
            }
            info.append({"conservative": metrics})
        return observations, rewards, terminals, truncations, info
