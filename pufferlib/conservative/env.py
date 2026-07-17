import warnings

import numpy as np

from pufferlib.ocean.drive.drive import Drive

from pufferlib.conservative.config import parse_conservative_config
from pufferlib.conservative.policy import register_conservative_policies

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


def normalize_role_ids(policy_log_ids, num_agents):
    if not policy_log_ids:
        raise ValueError("ConservativeMixDrive requires mix_ppo policy_log_ids")
    ids = np.asarray(policy_log_ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) == 0:
        raise ValueError("policy_log_ids must be a non-empty 1D sequence")
    repeats = (num_agents + len(ids) - 1) // len(ids)
    ids = np.tile(ids, repeats)[:num_agents]
    unknown = np.setdiff1d(np.unique(ids), np.asarray([EGO, PARTNER]))
    if len(unknown):
        raise ValueError(f"V1 supports only policy ids 0 and 1, got {unknown}")
    return ids


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
        if self.policy_log_count != 2:
            raise ValueError(
                "ConservativeMixDrive requires exactly 2 mix_ppo policies"
            )
        register_conservative_policies()
        self._role_ids = normalize_role_ids(self.policy_log_ids, self.num_agents)
        self._assignment_metrics = audit_scene_roles(
            self.agent_offsets, self._role_ids
        )
        self._check_assignment_thresholds()

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

    # Phase A: no reward override — inherit Drive.step
