from dataclasses import dataclass
from typing import Optional

_VALID_MODES = frozenset(
    {"action_constraint", "reward_shaping", "both", "off"}
)


def _as_bool(v, default=False):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _as_float(v, default):
    if v is None or v == "":
        return float(default)
    return float(v)


@dataclass(frozen=True)
class ConservativePartnerConfig:
    partner_mode: str = "action_constraint"
    partner_max_abs_steer: float = 0.333
    partner_constrain_accel: bool = False
    partner_target_headway: float = 1.8
    w_center: float = 0.05
    w_align: float = 0.05
    w_steer: float = 0.05
    w_gap: float = 0.10
    warn_mixed_scene_rate_below: float = 0.8
    fail_mixed_scene_rate_below: Optional[float] = None

    @property
    def use_action_constraint(self) -> bool:
        return self.partner_mode in ("action_constraint", "both")

    @property
    def use_reward_shaping(self) -> bool:
        return self.partner_mode in ("reward_shaping", "both")


def parse_conservative_config(env_kwargs: dict) -> ConservativePartnerConfig:
    kw = dict(env_kwargs or {})
    mode = str(kw.get("partner_mode", "action_constraint")).strip()
    if mode not in _VALID_MODES:
        raise ValueError(
            f"partner_mode must be one of {sorted(_VALID_MODES)}, got {mode!r}"
        )
    headway = _as_float(kw.get("partner_target_headway"), 1.8)
    if headway <= 0:
        raise ValueError("partner_target_headway must be > 0")
    fail_below = kw.get("fail_mixed_scene_rate_below", None)
    if fail_below in ("", "null", "None"):
        fail_below = None
    elif fail_below is not None:
        fail_below = float(fail_below)
    return ConservativePartnerConfig(
        partner_mode=mode,
        partner_max_abs_steer=_as_float(kw.get("partner_max_abs_steer"), 0.333),
        partner_constrain_accel=_as_bool(
            kw.get("partner_constrain_accel"), False
        ),
        partner_target_headway=headway,
        w_center=_as_float(kw.get("partner_w_center"), 0.05),
        w_align=_as_float(kw.get("partner_w_align"), 0.05),
        w_steer=_as_float(kw.get("partner_w_steer"), 0.05),
        w_gap=_as_float(kw.get("partner_w_gap"), 0.10),
        warn_mixed_scene_rate_below=_as_float(
            kw.get("warn_mixed_scene_rate_below"), 0.8
        ),
        fail_mixed_scene_rate_below=fail_below,
    )
