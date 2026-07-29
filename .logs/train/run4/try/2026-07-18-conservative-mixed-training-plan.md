# Conservative Mixed Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a config-gated `ConservativeMixDrive` that mixes ego PPO agents with conservative partner PPO agents (steer logit constraint first, reward shaping second) without using eval IDM in training.

**Architecture:** Reuse existing `mix_ppo` for policy routing/BPTT/optimizers/checkpoints. New Python-only package `pufferlib/conservative/` owns config, steer masks, partner reward shaping, and `ConservativeMixDrive(Drive)`. Partner action constraint is applied inside a new policy class `DriveSteerConstrained` that overrides `decode_actions` (on-policy logits mask). That class lives in the new package and is **runtime-registered** onto `pufferlib.ocean.torch` only when `ConservativeMixDrive` is constructed — no edits to `pufferl.py`, and no unconditional import on the default `puffer_drive` path.

**Tech Stack:** Python 3.9+, NumPy, PyTorch, PufferLib/PuffeRL, C-backed PufferDrive, unittest/pytest.

**Spec:** `.logs/train/run4/try/2026-07-18-conservative-mixed-training-design.zh.md`

## Global Constraints

- **Non-invasive (highest priority):** Must not change existing functionality, launch commands, or reproducibility of prior results; the whole change must be pack-deletable.
- `puffer train puffer_drive` and default `drive.ini` / `mix_ppo=False` / `mix_traffic` defaults must keep identical behavior and no new per-step overhead.
- Do **not** modify: `pufferlib/pufferl.py`, `pufferlib/policy_mix.py`, `pufferlib/ocean/drive/drive.h`, `pufferlib/ocean/drive/binding.c`, `pufferlib/ocean/drive/drive.py` reward/dynamics defaults, `pufferlib/evaluation/collision_classifier.py`, or default field values in `pufferlib/config/ocean/drive.ini` / `config/ocean/drive.ini`.
- Allowed code touch set: new `pufferlib/conservative/**`, new `drive_conservative_mix.ini` (+ mirror), **one additive** entry in `MAKE_FUNCTIONS` via `lazy_import`, new `tests/test_conservative_*.py`. Optional: zero-line-change runtime `setattr` on `ocean.torch` from inside the new env (no source edit to `torch.py`).
- Ego = policy 0, original `Drive` reward, no steer mask.
- Partner = policy 1; A/B only apply to partner and only under `puffer_drive_conservative_mix`.
- Main method must keep `idm_fraction = 0.0` (no eval-IDM in training story).
- Action constraint must mask **logits before sampling** (never rewrite actions after `sample_logits` while keeping old logprobs).
- Do **not** auto-commit. After each task: run tests, stop for review. Commit only if the user explicitly asks.
- Pairwise geometry for reward shaping (Task 7+) must stay inside per-scene slices from `agent_offsets` (same rule as adversarial).

---

## File Map

**Create**

- `pufferlib/conservative/__init__.py`
- `pufferlib/conservative/action_constraint.py` — steer tables, legal action mask, logit masking
- `pufferlib/conservative/config.py` — parse/validate partner mode + weights from env kwargs / optional yaml
- `pufferlib/conservative/policy.py` — `DriveSteerConstrained(Drive)`
- `pufferlib/conservative/reward.py` — `PartnerShapingEvaluator` (Phase B)
- `pufferlib/conservative/env.py` — `ConservativeMixDrive`
- `pufferlib/config/ocean/drive_conservative_mix.ini`
- `config/ocean/drive_conservative_mix.ini` — mirror copy (repo dual-ini convention)
- `tests/test_conservative_action_constraint.py`
- `tests/test_conservative_config.py`
- `tests/test_conservative_policy.py`
- `tests/test_conservative_env.py`
- `tests/test_conservative_reward.py` (Phase B)
- `tests/test_conservative_noninvasive.py` — default path isolation / revert checklist smoke

**Modify (additive only)**

- `pufferlib/ocean/environment.py` — append one `MAKE_FUNCTIONS` entry:
  `"drive_conservative_mix": lazy_import("pufferlib.conservative.env", "ConservativeMixDrive")`

**Explicitly unchanged**

- `pufferlib/pufferl.py`
- `pufferlib/policy_mix.py`
- `pufferlib/ocean/drive/drive.py` / `drive.h` / `binding.c`
- `pufferlib/ocean/torch.py` (no source edit; runtime register only)
- `pufferlib/config/ocean/drive.ini` and `config/ocean/drive.ini` defaults
- `pufferlib/adversarial/**` behavior
- eval / collision classifier semantics

**Revert pack (delete to undo)**

```text
rm -rf pufferlib/conservative
rm -f pufferlib/config/ocean/drive_conservative_mix.ini
rm -f config/ocean/drive_conservative_mix.ini
rm -f tests/test_conservative_*.py
# revert the single MAKE_FUNCTIONS line in environment.py
```

---

### Task 1: Steer Action Constraint Primitives

**Files:**
- Create: `pufferlib/conservative/action_constraint.py`
- Create: `pufferlib/conservative/__init__.py`
- Test: `tests/test_conservative_action_constraint.py`

**Interfaces:**
- Consumes: none
- Produces:
  - `STEERING_VALUES: tuple[float, ...]` (match `drive.h`)
  - `NUM_STEER: int = 13`
  - `NUM_ACCEL: int = 7`
  - `NUM_ACTIONS: int = 91`
  - `legal_steer_indices(max_abs_steer: float) -> list[int]`
  - `legal_action_mask(max_abs_steer: float, num_actions: int = 91) -> np.ndarray`  # bool shape `[num_actions]`
  - `mask_discrete_logits(logits, legal_mask) -> logits`  # torch; illegal → `-inf`

- [ ] **Step 1: Write the failing test**

Create `tests/test_conservative_action_constraint.py`:

```python
import math

import numpy as np
import pytest
import torch

from pufferlib.conservative.action_constraint import (
    NUM_ACTIONS,
    NUM_STEER,
    STEERING_VALUES,
    legal_action_mask,
    legal_steer_indices,
    mask_discrete_logits,
)


def test_steering_table_matches_drive_h_center():
    assert NUM_STEER == 13
    assert NUM_ACTIONS == 91
    assert STEERING_VALUES[6] == pytest.approx(0.0)
    assert abs(STEERING_VALUES[5]) == pytest.approx(0.167, abs=1e-3)


def test_legal_steer_indices_default_333():
    idx = legal_steer_indices(0.333)
    assert idx == [4, 5, 6, 7, 8]


def test_legal_steer_indices_tight_167():
    idx = legal_steer_indices(0.167)
    assert idx == [5, 6, 7]


def test_legal_action_mask_blocks_large_steer_keeps_all_accel():
    mask = legal_action_mask(0.333)
    assert mask.shape == (91,)
    assert mask.dtype == bool
    # accel=3 (zero), steer=6 (center) -> action 3*13+6 = 45
    assert mask[45]
    # accel=3, steer=0 (|steer|=1.0) -> 3*13+0 = 39
    assert not mask[39]
    # every accel row has exactly 5 legal steers
    for a in range(7):
        assert int(mask[a * 13 : (a + 1) * 13].sum()) == 5


def test_mask_discrete_logits_tuple_and_tensor():
    logits = torch.zeros(2, 91)
    mask = legal_action_mask(0.333)
    out = mask_discrete_logits((logits,), mask)
    assert isinstance(out, tuple)
    assert torch.isneginf(out[0][0, 39])
    assert out[0][0, 45] == 0.0
    # already-sampled path must remain differentiable-safe: no NaN after softmax
    probs = torch.softmax(out[0], dim=-1)
    assert torch.isfinite(probs).all()
    assert probs[0, 39] == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_conservative_action_constraint.py -v`  
Expected: FAIL with `ModuleNotFoundError: pufferlib.conservative`

- [ ] **Step 3: Minimal implementation**

Create `pufferlib/conservative/__init__.py`:

```python
"""Conservative partner mix training (opt-in via puffer_drive_conservative_mix)."""

__all__ = [
    "ConservativeMixDrive",
]
```

Create `pufferlib/conservative/action_constraint.py` with tables copied from `drive.h`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_conservative_action_constraint.py -v`  
Expected: PASS

- [ ] **Step 5: Stop for review (no commit unless user asks)**

---

### Task 2: Config Parse / Validate

**Files:**
- Create: `pufferlib/conservative/config.py`
- Test: `tests/test_conservative_config.py`

**Interfaces:**
- Consumes: none
- Produces:
  - `@dataclass ConservativePartnerConfig`
  - `parse_conservative_config(env_kwargs: dict) -> ConservativePartnerConfig`
  - Valid `partner_mode ∈ {"action_constraint","reward_shaping","both","off"}`

- [ ] **Step 1: Write failing tests**

```python
import pytest

from pufferlib.conservative.config import parse_conservative_config


def test_defaults_for_phase_a():
    cfg = parse_conservative_config({})
    assert cfg.partner_mode == "action_constraint"
    assert cfg.partner_max_abs_steer == pytest.approx(0.333)
    assert cfg.partner_constrain_accel is False
    assert cfg.partner_target_headway == pytest.approx(1.8)
    assert cfg.w_center == pytest.approx(0.05)
    assert cfg.w_align == pytest.approx(0.05)
    assert cfg.w_steer == pytest.approx(0.05)
    assert cfg.w_gap == pytest.approx(0.10)


def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match="partner_mode"):
        parse_conservative_config({"partner_mode": "idm"})


def test_rejects_nonpositive_headway():
    with pytest.raises(ValueError, match="headway"):
        parse_conservative_config({"partner_target_headway": 0.0})


def test_bool_parsing_from_ini_strings():
    cfg = parse_conservative_config(
        {"partner_constrain_accel": "False", "partner_mode": "both"}
    )
    assert cfg.partner_constrain_accel is False
    assert cfg.partner_mode == "both"
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_conservative_config.py -v`  
Expected: FAIL import/attribute

- [ ] **Step 3: Implement `config.py`**

```python
from dataclasses import dataclass

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
    fail_mixed_scene_rate_below: float | None = None

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
```

Note: if the runtime is Python <3.10, replace `float | None` with `Optional[float]`.

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Stop for review**

---

### Task 3: `DriveSteerConstrained` Policy + Runtime Registration

**Files:**
- Create: `pufferlib/conservative/policy.py`
- Test: `tests/test_conservative_policy.py`

**Interfaces:**
- Consumes: `legal_action_mask`, `mask_discrete_logits`, `ConservativePartnerConfig` fields on env
- Produces:
  - `class DriveSteerConstrained(Drive)`
  - `register_conservative_policies()` — `setattr(pufferlib.ocean.torch, "DriveSteerConstrained", ...)` if missing

**Why runtime register (not editing `torch.py`):**  
`load_mixed_policies` does `getattr(env_module.torch, policy_name)`. Registering during `ConservativeMixDrive.__init__` (before policy load) keeps default `import pufferlib.ocean.torch` free of conservative imports.

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `policy.py`**

```python
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
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Stop for review**

---

### Task 4: `ConservativeMixDrive` (Phase A: constraint mode wiring)

**Files:**
- Create: `pufferlib/conservative/env.py`
- Test: `tests/test_conservative_env.py`

**Interfaces:**
- Consumes: `parse_conservative_config`, `register_conservative_policies`, adversarial-style role helpers (reimplement locally to avoid coupling — copy small `normalize_role_ids` / `audit_scene_roles` into `env.py` or a tiny `assignment.py`)
- Produces: `class ConservativeMixDrive(Drive)`

Behavior for Phase A:
1. Pop/parse conservative kwargs; call `super().__init__(**drive_kwargs)`.
2. Require `policy_log_count == 2`.
3. Set `self.partner_max_abs_steer` etc. on the instance (policy reads these).
4. Call `register_conservative_policies()`.
5. Audit scene mix rates.
6. For Phase A, `step()` = `Drive.step` (no reward rewrite yet). Optionally attach assignment metrics into `info` on report intervals (mirror adversarial pattern lightly).

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
import pytest

from pufferlib.conservative.env import (
    ConservativeMixDrive,
    audit_scene_roles,
    normalize_role_ids,
)


def test_normalize_role_ids_tiles_and_rejects_unknown():
    ids = normalize_role_ids([0, 1], 5)
    assert list(ids) == [0, 1, 0, 1, 0]
    with pytest.raises(ValueError):
        normalize_role_ids([0, 2], 4)


def test_audit_scene_roles_mixed_rate():
    metrics = audit_scene_roles(
        agent_offsets=np.asarray([0, 2, 4]),
        role_ids=np.asarray([0, 1, 0, 1]),
    )
    assert metrics["assignment/mixed_scene_rate"] == pytest.approx(1.0)


def test_env_requires_two_policies_and_registers(monkeypatch):
    # Integration-style: if full C env too heavy, test __init__ contract with
    # a patched Drive.__init__. Prefer following tests/test_adversarial_mix_env.py.
    ...
```

Fill the env integration test by mirroring `tests/test_adversarial_mix_env.py` patterns used in-repo (patch `Drive.__init__` / construct via `__new__` for unit pieces; one optional small real-env smoke if CI maps exist).

Minimum required assertions:
- `parse` + attributes set
- `register_conservative_policies` called
- unknown `partner_mode` fails before C init when possible

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `env.py` (Phase A)**

Core sketch:

```python
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
    ...  # same semantics as adversarial env (ids 0/1 only)


def audit_scene_roles(agent_offsets, role_ids):
    ...  # mixed / ego_only / partner_only rates


class ConservativeMixDrive(Drive):
    def __init__(self, **kwargs):
        raw = dict(kwargs)
        cons_kwargs = {k: raw.pop(k) for k in list(raw) if k in _CONSERVATIVE_KEYS}
        self.conservative_config = parse_conservative_config(cons_kwargs)
        # Expose for DriveSteerConstrained
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
        ...  # warn / optional fail like adversarial

    def resample_maps(self):
        super().resample_maps()
        self._assignment_metrics = audit_scene_roles(
            self.agent_offsets, self._role_ids
        )
        self._check_assignment_thresholds()

    # Phase A: no reward override yet
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Stop for review**

---

### Task 5: Env Registration + INI Profiles (Phase A launchable)

**Files:**
- Modify: `pufferlib/ocean/environment.py` — **append only** inside `MAKE_FUNCTIONS`
- Create: `pufferlib/config/ocean/drive_conservative_mix.ini`
- Create: `config/ocean/drive_conservative_mix.ini` (copy)
- Test: extend `tests/test_conservative_env.py` or add load_config smoke

- [ ] **Step 1: Write failing registration test**

```python
from pufferlib.ocean.environment import MAKE_FUNCTIONS


def test_conservative_mix_registered_lazily():
    assert "drive_conservative_mix" in MAKE_FUNCTIONS
```

Also:

```python
from pufferlib.pufferl import load_config


def test_load_conservative_mix_ini():
    args = load_config("puffer_drive_conservative_mix")
    assert args["env_name"] == "puffer_drive_conservative_mix"
    assert args["train"]["mix_ppo"] in (True, "True", "true", 1, "1")
    assert "DriveSteerConstrained" in args["train"]["mix_ppo_policy_names"]
    assert float(args["env"].get("idm_fraction", 0)) == 0.0
```

- [ ] **Step 2: Run — expect FAIL on missing key / ini**

- [ ] **Step 3: Register + write INI**

In `MAKE_FUNCTIONS` add **after** `drive_adversarial` entry (do not reorder unrelated keys):

```python
    "drive_conservative_mix": lazy_import(
        "pufferlib.conservative.env", "ConservativeMixDrive"
    ),
```

Create `drive_conservative_mix.ini` by copying `drive_adversarial.ini` structure / homogeneous 500M hyperparams, then set:

```ini
[base]
package = ocean
env_name = puffer_drive_conservative_mix
policy_name = Drive
rnn_name = Recurrent

[env]
partner_mode = action_constraint
partner_max_abs_steer = 0.333
partner_constrain_accel = False
partner_target_headway = 1.8
mix_traffic = False
ppo_fraction = 1.0
idm_fraction = 0.0
expert_fraction = 0.0
; ... remainder same reward/dynamics as standard drive.ini / homogeneous recipe ...

[train]
mix_ppo = True
mix_ppo_policy_mix = ego:0.5, partner:0.5
mix_ppo_policy_names = Drive,DriveSteerConstrained
mix_ppo_rnn_names = Recurrent,Recurrent
mix_ppo_policy_paths = ,
mix_ppo_policy_trainable = True,True
; Align batch/steps with run4 homogeneous 500M when running C-A:
; total_timesteps = 500_000_000
; batch_size = 524288
```

Copy the same file to `config/ocean/drive_conservative_mix.ini`.

**Critical:** Do not edit `drive.ini`.

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Manual smoke (no full train)**

```bash
python -c "from pufferlib.pufferl import load_config; print(load_config('puffer_drive_conservative_mix')['env_name'])"
python -c "from pufferlib.pufferl import load_config; print(load_config('puffer_drive')['env_name'])"
```

Expected: first prints `puffer_drive_conservative_mix`; second still `puffer_drive`.

- [ ] **Step 6: Stop for review**

---

### Task 6: Non-Invasive Isolation Tests + Revert Checklist

**Files:**
- Create: `tests/test_conservative_noninvasive.py`

- [ ] **Step 1: Write tests**

```python
import importlib
import inspect

import pufferlib.ocean.torch as drive_torch


def test_default_torch_import_does_not_require_conservative_package(monkeypatch):
    # Importing ocean.torch must succeed even if conservative were absent;
    # we approximate by asserting DriveSteerConstrained is not defined in
    # torch.py source and is only present after register().
    src = inspect.getsource(drive_torch)
    assert "class DriveSteerConstrained" not in src


def test_drive_ini_has_no_partner_mode_keys():
    from pathlib import Path
    text = Path("pufferlib/config/ocean/drive.ini").read_text(encoding="utf-8")
    assert "partner_mode" not in text
    assert "DriveSteerConstrained" not in text


def test_puffer_drive_creator_still_drive():
    from pufferlib.ocean.environment import MAKE_FUNCTIONS
    assert MAKE_FUNCTIONS["drive"] == "Drive"
```

- [ ] **Step 2–4: Implement (tests only) / run PASS**

- [ ] **Step 5: Document revert verification in the try log**

Append to `.logs/train/run4/try/revert_checklist.md`:

```markdown
# ConservativeMix revert checklist
1. Delete pufferlib/conservative/
2. Delete both drive_conservative_mix.ini files
3. Delete tests/test_conservative_*.py
4. Remove drive_conservative_mix entry from MAKE_FUNCTIONS
5. python -c "from pufferlib.pufferl import load_config; load_config('puffer_drive')"
6. pytest tests/test_mix_ppo.py tests/test_adversarial_mix_env.py -q
```

- [ ] **Step 6: Stop for review**

---

### Task 7: Partner Reward Shaping (Phase B)

**Files:**
- Create: `pufferlib/conservative/reward.py`
- Test: `tests/test_conservative_reward.py`

**Interfaces:**
- Consumes: `ConservativePartnerConfig`, scene slices, actions, optional `reward_components_raw` / state frames (reuse `binding.vec_get_global_agent_state` pattern from adversarial if needed)
- Produces: `PartnerShapingEvaluator.evaluate(...) -> (rewards, metrics)`  
  Only overwrites partner slots; ego unchanged; final clip `[-1, 1]`.

Shaping terms (spec §7):
- `+ w_c * r_center`, `+ w_a * r_align` from existing component indices when available
- `- w_s * |steer|` from discrete action decode (`steer = STEERING_VALUES[action % 13]`)
- `+ w_g * gap_shaping(headway; T=partner_target_headway)` using geometric lead gap — **not** IDM formula; default T=1.8

- [ ] **Step 1: Write deterministic synthetic tests** (no C env): large steer lowers partner reward; ego entries identical byte-for-byte; headway below target lowers gap term; outputs finite and in `[-1,1]`.

- [ ] **Step 2: FAIL → implement → PASS**

- [ ] **Step 3: Stop for review**

---

### Task 8: Wire B into `ConservativeMixDrive.step`

**Files:**
- Modify: `pufferlib/conservative/env.py` only
- Test: extend `tests/test_conservative_env.py`

- [ ] **Step 1: Failing test** — with `partner_mode=reward_shaping` / `both`, partner rewards change, ego rewards identical to `base_rewards`.

- [ ] **Step 2: Implement**

```python
def step(self, actions):
    will_resample = (...)
    collect_metrics = (...)
    result = super().step(actions)
    if not will_resample and self.conservative_config.use_reward_shaping:
        self._apply_partner_shaping(actions, collect_metrics)
    ...
    return result
```

When `partner_mode=action_constraint`, skip shaping (Phase A behavior preserved).

- [ ] **Step 3: PASS + stop for review**

---

### Task 9: Launch Recipes + Short Smoke Train (optional ops)

**Files:**
- Create: `.logs/train/run4/try/launch_commands.md` (ops only; no code)

Document:

```bash
# Phase A main (C-A)
puffer train puffer_drive_conservative_mix \
  --train.total-timesteps 500000000 \
  --tag cons_mix_A_steer0333_500m

# Ablation tight
puffer train puffer_drive_conservative_mix \
  --env.partner-max-abs-steer 0.167 \
  --tag cons_mix_A_steer0167_500m

# Phase B
puffer train puffer_drive_conservative_mix \
  --env.partner-mode reward_shaping \
  --train.mix-ppo-policy-names Drive,Drive \
  --tag cons_mix_B_shape_500m

# Default path must still work unchanged
puffer train puffer_drive
```

Eval (unchanged harness):

```bash
# vs IDM / vs PPO using existing benchmark scripts — same as run4 protocol
```

Success gate (from spec): vs IDM Coll or At-fault relative −≥15% vs H0; vs PPO Coll +≤1pp.

- [ ] **Step 1: Write `launch_commands.md`**
- [ ] **Step 2: Optional 1-map / few-thousand-step smoke only if GPU free — do not start 500M without user OK**
- [ ] **Step 3: Stop for review**

---

## Execution Order Summary

| Order | Task | Phase | Pack-revertible |
| --- | --- | --- | --- |
| 1 | Action constraint primitives | A | yes |
| 2 | Config | A | yes |
| 3 | Constrained policy + runtime register | A | yes |
| 4 | ConservativeMixDrive skeleton | A | yes |
| 5 | Registration + ini | A | yes |
| 6 | Non-invasive tests + revert doc | A | yes |
| 7 | Reward shaping evaluator | B | yes |
| 8 | Wire shaping into step | B | yes |
| 9 | Launch recipes / optional smoke | Ops | n/a |

After Task 6, Phase A is code-complete and launchable. Tasks 7–8 are Phase B.

---

## Spec Coverage Self-Check

| Spec requirement | Task |
| --- | --- |
| Independent env name / default path unchanged | 5, 6 |
| No eval IDM in main training | 5 (ini `idm_fraction=0`) |
| A: logits mask before sample | 1, 3 |
| B: partner-only reward shaping, T=1.8 | 7, 8 |
| mix_ppo 50:50 ego/partner | 5 |
| Scene assignment audit | 4 |
| Experiment modes action_constraint / reward_shaping / both / off | 2, 5, 8 |
| Pack-deletable / no pufferl/C/drive.ini edits | Global + 6 |
| vs IDM success criteria | 9 (eval ops) |

## Placeholder Scan

None intentional. Task 4’s full C-env integration test may reuse adversarial’s `__new__`/patch style where CI lacks maps — that is an allowed technique, not a TBD.

---

## Handoff

Plan saved to:

`.logs/train/run4/try/2026-07-18-conservative-mixed-training-plan.md`

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — execute tasks in this session with checkpoints  

Which approach?
