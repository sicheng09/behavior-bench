"""Logical per-policy sampling pools over a shared rollout buffer."""

from dataclasses import dataclass

import torch

import pufferlib


@dataclass(frozen=True)
class StratifiedPolicySpec:
    minibatch_size: int
    update_steps: int | None


def _split_ints(value, *, name, count, allow_empty=False):
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raw = [item.strip() for item in str(value).split(",")]
    if len(raw) != count:
        raise pufferlib.APIUsageError(
            f"{name} must contain exactly {count} comma-separated values"
        )
    values = []
    for item in raw:
        if allow_empty and (item is None or str(item).strip() == ""):
            values.append(None)
        else:
            values.append(int(item))
    return values


def parse_stratified_policy_specs(config, policy_count, horizon, trainable):
    """Parse per-policy minibatch sizes and optional fixed update counts."""
    minibatches = _split_ints(
        config.get("mix_ppo_policy_minibatch_sizes"),
        name="mix_ppo_policy_minibatch_sizes",
        count=policy_count,
    )
    if not minibatches:
        raise pufferlib.APIUsageError(
            "mix_ppo_sampling=stratified requires "
            "mix_ppo_policy_minibatch_sizes"
        )

    configured_steps = _split_ints(
        config.get("mix_ppo_policy_update_steps"),
        name="mix_ppo_policy_update_steps",
        count=policy_count,
        allow_empty=True,
    )
    if not configured_steps:
        configured_steps = [None] * policy_count

    specs = []
    for policy_idx, (minibatch, steps) in enumerate(
        zip(minibatches, configured_steps)
    ):
        if minibatch < 0 or (trainable[policy_idx] and minibatch == 0):
            raise pufferlib.APIUsageError(
                f"policy {policy_idx} minibatch size must be positive when trainable"
            )
        if minibatch and minibatch % horizon != 0:
            raise pufferlib.APIUsageError(
                f"policy {policy_idx} minibatch size {minibatch} must be "
                f"divisible by bptt_horizon {horizon}"
            )
        if steps is not None and steps < 0:
            raise pufferlib.APIUsageError(
                f"policy {policy_idx} update steps must be non-negative"
            )
        specs.append(StratifiedPolicySpec(minibatch, steps))
    return specs


def build_policy_segment_pools(segment_policy_ids, policy_count):
    """Return index tensors into shared rollout storage, one per policy."""
    return [
        torch.nonzero(segment_policy_ids == policy_idx, as_tuple=False).flatten()
        for policy_idx in range(policy_count)
    ]


def resolve_policy_update_steps(
    specs,
    pools,
    *,
    horizon,
    update_epochs,
    trainable,
):
    """Resolve fixed or pool-size-derived optimizer steps for one rollout."""
    resolved = []
    for policy_idx, (spec, pool) in enumerate(zip(specs, pools)):
        if not trainable[policy_idx]:
            resolved.append(0)
            continue
        pool_steps = int(pool.numel()) * horizon
        if pool_steps < spec.minibatch_size:
            raise pufferlib.APIUsageError(
                f"policy {policy_idx} rollout pool has {pool_steps} transitions, "
                f"smaller than minibatch {spec.minibatch_size}"
            )
        steps = spec.update_steps
        if steps is None:
            steps = int(update_epochs * pool_steps / spec.minibatch_size)
        if steps <= 0:
            raise pufferlib.APIUsageError(
                f"policy {policy_idx} resolved to zero optimizer steps"
            )
        resolved.append(steps)
    return resolved
