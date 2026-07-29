import pytest
import torch

import pufferlib
from pufferlib.policy_sampling import (
    build_policy_segment_pools,
    parse_stratified_policy_specs,
    resolve_policy_update_steps,
)


def test_build_policy_segment_pools_keeps_shared_storage_indices():
    policy_ids = torch.tensor([0, 1, 0, 2, 1, 0])
    pools = build_policy_segment_pools(policy_ids, 3)
    assert [pool.tolist() for pool in pools] == [[0, 2, 5], [1, 4], [3]]


def test_parse_and_auto_resolve_policy_updates():
    config = {
        "mix_ppo_policy_minibatch_sizes": "32,16,0",
        "mix_ppo_policy_update_steps": "8,,0",
    }
    trainable = [True, True, False]
    specs = parse_stratified_policy_specs(config, 3, 8, trainable)
    pools = [torch.arange(8), torch.arange(4), torch.arange(2)]
    steps = resolve_policy_update_steps(
        specs,
        pools,
        horizon=8,
        update_epochs=1,
        trainable=trainable,
    )
    assert [spec.minibatch_size for spec in specs] == [32, 16, 0]
    assert steps == [8, 2, 0]


def test_stratified_requires_explicit_per_policy_minibatches():
    with pytest.raises(pufferlib.APIUsageError, match="minibatch_sizes"):
        parse_stratified_policy_specs({}, 2, 4, [True, True])


def test_stratified_rejects_minibatch_larger_than_pool():
    specs = parse_stratified_policy_specs(
        {"mix_ppo_policy_minibatch_sizes": "16,16"},
        2,
        4,
        [True, True],
    )
    with pytest.raises(pufferlib.APIUsageError, match="smaller than minibatch"):
        resolve_policy_update_steps(
            specs,
            [torch.arange(3), torch.arange(4)],
            horizon=4,
            update_epochs=1,
            trainable=[True, True],
        )
