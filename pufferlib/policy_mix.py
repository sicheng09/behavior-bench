# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0
#
# This source code is derived from PufferDrive V2.0
# (https://github.com/Emerge-Lab/PufferDrive/)
# Copyright (c) 2026 PufferDrive, licensed under the MIT license.

import torch


POLICY_STAT_KEYS = (
    "n",
    "score",
    "offroad_rate",
    "collision_rate",
    "episode_length",
    "episode_return",
    "dnf_rate",
    "completion_rate",
    "lane_alignment_rate",
    "lane_aligned_steps",
    "lane_distance_count",
    "speed_limit_rate",
    "lane_distance_avg",
    "velocity_reward_total",
    "comfort_violations",
    "offroad_per_agent",
    "collisions_per_agent",
    "goals_sampled_this_episode",
    "goals_reached_this_episode",
    "speed_at_goal",
)

LOSS_DISPLAY_NAMES = {
    "policy_loss": "policy",
    "value_loss": "value",
    "old_approx_kl": "old_kl",
    "approx_kl": "kl",
    "importance": "ratio",
}


def parse_policy_mix(spec):
    """Parse a comma-separated policy mix specification.

    Example: "learner:0.5, transformer:0.25, gameformer:0.25"
    """
    if spec is None or spec == "":
        return ["learner"], [1.0]

    names = []
    fractions = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            names.append(item)
            fractions.append(1.0)
            continue

        name, fraction = item.split(":", 1)
        names.append(name.strip())
        fractions.append(float(fraction.strip()))

    if not names:
        return ["learner"], [1.0]

    total = sum(fractions)
    if total <= 0:
        raise ValueError("Policy mix fractions must sum to a positive value")

    fractions = [f / total for f in fractions]
    return names, fractions


def assign_policy_ids(total_agents, fractions, device=None):
    """Assign each PPO-controlled agent slot to a policy id by fraction.

    The assignment is stable within an epoch/rollout and deterministic by
    default. This mirrors mix_traffic's deficit-based interleaving without
    touching the C movement-mode split.
    """
    if total_agents <= 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    if not fractions:
        fractions = [1.0]

    total = sum(fractions)
    if total <= 0:
        raise ValueError("Policy fractions must sum to a positive value")

    fractions = [float(f) / total for f in fractions]
    counts = [0] * len(fractions)
    policy_ids = []
    for total_assigned in range(1, total_agents + 1):
        deficits = [
            fractions[policy_id] * total_assigned - counts[policy_id]
            for policy_id in range(len(fractions))
        ]
        policy_id = max(range(len(fractions)), key=lambda i: (deficits[i], -i))
        counts[policy_id] += 1
        policy_ids.append(policy_id)

    return torch.tensor(policy_ids, dtype=torch.long, device=device)


def group_policy_losses(losses):
    grouped = {}
    for key, value in losses.items():
        if not key.startswith("policy_") or "/" not in key:
            continue

        policy, metric = key.split("/", 1)
        try:
            policy_idx = int(policy.split("_", 1)[1])
        except (IndexError, ValueError):
            continue

        grouped.setdefault(policy_idx, {})[metric] = value

    return grouped


def flatten_policy_stats(policy_stats):
    flat = {}
    for policy_idx, stats in policy_stats.items():
        for key, value in stats.items():
            flat[f"mix_ppo/policy_{policy_idx}/{key}"] = value
    return flat
