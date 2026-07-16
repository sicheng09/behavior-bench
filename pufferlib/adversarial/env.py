import numpy as np

from pufferlib.ocean.drive.drive import Drive

from .config import load_adversarial_config
from .reward import AsymmetricRewardEvaluator, StateFrame


def normalize_role_ids(policy_log_ids, num_agents):
    if not policy_log_ids:
        raise ValueError("AdversarialMixDrive requires mix_ppo policy_log_ids")
    ids = np.asarray(policy_log_ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) == 0:
        raise ValueError("policy_log_ids must be a non-empty 1D sequence")
    repeats = (num_agents + len(ids) - 1) // len(ids)
    ids = np.tile(ids, repeats)[:num_agents]
    unknown = np.setdiff1d(np.unique(ids), np.asarray([0, 1]))
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

    mixed = ego_only = opponent_only = 0
    size_buckets = {}
    for start, stop in zip(offsets[:-1], offsets[1:]):
        roles = role_ids[start:stop]
        has_ego = np.any(roles == 0)
        has_opponent = np.any(roles == 1)
        kind = (
            "mixed"
            if has_ego and has_opponent
            else ("ego_only" if has_ego else "opponent_only")
        )
        mixed += kind == "mixed"
        ego_only += kind == "ego_only"
        opponent_only += kind == "opponent_only"
        size = int(stop - start)
        bucket = size_buckets.setdefault(size, {"total": 0, "mixed": 0})
        bucket["total"] += 1
        bucket["mixed"] += kind == "mixed"

    total = max(len(offsets) - 1, 1)
    metrics = {
        "assignment/mixed_scene_rate": mixed / total,
        "assignment/ego_only_scene_rate": ego_only / total,
        "assignment/opponent_only_scene_rate": opponent_only / total,
    }
    for size, counts in size_buckets.items():
        metrics[f"assignment/size_{size}/mixed_rate"] = (
            counts["mixed"] / counts["total"]
        )
    return metrics


class AdversarialMixDrive(Drive):
    def __init__(self, adversarial_config_path, **kwargs):
        self.adversarial_config_path = str(adversarial_config_path)
        self.adversarial_config = load_adversarial_config(
            self.adversarial_config_path
        )
        super().__init__(**kwargs)
        if self.policy_log_count != 2:
            raise ValueError(
                "AdversarialMixDrive requires exactly 2 mix_ppo policies"
            )
        self._role_ids = normalize_role_ids(
            self.policy_log_ids, self.num_agents
        )
        self._adversarial_evaluator = AsymmetricRewardEvaluator(
            self.adversarial_config.reward, dt=self.dt
        )
        self._assignment_metrics = audit_scene_roles(
            self.agent_offsets, self._role_ids
        )
        self._pending_adversarial_metrics = {}
        self._invalid_reward_events = 0
        self._pre_state = StateFrame.from_mapping(
            self.get_global_agent_state()
        )
        self._pre_observations = self.observations.copy()
        self._check_assignment_thresholds()

    def _check_assignment_thresholds(self):
        import warnings

        mixed_rate = self._assignment_metrics["assignment/mixed_scene_rate"]
        warn_below = (
            self.adversarial_config.role_assignment.warn_mixed_scene_rate_below
        )
        if mixed_rate < warn_below:
            warnings.warn(
                f"mixed_scene_rate {mixed_rate:.3f} is below "
                f"warning threshold {warn_below:.3f}",
                RuntimeWarning,
            )
        fail_below = (
            self.adversarial_config.role_assignment.fail_mixed_scene_rate_below
        )
        if fail_below is not None and mixed_rate < fail_below:
            raise ValueError(
                f"mixed_scene_rate {mixed_rate:.3f} is below "
                f"configured failure threshold {fail_below:.3f}"
            )

    def reset(self, seed=0):
        observations, info = super().reset(seed)
        self._adversarial_evaluator.reset(self.num_agents)
        self._pre_state = StateFrame.from_mapping(
            self.get_global_agent_state()
        )
        self._pre_observations = self.observations.copy()
        return observations, info

    def resample_maps(self):
        super().resample_maps()
        self._assignment_metrics = audit_scene_roles(
            self.agent_offsets, self._role_ids
        )
        self._check_assignment_thresholds()
        self._adversarial_evaluator.reset(self.num_agents)
        self._pre_state = StateFrame.from_mapping(
            self.get_global_agent_state()
        )
        self._pre_observations = self.observations.copy()

    def _apply_adversarial_rewards(self, actions):
        base_rewards = self.rewards.copy()
        post_state = StateFrame.from_mapping(self.get_global_agent_state())
        result = self._adversarial_evaluator.evaluate(
            base_rewards=base_rewards,
            pre_state=self._pre_state,
            post_state=post_state,
            actions=actions,
            pre_observations=self._pre_observations,
            post_observations=self.observations,
            raw_components=self.reward_components_raw,
            agent_offsets=self.agent_offsets,
            role_ids=self._role_ids,
            action_type=(
                "discrete" if self._action_type_flag == 0 else "continuous"
            ),
        )
        self.rewards[:] = result.rewards
        self._pending_adversarial_metrics = result.metrics
        self._invalid_reward_events += int(
            result.metrics.get("adv/invalid_reward_events", 0)
        )
        if (
            self._invalid_reward_events
            > self.adversarial_config.reward.limits.max_invalid_reward_events
        ):
            raise FloatingPointError(
                "Adversarial reward exceeded max_invalid_reward_events"
            )
        self._pre_state = post_state
        self._pre_observations = self.observations.copy()

    def step(self, actions):
        will_resample = (
            self.tick > 0
            and self.resample_frequency > 0
            and self.tick % self.resample_frequency == 0
        )
        result = super().step(actions)
        if not will_resample:
            self._apply_adversarial_rewards(actions)
        observations, rewards, terminals, truncations, info = result
        if self.tick % self.report_interval == 0:
            metrics = {
                **self._assignment_metrics,
                **self._pending_adversarial_metrics,
            }
            info.append({"adversarial": metrics})
        return observations, rewards, terminals, truncations, info
