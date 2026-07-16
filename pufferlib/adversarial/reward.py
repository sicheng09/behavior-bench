from dataclasses import dataclass
from typing import Dict

import numpy as np

from .config import RewardConfig


MAX_SPEED_MPS = 100.0
COLLISION_COMPONENT = 0
OFFROAD_COMPONENT = 1
REVERSE_COMPONENT = 9
SPEED_LIMIT_COMPONENT = 10


@dataclass(frozen=True)
class StateFrame:
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    length: np.ndarray
    width: np.ndarray
    valid: np.ndarray

    @classmethod
    def from_mapping(cls, state):
        x = np.asarray(state["x"], dtype=np.float32)
        y = np.asarray(state["y"], dtype=np.float32)
        return cls(
            x=x,
            y=y,
            heading=np.asarray(state["heading"], dtype=np.float32),
            length=np.asarray(state["length"], dtype=np.float32),
            width=np.asarray(state["width"], dtype=np.float32),
            valid=(
                np.isfinite(x)
                & np.isfinite(y)
                & (x > -9000)
                & (y > -9000)
            ),
        )


@dataclass(frozen=True)
class RewardEvaluation:
    rewards: np.ndarray
    components: Dict[str, np.ndarray]
    metrics: Dict[str, float]


class AsymmetricRewardEvaluator:
    def __init__(self, config: RewardConfig, dt: float):
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.config = config
        self.dt = float(dt)
        self._previous_accel = None
        self._previous_steer = None
        self._has_history = False

    def reset(self, num_agents: int) -> None:
        self._previous_accel = np.zeros(num_agents, dtype=np.float32)
        self._previous_steer = np.zeros(num_agents, dtype=np.float32)
        self._has_history = False

    def evaluate(
        self,
        *,
        base_rewards,
        pre_state,
        post_state,
        actions,
        pre_observations,
        post_observations,
        raw_components,
        agent_offsets,
        role_ids,
        action_type,
    ) -> RewardEvaluation:
        base_rewards = np.asarray(base_rewards, dtype=np.float32)
        rewards = base_rewards.copy()
        role_ids = np.asarray(role_ids, dtype=np.int64)
        raw_components = np.asarray(raw_components, dtype=np.float32)
        agent_offsets = np.asarray(agent_offsets, dtype=np.int64)
        n = len(base_rewards)
        self._validate_shapes(
            n,
            pre_state,
            post_state,
            raw_components,
            agent_offsets,
            role_ids,
        )
        if self._previous_accel is None or len(self._previous_accel) != n:
            self.reset(n)

        pre_speed = self._observation_speed(pre_observations, n)
        post_speed = self._observation_speed(post_observations, n)
        accel = (post_speed - pre_speed) / self.dt
        steer = self._decode_steer(actions, action_type, n)
        if self._has_history:
            jerk = (accel - self._previous_accel) / self.dt
            steer_rate = (steer - self._previous_steer) / self.dt
        else:
            jerk = np.zeros(n, dtype=np.float32)
            steer_rate = np.zeros(n, dtype=np.float32)

        ego_cost = np.zeros(n, dtype=np.float32)
        fault_penalty = np.zeros(n, dtype=np.float32)
        kinematics_cost = np.zeros(n, dtype=np.float32)
        normality = np.zeros(n, dtype=np.float32)
        min_ttc = np.full(n, np.inf, dtype=np.float32)
        hard_brake_events = np.zeros(n, dtype=np.float32)
        ambiguous_collision_events = np.zeros(n, dtype=np.float32)

        for start, stop in zip(agent_offsets[:-1], agent_offsets[1:]):
            scene_roles = role_ids[start:stop]
            ego_indices = start + np.flatnonzero(scene_roles == 0)
            opponent_indices = start + np.flatnonzero(scene_roles == 1)
            for opponent_idx in opponent_indices:
                (
                    ego_cost[opponent_idx],
                    min_ttc[opponent_idx],
                    hard_brake_events[opponent_idx],
                ) = self._max_causal_ego_cost(
                    opponent_idx,
                    ego_indices,
                    pre_state,
                    post_state,
                    pre_speed,
                    post_speed,
                    accel,
                )
                (
                    fault_penalty[opponent_idx],
                    ambiguous_collision_events[opponent_idx],
                ) = self._fault_penalty(
                    opponent_idx,
                    ego_indices,
                    pre_state,
                    pre_speed,
                    accel,
                    steer,
                    raw_components,
                )
                kinematics_cost[opponent_idx] = self._kinematics_cost(
                    opponent_idx,
                    pre_state,
                    post_state,
                    post_speed,
                    accel,
                    jerk,
                    steer,
                    steer_rate,
                    raw_components,
                )
                normality[opponent_idx] = np.clip(
                    post_speed[opponent_idx]
                    / self.config.thresholds.normal_speed_mps,
                    0.0,
                    1.0,
                )

        opponent_mask = role_ids == 1
        weights = self.config.weights
        raw_adversarial = (
            weights.ego_cost * ego_cost
            - weights.fault * fault_penalty
            - weights.kinematics * kinematics_cost
            + weights.normality * normality
        )
        adversarial = np.where(
            raw_adversarial > 0,
            np.minimum(
                raw_adversarial,
                self.config.limits.positive_reward_cap,
            ),
            raw_adversarial,
        )
        adversarial = np.clip(adversarial, -1.0, 1.0)
        adversarial[fault_penalty > 0] = -1.0
        rewards[opponent_mask] = adversarial[opponent_mask]

        invalid_reward_mask = ~np.isfinite(base_rewards) | ~np.isfinite(rewards)
        rewards[invalid_reward_mask] = -1.0
        self._previous_accel = accel.astype(np.float32, copy=True)
        self._previous_steer = steer.astype(np.float32, copy=True)
        self._has_history = True

        components = {
            "ego_cost": ego_cost,
            "fault_penalty": fault_penalty,
            "kinematics_cost": kinematics_cost,
            "normality": normality,
            "min_ttc": min_ttc,
        }
        metrics = self._aggregate_metrics(
            components,
            rewards,
            opponent_mask,
            hard_brake_events,
            ambiguous_collision_events,
        )
        metrics["adv/invalid_reward_events"] = float(
            invalid_reward_mask.sum()
        )
        return RewardEvaluation(rewards, components, metrics)

    @staticmethod
    def _observation_speed(observations, num_agents):
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[0] != num_agents:
            raise ValueError("observations must have one row per agent")
        if observations.shape[1] < 3:
            raise ValueError("observations must contain normalized speed")
        return np.abs(observations[:, 2]) * MAX_SPEED_MPS

    @staticmethod
    def _velocity(speed, heading):
        return np.stack(
            (speed * np.cos(heading), speed * np.sin(heading)),
            axis=-1,
        )

    def _pair_ttc(
        self,
        ego_idx,
        opponent_idx,
        state,
        speed,
    ):
        rel = np.asarray(
            [
                state.x[opponent_idx] - state.x[ego_idx],
                state.y[opponent_idx] - state.y[ego_idx],
            ],
            dtype=np.float32,
        )
        distance = float(np.linalg.norm(rel))
        if distance <= 1e-6:
            return 0.0, 0.0, 0.0
        ego_forward = np.asarray(
            [np.cos(state.heading[ego_idx]), np.sin(state.heading[ego_idx])]
        )
        longitudinal = float(np.dot(rel, ego_forward))
        lateral = float(ego_forward[0] * rel[1] - ego_forward[1] * rel[0])
        velocities = self._velocity(speed, state.heading)
        closing = float(
            np.dot(velocities[ego_idx] - velocities[opponent_idx], rel)
            / distance
        )
        bumper_distance = max(
            distance
            - 0.5 * (state.length[ego_idx] + state.length[opponent_idx]),
            0.0,
        )
        if closing <= 1e-6:
            return np.inf, longitudinal, lateral
        ttc = max(
            bumper_distance - self.config.thresholds.safe_distance_m,
            0.0,
        ) / closing
        return ttc, longitudinal, lateral

    def _max_causal_ego_cost(
        self,
        opponent_idx,
        ego_indices,
        pre_state,
        post_state,
        pre_speed,
        post_speed,
        accel,
    ):
        best_cost = 0.0
        best_ttc = np.inf
        hard_brake_event = 0.0
        threshold = self.config.thresholds.ttc_seconds
        for ego_idx in ego_indices:
            if not pre_state.valid[ego_idx] or not pre_state.valid[opponent_idx]:
                continue
            pre_ttc, longitudinal, lateral = self._pair_ttc(
                ego_idx, opponent_idx, pre_state, pre_speed
            )
            lateral_limit = max(
                float(pre_state.width[ego_idx]),
                float(pre_state.width[opponent_idx]),
            )
            if longitudinal <= 0 or abs(lateral) > lateral_limit:
                continue
            if np.isfinite(pre_ttc):
                best_ttc = min(best_ttc, pre_ttc)
            if post_state.valid[ego_idx] and post_state.valid[opponent_idx]:
                post_ttc, _, _ = self._pair_ttc(
                    ego_idx, opponent_idx, post_state, post_speed
                )
            else:
                post_ttc = pre_ttc
            pre_risk = self._ttc_risk(pre_ttc, threshold)
            post_risk = self._ttc_risk(post_ttc, threshold)
            risk_increase = max(0.0, post_risk - pre_risk)
            brake = max(
                0.0,
                -float(accel[ego_idx])
                - self.config.thresholds.hard_brake_mps2,
            ) / self.config.thresholds.hard_brake_mps2
            is_near_term_risk = min(pre_ttc, post_ttc) <= threshold
            if is_near_term_risk and brake > 0:
                hard_brake_event = 1.0
            if not is_near_term_risk:
                brake = 0.0
            cost = risk_increase + min(brake, 1.0)
            best_cost = max(best_cost, min(cost, 1.0))
        return best_cost, best_ttc, hard_brake_event

    @staticmethod
    def _ttc_risk(ttc, threshold):
        if not np.isfinite(ttc):
            return 0.0
        return float(np.clip((threshold - ttc) / threshold, 0.0, 1.0))

    def _fault_penalty(
        self,
        opponent_idx,
        ego_indices,
        pre_state,
        pre_speed,
        accel,
        steer,
        raw_components,
    ):
        if raw_components[opponent_idx, COLLISION_COMPONENT] <= 0:
            return 0.0, 0.0
        if (
            accel[opponent_idx]
            < -self.config.thresholds.hard_brake_mps2
            or abs(steer[opponent_idx])
            > self.config.thresholds.hard_steer_rad
        ):
            return 1.0, 0.0

        velocities = self._velocity(pre_speed, pre_state.heading)
        for ego_idx in ego_indices:
            if not pre_state.valid[ego_idx] or not pre_state.valid[opponent_idx]:
                continue
            rel_from_opponent = np.asarray(
                [
                    pre_state.x[ego_idx] - pre_state.x[opponent_idx],
                    pre_state.y[ego_idx] - pre_state.y[opponent_idx],
                ],
                dtype=np.float32,
            )
            distance = float(np.linalg.norm(rel_from_opponent))
            if distance <= 1e-6:
                continue
            opponent_forward = np.asarray(
                [
                    np.cos(pre_state.heading[opponent_idx]),
                    np.sin(pre_state.heading[opponent_idx]),
                ]
            )
            longitudinal = float(np.dot(rel_from_opponent, opponent_forward))
            lateral = float(
                opponent_forward[0] * rel_from_opponent[1]
                - opponent_forward[1] * rel_from_opponent[0]
            )
            vector_ego_to_opponent = -rel_from_opponent
            closing = float(
                np.dot(
                    velocities[ego_idx] - velocities[opponent_idx],
                    vector_ego_to_opponent,
                )
                / distance
            )
            max_contact_distance = (
                0.5
                * (
                    pre_state.length[ego_idx]
                    + pre_state.length[opponent_idx]
                )
                + self.config.thresholds.safe_distance_m
            )
            if (
                longitudinal < 0
                and abs(lateral)
                <= max(
                    pre_state.width[ego_idx],
                    pre_state.width[opponent_idx],
                )
                and closing > 0
                and distance <= max_contact_distance
            ):
                return 0.0, 0.0
        return 1.0, 1.0

    def _kinematics_cost(
        self,
        opponent_idx,
        pre_state,
        post_state,
        post_speed,
        accel,
        jerk,
        steer,
        steer_rate,
        raw_components,
    ):
        thresholds = self.config.thresholds
        heading_delta = self._wrap_angle(
            post_state.heading[opponent_idx]
            - pre_state.heading[opponent_idx]
        )
        lateral_accel = (
            post_speed[opponent_idx] * heading_delta / self.dt
        )
        violations = [
            self._hinge(abs(accel[opponent_idx]), thresholds.max_accel_mps2),
            self._hinge(
                abs(lateral_accel),
                thresholds.max_lateral_accel_mps2,
            ),
            self._hinge(abs(jerk[opponent_idx]), thresholds.max_jerk_mps3),
            self._hinge(
                abs(steer_rate[opponent_idx]),
                thresholds.max_steer_rate_radps,
            ),
            max(0.0, float(raw_components[opponent_idx, OFFROAD_COMPONENT])),
            max(0.0, float(raw_components[opponent_idx, REVERSE_COMPONENT])),
            max(
                0.0,
                float(raw_components[opponent_idx, SPEED_LIMIT_COMPONENT]),
            ),
        ]
        max_cost = (
            self.config.limits.kinematics_penalty_cap
            / max(self.config.weights.kinematics, 1e-6)
        )
        return float(np.clip(sum(violations), 0.0, max_cost))

    @staticmethod
    def _hinge(value, limit):
        return max(0.0, float(value) - limit) / max(limit, 1e-6)

    @staticmethod
    def _wrap_angle(angle):
        return (float(angle) + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _decode_steer(actions, action_type, num_agents):
        actions = np.asarray(actions)
        if action_type == "continuous":
            if actions.shape != (num_agents, 2):
                raise ValueError("continuous actions must have shape (N, 2)")
            return actions[:, 1].astype(np.float32)
        if action_type != "discrete":
            raise ValueError(f"Unknown action_type: {action_type}")
        indices = actions.reshape(num_agents, -1)[:, 0].astype(np.int64)
        if np.any(indices < 0) or np.any(indices >= 91):
            raise ValueError("classic discrete action indices must be in [0, 90]")
        return np.linspace(-1.0, 1.0, 13, dtype=np.float32)[indices % 13]

    @staticmethod
    def _aggregate_metrics(
        components,
        rewards,
        opponent_mask,
        hard_brake_events,
        ambiguous_collision_events,
    ):
        if not np.any(opponent_mask):
            mean = lambda _: 0.0
        else:
            mean = lambda values: float(np.mean(values[opponent_mask]))
        finite_ttc = components["min_ttc"][
            opponent_mask & np.isfinite(components["min_ttc"])
        ]
        return {
            "adv/cost_ego": mean(components["ego_cost"]),
            "adv/penalty_fault": mean(components["fault_penalty"]),
            "adv/cost_kinematics": mean(components["kinematics_cost"]),
            "adv/normality": mean(components["normality"]),
            "adv/reward_total": mean(rewards),
            "adv/min_ttc": (
                float(np.min(finite_ttc)) if len(finite_ttc) else -1.0
            ),
            "adv/ego_hard_brake_events": float(
                np.sum(hard_brake_events[opponent_mask])
            ),
            "adv/fault_collision_events": float(
                np.sum(components["fault_penalty"][opponent_mask] > 0)
            ),
            "adv/ambiguous_collision_events": float(
                np.sum(ambiguous_collision_events[opponent_mask])
            ),
        }

    @staticmethod
    def _validate_shapes(
        num_agents,
        pre_state,
        post_state,
        raw_components,
        agent_offsets,
        role_ids,
    ):
        state_arrays = (
            pre_state.x,
            pre_state.y,
            pre_state.heading,
            pre_state.length,
            pre_state.width,
            pre_state.valid,
            post_state.x,
            post_state.y,
            post_state.heading,
            post_state.length,
            post_state.width,
            post_state.valid,
        )
        if any(len(values) != num_agents for values in state_arrays):
            raise ValueError("state arrays must have one value per agent")
        if role_ids.shape != (num_agents,):
            raise ValueError("role_ids must have shape (num_agents,)")
        if raw_components.shape != (num_agents, 11):
            raise ValueError("raw_components must have shape (num_agents, 11)")
        if (
            agent_offsets.ndim != 1
            or len(agent_offsets) < 2
            or agent_offsets[0] != 0
            or agent_offsets[-1] != num_agents
            or np.any(np.diff(agent_offsets) < 0)
        ):
            raise ValueError("agent_offsets must be monotonic and span agents")
