from dataclasses import dataclass
import math
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
    _DISCRETE_STEER = np.linspace(-1.0, 1.0, 13, dtype=np.float32)
    # Skip far pair evaluations inside a scene; keeps local TTC semantics.
    _MAX_PAIR_DISTANCE_M = 40.0

    def __init__(self, config: RewardConfig, dt: float):
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.config = config
        self.dt = float(dt)
        self._previous_accel = None
        self._previous_steer = None
        self._has_history = False
        self._shapes_validated = False

    def reset(self, num_agents: int) -> None:
        self._previous_accel = np.zeros(num_agents, dtype=np.float32)
        self._previous_steer = np.zeros(num_agents, dtype=np.float32)
        self._has_history = False
        self._shapes_validated = False

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
        collect_metrics: bool = True,
    ) -> RewardEvaluation:
        base_rewards = np.asarray(base_rewards, dtype=np.float32)
        rewards = base_rewards.copy()
        role_ids = np.asarray(role_ids, dtype=np.int64)
        raw_components = np.asarray(raw_components, dtype=np.float32)
        agent_offsets = np.asarray(agent_offsets, dtype=np.int64)
        n = len(base_rewards)
        if not self._shapes_validated:
            self._validate_shapes(
                n,
                pre_state,
                post_state,
                raw_components,
                agent_offsets,
                role_ids,
            )
            self._shapes_validated = True
        if self._previous_accel is None or len(self._previous_accel) != n:
            self.reset(n)
            self._validate_shapes(
                n,
                pre_state,
                post_state,
                raw_components,
                agent_offsets,
                role_ids,
            )
            self._shapes_validated = True

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

        opponent_mask = role_ids == 1
        if np.any(opponent_mask):
            self._fill_opponent_rewards(
                agent_offsets,
                role_ids,
                pre_state,
                post_state,
                pre_speed,
                post_speed,
                accel,
                jerk,
                steer,
                steer_rate,
                raw_components,
                ego_cost,
                fault_penalty,
                kinematics_cost,
                normality,
                min_ttc,
                hard_brake_events,
                ambiguous_collision_events,
            )

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
        np.copyto(self._previous_accel, accel)
        np.copyto(self._previous_steer, steer)
        self._has_history = True

        components = {
            "ego_cost": ego_cost,
            "fault_penalty": fault_penalty,
            "kinematics_cost": kinematics_cost,
            "normality": normality,
            "min_ttc": min_ttc,
        }
        if collect_metrics:
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
        else:
            metrics = {
                "adv/invalid_reward_events": float(invalid_reward_mask.sum()),
            }
        return RewardEvaluation(rewards, components, metrics)

    def _fill_opponent_rewards(
        self,
        agent_offsets,
        role_ids,
        pre_state,
        post_state,
        pre_speed,
        post_speed,
        accel,
        jerk,
        steer,
        steer_rate,
        raw_components,
        ego_cost,
        fault_penalty,
        kinematics_cost,
        normality,
        min_ttc,
        hard_brake_events,
        ambiguous_collision_events,
    ):
        thresholds = self.config.thresholds
        dt = self.dt
        kin_cap = (
            self.config.limits.kinematics_penalty_cap
            / max(self.config.weights.kinematics, 1e-6)
        )
        normal_speed = thresholds.normal_speed_mps
        ttc_threshold = thresholds.ttc_seconds
        safe_distance = thresholds.safe_distance_m
        hard_brake = thresholds.hard_brake_mps2
        hard_steer = thresholds.hard_steer_rad

        # Local references avoid repeated attribute lookups in hot loops.
        pre_x = pre_state.x
        pre_y = pre_state.y
        pre_h = pre_state.heading
        pre_len = pre_state.length
        pre_wid = pre_state.width
        pre_valid = pre_state.valid
        post_x = post_state.x
        post_y = post_state.y
        post_h = post_state.heading
        post_len = post_state.length
        post_valid = post_state.valid

        for start, stop in zip(agent_offsets[:-1], agent_offsets[1:]):
            ego_indices = []
            opponent_indices = []
            for idx in range(int(start), int(stop)):
                if role_ids[idx] == 0:
                    ego_indices.append(idx)
                elif role_ids[idx] == 1:
                    opponent_indices.append(idx)
            if not opponent_indices:
                continue

            for opponent_idx in opponent_indices:
                # Kinematics + normality (scalar, no numpy).
                heading_delta = self._wrap_angle(
                    post_h[opponent_idx] - pre_h[opponent_idx]
                )
                lateral_accel = post_speed[opponent_idx] * heading_delta / dt
                violations = (
                    self._hinge(abs(accel[opponent_idx]), thresholds.max_accel_mps2)
                    + self._hinge(
                        abs(lateral_accel), thresholds.max_lateral_accel_mps2
                    )
                    + self._hinge(abs(jerk[opponent_idx]), thresholds.max_jerk_mps3)
                    + self._hinge(
                        abs(steer_rate[opponent_idx]),
                        thresholds.max_steer_rate_radps,
                    )
                    + max(0.0, float(raw_components[opponent_idx, OFFROAD_COMPONENT]))
                    + max(0.0, float(raw_components[opponent_idx, REVERSE_COMPONENT]))
                    + max(
                        0.0,
                        float(raw_components[opponent_idx, SPEED_LIMIT_COMPONENT]),
                    )
                )
                kinematics_cost[opponent_idx] = min(max(violations, 0.0), kin_cap)
                normality[opponent_idx] = min(
                    max(post_speed[opponent_idx] / normal_speed, 0.0),
                    1.0,
                )

                best_cost = 0.0
                best_ttc = math.inf
                hard_brake_event = 0.0
                ox = float(pre_x[opponent_idx])
                oy = float(pre_y[opponent_idx])
                o_valid = bool(pre_valid[opponent_idx])
                o_len = float(pre_len[opponent_idx])
                o_wid = float(pre_wid[opponent_idx])
                o_speed = float(pre_speed[opponent_idx])
                o_post_x = float(post_x[opponent_idx])
                o_post_y = float(post_y[opponent_idx])
                o_post_valid = bool(post_valid[opponent_idx])
                o_post_speed = float(post_speed[opponent_idx])
                o_post_len = float(post_len[opponent_idx])
                o_h = float(pre_h[opponent_idx])
                o_cos = math.cos(o_h)
                o_sin = math.sin(o_h)
                o_vx = o_speed * o_cos
                o_vy = o_speed * o_sin
                o_post_h = float(post_h[opponent_idx])
                o_post_vx = o_post_speed * math.cos(o_post_h)
                o_post_vy = o_post_speed * math.sin(o_post_h)

                for ego_idx in ego_indices:
                    if not o_valid or not bool(pre_valid[ego_idx]):
                        continue
                    ex = float(pre_x[ego_idx])
                    ey = float(pre_y[ego_idx])
                    dx = ox - ex
                    dy = oy - ey
                    distance = math.hypot(dx, dy)
                    if distance <= 1e-6 or distance > self._MAX_PAIR_DISTANCE_M:
                        continue
                    e_h = float(pre_h[ego_idx])
                    e_cos = math.cos(e_h)
                    e_sin = math.sin(e_h)
                    longitudinal = dx * e_cos + dy * e_sin
                    lateral = e_cos * dy - e_sin * dx
                    lateral_limit = max(float(pre_wid[ego_idx]), o_wid)
                    if longitudinal <= 0.0 or abs(lateral) > lateral_limit:
                        continue
                    e_speed = float(pre_speed[ego_idx])
                    e_vx = e_speed * e_cos
                    e_vy = e_speed * e_sin
                    closing = ((e_vx - o_vx) * dx + (e_vy - o_vy) * dy) / distance
                    bumper = max(
                        distance - 0.5 * (float(pre_len[ego_idx]) + o_len),
                        0.0,
                    )
                    if closing <= 1e-6:
                        pre_ttc = math.inf
                    else:
                        pre_ttc = max(bumper - safe_distance, 0.0) / closing
                    if math.isfinite(pre_ttc):
                        best_ttc = min(best_ttc, pre_ttc)

                    if o_post_valid and bool(post_valid[ego_idx]):
                        pdx = o_post_x - float(post_x[ego_idx])
                        pdy = o_post_y - float(post_y[ego_idx])
                        pdist = math.hypot(pdx, pdy)
                        if pdist <= 1e-6:
                            post_ttc = 0.0
                        else:
                            pe_h = float(post_h[ego_idx])
                            pe_speed = float(post_speed[ego_idx])
                            pe_vx = pe_speed * math.cos(pe_h)
                            pe_vy = pe_speed * math.sin(pe_h)
                            pclosing = (
                                (pe_vx - o_post_vx) * pdx
                                + (pe_vy - o_post_vy) * pdy
                            ) / pdist
                            pbumper = max(
                                pdist
                                - 0.5
                                * (float(post_len[ego_idx]) + o_post_len),
                                0.0,
                            )
                            if pclosing <= 1e-6:
                                post_ttc = math.inf
                            else:
                                post_ttc = (
                                    max(pbumper - safe_distance, 0.0) / pclosing
                                )
                    else:
                        post_ttc = pre_ttc

                    pre_risk = self._ttc_risk(pre_ttc, ttc_threshold)
                    post_risk = self._ttc_risk(post_ttc, ttc_threshold)
                    risk_increase = max(0.0, post_risk - pre_risk)
                    brake = max(
                        0.0,
                        -float(accel[ego_idx]) - hard_brake,
                    ) / hard_brake
                    is_near = min(pre_ttc, post_ttc) <= ttc_threshold
                    if is_near and brake > 0.0:
                        hard_brake_event = 1.0
                    if not is_near:
                        brake = 0.0
                    cost = risk_increase + min(brake, 1.0)
                    best_cost = max(best_cost, min(cost, 1.0))

                ego_cost[opponent_idx] = best_cost
                min_ttc[opponent_idx] = best_ttc
                hard_brake_events[opponent_idx] = hard_brake_event

                if raw_components[opponent_idx, COLLISION_COMPONENT] <= 0:
                    continue
                if (
                    accel[opponent_idx] < -hard_brake
                    or abs(steer[opponent_idx]) > hard_steer
                ):
                    fault_penalty[opponent_idx] = 1.0
                    continue

                # Conservative fault: only clear for high-confidence ego rear-end.
                fault = 1.0
                ambiguous = 1.0
                for ego_idx in ego_indices:
                    if not o_valid or not bool(pre_valid[ego_idx]):
                        continue
                    rx = float(pre_x[ego_idx]) - ox
                    ry = float(pre_y[ego_idx]) - oy
                    distance = math.hypot(rx, ry)
                    if distance <= 1e-6:
                        continue
                    longitudinal = rx * o_cos + ry * o_sin
                    lateral = o_cos * ry - o_sin * rx
                    e_speed = float(pre_speed[ego_idx])
                    e_h = float(pre_h[ego_idx])
                    e_vx = e_speed * math.cos(e_h)
                    e_vy = e_speed * math.sin(e_h)
                    closing = (
                        (e_vx - o_vx) * (-rx) + (e_vy - o_vy) * (-ry)
                    ) / distance
                    max_contact = (
                        0.5 * (float(pre_len[ego_idx]) + o_len) + safe_distance
                    )
                    if (
                        longitudinal < 0.0
                        and abs(lateral) <= max(float(pre_wid[ego_idx]), o_wid)
                        and closing > 0.0
                        and distance <= max_contact
                    ):
                        fault = 0.0
                        ambiguous = 0.0
                        break
                fault_penalty[opponent_idx] = fault
                ambiguous_collision_events[opponent_idx] = ambiguous

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
        rel_x = float(state.x[opponent_idx] - state.x[ego_idx])
        rel_y = float(state.y[opponent_idx] - state.y[ego_idx])
        distance = math.hypot(rel_x, rel_y)
        if distance <= 1e-6:
            return 0.0, 0.0, 0.0
        ego_fx = math.cos(float(state.heading[ego_idx]))
        ego_fy = math.sin(float(state.heading[ego_idx]))
        longitudinal = rel_x * ego_fx + rel_y * ego_fy
        lateral = ego_fx * rel_y - ego_fy * rel_x
        ego_speed = float(speed[ego_idx])
        opp_speed = float(speed[opponent_idx])
        ego_vx = ego_speed * ego_fx
        ego_vy = ego_speed * ego_fy
        opp_h = float(state.heading[opponent_idx])
        opp_vx = opp_speed * math.cos(opp_h)
        opp_vy = opp_speed * math.sin(opp_h)
        closing = ((ego_vx - opp_vx) * rel_x + (ego_vy - opp_vy) * rel_y) / distance
        bumper_distance = max(
            distance
            - 0.5
            * (
                float(state.length[ego_idx])
                + float(state.length[opponent_idx])
            ),
            0.0,
        )
        if closing <= 1e-6:
            return math.inf, longitudinal, lateral
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
        best_ttc = math.inf
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
            if math.isfinite(pre_ttc):
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
        if not math.isfinite(ttc):
            return 0.0
        return min(max((threshold - ttc) / threshold, 0.0), 1.0)

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

        ox = float(pre_state.x[opponent_idx])
        oy = float(pre_state.y[opponent_idx])
        o_h = float(pre_state.heading[opponent_idx])
        o_cos = math.cos(o_h)
        o_sin = math.sin(o_h)
        o_speed = float(pre_speed[opponent_idx])
        o_vx = o_speed * o_cos
        o_vy = o_speed * o_sin
        for ego_idx in ego_indices:
            if not pre_state.valid[ego_idx] or not pre_state.valid[opponent_idx]:
                continue
            rx = float(pre_state.x[ego_idx]) - ox
            ry = float(pre_state.y[ego_idx]) - oy
            distance = math.hypot(rx, ry)
            if distance <= 1e-6:
                continue
            longitudinal = rx * o_cos + ry * o_sin
            lateral = o_cos * ry - o_sin * rx
            e_speed = float(pre_speed[ego_idx])
            e_h = float(pre_state.heading[ego_idx])
            e_vx = e_speed * math.cos(e_h)
            e_vy = e_speed * math.sin(e_h)
            closing = ((e_vx - o_vx) * (-rx) + (e_vy - o_vy) * (-ry)) / distance
            max_contact_distance = (
                0.5
                * (
                    float(pre_state.length[ego_idx])
                    + float(pre_state.length[opponent_idx])
                )
                + self.config.thresholds.safe_distance_m
            )
            if (
                longitudinal < 0
                and abs(lateral)
                <= max(
                    float(pre_state.width[ego_idx]),
                    float(pre_state.width[opponent_idx]),
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
        violations = (
            self._hinge(abs(accel[opponent_idx]), thresholds.max_accel_mps2)
            + self._hinge(
                abs(lateral_accel),
                thresholds.max_lateral_accel_mps2,
            )
            + self._hinge(abs(jerk[opponent_idx]), thresholds.max_jerk_mps3)
            + self._hinge(
                abs(steer_rate[opponent_idx]),
                thresholds.max_steer_rate_radps,
            )
            + max(0.0, float(raw_components[opponent_idx, OFFROAD_COMPONENT]))
            + max(0.0, float(raw_components[opponent_idx, REVERSE_COMPONENT]))
            + max(
                0.0,
                float(raw_components[opponent_idx, SPEED_LIMIT_COMPONENT]),
            )
        )
        max_cost = (
            self.config.limits.kinematics_penalty_cap
            / max(self.config.weights.kinematics, 1e-6)
        )
        return min(max(violations, 0.0), max_cost)

    @staticmethod
    def _hinge(value, limit):
        return max(0.0, float(value) - limit) / max(limit, 1e-6)

    @staticmethod
    def _wrap_angle(angle):
        return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

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
        return AsymmetricRewardEvaluator._DISCRETE_STEER[indices % 13]

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
