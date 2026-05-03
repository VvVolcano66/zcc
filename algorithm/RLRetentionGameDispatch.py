import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(1e-6, float(temperature))
    shifted = (logits - np.max(logits)) / temperature
    exp_values = np.exp(shifted)
    total = float(np.sum(exp_values))
    if total <= 0.0:
        return np.ones_like(exp_values) / max(1, exp_values.size)
    return exp_values / total


def _safe_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr))


def _euclidean(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    return float(math.hypot(lon1 - lon2, lat1 - lat2))


def _compute_service_ratio(
    assigned_tasks_by_region: Dict[int, int],
    total_tasks_by_region: Dict[int, int],
) -> Dict[int, float]:
    ratios: Dict[int, float] = {}
    for rid, total_tasks in total_tasks_by_region.items():
        if total_tasks <= 0:
            ratios[rid] = 1.0
        else:
            ratios[rid] = min(1.0, float(assigned_tasks_by_region.get(rid, 0)) / float(total_tasks))
    return ratios


def _compute_pairwise_unfairness(service_ratio: Dict[int, float]) -> Dict[int, float]:
    region_ids = sorted(service_ratio.keys())
    if len(region_ids) <= 1:
        return {rid: 0.0 for rid in region_ids}

    unfairness: Dict[int, float] = {}
    for rid in region_ids:
        disparity = 0.0
        for other in region_ids:
            if other == rid:
                continue
            disparity += abs(service_ratio[rid] - service_ratio[other])
        unfairness[rid] = disparity / max(1, len(region_ids) - 1)
    return unfairness


@dataclass
class RLRetentionBilateralState:
    region_ids: List[int]
    action_ratios: Tuple[float, ...] = (-0.30, -0.15, 0.0, 0.15, 0.30)
    learning_rate: float = 0.03
    temperature: float = 0.85
    exploration_prob: float = 0.12
    service_debt_decay: float = 0.85
    max_service_debt: float = 4.0
    move_history_size: int = 8
    random_seed: int = 42
    feature_dim: int = 13
    prediction_error_decay: float = 0.80
    prediction_bias_clip_ratio: float = 0.75
    prediction_bias_ema: Dict[int, float] = field(default_factory=dict)
    prediction_abs_error_ema: Dict[int, float] = field(default_factory=dict)
    policy_weights: Dict[int, np.ndarray] = field(default_factory=dict)
    reward_baseline: Dict[int, float] = field(default_factory=dict)
    service_debt: Dict[int, float] = field(default_factory=dict)
    worker_move_slots: Dict[str, Deque[int]] = field(default_factory=dict)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)
        action_count = len(self.action_ratios)
        for rid in self.region_ids:
            self.policy_weights.setdefault(rid, np.zeros((action_count, self.feature_dim), dtype=np.float32))
            self.reward_baseline.setdefault(rid, 0.0)
            self.service_debt.setdefault(rid, 0.0)
            self.prediction_bias_ema.setdefault(rid, 0.0)
            self.prediction_abs_error_ema.setdefault(rid, 0.0)

    def record_moves(self, slot_idx: int, moved_workers: Iterable[str]) -> None:
        for wid in moved_workers:
            history = self.worker_move_slots.setdefault(wid, deque(maxlen=self.move_history_size))
            history.append(int(slot_idx))

    def record_prediction_feedback(
        self,
        predicted_region_demand: Dict[int, int],
        actual_region_demand: Dict[int, int],
    ) -> None:
        decay = float(np.clip(self.prediction_error_decay, 0.0, 0.999))
        clip_ratio = max(0.0, float(self.prediction_bias_clip_ratio))
        for rid in self.region_ids:
            pred = float(predicted_region_demand.get(rid, 0.0))
            actual = float(actual_region_demand.get(rid, 0.0))
            err = pred - actual
            scale = max(1.0, pred, actual)
            clipped_err = float(np.clip(err, -clip_ratio * scale, clip_ratio * scale))
            prev_bias = float(self.prediction_bias_ema.get(rid, 0.0))
            prev_abs = float(self.prediction_abs_error_ema.get(rid, 0.0))
            self.prediction_bias_ema[rid] = decay * prev_bias + (1.0 - decay) * clipped_err
            self.prediction_abs_error_ema[rid] = decay * prev_abs + (1.0 - decay) * abs(clipped_err)

    def build_features(
        self,
        region_id: int,
        demand_profile: Dict[int, Dict[str, float]],
        available_workers: Dict[int, int],
        base_keep: int,
        shortage_workers: float,
        neighbor_backlog_pressure: float,
        max_tasks_per_worker: int,
    ) -> np.ndarray:
        profile = demand_profile[region_id]
        scale = max(
            1.0,
            float(profile.get("mu", 0.0) + profile.get("backlog", 0.0)),
            float(available_workers.get(region_id, 0) * max_tasks_per_worker),
            float(profile.get("effective_demand", 0.0)),
        )
        debt = float(self.service_debt.get(region_id, 0.0))
        bias_ema = float(profile.get("bias_ema", 0.0))
        abs_err_ema = float(profile.get("abs_err_ema", 0.0))
        features = np.asarray(
            [
                1.0,
                float(profile.get("mu", 0.0)) / scale,
                float(profile.get("sigma", 0.0)) / scale,
                float(profile.get("tail_gap", 0.0)) / scale,
                float(profile.get("burst_prob", 0.0)),
                float(profile.get("backlog", 0.0)) / scale,
                float(available_workers.get(region_id, 0) * max_tasks_per_worker) / scale,
                float(base_keep * max_tasks_per_worker) / scale,
                float(shortage_workers * max_tasks_per_worker) / scale,
                float(neighbor_backlog_pressure) / scale,
                bias_ema / scale,
                abs_err_ema / scale,
                debt / max(1.0, self.max_service_debt),
            ],
            dtype=np.float32,
        )
        return features

    def sample_action(self, region_id: int, features: np.ndarray) -> Tuple[int, float, np.ndarray]:
        weights = self.policy_weights[region_id]
        logits = weights @ features
        probs = _softmax(logits, self.temperature)
        if self.rng.random() < self.exploration_prob:
            action_idx = int(self.rng.integers(0, len(self.action_ratios)))
        else:
            action_idx = int(self.rng.choice(len(self.action_ratios), p=probs))
        return action_idx, float(self.action_ratios[action_idx]), probs

    def update_policy(
        self,
        transitions: Dict[int, Dict[str, Any]],
        reward_by_region: Dict[int, float],
        total_tasks_by_region: Dict[int, int],
        assigned_tasks_by_region: Dict[int, int],
    ) -> None:
        service_ratio = _compute_service_ratio(assigned_tasks_by_region, total_tasks_by_region)
        for rid, total_tasks in total_tasks_by_region.items():
            prev_debt = float(self.service_debt.get(rid, 0.0))
            if total_tasks <= 0:
                self.service_debt[rid] = max(0.0, prev_debt * self.service_debt_decay)
                continue
            updated = prev_debt * self.service_debt_decay + max(0.0, 1.0 - service_ratio.get(rid, 1.0))
            self.service_debt[rid] = min(self.max_service_debt, updated)

        for rid, transition in transitions.items():
            reward = float(reward_by_region.get(rid, 0.0))
            baseline = float(self.reward_baseline.get(rid, 0.0))
            advantage = reward - baseline
            self.reward_baseline[rid] = 0.9 * baseline + 0.1 * reward

            features = transition["features"]
            probs = transition["probs"]
            chosen_idx = int(transition["action_idx"])
            weights = self.policy_weights[rid]
            for action_idx in range(weights.shape[0]):
                indicator = 1.0 if action_idx == chosen_idx else 0.0
                grad_scale = (indicator - float(probs[action_idx])) * advantage
                weights[action_idx] += self.learning_rate * grad_scale * features

    def imitation_update(
        self,
        region_id: int,
        features: np.ndarray,
        target_action_idx: int,
        strength: float = 1.0,
    ) -> None:
        weights = self.policy_weights[region_id]
        logits = weights @ features
        probs = _softmax(logits, self.temperature)
        for action_idx in range(weights.shape[0]):
            indicator = 1.0 if action_idx == int(target_action_idx) else 0.0
            grad_scale = (indicator - float(probs[action_idx])) * float(strength)
            weights[action_idx] += self.learning_rate * grad_scale * features


@dataclass
class PlatformTaskFirstRLState:
    region_ids: List[int]
    action_profiles: Tuple[Tuple[float, float, float, float, float, float], ...] = (
        (1.30, 1.20, 0.95, 0.82, 1.18, 0.10),
        (1.15, 1.10, 1.00, 0.92, 1.08, 0.18),
        (1.00, 1.00, 1.08, 1.00, 0.98, 0.28),
        (0.92, 0.95, 1.20, 1.10, 0.88, 0.40),
    )
    learning_rate: float = 0.03
    temperature: float = 0.90
    exploration_prob: float = 0.10
    random_seed: int = 7
    feature_dim: int = 9
    reward_baseline: float = 0.0
    completion_rate_ema: float = 0.0
    unfairness_ema: float = 0.0
    policy_weights: np.ndarray = field(init=False)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)
        self.policy_weights = np.zeros((len(self.action_profiles), self.feature_dim), dtype=np.float32)

    def build_features(
        self,
        demand_profile: Dict[int, Dict[str, float]],
        available_workers: Dict[int, int],
        desired_workers: Dict[int, int],
        max_tasks_per_worker: int,
    ) -> np.ndarray:
        total_effective = float(sum(profile.get("effective_demand", 0.0) for profile in demand_profile.values()))
        total_backlog = float(sum(profile.get("backlog", 0.0) for profile in demand_profile.values()))
        mean_sigma = float(np.mean([profile.get("sigma", 0.0) for profile in demand_profile.values()])) if demand_profile else 0.0
        mean_bias = float(np.mean([profile.get("bias_ema", 0.0) for profile in demand_profile.values()])) if demand_profile else 0.0
        mean_abs_error = float(np.mean([profile.get("abs_err_ema", 0.0) for profile in demand_profile.values()])) if demand_profile else 0.0
        total_capacity = float(sum(int(available_workers.get(rid, 0)) * max_tasks_per_worker for rid in self.region_ids))
        total_shortage = float(sum(max(0, desired_workers.get(rid, 0) - int(available_workers.get(rid, 0))) for rid in self.region_ids))
        scale = max(1.0, total_effective + total_backlog, total_capacity)
        features = np.asarray(
            [
                1.0,
                total_effective / scale,
                total_backlog / scale,
                mean_sigma / scale,
                total_capacity / scale,
                total_shortage / max(1.0, total_capacity / max(1, max_tasks_per_worker)),
                mean_bias / scale,
                mean_abs_error / scale,
                float(self.unfairness_ema),
            ],
            dtype=np.float32,
        )
        return features

    def sample_action(self, features: np.ndarray) -> Tuple[int, Tuple[float, float, float, float, float, float], np.ndarray]:
        logits = self.policy_weights @ features
        probs = _softmax(logits, self.temperature)
        if self.rng.random() < self.exploration_prob:
            action_idx = int(self.rng.integers(0, len(self.action_profiles)))
        else:
            action_idx = int(self.rng.choice(len(self.action_profiles), p=probs))
        return action_idx, self.action_profiles[action_idx], probs

    def update_policy(
        self,
        features: np.ndarray,
        probs: np.ndarray,
        action_idx: int,
        reward: float,
        completion_rate: float,
        unfairness: float,
    ) -> None:
        baseline = float(self.reward_baseline)
        advantage = float(reward) - baseline
        self.reward_baseline = 0.9 * baseline + 0.1 * float(reward)
        self.completion_rate_ema = 0.85 * float(self.completion_rate_ema) + 0.15 * float(completion_rate)
        self.unfairness_ema = 0.85 * float(self.unfairness_ema) + 0.15 * float(unfairness)

        for idx in range(self.policy_weights.shape[0]):
            indicator = 1.0 if idx == int(action_idx) else 0.0
            grad_scale = (indicator - float(probs[idx])) * advantage
            self.policy_weights[idx] += self.learning_rate * grad_scale * features


def _build_movable_workers(
    worker_sim,
    centers: Dict[int, Any],
    next_slot_start_seconds: float,
) -> Dict[int, List[str]]:
    movable_workers = {rid: [] for rid in centers.keys()}
    for wid, region_id in worker_sim.worker_center_map.items():
        if region_id not in centers or wid not in worker_sim.worker_positions:
            continue

        status = worker_sim.worker_status.get(wid, "idle")
        busy_until = worker_sim.worker_busy_until.get(wid, 0.0)
        if status == "en_route_to_task" and busy_until > next_slot_start_seconds:
            continue

        movable_workers[region_id].append(wid)
    return movable_workers


def _build_demand_profile(
    region_ids: List[int],
    predicted_demand: Dict[int, int],
    backlog_counts: Dict[int, int],
    state: RLRetentionBilateralState,
    predicted_distribution: Optional[Dict[int, Dict[str, float]]],
    backlog_weight: float,
    uncertainty_weight: float,
    quantile_weight: float,
    burst_weight: float,
    calibration_bias_weight: float,
    calibration_shrink_weight: float,
    calibration_sigma_boost: float,
    calibration_min_scale: float,
) -> Dict[int, Dict[str, float]]:
    demand_profile: Dict[int, Dict[str, float]] = {}
    for rid in region_ids:
        dist_profile = (predicted_distribution or {}).get(rid, {})
        raw_mu = float(max(0.0, dist_profile.get("mu", predicted_demand.get(rid, 0))))
        backlog = float(max(0.0, backlog_counts.get(rid, 0)))
        raw_sigma = float(max(0.0, dist_profile.get("sigma", math.sqrt(max(raw_mu, 1.0)) * 0.20)))
        raw_q90 = float(max(raw_mu, dist_profile.get("q90", raw_mu + raw_sigma)))
        raw_tail_gap = max(0.0, raw_q90 - raw_mu)
        burst = float(np.clip(dist_profile.get("burst_prob", min(1.0, raw_sigma / max(raw_mu + 1.0, 1.0))), 0.0, 1.0))
        debt = float(state.service_debt.get(rid, 0.0))
        bias_ema = float(state.prediction_bias_ema.get(rid, 0.0))
        abs_err_ema = float(state.prediction_abs_error_ema.get(rid, 0.0))
        hist_bias = float(dist_profile.get("hist_bias", 0.0))
        hist_abs_bias = float(dist_profile.get("hist_abs_bias", 0.0))
        combined_bias = 0.60 * hist_bias + 0.40 * bias_ema
        combined_abs_bias = 0.60 * hist_abs_bias + 0.40 * abs_err_ema
        relative_error = combined_abs_bias / max(1.0, raw_mu)

        corrected_mu = max(0.0, raw_mu - calibration_bias_weight * combined_bias)
        shrink_scale = 1.0
        mu = corrected_mu
        sigma = max(
            raw_sigma,
            raw_sigma * (1.0 + calibration_sigma_boost * relative_error),
            0.35 * combined_abs_bias,
        )
        q90 = max(mu, mu + raw_tail_gap)
        tail_gap = max(0.0, q90 - mu)
        base_demand = mu + backlog_weight * backlog
        underpredict_ratio = max(0.0, -combined_bias) / max(1.0, raw_mu)
        overpredict_ratio = max(0.0, combined_bias) / max(1.0, raw_mu)
        risk_scale = float(np.clip(
            0.25
            + 0.90 * underpredict_ratio
            + 0.20 * burst
            - 0.35 * overpredict_ratio,
            0.10,
            1.05,
        ))
        uncertainty_component = (
            uncertainty_weight * sigma
            + quantile_weight * tail_gap
            + burst_weight * burst * max(1.0, sigma)
        )
        capped_uncertainty = min(
            risk_scale * uncertainty_component,
            0.28 * max(1.0, base_demand),
        )
        effective = base_demand + capped_uncertainty
        demand_profile[rid] = {
            "mu": mu,
            "sigma": sigma,
            "q90": q90,
            "tail_gap": tail_gap,
            "burst_prob": burst,
            "backlog": backlog,
            "debt": debt,
            "effective_demand": effective,
            "base_demand": base_demand,
            "risk_scale": risk_scale,
            "capped_uncertainty": capped_uncertainty,
            "raw_mu": raw_mu,
            "hist_bias": hist_bias,
            "hist_abs_bias": hist_abs_bias,
            "bias_ema": bias_ema,
            "combined_bias": combined_bias,
            "abs_err_ema": abs_err_ema,
            "shrink_scale": shrink_scale,
        }
    return demand_profile


def _normalize_worker_targets(
    region_ids: List[int],
    demand_profile: Dict[int, Dict[str, float]],
    raw_desired_workers: Dict[int, int],
    total_available_workers: int,
) -> Dict[int, int]:
    total_available_workers = max(0, int(total_available_workers))
    total_raw_desired = int(sum(max(0, int(raw_desired_workers.get(rid, 0))) for rid in region_ids))
    if total_available_workers <= 0 or total_raw_desired <= total_available_workers:
        return {rid: max(0, int(raw_desired_workers.get(rid, 0))) for rid in region_ids}

    score_basis = {}
    for rid in region_ids:
        profile = demand_profile[rid]
        score_basis[rid] = max(
            1e-6,
            profile["effective_demand"] * (1.0 + 0.15 * profile["burst_prob"] + 0.10 * profile["debt"])
        )
    score_sum = float(sum(score_basis.values()))
    if score_sum <= 0.0:
        score_sum = float(len(region_ids))

    normalized_targets = {}
    fractional_targets = []
    allocated = 0
    for rid in region_ids:
        target_float = total_available_workers * score_basis[rid] / score_sum
        target_floor = int(math.floor(target_float))
        capped_floor = min(max(0, int(raw_desired_workers.get(rid, 0))), target_floor)
        normalized_targets[rid] = capped_floor
        allocated += capped_floor
        fractional_targets.append((target_float - target_floor, rid))

    remaining = max(0, total_available_workers - allocated)
    for _, rid in sorted(fractional_targets, reverse=True):
        if remaining <= 0:
            break
        raw_target = max(0, int(raw_desired_workers.get(rid, 0)))
        if normalized_targets[rid] >= raw_target:
            continue
        normalized_targets[rid] += 1
        remaining -= 1

    return normalized_targets


def _select_retained_and_releasable_workers(
    region_id: int,
    worker_ids: List[str],
    retain_count: int,
    shortage_receivers: List[int],
    worker_sim,
    centers: Dict[int, Any],
    state: RLRetentionBilateralState,
) -> Tuple[List[str], List[str]]:
    if not worker_ids:
        return [], []

    donor_center = centers[region_id]
    donor_center_info = worker_sim.worker_positions.get(worker_ids[0])
    donor_center_lon = donor_center_lat = None
    scored_workers: List[Tuple[float, str]] = []
    for wid in worker_ids:
        worker_node, worker_lon, worker_lat = worker_sim.worker_positions[wid]
        if donor_center_lon is None:
            donor_center_lon = worker_lon
            donor_center_lat = worker_lat

        donor_anchor = worker_sim.G.nodes[donor_center]
        donor_lon = donor_anchor.get("x", donor_anchor.get("lon"))
        donor_lat = donor_anchor.get("y", donor_anchor.get("lat"))
        donor_distance = _euclidean(worker_lon, worker_lat, donor_lon, donor_lat)

        receiver_gain = 0.0
        for receiver in shortage_receivers:
            if receiver == region_id:
                continue
            receiver_anchor = worker_sim.G.nodes[centers[receiver]]
            receiver_lon = receiver_anchor.get("x", receiver_anchor.get("lon"))
            receiver_lat = receiver_anchor.get("y", receiver_anchor.get("lat"))
            receiver_gain = max(receiver_gain, max(0.0, donor_distance - _euclidean(worker_lon, worker_lat, receiver_lon, receiver_lat)))

        move_history = state.worker_move_slots.get(wid)
        recent_move_penalty = 0.15 * len(move_history) if move_history is not None else 0.0
        keep_score = -1.0 * donor_distance - 0.35 * receiver_gain + 0.20 * recent_move_penalty
        scored_workers.append((keep_score, wid))

    scored_workers.sort(reverse=True)
    retained = [wid for _, wid in scored_workers[:retain_count]]
    releasable = [wid for _, wid in scored_workers[retain_count:]]
    return retained, releasable


def _choose_worker_for_receiver(
    donor_region: int,
    receiver_region: int,
    releasable_workers: List[str],
    worker_sim,
    centers: Dict[int, Any],
    state: RLRetentionBilateralState,
    candidate_k: int,
) -> Optional[str]:
    if not releasable_workers:
        return None

    receiver_anchor = worker_sim.G.nodes[centers[receiver_region]]
    receiver_lon = receiver_anchor.get("x", receiver_anchor.get("lon"))
    receiver_lat = receiver_anchor.get("y", receiver_anchor.get("lat"))

    donor_anchor = worker_sim.G.nodes[centers[donor_region]]
    donor_lon = donor_anchor.get("x", donor_anchor.get("lon"))
    donor_lat = donor_anchor.get("y", donor_anchor.get("lat"))

    ranked: List[Tuple[float, str]] = []
    for wid in releasable_workers:
        _, worker_lon, worker_lat = worker_sim.worker_positions[wid]
        receiver_distance = _euclidean(worker_lon, worker_lat, receiver_lon, receiver_lat)
        donor_distance = _euclidean(worker_lon, worker_lat, donor_lon, donor_lat)
        move_history = state.worker_move_slots.get(wid)
        move_penalty = 0.12 * len(move_history) if move_history is not None else 0.0
        score = receiver_distance - 0.25 * donor_distance + move_penalty
        ranked.append((score, wid))

    ranked.sort()
    top_candidates = ranked[:max(1, int(candidate_k))]
    return top_candidates[0][1] if top_candidates else None


def sample_platform_task_first_control(
    region_ids: List[int],
    predicted_demand: Dict[int, int],
    backlog_counts: Dict[int, int],
    available_workers: Dict[int, int],
    max_tasks_per_worker: int,
    retention_state: RLRetentionBilateralState,
    platform_state: PlatformTaskFirstRLState,
    predicted_distribution: Optional[Dict[int, Dict[str, float]]],
    backlog_weight: float,
    uncertainty_weight: float,
    quantile_weight: float,
    burst_weight: float,
    calibration_bias_weight: float,
    calibration_shrink_weight: float,
    calibration_sigma_boost: float,
    calibration_min_scale: float,
    base_platform_task_weight: float,
    base_platform_gap_weight: float,
    base_platform_release_credit_weight: float,
) -> Dict[str, Any]:
    demand_profile = _build_demand_profile(
        region_ids=region_ids,
        predicted_demand=predicted_demand,
        backlog_counts=backlog_counts,
        state=retention_state,
        predicted_distribution=predicted_distribution,
        backlog_weight=backlog_weight,
        uncertainty_weight=uncertainty_weight,
        quantile_weight=quantile_weight,
        burst_weight=burst_weight,
        calibration_bias_weight=calibration_bias_weight,
        calibration_shrink_weight=calibration_shrink_weight,
        calibration_sigma_boost=calibration_sigma_boost,
        calibration_min_scale=calibration_min_scale,
    )
    raw_desired_workers = {
        rid: int(math.ceil(demand_profile[rid]["effective_demand"] / max(1, int(max_tasks_per_worker))))
        for rid in region_ids
    }
    desired_workers = _normalize_worker_targets(
        region_ids=region_ids,
        demand_profile=demand_profile,
        raw_desired_workers=raw_desired_workers,
        total_available_workers=int(sum(max(0, int(available_workers.get(rid, 0))) for rid in region_ids)),
    )
    desired_workers = {
        rid: max(
            int(math.ceil(demand_profile[rid]["backlog"] / max(1, int(max_tasks_per_worker)))),
            desired_workers[rid] + max(
                0,
                int(math.ceil(max(0.0, -demand_profile[rid].get("combined_bias", 0.0)) / max(1, int(max_tasks_per_worker)) * 0.5)),
            ),
        )
        for rid in region_ids
    }
    features = platform_state.build_features(
        demand_profile=demand_profile,
        available_workers=available_workers,
        desired_workers=desired_workers,
        max_tasks_per_worker=max_tasks_per_worker,
    )
    action_idx, action_profile, probs = platform_state.sample_action(features)
    task_scale, gap_scale, release_scale, keep_scale, need_scale, fairness_weight = action_profile
    return {
        "features": features,
        "probs": probs,
        "action_idx": action_idx,
        "task_weight": float(base_platform_task_weight * task_scale),
        "gap_weight": float(base_platform_gap_weight * gap_scale),
        "release_credit_weight": float(base_platform_release_credit_weight * release_scale),
        "keep_scale": float(keep_scale),
        "need_scale": float(need_scale),
        "fairness_weight": float(fairness_weight),
        "desired_workers": desired_workers,
        "demand_profile": demand_profile,
    }


def rl_retention_bilateral_predispatch_workers(
    G: nx.Graph,
    worker_sim,
    centers: Dict[int, Any],
    predicted_demand: Dict[int, int],
    state: RLRetentionBilateralState,
    slot_idx: int,
    next_slot_start_seconds: float,
    predicted_distribution: Optional[Dict[int, Dict[str, float]]] = None,
    max_tasks_per_worker: int = 4,
    backlog_counts: Optional[Dict[int, int]] = None,
    backlog_weight: float = 1.0,
    uncertainty_weight: float = 0.45,
    quantile_weight: float = 0.55,
    burst_weight: float = 1.2,
    calibration_bias_weight: float = 0.60,
    calibration_shrink_weight: float = 0.55,
    calibration_sigma_boost: float = 0.75,
    calibration_min_scale: float = 0.55,
    platform_task_weight: float = 0.30,
    platform_gap_weight: float = 0.55,
    platform_release_credit_weight: float = 0.35,
    platform_keep_scale: float = 1.0,
    platform_need_scale: float = 1.0,
    center_local_task_weight: float = 1.0,
    worker_completion_bonus: float = 0.20,
    worker_distance_penalty: float = 0.015,
    same_worker_chain_bonus: float = 0.08,
    min_buffer_workers: int = 1,
    reserve_ratio: float = 0.10,
    bid_shortage_weight: float = 0.90,
    bid_backlog_weight: float = 0.60,
    bid_debt_weight: float = 0.75,
    bid_burst_weight: float = 0.40,
    ask_shortage_weight: float = 0.85,
    ask_uncertainty_weight: float = 0.55,
    hoard_discount_weight: float = 0.40,
    move_cost_weight: float = 0.02,
    distance_penalty: float = 0.003,
    candidate_k: int = 12,
    edge_epsilon: float = 0.05,
    record_transition: bool = True,
) -> Dict[str, Any]:
    max_tasks_per_worker = max(1, int(max_tasks_per_worker))
    backlog_counts = backlog_counts or {}
    region_ids = sorted(centers.keys())
    movable_workers = _build_movable_workers(worker_sim, centers, next_slot_start_seconds)
    available_workers = {rid: len(movable_workers[rid]) for rid in region_ids}

    demand_profile = _build_demand_profile(
        region_ids=region_ids,
        predicted_demand=predicted_demand,
        backlog_counts=backlog_counts,
        state=state,
        predicted_distribution=predicted_distribution,
        backlog_weight=backlog_weight,
        uncertainty_weight=uncertainty_weight,
        quantile_weight=quantile_weight,
        burst_weight=burst_weight,
        calibration_bias_weight=calibration_bias_weight,
        calibration_shrink_weight=calibration_shrink_weight,
        calibration_sigma_boost=calibration_sigma_boost,
        calibration_min_scale=calibration_min_scale,
    )

    raw_desired_workers = {
        rid: int(math.ceil(demand_profile[rid]["effective_demand"] / max_tasks_per_worker))
        for rid in region_ids
    }
    normalized_desired_workers = _normalize_worker_targets(
        region_ids=region_ids,
        demand_profile=demand_profile,
        raw_desired_workers=raw_desired_workers,
        total_available_workers=int(sum(available_workers.values())),
    )
    stabilized_desired_workers = {
        rid: max(
            int(math.ceil(demand_profile[rid]["backlog"] / max_tasks_per_worker)),
            normalized_desired_workers[rid] + max(
                0,
                int(math.ceil(max(0.0, -demand_profile[rid].get("combined_bias", 0.0)) / max_tasks_per_worker * 0.5)),
            ),
        )
        for rid in region_ids
    }
    desired_workers = {
        rid: max(
            0,
            int(math.ceil(stabilized_desired_workers[rid] * max(0.65, float(platform_need_scale)))),
        )
        for rid in region_ids
    }
    shortage_guess = {
        rid: max(0.0, float(desired_workers[rid] - available_workers.get(rid, 0)))
        for rid in region_ids
    }

    transitions: Dict[int, Dict[str, Any]] = {}
    retained_workers_by_region: Dict[int, List[str]] = {rid: [] for rid in region_ids}
    releasable_workers_by_region: Dict[int, List[str]] = {rid: [] for rid in region_ids}
    retain_count_by_region: Dict[int, int] = {}
    safe_reserve_by_region: Dict[int, int] = {}
    hoard_penalty_by_region: Dict[int, float] = {}
    action_ratio_by_region: Dict[int, float] = {}
    action_index_by_region: Dict[int, int] = {}
    platform_reward_weight_by_region: Dict[int, float] = {}
    platform_release_credit_by_region: Dict[int, float] = {}

    avg_effective_demand = max(
        1.0,
        float(np.mean([demand_profile[rid]["effective_demand"] for rid in region_ids])) if region_ids else 1.0,
    )
    total_shortage_guess = max(1.0, float(sum(shortage_guess.values())))

    total_backlog = float(sum(max(0, backlog_counts.get(rid, 0)) for rid in region_ids))
    for rid in region_ids:
        idle_count = available_workers.get(rid, 0)
        bias_keep_bonus = max(
            0,
            int(math.ceil(max(0.0, -demand_profile[rid].get("combined_bias", 0.0)) / max_tasks_per_worker * 0.5)),
        )
        base_keep = min(
            idle_count,
            max(
                min_buffer_workers,
                int(math.ceil(desired_workers[rid] * (1.0 + max(0.0, reserve_ratio)))) + bias_keep_bonus,
            ),
        )
        keep_scale_effective = max(0.75, float(platform_keep_scale))
        base_keep = min(
            idle_count,
            max(
                min_buffer_workers,
                int(round(base_keep * keep_scale_effective)),
            ),
        )
        safe_reserve_by_region[rid] = base_keep
        neighbor_backlog_pressure = max(0.0, total_backlog - float(backlog_counts.get(rid, 0)))
        features = state.build_features(
            region_id=rid,
            demand_profile=demand_profile,
            available_workers=available_workers,
            base_keep=base_keep,
            shortage_workers=shortage_guess[rid],
            neighbor_backlog_pressure=neighbor_backlog_pressure,
            max_tasks_per_worker=max_tasks_per_worker,
        )
        action_idx, action_ratio, probs = state.sample_action(rid, features)
        delta_keep = int(round(action_ratio * max(1, idle_count)))
        retain_count = int(np.clip(base_keep + delta_keep, 0, idle_count))
        retain_count_by_region[rid] = retain_count
        action_ratio_by_region[rid] = action_ratio
        action_index_by_region[rid] = action_idx
        hoard_penalty_by_region[rid] = max(0.0, retain_count - base_keep) * neighbor_backlog_pressure
        platform_reward_weight_by_region[rid] = float(np.clip(
            1.0
            + platform_task_weight * (demand_profile[rid]["effective_demand"] / avg_effective_demand - 1.0)
            + platform_gap_weight * (shortage_guess[rid] / total_shortage_guess),
            0.70,
            2.50,
        ))
        platform_release_credit_by_region[rid] = platform_release_credit_weight * max(
            0.0,
            idle_count - retain_count,
        )
        if record_transition:
            transitions[rid] = {
                "features": features,
                "probs": probs,
                "action_idx": action_idx,
                "retain_count": retain_count,
                "base_keep": base_keep,
                "platform_reward_weight": platform_reward_weight_by_region[rid],
            }

    shortage_receivers = [rid for rid in region_ids if max(0, desired_workers[rid] - retain_count_by_region[rid]) > 0]
    for rid in region_ids:
        retained, releasable = _select_retained_and_releasable_workers(
            region_id=rid,
            worker_ids=movable_workers[rid],
            retain_count=retain_count_by_region[rid],
            shortage_receivers=shortage_receivers,
            worker_sim=worker_sim,
            centers=centers,
            state=state,
        )
        retained_workers_by_region[rid] = retained
        releasable_workers_by_region[rid] = releasable

    receiver_need = {
        rid: max(0, desired_workers[rid] - retain_count_by_region[rid])
        for rid in region_ids
    }
    donor_supply = {
        rid: len(releasable_workers_by_region[rid])
        for rid in region_ids
    }
    moves: List[Dict[str, Any]] = []
    move_cost_by_region = {rid: 0.0 for rid in region_ids}
    center_distance_cache: Dict[Tuple[Any, Any], float] = {}

    def get_center_distance(donor_region: int, receiver_region: int) -> float:
        donor_node = centers[donor_region]
        receiver_node = centers[receiver_region]
        cache_key = (donor_node, receiver_node)
        if cache_key in center_distance_cache:
            return center_distance_cache[cache_key]
        try:
            distance = float(nx.shortest_path_length(G, source=donor_node, target=receiver_node, weight="length"))
        except nx.NetworkXNoPath:
            distance = float("inf")
        center_distance_cache[cache_key] = distance
        return distance

    while True:
        best_edge = None
        best_gain = edge_epsilon
        for donor in region_ids:
            if donor_supply[donor] <= 0:
                continue
            local_gap = max(0.0, float(desired_workers[donor] - retain_count_by_region[donor]))
            for receiver in region_ids:
                if receiver == donor or receiver_need[receiver] <= 0:
                    continue
                center_distance = get_center_distance(donor, receiver)
                if not np.isfinite(center_distance):
                    continue

                receiver_profile = demand_profile[receiver]
                donor_profile = demand_profile[donor]
                bid = (
                    center_local_task_weight * float(receiver_need[receiver])
                    + platform_reward_weight_by_region[receiver]
                    + bid_shortage_weight * float(receiver_need[receiver])
                    + bid_backlog_weight * receiver_profile["backlog"]
                    + bid_debt_weight * receiver_profile["debt"]
                    + bid_burst_weight * receiver_profile["burst_prob"] * max(1.0, receiver_profile["sigma"])
                )
                ask = (
                    center_local_task_weight * local_gap
                    + ask_shortage_weight * local_gap
                    + ask_uncertainty_weight * donor_profile["sigma"]
                    + move_cost_weight * center_distance / 1000.0
                    - platform_release_credit_by_region.get(donor, 0.0)
                    - hoard_discount_weight * hoard_penalty_by_region.get(donor, 0.0)
                )
                gain = bid - ask - distance_penalty * center_distance / 1000.0
                if gain > best_gain:
                    best_gain = gain
                    best_edge = (donor, receiver, center_distance)

        if best_edge is None:
            break

        donor, receiver, center_distance = best_edge
        chosen_worker = _choose_worker_for_receiver(
            donor_region=donor,
            receiver_region=receiver,
            releasable_workers=releasable_workers_by_region[donor],
            worker_sim=worker_sim,
            centers=centers,
            state=state,
            candidate_k=candidate_k,
        )
        if chosen_worker is None:
            donor_supply[donor] = 0
            continue

        releasable_workers_by_region[donor].remove(chosen_worker)
        donor_supply[donor] -= 1
        receiver_need[receiver] = max(0, receiver_need[receiver] - 1)
        move_cost_by_region[donor] += center_distance / 1000.0
        worker_sim.worker_center_map[chosen_worker] = receiver
        moves.append(
            {
                "wid": chosen_worker,
                "from_region": donor,
                "to_region": receiver,
                "distance_m": center_distance,
            }
        )

    if moves:
        state.record_moves(slot_idx=slot_idx, moved_workers=[move["wid"] for move in moves])

    diagnostics = {
        "retain_count": retain_count_by_region,
        "safe_reserve": safe_reserve_by_region,
        "receiver_need_after_trade": receiver_need,
        "donor_supply_after_trade": donor_supply,
        "hoard_penalty": hoard_penalty_by_region,
        "action_ratio": action_ratio_by_region,
        "action_index": action_index_by_region,
        "platform_reward_weight": platform_reward_weight_by_region,
        "platform_release_credit": platform_release_credit_by_region,
    }

    return {
        "moves": moves,
        "available_workers": available_workers,
        "desired_workers": desired_workers,
        "retain_count": retain_count_by_region,
        "retained_workers": retained_workers_by_region,
        "releasable_workers": releasable_workers_by_region,
        "demand_profile": demand_profile,
        "effective_demand": {rid: demand_profile[rid]["effective_demand"] for rid in region_ids},
        "hoard_penalty": hoard_penalty_by_region,
        "move_cost_by_region": move_cost_by_region,
        "transitions": transitions,
        "diagnostics": diagnostics,
        "stackelberg_control": {
            "region_priority_weight": platform_reward_weight_by_region,
            "worker_completion_bonus": float(worker_completion_bonus),
            "worker_distance_penalty": float(worker_distance_penalty),
            "same_worker_chain_bonus": float(same_worker_chain_bonus),
            "platform_release_credit": platform_release_credit_by_region,
        },
    }


def update_rl_retention_bilateral_state(
    state: RLRetentionBilateralState,
    transitions: Dict[int, Dict[str, Any]],
    assigned_tasks_by_region: Dict[int, int],
    total_tasks_by_region: Dict[int, int],
    hoard_penalty_by_region: Dict[int, float],
    move_cost_by_region: Dict[int, float],
    hoard_penalty_weight: float = 0.02,
    move_cost_weight: float = 0.08,
    unfairness_weight: float = 1.0,
) -> Dict[int, float]:
    service_ratio = _compute_service_ratio(assigned_tasks_by_region, total_tasks_by_region)
    unfairness_by_region = _compute_pairwise_unfairness(service_ratio)
    reward_by_region: Dict[int, float] = {}
    for rid in state.region_ids:
        served = float(assigned_tasks_by_region.get(rid, 0))
        platform_reward_weight = float(transitions.get(rid, {}).get("platform_reward_weight", 1.0))
        reward_by_region[rid] = (
            platform_reward_weight * served
            - hoard_penalty_weight * float(hoard_penalty_by_region.get(rid, 0.0))
            - move_cost_weight * float(move_cost_by_region.get(rid, 0.0))
            - unfairness_weight * float(unfairness_by_region.get(rid, 0.0))
        )
    state.update_policy(
        transitions=transitions,
        reward_by_region=reward_by_region,
        total_tasks_by_region=total_tasks_by_region,
        assigned_tasks_by_region=assigned_tasks_by_region,
    )
    return reward_by_region


def update_platform_task_first_state(
    state: PlatformTaskFirstRLState,
    transition: Dict[str, Any],
    assigned_tasks_by_region: Dict[int, int],
    total_tasks_by_region: Dict[int, int],
    fairness_secondary_weight: float = 0.20,
) -> Dict[str, float]:
    total_assigned = float(sum(assigned_tasks_by_region.values()))
    total_tasks = float(sum(total_tasks_by_region.values()))
    completion_rate = (total_assigned / total_tasks) if total_tasks > 0 else 0.0
    service_ratio = _compute_service_ratio(assigned_tasks_by_region, total_tasks_by_region)
    unfairness_by_region = _compute_pairwise_unfairness(service_ratio)
    mean_unfairness = float(np.mean(list(unfairness_by_region.values()))) if unfairness_by_region else 0.0
    reward = total_assigned - float(fairness_secondary_weight) * mean_unfairness
    state.update_policy(
        features=transition["features"],
        probs=transition["probs"],
        action_idx=int(transition["action_idx"]),
        reward=reward,
        completion_rate=completion_rate,
        unfairness=mean_unfairness,
    )
    return {
        "platform_reward": reward,
        "completion_rate": completion_rate,
        "mean_unfairness": mean_unfairness,
    }


def offline_warm_start_retention_policy(
    state: RLRetentionBilateralState,
    historical_samples: List[Dict[str, Any]],
    max_tasks_per_worker: int = 4,
    min_buffer_workers: int = 1,
    reserve_ratio: float = 0.10,
    backlog_weight: float = 1.0,
    uncertainty_weight: float = 0.45,
    quantile_weight: float = 0.55,
    burst_weight: float = 1.2,
    epochs: int = 3,
) -> Dict[str, Any]:
    if not historical_samples:
        return {"sample_count": 0, "epochs": 0, "action_histogram": {}}

    action_histogram = {idx: 0 for idx in range(len(state.action_ratios))}
    max_tasks_per_worker = max(1, int(max_tasks_per_worker))
    epochs = max(1, int(epochs))

    for _ in range(epochs):
        state.service_debt = {rid: 0.0 for rid in state.region_ids}
        for sample in historical_samples:
            available_workers = {
                int(rid): int(count)
                for rid, count in sample.get("available_workers", {}).items()
            }
            backlog_counts = {
                int(rid): int(count)
                for rid, count in sample.get("backlog_counts", {}).items()
            }
            actual_counts = {
                int(rid): float(count)
                for rid, count in sample.get("actual_counts", {}).items()
            }
            sigma_map = {
                int(rid): float(value)
                for rid, value in sample.get("sigma_map", {}).items()
            }
            q90_map = {
                int(rid): float(value)
                for rid, value in sample.get("q90_map", {}).items()
            }
            burst_map = {
                int(rid): float(value)
                for rid, value in sample.get("burst_map", {}).items()
            }

            demand_profile: Dict[int, Dict[str, float]] = {}
            desired_workers: Dict[int, int] = {}
            base_keep_by_region: Dict[int, int] = {}

            for rid in state.region_ids:
                mu = max(0.0, actual_counts.get(rid, 0.0))
                backlog = max(0.0, float(backlog_counts.get(rid, 0)))
                sigma = max(0.0, sigma_map.get(rid, math.sqrt(max(mu, 1.0)) * 0.20))
                q90 = max(mu, q90_map.get(rid, mu + sigma))
                tail_gap = max(0.0, q90 - mu)
                burst = float(np.clip(burst_map.get(rid, 0.0), 0.0, 1.0))
                debt = float(state.service_debt.get(rid, 0.0))
                effective = (
                    mu
                    + backlog_weight * backlog
                    + uncertainty_weight * sigma
                    + quantile_weight * tail_gap
                    + burst_weight * burst * max(1.0, sigma)
                )
                demand_profile[rid] = {
                    "mu": mu,
                    "sigma": sigma,
                    "q90": q90,
                    "tail_gap": tail_gap,
                    "burst_prob": burst,
                    "backlog": backlog,
                    "debt": debt,
                    "effective_demand": effective,
                }
                desired_workers[rid] = int(math.ceil(effective / max_tasks_per_worker))
                idle_count = int(available_workers.get(rid, 0))
                base_keep_by_region[rid] = min(
                    idle_count,
                    max(
                        min_buffer_workers,
                        int(math.ceil(desired_workers[rid] * (1.0 + max(0.0, reserve_ratio)))),
                    ),
                )

            total_shortage = sum(
                max(0, desired_workers[rid] - base_keep_by_region[rid])
                for rid in state.region_ids
            )

            served_proxy_by_region: Dict[int, int] = {}
            total_tasks_by_region: Dict[int, int] = {}
            for rid in state.region_ids:
                idle_count = int(available_workers.get(rid, 0))
                base_keep = int(base_keep_by_region[rid])
                local_shortage = max(0, desired_workers[rid] - base_keep)
                other_shortage = max(0, total_shortage - local_shortage)
                neighbor_backlog_pressure = max(
                    0.0,
                    float(sum(backlog_counts.get(other, 0) for other in state.region_ids if other != rid))
                )

                features = state.build_features(
                    region_id=rid,
                    demand_profile=demand_profile,
                    available_workers=available_workers,
                    base_keep=base_keep,
                    shortage_workers=float(local_shortage),
                    neighbor_backlog_pressure=neighbor_backlog_pressure,
                    max_tasks_per_worker=max_tasks_per_worker,
                )

                target_ratio = 0.0
                if idle_count > 0:
                    shortage_ratio = float(local_shortage) / max(1.0, float(idle_count))
                    slack_workers = max(0, idle_count - base_keep)
                    if local_shortage > 0:
                        if (
                            shortage_ratio >= 0.25
                            or demand_profile[rid]["burst_prob"] >= 0.35
                            or demand_profile[rid]["debt"] >= 1.0
                        ):
                            target_ratio = 0.30
                        else:
                            target_ratio = 0.15
                    elif slack_workers > 0 and other_shortage > 0:
                        release_ratio = float(min(slack_workers, other_shortage)) / max(1.0, float(idle_count))
                        if release_ratio >= 0.25:
                            target_ratio = -0.30
                        else:
                            target_ratio = -0.15

                action_idx = int(np.argmin(np.abs(np.asarray(state.action_ratios) - target_ratio)))
                action_histogram[action_idx] += 1
                confidence = 1.0 + abs(target_ratio) * 2.0
                state.imitation_update(
                    region_id=rid,
                    features=features,
                    target_action_idx=action_idx,
                    strength=confidence,
                )

                retain_count = int(np.clip(
                    base_keep + round(float(state.action_ratios[action_idx]) * max(1, idle_count)),
                    0,
                    idle_count,
                ))
                served_proxy_by_region[rid] = min(
                    int(round(demand_profile[rid]["mu"] + demand_profile[rid]["backlog"])),
                    retain_count * max_tasks_per_worker,
                )
                total_tasks_by_region[rid] = max(
                    0,
                    int(round(demand_profile[rid]["mu"] + demand_profile[rid]["backlog"]))
                )

            service_ratio = _compute_service_ratio(served_proxy_by_region, total_tasks_by_region)
            for rid in state.region_ids:
                prev_debt = float(state.service_debt.get(rid, 0.0))
                total_tasks = total_tasks_by_region.get(rid, 0)
                if total_tasks <= 0:
                    state.service_debt[rid] = max(0.0, prev_debt * state.service_debt_decay)
                else:
                    updated = prev_debt * state.service_debt_decay + max(0.0, 1.0 - service_ratio.get(rid, 1.0))
                    state.service_debt[rid] = min(state.max_service_debt, updated)

    return {
        "sample_count": len(historical_samples),
        "epochs": epochs,
        "action_histogram": action_histogram,
    }
