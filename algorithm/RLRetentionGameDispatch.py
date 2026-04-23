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
    feature_dim: int = 11
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

    def record_moves(self, slot_idx: int, moved_workers: Iterable[str]) -> None:
        for wid in moved_workers:
            history = self.worker_move_slots.setdefault(wid, deque(maxlen=self.move_history_size))
            history.append(int(slot_idx))

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
) -> Dict[int, Dict[str, float]]:
    demand_profile: Dict[int, Dict[str, float]] = {}
    for rid in region_ids:
        dist_profile = (predicted_distribution or {}).get(rid, {})
        mu = float(max(0.0, dist_profile.get("mu", predicted_demand.get(rid, 0))))
        backlog = float(max(0.0, backlog_counts.get(rid, 0)))
        sigma = float(max(0.0, dist_profile.get("sigma", math.sqrt(max(mu, 1.0)) * 0.20)))
        q90 = float(max(mu, dist_profile.get("q90", mu + sigma)))
        tail_gap = max(0.0, q90 - mu)
        burst = float(np.clip(dist_profile.get("burst_prob", min(1.0, sigma / max(mu + 1.0, 1.0))), 0.0, 1.0))
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
    return demand_profile


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
    )

    desired_workers = {
        rid: int(math.ceil(demand_profile[rid]["effective_demand"] / max_tasks_per_worker))
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

    total_backlog = float(sum(max(0, backlog_counts.get(rid, 0)) for rid in region_ids))
    for rid in region_ids:
        idle_count = available_workers.get(rid, 0)
        base_keep = min(
            idle_count,
            max(
                min_buffer_workers,
                int(math.ceil(desired_workers[rid] * (1.0 + max(0.0, reserve_ratio)))),
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
        if record_transition:
            transitions[rid] = {
                "features": features,
                "probs": probs,
                "action_idx": action_idx,
                "retain_count": retain_count,
                "base_keep": base_keep,
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
                    bid_shortage_weight * float(receiver_need[receiver])
                    + bid_backlog_weight * receiver_profile["backlog"]
                    + bid_debt_weight * receiver_profile["debt"]
                    + bid_burst_weight * receiver_profile["burst_prob"] * max(1.0, receiver_profile["sigma"])
                )
                ask = (
                    ask_shortage_weight * local_gap
                    + ask_uncertainty_weight * donor_profile["sigma"]
                    + move_cost_weight * center_distance / 1000.0
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
        reward_by_region[rid] = (
            served
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
