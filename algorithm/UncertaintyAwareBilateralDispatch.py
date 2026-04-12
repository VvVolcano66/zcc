import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import KDTree


def _safe_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr))


def _safe_quantile(values: Iterable[float], q: float, fallback: float) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return float(fallback)
    return float(np.quantile(arr, q))


def _compute_service_ratio(
    worker_counts: Dict[int, int],
    effective_demand: Dict[int, float],
    max_tasks_per_worker: int,
) -> Dict[int, float]:
    ratios: Dict[int, float] = {}
    for rid, demand in effective_demand.items():
        if demand <= 0.0:
            ratios[rid] = 1.0
            continue
        capacity = float(worker_counts.get(rid, 0) * max_tasks_per_worker)
        ratios[rid] = min(1.0, capacity / max(demand, 1.0))
    return ratios


def _compute_weighted_unfairness(
    service_ratio: Dict[int, float],
    effective_demand: Dict[int, float],
    burst_prob: Dict[int, float],
) -> float:
    total_effective = max(1.0, float(sum(max(v, 0.0) for v in effective_demand.values())))
    unfairness = 0.0
    for rid, ratio in service_ratio.items():
        demand_weight = max(0.0, effective_demand.get(rid, 0.0)) / total_effective
        risk_weight = 0.5 + 0.5 * float(np.clip(burst_prob.get(rid, 0.0), 0.0, 1.0))
        unfairness += demand_weight * risk_weight * ((1.0 - ratio) ** 2)
    return float(unfairness)


@dataclass
class UncertaintyAwareBilateralState:
    region_ids: List[int]
    history_size: int = 12
    service_debt_decay: float = 0.85
    max_service_debt: float = 4.0
    move_history_size: int = 8
    prediction_errors: Dict[int, Deque[float]] = field(default_factory=dict)
    arrival_history: Dict[int, Deque[float]] = field(default_factory=dict)
    service_debt: Dict[int, float] = field(default_factory=dict)
    worker_move_slots: Dict[str, Deque[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for rid in self.region_ids:
            self.prediction_errors.setdefault(rid, deque(maxlen=self.history_size))
            self.arrival_history.setdefault(rid, deque(maxlen=self.history_size))
            self.service_debt.setdefault(rid, 0.0)

    def record_prediction_feedback(
        self,
        predicted_region_demand: Optional[Dict[int, int]],
        actual_region_demand: Dict[int, int],
    ) -> None:
        if predicted_region_demand is None:
            return
        for rid in self.region_ids:
            pred = float(predicted_region_demand.get(rid, 0.0))
            actual = float(actual_region_demand.get(rid, 0.0))
            self.prediction_errors[rid].append(actual - pred)
            self.arrival_history[rid].append(actual)

    def record_service_outcome(
        self,
        total_tasks_by_region: Dict[int, int],
        assigned_tasks_by_region: Dict[int, int],
    ) -> None:
        for rid in self.region_ids:
            total_tasks = float(total_tasks_by_region.get(rid, 0))
            prev_debt = float(self.service_debt.get(rid, 0.0))
            if total_tasks <= 0.0:
                self.service_debt[rid] = max(0.0, prev_debt * self.service_debt_decay)
                continue

            served = float(assigned_tasks_by_region.get(rid, 0))
            service_ratio = min(1.0, served / max(total_tasks, 1.0))
            updated = prev_debt * self.service_debt_decay + max(0.0, 1.0 - service_ratio)
            self.service_debt[rid] = min(self.max_service_debt, updated)

    def record_moves(self, slot_idx: int, moved_workers: Iterable[str]) -> None:
        for wid in moved_workers:
            history = self.worker_move_slots.setdefault(wid, deque(maxlen=self.move_history_size))
            history.append(int(slot_idx))


def uncertainty_aware_bilateral_predispatch_workers(
    G: nx.Graph,
    worker_sim,
    centers: Dict[int, Any],
    predicted_demand: Dict[int, int],
    state: UncertaintyAwareBilateralState,
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
    reserve_ratio: float = 0.1,
    max_rebalance_share: float = 0.6,
    max_distance_km: Optional[float] = None,
    donor_sigma_buffer: float = 0.3,
    donor_tail_buffer: float = 0.4,
    donor_debt_buffer: float = 0.35,
    bid_shortage_weight: float = 0.9,
    bid_service_weight: float = 0.7,
    bid_backlog_weight: float = 0.45,
    bid_burst_weight: float = 0.6,
    bid_debt_weight: float = 0.85,
    ask_shortage_weight: float = 0.85,
    ask_fairness_weight: float = 0.7,
    ask_uncertainty_weight: float = 0.65,
    distance_penalty: float = 0.004,
    opportunity_eta_weight: float = 0.015,
    opportunity_capture_weight: float = 0.9,
    opportunity_return_weight: float = 0.06,
    remote_worker_bonus: float = 0.05,
    switch_cooldown_slots: int = 2,
    switch_recent_penalty: float = 0.6,
    switch_repeat_penalty: float = 0.25,
    switch_lookback_slots: int = 4,
    candidate_k: int = 16,
    edge_epsilon: float = 0.05,
) -> Dict[str, Any]:
    max_tasks_per_worker = max(1, int(max_tasks_per_worker))
    backlog_counts = backlog_counts or {}
    max_distance_m = float("inf") if max_distance_km is None else max(0.0, float(max_distance_km)) * 1000.0
    region_ids = sorted(centers.keys())

    movable_workers: Dict[int, List[str]] = {rid: [] for rid in region_ids}
    for wid, region_id in worker_sim.worker_center_map.items():
        if region_id not in centers or wid not in worker_sim.worker_positions:
            continue

        status = worker_sim.worker_status.get(wid, "idle")
        busy_until = worker_sim.worker_busy_until.get(wid, 0.0)
        if status == "en_route_to_task" and busy_until > next_slot_start_seconds:
            continue
        movable_workers[region_id].append(wid)

    available_workers = {rid: len(movable_workers[rid]) for rid in region_ids}
    demand_profile: Dict[int, Dict[str, float]] = {}
    effective_demand: Dict[int, float] = {}
    burst_prob: Dict[int, float] = {}

    for rid in region_ids:
        dist_profile = (predicted_distribution or {}).get(rid, {})
        mu = float(max(0.0, dist_profile.get("mu", predicted_demand.get(rid, 0))))
        backlog = float(max(0.0, backlog_counts.get(rid, 0)))
        if dist_profile:
            sigma = float(max(0.0, dist_profile.get("sigma", 0.0)))
            q90 = float(max(mu, dist_profile.get("q90", mu + sigma)))
            tail_gap = max(0.0, q90 - mu)
            burst = float(np.clip(dist_profile.get("burst_prob", 0.0), 0.0, 1.0))
        else:
            errors = list(state.prediction_errors.get(rid, ()))
            arrivals = list(state.arrival_history.get(rid, ()))
            positive_errors = [max(0.0, e) for e in errors]

            sigma_error = _safe_std(errors)
            sigma_arrival = _safe_std(arrivals)
            sigma_floor = math.sqrt(max(mu, 1.0)) * 0.2
            sigma = max(sigma_floor, 0.5 * sigma_error + 0.35 * sigma_arrival + 0.15 * sigma_floor)

            tail_gap = max(
                sigma,
                _safe_quantile(positive_errors, 0.9, sigma),
                _safe_quantile(arrivals, 0.9, mu) - mu,
            )

            arrival_mean = float(np.mean(arrivals)) if arrivals else mu
            arrival_std = sigma_arrival if sigma_arrival > 0.0 else sigma_floor
            burst_threshold = arrival_mean + arrival_std
            empirical_burst = float(np.mean([1.0 if val >= burst_threshold else 0.0 for val in arrivals])) if arrivals else 0.0
            underpredict_rate = float(np.mean([1.0 if err > 0 else 0.0 for err in errors])) if errors else 0.0
            burst = float(np.clip(0.5 * empirical_burst + 0.5 * underpredict_rate, 0.0, 1.0))

        debt = float(state.service_debt.get(rid, 0.0))
        effective = (
            mu
            + backlog_weight * backlog
            + uncertainty_weight * sigma
            + quantile_weight * max(0.0, tail_gap)
            + burst_weight * burst * max(1.0, sigma)
        )

        demand_profile[rid] = {
            "mu": mu,
            "sigma": sigma,
            "q90": mu + tail_gap,
            "tail_gap": tail_gap,
            "burst_prob": burst,
            "backlog": backlog,
            "debt": debt,
            "effective_demand": effective,
        }
        effective_demand[rid] = effective
        burst_prob[rid] = burst

    required_workers = {
        rid: int(math.ceil(effective_demand[rid] / max_tasks_per_worker))
        for rid in region_ids
    }
    total_workers = int(sum(available_workers.values()))
    total_required_workers = int(sum(required_workers.values()))
    global_shortage_mode = total_required_workers > total_workers
    allocation_target_workers = {rid: 0 for rid in region_ids}
    local_guard_workers = {rid: 0 for rid in region_ids}

    score_basis = {}
    for rid in region_ids:
        profile = demand_profile[rid]
        score_basis[rid] = max(
            1e-6,
            profile["effective_demand"] * (1.0 + 0.20 * profile["burst_prob"] + 0.15 * profile["debt"])
        )
    score_sum = float(sum(score_basis.values()))
    if score_sum <= 0.0:
        score_sum = float(len(region_ids))

    base_targets = {}
    fractional_targets = []
    allocated_workers = 0
    for rid in region_ids:
        target_float = total_workers * score_basis[rid] / score_sum
        target_floor = int(math.floor(target_float))
        base_targets[rid] = target_floor
        allocated_workers += target_floor
        fractional_targets.append((target_float - target_floor, rid))

    for _, rid in sorted(fractional_targets, reverse=True)[:max(0, total_workers - allocated_workers)]:
        base_targets[rid] += 1
    allocation_target_workers.update(base_targets)

    protected_supply = {}
    max_outbound = {}
    for rid in region_ids:
        local_guard_demand = (
            demand_profile[rid]["mu"]
            + backlog_weight * demand_profile[rid]["backlog"]
            + 0.25 * demand_profile[rid]["sigma"]
            + 0.15 * demand_profile[rid]["tail_gap"]
        )
        local_guard_from_demand = int(math.ceil(local_guard_demand / max_tasks_per_worker))
        if global_shortage_mode:
            local_guard_workers[rid] = max(int(min_buffer_workers), min(allocation_target_workers[rid], local_guard_from_demand))
        else:
            local_guard_workers[rid] = local_guard_from_demand
        risk_buffer = (
            donor_sigma_buffer * demand_profile[rid]["sigma"]
            + donor_tail_buffer * max(0.0, demand_profile[rid]["q90"] - demand_profile[rid]["mu"])
        ) / max_tasks_per_worker
        debt_buffer = donor_debt_buffer * demand_profile[rid]["debt"]
        reserve_base = allocation_target_workers[rid] if global_shortage_mode else required_workers[rid]
        if global_shortage_mode:
            protected_value = reserve_base * max(0.0, reserve_ratio) + 0.25 * debt_buffer
        else:
            protected_value = reserve_base * max(0.0, reserve_ratio) + risk_buffer + debt_buffer
        protected_supply[rid] = max(
            int(min_buffer_workers),
            int(math.ceil(protected_value))
        )
        max_outbound[rid] = int(math.floor(available_workers[rid] * max(0.0, min(1.0, max_rebalance_share))))

    worker_counts = available_workers.copy()
    target_workers = allocation_target_workers if global_shortage_mode else required_workers
    receiver_shortage = {
        rid: max(0, target_workers[rid] - worker_counts[rid])
        for rid in region_ids
    }
    donor_supply = {}
    for rid in region_ids:
        keep_floor = (
            local_guard_workers[rid] + protected_supply[rid]
            if global_shortage_mode
            else required_workers[rid] + protected_supply[rid]
        )
        donor_supply[rid] = max(0, min(worker_counts[rid] - keep_floor, max_outbound[rid]))

    current_service_ratio = _compute_service_ratio(worker_counts, effective_demand, max_tasks_per_worker)
    current_unfairness = _compute_weighted_unfairness(current_service_ratio, effective_demand, burst_prob)
    distance_cache: Dict[Tuple[Any, Any], float] = {}

    def get_dist(n1: Any, n2: Any) -> float:
        if n1 == n2:
            return 0.0
        pair = (n1, n2) if str(n1) < str(n2) else (n2, n1)
        if pair not in distance_cache:
            try:
                distance_cache[pair] = nx.shortest_path_length(G, source=n1, target=n2, weight="length")
            except nx.NetworkXNoPath:
                distance_cache[pair] = float("inf")
        return distance_cache[pair]

    def get_candidate_workers(donor_region: int, receiver_region: int) -> List[Tuple[str, float, float]]:
        donor_workers = movable_workers[donor_region]
        if not donor_workers:
            return []

        receiver_center_node = centers[receiver_region]
        donor_center_node = centers[donor_region]
        center_lon = G.nodes[receiver_center_node].get("x", G.nodes[receiver_center_node].get("lon"))
        center_lat = G.nodes[receiver_center_node].get("y", G.nodes[receiver_center_node].get("lat"))

        coords = []
        valid_worker_ids = []
        for wid in donor_workers:
            worker_info = worker_sim.worker_positions.get(wid)
            if worker_info is None:
                continue
            coords.append((worker_info[1], worker_info[2]))
            valid_worker_ids.append(wid)

        if not valid_worker_ids:
            return []

        if len(valid_worker_ids) <= candidate_k:
            query_indices = list(range(len(valid_worker_ids)))
        else:
            tree = KDTree(np.asarray(coords, dtype=np.float32))
            _, query_indices = tree.query([center_lon, center_lat], k=min(candidate_k, len(valid_worker_ids)))
            query_indices = np.atleast_1d(query_indices).tolist()

        candidates: List[Tuple[str, float, float]] = []
        for idx in query_indices:
            wid = valid_worker_ids[int(idx)]
            worker_node = worker_sim.worker_positions[wid][0]
            dist_to_receiver = get_dist(worker_node, receiver_center_node)
            if dist_to_receiver > max_distance_m or not math.isfinite(dist_to_receiver):
                continue
            dist_to_donor = get_dist(worker_node, donor_center_node)
            candidates.append((wid, dist_to_receiver, dist_to_donor))

        if candidates:
            candidates.sort(key=lambda item: (item[1], -item[2]))
        return candidates

    def compute_switch_penalty(wid: str) -> float:
        history = list(state.worker_move_slots.get(wid, ()))
        if not history:
            return 0.0

        recent_count = sum(1 for prev_slot in history if slot_idx - prev_slot <= switch_lookback_slots)
        last_slot = history[-1]
        penalty = switch_repeat_penalty * recent_count
        if slot_idx - last_slot <= switch_cooldown_slots:
            penalty += switch_recent_penalty
        return penalty

    edge_candidates = []
    candidate_pair_count = 0
    for receiver in region_ids:
        if receiver_shortage[receiver] <= 0:
            continue

        receiver_profile = demand_profile[receiver]
        receiver_old_ratio = current_service_ratio[receiver]
        receiver_new_ratio = min(
            1.0,
            ((worker_counts[receiver] + 1) * max_tasks_per_worker) / max(receiver_profile["effective_demand"], 1.0),
        )
        receiver_gap_before = max(0, target_workers[receiver] - worker_counts[receiver])
        receiver_gap_after = max(0, target_workers[receiver] - (worker_counts[receiver] + 1))
        shortage_relief = (receiver_gap_before - receiver_gap_after) / max(target_workers[receiver], 1)
        receiver_gap_pressure = receiver_gap_before / max(target_workers[receiver], 1)
        service_gain = receiver_new_ratio - receiver_old_ratio
        backlog_relief = min(receiver_profile["backlog"], max_tasks_per_worker) / max(max_tasks_per_worker, 1)
        burst_pressure = receiver_profile["burst_prob"] + receiver_profile["tail_gap"] / max(
            receiver_profile["effective_demand"], 1.0
        )
        receiver_bid = (
            bid_shortage_weight * shortage_relief
            + bid_service_weight * service_gain
            + bid_backlog_weight * backlog_relief
            + bid_burst_weight * burst_pressure
            + bid_debt_weight * receiver_profile["debt"]
        )
        if global_shortage_mode:
            receiver_bid += 0.75 * receiver_gap_pressure + 0.35 * max(0.0, 1.0 - current_service_ratio[receiver])

        for donor in region_ids:
            if donor == receiver or donor_supply[donor] <= 0:
                continue

            simulated_counts = worker_counts.copy()
            simulated_counts[donor] -= 1
            simulated_counts[receiver] += 1
            donor_profile = demand_profile[donor]

            donor_new_ratio = min(
                1.0,
                (simulated_counts[donor] * max_tasks_per_worker) / max(donor_profile["effective_demand"], 1.0),
            )
            local_guard_gap = max(0, local_guard_workers[donor] - simulated_counts[donor])
            local_shortage_risk = local_guard_gap / max(local_guard_workers[donor], 1)
            new_service_ratio = _compute_service_ratio(simulated_counts, effective_demand, max_tasks_per_worker)
            fairness_damage = max(
                0.0,
                _compute_weighted_unfairness(new_service_ratio, effective_demand, burst_prob) - current_unfairness,
            )
            if global_shortage_mode:
                donor_gap_after = max(0, target_workers[donor] - simulated_counts[donor])
                donor_uncertainty = (
                    0.35 * donor_gap_after / max(target_workers[donor], 1)
                    + 0.20 * donor_profile["burst_prob"]
                    + 0.20 * donor_profile["debt"]
                    + 0.25 * max(0.0, current_service_ratio[donor] - donor_new_ratio)
                )
            else:
                donor_uncertainty = (
                    donor_profile["sigma"] / max(max_tasks_per_worker, 1)
                    + donor_profile["burst_prob"]
                    + 0.5 * donor_profile["debt"]
                    + max(0.0, current_service_ratio[donor] - donor_new_ratio)
                )
            donor_ask = (
                ask_shortage_weight * local_shortage_risk
                + ask_fairness_weight * fairness_damage
                + ask_uncertainty_weight * donor_uncertainty
            )

            center_distance_km = get_dist(centers[donor], centers[receiver]) / 1000.0
            for wid, dist_to_receiver, dist_to_donor in get_candidate_workers(donor, receiver):
                candidate_pair_count += 1
                eta_to_donor_minutes = dist_to_donor / max(1e-6, worker_sim.config.WORKER_SPEED_MS) / 60.0
                donor_pressure = min(
                    1.0,
                    donor_profile["effective_demand"] / max(max_tasks_per_worker * max(worker_counts[donor], 1), 1.0),
                )
                local_capture_prob = math.exp(-dist_to_donor / 1200.0) * donor_pressure
                opportunity_cost = (
                    opportunity_eta_weight * eta_to_donor_minutes
                    + opportunity_capture_weight * local_capture_prob
                    + opportunity_return_weight * center_distance_km
                )
                opportunity_cost = max(0.0, opportunity_cost - remote_worker_bonus * (dist_to_donor / 1000.0))
                move_cost = distance_penalty * (dist_to_receiver / 1000.0)
                switch_penalty = compute_switch_penalty(wid)
                edge_weight = receiver_bid - donor_ask - move_cost - opportunity_cost - switch_penalty
                if edge_weight <= edge_epsilon:
                    continue

                edge_candidates.append(
                    {
                        "wid": wid,
                        "from_region": donor,
                        "to_region": receiver,
                        "trade_surplus": receiver_bid - donor_ask,
                        "edge_weight": edge_weight,
                        "distance_to_target_center": dist_to_receiver,
                        "distance_from_donor_center": dist_to_donor,
                        "move_cost": move_cost,
                        "opportunity_cost": opportunity_cost,
                        "switch_penalty": switch_penalty,
                        "receiver_bid": receiver_bid,
                        "donor_ask": donor_ask,
                    }
                )

    edge_candidates.sort(key=lambda item: item["edge_weight"], reverse=True)

    selected_workers = set()
    outbound_used = {rid: 0 for rid in region_ids}
    inbound_used = {rid: 0 for rid in region_ids}
    moves = []

    for edge in edge_candidates:
        wid = edge["wid"]
        donor = edge["from_region"]
        receiver = edge["to_region"]
        if wid in selected_workers:
            continue
        if outbound_used[donor] >= donor_supply[donor]:
            continue
        if inbound_used[receiver] >= receiver_shortage[receiver]:
            continue

        selected_workers.add(wid)
        outbound_used[donor] += 1
        inbound_used[receiver] += 1
        worker_sim.worker_center_map[wid] = receiver
        worker_counts[donor] -= 1
        worker_counts[receiver] += 1
        moves.append(edge)

    final_service_ratio = _compute_service_ratio(worker_counts, effective_demand, max_tasks_per_worker)
    final_unfairness = _compute_weighted_unfairness(final_service_ratio, effective_demand, burst_prob)

    return {
        "predicted_demand": predicted_demand,
        "demand_profile": demand_profile,
        "effective_demand": {rid: int(round(val)) for rid, val in effective_demand.items()},
        "available_workers": available_workers,
        "required_workers": required_workers,
        "target_workers": target_workers,
        "local_guard_workers": local_guard_workers,
        "protected_supply": protected_supply,
        "final_worker_counts": worker_counts,
        "service_ratio": final_service_ratio,
        "weighted_unfairness": final_unfairness,
        "diagnostics": {
            "global_shortage_mode": global_shortage_mode,
            "total_workers": total_workers,
            "total_required_workers": total_required_workers,
            "active_receivers": sum(1 for rid in region_ids if receiver_shortage[rid] > 0),
            "active_donors": sum(1 for rid in region_ids if donor_supply[rid] > 0),
            "candidate_pairs": candidate_pair_count,
            "positive_edges": len(edge_candidates),
            "selected_moves": len(moves),
            "max_receiver_gap": max(receiver_shortage.values()) if receiver_shortage else 0,
            "max_donor_supply": max(donor_supply.values()) if donor_supply else 0,
        },
        "moves": moves,
    }
