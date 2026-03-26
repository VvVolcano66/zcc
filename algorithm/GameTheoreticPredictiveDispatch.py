import math
from typing import Any, Dict, List, Tuple

import networkx as nx


def _pairwise_unfairness(service_ratio: Dict[int, float]) -> float:
    region_ids = sorted(service_ratio.keys())
    if len(region_ids) <= 1:
        return 0.0

    total = 0.0
    for i in range(len(region_ids)):
        for j in range(len(region_ids)):
            if i != j:
                total += abs(service_ratio[region_ids[i]] - service_ratio[region_ids[j]])
    return total / (len(region_ids) * (len(region_ids) - 1))


def _compute_service_ratio(
        worker_counts: Dict[int, int],
        effective_demand: Dict[int, int],
        max_tasks_per_worker: int
) -> Dict[int, float]:
    ratio = {}
    for rid, demand in effective_demand.items():
        if demand <= 0:
            ratio[rid] = 1.0
            continue
        service_capacity = worker_counts.get(rid, 0) * max_tasks_per_worker
        ratio[rid] = min(1.0, service_capacity / max(demand, 1))
    return ratio


def _compute_center_utility(
        region_id: int,
        service_ratio: Dict[int, float],
        fairness_weight: float
) -> float:
    other_ids = [rid for rid in service_ratio.keys() if rid != region_id]
    if not other_ids:
        return service_ratio[region_id]

    disparity = sum(abs(service_ratio[region_id] - service_ratio[rid]) for rid in other_ids) / len(other_ids)
    return service_ratio[region_id] - fairness_weight * disparity


def _compute_potential(
        worker_counts: Dict[int, int],
        effective_demand: Dict[int, int],
        max_tasks_per_worker: int,
        fairness_weight: float
) -> float:
    service_ratio = _compute_service_ratio(worker_counts, effective_demand, max_tasks_per_worker)
    covered_tasks = 0.0
    for rid, demand in effective_demand.items():
        covered_tasks += min(demand, worker_counts.get(rid, 0) * max_tasks_per_worker)
    unfairness = _pairwise_unfairness(service_ratio)
    total_demand = max(1.0, float(sum(effective_demand.values())))
    return covered_tasks / total_demand - fairness_weight * unfairness


def game_theoretic_predispatch_workers(
        G: nx.Graph,
        worker_sim,
        centers: Dict[int, Any],
        predicted_demand: Dict[int, int],
        next_slot_start_seconds: float,
        max_tasks_per_worker: int = 4,
        backlog_counts: Dict[int, int] = None,
        backlog_weight: float = 1.0,
        min_buffer_workers: int = 3,
        reserve_ratio: float = 0.15,
        max_rebalance_share: float = 0.35,
        max_distance_km: float = 8.0,
        fairness_weight: float = 0.5,
        distance_penalty: float = 0.015,
        donor_max_utility_drop: float = 0.04,
        receiver_min_utility_gain: float = 0.01,
        max_iterations: int = 120
) -> Dict[str, Any]:
    """
    Potential-game style pre-dispatch.

    Each center is treated as a player. A worker transfer is accepted only when:
    1. the receiver's utility improves enough,
    2. the donor's utility does not fall too much,
    3. the global potential increases after accounting for travel distance.
    """
    max_tasks_per_worker = max(1, int(max_tasks_per_worker))
    backlog_counts = backlog_counts or {}
    backlog_weight = max(0.0, float(backlog_weight))
    min_buffer_workers = max(0, int(min_buffer_workers))
    reserve_ratio = max(0.0, float(reserve_ratio))
    max_rebalance_share = min(1.0, max(0.0, float(max_rebalance_share)))
    max_distance_m = max(0.0, float(max_distance_km)) * 1000.0
    max_iterations = max(1, int(max_iterations))

    region_ids = sorted(centers.keys())
    movable_workers = {rid: [] for rid in region_ids}

    for wid, region_id in worker_sim.worker_center_map.items():
        if region_id not in centers or wid not in worker_sim.worker_positions:
            continue

        status = worker_sim.worker_status.get(wid, 'idle')
        busy_until = worker_sim.worker_busy_until.get(wid, 0.0)
        if status == 'en_route_to_task' and busy_until > next_slot_start_seconds:
            continue

        movable_workers[region_id].append(wid)

    available_workers = {rid: len(movable_workers[rid]) for rid in region_ids}
    effective_demand = {
        rid: max(0, int(round(predicted_demand.get(rid, 0) + backlog_weight * backlog_counts.get(rid, 0))))
        for rid in region_ids
    }
    required_workers = {
        rid: int(math.ceil(effective_demand[rid] / max_tasks_per_worker))
        for rid in region_ids
    }
    protected_supply = {
        rid: max(min_buffer_workers, int(math.ceil(required_workers[rid] * reserve_ratio)))
        for rid in region_ids
    }
    max_outbound = {
        rid: int(math.floor(available_workers[rid] * max_rebalance_share))
        for rid in region_ids
    }

    worker_counts = available_workers.copy()
    outbound_counts = {rid: 0 for rid in region_ids}
    distance_cache: Dict[Tuple[Any, Any], float] = {}
    moves = []

    def get_dist(n1: Any, n2: Any) -> float:
        if n1 == n2:
            return 0.0
        pair = (n1, n2) if str(n1) < str(n2) else (n2, n1)
        if pair not in distance_cache:
            try:
                distance_cache[pair] = nx.shortest_path_length(G, source=n1, target=n2, weight='length')
            except nx.NetworkXNoPath:
                distance_cache[pair] = float('inf')
        return distance_cache[pair]

    def donor_can_send(region_id: int) -> bool:
        keep_floor = required_workers[region_id] + protected_supply[region_id]
        if worker_counts[region_id] - 1 < keep_floor:
            return False
        if outbound_counts[region_id] >= max_outbound[region_id]:
            return False
        return True

    for _ in range(max_iterations):
        current_service = _compute_service_ratio(worker_counts, effective_demand, max_tasks_per_worker)
        current_utilities = {
            rid: _compute_center_utility(rid, current_service, fairness_weight)
            for rid in region_ids
        }
        current_potential = _compute_potential(worker_counts, effective_demand, max_tasks_per_worker, fairness_weight)

        best_move = None
        best_gain = 0.0

        shortage_regions = sorted(
            region_ids,
            key=lambda rid: (
                required_workers[rid] - worker_counts[rid],
                effective_demand[rid]
            ),
            reverse=True
        )

        for receiver in shortage_regions:
            if worker_counts[receiver] >= required_workers[receiver]:
                continue

            receiver_center = centers[receiver]
            for donor in region_ids:
                if donor == receiver or not donor_can_send(donor):
                    continue

                best_worker_for_pair = None
                best_distance_for_pair = float('inf')
                for wid in movable_workers[donor]:
                    worker_node = worker_sim.worker_positions[wid][0]
                    dist = get_dist(worker_node, receiver_center)
                    if dist <= max_distance_m and dist < best_distance_for_pair:
                        best_distance_for_pair = dist
                        best_worker_for_pair = wid

                if best_worker_for_pair is None:
                    continue

                simulated_counts = worker_counts.copy()
                simulated_counts[donor] -= 1
                simulated_counts[receiver] += 1

                new_service = _compute_service_ratio(simulated_counts, effective_demand, max_tasks_per_worker)
                donor_old_utility = current_utilities[donor]
                receiver_old_utility = current_utilities[receiver]
                donor_new_utility = _compute_center_utility(donor, new_service, fairness_weight)
                receiver_new_utility = _compute_center_utility(receiver, new_service, fairness_weight)

                receiver_gain = receiver_new_utility - receiver_old_utility
                donor_drop = donor_old_utility - donor_new_utility
                if receiver_gain < receiver_min_utility_gain:
                    continue
                if donor_drop > donor_max_utility_drop:
                    continue

                move_cost = distance_penalty * (best_distance_for_pair / 1000.0)
                new_potential = _compute_potential(simulated_counts, effective_demand, max_tasks_per_worker, fairness_weight)
                potential_gain = new_potential - current_potential - move_cost

                if potential_gain > best_gain:
                    best_gain = potential_gain
                    best_move = {
                        'wid': best_worker_for_pair,
                        'from_region': donor,
                        'to_region': receiver,
                        'distance_to_target_center': best_distance_for_pair,
                        'receiver_gain': receiver_gain,
                        'donor_drop': donor_drop,
                        'potential_gain': potential_gain
                    }

        if best_move is None or best_gain <= 1e-9:
            break

        wid = best_move['wid']
        donor = best_move['from_region']
        receiver = best_move['to_region']

        movable_workers[donor].remove(wid)
        movable_workers[receiver].append(wid)
        worker_sim.worker_center_map[wid] = receiver
        worker_counts[donor] -= 1
        worker_counts[receiver] += 1
        outbound_counts[donor] += 1
        moves.append(best_move)

    final_service = _compute_service_ratio(worker_counts, effective_demand, max_tasks_per_worker)
    final_utilities = {
        rid: _compute_center_utility(rid, final_service, fairness_weight)
        for rid in region_ids
    }

    return {
        'predicted_demand': predicted_demand,
        'effective_demand': effective_demand,
        'available_workers': available_workers,
        'required_workers': required_workers,
        'protected_supply': protected_supply,
        'final_worker_counts': worker_counts,
        'service_ratio': final_service,
        'center_utility': final_utilities,
        'moves': moves
    }
