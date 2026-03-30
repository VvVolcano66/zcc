import math
from typing import Any, Dict, List, Tuple

import networkx as nx


def predict_next_slot_demand(
        observed_arrivals_history: List[Dict[int, int]],
        backlog_counts: Dict[int, int],
        centers: Dict[int, Any]
) -> Dict[int, int]:
    """
    Predict the service demand of the next slot for each center.

    The predictor is intentionally lightweight so it can run inline in the
    dispatch simulation:
    - short-term momentum from the latest slot,
    - recent average over the last three observed slots,
    - overall average of observed slots so far,
    - current backlog that will remain into the next slot.
    """
    region_ids = sorted(centers.keys())
    if not observed_arrivals_history:
        return {rid: int(backlog_counts.get(rid, 0)) for rid in region_ids}

    latest = observed_arrivals_history[-1]
    recent_window = observed_arrivals_history[-min(3, len(observed_arrivals_history)):]

    predicted = {}
    for rid in region_ids:
        latest_count = latest.get(rid, 0)
        recent_avg = sum(slot.get(rid, 0) for slot in recent_window) / len(recent_window)
        overall_avg = sum(slot.get(rid, 0) for slot in observed_arrivals_history) / len(observed_arrivals_history)

        positive_trend = 0.0
        if len(observed_arrivals_history) >= 2:
            prev_count = observed_arrivals_history[-2].get(rid, 0)
            positive_trend = max(0.0, latest_count - prev_count) * 0.3

        predicted_new_tasks = 0.5 * latest_count + 0.3 * recent_avg + 0.2 * overall_avg + positive_trend
        predicted[rid] = max(0, int(round(predicted_new_tasks + backlog_counts.get(rid, 0))))

    return predicted


def predispatch_workers_for_next_slot(
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
        max_distance_km: float = 8.0
) -> Dict[str, Any]:
    """
    Reassign workers that are expected to be available in the next slot from
    supply-rich centers to demand-heavy centers.
    """
    max_tasks_per_worker = max(1, int(max_tasks_per_worker))
    backlog_counts = backlog_counts or {}
    backlog_weight = max(0.0, float(backlog_weight))
    min_buffer_workers = max(0, int(min_buffer_workers))
    reserve_ratio = max(0.0, float(reserve_ratio))
    max_rebalance_share = min(1.0, max(0.0, float(max_rebalance_share)))
    max_distance_m = max(0.0, float(max_distance_km)) * 1000.0
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
        rid: max(
            min_buffer_workers,
            int(math.ceil(required_workers[rid] * reserve_ratio))
        )
        for rid in region_ids
    }
    max_outbound = {
        rid: int(math.floor(available_workers[rid] * max_rebalance_share))
        for rid in region_ids
    }

    surplus = {
        rid: max(0, min(available_workers[rid] - required_workers[rid] - protected_supply[rid], max_outbound[rid]))
        for rid in region_ids
    }
    shortage = {
        rid: max(0, required_workers[rid] - available_workers[rid])
        for rid in region_ids
    }

    distance_cache: Dict[Tuple[Any, Any], float] = {}

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

    moves = []
    for target_region in sorted(region_ids, key=lambda rid: shortage[rid], reverse=True):
        while shortage[target_region] > 0:
            best_choice = None
            best_distance = float('inf')
            target_center_node = centers[target_region]

            donor_regions = sorted(
                [rid for rid in region_ids if surplus[rid] > 0 and rid != target_region],
                key=lambda rid: surplus[rid],
                reverse=True
            )
            if not donor_regions:
                break

            for donor_region in donor_regions:
                for wid in movable_workers[donor_region]:
                    worker_node = worker_sim.worker_positions[wid][0]
                    dist = get_dist(worker_node, target_center_node)
                    if dist <= max_distance_m and dist < best_distance:
                        best_distance = dist
                        best_choice = (wid, donor_region)

            if best_choice is None or best_distance == float('inf'):
                break

            wid, donor_region = best_choice
            movable_workers[donor_region].remove(wid)
            worker_sim.worker_center_map[wid] = target_region
            movable_workers[target_region].append(wid)

            surplus[donor_region] -= 1
            shortage[target_region] -= 1
            moves.append({
                'wid': wid,
                'from_region': donor_region,
                'to_region': target_region,
                'distance_to_target_center': best_distance
            })

    return {
        'predicted_demand': predicted_demand,
        'effective_demand': effective_demand,
        'available_workers': available_workers,
        'required_workers': required_workers,
        'protected_supply': protected_supply,
        'moves': moves
    }
