from typing import Any, Dict, List, Tuple

import networkx as nx


def greedy_assignment_with_center_pickup(
    G: nx.Graph,
    config,
    centers: Dict[int, Any],
    partition: Dict[Any, int],
    workers_per_center: Dict[int, List[Tuple[Any, str, float, float, Any]]],
    tasks_per_center: Dict[int, List[Tuple[Any, str, float, float, float]]],
    slot_start_seconds: float = 0.0,
    slot_end_seconds: float = float("inf"),
) -> Tuple[Dict[Tuple[str, str], float], float, List[Dict]]:
    print(">> Start greedy assignment (task-count objective, time-window constrained)...")

    all_assignments: Dict[Tuple[str, str], float] = {}
    total_score = 0.0
    assignment_details: List[Dict] = []
    path_cache: Dict[Tuple[Any, Any], float] = {}
    source_cache: Dict[Any, Dict[Any, float]] = {}

    def get_dist(n1: Any, n2: Any) -> float:
        if n1 == n2:
            return 0.0
        pair = (n1, n2) if str(n1) < str(n2) else (n2, n1)
        if pair not in path_cache:
            if n1 not in source_cache:
                source_cache[n1] = nx.single_source_dijkstra_path_length(G, n1, weight="length")
            path_cache[pair] = source_cache[n1].get(n2, float("inf"))
        return path_cache[pair]

    for region_id, center_node in centers.items():
        workers = workers_per_center.get(region_id, [])
        tasks = [
            t for t in tasks_per_center.get(region_id, [])
            if partition.get(t[0]) == region_id
        ]

        if not workers or not tasks:
            continue

        region_score, region_assignments, region_details = _assign_single_region_dynamic_greedy(
            config=config,
            region_id=region_id,
            workers=workers,
            tasks=tasks,
            center_node=center_node,
            get_dist=get_dist,
            slot_start_seconds=slot_start_seconds,
            slot_end_seconds=slot_end_seconds,
        )

        all_assignments.update(region_assignments)
        total_score += region_score
        assignment_details.extend(region_details)

    print(f"✅ Greedy finished: assigned={len(all_assignments)}, objective_score={total_score:.2f}")
    return all_assignments, total_score, assignment_details


def _assign_single_region_dynamic_greedy(
    config,
    region_id: int,
    workers: List[Tuple],
    tasks: List[Tuple],
    center_node: Any,
    get_dist,
    slot_start_seconds: float,
    slot_end_seconds: float,
) -> Tuple[float, Dict[Tuple[str, str], float], List[Dict]]:
    if slot_end_seconds == float("inf"):
        slot_minutes = float(getattr(config, "EXPERIMENT_TIME_SLOT_MINUTES", 15))
        slot_end_seconds = slot_start_seconds + slot_minutes * 60

    available_tasks = {t[1] for t in tasks}
    task_nodes = {t[1]: t[0] for t in tasks}
    task_rewards = {t[1]: t[2] for t in tasks}
    task_expires = {t[1]: (t[3] if len(t) > 3 else float("inf")) for t in tasks}
    task_releases = {t[1]: (t[4] if len(t) > 4 else -float("inf")) for t in tasks}
    task_regions = {t[1]: region_id for t in tasks}

    worker_nodes = {w[1]: w[0] for w in workers}
    worker_round_load = {w[1]: 0 for w in workers}
    worker_virtual_loc = {w[1]: center_node for w in workers}
    worker_paid_center_cost = {w[1]: False for w in workers}
    worker_current_time = {w[1]: slot_start_seconds for w in workers}
    dist_worker_to_center = {wid: get_dist(w_node, center_node) for wid, w_node in worker_nodes.items()}

    center_to_task_dist = {tid: get_dist(center_node, task_nodes[tid]) for tid in task_nodes}
    task_to_center_dist = {tid: get_dist(task_nodes[tid], center_node) for tid in task_nodes}

    assignments: Dict[Tuple[str, str], float] = {}
    details: List[Dict] = []
    region_score = 0.0

    while available_tasks:
        best_pair = None
        best_priority = None
        best_dist_to_center = 0.0
        best_dist_to_task = 0.0
        best_return_dist = 0.0
        best_finish_time = slot_start_seconds
        best_end_time = slot_start_seconds
        best_end_node = center_node
        best_returned_to_center = False
        made_round_reset = False

        for wid in worker_nodes.keys():
            if worker_round_load[wid] >= config.MAX_TASKS_PER_WORKER:
                return_dist = get_dist(worker_virtual_loc[wid], center_node)
                if return_dist == float("inf"):
                    continue

                return_finish_time = worker_current_time[wid] + return_dist / config.WORKER_SPEED_MS
                if return_finish_time > slot_end_seconds:
                    continue

                worker_virtual_loc[wid] = center_node
                worker_current_time[wid] = return_finish_time
                worker_round_load[wid] = 0
                worker_paid_center_cost[wid] = True
                made_round_reset = True

            for tid in available_tasks:
                if task_regions.get(tid) != region_id:
                    continue

                release_time = task_releases[tid]
                if worker_virtual_loc[wid] == center_node:
                    dist_to_task = center_to_task_dist[tid]
                else:
                    dist_to_task = get_dist(worker_virtual_loc[wid], task_nodes[tid])
                if dist_to_task == float("inf"):
                    continue

                dist_to_center = 0.0 if worker_paid_center_cost[wid] else dist_worker_to_center[wid]
                total_dist = dist_to_center + dist_to_task
                travel_time = total_dist / config.WORKER_SPEED_MS
                arrival_time = max(worker_current_time[wid] + travel_time, release_time)
                if arrival_time > task_expires[tid]:
                    continue

                candidate_end_time = arrival_time
                candidate_end_node = task_nodes[tid]
                candidate_returned_to_center = False
                candidate_return_dist = 0.0

                if worker_round_load[wid] + 1 >= config.MAX_TASKS_PER_WORKER:
                    return_dist = task_to_center_dist[tid]
                    if return_dist != float("inf"):
                        return_finish_time = arrival_time + return_dist / config.WORKER_SPEED_MS
                        if return_finish_time <= slot_end_seconds:
                            candidate_end_time = return_finish_time
                            candidate_end_node = center_node
                            candidate_returned_to_center = True
                            candidate_return_dist = return_dist

                # Task-maximizing priority:
                # 1. smaller slack first (avoid losing urgent tasks)
                # 2. earlier end time (free worker sooner)
                # 3. shorter travel first
                candidate_priority = (
                    task_expires[tid] - arrival_time,
                    candidate_end_time,
                    dist_to_task + candidate_return_dist,
                    task_expires[tid],
                    tid,
                    wid,
                )
                if best_priority is not None and candidate_priority >= best_priority:
                    continue

                best_pair = (wid, tid)
                best_priority = candidate_priority
                best_dist_to_center = dist_to_center
                best_dist_to_task = dist_to_task
                best_return_dist = candidate_return_dist
                best_finish_time = arrival_time
                best_end_time = candidate_end_time
                best_end_node = candidate_end_node
                best_returned_to_center = candidate_returned_to_center

        if best_pair is None and made_round_reset:
            continue
        if best_pair is None:
            break

        best_wid, best_tid = best_pair
        assignments[(best_wid, best_tid)] = 1.0

        worker_round_load[best_wid] += 1
        worker_virtual_loc[best_wid] = best_end_node
        worker_paid_center_cost[best_wid] = True
        worker_current_time[best_wid] = best_end_time
        if best_returned_to_center:
            worker_round_load[best_wid] = 0

        region_score += 1.0

        actual_cost = (best_dist_to_center + best_dist_to_task + best_return_dist) * config.TRAVEL_COST_PER_METER
        actual_profit = task_rewards[best_tid] - actual_cost
        details.append(
            {
                "region_id": region_id,
                "wid": best_wid,
                "task_id": best_tid,
                "dist_to_center": best_dist_to_center,
                "dist_to_task": best_dist_to_task,
                "return_to_center_dist": best_return_dist,
                "task_node": best_end_node,
                "service_node": task_nodes[best_tid],
                "reward": task_rewards[best_tid],
                "cost": actual_cost,
                "finish_time": best_end_time,
                "service_finish_time": best_finish_time,
                "end_time": best_end_time,
                "end_node": best_end_node,
                "profit": actual_profit,
                "objective_score": 1.0,
            }
        )

        available_tasks.remove(best_tid)

        if len(details) % 100 == 0:
            print(f"   processed {len(details)} assignments...")

    print(f"✅ Region {region_id} greedy finished: assigned={len(assignments)}, objective_score={region_score:.2f}")
    return region_score, assignments, details
