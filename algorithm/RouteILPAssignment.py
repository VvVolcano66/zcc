from typing import Any, Dict, List, Tuple

import networkx as nx
try:
    import pulp
except Exception:
    pulp = None

from algorithm.Greedy import _assign_single_region_dynamic_greedy


def _select_routes_greedily(
    all_routes: List[Tuple[str, int, Dict]],
) -> Tuple[float, Dict[Tuple[str, str], float], List[Dict]]:
    assignments: Dict[Tuple[str, str], float] = {}
    details: List[Dict] = []
    total_score = 0.0
    used_workers = set()
    used_tasks = set()

    ranked_routes = sorted(
        all_routes,
        key=lambda item: (
            -float(item[2]["objective_score"]),
            -len(item[2]["task_ids"]),
            item[0],
            item[1],
        ),
    )

    for wid, _route_idx, route in ranked_routes:
        if wid in used_workers:
            continue
        task_ids = route["task_ids"]
        if any(tid in used_tasks for tid in task_ids):
            continue

        used_workers.add(wid)
        used_tasks.update(task_ids)
        total_score += float(route["objective_score"])
        for detail in route["details"]:
            assignments[(wid, detail["task_id"])] = 1.0
            details.append(detail)

    return total_score, assignments, details


def route_ilp_assignment_with_center_pickup(
    G: nx.Graph,
    config,
    centers: Dict[int, Any],
    partition: Dict[Any, int],
    workers_per_center: Dict[int, List[Tuple[Any, str, float, float, Any]]],
    tasks_per_center: Dict[int, List[Tuple[Any, str, float, float, float]]],
    slot_start_seconds: float = 0.0,
    slot_end_seconds: float = float("inf"),
) -> Tuple[Dict[Tuple[str, str], float], float, List[Dict]]:
    print(">> Start route-ILP assignment (predictive/game variants)...")

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

        region_score, region_assignments, region_details = _assign_single_region_route_ilp(
            config=config,
            region_id=region_id,
            workers=workers,
            tasks=tasks,
            center_node=center_node,
            get_dist=get_dist,
            slot_start_seconds=slot_start_seconds,
            slot_end_seconds=slot_end_seconds,
        )

        if region_score is None:
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

    print(f"✅ Route-ILP finished: assigned={len(all_assignments)}, objective_score={total_score:.2f}")
    return all_assignments, total_score, assignment_details


def _assign_single_region_route_ilp(
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

    task_nodes = {t[1]: t[0] for t in tasks}
    task_rewards = {t[1]: t[2] for t in tasks}
    task_expires = {t[1]: (t[3] if len(t) > 3 else float("inf")) for t in tasks}
    task_releases = {t[1]: (t[4] if len(t) > 4 else -float("inf")) for t in tasks}
    task_ids = [t[1] for t in tasks]

    worker_nodes = {w[1]: w[0] for w in workers}
    dist_worker_to_center = {wid: get_dist(w_node, center_node) for wid, w_node in worker_nodes.items()}
    center_to_task_dist = {tid: get_dist(center_node, task_nodes[tid]) for tid in task_ids}
    task_to_center_dist = {tid: get_dist(task_nodes[tid], center_node) for tid in task_ids}

    max_tasks_per_worker = int(getattr(config, "MAX_TASKS_PER_WORKER", 4))
    max_route_tasks = int(getattr(config, "ROUTE_ILP_MAX_ROUTE_TASKS", max_tasks_per_worker * 3))
    candidate_task_limit = int(getattr(config, "ROUTE_ILP_CANDIDATE_TASKS_PER_WORKER", 18))
    branch_limit = int(getattr(config, "ROUTE_ILP_BRANCH_LIMIT", 6))
    max_routes_per_worker = int(getattr(config, "ROUTE_ILP_MAX_ROUTES_PER_WORKER", 80))
    ilp_time_limit = int(getattr(config, "ROUTE_ILP_TIME_LIMIT_SECONDS", 8))

    route_candidates_by_worker: Dict[str, List[Dict]] = {}

    for worker in workers:
        wid = worker[1]
        worker_center_arrival = slot_start_seconds + dist_worker_to_center[wid] / config.WORKER_SPEED_MS
        feasible_first_tasks = []
        for tid in task_ids:
            center_dist = center_to_task_dist[tid]
            if center_dist == float("inf"):
                continue
            first_departure = max(worker_center_arrival, task_releases[tid])
            first_arrival = first_departure + center_dist / config.WORKER_SPEED_MS
            if first_arrival > task_expires[tid]:
                continue
            feasible_first_tasks.append(
                (
                    task_expires[tid] - first_arrival,
                    first_arrival,
                    center_dist,
                    task_expires[tid],
                    task_releases[tid],
                    tid,
                )
            )

        candidate_task_ids = [
            item[-1]
            for item in sorted(feasible_first_tasks)[:candidate_task_limit]
        ]

        route_candidates_by_worker[wid] = _enumerate_worker_routes(
            wid=wid,
            region_id=region_id,
            center_node=center_node,
            candidate_task_ids=candidate_task_ids,
            task_nodes=task_nodes,
            task_rewards=task_rewards,
            task_expires=task_expires,
            task_releases=task_releases,
            dist_worker_to_center=dist_worker_to_center[wid],
            center_to_task_dist=center_to_task_dist,
            task_to_center_dist=task_to_center_dist,
            get_dist=get_dist,
            speed_ms=config.WORKER_SPEED_MS,
            travel_cost_per_meter=config.TRAVEL_COST_PER_METER,
            slot_start_seconds=slot_start_seconds,
            slot_end_seconds=slot_end_seconds,
            max_tasks_per_worker=max_tasks_per_worker,
            max_route_tasks=max_route_tasks,
            branch_limit=branch_limit,
            max_routes=max_routes_per_worker,
        )

    all_routes = []
    for wid, routes in route_candidates_by_worker.items():
        for route_idx, route in enumerate(routes):
            all_routes.append((wid, route_idx, route))

    if not all_routes:
        return None, {}, []

    if pulp is None:
        total_score, assignments, details = _select_routes_greedily(all_routes)
        used_workers = {wid for wid, _ in assignments.keys()}
        used_tasks = {tid for _, tid in assignments.keys()}
        remaining_workers = [worker for worker in workers if worker[1] not in used_workers]
        remaining_tasks = [task for task in tasks if task[1] not in used_tasks]
        if remaining_workers and remaining_tasks:
            extra_score, extra_assignments, extra_details = _assign_single_region_dynamic_greedy(
                config=config,
                region_id=region_id,
                workers=remaining_workers,
                tasks=remaining_tasks,
                center_node=center_node,
                get_dist=get_dist,
                slot_start_seconds=slot_start_seconds,
                slot_end_seconds=slot_end_seconds,
            )
            assignments.update(extra_assignments)
            details.extend(extra_details)
            total_score += extra_score
        print(
            f"✅ Region {region_id} route-pack fallback finished: "
            f"assigned={len(assignments)}, objective_score={total_score:.2f}"
        )
        return total_score, assignments, details

    model = pulp.LpProblem(f"route_pack_region_{region_id}", pulp.LpMaximize)
    y_vars = {
        (wid, route_idx): pulp.LpVariable(f"y_{region_id}_{wid}_{route_idx}", cat="Binary")
        for wid, route_idx, _ in all_routes
    }

    model += pulp.lpSum(
        route["objective_score"] * y_vars[(wid, route_idx)]
        for wid, route_idx, route in all_routes
    )

    for wid in route_candidates_by_worker.keys():
        worker_route_indices = [(w, idx) for w, idx, _ in all_routes if w == wid]
        model += pulp.lpSum(y_vars[key] for key in worker_route_indices) <= 1

    for tid in task_ids:
        covering_keys = [
            (wid, route_idx)
            for wid, route_idx, route in all_routes
            if tid in route["task_ids"]
        ]
        if covering_keys:
            model += pulp.lpSum(y_vars[key] for key in covering_keys) <= 1

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=ilp_time_limit)
    model.solve(solver)
    status = pulp.LpStatus.get(model.status, "Unknown")
    if status not in {"Optimal", "Integer Feasible", "Feasible"}:
        return None, {}, []

    assignments: Dict[Tuple[str, str], float] = {}
    details: List[Dict] = []
    total_score = 0.0

    for wid, route_idx, route in all_routes:
        value = pulp.value(y_vars[(wid, route_idx)])
        if value is None or value < 0.5:
            continue
        total_score += float(route["objective_score"])
        for detail in route["details"]:
            assignments[(wid, detail["task_id"])] = 1.0
            details.append(detail)

    used_workers = {wid for wid, _ in assignments.keys()}
    used_tasks = {tid for _, tid in assignments.keys()}
    remaining_workers = [worker for worker in workers if worker[1] not in used_workers]
    remaining_tasks = [task for task in tasks if task[1] not in used_tasks]
    if remaining_workers and remaining_tasks:
        extra_score, extra_assignments, extra_details = _assign_single_region_dynamic_greedy(
            config=config,
            region_id=region_id,
            workers=remaining_workers,
            tasks=remaining_tasks,
            center_node=center_node,
            get_dist=get_dist,
            slot_start_seconds=slot_start_seconds,
            slot_end_seconds=slot_end_seconds,
        )
        assignments.update(extra_assignments)
        details.extend(extra_details)
        total_score += extra_score

    print(f"✅ Region {region_id} route-ILP finished: assigned={len(assignments)}, objective_score={total_score:.2f}")
    return total_score, assignments, details


def _enumerate_worker_routes(
    wid: str,
    region_id: int,
    center_node: Any,
    candidate_task_ids: List[str],
    task_nodes: Dict[str, Any],
    task_rewards: Dict[str, float],
    task_expires: Dict[str, float],
    task_releases: Dict[str, float],
    dist_worker_to_center: float,
    center_to_task_dist: Dict[str, float],
    task_to_center_dist: Dict[str, float],
    get_dist,
    speed_ms: float,
    travel_cost_per_meter: float,
    slot_start_seconds: float,
    slot_end_seconds: float,
    max_tasks_per_worker: int,
    max_route_tasks: int,
    branch_limit: int,
    max_routes: int,
) -> List[Dict]:
    routes: List[Dict] = []
    if dist_worker_to_center == float("inf"):
        return routes

    worker_arrival_at_center = slot_start_seconds + dist_worker_to_center / speed_ms

    def add_route(
        seq: List[str],
        detail_states: List[Dict],
        total_distance: float,
        returned_to_center: bool,
    ) -> None:
        if not seq:
            return
        route_score = len(seq) - 1e-6 * total_distance
        routes.append(
            {
                "task_ids": tuple(seq),
                "details": [dict(d) for d in detail_states],
                "objective_score": route_score,
                "returned_to_center": returned_to_center,
            }
        )

    def dfs(
        current_node: Any,
        current_time: float,
        seq: List[str],
        used: set,
        detail_states: List[Dict],
        total_distance: float,
        round_load: int,
    ) -> None:
        if len(routes) >= max_routes:
            return
        if len(seq) >= max_route_tasks:
            return

        candidates = []
        for tid in candidate_task_ids:
            if tid in used:
                continue

            if not seq:
                dist_to_task = center_to_task_dist[tid]
            else:
                dist_to_task = get_dist(current_node, task_nodes[tid])
            if dist_to_task == float("inf"):
                continue

            departure_time = max(current_time, task_releases[tid])
            arrival_time = departure_time + dist_to_task / speed_ms
            if arrival_time > task_expires[tid]:
                continue

            end_time = arrival_time
            end_node = task_nodes[tid]
            return_dist = 0.0
            returned_to_center = False
            total_distance_after = total_distance + dist_to_task

            next_round_load = round_load + 1
            if next_round_load >= max_tasks_per_worker:
                return_dist = task_to_center_dist[tid]
                if return_dist == float("inf"):
                    continue
                return_finish_time = arrival_time + return_dist / speed_ms
                end_time = return_finish_time
                end_node = center_node
                returned_to_center = True
                total_distance_after += return_dist

            slack = task_expires[tid] - arrival_time
            candidates.append(
                (
                    slack,
                    end_time,
                    dist_to_task + return_dist,
                    tid,
                    arrival_time,
                    end_node,
                    end_time,
                    return_dist,
                    returned_to_center,
                    total_distance_after,
                    next_round_load,
                )
            )

        candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        for candidate in candidates[:branch_limit]:
            (
                _slack,
                _candidate_end_time,
                _candidate_travel,
                tid,
                arrival_time,
                end_node,
                end_time,
                return_dist,
                returned_to_center,
                total_distance_after,
                next_round_load,
            ) = candidate

            dist_to_center = dist_worker_to_center if not seq else 0.0
            dist_to_task = center_to_task_dist[tid] if not seq else get_dist(current_node, task_nodes[tid])
            actual_cost = (dist_to_center + dist_to_task + return_dist) * travel_cost_per_meter
            actual_profit = task_rewards[tid] - actual_cost
            next_details = detail_states + [
                {
                    "region_id": region_id,
                    "wid": wid,
                    "task_id": tid,
                    "dist_to_center": dist_to_center,
                    "dist_to_task": dist_to_task,
                    "return_to_center_dist": return_dist,
                    "task_node": end_node,
                    "service_node": task_nodes[tid],
                    "reward": task_rewards[tid],
                    "cost": actual_cost,
                    "finish_time": end_time,
                    "service_finish_time": arrival_time,
                    "end_time": end_time,
                    "end_node": end_node,
                    "profit": actual_profit,
                    "objective_score": 1.0,
                }
            ]
            next_seq = seq + [tid]
            next_used = used | {tid}
            add_route(
                seq=next_seq,
                detail_states=next_details,
                total_distance=total_distance_after,
                returned_to_center=returned_to_center,
            )

            if len(next_seq) >= max_route_tasks:
                continue

            if returned_to_center:
                dfs(
                    current_node=center_node,
                    current_time=end_time,
                    seq=next_seq,
                    used=next_used,
                    detail_states=next_details,
                    total_distance=total_distance_after,
                    round_load=0,
                )
                continue

            dfs(
                current_node=task_nodes[tid],
                current_time=arrival_time,
                seq=next_seq,
                used=next_used,
                detail_states=next_details,
                total_distance=total_distance_after,
                round_load=next_round_load,
            )

    dfs(
        current_node=center_node,
        current_time=worker_arrival_at_center,
        seq=[],
        used=set(),
        detail_states=[],
        total_distance=dist_worker_to_center,
        round_load=0,
    )

    deduped: Dict[Tuple[str, ...], Dict] = {}
    for route in routes:
        key = tuple(route["task_ids"])
        old = deduped.get(key)
        if old is None or route["objective_score"] > old["objective_score"]:
            deduped[key] = route

    sorted_routes = sorted(
        deduped.values(),
        key=lambda r: (-r["objective_score"], len(r["task_ids"]), r["task_ids"]),
    )
    return sorted_routes[:max_routes]
