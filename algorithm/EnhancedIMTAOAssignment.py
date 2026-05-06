from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from algorithm.IMTAO import (
    Center as IMTAOCenter,
    IMTAO_Framework,
    Task as IMTAOTask,
    Worker as IMTAOWorker,
)


class EnhancedIMTAOAssignmentFramework(IMTAO_Framework):
    """
    Our assignment backend keeps IMTAO's center-pickup sequencing structure but
    improves local task choice and worker ordering. Workers receive a pre-packed
    task sequence at the center before departure; they do not accept new tasks
    after leaving the center within the same dispatch round. The original IMTAO
    baseline remains unchanged in `algorithm/IMTAO.py`.
    """

    def __init__(
        self,
        *args,
        task_reward_map: Optional[Dict[str, float]] = None,
        region_priority_weight: Optional[Dict[int, float]] = None,
        worker_completion_bonus: float = 0.0,
        worker_distance_penalty: float = 0.0,
        same_worker_chain_bonus: float = 0.0,
        force_center_pickup_on_first_departure: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.task_reward_map = task_reward_map or {}
        self.region_priority_weight = region_priority_weight or {}
        self.worker_completion_bonus = float(worker_completion_bonus)
        self.worker_distance_penalty = float(worker_distance_penalty)
        self.same_worker_chain_bonus = float(same_worker_chain_bonus)
        self.use_stackelberg_worker_utility = bool(self.region_priority_weight)
        self.force_center_pickup_on_first_departure = bool(force_center_pickup_on_first_departure)

    def _stackelberg_task_score(
        self,
        center: IMTAOCenter,
        task: IMTAOTask,
        travel_t: float,
        finish_time: float,
        deadline: float,
        follow_up: bool,
    ) -> float:
        reward = float(self.task_reward_map.get(task.id, 1.0))
        region_weight = float(self.region_priority_weight.get(center.id, 1.0))
        slack = max(0.0, deadline - finish_time)
        urgency_bonus = 1.0 / max(1.0, slack)
        return (
            region_weight * reward
            + self.worker_completion_bonus
            + (self.same_worker_chain_bonus if follow_up else 0.0)
            + 0.35 * urgency_bonus
        )

    def _find_nearest_feasible_task(
        self,
        center: IMTAOCenter,
        current_ref,
        current_time: float,
        remaining_tasks: Sequence[IMTAOTask],
    ) -> Tuple[Optional[IMTAOTask], Optional[float]]:
        best_task = None
        best_finish = None
        best_priority = None
        hard_deadline = self.slot_duration_seconds

        for task in remaining_tasks:
            travel_t = self._travel_time(current_ref, task)
            if not np.isfinite(travel_t):
                continue

            finish_time = max(current_time + travel_t, task.r)
            deadline = min(task.e, hard_deadline)
            if finish_time > deadline:
                continue

            if self.use_stackelberg_worker_utility:
                priority = (
                    -self._stackelberg_task_score(
                        center=center,
                        task=task,
                        travel_t=travel_t,
                        finish_time=finish_time,
                        deadline=deadline,
                        follow_up=False,
                    ),
                    finish_time,
                    travel_t,
                    str(task.id),
                )
            else:
                priority = (
                    deadline - finish_time,
                    finish_time,
                    travel_t,
                    str(task.id),
                )
            if best_priority is None or priority < best_priority:
                best_task = task
                best_finish = finish_time
                best_priority = priority

        return best_task, best_finish

    def _find_same_worker_preferred_task(
        self,
        center: IMTAOCenter,
        current_ref,
        current_time: float,
        remaining_tasks: Sequence[IMTAOTask],
    ) -> Tuple[Optional[IMTAOTask], Optional[float]]:
        best_task = None
        best_finish = None
        best_priority = None
        hard_deadline = self.slot_duration_seconds

        for task in remaining_tasks:
            travel_t = self._travel_time(current_ref, task)
            if not np.isfinite(travel_t):
                continue

            finish_time = max(current_time + travel_t, task.r)
            deadline = min(task.e, hard_deadline)
            if finish_time > deadline:
                continue

            if self.use_stackelberg_worker_utility:
                priority = (
                    -self._stackelberg_task_score(
                        center=center,
                        task=task,
                        travel_t=travel_t,
                        finish_time=finish_time,
                        deadline=deadline,
                        follow_up=True,
                    ),
                    travel_t,
                    finish_time,
                    str(task.id),
                )
            else:
                # Still in center-side pre-packing: for follow-up tasks within the
                # same planned sequence, prefer nearby feasible tasks for the same
                # worker before activating more workers.
                priority = (
                    travel_t,
                    deadline - finish_time,
                    finish_time,
                    str(task.id),
                )
            if best_priority is None or priority < best_priority:
                best_task = task
                best_finish = finish_time
                best_priority = priority

        return best_task, best_finish

    def _build_task_sequence(
        self,
        center: IMTAOCenter,
        worker: IMTAOWorker,
        candidate_tasks: Sequence[IMTAOTask],
    ) -> Tuple[List[IMTAOTask], float]:
        if self.force_center_pickup_on_first_departure:
            current_time = self._travel_time(worker, center)
            if not np.isfinite(current_time) or current_time > self.slot_duration_seconds:
                return [], current_time
            current_ref = center
        else:
            current_time = 0.0
            current_ref = worker
        remaining_tasks = list(candidate_tasks)
        selected_tasks: List[IMTAOTask] = []
        round_load = 0

        while remaining_tasks:
            if round_load >= worker.maxT:
                return_time = self._travel_time(current_ref, center)
                if not np.isfinite(return_time) or current_time + return_time > self.slot_duration_seconds:
                    break
                current_time += return_time
                current_ref = center
                round_load = 0

            if selected_tasks and current_ref is not center:
                next_task, next_finish_time = self._find_same_worker_preferred_task(
                    center=center,
                    current_ref=current_ref,
                    current_time=current_time,
                    remaining_tasks=remaining_tasks,
                )
            else:
                next_task, next_finish_time = self._find_nearest_feasible_task(
                    center=center,
                    current_ref=current_ref,
                    current_time=current_time,
                    remaining_tasks=remaining_tasks,
                )

            if next_task is None:
                break

            selected_tasks.append(next_task)
            remaining_tasks = [task for task in remaining_tasks if task.id != next_task.id]
            current_time = next_finish_time
            current_ref = next_task
            round_load += 1

        return selected_tasks, current_time

    def _run_sequential_assignment(self, center: IMTAOCenter, workers_to_assign: Sequence[IMTAOWorker]):
        assignments: List[Tuple[IMTAOWorker, List[IMTAOTask]]] = []
        remaining_tasks = list(center.S)
        unused_workers: List[IMTAOWorker] = []

        workers_sorted = sorted(
            workers_to_assign,
            key=lambda worker: (
                self._travel_time(worker, center) if self.force_center_pickup_on_first_departure else 0.0,
                str(worker.id),
            ),
        )

        for worker in workers_sorted:
            task_seq, _ = self._build_task_sequence(center, worker, remaining_tasks)
            if task_seq:
                assigned_ids = {task.id for task in task_seq}
                remaining_tasks = [task for task in remaining_tasks if task.id not in assigned_ids]
            else:
                unused_workers.append(worker)
            assignments.append((worker, task_seq))

        rho = self._compute_rho(center, assignments)
        return {
            "rho": rho,
            "A": assignments,
            "S_left": remaining_tasks,
            "W_left": unused_workers,
        }


def enhanced_imtao_assignment_with_center_pickup(
    G,
    config,
    centers_dict,
    workers_per_center,
    tasks_per_center,
    slot_start_seconds,
    slot_end_seconds=None,
    stackelberg_control=None,
    force_center_pickup_on_first_departure=True,
):
    if slot_end_seconds is None:
        slot_end_seconds = slot_start_seconds + float(getattr(config, "EXPERIMENT_TIME_SLOT_MINUTES", 15)) * 60

    print(">> Start enhanced IMTAO-style center-prepacked assignment (our backend)...")

    imtao_centers = []
    imtao_workers = []
    imtao_tasks = []
    center_worker_map = {}
    center_task_map = {}

    worker_node_map = {}
    task_node_map = {}
    task_reward_map = {}
    task_expire_map = {}
    path_time_cache = {}

    def route_travel_time(src_node, dst_node):
        if src_node == dst_node:
            return 0.0
        pair = (src_node, dst_node) if str(src_node) < str(dst_node) else (dst_node, src_node)
        if pair not in path_time_cache:
            try:
                dist = nx.shortest_path_length(G, source=src_node, target=dst_node, weight="length")
                path_time_cache[pair] = dist / config.WORKER_SPEED_MS
            except nx.NetworkXNoPath:
                path_time_cache[pair] = float("inf")
        return path_time_cache[pair]

    for rid, c_node in centers_dict.items():
        c_lon = G.nodes[c_node].get("x", G.nodes[c_node].get("lon"))
        c_lat = G.nodes[c_node].get("y", G.nodes[c_node].get("lat"))
        center = IMTAOCenter(rid, c_lon, c_lat, node=c_node)
        imtao_centers.append(center)
        center_worker_map[rid] = []
        center_task_map[rid] = []

    for rid, w_list in workers_per_center.items():
        for w in w_list:
            w_node, wid, w_lon, w_lat, _ = w
            worker = IMTAOWorker(wid, w_lon, w_lat, max_t=config.MAX_TASKS_PER_WORKER, node=w_node)
            imtao_workers.append(worker)
            center_worker_map[rid].append(worker)
            worker_node_map[wid] = w_node

    for rid, t_list in tasks_per_center.items():
        for t in t_list:
            t_node, tid, reward, expire_seconds = t[0], t[1], t[2], t[3]
            release_seconds = t[4] if len(t) > 4 else slot_start_seconds

            t_lon = G.nodes[t_node].get("x", G.nodes[t_node].get("lon"))
            t_lat = G.nodes[t_node].get("y", G.nodes[t_node].get("lat"))

            relative_expire_seconds = max(0, expire_seconds - slot_start_seconds)
            relative_release_seconds = max(0, release_seconds - slot_start_seconds)
            task = IMTAOTask(
                tid,
                t_lon,
                t_lat,
                expire_time=relative_expire_seconds,
                release_time=relative_release_seconds,
                node=t_node,
            )
            imtao_tasks.append(task)
            center_task_map[rid].append(task)
            task_node_map[tid] = t_node
            task_reward_map[tid] = reward
            task_expire_map[tid] = expire_seconds

    if not imtao_tasks:
        return {}, 0, []

    stackelberg_control = stackelberg_control or {}

    framework = EnhancedIMTAOAssignmentFramework(
        imtao_centers,
        imtao_tasks,
        imtao_workers,
        travel_time_func=route_travel_time,
        slot_duration_seconds=max(0.0, slot_end_seconds - slot_start_seconds),
        task_reward_map=task_reward_map,
        region_priority_weight=stackelberg_control.get("region_priority_weight"),
        worker_completion_bonus=stackelberg_control.get("worker_completion_bonus", 0.0),
        worker_distance_penalty=stackelberg_control.get("worker_distance_penalty", 0.0),
        same_worker_chain_bonus=stackelberg_control.get("same_worker_chain_bonus", 0.0),
        force_center_pickup_on_first_departure=force_center_pickup_on_first_departure,
    )
    framework.initialize_existing_partition(center_task_map, center_worker_map)

    for center in framework.centers:
        framework.algo2_sequential_assignment(center, center.W)

    slot_assignments = {}
    slot_details = []
    slot_profit = 0.0

    for c in framework.centers:
        center_node = centers_dict[c.id]
        for w, assigned_tasks in c.A:
            if not assigned_tasks:
                continue

            worker_node = worker_node_map[w.id]
            if force_center_pickup_on_first_departure:
                try:
                    dist_to_center = nx.shortest_path_length(G, worker_node, center_node, weight="length")
                except nx.NetworkXNoPath:
                    continue
                current_node = center_node
                current_finish_time = slot_start_seconds + dist_to_center / config.WORKER_SPEED_MS
                first_departure_pending = True
            else:
                dist_to_center = 0.0
                current_node = worker_node
                current_finish_time = slot_start_seconds
                first_departure_pending = False
            round_load = 0

            for task in assigned_tasks:
                if round_load >= config.MAX_TASKS_PER_WORKER:
                    try:
                        return_dist = nx.shortest_path_length(G, current_node, center_node, weight="length")
                    except nx.NetworkXNoPath:
                        break

                    return_finish_time = current_finish_time + return_dist / config.WORKER_SPEED_MS
                    if return_finish_time > slot_end_seconds:
                        break

                    current_finish_time = return_finish_time
                    current_node = center_node
                    round_load = 0

                task_node = task_node_map[task.id]
                try:
                    dist_to_task = nx.shortest_path_length(G, current_node, task_node, weight="length")
                except nx.NetworkXNoPath:
                    continue

                dist_to_center_cost = dist_to_center if first_departure_pending else 0.0
                reward = task_reward_map[task.id]
                candidate_finish_time = max(
                    current_finish_time + dist_to_task / config.WORKER_SPEED_MS,
                    slot_start_seconds + task.r,
                )
                if candidate_finish_time > task_expire_map[task.id] or candidate_finish_time > slot_end_seconds:
                    break

                end_time = candidate_finish_time
                end_node = task_node
                service_finish_time = candidate_finish_time
                next_round_load = round_load + 1
                return_dist_cost = 0.0

                if next_round_load >= config.MAX_TASKS_PER_WORKER:
                    try:
                        return_dist = nx.shortest_path_length(G, task_node, center_node, weight="length")
                    except nx.NetworkXNoPath:
                        return_dist = float("inf")

                    if return_dist != float("inf"):
                        return_finish_time = candidate_finish_time + return_dist / config.WORKER_SPEED_MS
                        if return_finish_time <= slot_end_seconds:
                            return_dist_cost = return_dist
                            end_time = return_finish_time
                            end_node = center_node

                total_cost = (dist_to_center_cost + dist_to_task + return_dist_cost) * config.TRAVEL_COST_PER_METER
                profit = reward - total_cost

                slot_assignments[(w.id, task.id)] = profit
                slot_details.append(
                    {
                        "region_id": c.id,
                        "wid": w.id,
                        "task_id": task.id,
                        "dist_to_center": dist_to_center_cost,
                        "dist_to_task": dist_to_task,
                        "return_to_center_dist": return_dist_cost,
                        "task_node": end_node,
                        "service_node": task_node,
                        "reward": reward,
                        "cost": total_cost,
                        "finish_time": end_time,
                        "service_finish_time": service_finish_time,
                        "end_time": end_time,
                        "end_node": end_node,
                        "profit": profit,
                    }
                )
                slot_profit += profit

                current_finish_time = end_time
                current_node = end_node
                round_load = 0 if end_node == center_node else next_round_load
                first_departure_pending = False

    print(
        f"✅ Enhanced IMTAO-style assignment finished: "
        f"assigned={len(slot_assignments)}, profit={slot_profit:.2f}"
    )
    return slot_assignments, slot_profit, slot_details
