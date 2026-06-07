import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import config


IMTAO_MODE_BDC = "bdc"
IMTAO_MODE_RBDC = "rbdc"
IMTAO_MODE_DC = "dc"
IMTAO_MODE_WO_C = "wo_c"
IMTAO_SUPPORTED_MODES = {IMTAO_MODE_BDC, IMTAO_MODE_RBDC, IMTAO_MODE_DC, IMTAO_MODE_WO_C}

IMTAO_SELECT_LOWEST_RHO = "lowest_rho"
IMTAO_SELECT_RANDOM = "random"
IMTAO_SUPPORTED_SELECTIONS = {IMTAO_SELECT_LOWEST_RHO, IMTAO_SELECT_RANDOM}

IMTAO_DEFAULT_MAX_COLLAB_ITERATIONS = 20


@dataclass(eq=False)
class Task:
    id: str
    lon: float
    lat: float
    e: float
    r: float = 0.0
    node: Optional[object] = None

    def __init__(
        self,
        t_id,
        lon,
        lat,
        expire_time=None,
        release_time=0.0,
        node=None,
        e=None,
        r=None,
    ):
        self.id = t_id
        self.lon = lon
        self.lat = lat
        self.e = expire_time if expire_time is not None else e
        if self.e is None:
            raise ValueError("Task requires expire_time or e")
        self.r = release_time if r is None else r
        self.node = node


@dataclass(eq=False)
class Worker:
    id: str
    lon: float
    lat: float
    maxT: int = 4
    node: Optional[object] = None

    def __init__(
        self,
        w_id,
        lon,
        lat,
        max_t=None,
        node=None,
        maxT=None,
    ):
        self.id = w_id
        self.lon = lon
        self.lat = lat
        self.maxT = max_t if max_t is not None else (maxT if maxT is not None else 4)
        self.node = node


@dataclass(eq=False)
class Center:
    id: int
    lon: float
    lat: float
    node: Optional[object] = None
    S: List[Task] = field(default_factory=list)
    W: List[Worker] = field(default_factory=list)
    S_left: List[Task] = field(default_factory=list)
    W_left: List[Worker] = field(default_factory=list)
    A: List[Tuple[Worker, List[Task]]] = field(default_factory=list)
    rho: float = 0.0
    borrowed_workers: List[Worker] = field(default_factory=list)


def calculate_travel_time(lon1, lat1, lon2, lat2, speed=None):
    if speed is None:
        speed = config.WORKER_SPEED_MS

    if abs(lon1) > 180 or abs(lat1) > 90:
        dist_m = np.sqrt((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2)
    else:
        mean_lat_rad = np.radians((lat1 + lat2) / 2.0)
        dx = (lon1 - lon2) * 111320.0 * np.cos(mean_lat_rad)
        dy = (lat1 - lat2) * 111320.0
        dist_m = np.sqrt(dx ** 2 + dy ** 2)

    return dist_m / speed


class IMTAO_Framework:
    """
    IMTAO task-assignment framework.

    This implementation follows the paper's two-phase structure more closely:
    1. Center-independent task assignment.
    2. Game-theoretic multi-center collaboration with explicit BDC / RBDC / DC modes.

    If no partition has been initialized beforehand, the framework falls back to the
    paper-style Voronoi partition before running the two phases.
    """

    def __init__(
        self,
        centers: List[Center],
        tasks: List[Task],
        workers: List[Worker],
        travel_time_func=None,
        slot_duration_seconds: float = float("inf"),
        random_seed: int = 42,
        fairness_penalty_weight: float = 1.0,
        collaboration_max_iterations: int = IMTAO_DEFAULT_MAX_COLLAB_ITERATIONS,
    ):
        self.centers = centers
        self.tasks = tasks
        self.workers = workers
        self.travel_time_func = travel_time_func
        self.slot_duration_seconds = slot_duration_seconds
        self.rng = random.Random(random_seed)
        self.fairness_penalty_weight = float(fairness_penalty_weight)
        self.collaboration_max_iterations = max(1, int(collaboration_max_iterations))
        self.worker_home_center: Dict[str, Center] = {}
        self.partition_initialized = False

    def _travel_time(self, src, dst) -> float:
        src_node = getattr(src, "node", None)
        dst_node = getattr(dst, "node", None)
        if self.travel_time_func is not None and src_node is not None and dst_node is not None:
            travel_t = self.travel_time_func(src_node, dst_node)
            if travel_t is not None:
                return travel_t
        return calculate_travel_time(src.lon, src.lat, dst.lon, dst.lat)

    def _calculate_collaboration_unfairness(self) -> float:
        if len(self.centers) <= 1:
            return 0.0

        rhos = [c.rho for c in self.centers]
        u_rho = 0.0
        for i in range(len(rhos)):
            for j in range(len(rhos)):
                if i != j:
                    u_rho += abs(rhos[i] - rhos[j])
        return u_rho / (len(self.centers) * (len(self.centers) - 1))

    def _calculate_center_utility_from_rho(self, center_id: int, rho_value: float) -> float:
        if len(self.centers) <= 1:
            return rho_value

        other_rhos = [c.rho for c in self.centers if c.id != center_id]
        if not other_rhos:
            return rho_value
        unfairness_penalty = sum(abs(rho_value - other_rho) for other_rho in other_rhos) / len(other_rhos)
        return rho_value - self.fairness_penalty_weight * unfairness_penalty

    def _calculate_center_utility(self, center: Center) -> float:
        return self._calculate_center_utility_from_rho(center.id, center.rho)

    def _compute_rho(self, center: Center, assignments: Sequence[Tuple[Worker, List[Task]]]) -> float:
        if not center.S:
            return 1.0
        assigned_tasks = sum(len(task_seq) for _, task_seq in assignments)
        return assigned_tasks / len(center.S)

    def _clone_assignment_state(self, center: Center, borrowed_workers: Optional[List[Worker]] = None):
        return {
            "rho": center.rho,
            "A": [(worker, list(task_seq)) for worker, task_seq in center.A],
            "S_left": list(center.S_left),
            "W_left": list(center.W_left),
            "borrowed_workers": list(center.borrowed_workers if borrowed_workers is None else borrowed_workers),
        }

    def _apply_assignment_state(self, center: Center, state) -> None:
        center.rho = state["rho"]
        center.A = [(worker, list(task_seq)) for worker, task_seq in state["A"]]
        center.S_left = list(state["S_left"])
        center.W_left = list(state["W_left"])
        center.borrowed_workers = list(state["borrowed_workers"])

    def _find_nearest_feasible_task(
        self,
        center: Center,
        current_ref,
        current_time: float,
        remaining_tasks: Sequence[Task],
    ) -> Tuple[Optional[Task], Optional[float]]:
        nearest_task = None
        nearest_travel = None
        nearest_finish = None
        hard_deadline = self.slot_duration_seconds

        for task in remaining_tasks:
            travel_t = self._travel_time(current_ref, task)
            if not np.isfinite(travel_t):
                continue

            depart_time = max(current_time, task.r)
            finish_time = depart_time + travel_t
            deadline = min(task.e, hard_deadline)
            if finish_time > deadline:
                continue

            if nearest_task is None or travel_t < nearest_travel:
                nearest_task = task
                nearest_travel = travel_t
                nearest_finish = finish_time

        return nearest_task, nearest_finish

    def _build_task_sequence(
        self,
        center: Center,
        worker: Worker,
        candidate_tasks: Sequence[Task],
    ) -> Tuple[List[Task], float]:
        current_time = self._travel_time(worker, center)
        if not np.isfinite(current_time) or current_time > self.slot_duration_seconds:
            return [], current_time

        current_ref = center
        remaining_tasks = list(candidate_tasks)
        selected_tasks: List[Task] = []
        round_load = 0

        while remaining_tasks:
            if round_load >= worker.maxT:
                return_time = self._travel_time(current_ref, center)
                if not np.isfinite(return_time) or current_time + return_time > self.slot_duration_seconds:
                    break
                current_time += return_time
                current_ref = center
                round_load = 0

            nearest_task, next_finish_time = self._find_nearest_feasible_task(
                center=center,
                current_ref=current_ref,
                current_time=current_time,
                remaining_tasks=remaining_tasks,
            )
            if nearest_task is None:
                break

            selected_tasks.append(nearest_task)
            remaining_tasks = [task for task in remaining_tasks if task.id != nearest_task.id]
            current_time = next_finish_time
            current_ref = nearest_task
            round_load += 1

        return selected_tasks, current_time

    def _run_sequential_assignment(self, center: Center, workers_to_assign: Sequence[Worker]):
        assignments: List[Tuple[Worker, List[Task]]] = []
        remaining_tasks = list(center.S)
        unused_workers: List[Worker] = []

        workers_sorted = sorted(
            workers_to_assign,
            key=lambda worker: self._travel_time(worker, center),
            reverse=True,
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

    def _run_decomposed_assignment(self, center: Center, dispatched_worker: Worker):
        remaining_tasks = list(center.S_left)
        task_seq, _ = self._build_task_sequence(center, dispatched_worker, remaining_tasks)
        if task_seq:
            assigned_ids = {task.id for task in task_seq}
            remaining_tasks = [task for task in remaining_tasks if task.id not in assigned_ids]

        assignments = [(worker, list(task_seq_existing)) for worker, task_seq_existing in center.A]
        assignments.append((dispatched_worker, task_seq))

        unused_workers = [worker for worker in center.W_left if worker.id != dispatched_worker.id]
        if not task_seq:
            unused_workers.append(dispatched_worker)

        rho = self._compute_rho(center, assignments)
        return {
            "rho": rho,
            "A": assignments,
            "S_left": remaining_tasks,
            "W_left": unused_workers,
        }

    def _select_best_worker_for_center(
        self,
        center: Center,
        available_workers: Sequence[Worker],
        collaboration_mode: str,
    ):
        best_worker = None
        best_state = None
        best_utility = self._calculate_center_utility(center)

        for worker in available_workers:
            if collaboration_mode == IMTAO_MODE_DC:
                candidate_state = self._run_decomposed_assignment(center, worker)
            else:
                candidate_workers = list(center.W) + list(center.borrowed_workers) + [worker]
                candidate_state = self._run_sequential_assignment(center, candidate_workers)

            candidate_utility = self._calculate_center_utility_from_rho(center.id, candidate_state["rho"])
            if (
                candidate_utility > best_utility
                or (
                    np.isclose(candidate_utility, best_utility)
                    and best_state is not None
                    and candidate_state["rho"] > best_state["rho"]
                )
            ):
                best_worker = worker
                best_utility = candidate_utility
                best_state = {
                    **candidate_state,
                    "borrowed_workers": list(center.borrowed_workers) + [worker],
                }

        return best_worker, best_state

    def _resolve_center_selection(
        self,
        collaboration_mode: str,
        center_selection: Optional[str],
    ) -> str:
        if collaboration_mode == IMTAO_MODE_RBDC:
            return IMTAO_SELECT_RANDOM
        if center_selection is None:
            return IMTAO_SELECT_LOWEST_RHO
        return center_selection

    def _select_recipient_center(
        self,
        pending_centers: Sequence[Center],
        center_selection: str,
    ) -> Center:
        if center_selection == IMTAO_SELECT_RANDOM:
            return self.rng.choice(list(pending_centers))
        return min(
            pending_centers,
            key=lambda center: (
                self._calculate_center_utility(center),
                center.rho,
                center.id,
            ),
        )

    def algo1_voronoi_partition(self):
        for center in self.centers:
            center.S = []
            center.W = []

        for task in self.tasks:
            nearest_center = min(self.centers, key=lambda center: self._travel_time(task, center))
            nearest_center.S.append(task)

        for worker in self.workers:
            nearest_center = min(self.centers, key=lambda center: self._travel_time(worker, center))
            nearest_center.W.append(worker)
            self.worker_home_center[worker.id] = nearest_center

        for center in self.centers:
            center.S_left = list(center.S)
            center.W_left = list(center.W)
            center.A = []
            center.rho = 0.0
            center.borrowed_workers = []
        self.partition_initialized = True

    def initialize_existing_partition(
        self,
        center_to_tasks: Dict[int, List[Task]],
        center_to_workers: Dict[int, List[Worker]],
    ):
        self.worker_home_center = {}
        for center in self.centers:
            center.S = list(center_to_tasks.get(center.id, []))
            center.W = list(center_to_workers.get(center.id, []))
            center.S_left = list(center.S)
            center.W_left = list(center.W)
            center.A = []
            center.rho = 0.0
            center.borrowed_workers = []
            for worker in center.W:
                self.worker_home_center[worker.id] = center
        self.partition_initialized = True

    def algo2_sequential_assignment(self, center: Center, workers_to_assign: Optional[Sequence[Worker]] = None):
        candidate_workers = center.W if workers_to_assign is None else list(workers_to_assign)
        assignment_state = self._run_sequential_assignment(center, candidate_workers)
        self._apply_assignment_state(
            center,
            {
                **assignment_state,
                "borrowed_workers": list(center.borrowed_workers),
            },
        )

    def algo3_game_theoretic_collaboration(
        self,
        repartition: bool = False,
        collaboration_mode: str = IMTAO_MODE_BDC,
        center_selection: Optional[str] = None,
    ):
        if collaboration_mode not in IMTAO_SUPPORTED_MODES:
            raise ValueError(f"Unsupported IMTAO collaboration mode: {collaboration_mode}")
        if center_selection is not None and center_selection not in IMTAO_SUPPORTED_SELECTIONS:
            raise ValueError(f"Unsupported IMTAO center selection mode: {center_selection}")

        if repartition or not self.partition_initialized:
            self.algo1_voronoi_partition()

        effective_center_selection = self._resolve_center_selection(
            collaboration_mode=collaboration_mode,
            center_selection=center_selection,
        )

        for center in self.centers:
            center.borrowed_workers = []
            self.algo2_sequential_assignment(center, center.W)

        if collaboration_mode == IMTAO_MODE_WO_C:
            total_assigned = sum(sum(len(task_seq) for _, task_seq in center.A) for center in self.centers)
            return total_assigned, self._calculate_collaboration_unfairness()

        pending_centers = [center for center in self.centers if center.rho < 1.0]
        global_w_left = [worker for center in self.centers for worker in center.W_left]
        iteration_count = 0

        while pending_centers and global_w_left and iteration_count < self.collaboration_max_iterations:
            iteration_count += 1
            current_center = self._select_recipient_center(
                pending_centers=pending_centers,
                center_selection=effective_center_selection,
            )

            worker_to_move, best_state = self._select_best_worker_for_center(
                current_center,
                global_w_left,
                collaboration_mode=IMTAO_MODE_DC if collaboration_mode == IMTAO_MODE_DC else IMTAO_MODE_BDC,
            )

            if worker_to_move is None or best_state is None or best_state["rho"] <= current_center.rho:
                pending_centers = [center for center in pending_centers if center.id != current_center.id]
                continue

            self._apply_assignment_state(current_center, best_state)
            global_w_left = [worker for worker in global_w_left if worker.id != worker_to_move.id]

            donor_center = self.worker_home_center.get(worker_to_move.id)
            if donor_center is not None:
                donor_center.W_left = [worker for worker in donor_center.W_left if worker.id != worker_to_move.id]

            pending_centers = [center for center in self.centers if center.rho < 1.0]

        total_assigned = sum(sum(len(task_seq) for _, task_seq in center.A) for center in self.centers)
        return total_assigned, self._calculate_collaboration_unfairness()
