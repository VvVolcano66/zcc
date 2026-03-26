import numpy as np
import copy
from typing import List, Dict

import config


class Task:
    def __init__(self, t_id, lon, lat, expire_time, release_time=0, node=None):
        self.id = t_id
        self.lon = lon
        self.lat = lat
        self.e = expire_time  # 论文中的 s.e (expiration time)
        self.r = release_time
        self.node = node


class Worker:
    def __init__(self, w_id, lon, lat, max_t=4, node=None):
        self.id = w_id
        self.lon = lon
        self.lat = lat
        self.maxT = max_t  # 论文中的 w.maxT (task capacity)
        self.node = node


class Center:
    def __init__(self, c_id, lon, lat, node=None):
        self.id = c_id
        self.lon = lon
        self.lat = lat
        self.node = node
        self.S = []  # 分配给该中心的任务 c.S
        self.W = []  # 该中心的原始工人 c.W

        # 算法运行过程中的状态
        self.S_left = []  # 未分配的任务 c.S_left
        self.W_left = []  # 未使用的工人 c.W_left
        self.A = []  # 任务分配结果 A(c) -> [(worker, [task1, task2...])]
        self.rho = 0.0  # 任务分配率 \rho_i


def calculate_travel_time(lon1, lat1, lon2, lat2, speed=None):
    """
    计算两点间的行驶时间
    💡 核心修复：坐标系投影自适应！
    """
    if speed is None:
        speed = config.WORKER_SPEED_MS

    # 如果坐标绝对值大于 180，说明传入的已经是投影后的米制坐标 (如 UTM)，无需经纬度换算
    if abs(lon1) > 180 or abs(lat1) > 90:
        dist_m = np.sqrt((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2)
    else:
        # 传入的是标准 GPS 经纬度 (EPSG:4326)，需要乘以地球赤道常数换算成米
        mean_lat_rad = np.radians((lat1 + lat2) / 2.0)
        dx = (lon1 - lon2) * 111320.0 * np.cos(mean_lat_rad)
        dy = (lat1 - lat2) * 111320.0
        dist_m = np.sqrt(dx ** 2 + dy ** 2)

    return dist_m / speed


class IMTAO_Framework:
    def __init__(self, centers: List[Center], tasks: List[Task], workers: List[Worker], travel_time_func=None):
        self.centers = centers
        self.tasks = tasks
        self.workers = workers
        self.travel_time_func = travel_time_func

    def _travel_time(self, src, dst) -> float:
        src_node = getattr(src, 'node', None)
        dst_node = getattr(dst, 'node', None)
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

    def _calculate_center_utility(self, center: Center) -> float:
        """
        Paper Eq. (5): UUP(ci, BWS(ci)) = rho_i - average_{j != i} rho_j
        """
        if len(self.centers) <= 1:
            return center.rho

        other_rhos = [c.rho for c in self.centers if c.id != center.id]
        return center.rho - sum(other_rhos) / len(other_rhos)

    def _select_best_worker_for_center(self, center: Center, available_workers: List[Worker]):
        """
        Best-response step in the paper:
        evaluate each candidate borrowed worker, re-run the sequential
        assignment for the recipient center, and keep the worker that
        maximizes the recipient center's utility.
        """
        best_worker = None
        best_utility = self._calculate_center_utility(center)
        best_snapshot = None

        original_state = (center.rho, copy.deepcopy(center.A), copy.deepcopy(center.S_left), copy.deepcopy(center.W_left))

        for worker in available_workers:
            combined_workers = center.W + [worker]
            self.algo2_sequential_assignment(center, combined_workers)
            candidate_utility = self._calculate_center_utility(center)

            if candidate_utility > best_utility:
                best_worker = worker
                best_utility = candidate_utility
                best_snapshot = (
                    center.rho,
                    copy.deepcopy(center.A),
                    copy.deepcopy(center.S_left),
                    copy.deepcopy(center.W_left)
                )

        center.rho, center.A, center.S_left, center.W_left = original_state
        return best_worker, best_snapshot

    # =====================================================================
    # Algorithm 1: Voronoi-based Service Area Partition
    # =====================================================================
    def algo1_voronoi_partition(self):
        for c in self.centers:
            c.S = []
            c.W = []

        for s in self.tasks:
            nearest_c = min(self.centers, key=lambda c: self._travel_time(s, c))
            nearest_c.S.append(s)

        for w in self.workers:
            nearest_c = min(self.centers, key=lambda c: self._travel_time(w, c))
            nearest_c.W.append(w)

        for c in self.centers:
            c.S_left = c.S.copy()
            c.W_left = c.W.copy()
            c.A = []
            c.rho = 0.0

    def initialize_existing_partition(self, center_to_tasks: Dict[int, List[Task]], center_to_workers: Dict[int, List[Worker]]):
        for c in self.centers:
            c.S = list(center_to_tasks.get(c.id, []))
            c.W = list(center_to_workers.get(c.id, []))
            c.S_left = c.S.copy()
            c.W_left = c.W.copy()
            c.A = []
            c.rho = 0.0

    # =====================================================================
    # Algorithm 2: Sequential Task Assignment Algorithm
    # =====================================================================
    def algo2_sequential_assignment(self, center: Center, workers_to_assign: List[Worker]):
        center.A = []
        center.S_left = center.S.copy()
        used_workers = []

        # 按照工人到中心的距离降序排列 (优先分配边缘工人)
        workers_sorted = sorted(
            workers_to_assign,
            key=lambda w: self._travel_time(w, center),
            reverse=True
        )

        for w in workers_sorted:
            S_w = []
            current_time = self._travel_time(w, center)
            current_ref = center

            while len(S_w) < w.maxT and len(center.S_left) > 0:
                valid_tasks = []
                for s in center.S_left:
                    travel_t = self._travel_time(current_ref, s)
                    finish_time = max(current_time + travel_t, s.r)
                    if finish_time <= s.e:
                        valid_tasks.append((s, travel_t, finish_time))

                if not valid_tasks:
                    break

                nearest_task, best_travel_time, best_finish_time = min(valid_tasks, key=lambda x: x[1])

                center.S_left.remove(nearest_task)
                S_w.append(nearest_task)
                current_time = best_finish_time
                current_ref = nearest_task

            if len(S_w) > 0:
                center.A.append((w, S_w))
                used_workers.append(w)

        center.W_left = [w for w in workers_to_assign if w not in used_workers]

        assigned_tasks_count = sum(len(tasks) for worker, tasks in center.A)
        if len(center.S) > 0:
            center.rho = assigned_tasks_count / len(center.S)
        else:
            center.rho = 1.0

    # =====================================================================
    # Algorithm 3: Game-Theoretic Multi-Center Collaboration
    # =====================================================================
    def algo3_game_theoretic_collaboration(self, repartition: bool = True):
        if repartition:
            self.algo1_voronoi_partition()
        for c in self.centers:
            self.algo2_sequential_assignment(c, c.W)

        C_prime = [c for c in self.centers if c.rho < 1.0]
        worker_owner = {w.id: c for c in self.centers for w in c.W}

        iteration = 1
        while True:
            global_W_left = [w for c in self.centers for w in c.W_left]
            if len(C_prime) == 0 or len(global_W_left) == 0:
                break

            c_i = min(C_prime, key=lambda c: c.rho)

            w_move, best_snapshot = self._select_best_worker_for_center(c_i, global_W_left)

            if w_move is not None and best_snapshot is not None:
                c_i.rho, c_i.A, c_i.S_left, c_i.W_left = best_snapshot
                if all(existing.id != w_move.id for existing in c_i.W):
                    c_i.W.append(w_move)
                donor_center = worker_owner.get(w_move.id)
                if donor_center is not None:
                    donor_center.W_left = [w for w in donor_center.W_left if w.id != w_move.id]
            else:
                C_prime.remove(c_i)

            iteration += 1

        u_rho = self._calculate_collaboration_unfairness()
        total_assigned = sum(sum(len(tasks) for w, tasks in c.A) for c in self.centers)
        return total_assigned, u_rho
