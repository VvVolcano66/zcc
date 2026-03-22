import numpy as np
import copy
from typing import List, Dict


class Task:
    def __init__(self, t_id, lon, lat, expire_time):
        self.id = t_id
        self.lon = lon
        self.lat = lat
        self.e = expire_time  # 论文中的 s.e (expiration time)


class Worker:
    def __init__(self, w_id, lon, lat, max_t=4):
        self.id = w_id
        self.lon = lon
        self.lat = lat
        self.maxT = max_t  # 论文中的 w.maxT (task capacity)


class Center:
    def __init__(self, c_id, lon, lat):
        self.id = c_id
        self.lon = lon
        self.lat = lat
        self.S = []  # 分配给该中心的任务 c.S
        self.W = []  # 该中心的原始工人 c.W

        # 算法运行过程中的状态
        self.S_left = []  # 未分配的任务 c.S_left
        self.W_left = []  # 未使用的工人 c.W_left
        self.A = []  # 任务分配结果 A(c) -> [(worker, [task1, task2...])]
        self.rho = 0.0  # 任务分配率 \rho_i


def calculate_travel_time(lon1, lat1, lon2, lat2, speed=5.0):
    """
    计算两点间的行驶时间
    加入了对成都纬度(北纬30度左右)的 cos 补偿，使得空间距离估算更准。
    """
    mean_lat_rad = np.radians((lat1 + lat2) / 2.0)
    dx = (lon1 - lon2) * 111320.0 * np.cos(mean_lat_rad)
    dy = (lat1 - lat2) * 111320.0
    dist_m = np.sqrt(dx ** 2 + dy ** 2)
    return dist_m / speed


class IMTAO_Framework:
    def __init__(self, centers: List[Center], tasks: List[Task], workers: List[Worker]):
        self.centers = centers
        self.tasks = tasks
        self.workers = workers

    # =====================================================================
    # Algorithm 1: Voronoi-based Service Area Partition
    # =====================================================================
    def algo1_voronoi_partition(self):
        for c in self.centers:
            c.S = []
            c.W = []

        for s in self.tasks:
            nearest_c = min(self.centers, key=lambda c: calculate_travel_time(s.lon, s.lat, c.lon, c.lat))
            nearest_c.S.append(s)

        for w in self.workers:
            nearest_c = min(self.centers, key=lambda c: calculate_travel_time(w.lon, w.lat, c.lon, c.lat))
            nearest_c.W.append(w)

        for c in self.centers:
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
            key=lambda w: calculate_travel_time(w.lon, w.lat, center.lon, center.lat),
            reverse=True
        )

        for w in workers_sorted:
            S_w = []
            current_time = calculate_travel_time(w.lon, w.lat, center.lon, center.lat)
            current_lon, current_lat = center.lon, center.lat

            while len(S_w) < w.maxT and len(center.S_left) > 0:
                # 💡 核心逻辑优化：接单时，只在“自己能在死线前赶到的任务”里挑
                valid_tasks = []
                for s in center.S_left:
                    travel_t = calculate_travel_time(current_lon, current_lat, s.lon, s.lat)
                    # 校验时间窗约束
                    if current_time + travel_t <= s.e:
                        valid_tasks.append((s, travel_t))

                # 如果这个工人发现剩下的所有订单自己都来不及送了
                if not valid_tasks:
                    break  # 该工人停止接单，把任务留在池子里给其他更近的工人

                # 在所有能赶得及的任务里，挑一个距离最近的
                nearest_task, best_travel_time = min(valid_tasks, key=lambda x: x[1])

                # 落实分配
                center.S_left.remove(nearest_task)
                S_w.append(nearest_task)
                current_time += best_travel_time
                current_lon, current_lat = nearest_task.lon, nearest_task.lat

            if len(S_w) > 0:
                center.A.append((w, S_w))
                used_workers.append(w)

        center.W_left = [w for w in workers_to_assign if w not in used_workers]

        # 分配率 rho 的计算
        assigned_tasks_count = sum(len(tasks) for worker, tasks in center.A)
        if len(center.S) > 0:
            center.rho = assigned_tasks_count / len(center.S)
        else:
            center.rho = 1.0

    # =====================================================================
    # Algorithm 3: Game-Theoretic Multi-Center Collaboration
    # =====================================================================
    def algo3_game_theoretic_collaboration(self):
        self.algo1_voronoi_partition()
        for c in self.centers:
            self.algo2_sequential_assignment(c, c.W)

        global_W_left = []
        for c in self.centers:
            global_W_left.extend(c.W_left)

        C_prime = [c for c in self.centers if c.rho < 1.0]

        iteration = 1
        while True:
            if len(C_prime) == 0 or len(global_W_left) == 0:
                break

            c_i = min(C_prime, key=lambda c: c.rho)
            w_move = global_W_left[0]

            original_rho = c_i.rho
            original_A = copy.deepcopy(c_i.A)
            original_S_left = copy.deepcopy(c_i.S_left)

            combined_workers = c_i.W + [w_move]
            self.algo2_sequential_assignment(c_i, combined_workers)

            if c_i.rho > original_rho:
                global_W_left.pop(0)
                c_i.W.append(w_move)
            else:
                c_i.rho = original_rho
                c_i.A = original_A
                c_i.S_left = original_S_left
                C_prime.remove(c_i)

            iteration += 1

        rhos = [c.rho for c in self.centers]
        u_rho = 0
        for i in range(len(rhos)):
            for j in range(len(rhos)):
                if i != j:
                    u_rho += abs(rhos[i] - rhos[j])
        u_rho = u_rho / (len(self.centers) * (len(self.centers) - 1))

        total_assigned = sum(sum(len(tasks) for w, tasks in c.A) for c in self.centers)
        return total_assigned, u_rho