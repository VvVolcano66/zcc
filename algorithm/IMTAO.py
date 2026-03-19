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
    计算两点间的行驶时间 (对应论文中的 tt(l_a, l_b))
    为了算法复现速度，这里使用欧式距离除以速度的近似。
    你也可以替换为调用 get_distance.py 中的路网距离。
    """
    # 粗略将经纬度欧式距离转为米 (仅作算法逻辑复现)
    dist_m = np.sqrt((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2) * 111320
    return dist_m / speed


class IMTAO_Framework:
    def __init__(self, centers: List[Center], tasks: List[Task], workers: List[Worker]):
        self.centers = centers
        self.tasks = tasks
        self.workers = workers

    # =====================================================================
    # Algorithm 1: Voronoi-based Service Area Partition (Voronoi 区域划分)
    # =====================================================================
    def algo1_voronoi_partition(self):
        """将全局的任务和工人分配到距离最近的中心"""
        for c in self.centers:
            c.S = []
            c.W = []

        # 划分任务
        for s in self.tasks:
            nearest_c = min(self.centers, key=lambda c: calculate_travel_time(s.lon, s.lat, c.lon, c.lat))
            nearest_c.S.append(s)

        # 划分工人
        for w in self.workers:
            nearest_c = min(self.centers, key=lambda c: calculate_travel_time(w.lon, w.lat, c.lon, c.lat))
            nearest_c.W.append(w)

        # 初始化分配状态
        for c in self.centers:
            c.S_left = c.S.copy()
            c.W_left = c.W.copy()
            c.A = []
            c.rho = 0.0

    # =====================================================================
    # Algorithm 2: Sequential Task Assignment Algorithm (顺序任务分配算法)
    # =====================================================================
    def algo2_sequential_assignment(self, center: Center, workers_to_assign: List[Worker]):
        """
        对指定中心的工人进行贪心顺序分配 (考虑时空约束)
        """
        center.A = []
        center.S_left = center.S.copy()
        used_workers = []

        # 1. 按照工人到中心的距离降序排列 (优先分配边缘工人，论文核心策略)
        workers_sorted = sorted(
            workers_to_assign,
            key=lambda w: calculate_travel_time(w.lon, w.lat, center.lon, center.lat),
            reverse=True
        )

        for w in workers_sorted:
            S_w = []  # 分配给该工人的任务序列 VTDS(w)

            # 初始时间: 工人前往中心取货的时间
            current_time = calculate_travel_time(w.lon, w.lat, center.lon, center.lat)
            current_lon, current_lat = center.lon, center.lat  # 取货后位置在中心

            while len(S_w) < w.maxT and len(center.S_left) > 0:
                # 寻找距离当前位置最近的未分配任务
                nearest_task = min(
                    center.S_left,
                    key=lambda s: calculate_travel_time(current_lon, current_lat, s.lon, s.lat)
                )

                travel_time = calculate_travel_time(current_lon, current_lat, nearest_task.lon, nearest_task.lat)

                # 检查时间约束: 当前时间 + 行驶时间 < 任务过期时间 e
                if current_time + travel_time < nearest_task.e:
                    center.S_left.remove(nearest_task)
                    S_w.append(nearest_task)
                    current_time += travel_time
                    current_lon, current_lat = nearest_task.lon, nearest_task.lat
                else:
                    # 最近的任务都超时了，退出该工人的分配
                    break

            if len(S_w) > 0:
                center.A.append((w, S_w))
                used_workers.append(w)

        # 计算该中心剩余的空闲工人和分配率 \rho
        center.W_left = [w for w in workers_to_assign if w not in used_workers]
        center.rho = len(center.S) - len(center.S_left)
        if len(center.S) > 0:
            center.rho = center.rho / len(center.S)
        else:
            center.rho = 1.0

    # =====================================================================
    # Algorithm 3: Game-Theoretic Multi-Center Collaboration (博弈论协同)
    # =====================================================================
    def algo3_game_theoretic_collaboration(self):
        """
        跨中心劳动力转移 (Inter-center Workforce Transfer)
        目标: 最小化协同不公平性 U_\rho，最大化总任务数
        """
        # 第一阶段: 独立分配 (Center-independent)
        self.algo1_voronoi_partition()
        for c in self.centers:
            self.algo2_sequential_assignment(c, c.W)

        # 收集全局所有中心的剩余空闲工人 (C.W_left)
        global_W_left = []
        for c in self.centers:
            global_W_left.extend(c.W_left)

        # 寻找需要援助的接收中心 (rho < 1.0)
        C_prime = [c for c in self.centers if c.rho < 1.0]

        iteration = 1
        while True:
            if len(C_prime) == 0 or len(global_W_left) == 0:
                break  # 没有接收中心或没有多余工人，达到纳什均衡

            # 1. Best-response 机制: 挑选当前分配率 rho 最低的中心 c_i
            c_i = min(C_prime, key=lambda c: c.rho)

            # 2. 尝试派遣一个空闲工人给 c_i
            w_move = global_W_left[0]

            # 备份当前状态以供回滚
            original_rho = c_i.rho
            original_A = copy.deepcopy(c_i.A)
            original_S_left = copy.deepcopy(c_i.S_left)

            # 3. Bi-directional Optimization (BDC): 将原始工人和新借来的工人放在一起重新运行 Algo2
            combined_workers = c_i.W + [w_move]
            self.algo2_sequential_assignment(c_i, combined_workers)

            # 4. 判断收益
            if c_i.rho > original_rho:
                # 策略有效：接受该调度
                global_W_left.pop(0)  # 正式移除该工人
                c_i.W.append(w_move)  # 将工人正式编制入该中心
            else:
                # 策略无效：回滚状态，并将 c_i 移出候选名单
                c_i.rho = original_rho
                c_i.A = original_A
                c_i.S_left = original_S_left
                C_prime.remove(c_i)

            iteration += 1

        # 计算最终的不公平性 U_rho
        rhos = [c.rho for c in self.centers]
        u_rho = 0
        for i in range(len(rhos)):
            for j in range(len(rhos)):
                if i != j:
                    u_rho += abs(rhos[i] - rhos[j])
        u_rho = u_rho / (len(self.centers) * (len(self.centers) - 1))

        total_assigned = sum(len(c.S) - len(c.S_left) for c in self.centers)
        return total_assigned, u_rho