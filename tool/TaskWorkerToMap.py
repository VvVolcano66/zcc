from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import networkx as nx
import config


def snap_to_network(lon: float, lat: float, coords: np.ndarray, nodes: list) -> Any:
    """
    将经纬度坐标映射到路网中最近的节点
    注意：此函数仅供单次、少量调用。若处理大批量数据，请使用向量化批量查询。
    """
    tree = KDTree(coords)
    _, idx = tree.query([[lon, lat]])
    return nodes[idx[0]]


def _out_of_china(lon: float, lat: float) -> bool:
    return not (73.66 < lon < 135.05 and 3.86 < lat < 53.55)


def _transform_lat(x: float, y: float) -> float:
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * np.sqrt(abs(x))
    ret += (20.0 * np.sin(6.0 * x * np.pi) + 20.0 * np.sin(2.0 * x * np.pi)) * 2.0 / 3.0
    ret += (20.0 * np.sin(y * np.pi) + 40.0 * np.sin(y / 3.0 * np.pi)) * 2.0 / 3.0
    ret += (160.0 * np.sin(y / 12.0 * np.pi) + 320 * np.sin(y * np.pi / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lon(x: float, y: float) -> float:
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * np.sqrt(abs(x))
    ret += (20.0 * np.sin(6.0 * x * np.pi) + 20.0 * np.sin(2.0 * x * np.pi)) * 2.0 / 3.0
    ret += (20.0 * np.sin(x * np.pi) + 40.0 * np.sin(x / 3.0 * np.pi)) * 2.0 / 3.0
    ret += (150.0 * np.sin(x / 12.0 * np.pi) + 300.0 * np.sin(x / 30.0 * np.pi)) * 2.0 / 3.0
    return ret


def gcj02_to_wgs84(lon: float, lat: float) -> Tuple[float, float]:
    """
    将 GCJ-02 坐标近似转换为 WGS84，便于和 OSM 路网对齐。
    """
    if _out_of_china(lon, lat):
        return lon, lat

    a = 6378245.0
    ee = 0.00669342162296594323
    dlat = _transform_lat(lon - 105.0, lat - 35.0)
    dlon = _transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * np.pi
    magic = np.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = np.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * np.pi)
    dlon = (dlon * 180.0) / (a / sqrtmagic * np.cos(radlat) * np.pi)
    mg_lat = lat + dlat
    mg_lon = lon + dlon
    return lon * 2 - mg_lon, lat * 2 - mg_lat


class WorkerSimulator:
    """
    工人位置模拟器（带中心取货逻辑）

    工作流程：
    1. 初始化时加载真实工人位置（测试前 5 分钟的数据）
    2. 每次分配任务时，计算路径：工人当前位置 → 中心 → 任务点
    3. 更新位置时，直接更新到任务地点（模拟完成整个配送过程）

    状态说明：
    - 'idle': 空闲，可以接单
    - 'en_route_to_center': 已接单但还在去中心路上，可以继续接单
    - 'en_route_to_task': 已到中心取货，正在送货，不能接单
    """

    def __init__(self, G: nx.Graph, config):
        self.G = G
        self.config = config
        self.worker_positions = {}  # {wid: (node, lon, lat)}
        self.worker_status = {}  # {wid: 'idle', 'en_route_to_center', or 'en_route_to_task'}
        self.worker_center_map = {}  # {wid: region_id} 工人所属的中心区域
        self.worker_busy_until = {}  # {wid: timestamp} 工人忙碌到的时间点
        self.worker_available_from = {}  # {wid: timestamp} 工人从该时刻起可在当前位置自由移动

    def get_available_workers_with_center_info(
            self,
            region_id: int,
            current_time: float = None
    ) -> List[Tuple[Any, str, float, float, Any]]:
        """
        获取指定区域的可用工人列表（包含中心节点信息）

        可用工人包括：
        1. idle 状态的工人
        2. en_route_to_center 状态的工人（还在去中心路上，可以接新任务）
        """
        available = []
        for wid, cur_region_id in self.worker_center_map.items():
            if cur_region_id == region_id:
                status = self.worker_status.get(wid, 'idle')

                if status == 'en_route_to_task':
                    busy_until = self.worker_busy_until.get(wid)
                    if current_time is not None and busy_until is not None and current_time >= busy_until:
                        self.worker_status[wid] = 'en_route_to_center'
                        self.worker_available_from[wid] = busy_until
                        status = 'en_route_to_center'

                if status in ['idle', 'en_route_to_center']:
                    node, lon, lat = self.worker_positions[wid]
                    available.append((node, wid, lon, lat, None))

        return available

    def set_worker_en_route_to_center(self, wid: str, until_timestamp: float) -> None:
        """设置工人正在前往中心（仍可接单）"""
        self.worker_status[wid] = 'en_route_to_center'
        self.worker_busy_until[wid] = until_timestamp
        self.worker_available_from[wid] = until_timestamp

    def set_worker_en_route_to_task(self, wid: str, until_timestamp: float) -> None:
        """设置工人正在前往任务点（不能接单）"""
        self.worker_status[wid] = 'en_route_to_task'
        self.worker_busy_until[wid] = until_timestamp
        self.worker_available_from[wid] = until_timestamp

    def initialize_from_real_data(
            self,
            date: str,
            test_start_hour: int,
            prep_minutes: int = 5,
            coords: np.ndarray = None,
            nodes: list = None,
            partition: Dict[Any, int] = None,
            centers: Dict[int, Any] = None
    ) -> None:
        """
        从真实数据初始化工人位置，并将工人【平均分配】到各个区域中心
        """
        import pandas as pd
        import os

        # 计算时间窗口
        end_timestamp = test_start_hour * 3600
        start_timestamp = end_timestamp - prep_minutes * 60

        file_path = f"D:/biyelunwen/data/worker/workers_{date}.csv"

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在：{file_path}")

        print(f">> 初始化加载工人位置数据...")
        print(f"   - 日期：{date}")
        print(f"   - 测试时间：{test_start_hour}:00")
        print(
            f"   - 使用时间窗口：{start_timestamp // 3600:02d}:{start_timestamp % 3600 // 60:02d} - {end_timestamp // 3600:02d}:{end_timestamp % 3600 // 60:02d}")

        df = pd.read_csv(file_path)
        df['seconds_of_day'] = pd.to_datetime(df['time_str']).dt.hour * 3600 + \
                               pd.to_datetime(df['time_str']).dt.minute * 60 + \
                               pd.to_datetime(df['time_str']).dt.second

        # 筛选时间窗口内的数据
        mask = (df['seconds_of_day'] >= start_timestamp) & \
               (df['seconds_of_day'] < end_timestamp)
        workers_in_window = df[mask].copy()

        if len(workers_in_window) == 0:
            raise ValueError(f"在时间窗口内没有找到工人数据")

        # 取每个工人的最后一个位置
        latest_positions = workers_in_window.sort_values('timestamp').groupby('wid').last().reset_index()

        # 🚀 性能优化：在循环外统一构建 KDTree 并批量查询所有工人的最近路网节点
        if coords is not None and nodes is not None:
            from scipy.spatial import KDTree
            tree = KDTree(coords)
            worker_wgs84 = latest_positions.apply(
                lambda row: gcj02_to_wgs84(row['lon_gcj'], row['lat_gcj']),
                axis=1
            )
            worker_coords = np.array(worker_wgs84.tolist())
            _, idxs = tree.query(worker_coords)
            latest_positions['nearest_node'] = [nodes[i] for i in idxs]
            latest_positions[['lon_wgs84', 'lat_wgs84']] = worker_coords
        else:
            latest_positions['nearest_node'] = None
            latest_positions['lon_wgs84'] = latest_positions['lon_gcj']
            latest_positions['lat_wgs84'] = latest_positions['lat_gcj']

        # 获取所有的中心区域 ID
        region_ids = list(centers.keys()) if centers else []
        num_regions = len(region_ids)

        # 初始化位置，并执行【平均分配】逻辑
        assigned_count = 0
        for idx, row in latest_positions.iterrows():
            wid = row['wid']
            lon = row['lon_wgs84']
            lat = row['lat_wgs84']
            node = row['nearest_node']

            if partition is not None and node in partition:
                assigned_region_id = partition[node]
                self.worker_center_map[wid] = assigned_region_id
                assigned_count += 1
            elif num_regions > 0:
                # 回退策略：如果当前节点未落入任何分区，再做均匀分配。
                assigned_region_id = region_ids[idx % num_regions]
                self.worker_center_map[wid] = assigned_region_id
                assigned_count += 1

            # 保留工人真实的初始坐标
            self.worker_positions[wid] = (node, lon, lat)
            self.worker_status[wid] = 'idle'
            self.worker_available_from[wid] = end_timestamp

        print(f"✅ 初始化完成：共 {len(self.worker_positions)} 个工人")
        if num_regions > 0:
            print(
                f"   - 已将 {assigned_count} 名工人强制平均分配至 {num_regions} 个中心 (每个中心约 {assigned_count // num_regions} 人)")

    def calculate_route_with_center(
            self,
            wid: str,
            task_node: Any,
            centers: Dict[int, Any]
    ) -> Tuple[float, float]:
        """
        计算工人取货路线的总距离和时间
        """
        if wid not in self.worker_center_map:
            return float('inf'), float('inf')

        region_id = self.worker_center_map[wid]
        center_node = centers[region_id]
        worker_node = self.worker_positions[wid][0]

        try:
            # 第一段：工人 → 中心
            dist_to_center = nx.shortest_path_length(
                self.G, source=worker_node, target=center_node, weight='length'
            )

            # 第二段：中心 → 任务
            dist_to_task = nx.shortest_path_length(
                self.G, source=center_node, target=task_node, weight='length'
            )

            total_distance = dist_to_center + dist_to_task

            # 计算时间（假设速度恒定）
            total_time = total_distance / self.config.WORKER_SPEED_MS

            return total_distance, total_time

        except nx.NetworkXNoPath:
            return float('inf'), float('inf')

    def update_worker_position(
            self,
            wid: str,
            new_node: Any,
            new_lon: float,
            new_lat: float
    ) -> None:
        """更新工人位置（在完成分配任务后调用）"""
        self.worker_positions[wid] = (new_node, new_lon, new_lat)

    def set_worker_busy(self, wid: str) -> None:
        """设置工人忙碌状态"""
        self.worker_status[wid] = 'en_route_to_task'

    def set_worker_idle(self, wid: str) -> None:
        """设置工人空闲状态（完成任务后）"""
        self.worker_status[wid] = 'idle'
        self.worker_available_from[wid] = self.worker_busy_until.get(wid, self.worker_available_from.get(wid, 0.0))
        # 注意：位置已经在 update_worker_position 中更新过了

    def _move_worker_towards_center(
            self,
            wid: str,
            centers: Dict[int, Any],
            time_delta_seconds: float
    ) -> bool:
        """
        将单个工人在给定空闲时间内向所属中心推进，返回是否发生位移。
        """
        if time_delta_seconds <= 0:
            return False

        region_id = self.worker_center_map.get(wid)
        if region_id is None or region_id not in centers:
            return False

        curr_node, _, _ = self.worker_positions[wid]
        center_node = centers[region_id]

        if curr_node == center_node:
            return False

        max_travel_dist = time_delta_seconds * self.config.WORKER_SPEED_MS

        try:
            path = nx.shortest_path(self.G, source=curr_node, target=center_node, weight='length')
        except nx.NetworkXNoPath:
            return False

        if len(path) <= 1:
            return False

        traveled_dist = 0.0
        new_node = curr_node

        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            edge_data = self.G.get_edge_data(u, v)
            if isinstance(edge_data, dict) and 0 in edge_data:
                length = edge_data[0].get('length', 0)
            else:
                length = edge_data.get('length', 0) if edge_data else 0

            if traveled_dist + length <= max_travel_dist:
                traveled_dist += length
                new_node = v
            else:
                break

        if new_node == curr_node:
            return False

        node_data = self.G.nodes[new_node]
        new_lon = node_data.get('x', node_data.get('lon'))
        new_lat = node_data.get('y', node_data.get('lat'))
        self.worker_positions[wid] = (new_node, new_lon, new_lat)
        if new_node == center_node:
            self.worker_status[wid] = 'idle'
        else:
            self.worker_status[wid] = 'en_route_to_center'
        return True

    def advance_workers_to_time(
            self,
            centers: Dict[int, Any],
            current_time: float
    ) -> None:
        """
        将工人状态推进到 current_time。

        规则：
        1. 仍在送货且尚未完成的工人保持忙碌。
        2. 已完成送货的工人，从完成时刻开始，在空闲时间内向所属中心移动。
        3. 原本就空闲的工人，从上次可自由移动的时刻开始继续向中心移动。
        """
        moved_count = 0

        for wid in list(self.worker_positions.keys()):
            status = self.worker_status.get(wid, 'idle')
            busy_until = self.worker_busy_until.get(wid)
            idle_start = self.worker_available_from.get(wid, current_time)

            if status == 'en_route_to_task':
                if busy_until is None or busy_until > current_time:
                    continue
                self.worker_status[wid] = 'en_route_to_center'
                idle_start = busy_until

            status = self.worker_status.get(wid, 'idle')
            if status == 'en_route_to_center':
                if current_time > idle_start:
                    if self._move_worker_towards_center(wid, centers, current_time - idle_start):
                        moved_count += 1
                    self.worker_available_from[wid] = current_time
                continue

            if status != 'idle':
                continue

            if current_time > idle_start:
                if self._move_worker_towards_center(wid, centers, current_time - idle_start):
                    moved_count += 1
                self.worker_available_from[wid] = current_time

        if moved_count > 0:
            print(f"   [位置推进] {moved_count} 名工人在空闲时间内向中心移动")

    def move_idle_workers_towards_center(
            self,
            centers: Dict[int, Any],
            time_delta_seconds: float
    ) -> None:
        """
        兼容旧调用：将当前已空闲的工人推进固定时长。
        """
        for wid, status in list(self.worker_status.items()):
            if status == 'idle':
                idle_start = self.worker_available_from.get(wid, 0.0)
                self.worker_available_from[wid] = idle_start + time_delta_seconds
        self.advance_workers_to_time(
            centers,
            max(self.worker_available_from.values(), default=0.0)
        )


def load_task_locations(
        date: str,
        partition: Dict[Any, int],
        centers: Dict[int, Any],
        coords: np.ndarray,
        nodes: list,
        start_hour: Optional[int] = None,
        end_hour: Optional[int] = None,
        sample_size: Optional[int] = None,
        reward_range: Optional[Tuple[float, float]] = None
) -> Dict[int, List[Tuple[Any, str, float]]]:
    """
    加载指定日期的任务位置数据，并按中心分组
    """
    print(f">> 加载 {date} 的任务数据...")

    if start_hour is not None or end_hour is not None:
        time_desc = (
            f"时间段 {start_hour}:00-{end_hour}:00"
            if start_hour is not None and end_hour is not None
            else "全时段"
        )
        print(f"   筛选条件：{time_desc}")

    file_path = f"D:/biyelunwen/data/task/tasks_{date}.csv"

    df = pd.read_csv(file_path)

    df['first_time'] = pd.to_datetime(df['first_time'])
    df['hour'] = df['first_time'].dt.hour

    if start_hour is not None and end_hour is not None:
        if start_hour <= end_hour:
            df = df[(df['hour'] >= start_hour) & (df['hour'] < end_hour)]
        else:
            df = df[(df['hour'] >= start_hour) | (df['hour'] < end_hour)]

    if sample_size is not None and len(df) > sample_size:
        print(f"   随机采样 {sample_size} 个任务...")
        df = df.sample(n=sample_size, random_state=42)

    tasks_per_center = {region_id: [] for region_id in centers.keys()}

    if len(df) == 0:
        print("   ⚠️ 没有满足时间或采样条件的任务")
        return tasks_per_center

    # 🚀 性能优化：在循环外统一构建 KDTree，进行批量向量化查询 (彻底告别单行建树缓慢的问题)
    tree = KDTree(coords)
    task_coords = df[['first_lon', 'first_lat']].values
    _, idxs = tree.query(task_coords)

    # 将批量查询到的 nearest_node 保存回 df，方便后续遍历使用
    df['nearest_node'] = [nodes[i] for i in idxs]

    matched_count = 0
    # 由于已经有了路网点映射，这里的遍历速度极快
    for _, row in df.iterrows():
        task_id = row['task_id']
        nearest_node = row['nearest_node']

        if nearest_node in partition:
            region_id = partition[nearest_node]
            if region_id in tasks_per_center:
                reward = config.TASK_BASE_REWARD
                tasks_per_center[region_id].append(
                    (nearest_node, task_id, reward)
                )
                matched_count += 1

    total_tasks = sum(len(tasks) for tasks in tasks_per_center.values())
    print(
        f"✅ 加载完成：共 {len(df)} 个任务，"
        f"{matched_count} 个匹配到路网，最终分配 {total_tasks} 个任务"
    )

    return tasks_per_center
