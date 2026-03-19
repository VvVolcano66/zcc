from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import networkx as nx


def snap_to_network(lon: float, lat: float, coords: np.ndarray, nodes: list) -> Any:
    """将经纬度坐标映射到路网中最近的节点"""
    tree = KDTree(coords)
    _, idx = tree.query([[lon, lat]])
    return nodes[idx[0]]


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

                if status in ['en_route_to_center', 'en_route_to_task']:
                    if wid in self.worker_busy_until:
                        if current_time is not None and current_time >= self.worker_busy_until[wid]:
                            self.worker_status[wid] = 'idle'
                            status = 'idle'

                if status in ['idle', 'en_route_to_center']:
                    node, lon, lat = self.worker_positions[wid]
                    available.append((node, wid, lon, lat, None))

        return available

    def set_worker_en_route_to_center(self, wid: str, until_timestamp: float) -> None:
        """设置工人正在前往中心（仍可接单）"""
        self.worker_status[wid] = 'en_route_to_center'
        self.worker_busy_until[wid] = until_timestamp

    def set_worker_en_route_to_task(self, wid: str, until_timestamp: float) -> None:
        """设置工人正在前往任务点（不能接单）"""
        self.worker_status[wid] = 'en_route_to_task'
        self.worker_busy_until[wid] = until_timestamp

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
        从真实数据初始化工人位置，并绑定到对应区域中心
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

        # 初始化位置，并绑定到对应的中心区域
        assigned_count = 0
        for _, row in latest_positions.iterrows():
            wid = row['wid']
            lon = row['lon_gcj']
            lat = row['lat_gcj']

            # 映射到路网节点
            if coords is not None and nodes is not None:
                node = snap_to_network(lon, lat, coords, nodes)

                # 确定工人属于哪个中心区域
                if partition is not None and node in partition:
                    region_id = partition[node]
                    self.worker_center_map[wid] = region_id
                    assigned_count += 1
            else:
                node = None

            self.worker_positions[wid] = (node, lon, lat)
            self.worker_status[wid] = 'idle'

        print(f"✅ 初始化完成：共 {len(self.worker_positions)} 个工人")
        print(f"   - 已分配到中心区域：{assigned_count} 个工人")

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
        self.worker_status[wid] = 'busy'

    def set_worker_idle(self, wid: str) -> None:
        """设置工人空闲状态（完成任务后）"""
        self.worker_status[wid] = 'idle'
        # 注意：位置已经在 update_worker_position 中更新过了

    def move_idle_workers_towards_center(
            self,
            centers: Dict[int, Any],
            time_delta_seconds: float
    ) -> None:
        """
        【新增功能】模拟时间流逝，将所有空闲 (idle) 的工人向所属区域的中心移动。
        如果在 time_delta_seconds 时间内能到达中心，则位置更新为中心；
        否则沿着最短路移动相应的距离，停在途中的某个节点上。
        """
        # 计算这段时间内工人最多能走多远
        max_travel_dist = time_delta_seconds * self.config.WORKER_SPEED_MS

        moved_count = 0
        reached_center_count = 0

        for wid, status in self.worker_status.items():
            # 只移动空闲的工人
            if status == 'idle':
                region_id = self.worker_center_map.get(wid)
                if region_id is None or region_id not in centers:
                    continue

                curr_node, _, _ = self.worker_positions[wid]
                center_node = centers[region_id]

                if curr_node == center_node:
                    continue  # 已经在中心待命，不需要移动

                try:
                    # 计算从当前位置到中心的最短路径
                    path = nx.shortest_path(self.G, source=curr_node, target=center_node, weight='length')

                    if len(path) <= 1:
                        continue

                    traveled_dist = 0.0
                    new_node = curr_node

                    # 沿着最短路径的节点一截一截走
                    for i in range(len(path) - 1):
                        u = path[i]
                        v = path[i + 1]

                        # 获取边长数据 (兼容 Graph 和 MultiGraph)
                        edge_data = self.G.get_edge_data(u, v)
                        if isinstance(edge_data, dict) and 0 in edge_data:
                            length = edge_data[0].get('length', 0)
                        else:
                            length = edge_data.get('length', 0) if edge_data else 0

                        # 如果这段路能在剩余时间内走完
                        if traveled_dist + length <= max_travel_dist:
                            traveled_dist += length
                            new_node = v
                        else:
                            # 走不完，为了简化节点映射，停在当前边起点的节点 u
                            break

                    # 提取到达的新节点坐标并更新工人位置
                    node_data = self.G.nodes[new_node]
                    new_lon = node_data.get('x', node_data.get('lon'))
                    new_lat = node_data.get('y', node_data.get('lat'))

                    self.worker_positions[wid] = (new_node, new_lon, new_lat)

                    moved_count += 1
                    if new_node == center_node:
                        reached_center_count += 1

                except nx.NetworkXNoPath:
                    # 如果图不连通找不到路，就原地不动
                    pass

        if moved_count > 0:
            print(f"   [主动返航] {moved_count} 名空闲工人向中心移动 (其中 {reached_center_count} 人已就位)")


def load_task_locations(
        date: str,
        partition: Dict[Any, int],
        centers: Dict[int, Any],
        coords: np.ndarray,
        nodes: list,
        start_hour: Optional[int] = None,
        end_hour: Optional[int] = None,
        sample_size: Optional[int] = None,
        reward_range: Tuple[float, float] = (8.0, 15.0)
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

    matched_count = 0
    for _, row in df.iterrows():
        lon = row['first_lon']
        lat = row['first_lat']
        task_id = row['task_id']

        nearest_node = snap_to_network(lon, lat, coords, nodes)

        if nearest_node in partition:
            region_id = partition[nearest_node]
            if region_id in tasks_per_center:
                reward = np.random.uniform(reward_range[0], reward_range[1])
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