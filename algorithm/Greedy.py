from typing import List, Any, Tuple, Dict

import networkx as nx


def calculate_task_profit_with_center_pickup(
        G: nx.Graph,
        config,
        worker_node: Any,
        center_node: Any,
        task_node: Any,
        task_reward: float = None
) -> float:
    """
    计算工人完成某个任务的净收益（需要先去中心取货）

    路径：工人当前位置 → 中心取货 → 任务点送货

    净收益 = 任务奖励 - 行驶成本（两段路程）

    Args:
        G: 路网图
        config: 配置对象
        worker_node: 工人当前所在节点
        center_node: 所属区域中心节点
        task_node: 任务所在节点
        task_reward: 任务奖励，如为 None 则使用基础奖励

    Returns:
        净收益值，如果无法到达则返回负无穷
    """
    if task_reward is None:
        task_reward = config.TASK_BASE_REWARD

    try:
        # 第一段：工人 → 中心
        dist_to_center = nx.shortest_path_length(
            G,
            source=worker_node,
            target=center_node,
            weight='length'
        )

        # 第二段：中心 → 任务
        dist_to_task = nx.shortest_path_length(
            G,
            source=center_node,
            target=task_node,
            weight='length'
        )

        # 总行驶距离
        total_distance = dist_to_center + dist_to_task

        # 计算行驶成本
        travel_cost = total_distance * config.TRAVEL_COST_PER_METER

        # 净收益
        profit = task_reward - travel_cost
        return profit

    except nx.NetworkXNoPath:
        return float('-inf')


def calculate_task_profit(G: nx.Graph, config, worker_node: Any,
                          task_node: Any, task_reward: float = None) -> float:
    """
    计算工人完成某个任务的净收益（不经过中心，直接前往）

    净收益 = 任务奖励 - 行驶成本

    Args:
        G: 路网图
        config: 配置对象
        worker_node: 工人所在节点
        task_node: 任务所在节点
        task_reward: 任务奖励，如为 None 则使用基础奖励

    Returns:
        净收益值，如果无法到达则返回负无穷
    """
    if task_reward is None:
        task_reward = config.TASK_BASE_REWARD

    try:
        shortest_path = nx.shortest_path_length(
            G,
            source=worker_node,
            target=task_node,
            weight='length'
        )
        travel_cost = shortest_path * config.TRAVEL_COST_PER_METER
        profit = task_reward - travel_cost
        return profit
    except nx.NetworkXNoPath:
        return float('-inf')


def greedy_assignment_with_center_pickup(
        G: nx.Graph,
        config,
        centers: Dict[int, Any],
        partition: Dict[Any, int],
        workers_per_center: Dict[int, List[Tuple[Any, str, float, float]]],
        tasks_per_center: Dict[int, List[Tuple[Any, str, float]]]
) -> Tuple[Dict[Tuple[str, str], float], float, List[Dict]]:
    """
    执行贪心任务分配算法（工人需要先去中心取货）

    目标：最大化总利益（收益 - 成本）

    核心约束：
    1. 每个任务只能分配给同一中心的工人
    2. 每个工人只能完成同一中心的任务
    3. 每个工人最多只能完成一个任务
    4. 工人必须先从当前位置到中心取货，再去任务点

    Args:
        G: 路网图
        config: 配置对象
        centers: {region_id: center_node} 每个区域的中心节点
        partition: {node: region_id} 节点到区域的映射
        workers_per_center: {region_id: [(worker_node, wid, lon, lat), ...]}
        tasks_per_center: {region_id: [(task_node, task_id, reward), ...]}

    Returns:
        assignments: {(wid, task_id): profit} 分配方案
        total_profit: 总利润
        assignment_details: 分配详情列表
    """
    print(">> 开始执行贪心任务分配算法（带中心取货）...")

    all_assignments = {}
    total_profit = 0
    assignment_details = []

    # 对每个区域独立进行分配
    for region_id in centers.keys():
        print(f"\n处理区域 {region_id}...")

        workers = workers_per_center.get(region_id, [])
        tasks = tasks_per_center.get(region_id, [])

        if not workers or not tasks:
            print(f"  区域 {region_id} 没有足够的工人或任务，跳过")
            continue

        center_node = centers[region_id]
        region_profit, region_assignments, region_details = \
            _assign_single_region_with_center_pickup(
                G, config, region_id, workers, tasks, center_node
            )

        all_assignments.update(region_assignments)
        total_profit += region_profit
        assignment_details.extend(region_details)

        print(f"  区域 {region_id} 完成分配："
              f"{len(region_assignments)} 个任务，利润：{region_profit:.2f} 元")

    print(f"\n✅ 贪心分配完成！")
    print(f"   总分配任务数：{len(all_assignments)}")
    print(f"   总利润：{total_profit:.2f} 元")

    return all_assignments, total_profit, assignment_details


def _assign_single_region_with_center_pickup(
        G: nx.Graph,
        config,
        region_id: int,
        workers: List[Tuple[Any, str, float, float]],
        tasks: List[Tuple[Any, str, float]],
        center_node: Any
) -> Tuple[float, Dict[Tuple[str, str], float], List[Dict]]:
    """
    对单个区域内的工人和任务进行贪心分配（工人需要先去中心）

    贪心策略：
    1. 计算所有可能的 (工人，任务) 配对的收益（考虑去中心的路程）
    2. 按收益从高到低排序
    3. 依次选择收益最高的配对，确保工人和任务都未被分配

    Args:
        G: 路网图
        config: 配置对象
        region_id: 区域 ID
        workers: 该区域的工人列表 [(worker_node, wid, lon, lat), ...]
        tasks: 该区域的任务列表 [(task_node, task_id, reward), ...]
        center_node: 中心节点

    Returns:
        region_profit: 该区域总利润
        assignments: {(wid, task_id): profit} 该区域的分配方案
        details: 分配详情列表
    """
    # 可用工人集合（用工人的 ID 标识）
    available_workers = set([w[1] for w in workers])
    # 可用任务集合（用任务 ID 标识）
    available_tasks = set([t[1] for t in tasks])

    # 建立 ID 到节点的映射
    worker_nodes = {w[1]: w[0] for w in workers}
    task_nodes = {t[1]: t[0] for t in tasks}
    task_rewards = {t[1]: t[2] for t in tasks}

    assignments = {}
    details = []
    region_profit = 0

    # 计算所有可能的 (工人，任务) 配对的收益（考虑去中心）
    all_pairs = []

    for worker in workers:
        worker_node = worker[0]
        wid = worker[1]

        for task in tasks:
            task_node = task[0]
            task_id = task[1]
            reward = task[2]

            # 使用带中心取货的收益计算
            profit = calculate_task_profit_with_center_pickup(
                G, config, worker_node, center_node, task_node, reward
            )

            # 只考虑有正收益的配对
            if profit > 0:
                # 计算两段距离
                try:
                    dist_to_center = nx.shortest_path_length(
                        G, source=worker_node, target=center_node, weight='length'
                    )
                    dist_to_task = nx.shortest_path_length(
                        G, source=center_node, target=task_node, weight='length'
                    )
                    total_distance = dist_to_center + dist_to_task
                except:
                    total_distance = float('inf')

                all_pairs.append({
                    'worker_node': worker_node,
                    'wid': wid,
                    'task_node': task_node,
                    'task_id': task_id,
                    'profit': profit,
                    'reward': reward,
                    'distance': total_distance,
                    'dist_to_center': dist_to_center,
                    'dist_to_task': dist_to_task
                })

    # 按收益从高到低排序
    all_pairs.sort(key=lambda x: x['profit'], reverse=True)

    # 贪心选择：依次选择收益最高的配对
    for pair in all_pairs:
        wid = pair['wid']
        task_id = pair['task_id']

        # 确保工人和任务都未被分配
        if wid in available_workers and task_id in available_tasks:
            available_workers.remove(wid)
            available_tasks.remove(task_id)

            assignments[(wid, task_id)] = pair['profit']
            region_profit += pair['profit']

            details.append({
                'region_id': region_id,
                'wid': wid,
                'task_id': task_id,
                'task_node': task_node,
                'reward': pair['reward'],
                'distance': pair['distance'],
                'dist_to_center': pair['dist_to_center'],
                'dist_to_task': pair['dist_to_task'],
                'cost': pair['reward'] - pair['profit'],
                'profit': pair['profit']
            })

    return region_profit, assignments, details