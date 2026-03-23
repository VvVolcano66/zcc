from typing import List, Any, Tuple, Dict
import networkx as nx


def greedy_assignment_with_center_pickup(
        G: nx.Graph,
        config,
        centers: Dict[int, Any],
        partition: Dict[Any, int],
        workers_per_center: Dict[int, List[Tuple[Any, str, float, float, Any]]],
        tasks_per_center: Dict[int, List[Tuple[Any, str, float, float]]],
        slot_start_seconds: float = 0.0
) -> Tuple[Dict[Tuple[str, str], float], float, List[Dict]]:
    """
    执行动态贪心任务分配算法（带时间窗约束 + 每人负载上限 4 单）
    """
    print(">> 开始执行贪心任务分配算法（带时间窗死线约束）...")

    all_assignments = {}
    total_profit = 0
    assignment_details = []

    # 性能优化：全局最短路径缓存
    path_cache = {}

    def get_dist(n1, n2):
        if n1 == n2:
            return 0.0
        pair = (n1, n2) if str(n1) < str(n2) else (n2, n1)
        if pair not in path_cache:
            try:
                path_cache[pair] = nx.shortest_path_length(G, source=n1, target=n2, weight='length')
            except nx.NetworkXNoPath:
                path_cache[pair] = float('inf')
        return path_cache[pair]

    for region_id, center_node in centers.items():
        workers = workers_per_center.get(region_id, [])
        tasks = tasks_per_center.get(region_id, [])

        if not workers or not tasks:
            continue

        region_profit, region_assignments, region_details = \
            _assign_single_region_dynamic_greedy(
                G, config, region_id, workers, tasks, center_node, get_dist, slot_start_seconds
            )

        all_assignments.update(region_assignments)
        total_profit += region_profit
        assignment_details.extend(region_details)

    print(f"✅ 贪心分配完成！总分配任务数：{len(all_assignments)}，总利润：{total_profit:.2f} 元")
    return all_assignments, total_profit, assignment_details


def _assign_single_region_dynamic_greedy(
        G: nx.Graph,
        config,
        region_id: int,
        workers: List[Tuple],
        tasks: List[Tuple],
        center_node: Any,
        get_dist,
        slot_start_seconds: float
) -> Tuple[float, Dict[Tuple[str, str], float], List[Dict]]:
    available_tasks = set([t[1] for t in tasks])
    task_nodes = {t[1]: t[0] for t in tasks}
    task_rewards = {t[1]: t[2] for t in tasks}
    task_expires = {t[1]: (t[3] if len(t) > 3 else float('inf')) for t in tasks}

    worker_nodes = {w[1]: w[0] for w in workers}
    worker_capacity = {w[1]: 4 for w in workers}

    # 预计算：工人初始真实位置到中心的距离
    dist_worker_to_center = {}
    for wid, w_node in worker_nodes.items():
        dist_worker_to_center[wid] = get_dist(w_node, center_node)

    # 💡 修复 Bug 2：工人的第一单接力起点必须是中心点！
    # 因为完整的轨迹是：真实位置 -> [中心取货] -> 任务点1
    worker_virtual_loc = {w[1]: center_node for w in workers}
    worker_paid_center_cost = {w[1]: False for w in workers}
    worker_current_time = {w[1]: slot_start_seconds for w in workers}

    assignments = {}
    details = []
    region_profit = 0

    while available_tasks:
        best_pair = None
        best_profit = -float('inf')
        best_dist_to_center = 0
        best_dist_to_task = 0

        for wid, capacity in worker_capacity.items():
            # 容量不足的工人直接跳过
            if capacity <= 0:
                continue

            for tid in available_tasks:
                t_node = task_nodes[tid]
                reward = task_rewards[tid]

                # 从工人当前虚拟位置到任务点的距离
                dist_to_task = get_dist(worker_virtual_loc[wid], t_node)
                if dist_to_task == float('inf'):
                    continue

                # 如果工人还没有支付过中心取货成本，则加上去中心的距离
                if not worker_paid_center_cost[wid]:
                    dist_to_center = dist_worker_to_center[wid]
                else:
                    dist_to_center = 0.0

                total_dist = dist_to_center + dist_to_task

                # 时间窗死线校验
                travel_time = total_dist / config.WORKER_SPEED_MS
                arrival_time = worker_current_time[wid] + travel_time

                if arrival_time > task_expires[tid]:
                    continue

                # 核算经济利润
                travel_cost = total_dist * config.TRAVEL_COST_PER_METER
                profit = reward - travel_cost

                if profit > best_profit and profit > 0:
                    best_profit = profit
                    best_pair = (wid, tid)
                    best_dist_to_center = dist_to_center
                    best_dist_to_task = dist_to_task

        # 如果找不到能赚钱且不超时的订单，提前结束该区域分配
        if best_pair is None:
            break

        # ==================================
        # 落实分配操作，更新所有物理状态
        # ==================================
        best_wid, best_tid = best_pair

        assignments[(best_wid, best_tid)] = best_profit

        # 💡 修复 Bug 1：必须扣减该工人的可接单容量，防止变身超人！
        worker_capacity[best_wid] -= 1

        # 更新工人的虚拟位置到任务点 (模拟完成配送，准备从这里去接下一单)
        worker_virtual_loc[best_wid] = task_nodes[best_tid]
        # 标记该工人已经跑过中心取完货了
        worker_paid_center_cost[best_wid] = True

        # 更新工人完成当前任务后的绝对物理时间
        travel_time = (best_dist_to_center + best_dist_to_task) / config.WORKER_SPEED_MS
        worker_current_time[best_wid] += travel_time

        region_profit += best_profit

        details.append({
            'wid': best_wid,
            'task_id': best_tid,
            'dist_to_center': best_dist_to_center,
            'dist_to_task': best_dist_to_task,
            'task_node': task_nodes[best_tid],
            'profit': best_profit
        })

        # 把已经被接走的外卖从池子里删掉
        available_tasks.remove(best_tid)

        if len(details) % 100 == 0:
            print(f"   已处理 {len(details)} 个分配...")

    print(f"✅ 区域 {region_id} 贪心分配完成！分配任务数：{len(assignments)}，利润：{region_profit:.2f} 元")
    return region_profit, assignments, details