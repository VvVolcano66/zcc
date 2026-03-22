from typing import List, Any, Tuple, Dict
import networkx as nx


def greedy_assignment_with_center_pickup(
        G: nx.Graph,
        config,
        centers: Dict[int, Any],
        partition: Dict[Any, int],
        workers_per_center: Dict[int, List[Tuple[Any, str, float, float, Any]]],
        tasks_per_center: Dict[int, List[Tuple[Any, str, float, float]]],
        slot_start_seconds: float = 0.0  # 💡 新增：传入当前时间槽的绝对起始时间
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

    dist_worker_to_center = {}
    for wid, w_node in worker_nodes.items():
        dist_worker_to_center[wid] = get_dist(w_node, center_node)

    # 💡 修复：工人的虚拟位置应该是其真实位置，而不是中心节点
    # 这样在计算距离时会正确计算：工人真实位置 → 中心 → 任务
    worker_virtual_loc = {w[1]: w[0] for w in workers}
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
            if capacity <= 0:
                continue

            for tid in available_tasks:
                t_node = task_nodes[tid]
                reward = task_rewards[tid]

                # 💡 修复：从工人虚拟位置 (真实位置) 到任务点的距离
                dist_to_task = get_dist(worker_virtual_loc[wid], t_node)
                if dist_to_task == float('inf'):
                    continue

                # 💡 修复：如果工人还没有支付过中心取货成本，则需要加上这段距离
                if not worker_paid_center_cost[wid]:
                    dist_to_center = dist_worker_to_center[wid]
                else:
                    dist_to_center = 0.0

                total_dist = dist_to_center + dist_to_task

                travel_time = total_dist / config.WORKER_SPEED_MS
                arrival_time = worker_current_time[wid] + travel_time

                if arrival_time > task_expires[tid]:
                    continue

                travel_cost = total_dist * config.TRAVEL_COST_PER_METER
                profit = reward - travel_cost

                if profit > best_profit and profit > 0:
                    best_profit = profit
                    best_pair = (wid, tid)
                    best_dist_to_center = dist_to_center
                    best_dist_to_task = dist_to_task

        if best_pair is None:
            break

        best_wid, best_tid = best_pair

        assignments[(best_wid, best_tid)] = best_profit

        # 💡 修复：更新工人的虚拟位置到任务点 (模拟完成配送)
        worker_virtual_loc[best_wid] = task_nodes[best_tid]
        # 标记该工人已经支付过中心取货成本
        worker_paid_center_cost[best_wid] = True

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

        available_tasks.remove(best_tid)

        if len(details) % 100 == 0:
            print(f"   已处理 {len(details)} 个分配...")

    print(f"✅ 区域 {region_id} 贪心分配完成！分配任务数：{len(assignments)}，利润：{region_profit:.2f} 元")
    return region_profit, assignments, details