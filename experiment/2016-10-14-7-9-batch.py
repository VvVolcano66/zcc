import time
import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import KDTree

import config
from algorithm.Greedy import greedy_assignment_with_center_pickup
from tool.TaskWorkerToMap import WorkerSimulator
from tool.data_loader import get_real_road_network
from tool.map_algorithms import run_kmeans_baseline, run_rcc_algorithm, find_region_centers

from algorithm.IMTAO import Center as IMTAOCenter, Worker as IMTAOWorker, Task as IMTAOTask, IMTAO_Framework


def run_imtao_for_slot(G, config, centers_dict, workers_per_center, tasks_per_center, slot_start_seconds):
    """
    IMTAO 算法适配器（修复版：强制增加中心取货约束）
    """
    imtao_centers = []
    imtao_workers = []
    imtao_tasks = []

    worker_node_map = {}
    task_node_map = {}

    for rid, c_node in centers_dict.items():
        c_lon = G.nodes[c_node].get('x', G.nodes[c_node].get('lon'))
        c_lat = G.nodes[c_node].get('y', G.nodes[c_node].get('lat'))
        imtao_centers.append(IMTAOCenter(rid, c_lon, c_lat))

    for rid, w_list in workers_per_center.items():
        for w in w_list:
            w_node, wid, w_lon, w_lat, _ = w
            # 💡 修复：将工人的位置设置为其所属区域的中心点 (强制中心取货约束)
            center_node = centers_dict[rid]
            center_lon = G.nodes(center_node).get('x', G.nodes(center_node).get('lon'))
            center_lat = G.nodes(center_node).get('y', G.nodes(center_node).get('lat'))
            imtao_workers.append(IMTAOWorker(wid, center_lon, center_lat, max_t=4))
            worker_node_map[wid] = w_node

    for rid, t_list in tasks_per_center.items():
        for t in t_list:
            t_node, tid, reward, expire_seconds = t[0], t[1], t[2], t[3]

            t_lon = G.nodes[t_node].get('x', G.nodes[t_node].get('lon'))
            t_lat = G.nodes[t_node].get('y', G.nodes[t_node].get('lat'))

            relative_expire_seconds = max(0, expire_seconds - slot_start_seconds)
            imtao_tasks.append(IMTAOTask(tid, t_lon, t_lat, expire_time=relative_expire_seconds))
            task_node_map[tid] = t_node

    if not imtao_tasks:
        return {}, 0, []

    framework = IMTAO_Framework(imtao_centers, imtao_tasks, imtao_workers)
    framework.algo3_game_theoretic_collaboration()

    slot_assignments = {}
    slot_details = []
    slot_profit = 0

    for c in framework.centers:
        center_node = centers_dict[c.id]
        for w, assigned_tasks in c.A:
            worker_node = worker_node_map[w.id]

            try:
                dist_to_center = nx.shortest_path_length(G, worker_node, center_node, weight='length')
            except nx.NetworkXNoPath:
                continue

            prev_node = center_node
            current_dist_to_center = dist_to_center

            for task in assigned_tasks:
                task_node = task_node_map[task.id]
                try:
                    dist_to_task = nx.shortest_path_length(G, prev_node, task_node, weight='length')
                except nx.NetworkXNoPath:
                    continue

                travel_cost = (current_dist_to_center + dist_to_task) * config.TRAVEL_COST_PER_METER
                profit = config.TASK_BASE_REWARD - travel_cost

                slot_assignments[(w.id, task.id)] = profit
                slot_details.append({
                    'wid': w.id,
                    'task_id': task.id,
                    'dist_to_center': current_dist_to_center,
                    'dist_to_task': dist_to_task,
                    'task_node': task_node,
                    'profit': profit
                })
                slot_profit += profit

                prev_node = task_node
                current_dist_to_center = 0

    return slot_assignments, slot_profit, slot_details


def run_online_simulation_with_center_pickup(
        algo_name: str = 'greedy',
        test_date: str = '2016-10-14',
        test_start_hour: int = 7,
        test_end_hour: int = 9,
        time_slot_minutes: int = 15
):
    print("=" * 70)
    print(f"在线物流调度仿真实验 (算法: {algo_name.upper()})")
    print("=" * 70)
    print(f"测试日期：{test_date} | 时段：{test_start_hour}:00-{test_end_hour}:00 | 时间槽：{time_slot_minutes} 分钟")
    print("=" * 70)

    # ==================== 1. 加载路网和分区 ====================
    print("\n【阶段 1-3】加载路网数据与中心划分...")
    G, coords, nodes = get_real_road_network(config.CHENGDU_CENTER, dist=config.DOWNLOAD_DIST)
    kmeans_partition = run_kmeans_baseline(coords, nodes, k=config.NUM_ZONES)
    rcc_partition = run_rcc_algorithm(G, kmeans_partition, k=config.NUM_ZONES)
    centers = find_region_centers(G, rcc_partition, weight='length')

    # ==================== 2. 初始化全局订单池 ====================
    print("\n【阶段 3.0】预加载并构建全局订单池 (精确到秒级划分)...")
    task_file = f"D:/biyelunwen/data/task/tasks_{test_date}.csv"
    if os.path.exists(task_file):
        df_tasks = pd.read_csv(task_file)
        df_tasks['first_time'] = pd.to_datetime(df_tasks['first_time'])
        df_tasks['seconds_of_day'] = df_tasks['first_time'].dt.hour * 3600 + df_tasks['first_time'].dt.minute * 60 + \
                                     df_tasks['first_time'].dt.second

        tree = KDTree(coords)
        task_coords = df_tasks[['first_lon', 'first_lat']].values
        _, idxs = tree.query(task_coords)
        df_tasks['nearest_node'] = [nodes[i] for i in idxs]
        print(f"✅ 全局订单池就绪，共 {len(df_tasks)} 个任务待命。")
    else:
        print(f"⚠️ 找不到订单文件: {task_file}")
        df_tasks = pd.DataFrame()

    unassigned_tasks_pool = {region_id: [] for region_id in centers.keys()}
    total_expired_tasks_global = 0

    # ==================== 3. 初始化工人位置 ====================
    print("\n【阶段 4】初始化工人真实位置...")
    worker_sim = WorkerSimulator(G, config)
    worker_sim.initialize_from_real_data(
        date=test_date, test_start_hour=test_start_hour, prep_minutes=5,
        coords=coords, nodes=nodes, partition=rcc_partition, centers=centers
    )

    # ==================== 4. 在线仿真实验 ====================
    print("\n【阶段 5】开始在线流式仿真...")
    all_assignments = {}
    all_details = []
    total_profit = 0
    total_dist_to_center = 0
    total_dist_to_task = 0

    total_minutes = (test_end_hour - test_start_hour) * 60
    num_slots = total_minutes // time_slot_minutes

    for slot_idx in range(num_slots):
        slot_start_minute = slot_idx * time_slot_minutes
        slot_end_minute = (slot_idx + 1) * time_slot_minutes
        current_hour = test_start_hour + slot_start_minute // 60
        current_minute = slot_start_minute % 60
        next_hour = test_start_hour + slot_end_minute // 60
        next_minute = slot_end_minute % 60

        print(
            f"\n--- 时间槽 {slot_idx + 1}/{num_slots}: {current_hour:02d}:{current_minute:02d} - {next_hour:02d}:{next_minute:02d} ---")

        time_delta_seconds = time_slot_minutes * 60
        print(f">> 空闲工人正在向所属中心返航中...")
        worker_sim.move_idle_workers_towards_center(centers, time_delta_seconds)

        # 4.1 提取当前时间槽的订单，并合并上轮积压的订单
        slot_start_seconds = test_start_hour * 3600 + slot_start_minute * 60
        slot_end_seconds = test_start_hour * 3600 + slot_end_minute * 60

        tasks_per_center = {region_id: [] for region_id in centers.keys()}

        for rid in centers.keys():
            tasks_per_center[rid].extend(unassigned_tasks_pool[rid])

        new_tasks_count = 0
        if not df_tasks.empty:
            mask = (df_tasks['seconds_of_day'] >= slot_start_seconds) & (df_tasks['seconds_of_day'] < slot_end_seconds)
            current_tasks = df_tasks[mask]

            for _, row in current_tasks.iterrows():
                nearest_node = row['nearest_node']
                if nearest_node in rcc_partition:
                    region_id = rcc_partition[nearest_node]
                    reward = np.random.uniform(config.TASK_BASE_REWARD, 15.0)
                    expire_seconds = row['seconds_of_day'] + config.TASK_EXPIRE_MINUTES * 60

                    tasks_per_center[region_id].append((nearest_node, row['task_id'], reward, expire_seconds))
                    new_tasks_count += 1

        # 4.2 获取可用工人
        workers_per_center = {}
        for region_id in centers.keys():
            workers = worker_sim.get_available_workers_with_center_info(region_id)
            workers_per_center[region_id] = [(w[0], w[1], w[2], w[3], centers[region_id]) for w in workers]

        total_workers = sum(len(w) for w in workers_per_center.values())
        total_current_tasks = sum(len(t) for t in tasks_per_center.values())
        print(f"可用工人: {total_workers} 个 | 新增订单: {new_tasks_count} 个 | 池内总单量: {total_current_tasks} 个")

        if total_current_tasks == 0:
            print("本时段无订单，跳过分配。")
            continue

        # 4.3 执行调度分配算法
        if algo_name.lower() == 'greedy':
            slot_assignments, slot_profit, slot_details = greedy_assignment_with_center_pickup(
                G=G, config=config, centers=centers, partition=rcc_partition,
                workers_per_center=workers_per_center, tasks_per_center=tasks_per_center,
                slot_start_seconds=slot_start_seconds  # 💡 补全了这个漏掉的参数！
            )
        elif algo_name.lower() == 'imtao':
            slot_assignments, slot_profit, slot_details = run_imtao_for_slot(
                G=G, config=config, centers_dict=centers,
                workers_per_center=workers_per_center, tasks_per_center=tasks_per_center,
                slot_start_seconds=slot_start_seconds
            )
        else:
            raise ValueError(f"未知的算法: {algo_name}")

        # 4.4 更新物理位置状态
        slot_dist_to_center = 0
        slot_dist_to_task = 0

        for detail in slot_details:
            wid = detail['wid']
            task_node = detail['task_node']
            slot_dist_to_center += detail['dist_to_center']
            slot_dist_to_task += detail['dist_to_task']

            if task_node in G.nodes:
                task_lon, task_lat = G.nodes[task_node].get('x', G.nodes[task_node].get('lon')), G.nodes[task_node].get(
                    'y', G.nodes[task_node].get('lat'))
                worker_sim.update_worker_position(wid, task_node, task_lon, task_lat)
                worker_sim.set_worker_busy(wid)

        # =========================================================
        # 💡 4.5 核心修改：结算积压订单 (已删除扣钱逻辑)
        # =========================================================
        assigned_task_ids = set([k[1] for k in slot_assignments.keys()])
        slot_expired_count = 0

        for rid in centers.keys():
            new_pool = []
            for t in tasks_per_center[rid]:
                if t[1] not in assigned_task_ids:
                    expire_seconds = t[3]
                    # 判断该订单是否已经超过其存活时间
                    if slot_end_seconds >= expire_seconds:
                        slot_expired_count += 1
                        # 现在超时订单只会被淘汰，不再倒扣系统的利润
                    else:
                        new_pool.append(t)
            unassigned_tasks_pool[rid] = new_pool

        leftover_count = sum(len(pool) for pool in unassigned_tasks_pool.values())
        total_expired_tasks_global += slot_expired_count

        print(
            f"分配结果: 成交 {len(slot_assignments)} 单, 调度 {len(set(k[0] for k in slot_assignments.keys()))} 名工人")

        if slot_expired_count > 0:
            print(f"   ❌ 超时淘汰订单: {slot_expired_count} 个 (已自动取消该订单，不扣除利润)")

        if leftover_count > 0:
            print(f"   ⏳ 剩余积压订单: {leftover_count} 个 (安全范围内，自动滚入下一轮)")

        total_dist_to_center += slot_dist_to_center
        total_dist_to_task += slot_dist_to_task
        all_assignments.update(slot_assignments)
        all_details.extend(slot_details)
        total_profit += slot_profit

        # 4.6 时间槽末尾统一释放完成任务的工人
        for wid, _ in slot_assignments.keys():
            worker_sim.set_worker_idle(wid)

    print(f"\n[{algo_name.upper()}] 仿真完成:")
    print(f"  - 总利润: {total_profit:.2f} 元")
    print(f"  - 总成交: {len(all_assignments)} 单")
    print(f"  - 累计超时淘汰: {total_expired_tasks_global} 单")

    return all_assignments, all_details, total_profit, total_expired_tasks_global


if __name__ == "__main__":
    # ========================================================
    # 毕业论文核心实验：多模型调度效果在线横向对比
    # ========================================================

    # 1. 运行传统的贪心基线算法
    greedy_assignments, greedy_details, greedy_profit, g_expired = run_online_simulation_with_center_pickup(
        algo_name='greedy', test_date='2016-10-14', test_start_hour=7, test_end_hour=9, time_slot_minutes=15
    )

    # 2. 运行 ICDE 论文复现算法 (博弈论多中心协同)
    imtao_assignments, imtao_details, imtao_profit, i_expired = run_online_simulation_with_center_pickup(
        algo_name='imtao', test_date='2016-10-14', test_start_hour=7, test_end_hour=9, time_slot_minutes=15
    )

    # 3. 打印论文所需的大表对比数据
    print("\n\n" + "=" * 80)
    print("🏆 毕业论文实验：多中心调度算法性能横向对比 (无超时惩罚金版)")
    print("=" * 80)
    print(f"{'指标':<25} | {'Greedy (局部基线)':<22} | {'IMTAO (多中心协同)':<22}")
    print("-" * 80)
    print(f"{'总分配任务数 (完单量)':<20} | {len(greedy_assignments):<24} | {len(imtao_assignments):<22}")
    print(f"{'系统总净利润 (元)':<22} | {greedy_profit:<24.2f} | {imtao_profit:<22.2f}")
    print(f"{'超时淘汰遗弃单量':<21} | {g_expired:<24} | {i_expired:<22}")

    # 统计空载率
    greedy_dist_c = sum(d['dist_to_center'] for d in greedy_details)
    greedy_dist_t = sum(d['dist_to_task'] for d in greedy_details)
    imtao_dist_c = sum(d['dist_to_center'] for d in imtao_details)
    imtao_dist_t = sum(d['dist_to_task'] for d in imtao_details)

    g_empty_rate = (greedy_dist_c / (greedy_dist_c + greedy_dist_t) * 100) if (greedy_dist_c + greedy_dist_t) > 0 else 0
    i_empty_rate = (imtao_dist_c / (imtao_dist_c + imtao_dist_t) * 100) if (imtao_dist_c + imtao_dist_t) > 0 else 0

    print(f"{'空载行驶占比 (越低越好)':<20} | {g_empty_rate:<23.2f}% | {i_empty_rate:<21.2f}%")
    print("=" * 80)