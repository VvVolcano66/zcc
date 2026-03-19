import time
import networkx as nx
import config
from algorithm.Greedy import greedy_assignment_with_center_pickup
from tool.TaskWorkerToMap import WorkerSimulator, load_task_locations
from tool.data_loader import get_real_road_network
from tool.map_algorithms import run_kmeans_baseline, run_rcc_algorithm, find_region_centers

# 引入我们复现的 ICDE 论文核心算法
from algorithm.IMTAO import Center as IMTAOCenter, Worker as IMTAOWorker, Task as IMTAOTask, IMTAO_Framework


def run_imtao_for_slot(G, config, centers_dict, workers_per_center, tasks_per_center, time_slot_minutes):
    """
    IMTAO 算法适配器：将当前时间槽的数据转为 IMTAO 所需格式，运行协同分配后返回标准结果
    """
    imtao_centers = []
    imtao_workers = []
    imtao_tasks = []

    worker_node_map = {}
    task_node_map = {}

    # 1. 构造 IMTAO Centers
    for rid, c_node in centers_dict.items():
        c_lon = G.nodes[c_node].get('x', G.nodes[c_node].get('lon'))
        c_lat = G.nodes[c_node].get('y', G.nodes[c_node].get('lat'))
        imtao_centers.append(IMTAOCenter(rid, c_lon, c_lat))

    # 2. 构造 IMTAO Workers
    for rid, w_list in workers_per_center.items():
        for w in w_list:
            # worker format: (wid, w_node, lon, lat, center_node)
            wid, w_node, w_lon, w_lat, _ = w
            # 根据论文设定，每个工人一个批次最多可送 4 单
            imtao_workers.append(IMTAOWorker(wid, w_lon, w_lat, max_t=4))
            worker_node_map[wid] = w_node

    # 3. 构造 IMTAO Tasks
    for rid, t_list in tasks_per_center.items():
        for t in t_list:
            tid, t_node = t[0], t[1]
            if len(t) >= 4:
                t_lon, t_lat = t[2], t[3]
            else:
                t_lon = G.nodes[t_node].get('x', G.nodes[t_node].get('lon'))
                t_lat = G.nodes[t_node].get('y', G.nodes[t_node].get('lat'))

            # 过期时间设为 1 个时间槽的时长 (转化为秒)
            expire_seconds = time_slot_minutes * 60
            imtao_tasks.append(IMTAOTask(tid, t_lon, t_lat, expire_time=expire_seconds))
            task_node_map[tid] = t_node

    if not imtao_tasks:
        return {}, 0, []

    # 4. 运行博弈论多中心协同框架
    framework = IMTAO_Framework(imtao_centers, imtao_tasks, imtao_workers)
    framework.algo3_game_theoretic_collaboration()

    # 5. 解析结果，转回 Greedy 返回的通用格式
    slot_assignments = {}
    slot_details = []
    slot_profit = 0

    for c in framework.centers:
        center_node = centers_dict[c.id]
        for w, assigned_tasks in c.A:
            worker_node = worker_node_map[w.id]

            # 论文逻辑：工人 -> 协同中心取货 -> 任务1 -> 任务2 -> ...
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

                # 经济学核算
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

                # 序列中的下一个任务不需要再去中心，直接从上一个任务点出发
                prev_node = task_node
                current_dist_to_center = 0

    return slot_assignments, slot_profit, slot_details


def run_online_simulation_with_center_pickup(
        algo_name: str = 'greedy',  # 新增算法选择参数: 'greedy' 或 'imtao'
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

    # ==================== 2. 初始化工人位置 ====================
    print("\n【阶段 4】初始化工人真实位置...")
    worker_sim = WorkerSimulator(G, config)
    worker_sim.initialize_from_real_data(
        date=test_date, test_start_hour=test_start_hour, prep_minutes=5,
        coords=coords, nodes=nodes, partition=rcc_partition, centers=centers
    )

    # ==================== 3. 在线仿真实验 ====================
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
        # =======================================================

        # 3.1 加载订单
        tasks_per_center = load_task_locations(
            date=test_date, partition=rcc_partition, centers=centers, coords=coords, nodes=nodes,
            start_hour=current_hour if current_minute == 0 else current_hour,
            end_hour=next_hour if next_minute == 0 else next_hour,
            sample_size=None
        )

        # 3.2 获取工人
        workers_per_center = {}
        for region_id in centers.keys():
            workers = worker_sim.get_available_workers_with_center_info(region_id)
            workers_per_center[region_id] = [(w[0], w[1], w[2], w[3], centers[region_id]) for w in workers]
        total_workers = sum(len(w) for w in workers_per_center.values())
        print(f"可用工人: {total_workers} 个 | 当前订单: {sum(len(t) for t in tasks_per_center.values())} 个")

        # 3.3 执行调度分配算法 ==================== (核心修改点)
        if algo_name.lower() == 'greedy':
            slot_assignments, slot_profit, slot_details = greedy_assignment_with_center_pickup(
                G=G, config=config, centers=centers, partition=rcc_partition,
                workers_per_center=workers_per_center, tasks_per_center=tasks_per_center
            )
        elif algo_name.lower() == 'imtao':
            slot_assignments, slot_profit, slot_details = run_imtao_for_slot(
                G=G, config=config, centers_dict=centers,
                workers_per_center=workers_per_center, tasks_per_center=tasks_per_center,
                time_slot_minutes=time_slot_minutes
            )
        else:
            raise ValueError(f"未知的算法: {algo_name}")

        # 3.4 更新物理位置状态
        updated_count = 0
        slot_dist_to_center = 0
        slot_dist_to_task = 0

        # 由于 IMTAO 可能一个工人接多单，我们需要按顺序更新其最终位置
        for detail in slot_details:
            wid = detail['wid']
            task_node = detail['task_node']
            slot_dist_to_center += detail['dist_to_center']
            slot_dist_to_task += detail['dist_to_task']

            if task_node in G.nodes:
                task_lon, task_lat = G.nodes[task_node]['x'], G.nodes[task_node]['y']
                worker_sim.update_worker_position(wid, task_node, task_lon, task_lat)
                worker_sim.set_worker_busy(wid)
                updated_count += 1

        print(
            f"分配结果: 成交 {len(slot_assignments)} 单, 调度 {len(set(k[0] for k in slot_assignments.keys()))} 名工人")

        total_dist_to_center += slot_dist_to_center
        total_dist_to_task += slot_dist_to_task
        all_assignments.update(slot_assignments)
        all_details.extend(slot_details)
        total_profit += slot_profit

        # 3.6 完成任务释放工人 (时间槽末尾统一释放)
        for wid, _ in slot_assignments.keys():
            worker_sim.set_worker_idle(wid)

    print(f"\n[{algo_name.upper()}] 仿真完成: 总利润 {total_profit:.2f} 元, 总成交 {len(all_assignments)} 单")
    return all_assignments, all_details, total_profit


if __name__ == "__main__":
    # ========================================================
    # 毕业论文核心实验：多模型调度效果在线横向对比
    # ========================================================

    # 1. 运行传统的贪心基线算法 (无协同)
    greedy_assignments, greedy_details, greedy_profit = run_online_simulation_with_center_pickup(
        algo_name='greedy', test_date='2016-10-14', test_start_hour=7, test_end_hour=9, time_slot_minutes=15
    )

    # 2. 运行 ICDE 论文复现算法 (博弈论多中心协同)
    imtao_assignments, imtao_details, imtao_profit = run_online_simulation_with_center_pickup(
        algo_name='imtao', test_date='2016-10-14', test_start_hour=7, test_end_hour=9, time_slot_minutes=15
    )

    # 3. 打印论文所需的大表对比数据
    print("\n\n" + "=" * 70)
    print("🏆 毕业论文实验：多中心调度算法性能横向对比")
    print("=" * 70)
    print(f"{'指标':<25} | {'Greedy (局部基线)':<20} | {'IMTAO (多中心协同)':<20}")
    print("-" * 70)
    print(f"{'总分配任务数 (完单量)':<20} | {len(greedy_assignments):<22} | {len(imtao_assignments):<20}")
    print(f"{'系统总净利润 (元)':<22} | {greedy_profit:<22.2f} | {imtao_profit:<20.2f}")

    # 统计空载率 (去中心的距离占总距离的比例)
    greedy_dist_c = sum(d['dist_to_center'] for d in greedy_details)
    greedy_dist_t = sum(d['dist_to_task'] for d in greedy_details)
    imtao_dist_c = sum(d['dist_to_center'] for d in imtao_details)
    imtao_dist_t = sum(d['dist_to_task'] for d in imtao_details)

    if (greedy_dist_c + greedy_dist_t) > 0:
        g_empty_rate = greedy_dist_c / (greedy_dist_c + greedy_dist_t) * 100
    else:
        g_empty_rate = 0
    if (imtao_dist_c + imtao_dist_t) > 0:
        i_empty_rate = imtao_dist_c / (imtao_dist_c + imtao_dist_t) * 100
    else:
        i_empty_rate = 0

    print(f"{'空载行驶占比 (越低越好)':<20} | {g_empty_rate:<21.2f}% | {i_empty_rate:<19.2f}%")
    print("=" * 70)