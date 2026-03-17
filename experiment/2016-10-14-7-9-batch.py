import time

import config
from algorithm.Greedy import greedy_assignment_with_center_pickup
from tool.TaskWorkerToMap import WorkerSimulator, load_task_locations
from tool.data_loader import get_real_road_network
from tool.map_algorithms import run_kmeans_baseline, run_rcc_algorithm, find_region_centers


def run_online_simulation_with_center_pickup(
        test_date: str = '2016-10-14',
        test_start_hour: int = 7,
        test_end_hour: int = 9,
        time_slot_minutes: int = 15
):
    """
    运行在线仿真实验（工人需要先去中心取货）

    流程：
    1. 初始化：加载测试前 5 分钟的工人真实位置
    2. 分时间槽进行实验（如 7:00-7:15, 7:15-7:30, ...）
    3. 每个时间槽：
       - 加载该时间段的订单
       - 使用贪心算法分配（考虑去中心取货）
       - 更新工人位置（移动到任务地点）
       - 不再读取新的工人位置数据
    """
    print("=" * 70)
    print("在线物流调度仿真实验（带中心取货）")
    print("=" * 70)
    print(f"测试日期：{test_date}")
    print(f"测试时段：{test_start_hour}:00 - {test_end_hour}:00")
    print(f"时间槽长度：{time_slot_minutes} 分钟")
    print("=" * 70)

    # ==================== 1. 加载路网和分区 ====================
    print("\n【阶段 1】加载路网数据...")
    t0 = time.time()
    G, coords, nodes = get_real_road_network(config.CHENGDU_CENTER, dist=config.DOWNLOAD_DIST)
    print(f"[耗时] {time.time() - t0:.2f} 秒")

    print("\n【阶段 2】计算区域划分...")
    kmeans_partition = run_kmeans_baseline(coords, nodes, k=config.NUM_ZONES)
    rcc_partition = run_rcc_algorithm(G, kmeans_partition, k=config.NUM_ZONES)
    centers = find_region_centers(G, rcc_partition, weight='length')

    print("\n【阶段 3】显示各区域中心信息...")
    for region_id, center_node in centers.items():
        region_workers = [n for n in rcc_partition if rcc_partition[n] == region_id]
        print(f"   区域 {region_id}: 中心节点={center_node}, 包含{len(region_workers)}个节点")

    # ==================== 2. 初始化工人位置 ====================
    print("\n【阶段 4】初始化工人位置（从真实数据）...")
    worker_sim = WorkerSimulator(G, config)
    worker_sim.initialize_from_real_data(
        date=test_date,
        test_start_hour=test_start_hour,
        prep_minutes=5,
        coords=coords,
        nodes=nodes,
        partition=rcc_partition,
        centers=centers
    )

    # ==================== 3. 在线仿真实验 ====================
    print("\n【阶段 5】开始在线仿真实验...")

    all_assignments = {}
    all_details = []
    total_profit = 0
    total_dist_to_center = 0
    total_dist_to_task = 0

    # 计算总的时间槽数量
    total_minutes = (test_end_hour - test_start_hour) * 60
    num_slots = total_minutes // time_slot_minutes

    for slot_idx in range(num_slots):
        slot_start_minute = slot_idx * time_slot_minutes
        slot_end_minute = (slot_idx + 1) * time_slot_minutes

        current_hour = test_start_hour + slot_start_minute // 60
        current_minute = slot_start_minute % 60

        next_hour = test_start_hour + slot_end_minute // 60
        next_minute = slot_end_minute % 60

        print(f"\n{'=' * 70}")
        print(f"⏰ 时间槽 {slot_idx + 1}/{num_slots}: "
              f"{current_hour:02d}:{current_minute:02d} - {next_hour:02d}:{next_minute:02d}")
        print(f"{'=' * 70}")

        # 3.1 加载该时间段的订单（从真实数据）
        print(f">> 加载订单数据...")
        tasks_per_center = load_task_locations(
            date=test_date,
            partition=rcc_partition,
            centers=centers,
            coords=coords,
            nodes=nodes,
            start_hour=current_hour if current_minute == 0 else current_hour,
            end_hour=next_hour if next_minute == 0 else next_hour,
            sample_size=None
        )

        # 3.2 获取当前工人位置（从模拟器，不读取外部数据）
        print(f">> 获取工人位置（模拟器）...")
        workers_per_center = {}
        for region_id in centers.keys():
            workers = worker_sim.get_available_workers_with_center_info(region_id)
            # 添加中心节点信息
            workers_with_center = [
                (w[0], w[1], w[2], w[3], centers[region_id])
                for w in workers
            ]
            workers_per_center[region_id] = workers_with_center

        total_workers = sum(len(w) for w in workers_per_center.values())
        print(f"   当前可用工人：{total_workers} 个")

        # 3.3 执行贪心分配（带中心取货）
        print(f">> 执行贪心分配算法（带中心取货）...")
        slot_assignments, slot_profit, slot_details = greedy_assignment_with_center_pickup(
            G=G,
            config=config,
            centers=centers,
            partition=rcc_partition,
            workers_per_center=workers_per_center,
            tasks_per_center=tasks_per_center
        )

        # 3.4 更新工人位置（模拟移动到任务地点）
        print(f">> 更新工人位置...")
        updated_count = 0
        slot_dist_to_center = 0
        slot_dist_to_task = 0

        for (wid, task_id), profit in slot_assignments.items():
            # 找到任务位置和详细信息
            for detail in slot_details:
                if detail['wid'] == wid and detail['task_id'] == task_id:
                    slot_dist_to_center += detail['dist_to_center']
                    slot_dist_to_task += detail['dist_to_task']

                    # 获取任务节点坐标
                    task_node = detail['task_node']
                    if task_node in G.nodes:
                        task_lon = G.nodes[task_node]['x']
                        task_lat = G.nodes[task_node]['y']

                        # 更新工人位置到任务位置
                        worker_sim.update_worker_position(wid, task_node, task_lon, task_lat)
                        worker_sim.set_worker_busy(wid)
                        updated_count += 1
                    break

        print(f"   更新了 {updated_count} 个工人位置")
        print(f"   本时间槽总行驶距离：{slot_dist_to_center + slot_dist_to_task:.2f} 米")
        print(f"      - 到中心取货：{slot_dist_to_center:.2f} 米")
        print(f"      - 到任务送货：{slot_dist_to_task:.2f} 米")

        total_dist_to_center += slot_dist_to_center
        total_dist_to_task += slot_dist_to_task

        # 3.5 记录结果
        all_assignments.update(slot_assignments)
        all_details.extend(slot_details)
        total_profit += slot_profit

        # 3.6 完成任务，设为空闲
        for wid in slot_assignments.keys():
            worker_sim.set_worker_idle(wid)

    # ==================== 4. 输出实验结果 ====================
    print(f"\n{'=' * 70}")
    print("🎉 仿真实验完成！")
    print(f"{'=' * 70}")
    print(f"总时间槽数：{num_slots}")
    print(f"总分配任务数：{len(all_assignments)}")
    print(f"总利润：{total_profit:.2f} 元")
    print(f"总行驶距离：{total_dist_to_center + total_dist_to_task:.2f} 米")
    print(f"   - 到中心取货总距离：{total_dist_to_center:.2f} 米")
    print(f"   - 到任务送货总距离：{total_dist_to_task:.2f} 米")
    print(f"平均每单行驶距离：{(total_dist_to_center + total_dist_to_task) / max(1, len(all_assignments)):.2f} 米")

    return all_assignments, all_details, total_profit


if __name__ == "__main__":
    # 运行仿真：测试 7-9 点，每 15 分钟一个时间槽
    assignments, details, profit = run_online_simulation_with_center_pickup(
        test_date='2016-10-14',
        test_start_hour=7,
        test_end_hour=9,
        time_slot_minutes=15
    )