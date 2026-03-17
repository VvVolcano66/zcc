import time
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

import config
from algorithm.Greedy import greedy_assignment_with_center_pickup, calculate_task_profit_with_center_pickup
from tool.TaskWorkerToMap import WorkerSimulator, load_task_locations, snap_to_network
from tool.data_loader import get_real_road_network
from tool.map_algorithms import run_kmeans_baseline, run_rcc_algorithm, find_region_centers


def run_real_time_simulation_with_center_pickup(
        test_date: str = '2016-10-14',
        test_start_hour: int = 7,
        test_end_hour: int = 9,
):
    """
    运行实时仿真实验（事件驱动，订单逐个到达立即分配）

    流程：
    1. 初始化：加载测试前 5 分钟的工人真实位置
    2. 加载所有订单，按时间戳排序
    3. 按订单时间顺序逐个处理：
       - 获取当前时刻的可用工人
       - 使用贪心算法为该订单分配工人
       - 立即更新工人状态和位置
       - 推进仿真时钟
    """
    print("=" * 70)
    print("在线物流调度仿真实验（带中心取货）")
    print("=" * 70)
    print(f"测试日期：{test_date}")
    print(f"测试时段：{test_start_hour}:00 - {test_end_hour}:00")
    print(f"分配模式：实时分配（事件驱动）")
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

    # ==================== 3. 加载并排序订单 ====================
    print("\n【阶段 5】加载订单数据并按时间排序...")
    all_tasks = load_all_tasks_sorted(
        date=test_date,
        start_hour=test_start_hour,
        end_hour=test_end_hour,
        partition=rcc_partition,
        centers=centers,
        coords=coords,
        nodes=nodes
    )
    print(f"共加载 {len(all_tasks)} 个订单")

    # ==================== 4. 实时仿真实验 ====================
    print("\n【阶段 6】开始实时仿真实验...")

    all_assignments = {}
    all_details = []
    total_profit = 0
    total_dist_to_center = 0
    total_dist_to_task = 0

    assigned_count = 0
    unassigned_count = 0
    last_print_time = None

    for task_idx, task_info in enumerate(all_tasks):
        task_id = task_info['task_id']
        task_node = task_info['task_node']
        task_region = task_info['region_id']
        task_reward = task_info['reward']
        task_timestamp = task_info['timestamp']

        current_time_str = format_timestamp(task_timestamp, test_start_hour)

        if task_idx % 50 == 0 or task_idx == len(all_tasks) - 1:
            print(
                f"\n⏰ [{current_time_str}] 处理订单 {task_idx + 1}/{len(all_tasks)} (进度：{task_idx / len(all_tasks) * 100:.1f}%)")
        else:
            print(f"⏰ [{current_time_str}] 处理订单 {task_idx + 1}/{len(all_tasks)}", end='\r')

        workers_per_center = {}
        for region_id in centers.keys():
            workers = worker_sim.get_available_workers_with_center_info(region_id, current_time=task_timestamp)
            workers_with_center = [
                (w[0], w[1], w[2], w[3], centers[region_id])
                for w in workers
            ]
            workers_per_center[region_id] = workers_with_center

        total_workers = sum(len(w) for w in workers_per_center.values())

        if total_workers == 0:
            unassigned_count += 1
            continue

        tasks_for_this_order = {task_region: [(task_node, task_id, task_reward)]}

        slot_assignments, slot_profit, slot_details = greedy_assignment_with_center_pickup(
            G=G,
            config=config,
            centers=centers,
            partition=rcc_partition,
            workers_per_center=workers_per_center,
            tasks_per_center=tasks_for_this_order
        )

        if slot_assignments:
            for (wid, assigned_task_id), profit in slot_assignments.items():
                detail = next((d for d in slot_details if d['wid'] == wid and d['task_id'] == assigned_task_id), None)
                
                if detail:
                    total_dist_to_center += detail['dist_to_center']
                    total_dist_to_task += detail['dist_to_task']
                    travel_distance = detail['distance']
                    
                    assigned_task_node = detail['task_node']
                    if assigned_task_node in G.nodes:
                        task_lon = G.nodes[assigned_task_node]['x']
                        task_lat = G.nodes[assigned_task_node]['y']

                        worker_sim.update_worker_position(wid, assigned_task_node, task_lon, task_lat)

                        total_travel_time = travel_distance / config.WORKER_SPEED_MS
                        worker_sim.set_worker_en_route_to_task(wid, task_timestamp + total_travel_time)

                        assigned_count += 1
                        break

            all_assignments.update(slot_assignments)
            all_details.extend(slot_details)
            total_profit += slot_profit

            if task_idx % 100 == 0:
                print(f"   ✅ 订单 {assigned_task_id} -> 工人 {wid}, 利润：{profit:.2f} 元，距离：{travel_distance:.2f} 米")
        else:
            unassigned_count += 1

    # ==================== 5. 输出实验结果 ====================
    print(f"\n{'=' * 70}")
    print("🎉 仿真实验完成！")
    print(f"{'=' * 70}")
    print(f"总订单数：{len(all_tasks)}")
    print(f"成功分配订单数：{assigned_count}")
    print(f"未分配订单数：{unassigned_count}")
    print(f"分配成功率：{assigned_count / max(1, len(all_tasks)) * 100:.2f}%")
    print(f"总利润：{total_profit:.2f} 元")
    print(f"总行驶距离：{total_dist_to_center + total_dist_to_task:.2f} 米")
    print(f"   - 到中心取货总距离：{total_dist_to_center:.2f} 米")
    print(f"   - 到任务送货总距离：{total_dist_to_task:.2f} 米")
    print(f"平均每单行驶距离：{(total_dist_to_center + total_dist_to_task) / max(1, assigned_count):.2f} 米")
    print(f"平均每单利润：{total_profit / max(1, assigned_count):.2f} 元")

    return all_assignments, all_details, total_profit, assigned_count, unassigned_count


def load_all_tasks_sorted(
        date: str,
        start_hour: int,
        end_hour: int,
        partition,
        centers,
        coords,
        nodes,
        reward_range=(8.0, 15.0)
):
    """
    加载指定时间段的所有订单，并按时间戳排序

    Returns:
        List[Dict]: 按时间排序的订单列表，每个订单包含完整信息
    """
    import os

    file_path = f"D:/biyelunwen/data/task/tasks_{date}.csv"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在：{file_path}")

    df = pd.read_csv(file_path)
    df['first_time'] = pd.to_datetime(df['first_time'])
    df['timestamp'] = df['first_time'].dt.hour * 3600 + df['first_time'].dt.minute * 60 + df['first_time'].dt.second

    mask = (df['timestamp'] >= start_hour * 3600) & (df['timestamp'] < end_hour * 3600)
    filtered_df = df[mask].copy()

    filtered_df = filtered_df.sort_values('timestamp').reset_index(drop=True)

    tasks_list = []
    matched_count = 0

    for _, row in filtered_df.iterrows():
        lon = row['first_lon']
        lat = row['first_lat']
        task_id = row['task_id']
        timestamp = row['timestamp']

        nearest_node = snap_to_network(lon, lat, coords, nodes)

        if nearest_node in partition:
            region_id = partition[nearest_node]
            if region_id in centers:
                reward = np.random.uniform(reward_range[0], reward_range[1])

                tasks_list.append({
                    'task_id': task_id,
                    'task_node': nearest_node,
                    'region_id': region_id,
                    'reward': reward,
                    'timestamp': timestamp,
                    'original_lon': lon,
                    'original_lat': lat
                })
                matched_count += 1

    print(f"✅ 加载完成：原始订单 {len(filtered_df)} 个，匹配到路网 {matched_count} 个")

    return tasks_list


def format_timestamp(timestamp_seconds: float, base_hour: int) -> str:
    """将时间戳转换为可读格式"""
    hours = int(timestamp_seconds // 3600)
    minutes = int((timestamp_seconds % 3600) // 60)
    seconds = int(timestamp_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    assignments, details, profit, assigned, unassigned = run_real_time_simulation_with_center_pickup(
        test_date='2016-10-14',
        test_start_hour=7,
        test_end_hour=9,
    )