import os
import pickle
import time
from typing import Dict, List, Tuple, Any

import config
from algorithm.Greedy import greedy_assignment
from tool.data_loader import get_real_road_network
from tool.TaskWorkerToMap import load_worker_locations, load_task_locations
from tool.get_result import generate_assignment_report


def run_greedy_with_cached_data(
        date: str = '2016-10-14',
        start_hour: int = 7,
        end_hour: int = 9,
        worker_sample_size: int = 500,
        task_sample_size: int = 1000,
        reward_range: Tuple[float, float] = (8.0, 15.0)
) -> None:
    """
    使用缓存的分区和中心点数据运行 Greedy 任务分配算法

    Args:
        date: 分析日期，格式 'YYYY-MM-DD'
        start_hour: 起始小时 (0-23)
        end_hour: 结束小时 (0-23)
        worker_sample_size: 工人采样数量
        task_sample_size: 任务采样数量
        reward_range: 任务奖励范围 (min, max)
    """
    print("=" * 50)
    print("🚀 开始执行 Greedy 任务分配（使用缓存数据）")
    print("=" * 50)

    # ==================== 1. 导入依赖模块 ====================

    import config

    # ==================== 2. 加载路网数据 ====================
    print("\n[1/5] 加载路网数据...")
    t0 = time.time()
    G, coords, nodes = get_real_road_network(config.CHENGDU_CENTER, dist=config.DOWNLOAD_DIST)
    print(f"✅ 路网加载完成，用时：{time.time() - t0:.2f} 秒")

    # ==================== 3. 加载缓存的分区和中心点 ====================
    print("\n[2/5] 加载缓存的分区和中心点数据...")
    algo_cache_dir = r"D:\biyelunwen\tool\cache"
    algo_cache_file = os.path.join(
        algo_cache_dir,
        f"algo_results_{config.CITY_NAME}_k{config.NUM_ZONES}_dist{config.DOWNLOAD_DIST}.pkl"
    )

    if not os.path.exists(algo_cache_file):
        print(f"❌ 错误：未找到缓存文件 {algo_cache_file}")
        print("请先运行 main.py 生成缓存数据！")
        return

    with open(algo_cache_file, 'rb') as f:
        rcc_partition, centers = pickle.load(f)

    print(f"✅ 缓存加载成功：{len(centers)} 个中心区域")

    # ==================== 4. 加载工人和任务数据 ====================
    print(f"\n[3/5] 加载工人和任务数据...")
    print(f"   📅 日期：{date}")
    print(f"   ⏰ 时间段：{start_hour}:00 - {end_hour}:00")

    # 加载工人位置
    workers_per_center = load_worker_locations(
        date=date,
        partition=rcc_partition,
        centers=centers,
        coords=coords,
        nodes=nodes,
        start_hour=start_hour,
        end_hour=end_hour,
        sample_size=worker_sample_size
    )

    # 加载任务位置
    tasks_per_center = load_task_locations(
        date=date,
        partition=rcc_partition,
        centers=centers,
        coords=coords,
        nodes=nodes,
        start_hour=start_hour,
        end_hour=end_hour,
        sample_size=task_sample_size,
        reward_range=reward_range
    )

    # ==================== 5. 执行 Greedy 分配 ====================
    print(f"\n[4/5] 执行贪心任务分配算法...")
    t_start = time.time()

    assignments, total_profit, details = greedy_assignment(
        G=G,
        config=config,
        centers=centers,
        partition=rcc_partition,
        workers_per_center=workers_per_center,
        tasks_per_center=tasks_per_center
    )

    elapsed_time = time.time() - t_start
    print(f"⏱️  算法执行时间：{elapsed_time:.2f} 秒")

    # ==================== 6. 生成报告 ====================
    print(f"\n[5/5] 生成分配报告...")
        generate_assignment_report(details)

    # ==================== 7. 保存分配结果 ====================
    result_cache_file = os.path.join(
        algo_cache_dir,
        f"greedy_result_{date}_{start_hour}-{end_hour}.pkl"
    )

    result_data = {
        'date': date,
        'start_hour': start_hour,
        'end_hour': end_hour,
        'assignments': assignments,
        'total_profit': total_profit,
        'details': details,
        'execution_time': elapsed_time
    }

    with open(result_cache_file, 'wb') as f:
        pickle.dump(result_data, f)

    print(f"\n✅ 分配结果已保存到：{result_cache_file}")
    print("=" * 50)
    print("🎉 全部流程完成！")
    print("=" * 50)


def compare_time_periods():
    """
    对比不同时段的分配结果

    测试三个时段：
    1. 早高峰 (7:00-9:00)
    2. 午高峰 (11:00-13:00)
    3. 晚高峰 (17:00-19:00)
    """
    print("=" * 50)
    print("📊 开始对比不同时段的分配结果")
    print("=" * 50)

    time_periods = [
        (7, 9, "早高峰"),
        (11, 13, "午高峰"),
        (17, 19, "晚高峰")
    ]

    results = []

    for start_hour, end_hour, period_name in time_periods:
        print(f"\n{'=' * 50}")
        print(f"正在分析：{period_name} ({start_hour}:00-{end_hour}:00)")
        print(f"{'=' * 50}")



        # 加载路网
        G, coords, nodes = get_real_road_network(
            config.CHENGDU_CENTER,
            dist=config.DOWNLOAD_DIST
        )

        # 加载缓存
        algo_cache_dir = r"D:\biyelunwen\tool\cache"
        algo_cache_file = os.path.join(
            algo_cache_dir,
            f"algo_results_{config.CITY_NAME}_k{config.NUM_ZONES}_dist{config.DOWNLOAD_DIST}.pkl"
        )

        with open(algo_cache_file, 'rb') as f:
            rcc_partition, centers = pickle.load(f)

        # 加载数据
        workers = load_worker_locations(
            date='2016-10-14',
            partition=rcc_partition,
            centers=centers,
            coords=coords,
            nodes=nodes,
            start_hour=start_hour,
            end_hour=end_hour,
            sample_size=500
        )

        tasks = load_task_locations(
            date='2016-10-14',
            partition=rcc_partition,
            centers=centers,
            coords=coords,
            nodes=nodes,
            start_hour=start_hour,
            end_hour=end_hour,
            sample_size=1000
        )

        # 执行分配
        assignments, total_profit, details = greedy_assignment(
            G=G,
            config=config,
            centers=centers,
            partition=rcc_partition,
            workers_per_center=workers,
            tasks_per_center=tasks
        )

        results.append({
            'period': period_name,
            'start_hour': start_hour,
            'end_hour': end_hour,
            'assigned_tasks': len(assignments),
            'total_profit': total_profit,
            'profit_per_task': total_profit / len(assignments) if assignments else 0
        })

    # 打印对比结果
    print("\n" + "=" * 70)
    print("📈 各时段对比结果")
    print("=" * 70)
    print(f"{'时段':<10} {'分配任务数':<15} {'总利润 (元)':<15} {'平均利润 (元)':<15}")
    print("-" * 70)

    for result in results:
        print(f"{result['period']:<10} {result['assigned_tasks']:<15} "
              f"{result['total_profit']:<15.2f} {result['profit_per_task']:<15.2f}")

    print("=" * 70)

    # 找出最优时段
    best_period = max(results, key=lambda x: x['total_profit'])
    print(f"\n💡 最优时段：{best_period['period']} (总利润：{best_period['total_profit']:.2f} 元)")


if __name__ == "__main__":
    # 方式 1: 运行单个时段的分析
    run_greedy_with_cached_data(
        date='2016-10-14',
        start_hour=7,
        end_hour=9,
        worker_sample_size=500,
        task_sample_size=1000,
        reward_range=(8.0, 15.0)
    )

    # 方式 2: 对比多个时段（取消注释以启用）
    # compare_time_periods()