import time
import copy
import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import KDTree

import config
from algorithm.Greedy import greedy_assignment_with_center_pickup
from algorithm.PredictiveDispatch import predict_next_slot_demand, predispatch_workers_for_next_slot
from algorithm.GameTheoreticPredictiveDispatch import game_theoretic_predispatch_workers
from predicate.MCTGNetDispatchPredictor import MCTGNetDispatchPredictor
from tool.TaskWorkerToMap import WorkerSimulator
from tool.data_loader import get_real_road_network
from tool.map_algorithms import run_kmeans_baseline, run_rcc_algorithm, find_region_centers

from algorithm.IMTAO import Center as IMTAOCenter, Worker as IMTAOWorker, Task as IMTAOTask, IMTAO_Framework

DEFAULT_TEST_DATE = getattr(config, 'EXPERIMENT_TEST_DATE', '2016-10-31')
DEFAULT_START_HOUR = int(getattr(config, 'EXPERIMENT_START_HOUR', 7))
DEFAULT_END_HOUR = int(getattr(config, 'EXPERIMENT_END_HOUR', 9))
DEFAULT_TIME_SLOT_MINUTES = int(getattr(config, 'EXPERIMENT_TIME_SLOT_MINUTES', 15))
DEFAULT_PREP_MINUTES = int(getattr(config, 'WORKER_INIT_PREP_MINUTES', 5))

_SIMULATION_CONTEXT_CACHE = {}
_MCTG_PREDICTOR_CACHE = {}


def build_prediction_date_split(test_date: str, data_dir: str):
    test_ts = pd.Timestamp(test_date).normalize()
    val_days = max(1, int(getattr(config, 'DISPATCH_PRED_VAL_DAYS', 2)))

    available_dates = []
    for file_name in sorted(os.listdir(data_dir)):
        if not (file_name.startswith('tasks_') and file_name.endswith('.csv')):
            continue
        date_str = file_name[len('tasks_'):-len('.csv')]
        try:
            date_ts = pd.Timestamp(date_str).normalize()
        except ValueError:
            continue
        if date_ts < test_ts:
            available_dates.append(date_str)

    if len(available_dates) <= val_days:
        raise ValueError(
            f"Not enough historical task files before {test_date} to build train/val split: "
            f"{len(available_dates)} available, {val_days} reserved for validation."
        )

    train_dates = available_dates[:-val_days]
    val_dates = available_dates[-val_days:]
    return train_dates, val_dates


def _build_simulation_context(test_date: str, test_start_hour: int, test_end_hour: int):
    cache_key = (
        test_date,
        test_start_hour,
        test_end_hour,
        config.NUM_ZONES,
        config.CHENGDU_CENTER,
        config.DOWNLOAD_DIST,
        DEFAULT_PREP_MINUTES
    )
    if cache_key in _SIMULATION_CONTEXT_CACHE:
        print("\n【阶段 1-4】复用缓存的地图、订单与工人初始数据...")
        return _SIMULATION_CONTEXT_CACHE[cache_key]

    print("\n【阶段 1-3】加载路网数据与中心划分...")
    G, coords, nodes = get_real_road_network(config.CHENGDU_CENTER, dist=config.DOWNLOAD_DIST)
    kmeans_partition = run_kmeans_baseline(coords, nodes, k=config.NUM_ZONES)
    rcc_partition = run_rcc_algorithm(G, kmeans_partition, k=config.NUM_ZONES)
    centers = find_region_centers(G, rcc_partition, weight='length')

    print("\n【阶段 3.0】预加载并构建全局订单池 (精确到秒级划分)...")
    task_file = f"D:/biyelunwen/data/task/tasks_{test_date}.csv"
    if os.path.exists(task_file):
        df_tasks = pd.read_csv(task_file)
        df_tasks['first_time'] = pd.to_datetime(df_tasks['first_time'])
        df_tasks['seconds_of_day'] = (
            df_tasks['first_time'].dt.hour * 3600
            + df_tasks['first_time'].dt.minute * 60
            + df_tasks['first_time'].dt.second
        )

        tree = KDTree(coords)
        task_coords = df_tasks[['first_lon', 'first_lat']].values
        _, idxs = tree.query(task_coords)
        df_tasks['nearest_node'] = [nodes[i] for i in idxs]
        print(f"✅ 全局订单池就绪，共 {len(df_tasks)} 个任务待命。")

        eval_mask = (df_tasks['seconds_of_day'] >= test_start_hour * 3600) & \
                    (df_tasks['seconds_of_day'] < test_end_hour * 3600)
        eval_tasks = df_tasks[eval_mask].copy()
        eval_tasks = eval_tasks[eval_tasks['nearest_node'].isin(rcc_partition)].copy()
        eval_tasks['region_id'] = eval_tasks['nearest_node'].map(rcc_partition)
    else:
        print(f"⚠️ 未找到任务文件: {task_file}")
        df_tasks = pd.DataFrame()
        eval_tasks = pd.DataFrame(columns=['region_id'])

    total_tasks_per_center = {region_id: 0 for region_id in centers.keys()}
    if not eval_tasks.empty:
        for region_id, count in eval_tasks['region_id'].value_counts().items():
            total_tasks_per_center[region_id] = int(count)

    print("\n【阶段 4】初始化工人真实位置...")
    base_worker_sim = WorkerSimulator(G, config)
    base_worker_sim.initialize_from_real_data(
        date=test_date,
        test_start_hour=test_start_hour,
        prep_minutes=DEFAULT_PREP_MINUTES,
        coords=coords,
        nodes=nodes,
        partition=rcc_partition,
        centers=centers
    )

    context = {
        'G': G,
        'coords': coords,
        'nodes': nodes,
        'rcc_partition': rcc_partition,
        'centers': centers,
        'df_tasks': df_tasks,
        'eval_tasks': eval_tasks,
        'total_tasks_per_center': total_tasks_per_center,
        'worker_state': {
            'worker_positions': copy.deepcopy(base_worker_sim.worker_positions),
            'worker_status': copy.deepcopy(base_worker_sim.worker_status),
            'worker_center_map': copy.deepcopy(base_worker_sim.worker_center_map),
            'worker_busy_until': copy.deepcopy(base_worker_sim.worker_busy_until),
            'worker_available_from': copy.deepcopy(base_worker_sim.worker_available_from),
        }
    }
    _SIMULATION_CONTEXT_CACHE[cache_key] = context
    return context


def _restore_worker_simulator(G, worker_state):
    worker_sim = WorkerSimulator(G, config)
    worker_sim.worker_positions = copy.deepcopy(worker_state['worker_positions'])
    worker_sim.worker_status = copy.deepcopy(worker_state['worker_status'])
    worker_sim.worker_center_map = copy.deepcopy(worker_state['worker_center_map'])
    worker_sim.worker_busy_until = copy.deepcopy(worker_state['worker_busy_until'])
    worker_sim.worker_available_from = copy.deepcopy(worker_state['worker_available_from'])
    return worker_sim


def _get_or_train_mctg_predictor(
        test_date: str,
        test_start_hour: int,
        test_end_hour: int,
        time_slot_minutes: int,
        coords,
        nodes,
        rcc_partition,
        centers
):
    predictor_key = (
        test_date,
        test_start_hour,
        test_end_hour,
        time_slot_minutes,
        getattr(config, 'DISPATCH_PRED_SEQ_LEN', 4),
        getattr(config, 'DISPATCH_PRED_PRE_LEN', 1),
        getattr(config, 'DISPATCH_PRED_VAL_DAYS', 2),
        getattr(config, 'MCTGNET_DISPATCH_MAX_EPOCHS', 300),
        getattr(config, 'MCTGNET_DISPATCH_PATIENCE', 50),
        getattr(config, 'MCTGNET_DISPATCH_LR', 0.0005),
    )
    if predictor_key in _MCTG_PREDICTOR_CACHE:
        print("   - Reusing cached MCTGNet predictor")
        return _MCTG_PREDICTOR_CACHE[predictor_key]

    dispatch_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'task')
    train_dates, val_dates = build_prediction_date_split(test_date, dispatch_data_dir)
    history_span_minutes = getattr(config, 'DISPATCH_PRED_SEQ_LEN', 4) * time_slot_minutes
    history_start_hour = max(0, test_start_hour - int(np.ceil(history_span_minutes / 60.0)))
    print(f"\n[阶段 4.5] 训练 MCTGNet 预测器并为 {test_date} 调度准备历史上下文...")
    print(f"   - Train Dates: {train_dates[0]} ~ {train_dates[-1]}")
    print(f"   - Val Dates:   {val_dates[0]} ~ {val_dates[-1]}")
    print(f"   - Train Days:  {len(train_dates)} | Val Days: {len(val_dates)}")
    print(f"   - Target Date: {test_date}")
    print(f"   - History Window: {history_start_hour}:00 - {test_end_hour}:00")

    predictor = MCTGNetDispatchPredictor(
        data_dir=dispatch_data_dir,
        coords=coords,
        nodes=nodes,
        partition=rcc_partition,
        centers=centers,
        time_interval=time_slot_minutes,
        seq_len=getattr(config, 'DISPATCH_PRED_SEQ_LEN', 4),
        pre_len=getattr(config, 'DISPATCH_PRED_PRE_LEN', 1),
        max_epochs=getattr(config, 'MCTGNET_DISPATCH_MAX_EPOCHS', 300),
        patience=getattr(config, 'MCTGNET_DISPATCH_PATIENCE', 50),
        lr=getattr(config, 'MCTGNET_DISPATCH_LR', 0.0005),
        log_interval=getattr(config, 'MCTGNET_DISPATCH_LOG_INTERVAL', 20)
    )
    predictor.fit(
        train_dates=train_dates,
        val_dates=val_dates,
        target_date=test_date,
        history_start_hour=history_start_hour,
        end_hour=test_end_hour
    )
    print("   - MCTGNet predictor ready")
    _MCTG_PREDICTOR_CACHE[predictor_key] = predictor
    return predictor


def run_imtao_for_slot(G, config, centers_dict, workers_per_center, tasks_per_center, slot_start_seconds):
    """
    IMTAO 算法适配器（修复版：强制增加中心取货约束）
    """
    imtao_centers = []
    imtao_workers = []
    imtao_tasks = []
    center_worker_map = {}
    center_task_map = {}

    worker_node_map = {}
    task_node_map = {}
    task_reward_map = {}
    task_expire_map = {}
    path_time_cache = {}

    def route_travel_time(src_node, dst_node):
        if src_node == dst_node:
            return 0.0
        pair = (src_node, dst_node) if str(src_node) < str(dst_node) else (dst_node, src_node)
        if pair not in path_time_cache:
            try:
                dist = nx.shortest_path_length(G, source=src_node, target=dst_node, weight='length')
                path_time_cache[pair] = dist / config.WORKER_SPEED_MS
            except nx.NetworkXNoPath:
                path_time_cache[pair] = float('inf')
        return path_time_cache[pair]

    for rid, c_node in centers_dict.items():
        c_lon = G.nodes[c_node].get('x', G.nodes[c_node].get('lon'))
        c_lat = G.nodes[c_node].get('y', G.nodes[c_node].get('lat'))
        center = IMTAOCenter(rid, c_lon, c_lat, node=c_node)
        imtao_centers.append(center)
        center_worker_map[rid] = []
        center_task_map[rid] = []

    for rid, w_list in workers_per_center.items():
        for w in w_list:
            w_node, wid, w_lon, w_lat, _ = w
            # 论文中的 worker location 使用工人当前位置，而不是中心位置。
            worker = IMTAOWorker(wid, w_lon, w_lat, max_t=config.MAX_TASKS_PER_WORKER, node=w_node)
            imtao_workers.append(worker)
            center_worker_map[rid].append(worker)
            worker_node_map[wid] = w_node

    for rid, t_list in tasks_per_center.items():
        for t in t_list:
            t_node, tid, reward, expire_seconds = t[0], t[1], t[2], t[3]
            release_seconds = t[4] if len(t) > 4 else slot_start_seconds

            t_lon = G.nodes[t_node].get('x', G.nodes[t_node].get('lon'))
            t_lat = G.nodes[t_node].get('y', G.nodes[t_node].get('lat'))

            relative_expire_seconds = max(0, expire_seconds - slot_start_seconds)
            relative_release_seconds = max(0, release_seconds - slot_start_seconds)
            task = IMTAOTask(
                tid,
                t_lon,
                t_lat,
                expire_time=relative_expire_seconds,
                release_time=relative_release_seconds,
                node=t_node
            )
            imtao_tasks.append(task)
            center_task_map[rid].append(task)
            task_node_map[tid] = t_node
            task_reward_map[tid] = reward
            task_expire_map[tid] = expire_seconds

    if not imtao_tasks:
        return {}, 0, []

    framework = IMTAO_Framework(
        imtao_centers,
        imtao_tasks,
        imtao_workers,
        travel_time_func=route_travel_time
    )
    framework.initialize_existing_partition(center_task_map, center_worker_map)
    framework.algo3_game_theoretic_collaboration(repartition=False)

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
            current_finish_time = slot_start_seconds + dist_to_center / config.WORKER_SPEED_MS

            for task in assigned_tasks:
                task_node = task_node_map[task.id]
                try:
                    dist_to_task = nx.shortest_path_length(G, prev_node, task_node, weight='length')
                except nx.NetworkXNoPath:
                    continue

                travel_cost = (current_dist_to_center + dist_to_task) * config.TRAVEL_COST_PER_METER
                reward = task_reward_map[task.id]
                profit = reward - travel_cost
                candidate_finish_time = current_finish_time + dist_to_task / config.WORKER_SPEED_MS
                if candidate_finish_time > task_expire_map[task.id]:
                    break
                current_finish_time = candidate_finish_time

                slot_assignments[(w.id, task.id)] = profit
                slot_details.append({
                    'region_id': c.id,
                    'wid': w.id,
                    'task_id': task.id,
                    'dist_to_center': current_dist_to_center,
                    'dist_to_task': dist_to_task,
                    'task_node': task_node,
                    'reward': reward,
                    'cost': travel_cost,
                    'finish_time': current_finish_time,
                    'profit': profit
                })
                slot_profit += profit

                prev_node = task_node
                current_dist_to_center = 0

    return slot_assignments, slot_profit, slot_details


def calculate_collaboration_unfairness(total_tasks_per_center, assigned_tasks_per_center):
    """
    Paper Eq. (2)(3):
    rho_i = |A(c_i).S| / |c_i.S|
    U_rho = average pairwise absolute difference of rho_i
    """
    region_ids = sorted(total_tasks_per_center.keys())
    if len(region_ids) <= 1:
        return {rid: 1.0 for rid in region_ids}, 0.0

    rho = {}
    for rid in region_ids:
        total_tasks = total_tasks_per_center.get(rid, 0)
        assigned_tasks = assigned_tasks_per_center.get(rid, 0)
        rho[rid] = assigned_tasks / total_tasks if total_tasks > 0 else 1.0

    u_rho = 0.0
    for i in range(len(region_ids)):
        for j in range(len(region_ids)):
            if i != j:
                u_rho += abs(rho[region_ids[i]] - rho[region_ids[j]])
    u_rho /= (len(region_ids) * (len(region_ids) - 1))
    return rho, u_rho


def run_online_simulation_with_center_pickup(
        algo_name: str = 'greedy',
        test_date: str = DEFAULT_TEST_DATE,
        test_start_hour: int = DEFAULT_START_HOUR,
        test_end_hour: int = DEFAULT_END_HOUR,
        time_slot_minutes: int = DEFAULT_TIME_SLOT_MINUTES
):
    cpu_start = time.process_time()
    print("=" * 70)
    print(f"在线物流调度仿真实验 (算法: {algo_name.upper()})")
    print("=" * 70)
    print(f"测试日期：{test_date} | 时段：{test_start_hour}:00-{test_end_hour}:00 | 时间槽：{time_slot_minutes} 分钟")
    print("=" * 70)

    context = _build_simulation_context(test_date, test_start_hour, test_end_hour)
    G = context['G']
    rcc_partition = context['rcc_partition']
    centers = context['centers']
    df_tasks = context['df_tasks']
    total_tasks_per_center = copy.deepcopy(context['total_tasks_per_center'])
    worker_sim = _restore_worker_simulator(G, context['worker_state'])

    unassigned_tasks_pool = {region_id: [] for region_id in centers.keys()}
    total_expired_tasks_global = 0

    # ==================== 4. 在线仿真实验 ====================
    print("\n【阶段 5】开始在线流式仿真...")
    all_assignments = {}
    all_details = []
    total_profit = 0
    total_dist_to_center = 0
    total_dist_to_task = 0

    total_minutes = (test_end_hour - test_start_hour) * 60
    num_slots = total_minutes // time_slot_minutes
    observed_arrivals_history = []
    mctg_dispatch_predictor = None
    prediction_abs_errors = []
    prediction_sq_errors = []

    if algo_name.lower() in ['predictive_mctgnet', 'predictive_game_mctgnet', 'predictive_bstgcnet']:
        mctg_dispatch_predictor = _get_or_train_mctg_predictor(
            test_date=test_date,
            test_start_hour=test_start_hour,
            test_end_hour=test_end_hour,
            time_slot_minutes=time_slot_minutes,
            coords=context['coords'],
            nodes=context['nodes'],
            rcc_partition=rcc_partition,
            centers=centers
        )

    for slot_idx in range(num_slots):
        slot_start_minute = slot_idx * time_slot_minutes
        slot_end_minute = (slot_idx + 1) * time_slot_minutes
        current_hour = test_start_hour + slot_start_minute // 60
        current_minute = slot_start_minute % 60
        next_hour = test_start_hour + slot_end_minute // 60
        next_minute = slot_end_minute % 60

        print(
            f"\n--- 时间槽 {slot_idx + 1}/{num_slots}: {current_hour:02d}:{current_minute:02d} - {next_hour:02d}:{next_minute:02d} ---")

        # 4.1 提取当前时间槽的订单，并合并上轮积压的订单
        slot_start_seconds = test_start_hour * 3600 + slot_start_minute * 60
        slot_end_seconds = test_start_hour * 3600 + slot_end_minute * 60
        print(f">> 按当前时刻推进工人位置与状态...")
        worker_sim.advance_workers_to_time(centers, slot_start_seconds)
        current_slot_predicted_demand = None
        current_predict_label = None

        if algo_name.lower() in ['predictive_mctgnet', 'predictive_game_mctgnet', 'predictive_bstgcnet']:
            slot_timestamp = pd.Timestamp(test_date) + pd.Timedelta(seconds=slot_start_seconds)
            predicted_demand = mctg_dispatch_predictor.predict_region_demand(slot_timestamp)
            if predicted_demand is not None:
                current_slot_predicted_demand = predicted_demand
                backlog_counts = {rid: len(unassigned_tasks_pool[rid]) for rid in centers.keys()}
                if algo_name.lower() == 'predictive_game_mctgnet':
                    predispatch_result = game_theoretic_predispatch_workers(
                        G=G,
                        worker_sim=worker_sim,
                        centers=centers,
                        predicted_demand=predicted_demand,
                        next_slot_start_seconds=slot_start_seconds,
                        max_tasks_per_worker=getattr(config, 'MAX_TASKS_PER_WORKER', 4),
                        backlog_counts=backlog_counts,
                        backlog_weight=getattr(config, 'PREDISPATCH_BACKLOG_WEIGHT', 1.0),
                        min_buffer_workers=getattr(config, 'PREDISPATCH_MIN_BUFFER_WORKERS', 3),
                        reserve_ratio=getattr(config, 'PREDISPATCH_RESERVE_RATIO', 0.15),
                        max_rebalance_share=getattr(config, 'PREDISPATCH_MAX_SHARE_PER_DONOR', 0.35),
                        max_distance_km=getattr(config, 'PREDISPATCH_MAX_DISTANCE_KM', 8.0),
                        fairness_weight=getattr(config, 'GAME_DISPATCH_FAIRNESS_WEIGHT', 0.5),
                        distance_penalty=getattr(config, 'GAME_DISPATCH_DISTANCE_PENALTY', 0.015),
                        donor_max_utility_drop=getattr(config, 'GAME_DISPATCH_DONOR_MAX_UTILITY_DROP', 0.04),
                        receiver_min_utility_gain=getattr(config, 'GAME_DISPATCH_RECEIVER_MIN_GAIN', 0.01),
                        max_iterations=getattr(config, 'GAME_DISPATCH_MAX_ITERATIONS', 120)
                    )
                    predict_label = 'Game-MCTGNet Predict'
                else:
                    predispatch_result = predispatch_workers_for_next_slot(
                        G=G,
                        worker_sim=worker_sim,
                        centers=centers,
                        predicted_demand=predicted_demand,
                        next_slot_start_seconds=slot_start_seconds,
                        max_tasks_per_worker=getattr(config, 'MAX_TASKS_PER_WORKER', 4),
                        backlog_counts=backlog_counts,
                        backlog_weight=getattr(config, 'PREDISPATCH_BACKLOG_WEIGHT', 1.0),
                        min_buffer_workers=getattr(config, 'PREDISPATCH_MIN_BUFFER_WORKERS', 3),
                        reserve_ratio=getattr(config, 'PREDISPATCH_RESERVE_RATIO', 0.15),
                        max_rebalance_share=getattr(config, 'PREDISPATCH_MAX_SHARE_PER_DONOR', 0.35),
                        max_distance_km=getattr(config, 'PREDISPATCH_MAX_DISTANCE_KM', 8.0)
                    )
                    predict_label = 'MCTGNet Predict'
                current_predict_label = predict_label
                prediction_text = ", ".join(
                    [
                        f"R{rid}: pred={predicted_demand.get(rid, 0)}, "
                        f"eff={predispatch_result['effective_demand'].get(rid, 0)}"
                        for rid in sorted(centers.keys())
                    ]
                )
                print(f"   [{predict_label}] current-slot forecast: {prediction_text}")
                if predispatch_result['moves']:
                    move_summary = ", ".join(
                        [f"{m['wid']}:{m['from_region']}->{m['to_region']}" for m in predispatch_result['moves'][:8]]
                    )
                    if len(predispatch_result['moves']) > 8:
                        move_summary += f", ... (+{len(predispatch_result['moves']) - 8} more)"
                    print(f"   [{predict_label}] pre-dispatched {len(predispatch_result['moves'])} workers: {move_summary}")
                else:
                    print(f"   [{predict_label}] no worker rebalancing needed")
            else:
                print("   [MCTGNet Predict] insufficient same-day history, skip pre-dispatch for this slot")

        tasks_per_center = {region_id: [] for region_id in centers.keys()}
        slot_new_tasks_per_center = {region_id: 0 for region_id in centers.keys()}

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
                    reward = config.TASK_BASE_REWARD
                    release_seconds = row['seconds_of_day']
                    expire_seconds = row['seconds_of_day'] + config.TASK_EXPIRE_MINUTES * 60

                    tasks_per_center[region_id].append(
                        (nearest_node, row['task_id'], reward, expire_seconds, release_seconds)
                    )
                    slot_new_tasks_per_center[region_id] += 1
                    new_tasks_count += 1

        if current_slot_predicted_demand is not None:
            actual_text = ", ".join(
                [
                    f"R{rid}: actual={slot_new_tasks_per_center[rid]}, "
                    f"abs={abs(current_slot_predicted_demand.get(rid, 0) - slot_new_tasks_per_center[rid])}"
                    for rid in sorted(centers.keys())
                ]
            )
            print(f"   [{current_predict_label}] actual arrivals: {actual_text}")
            for rid in centers.keys():
                err = float(current_slot_predicted_demand.get(rid, 0) - slot_new_tasks_per_center[rid])
                prediction_abs_errors.append(abs(err))
                prediction_sq_errors.append(err * err)

        # 4.2 获取可用工人
        workers_per_center = {}
        for region_id in centers.keys():
            workers = worker_sim.get_available_workers_with_center_info(
                region_id,
                current_time=slot_start_seconds
            )
            workers_per_center[region_id] = [(w[0], w[1], w[2], w[3], centers[region_id]) for w in workers]

        total_workers = sum(len(w) for w in workers_per_center.values())
        total_current_tasks = sum(len(t) for t in tasks_per_center.values())
        print(f"可用工人: {total_workers} 个 | 新增订单: {new_tasks_count} 个 | 池内总单量: {total_current_tasks} 个")

        if total_current_tasks == 0:
            print("本时段无订单，跳过分配。")
            continue

        # 4.3 执行调度分配算法
        if algo_name.lower() in ['greedy', 'predictive_greedy', 'predictive_mctgnet', 'predictive_game_mctgnet', 'predictive_bstgcnet']:
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
        worker_final_state = {}

        for detail in slot_details:
            wid = detail['wid']
            task_node = detail['task_node']
            slot_dist_to_center += detail['dist_to_center']
            slot_dist_to_task += detail['dist_to_task']

            prev_detail = worker_final_state.get(wid)
            if prev_detail is None or detail['finish_time'] > prev_detail['finish_time']:
                worker_final_state[wid] = detail

        for wid, detail in worker_final_state.items():
            task_node = detail['task_node']
            if task_node in G.nodes:
                task_lon = G.nodes[task_node].get('x', G.nodes[task_node].get('lon'))
                task_lat = G.nodes[task_node].get('y', G.nodes[task_node].get('lat'))
                worker_sim.update_worker_position(wid, task_node, task_lon, task_lat)
                worker_sim.set_worker_en_route_to_task(wid, detail['finish_time'])

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

        if algo_name.lower() == 'predictive_greedy':
            observed_arrivals_history.append(slot_new_tasks_per_center)
            if slot_idx < num_slots - 1:
                predispatch_result = predispatch_workers_for_next_slot(
                    G=G,
                    worker_sim=worker_sim,
                    centers=centers,
                    predicted_demand=predict_next_slot_demand(
                        observed_arrivals_history=observed_arrivals_history,
                        backlog_counts={rid: len(unassigned_tasks_pool[rid]) for rid in centers.keys()},
                        centers=centers
                    ),
                    next_slot_start_seconds=slot_end_seconds,
                    max_tasks_per_worker=getattr(config, 'MAX_TASKS_PER_WORKER', 4)
                )
                prediction_text = ", ".join(
                    [
                        f"R{rid}: demand={predispatch_result['predicted_demand'][rid]}, "
                        f"workers={predispatch_result['available_workers'][rid]}->"
                        f"{predispatch_result['required_workers'][rid]}"
                        for rid in sorted(centers.keys())
                    ]
                )
                print(f"   [Predictive] next-slot forecast: {prediction_text}")
                moved_workers = predispatch_result['moves']
                if moved_workers:
                    move_summary = ", ".join(
                        [f"{m['wid']}:{m['from_region']}->{m['to_region']}" for m in moved_workers[:8]]
                    )
                    if len(moved_workers) > 8:
                        move_summary += f", ... (+{len(moved_workers) - 8} more)"
                    print(f"   [Predictive] pre-dispatched {len(moved_workers)} workers: {move_summary}")
                else:
                    print("   [Predictive] no worker rebalancing needed for next slot")

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

    assigned_tasks_per_center = {region_id: 0 for region_id in centers.keys()}
    for detail in all_details:
        assigned_tasks_per_center[detail['region_id']] += 1

    rho, u_rho = calculate_collaboration_unfairness(total_tasks_per_center, assigned_tasks_per_center)
    cpu_time = time.process_time() - cpu_start

    print(f"\n[{algo_name.upper()}] 仿真完成:")
    print(f"  - #Assigned Tasks: {len(all_assignments)}")
    print(f"  - Collaboration Unfairness (Uρ): {u_rho:.4f}")
    print(f"  - CPU Time: {cpu_time:.4f}s")
    pred_mae = None
    pred_rmse = None
    if prediction_abs_errors:
        pred_mae = float(np.mean(prediction_abs_errors))
        pred_rmse = float(np.sqrt(np.mean(prediction_sq_errors)))
        print(f"  - Prediction MAE: {pred_mae:.4f}")
        print(f"  - Prediction RMSE: {pred_rmse:.4f}")

    metrics = {
        'assigned_tasks': len(all_assignments),
        'rho': rho,
        'u_rho': u_rho,
        'cpu_time': cpu_time,
        'pred_mae': pred_mae,
        'pred_rmse': pred_rmse
    }

    return all_assignments, all_details, metrics


if __name__ == "__main__":
    # ========================================================
    # 毕业论文核心实验：多模型调度效果在线横向对比
    # ========================================================

    # 1. 运行传统的贪心基线算法
    greedy_assignments, greedy_details, greedy_metrics = run_online_simulation_with_center_pickup(
        algo_name='greedy',
        test_date=DEFAULT_TEST_DATE,
        test_start_hour=DEFAULT_START_HOUR,
        test_end_hour=DEFAULT_END_HOUR,
        time_slot_minutes=DEFAULT_TIME_SLOT_MINUTES
    )

    # 2. 运行 ICDE 论文复现算法 (博弈论多中心协同)
    imtao_assignments, imtao_details, imtao_metrics = run_online_simulation_with_center_pickup(
        algo_name='imtao',
        test_date=DEFAULT_TEST_DATE,
        test_start_hour=DEFAULT_START_HOUR,
        test_end_hour=DEFAULT_END_HOUR,
        time_slot_minutes=DEFAULT_TIME_SLOT_MINUTES
    )

    predictive_assignments, predictive_details, predictive_metrics = run_online_simulation_with_center_pickup(
        algo_name='predictive_mctgnet',
        test_date=DEFAULT_TEST_DATE,
        test_start_hour=DEFAULT_START_HOUR,
        test_end_hour=DEFAULT_END_HOUR,
        time_slot_minutes=DEFAULT_TIME_SLOT_MINUTES
    )
    game_predictive_assignments, game_predictive_details, game_predictive_metrics = run_online_simulation_with_center_pickup(
        algo_name='predictive_game_mctgnet',
        test_date=DEFAULT_TEST_DATE,
        test_start_hour=DEFAULT_START_HOUR,
        test_end_hour=DEFAULT_END_HOUR,
        time_slot_minutes=DEFAULT_TIME_SLOT_MINUTES
    )

    # 3. 打印与论文一致的评价指标
    print("\n\n" + "=" * 110)
    print("论文指标对齐：多中心调度算法横向对比")
    print("=" * 110)
    def _fmt_optional_metric(value):
        return f"{value:.4f}" if value is not None else "-"

    print(
        f"{'指标':<25} | {'Greedy':<12} | {'IMTAO':<12} | "
        f"{'Predictive-MCTGNet':<20} | {'Game-MCTGNet':<16}"
    )
    print("-" * 110)
    print(
        f"{'#Assigned Tasks':<25} | {greedy_metrics['assigned_tasks']:<12} | "
        f"{imtao_metrics['assigned_tasks']:<12} | {predictive_metrics['assigned_tasks']:<20} | "
        f"{game_predictive_metrics['assigned_tasks']:<16}"
    )
    print(
        f"{'Collaboration Unfairness':<25} | {greedy_metrics['u_rho']:<12.4f} | "
        f"{imtao_metrics['u_rho']:<12.4f} | {predictive_metrics['u_rho']:<20.4f} | "
        f"{game_predictive_metrics['u_rho']:<16.4f}"
    )
    print(
        f"{'CPU Time (s)':<25} | {greedy_metrics['cpu_time']:<12.4f} | "
        f"{imtao_metrics['cpu_time']:<12.4f} | {predictive_metrics['cpu_time']:<20.4f} | "
        f"{game_predictive_metrics['cpu_time']:<16.4f}"
    )
    print(
        f"{'Prediction MAE':<25} | {_fmt_optional_metric(greedy_metrics.get('pred_mae')):<12} | "
        f"{_fmt_optional_metric(imtao_metrics.get('pred_mae')):<12} | "
        f"{_fmt_optional_metric(predictive_metrics.get('pred_mae')):<20} | "
        f"{_fmt_optional_metric(game_predictive_metrics.get('pred_mae')):<16}"
    )
    print(
        f"{'Prediction RMSE':<25} | {_fmt_optional_metric(greedy_metrics.get('pred_rmse')):<12} | "
        f"{_fmt_optional_metric(imtao_metrics.get('pred_rmse')):<12} | "
        f"{_fmt_optional_metric(predictive_metrics.get('pred_rmse')):<20} | "
        f"{_fmt_optional_metric(game_predictive_metrics.get('pred_rmse')):<16}"
    )
    print("=" * 110)

