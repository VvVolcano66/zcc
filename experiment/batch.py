import time
import copy
import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import KDTree

import config
from algorithm.Greedy import greedy_assignment_with_center_pickup
from algorithm.EnhancedIMTAOAssignment import enhanced_imtao_assignment_with_center_pickup
from algorithm.PredictiveDispatch import predict_next_slot_demand, predispatch_workers_for_next_slot
from algorithm.GameTheoreticPredictiveDispatch import game_theoretic_predispatch_workers
from algorithm.UncertaintyAwareBilateralDispatch import (
    UncertaintyAwareBilateralState,
    uncertainty_aware_bilateral_predispatch_workers,
)
from algorithm.RLRetentionGameDispatch import (
    PlatformTaskFirstRLState,
    RLRetentionBilateralState,
    offline_warm_start_retention_policy,
    rl_retention_bilateral_predispatch_workers,
    sample_platform_task_first_control,
    update_platform_task_first_state,
    update_rl_retention_bilateral_state,
)
from predicate.MCTGNetDispatchPredictor import MCTGNetDispatchPredictor
from predicate.CenterPatternLSTMDispatchPredictor import CenterPatternLSTMDispatchPredictor
from tool.TaskWorkerToMap import WorkerSimulator
from tool.data_loader import get_real_road_network
from tool.map_algorithms import run_kmeans_baseline, run_rcc_algorithm, find_region_centers

from algorithm.IMTAO import (
    Center as IMTAOCenter,
    IMTAO_Framework,
    IMTAO_MODE_BDC,
    IMTAO_MODE_DC,
    IMTAO_MODE_WO_C,
    IMTAO_SELECT_LOWEST_RHO,
    IMTAO_SELECT_RANDOM,
    Task as IMTAOTask,
    Worker as IMTAOWorker,
)

DEFAULT_TEST_DATE = getattr(config, 'EXPERIMENT_TEST_DATE', '2016-10-31')
DEFAULT_START_HOUR = int(getattr(config, 'EXPERIMENT_START_HOUR', 7))
DEFAULT_END_HOUR = int(getattr(config, 'EXPERIMENT_END_HOUR', 9))
DEFAULT_TIME_SLOT_MINUTES = int(getattr(config, 'EXPERIMENT_TIME_SLOT_MINUTES', 15))
FIXED_INIT_PREP_MINUTES = int(getattr(config, 'WORKER_INIT_PREP_MINUTES', 2))
DEFAULT_COMPARE_SLOT_COUNT = int(getattr(config, 'EXPERIMENT_COMPARE_SLOT_COUNT', 8))
DEFAULT_LOOKAHEAD_SLOTS = int(getattr(config, 'PREDISPATCH_LOOKAHEAD_SLOTS', 3))
DEFAULT_LOOKAHEAD_DECAY = float(getattr(config, 'PREDISPATCH_LOOKAHEAD_DECAY', 0.7))
DEFAULT_LOOKAHEAD_PEAK_WEIGHT = float(getattr(config, 'PREDISPATCH_PEAK_WEIGHT', 0.3))
DEFAULT_PREDISPATCH_DEMAND_MARGIN = float(getattr(config, 'PREDISPATCH_DEMAND_MARGIN', 0.0))
DEFAULT_WORKER_LIMIT = getattr(config, 'EXPERIMENT_WORKER_LIMIT', None)
DEFAULT_WORKER_SAMPLING_MODE = getattr(config, 'EXPERIMENT_WORKER_SAMPLING_MODE', 'snapshot_global')
DEFAULT_WORKER_SAMPLE_SEED = int(getattr(config, 'EXPERIMENT_WORKER_SAMPLE_SEED', 42))

_SIMULATION_CONTEXT_CACHE = {}
_MCTG_PREDICTOR_CACHE = {}
_CENTER_LSTM_PREDICTOR_CACHE = {}
PREDICTIVE_UABG_ALGOS = {
    'predictive_uabg_mctgnet',
    'predictive_uncertainty_game_mctgnet',
    'predictive_bilateral_game_mctgnet',
}
PREDICTIVE_RBG_ALGOS = {
    'predictive_rl_game_mctgnet',
    'predictive_rbg_mctgnet',
    'rl_game_mctgnet',
}
PREDICTIVE_PLATFORM_RL_ALGOS = {
    'predictive_platform_rl_mctgnet',
    'platform_rl_mctgnet',
    'predictive_taskfirst_platform_mctgnet',
}
NO_PRED_GAME_ALGOS = {'game_only_dispatch', 'game_only', 'no_pred_game'}
INTRABATCH_ONLINE_ALGOS = {
    'greedy',
    'predictive_greedy',
    'imtao',
    'imtao_seq_bdc',
    'seq_bdc',
    'imtao_seq_rbdc',
    'seq_rbdc',
    'imtao_rbdc',
    'imtao_seq_dc',
    'seq_dc',
    'imtao_dc',
    'imtao_seq_wo_c',
    'seq_wo_c',
    'imtao_wo_c',
    'imtao_no_collab',
}
ROUTE_ILP_ASSIGNMENT_ALGOS = {
    'predictive_mctgnet',
    'predictive_game_mctgnet',
    'predictive_bstgcnet',
    'predictive_center_lstm',
    'predictive_game_center_lstm',
    *PREDICTIVE_UABG_ALGOS,
    *PREDICTIVE_RBG_ALGOS,
    *PREDICTIVE_PLATFORM_RL_ALGOS,
    *NO_PRED_GAME_ALGOS,
}


def _should_force_mctgnet_cpu() -> bool:
    env_value = os.environ.get("MCTGNET_DISPATCH_FORCE_CPU")
    if env_value is not None:
        return env_value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(getattr(config, 'MCTGNET_DISPATCH_FORCE_CPU', False))


def _compute_map_bbox(center_point, dist_m):
    center_lat, center_lon = center_point
    lat_delta = float(dist_m) / 111320.0
    lon_scale = max(np.cos(np.radians(center_lat)), 1e-6)
    lon_delta = float(dist_m) / (111320.0 * lon_scale)
    return (
        center_lon - lon_delta,
        center_lon + lon_delta,
        center_lat - lat_delta,
        center_lat + lat_delta,
    )


def _filter_df_by_map_bbox(df, lon_col: str, lat_col: str):
    if df.empty:
        return df, 0
    min_lon, max_lon, min_lat, max_lat = _compute_map_bbox(config.CHENGDU_CENTER, config.DOWNLOAD_DIST)
    mask = (
        df[lon_col].between(min_lon, max_lon)
        & df[lat_col].between(min_lat, max_lat)
    )
    return df[mask].copy(), int((~mask).sum())


def build_prediction_date_split(test_date: str, data_dir: str):
    test_ts = pd.Timestamp(test_date).normalize()
    # 显式定义学习数据的起始日期
    train_start_ts = pd.Timestamp('2016-10-01').normalize()

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

        # 核心修改：确保日期在 2016-10-01 之后，且在测试日期之前
        if train_start_ts <= date_ts < test_ts:
            available_dates.append(date_str)

    if len(available_dates) <= val_days:
        raise ValueError(
            f"从 {train_start_ts.date()} 到 {test_date} 之间的历史数据不足以构建训练/验证集: "
            f"仅找到 {len(available_dates)} 天数据，但需要预留 {val_days} 天进行验证。"
        )

    # 按照原逻辑切分：最后 val_days 天为验证集，其余为训练集
    train_dates = available_dates[:-val_days]
    val_dates = available_dates[-val_days:]
    return train_dates, val_dates


def _build_rbg_offline_historical_samples(
        test_date: str,
        test_start_hour: int,
        test_end_hour: int,
        time_slot_minutes: int,
        coords,
        nodes,
        rcc_partition,
        centers,
        worker_sim,
):
    region_ids = sorted(centers.keys())
    tree = KDTree(coords)

    available_workers = {rid: 0 for rid in region_ids}
    for wid, rid in worker_sim.worker_center_map.items():
        if rid in available_workers and wid in worker_sim.worker_positions:
            available_workers[rid] += 1

    max_tasks_per_worker = int(getattr(config, 'MAX_TASKS_PER_WORKER', 4))

    if bool(getattr(config, 'RBG_OFFLINE_USE_SAME_DAY_PREFIX', True)):
        prefix_start_hour = int(getattr(config, 'RBG_OFFLINE_SAME_DAY_START_HOUR', 7))
        prefix_start_hour = max(0, min(prefix_start_hour, test_start_hour))
        if prefix_start_hour < test_start_hour:
            same_day_file = os.path.join(config.TASK_DATA_DIR, f"tasks_{test_date}.csv")
            if os.path.exists(same_day_file):
                df = pd.read_csv(same_day_file)
                if not df.empty:
                    df, _ = _filter_df_by_map_bbox(df, 'first_lon', 'first_lat')
                    if not df.empty:
                        df['first_time'] = pd.to_datetime(df['first_time'])
                        df['hour'] = df['first_time'].dt.hour
                        df = df[(df['hour'] >= prefix_start_hour) & (df['hour'] < test_start_hour)].copy()
                        if not df.empty:
                            task_coords = df[['first_lon', 'first_lat']].values
                            _, idxs = tree.query(task_coords)
                            df['nearest_node'] = [nodes[i] for i in idxs]
                            df = df[df['nearest_node'].isin(rcc_partition)].copy()
                            if not df.empty:
                                df['region_id'] = df['nearest_node'].map(rcc_partition)
                                num_prefix_slots = max(
                                    1,
                                    ((test_start_hour - prefix_start_hour) * 60) // int(time_slot_minutes)
                                )
                                df['slot_id'] = (
                                    ((df['first_time'].dt.hour * 60 + df['first_time'].dt.minute) - prefix_start_hour * 60)
                                    // int(time_slot_minutes)
                                ).astype(int)
                                df = df[(df['slot_id'] >= 0) & (df['slot_id'] < num_prefix_slots)].copy()
                                if not df.empty:
                                    grouped = df.groupby(['slot_id', 'region_id']).size()
                                    rolling_history = {rid: [] for rid in region_ids}
                                    backlog_counts = {rid: 0 for rid in region_ids}
                                    same_day_samples = []

                                    for slot_id in range(num_prefix_slots):
                                        actual_counts = {
                                            rid: int(grouped.get((slot_id, rid), 0))
                                            for rid in region_ids
                                        }
                                        sigma_map = {}
                                        q90_map = {}
                                        burst_map = {}
                                        for rid in region_ids:
                                            history = rolling_history[rid]
                                            if history:
                                                hist_arr = np.asarray(history, dtype=np.float32)
                                                hist_mean = float(np.mean(hist_arr))
                                                hist_std = float(np.std(hist_arr)) if hist_arr.size > 1 else 0.0
                                                sigma_map[rid] = hist_std
                                                q90_map[rid] = float(np.quantile(hist_arr, 0.90))
                                                burst_map[rid] = 1.0 if actual_counts[rid] > hist_mean + hist_std else 0.0
                                            else:
                                                sigma_map[rid] = 0.0
                                                q90_map[rid] = float(actual_counts[rid])
                                                burst_map[rid] = 0.0

                                        same_day_samples.append(
                                            {
                                                'slot_id': slot_id,
                                                'actual_counts': actual_counts,
                                                'available_workers': dict(available_workers),
                                                'backlog_counts': dict(backlog_counts),
                                                'sigma_map': sigma_map,
                                                'q90_map': q90_map,
                                                'burst_map': burst_map,
                                                'source': 'same_day_prefix',
                                            }
                                        )

                                        for rid in region_ids:
                                            rolling_history[rid].append(actual_counts[rid])
                                            capacity = available_workers[rid] * max_tasks_per_worker
                                            backlog_counts[rid] = max(
                                                0,
                                                backlog_counts[rid] + actual_counts[rid] - capacity
                                            )

                                    if same_day_samples:
                                        return same_day_samples

    if not bool(getattr(config, 'RBG_OFFLINE_FALLBACK_TO_HISTORY_DAYS', True)):
        return []

    train_dates, val_dates = build_prediction_date_split(test_date, config.TASK_DATA_DIR)
    max_days = max(1, int(getattr(config, 'RBG_OFFLINE_MAX_DAYS', 21)))
    history_dates = (train_dates + val_dates)[-max_days:]
    if not history_dates:
        return []

    num_slots = max(1, ((test_end_hour - test_start_hour) * 60) // int(time_slot_minutes))

    daily_slot_counts = []
    slot_history = {
        slot_id: {rid: [] for rid in region_ids}
        for slot_id in range(num_slots)
    }

    for date_str in history_dates:
        file_path = os.path.join(config.TASK_DATA_DIR, f"tasks_{date_str}.csv")
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        if df.empty:
            continue
        df, _ = _filter_df_by_map_bbox(df, 'first_lon', 'first_lat')
        if df.empty:
            continue

        df['first_time'] = pd.to_datetime(df['first_time'])
        df['hour'] = df['first_time'].dt.hour
        df = df[(df['hour'] >= test_start_hour) & (df['hour'] < test_end_hour)].copy()
        if df.empty:
            continue

        task_coords = df[['first_lon', 'first_lat']].values
        _, idxs = tree.query(task_coords)
        df['nearest_node'] = [nodes[i] for i in idxs]
        df = df[df['nearest_node'].isin(rcc_partition)].copy()
        if df.empty:
            continue

        df['region_id'] = df['nearest_node'].map(rcc_partition)
        df['slot_id'] = (
            ((df['first_time'].dt.hour * 60 + df['first_time'].dt.minute) - test_start_hour * 60)
            // int(time_slot_minutes)
        ).clip(lower=0, upper=num_slots - 1)

        slot_counts = {
            slot_id: {rid: 0 for rid in region_ids}
            for slot_id in range(num_slots)
        }
        grouped = df.groupby(['slot_id', 'region_id']).size()
        for (slot_id, rid), count in grouped.items():
            slot_id = int(slot_id)
            rid = int(rid)
            slot_counts[slot_id][rid] = int(count)

        for slot_id in range(num_slots):
            for rid in region_ids:
                slot_history[slot_id][rid].append(int(slot_counts[slot_id][rid]))

        daily_slot_counts.append((date_str, slot_counts))

    if not daily_slot_counts:
        return []

    slot_stats = {
        slot_id: {
            rid: {
                'sigma': float(np.std(slot_history[slot_id][rid])) if len(slot_history[slot_id][rid]) > 1 else 0.0,
                'q90': float(np.quantile(slot_history[slot_id][rid], 0.90)) if slot_history[slot_id][rid] else 0.0,
                'burst': float(np.mean(
                    np.asarray(slot_history[slot_id][rid], dtype=np.float32)
                    > (np.mean(slot_history[slot_id][rid]) + np.std(slot_history[slot_id][rid]))
                )) if slot_history[slot_id][rid] else 0.0,
            }
            for rid in region_ids
        }
        for slot_id in range(num_slots)
    }

    samples = []
    for _, slot_counts in daily_slot_counts:
        backlog_counts = {rid: 0 for rid in region_ids}
        for slot_id in range(num_slots):
            actual_counts = {rid: int(slot_counts[slot_id][rid]) for rid in region_ids}
            samples.append(
                {
                    'slot_id': slot_id,
                    'actual_counts': actual_counts,
                    'available_workers': dict(available_workers),
                    'backlog_counts': dict(backlog_counts),
                    'sigma_map': {rid: slot_stats[slot_id][rid]['sigma'] for rid in region_ids},
                    'q90_map': {rid: slot_stats[slot_id][rid]['q90'] for rid in region_ids},
                    'burst_map': {rid: slot_stats[slot_id][rid]['burst'] for rid in region_ids},
                    'source': 'history_days',
                }
            )
            for rid in region_ids:
                capacity = available_workers[rid] * max_tasks_per_worker
                backlog_counts[rid] = max(0, backlog_counts[rid] + actual_counts[rid] - capacity)

    return samples


def _simulate_same_day_prefix_rbg_learning(
        state: RLRetentionBilateralState,
        test_date: str,
        test_start_hour: int,
        time_slot_minutes: int,
        G,
        coords,
        nodes,
        rcc_partition,
        centers,
        reference_worker_sim,
):
    if not bool(getattr(config, 'RBG_OFFLINE_PREFIX_SIMULATION', True)):
        return None

    prefix_start_hour = int(getattr(config, 'RBG_OFFLINE_SAME_DAY_START_HOUR', 7))
    prefix_start_hour = max(0, min(prefix_start_hour, test_start_hour))
    if prefix_start_hour >= test_start_hour:
        return None

    same_day_file = os.path.join(config.TASK_DATA_DIR, f"tasks_{test_date}.csv")
    if not os.path.exists(same_day_file):
        return None

    df = pd.read_csv(same_day_file)
    if df.empty:
        return None
    df, _ = _filter_df_by_map_bbox(df, 'first_lon', 'first_lat')
    if df.empty:
        return None

    df['first_time'] = pd.to_datetime(df['first_time'])
    df['seconds_of_day'] = (
        df['first_time'].dt.hour * 3600
        + df['first_time'].dt.minute * 60
        + df['first_time'].dt.second
    )
    prefix_start_seconds = prefix_start_hour * 3600
    prefix_end_seconds = test_start_hour * 3600
    df = df[(df['seconds_of_day'] >= prefix_start_seconds) & (df['seconds_of_day'] < prefix_end_seconds)].copy()
    if df.empty:
        return None

    tree = KDTree(coords)
    task_coords = df[['first_lon', 'first_lat']].values
    _, idxs = tree.query(task_coords)
    df['nearest_node'] = [nodes[i] for i in idxs]
    df = df[df['nearest_node'].isin(rcc_partition)].copy()
    if df.empty:
        return None
    df['region_id'] = df['nearest_node'].map(rcc_partition)

    prefix_worker_count = len(reference_worker_sim.worker_positions)
    prefix_worker_sim = WorkerSimulator(G, config)
    prefix_worker_sim.initialize_from_real_data(
        date=test_date,
        test_start_hour=prefix_start_hour,
        prep_minutes=FIXED_INIT_PREP_MINUTES,
        coords=coords,
        nodes=nodes,
        partition=rcc_partition,
        centers=centers,
        max_workers=prefix_worker_count,
        sampling_mode=DEFAULT_WORKER_SAMPLING_MODE,
        random_seed=DEFAULT_WORKER_SAMPLE_SEED,
    )

    unassigned_tasks_pool = {rid: [] for rid in centers.keys()}
    rolling_history = {rid: [] for rid in centers.keys()}
    backlog_counts = {rid: 0 for rid in centers.keys()}
    slot_count = max(1, ((test_start_hour - prefix_start_hour) * 60) // int(time_slot_minutes))

    for local_slot_idx in range(slot_count):
        slot_start_seconds = prefix_start_seconds + local_slot_idx * time_slot_minutes * 60
        slot_end_seconds = slot_start_seconds + time_slot_minutes * 60
        prefix_worker_sim.advance_workers_to_time(centers, slot_start_seconds)

        slot_mask = (df['seconds_of_day'] >= slot_start_seconds) & (df['seconds_of_day'] < slot_end_seconds)
        slot_df = df[slot_mask]
        slot_actual_counts = {rid: 0 for rid in centers.keys()}
        if not slot_df.empty:
            for _, row in slot_df.iterrows():
                rid = int(row['region_id'])
                task = (
                    row['nearest_node'],
                    row['task_id'],
                    config.TASK_BASE_REWARD,
                    row['seconds_of_day'] + config.TASK_EXPIRE_MINUTES * 60,
                    row['seconds_of_day'],
                )
                unassigned_tasks_pool[rid].append(task)
                slot_actual_counts[rid] += 1

        predicted_distribution = {}
        for rid in centers.keys():
            history = rolling_history[rid]
            if history:
                hist_arr = np.asarray(history, dtype=np.float32)
                mu = float(slot_actual_counts[rid])
                sigma = float(np.std(hist_arr)) if hist_arr.size > 1 else 0.0
                q90 = float(np.quantile(hist_arr, 0.90))
                hist_mean = float(np.mean(hist_arr))
                burst = 1.0 if slot_actual_counts[rid] > hist_mean + sigma else 0.0
            else:
                mu = float(slot_actual_counts[rid])
                sigma = 0.0
                q90 = mu
                burst = 0.0
            predicted_distribution[rid] = {
                'mu': mu,
                'sigma': sigma,
                'q90': max(mu, q90),
                'burst_prob': burst,
            }

        predispatch_result = rl_retention_bilateral_predispatch_workers(
            G=G,
            worker_sim=prefix_worker_sim,
            centers=centers,
            predicted_demand=slot_actual_counts,
            state=state,
            slot_idx=local_slot_idx,
            next_slot_start_seconds=slot_start_seconds,
            predicted_distribution=predicted_distribution,
            max_tasks_per_worker=getattr(config, 'MAX_TASKS_PER_WORKER', 4),
            backlog_counts=backlog_counts,
            backlog_weight=getattr(config, 'UABG_BACKLOG_WEIGHT', 1.0),
            uncertainty_weight=getattr(config, 'UABG_UNCERTAINTY_WEIGHT', 0.45),
            quantile_weight=getattr(config, 'UABG_QUANTILE_WEIGHT', 0.55),
            burst_weight=getattr(config, 'UABG_BURST_WEIGHT', 1.2),
            calibration_bias_weight=getattr(config, 'RBG_PREDICTION_BIAS_WEIGHT', 0.60),
            calibration_shrink_weight=getattr(config, 'RBG_PREDICTION_SHRINK_WEIGHT', 0.55),
            calibration_sigma_boost=getattr(config, 'RBG_PREDICTION_SIGMA_BOOST', 0.75),
            calibration_min_scale=getattr(config, 'RBG_PREDICTION_MIN_SCALE', 0.55),
            platform_task_weight=getattr(config, 'RBG_PLATFORM_TASK_WEIGHT', 0.30),
            platform_gap_weight=getattr(config, 'RBG_PLATFORM_GAP_WEIGHT', 0.55),
            platform_release_credit_weight=getattr(config, 'RBG_PLATFORM_RELEASE_CREDIT_WEIGHT', 0.35),
            center_local_task_weight=getattr(config, 'RBG_CENTER_LOCAL_TASK_WEIGHT', 1.0),
            worker_completion_bonus=getattr(config, 'RBG_WORKER_COMPLETION_BONUS', 0.20),
            worker_distance_penalty=getattr(config, 'RBG_WORKER_DISTANCE_PENALTY', 0.0),
            same_worker_chain_bonus=getattr(config, 'RBG_WORKER_CHAIN_BONUS', 0.08),
            min_buffer_workers=getattr(config, 'UABG_MIN_BUFFER_WORKERS', 1),
            reserve_ratio=getattr(config, 'UABG_RESERVE_RATIO', 0.1),
            bid_shortage_weight=getattr(config, 'UABG_BID_SHORTAGE_WEIGHT', 0.9),
            bid_backlog_weight=getattr(config, 'UABG_BID_BACKLOG_WEIGHT', 0.45),
            bid_debt_weight=getattr(config, 'UABG_BID_DEBT_WEIGHT', 0.85),
            bid_burst_weight=getattr(config, 'UABG_BID_BURST_WEIGHT', 0.6),
            ask_shortage_weight=getattr(config, 'UABG_ASK_SHORTAGE_WEIGHT', 0.85),
            ask_uncertainty_weight=getattr(config, 'UABG_ASK_UNCERTAINTY_WEIGHT', 0.65),
            dispatch_phase='slot_start',
            hoard_discount_weight=getattr(config, 'RBG_HOARD_DISCOUNT_WEIGHT', 0.40),
            move_cost_weight=getattr(config, 'RBG_MOVE_COST_WEIGHT', 0.02),
            distance_penalty=getattr(config, 'UABG_DISTANCE_PENALTY', 0.004),
            candidate_k=getattr(config, 'UABG_CANDIDATE_K', 16),
            edge_epsilon=getattr(config, 'UABG_EDGE_EPSILON', 0.05),
            record_transition=True,
        )

        workers_per_center = {}
        for rid in centers.keys():
            workers = prefix_worker_sim.get_available_workers_with_center_info(rid, current_time=slot_start_seconds)
            workers_per_center[rid] = [(w[0], w[1], w[2], w[3], centers[rid]) for w in workers]
        slot_total_tasks_per_center = {rid: len(unassigned_tasks_pool[rid]) for rid in centers.keys()}
        tasks_per_center = _build_intrabatch_online_tasks(
            unassigned_tasks_pool=unassigned_tasks_pool,
            current_time=slot_start_seconds,
        )
        slot_assignments, _, slot_details = _run_assignment_for_window(
            algo_name='predictive_rl_game_mctgnet',
            G=G,
            config=config,
            centers=centers,
            rcc_partition=rcc_partition,
            workers_per_center=workers_per_center,
            tasks_per_center=tasks_per_center,
            slot_start_seconds=slot_start_seconds,
            slot_end_seconds=slot_end_seconds,
            stackelberg_control=predispatch_result.get('stackelberg_control', {}),
        )
        _apply_assignment_results_to_workers(G, prefix_worker_sim, slot_details)
        slot_assigned_tasks_per_center = {rid: 0 for rid in centers.keys()}
        for detail in slot_details:
            slot_assigned_tasks_per_center[detail['region_id']] += 1

        assigned_task_ids = {k[1] for k in slot_assignments.keys()}
        for rid in centers.keys():
            new_pool = []
            for t in unassigned_tasks_pool[rid]:
                if t[1] in assigned_task_ids:
                    continue
                if slot_end_seconds >= t[3]:
                    continue
                new_pool.append(t)
            unassigned_tasks_pool[rid] = new_pool
            backlog_counts[rid] = len(new_pool)
            rolling_history[rid].append(slot_actual_counts[rid])

        state.record_prediction_feedback(
            predicted_region_demand=slot_actual_counts,
            actual_region_demand=slot_actual_counts,
        )
        update_rl_retention_bilateral_state(
            state=state,
            transitions=predispatch_result.get('transitions', {}),
            assigned_tasks_by_region=slot_assigned_tasks_per_center,
            total_tasks_by_region=slot_total_tasks_per_center,
            hoard_penalty_by_region=predispatch_result.get('hoard_penalty', {}),
            move_cost_by_region=predispatch_result.get('move_cost_by_region', {}),
            moves=predispatch_result.get('moves', []),
            hoard_penalty_weight=float(getattr(config, 'RBG_REWARD_HOARD_WEIGHT', 0.02)),
            move_cost_weight=float(getattr(config, 'RBG_REWARD_MOVE_WEIGHT', 0.08)),
            unfairness_weight=float(getattr(config, 'RBG_REWARD_UNFAIRNESS_WEIGHT', 1.0)),
        )

    return {
        'source': 'same_day_prefix_simulation',
        'slot_count': slot_count,
        'worker_count': prefix_worker_count,
    }


def _offline_warm_start_rbg_state(
        state: RLRetentionBilateralState,
        test_date: str,
        test_start_hour: int,
        test_end_hour: int,
        time_slot_minutes: int,
        G,
        coords,
        nodes,
        rcc_partition,
        centers,
        worker_sim,
):
    if not bool(getattr(config, 'RBG_OFFLINE_WARM_START', True)):
        return None

    prefix_sim_stats = _simulate_same_day_prefix_rbg_learning(
        state=state,
        test_date=test_date,
        test_start_hour=test_start_hour,
        time_slot_minutes=time_slot_minutes,
        G=G,
        coords=coords,
        nodes=nodes,
        rcc_partition=rcc_partition,
        centers=centers,
        reference_worker_sim=worker_sim,
    )
    if prefix_sim_stats is not None:
        print(
            f"   - RBG prefix simulation warm-start ready: slots={prefix_sim_stats['slot_count']}, "
            f"workers={prefix_sim_stats['worker_count']}, source={prefix_sim_stats['source']}"
        )
        return prefix_sim_stats

    historical_samples = _build_rbg_offline_historical_samples(
        test_date=test_date,
        test_start_hour=test_start_hour,
        test_end_hour=test_end_hour,
        time_slot_minutes=time_slot_minutes,
        coords=coords,
        nodes=nodes,
        rcc_partition=rcc_partition,
        centers=centers,
        worker_sim=worker_sim,
    )
    if not historical_samples:
        print("   - RBG offline warm-start skipped: no historical samples")
        return None

    stats = offline_warm_start_retention_policy(
        state=state,
        historical_samples=historical_samples,
        max_tasks_per_worker=int(getattr(config, 'MAX_TASKS_PER_WORKER', 4)),
        min_buffer_workers=int(getattr(config, 'UABG_MIN_BUFFER_WORKERS', 1)),
        reserve_ratio=float(getattr(config, 'UABG_RESERVE_RATIO', 0.10)),
        backlog_weight=float(getattr(config, 'UABG_BACKLOG_WEIGHT', 1.0)),
        uncertainty_weight=float(getattr(config, 'UABG_UNCERTAINTY_WEIGHT', 0.45)),
        quantile_weight=float(getattr(config, 'UABG_QUANTILE_WEIGHT', 0.55)),
        burst_weight=float(getattr(config, 'UABG_BURST_WEIGHT', 1.2)),
        epochs=int(getattr(config, 'RBG_OFFLINE_EPOCHS', 3)),
    )
    print(
        f"   - RBG offline warm-start ready: samples={stats['sample_count']}, "
        f"epochs={stats['epochs']}, source={historical_samples[0].get('source', 'unknown')}"
    )
    return stats

def _build_simulation_context(
        test_date: str,
        test_start_hour: int,
        test_end_hour: int,
        time_slot_minutes: int,
        slots_to_run: int
):
    compare_end_seconds = test_start_hour * 3600 + slots_to_run * time_slot_minutes * 60
    cache_key = (
        test_date,
        test_start_hour,
        test_end_hour,
        time_slot_minutes,
        slots_to_run,
        config.NUM_ZONES,
        config.CHENGDU_CENTER,
        config.DOWNLOAD_DIST,
        FIXED_INIT_PREP_MINUTES
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
    task_file = os.path.join(config.TASK_DATA_DIR, f"tasks_{test_date}.csv")
    if os.path.exists(task_file):
        df_tasks = pd.read_csv(task_file)
        df_tasks, filtered_out_count = _filter_df_by_map_bbox(df_tasks, 'first_lon', 'first_lat')
        if filtered_out_count > 0:
            print(f"   - Spatially filtered out {filtered_out_count} tasks outside current map boundary")
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
                    (df_tasks['seconds_of_day'] < compare_end_seconds)
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
        prep_minutes=FIXED_INIT_PREP_MINUTES,
        coords=coords,
        nodes=nodes,
        partition=rcc_partition,
        centers=centers,
        max_workers=DEFAULT_WORKER_LIMIT,
        sampling_mode=DEFAULT_WORKER_SAMPLING_MODE,
        random_seed=DEFAULT_WORKER_SAMPLE_SEED
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
        config.DOWNLOAD_DIST,
        config.NUM_ZONES,
        getattr(config, 'DISPATCH_PRED_SEQ_LEN', 4),
        getattr(config, 'DISPATCH_PRED_PRE_LEN', 1),
        getattr(config, 'DISPATCH_PRED_VAL_DAYS', 2),
        getattr(config, 'MCTGNET_DISPATCH_MAX_EPOCHS', 300),
        getattr(config, 'MCTGNET_DISPATCH_PATIENCE', 50),
        getattr(config, 'MCTGNET_DISPATCH_LR', 0.0005),
        getattr(config, 'MCTGNET_DISPATCH_USE_LSTM', True),
        getattr(config, 'MCTGNET_DISPATCH_LSTM_LAYERS', 1),
        getattr(config, 'MCTGNET_DISPATCH_LSTM_DROPOUT', 0.1),
        getattr(config, 'MCTGNET_DISPATCH_UQ_QUANTILE', 0.90),
        getattr(config, 'MCTGNET_DISPATCH_UQ_SLOT_BLEND', 0.65),
        getattr(config, 'MCTGNET_DISPATCH_UQ_ONLINE_ALPHA', 0.20),
        getattr(config, 'MCTGNET_DISPATCH_UQ_MIN_SIGMA_RATIO', 0.15),
        _should_force_mctgnet_cpu(),
    )
    if predictor_key in _MCTG_PREDICTOR_CACHE:
        print("   - Reusing cached MCTGNet predictor")
        return _MCTG_PREDICTOR_CACHE[predictor_key]

    dispatch_data_dir = config.TASK_DATA_DIR
    train_dates, val_dates = build_prediction_date_split(test_date, dispatch_data_dir)
    history_span_minutes = getattr(config, 'DISPATCH_PRED_SEQ_LEN', 4) * time_slot_minutes
    history_start_hour = max(0, test_start_hour - int(np.ceil(history_span_minutes / 60.0)))
    print(f"\n[阶段 4.5] 训练 MCTGNet 预测器并为 {test_date} 调度准备历史上下文...")
    print(f"   - Train Dates: {train_dates[0]} ~ {train_dates[-1]}")
    print(f"   - Val Dates:   {val_dates[0]} ~ {val_dates[-1]}")
    print(f"   - Train Days:  {len(train_dates)} | Val Days: {len(val_dates)}")
    print(f"   - Target Date: {test_date}")
    print(f"   - History Window: {history_start_hour}:00 - {test_end_hour}:00")
    force_cpu = _should_force_mctgnet_cpu()
    if force_cpu:
        print("   - MCTGNet dispatch predictor device: CPU-only (forced for this run)")

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
        log_interval=getattr(config, 'MCTGNET_DISPATCH_LOG_INTERVAL', 20),
        center_loss_weight=getattr(config, 'MCTGNET_DISPATCH_CENTER_LOSS_WEIGHT', 0.35),
        center_hotspot_alpha=getattr(config, 'MCTGNET_DISPATCH_CENTER_HOTSPOT_ALPHA', 1.5),
        center_underpredict_alpha=getattr(config, 'MCTGNET_DISPATCH_CENTER_UNDERPREDICT_ALPHA', 2.0),
        center_underpredict_power=getattr(config, 'MCTGNET_DISPATCH_CENTER_UNDERPREDICT_POWER', 1.0),
        use_lstm_branch=getattr(config, 'MCTGNET_DISPATCH_USE_LSTM', True),
        lstm_layers=getattr(config, 'MCTGNET_DISPATCH_LSTM_LAYERS', 1),
        lstm_dropout=getattr(config, 'MCTGNET_DISPATCH_LSTM_DROPOUT', 0.1),
        refit_on_all_pretarget=getattr(config, 'MCTGNET_DISPATCH_REFIT_ON_ALL_PRETARGET', True),
        use_online_adaptation=getattr(config, 'MCTGNET_DISPATCH_USE_ONLINE_ADAPTATION', True),
        online_bias_alpha=getattr(config, 'MCTGNET_DISPATCH_ONLINE_BIAS_ALPHA', 0.30),
        online_slot_bias_alpha=getattr(config, 'MCTGNET_DISPATCH_ONLINE_SLOT_BIAS_ALPHA', 0.40),
        online_scale_alpha=getattr(config, 'MCTGNET_DISPATCH_ONLINE_SCALE_ALPHA', 0.15),
        uncertainty_quantile=getattr(config, 'MCTGNET_DISPATCH_UQ_QUANTILE', 0.90),
        uncertainty_slot_blend=getattr(config, 'MCTGNET_DISPATCH_UQ_SLOT_BLEND', 0.65),
        online_uncertainty_alpha=getattr(config, 'MCTGNET_DISPATCH_UQ_ONLINE_ALPHA', 0.20),
        min_sigma_ratio=getattr(config, 'MCTGNET_DISPATCH_UQ_MIN_SIGMA_RATIO', 0.15),
        device='cpu' if force_cpu else None,
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


def _get_or_train_center_lstm_predictor(
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
        time_slot_minutes,
        getattr(config, 'CENTER_LSTM_DISPATCH_SEQ_LEN', 32),
        getattr(config, 'CENTER_LSTM_DISPATCH_PRE_LEN', 1),
        getattr(config, 'CENTER_LSTM_DISPATCH_MAX_EPOCHS', 400),
        getattr(config, 'CENTER_LSTM_DISPATCH_PATIENCE', 60),
        getattr(config, 'CENTER_LSTM_DISPATCH_LR', 0.0005),
        getattr(config, 'CENTER_LSTM_DISPATCH_HIDDEN_DIM', 128),
        getattr(config, 'CENTER_LSTM_DISPATCH_LSTM_LAYERS', 2),
        getattr(config, 'CENTER_LSTM_DISPATCH_DROPOUT', 0.15),
        getattr(config, 'CENTER_LSTM_DISPATCH_HOTSPOT_ALPHA', 2.5),
        getattr(config, 'CENTER_LSTM_DISPATCH_UNDERPREDICT_ALPHA', 3.0),
        getattr(config, 'CENTER_LSTM_DISPATCH_UNDERPREDICT_POWER', 1.0),
    )
    if predictor_key in _CENTER_LSTM_PREDICTOR_CACHE:
        print("   - Reusing cached CenterPatternLSTM predictor")
        return _CENTER_LSTM_PREDICTOR_CACHE[predictor_key]

    dispatch_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'task')
    train_dates, val_dates = build_prediction_date_split(test_date, dispatch_data_dir)
    print(f"\n[Phase 4.5] Training CenterPatternLSTM dispatch predictor for {test_date} ...")
    print(f"   - Train Dates: {train_dates[0]} ~ {train_dates[-1]}")
    print(f"   - Val Dates:   {val_dates[0]} ~ {val_dates[-1]}")
    print(f"   - Train Days:  {len(train_dates)} | Val Days: {len(val_dates)}")
    print(f"   - Target Date: {test_date}")
    print("   - Context Window: full-day center demand patterns with short-term autoregression")

    predictor = CenterPatternLSTMDispatchPredictor(
        data_dir=dispatch_data_dir,
        coords=coords,
        nodes=nodes,
        partition=rcc_partition,
        centers=centers,
        time_interval=time_slot_minutes,
        seq_len=getattr(config, 'CENTER_LSTM_DISPATCH_SEQ_LEN', 32),
        pre_len=getattr(config, 'CENTER_LSTM_DISPATCH_PRE_LEN', 1),
        max_epochs=getattr(config, 'CENTER_LSTM_DISPATCH_MAX_EPOCHS', 400),
        patience=getattr(config, 'CENTER_LSTM_DISPATCH_PATIENCE', 60),
        lr=getattr(config, 'CENTER_LSTM_DISPATCH_LR', 0.0005),
        hidden_dim=getattr(config, 'CENTER_LSTM_DISPATCH_HIDDEN_DIM', 128),
        lstm_layers=getattr(config, 'CENTER_LSTM_DISPATCH_LSTM_LAYERS', 2),
        dropout=getattr(config, 'CENTER_LSTM_DISPATCH_DROPOUT', 0.15),
        log_interval=getattr(config, 'CENTER_LSTM_DISPATCH_LOG_INTERVAL', 20),
        hotspot_alpha=getattr(config, 'CENTER_LSTM_DISPATCH_HOTSPOT_ALPHA', 2.5),
        underpredict_alpha=getattr(config, 'CENTER_LSTM_DISPATCH_UNDERPREDICT_ALPHA', 3.0),
        underpredict_power=getattr(config, 'CENTER_LSTM_DISPATCH_UNDERPREDICT_POWER', 1.0),
        refit_on_all_pretarget=getattr(config, 'CENTER_LSTM_DISPATCH_REFIT_ON_ALL_PRETARGET', True),
        use_online_adaptation=getattr(config, 'CENTER_LSTM_DISPATCH_USE_ONLINE_ADAPTATION', True),
        online_bias_alpha=getattr(config, 'CENTER_LSTM_DISPATCH_ONLINE_BIAS_ALPHA', 0.30),
        online_slot_bias_alpha=getattr(config, 'CENTER_LSTM_DISPATCH_ONLINE_SLOT_BIAS_ALPHA', 0.40),
        online_scale_alpha=getattr(config, 'CENTER_LSTM_DISPATCH_ONLINE_SCALE_ALPHA', 0.15),
    )
    predictor.fit(
        train_dates=train_dates,
        val_dates=val_dates,
        target_date=test_date,
        history_start_hour=test_start_hour,
        end_hour=test_end_hour
    )
    print("   - CenterPatternLSTM predictor ready")
    _CENTER_LSTM_PREDICTOR_CACHE[predictor_key] = predictor
    return predictor


def _build_lookahead_demand(
        predictor: MCTGNetDispatchPredictor,
        slot_timestamp: pd.Timestamp,
        time_slot_minutes: int,
        centers: dict,
        lookahead_slots: int,
        decay: float
):
    region_ids = sorted(centers.keys())
    lookahead_slots = max(1, int(lookahead_slots))
    decay = min(1.0, max(0.0, float(decay)))
    peak_weight = min(1.0, max(0.0, DEFAULT_LOOKAHEAD_PEAK_WEIGHT))
    demand_margin = max(0.0, DEFAULT_PREDISPATCH_DEMAND_MARGIN)

    forecast_trace = []
    weights = []
    for step in range(lookahead_slots):
        future_ts = slot_timestamp + pd.Timedelta(minutes=step * time_slot_minutes)
        forecast = predictor.predict_region_demand(future_ts)
        if forecast is None:
            continue
        weight = decay ** step
        forecast_trace.append((future_ts, forecast))
        weights.append(weight)

    if not forecast_trace:
        return None, []

    weight_sum = max(1e-6, float(sum(weights)))
    planning_demand = {}
    for rid in region_ids:
        weighted_mean = sum(weights[idx] * forecast_trace[idx][1].get(rid, 0) for idx in range(len(forecast_trace))) / weight_sum
        future_peak = max(item[1].get(rid, 0) for item in forecast_trace)
        blended = (1.0 - peak_weight) * weighted_mean + peak_weight * future_peak
        planning_demand[rid] = int(round(blended * (1.0 + demand_margin)))

    return planning_demand, forecast_trace


def run_imtao_for_slot(
        G,
        config,
        centers_dict,
        workers_per_center,
        tasks_per_center,
        slot_start_seconds,
        slot_end_seconds=None,
        collaboration_mode=IMTAO_MODE_BDC,
        center_selection=IMTAO_SELECT_LOWEST_RHO,
):
    """
    IMTAO 算法适配器（修复版：强制增加中心取货约束）
    """
    if slot_end_seconds is None:
        slot_end_seconds = slot_start_seconds + float(getattr(config, 'EXPERIMENT_TIME_SLOT_MINUTES', 15)) * 60

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
        travel_time_func=route_travel_time,
        slot_duration_seconds=max(0.0, slot_end_seconds - slot_start_seconds)
    )
    framework.initialize_existing_partition(center_task_map, center_worker_map)
    framework.algo3_game_theoretic_collaboration(
        repartition=False,
        collaboration_mode=collaboration_mode,
        center_selection=center_selection
    )

    slot_assignments = {}
    slot_details = []
    slot_profit = 0

    for c in framework.centers:
        center_node = centers_dict[c.id]
        for w, assigned_tasks in c.A:
            if not assigned_tasks:
                continue

            worker_node = worker_node_map[w.id]

            try:
                dist_to_center = nx.shortest_path_length(G, worker_node, center_node, weight='length')
            except nx.NetworkXNoPath:
                continue

            current_node = center_node
            current_finish_time = slot_start_seconds + dist_to_center / config.WORKER_SPEED_MS
            round_load = 0
            first_departure_pending = True

            for task in assigned_tasks:
                if round_load >= config.MAX_TASKS_PER_WORKER:
                    try:
                        return_dist = nx.shortest_path_length(G, current_node, center_node, weight='length')
                    except nx.NetworkXNoPath:
                        break

                    return_finish_time = current_finish_time + return_dist / config.WORKER_SPEED_MS
                    if return_finish_time > slot_end_seconds:
                        break

                    current_finish_time = return_finish_time
                    current_node = center_node
                    round_load = 0

                task_node = task_node_map[task.id]
                try:
                    dist_to_task = nx.shortest_path_length(G, current_node, task_node, weight='length')
                except nx.NetworkXNoPath:
                    continue

                dist_to_center_cost = dist_to_center if first_departure_pending else 0.0
                travel_cost = (dist_to_center_cost + dist_to_task) * config.TRAVEL_COST_PER_METER
                reward = task_reward_map[task.id]
                profit = reward - travel_cost
                candidate_finish_time = max(
                    current_finish_time + dist_to_task / config.WORKER_SPEED_MS,
                    slot_start_seconds + task.r
                )
                if candidate_finish_time > task_expire_map[task.id] or candidate_finish_time > slot_end_seconds:
                    break

                end_time = candidate_finish_time
                end_node = task_node
                service_finish_time = candidate_finish_time
                next_round_load = round_load + 1
                return_dist_cost = 0.0

                if next_round_load >= config.MAX_TASKS_PER_WORKER:
                    try:
                        return_dist = nx.shortest_path_length(G, task_node, center_node, weight='length')
                    except nx.NetworkXNoPath:
                        return_dist = float('inf')

                    if return_dist != float('inf'):
                        return_finish_time = candidate_finish_time + return_dist / config.WORKER_SPEED_MS
                        if return_finish_time <= slot_end_seconds:
                            return_dist_cost = return_dist
                            end_time = return_finish_time
                            end_node = center_node

                total_cost = (dist_to_center_cost + dist_to_task + return_dist_cost) * config.TRAVEL_COST_PER_METER
                profit = reward - total_cost

                slot_assignments[(w.id, task.id)] = profit
                slot_details.append({
                    'region_id': c.id,
                    'wid': w.id,
                    'task_id': task.id,
                    'dist_to_center': dist_to_center_cost,
                    'dist_to_task': dist_to_task,
                    'return_to_center_dist': return_dist_cost,
                    'task_node': end_node,
                    'service_node': task_node,
                    'reward': reward,
                    'cost': total_cost,
                    'finish_time': end_time,
                    'service_finish_time': service_finish_time,
                    'end_time': end_time,
                    'end_node': end_node,
                    'profit': profit
                })
                slot_profit += profit

                current_finish_time = end_time
                current_node = end_node
                round_load = 0 if end_node == center_node else next_round_load
                first_departure_pending = False

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


def _run_assignment_for_window(
        algo_name,
        G,
        config,
        centers,
        rcc_partition,
        workers_per_center,
        tasks_per_center,
        slot_start_seconds,
        slot_end_seconds,
        stackelberg_control=None,
):
    algo_key = algo_name.lower()
    if algo_key in ['greedy', 'predictive_greedy']:
        return greedy_assignment_with_center_pickup(
            G=G,
            config=config,
            centers=centers,
            partition=rcc_partition,
            workers_per_center=workers_per_center,
            tasks_per_center=tasks_per_center,
            slot_start_seconds=slot_start_seconds,
            slot_end_seconds=slot_end_seconds,
        )
    if algo_key in ROUTE_ILP_ASSIGNMENT_ALGOS:
        return enhanced_imtao_assignment_with_center_pickup(
            G=G,
            config=config,
            centers_dict=centers,
            workers_per_center=workers_per_center,
            tasks_per_center=tasks_per_center,
            slot_start_seconds=slot_start_seconds,
            slot_end_seconds=slot_end_seconds,
            stackelberg_control=stackelberg_control if algo_key in [*PREDICTIVE_RBG_ALGOS, *PREDICTIVE_PLATFORM_RL_ALGOS] else None,
        )
    if algo_key in ['imtao', 'imtao_seq_bdc', 'seq_bdc']:
        return run_imtao_for_slot(
            G=G,
            config=config,
            centers_dict=centers,
            workers_per_center=workers_per_center,
            tasks_per_center=tasks_per_center,
            slot_start_seconds=slot_start_seconds,
            slot_end_seconds=slot_end_seconds,
            collaboration_mode=IMTAO_MODE_BDC,
            center_selection=IMTAO_SELECT_LOWEST_RHO,
        )
    if algo_key in ['imtao_seq_rbdc', 'seq_rbdc', 'imtao_rbdc']:
        return run_imtao_for_slot(
            G=G,
            config=config,
            centers_dict=centers,
            workers_per_center=workers_per_center,
            tasks_per_center=tasks_per_center,
            slot_start_seconds=slot_start_seconds,
            slot_end_seconds=slot_end_seconds,
            collaboration_mode=IMTAO_MODE_BDC,
            center_selection=IMTAO_SELECT_RANDOM,
        )
    if algo_key in ['imtao_seq_dc', 'seq_dc', 'imtao_dc']:
        return run_imtao_for_slot(
            G=G,
            config=config,
            centers_dict=centers,
            workers_per_center=workers_per_center,
            tasks_per_center=tasks_per_center,
            slot_start_seconds=slot_start_seconds,
            slot_end_seconds=slot_end_seconds,
            collaboration_mode=IMTAO_MODE_DC,
            center_selection=IMTAO_SELECT_LOWEST_RHO,
        )
    if algo_key in ['imtao_seq_wo_c', 'seq_wo_c', 'imtao_wo_c', 'imtao_no_collab']:
        return run_imtao_for_slot(
            G=G,
            config=config,
            centers_dict=centers,
            workers_per_center=workers_per_center,
            tasks_per_center=tasks_per_center,
            slot_start_seconds=slot_start_seconds,
            slot_end_seconds=slot_end_seconds,
            collaboration_mode=IMTAO_MODE_WO_C,
            center_selection=IMTAO_SELECT_LOWEST_RHO,
        )
    raise ValueError(f"Unsupported algorithm: {algo_name}")


def _apply_assignment_results_to_workers(G, worker_sim, slot_details):
    slot_dist_to_center = 0.0
    slot_dist_to_task = 0.0
    worker_final_state = {}

    for detail in slot_details:
        wid = detail['wid']
        slot_dist_to_center += detail['dist_to_center']
        slot_dist_to_task += detail['dist_to_task']

        prev_detail = worker_final_state.get(wid)
        if prev_detail is None or detail['finish_time'] > prev_detail['finish_time']:
            worker_final_state[wid] = detail

    for wid, detail in worker_final_state.items():
        task_node = detail['task_node']
        if task_node not in G.nodes:
            continue
        task_lon = G.nodes[task_node].get('x', G.nodes[task_node].get('lon'))
        task_lat = G.nodes[task_node].get('y', G.nodes[task_node].get('lat'))
        worker_sim.update_worker_position(wid, task_node, task_lon, task_lat)
        worker_sim.set_worker_en_route_to_task(wid, detail['finish_time'])

    return slot_dist_to_center, slot_dist_to_task


def _build_microbatch_candidate_tasks(
        unassigned_tasks_pool,
        workers_per_center,
        current_time,
        slot_end_seconds,
):
    candidate_factor = float(getattr(config, 'MICROBATCH_TASK_CANDIDATE_FACTOR', 3.0))
    candidate_floor = int(getattr(config, 'MICROBATCH_TASK_CANDIDATE_FLOOR', 48))
    candidate_cap = int(getattr(config, 'MICROBATCH_TASK_CANDIDATE_CAP', 240))
    max_tasks_per_worker = int(getattr(config, 'MAX_TASKS_PER_WORKER', 4))

    tasks_per_center = {}
    for rid, pool in unassigned_tasks_pool.items():
        if not pool:
            tasks_per_center[rid] = []
            continue

        worker_count = len(workers_per_center.get(rid, []))
        if worker_count <= 0:
            tasks_per_center[rid] = []
            continue

        limit = int(round(worker_count * max_tasks_per_worker * candidate_factor))
        limit = max(candidate_floor, limit)
        limit = min(candidate_cap, limit, len(pool))

        prioritized = sorted(
            pool,
            key=lambda t: (
                max(0.0, float(t[3]) - current_time),
                0 if float(t[3]) <= slot_end_seconds else 1,
                float(t[4]),
                str(t[1]),
            )
        )
        tasks_per_center[rid] = prioritized[:limit]

    return tasks_per_center


def _build_intrabatch_online_tasks(
        unassigned_tasks_pool,
        current_time,
):
    tasks_per_center = {}
    for rid, pool in unassigned_tasks_pool.items():
        if not pool:
            tasks_per_center[rid] = []
            continue

        tasks_per_center[rid] = sorted(
            pool,
            key=lambda t: (
                max(0.0, float(t[3]) - current_time),
                float(t[4]),
                str(t[1]),
            )
        )

    return tasks_per_center


def _should_trigger_micro_redispatch(
        algo_name,
        current_time,
        slot_start_seconds,
        slot_end_seconds,
        available_workers_per_center,
        backlog_counts,
        observed_arrivals_so_far,
        predicted_total_demand,
):
    if current_time <= slot_start_seconds:
        return False

    max_tasks_per_worker = int(getattr(config, 'MAX_TASKS_PER_WORKER', 4))
    if algo_name.lower() in [*PREDICTIVE_RBG_ALGOS, *PREDICTIVE_PLATFORM_RL_ALGOS]:
        backlog_gap_threshold = int(getattr(config, 'RBG_MICROBATCH_REDISPATCH_BACKLOG_GAP_THRESHOLD', 4))
        underpredict_ratio = float(getattr(config, 'RBG_MICROBATCH_REDISPATCH_UNDERPREDICT_RATIO', 0.10))
        backlog_pressure_ratio = float(getattr(config, 'RBG_MICROBATCH_REDISPATCH_BACKLOG_PRESSURE_RATIO', 0.45))
    else:
        backlog_gap_threshold = int(getattr(config, 'MICROBATCH_REDISPATCH_BACKLOG_GAP_THRESHOLD', 8))
        underpredict_ratio = float(getattr(config, 'MICROBATCH_REDISPATCH_UNDERPREDICT_RATIO', 0.25))
        backlog_pressure_ratio = float(getattr(config, 'MICROBATCH_REDISPATCH_BACKLOG_PRESSURE_RATIO', 0.75))

    slot_duration = max(1.0, float(slot_end_seconds - slot_start_seconds))
    elapsed_ratio = min(1.0, max(0.0, (current_time - slot_start_seconds) / slot_duration))

    for rid, backlog in backlog_counts.items():
        available_workers = int(available_workers_per_center.get(rid, 0))
        capacity = max(1, available_workers * max_tasks_per_worker)
        if backlog >= backlog_gap_threshold and backlog >= capacity * backlog_pressure_ratio:
            return True

        predicted_total = 0 if predicted_total_demand is None else int(predicted_total_demand.get(rid, 0))
        if predicted_total <= 0:
            continue
        expected_cumulative = predicted_total * elapsed_ratio
        observed_cumulative = int(observed_arrivals_so_far.get(rid, 0))
        if observed_cumulative > expected_cumulative * (1.0 + underpredict_ratio):
            return True

    return False


def _run_triggered_micro_predispatch(
        algo_name,
        G,
        worker_sim,
        centers,
        current_time,
        slot_idx,
        micro_idx,
        slot_start_seconds,
        slot_end_seconds,
        current_slot_predicted_demand,
        slot_new_tasks_per_center,
        unassigned_tasks_pool,
        uncertainty_dispatch_state,
        dispatch_predictor,
        retention_game_state,
        platform_rl_state=None,
):
    backlog_counts = {rid: len(unassigned_tasks_pool[rid]) for rid in centers.keys()}
    available_workers = {
        rid: len(worker_sim.get_available_workers_with_center_info(rid, current_time=current_time))
        for rid in centers.keys()
    }

    if not _should_trigger_micro_redispatch(
        algo_name=algo_name,
        current_time=current_time,
        slot_start_seconds=slot_start_seconds,
        slot_end_seconds=slot_end_seconds,
        available_workers_per_center=available_workers,
        backlog_counts=backlog_counts,
        observed_arrivals_so_far=slot_new_tasks_per_center,
        predicted_total_demand=current_slot_predicted_demand,
    ):
        return None

    remaining_predicted = {}
    for rid in centers.keys():
        predicted_total = 0 if current_slot_predicted_demand is None else int(current_slot_predicted_demand.get(rid, 0))
        observed_so_far = int(slot_new_tasks_per_center.get(rid, 0))
        remaining_predicted[rid] = max(0, predicted_total - observed_so_far)

    if algo_name.lower() in [*PREDICTIVE_RBG_ALGOS, *PREDICTIVE_PLATFORM_RL_ALGOS]:
        platform_task_weight = getattr(config, 'RBG_PLATFORM_TASK_WEIGHT', 0.30)
        platform_gap_weight = getattr(config, 'RBG_PLATFORM_GAP_WEIGHT', 0.55)
        platform_release_credit_weight = getattr(config, 'RBG_PLATFORM_RELEASE_CREDIT_WEIGHT', 0.35)
        predicted_distribution = None
        if hasattr(dispatch_predictor, 'predict_region_distribution'):
            predicted_distribution = dispatch_predictor.predict_region_distribution(
                datetime.combine(
                    DEFAULT_TEST_DATE,
                    datetime.min.time(),
                ) + timedelta(seconds=float(current_time))
            )
        if algo_name.lower() in PREDICTIVE_PLATFORM_RL_ALGOS and platform_rl_state is not None:
            current_slot_platform_transition = sample_platform_task_first_control(
                region_ids=sorted(centers.keys()),
                predicted_demand=remaining_predicted,
                backlog_counts=backlog_counts,
                available_workers=available_workers,
                max_tasks_per_worker=getattr(config, 'MAX_TASKS_PER_WORKER', 4),
                retention_state=retention_game_state,
                platform_state=platform_rl_state,
                predicted_distribution=predicted_distribution,
                backlog_weight=getattr(config, 'UABG_BACKLOG_WEIGHT', 1.0),
                uncertainty_weight=getattr(config, 'UABG_UNCERTAINTY_WEIGHT', 0.45),
                quantile_weight=getattr(config, 'UABG_QUANTILE_WEIGHT', 0.55),
                burst_weight=getattr(config, 'UABG_BURST_WEIGHT', 1.2),
                calibration_bias_weight=getattr(config, 'RBG_PREDICTION_BIAS_WEIGHT', 0.60),
                calibration_shrink_weight=getattr(config, 'RBG_PREDICTION_SHRINK_WEIGHT', 0.55),
                calibration_sigma_boost=getattr(config, 'RBG_PREDICTION_SIGMA_BOOST', 0.75),
                calibration_min_scale=getattr(config, 'RBG_PREDICTION_MIN_SCALE', 0.55),
                base_platform_task_weight=platform_task_weight,
                base_platform_gap_weight=platform_gap_weight,
                base_platform_release_credit_weight=platform_release_credit_weight,
            )
            platform_task_weight = current_slot_platform_transition['task_weight']
            platform_gap_weight = current_slot_platform_transition['gap_weight']
            platform_release_credit_weight = current_slot_platform_transition['release_credit_weight']
        result = rl_retention_bilateral_predispatch_workers(
            G=G,
            worker_sim=worker_sim,
            centers=centers,
            predicted_demand=remaining_predicted,
            state=retention_game_state,
            slot_idx=slot_idx,
            next_slot_start_seconds=current_time,
            predicted_distribution=predicted_distribution,
            max_tasks_per_worker=getattr(config, 'MAX_TASKS_PER_WORKER', 4),
            backlog_counts=backlog_counts,
            backlog_weight=getattr(config, 'UABG_BACKLOG_WEIGHT', 1.0),
            uncertainty_weight=getattr(config, 'UABG_UNCERTAINTY_WEIGHT', 0.45),
            quantile_weight=getattr(config, 'UABG_QUANTILE_WEIGHT', 0.55),
            burst_weight=getattr(config, 'UABG_BURST_WEIGHT', 1.2),
            calibration_bias_weight=getattr(config, 'RBG_PREDICTION_BIAS_WEIGHT', 0.60),
            calibration_shrink_weight=getattr(config, 'RBG_PREDICTION_SHRINK_WEIGHT', 0.55),
            calibration_sigma_boost=getattr(config, 'RBG_PREDICTION_SIGMA_BOOST', 0.75),
            calibration_min_scale=getattr(config, 'RBG_PREDICTION_MIN_SCALE', 0.55),
            platform_task_weight=platform_task_weight,
            platform_gap_weight=platform_gap_weight,
            platform_release_credit_weight=platform_release_credit_weight,
            platform_fairness_weight=float(current_slot_platform_transition.get('fairness_weight', 0.0)) if current_slot_platform_transition else 0.0,
            platform_keep_scale=float(current_slot_platform_transition.get('keep_scale', 1.0)) if current_slot_platform_transition else 1.0,
            platform_need_scale=float(current_slot_platform_transition.get('need_scale', 1.0)) if current_slot_platform_transition else 1.0,
            center_local_task_weight=getattr(config, 'RBG_CENTER_LOCAL_TASK_WEIGHT', 1.0),
            worker_completion_bonus=getattr(config, 'RBG_WORKER_COMPLETION_BONUS', 0.20),
            worker_distance_penalty=getattr(config, 'RBG_WORKER_DISTANCE_PENALTY', 0.0),
            same_worker_chain_bonus=getattr(config, 'RBG_WORKER_CHAIN_BONUS', 0.08),
            min_buffer_workers=getattr(config, 'UABG_MIN_BUFFER_WORKERS', 1),
            reserve_ratio=getattr(config, 'UABG_RESERVE_RATIO', 0.1),
            bid_shortage_weight=getattr(config, 'UABG_BID_SHORTAGE_WEIGHT', 0.9),
            bid_backlog_weight=getattr(config, 'UABG_BID_BACKLOG_WEIGHT', 0.45),
            bid_debt_weight=getattr(config, 'UABG_BID_DEBT_WEIGHT', 0.85),
            bid_burst_weight=getattr(config, 'UABG_BID_BURST_WEIGHT', 0.6),
            ask_shortage_weight=getattr(config, 'UABG_ASK_SHORTAGE_WEIGHT', 0.85),
            ask_uncertainty_weight=getattr(config, 'UABG_ASK_UNCERTAINTY_WEIGHT', 0.65),
            hoard_discount_weight=getattr(config, 'RBG_HOARD_DISCOUNT_WEIGHT', 0.40),
            move_cost_weight=getattr(config, 'RBG_MOVE_COST_WEIGHT', 0.02),
            distance_penalty=getattr(config, 'UABG_DISTANCE_PENALTY', 0.004),
            candidate_k=getattr(config, 'UABG_CANDIDATE_K', 16),
            edge_epsilon=getattr(config, 'UABG_EDGE_EPSILON', 0.05),
            dispatch_phase='micro',
            record_transition=False,
        )
        label = 'Platform-RL-Micro' if algo_name.lower() in PREDICTIVE_PLATFORM_RL_ALGOS else 'RBG-Micro'
    elif algo_name.lower() in PREDICTIVE_UABG_ALGOS:
        predicted_distribution = None
        result = uncertainty_aware_bilateral_predispatch_workers(
            G=G,
            worker_sim=worker_sim,
            centers=centers,
            predicted_demand=remaining_predicted,
            state=uncertainty_dispatch_state,
            slot_idx=slot_idx,
            next_slot_start_seconds=current_time,
            predicted_distribution=predicted_distribution,
            max_tasks_per_worker=getattr(config, 'MAX_TASKS_PER_WORKER', 4),
            backlog_counts=backlog_counts,
            backlog_weight=getattr(config, 'UABG_BACKLOG_WEIGHT', 1.0),
            uncertainty_weight=getattr(config, 'UABG_UNCERTAINTY_WEIGHT', 0.45),
            quantile_weight=getattr(config, 'UABG_QUANTILE_WEIGHT', 0.55),
            burst_weight=getattr(config, 'UABG_BURST_WEIGHT', 1.2),
            min_buffer_workers=getattr(config, 'UABG_MIN_BUFFER_WORKERS', 1),
            reserve_ratio=getattr(config, 'UABG_RESERVE_RATIO', 0.1),
            max_rebalance_share=getattr(config, 'UABG_MAX_SHARE_PER_DONOR', 0.6),
            max_distance_km=getattr(config, 'UABG_MAX_DISTANCE_KM', getattr(config, 'PREDISPATCH_MAX_DISTANCE_KM', None)),
            donor_sigma_buffer=getattr(config, 'UABG_DONOR_SIGMA_BUFFER', 0.3),
            donor_tail_buffer=getattr(config, 'UABG_DONOR_TAIL_BUFFER', 0.4),
            donor_debt_buffer=getattr(config, 'UABG_DONOR_DEBT_BUFFER', 0.35),
            bid_shortage_weight=getattr(config, 'UABG_BID_SHORTAGE_WEIGHT', 0.9),
            bid_service_weight=getattr(config, 'UABG_BID_SERVICE_WEIGHT', 0.7),
            bid_backlog_weight=getattr(config, 'UABG_BID_BACKLOG_WEIGHT', 0.45),
            bid_burst_weight=getattr(config, 'UABG_BID_BURST_WEIGHT', 0.6),
            bid_debt_weight=getattr(config, 'UABG_BID_DEBT_WEIGHT', 0.85),
            ask_shortage_weight=getattr(config, 'UABG_ASK_SHORTAGE_WEIGHT', 0.85),
            ask_fairness_weight=getattr(config, 'UABG_ASK_FAIRNESS_WEIGHT', 0.7),
            ask_uncertainty_weight=getattr(config, 'UABG_ASK_UNCERTAINTY_WEIGHT', 0.65),
            distance_penalty=getattr(config, 'UABG_DISTANCE_PENALTY', 0.004),
            opportunity_eta_weight=getattr(config, 'UABG_OPPORTUNITY_ETA_WEIGHT', 0.015),
            opportunity_capture_weight=getattr(config, 'UABG_OPPORTUNITY_CAPTURE_WEIGHT', 0.90),
            opportunity_return_weight=getattr(config, 'UABG_OPPORTUNITY_RETURN_WEIGHT', 0.06),
            remote_worker_bonus=getattr(config, 'UABG_REMOTE_WORKER_BONUS', 0.05),
            switch_cooldown_slots=getattr(config, 'UABG_SWITCH_COOLDOWN_SLOTS', 2),
            switch_recent_penalty=getattr(config, 'UABG_SWITCH_RECENT_PENALTY', 0.60),
            switch_repeat_penalty=getattr(config, 'UABG_SWITCH_REPEAT_PENALTY', 0.25),
            switch_lookback_slots=getattr(config, 'UABG_SWITCH_LOOKBACK_SLOTS', 4),
            candidate_k=getattr(config, 'UABG_CANDIDATE_K', 16),
            edge_epsilon=getattr(config, 'UABG_EDGE_EPSILON', 0.05),
        )
        label = 'UABG-Micro'
    elif algo_name.lower() in ['predictive_game_mctgnet', 'predictive_game_center_lstm', *NO_PRED_GAME_ALGOS]:
        predicted_demand = remaining_predicted if algo_name.lower() not in NO_PRED_GAME_ALGOS else {rid: 0 for rid in centers.keys()}
        result = game_theoretic_predispatch_workers(
            G=G,
            worker_sim=worker_sim,
            centers=centers,
            predicted_demand=predicted_demand,
            next_slot_start_seconds=current_time,
            max_tasks_per_worker=getattr(config, 'MAX_TASKS_PER_WORKER', 4),
            backlog_counts=backlog_counts,
            backlog_weight=getattr(config, 'PREDISPATCH_BACKLOG_WEIGHT', 1.0),
            min_buffer_workers=getattr(config, 'PREDISPATCH_MIN_BUFFER_WORKERS', 3),
            reserve_ratio=getattr(config, 'PREDISPATCH_RESERVE_RATIO', 0.15),
            max_rebalance_share=getattr(config, 'PREDISPATCH_MAX_SHARE_PER_DONOR', 0.35),
            max_distance_km=getattr(config, 'PREDISPATCH_MAX_DISTANCE_KM', None),
            fairness_weight=getattr(config, 'GAME_DISPATCH_FAIRNESS_WEIGHT', 0.0),
            distance_penalty=getattr(config, 'GAME_DISPATCH_DISTANCE_PENALTY', 0.015),
            idle_penalty=getattr(config, 'PREDISPATCH_IDLE_PENALTY', 0.8),
            congestion_penalty=getattr(config, 'PREDISPATCH_CONGESTION_PENALTY', 0.35),
            remote_worker_bonus=getattr(config, 'PREDISPATCH_REMOTE_WORKER_BONUS', 0.03),
            donor_max_utility_drop=getattr(config, 'GAME_DISPATCH_DONOR_MAX_UTILITY_DROP', 0.04),
            receiver_min_utility_gain=getattr(config, 'GAME_DISPATCH_RECEIVER_MIN_GAIN', 0.01),
            max_iterations=getattr(config, 'GAME_DISPATCH_MAX_ITERATIONS', 120),
            burst_outbound_share=getattr(config, 'GAME_DISPATCH_BURST_OUTBOUND_SHARE', 0.6),
            high_demand_multiplier=getattr(config, 'GAME_DISPATCH_HIGH_DEMAND_MULTIPLIER', 1.25),
            high_demand_shortage_ratio=getattr(config, 'GAME_DISPATCH_HIGH_DEMAND_SHORTAGE_RATIO', 0.3),
            candidate_k=getattr(config, 'GAME_DISPATCH_CANDIDATE_K', 12),
            potential_gain_epsilon=getattr(config, 'GAME_DISPATCH_POTENTIAL_EPSILON', 1e-4),
        )
        label = 'Game-Micro'
    elif algo_name.lower() in ['predictive_mctgnet', 'predictive_center_lstm', 'predictive_bstgcnet']:
        result = predispatch_workers_for_next_slot(
            G=G,
            worker_sim=worker_sim,
            centers=centers,
            predicted_demand=remaining_predicted,
            next_slot_start_seconds=current_time,
            max_tasks_per_worker=getattr(config, 'MAX_TASKS_PER_WORKER', 4),
            backlog_counts=backlog_counts,
            backlog_weight=getattr(config, 'PREDISPATCH_BACKLOG_WEIGHT', 1.0),
            min_buffer_workers=getattr(config, 'PREDICTIVE_PREDISPATCH_MIN_BUFFER_WORKERS', getattr(config, 'PREDISPATCH_MIN_BUFFER_WORKERS', 3)),
            reserve_ratio=getattr(config, 'PREDICTIVE_PREDISPATCH_RESERVE_RATIO', getattr(config, 'PREDISPATCH_RESERVE_RATIO', 0.15)),
            max_rebalance_share=getattr(config, 'PREDICTIVE_PREDISPATCH_MAX_SHARE_PER_DONOR', getattr(config, 'PREDISPATCH_MAX_SHARE_PER_DONOR', 0.35)),
            max_distance_km=getattr(config, 'PREDISPATCH_MAX_DISTANCE_KM', None),
            idle_penalty=getattr(config, 'PREDICTIVE_PREDISPATCH_IDLE_PENALTY', getattr(config, 'PREDISPATCH_IDLE_PENALTY', 0.8)),
            congestion_penalty=getattr(config, 'PREDICTIVE_PREDISPATCH_CONGESTION_PENALTY', getattr(config, 'PREDISPATCH_CONGESTION_PENALTY', 0.35)),
            distance_penalty=getattr(config, 'PREDICTIVE_PREDISPATCH_DISTANCE_PENALTY', getattr(config, 'GAME_DISPATCH_DISTANCE_PENALTY', 0.015)),
            remote_worker_bonus=getattr(config, 'PREDICTIVE_PREDISPATCH_REMOTE_WORKER_BONUS', getattr(config, 'PREDISPATCH_REMOTE_WORKER_BONUS', 0.03)),
        )
        label = 'Predictive-Micro'
    else:
        return None

    if result.get('moves'):
        move_summary = ", ".join(
            [f"{m['wid']}:{m['from_region']}->{m['to_region']}" for m in result['moves'][:8]]
        )
        if len(result['moves']) > 8:
            move_summary += f", ... (+{len(result['moves']) - 8} more)"
        print(f"   [{label}] micro re-dispatch at batch {micro_idx + 1}: {move_summary}")
    else:
        print(f"   [{label}] micro re-dispatch triggered but no moves needed")

    return result


def _run_microbatch_simulation(
        algo_name,
        test_date,
        test_start_hour,
        time_slot_minutes,
        micro_batch_seconds,
        num_slots,
        G,
        centers,
        rcc_partition,
        df_tasks,
        worker_sim,
        unassigned_tasks_pool,
        dispatch_predictor,
        prediction_abs_errors,
        prediction_sq_errors,
        observed_arrivals_history,
        uncertainty_dispatch_state,
        retention_game_state,
        platform_rl_state,
):
    algo_key = algo_name.lower()
    slot_duration_seconds = int(time_slot_minutes) * 60
    is_intrabatch_online = algo_key in INTRABATCH_ONLINE_ALGOS or micro_batch_seconds >= slot_duration_seconds
    all_assignments = {}
    all_details = []
    total_profit = 0.0
    total_dist_to_center = 0.0
    total_dist_to_task = 0.0
    total_expired_tasks_global = 0

    for slot_idx in range(num_slots):
        slot_start_minute = slot_idx * time_slot_minutes
        slot_end_minute = (slot_idx + 1) * time_slot_minutes
        current_hour = test_start_hour + slot_start_minute // 60
        current_minute = slot_start_minute % 60
        next_hour = test_start_hour + slot_end_minute // 60
        next_minute = slot_end_minute % 60

        print(
            f"\n--- 时间槽 {slot_idx + 1}/{num_slots}: {current_hour:02d}:{current_minute:02d} - {next_hour:02d}:{next_minute:02d} ---")

        slot_start_seconds = test_start_hour * 3600 + slot_start_minute * 60
        slot_end_seconds = test_start_hour * 3600 + slot_end_minute * 60
        slot_timestamp = pd.Timestamp(test_date) + pd.Timedelta(seconds=slot_start_seconds)
        current_slot_predicted_demand = None
        current_predict_label = None
        current_slot_rbg_transitions = {}
        current_slot_rbg_hoard_penalty = {}
        current_slot_rbg_move_cost = {}
        current_slot_rbg_moves = []
        current_slot_rbg_stackelberg_control = {}
        current_slot_platform_transition = None
        current_slot_platform_fairness_weight = float(getattr(config, 'PFRL_FAIRNESS_SECONDARY_WEIGHT', 0.20))

        if algo_name.lower() in ['predictive_mctgnet', 'predictive_game_mctgnet', 'predictive_bstgcnet', 'predictive_center_lstm', 'predictive_game_center_lstm', *PREDICTIVE_UABG_ALGOS, *PREDICTIVE_RBG_ALGOS, *PREDICTIVE_PLATFORM_RL_ALGOS]:
            one_step_predicted_demand = dispatch_predictor.predict_region_demand(slot_timestamp)
            if one_step_predicted_demand is not None:
                current_slot_predicted_demand = one_step_predicted_demand
                backlog_counts = {rid: len(unassigned_tasks_pool[rid]) for rid in centers.keys()}
                displayed_plan_demand = dict(one_step_predicted_demand)
                predicted_distribution = None
                if algo_name.lower() in [*PREDICTIVE_UABG_ALGOS, *PREDICTIVE_RBG_ALGOS, *PREDICTIVE_PLATFORM_RL_ALGOS] and hasattr(dispatch_predictor, 'predict_region_distribution'):
                    predicted_distribution = dispatch_predictor.predict_region_distribution(slot_timestamp)

                if algo_name.lower() in [*PREDICTIVE_RBG_ALGOS, *PREDICTIVE_PLATFORM_RL_ALGOS]:
                    platform_task_weight = getattr(config, 'RBG_PLATFORM_TASK_WEIGHT', 0.30)
                    platform_gap_weight = getattr(config, 'RBG_PLATFORM_GAP_WEIGHT', 0.55)
                    platform_release_credit_weight = getattr(config, 'RBG_PLATFORM_RELEASE_CREDIT_WEIGHT', 0.35)
                    if algo_name.lower() in PREDICTIVE_PLATFORM_RL_ALGOS and platform_rl_state is not None:
                        available_workers_snapshot = {
                            rid: len(worker_sim.get_available_workers_with_center_info(rid, current_time=slot_start_seconds))
                            for rid in centers.keys()
                        }
                        current_slot_platform_transition = sample_platform_task_first_control(
                            region_ids=sorted(centers.keys()),
                            predicted_demand=displayed_plan_demand,
                            backlog_counts=backlog_counts,
                            available_workers=available_workers_snapshot,
                            max_tasks_per_worker=getattr(config, 'MAX_TASKS_PER_WORKER', 4),
                            retention_state=retention_game_state,
                            platform_state=platform_rl_state,
                            predicted_distribution=predicted_distribution,
                            backlog_weight=getattr(config, 'UABG_BACKLOG_WEIGHT', 1.0),
                            uncertainty_weight=getattr(config, 'UABG_UNCERTAINTY_WEIGHT', 0.45),
                            quantile_weight=getattr(config, 'UABG_QUANTILE_WEIGHT', 0.55),
                            burst_weight=getattr(config, 'UABG_BURST_WEIGHT', 1.2),
                            calibration_bias_weight=getattr(config, 'RBG_PREDICTION_BIAS_WEIGHT', 0.60),
                            calibration_shrink_weight=getattr(config, 'RBG_PREDICTION_SHRINK_WEIGHT', 0.55),
                            calibration_sigma_boost=getattr(config, 'RBG_PREDICTION_SIGMA_BOOST', 0.75),
                            calibration_min_scale=getattr(config, 'RBG_PREDICTION_MIN_SCALE', 0.55),
                            base_platform_task_weight=platform_task_weight,
                            base_platform_gap_weight=platform_gap_weight,
                            base_platform_release_credit_weight=platform_release_credit_weight,
                        )
                        platform_task_weight = current_slot_platform_transition['task_weight']
                        platform_gap_weight = current_slot_platform_transition['gap_weight']
                        platform_release_credit_weight = current_slot_platform_transition['release_credit_weight']
                        current_slot_platform_fairness_weight = float(current_slot_platform_transition['fairness_weight'])

                    predispatch_result = rl_retention_bilateral_predispatch_workers(
                        G=G,
                        worker_sim=worker_sim,
                        centers=centers,
                        predicted_demand=displayed_plan_demand,
                        state=retention_game_state,
                        slot_idx=slot_idx,
                        next_slot_start_seconds=slot_start_seconds,
                        predicted_distribution=predicted_distribution,
                        max_tasks_per_worker=getattr(config, 'MAX_TASKS_PER_WORKER', 4),
                        backlog_counts=backlog_counts,
                        backlog_weight=getattr(config, 'UABG_BACKLOG_WEIGHT', 1.0),
                        uncertainty_weight=getattr(config, 'UABG_UNCERTAINTY_WEIGHT', 0.45),
                        quantile_weight=getattr(config, 'UABG_QUANTILE_WEIGHT', 0.55),
                        burst_weight=getattr(config, 'UABG_BURST_WEIGHT', 1.2),
                        calibration_bias_weight=getattr(config, 'RBG_PREDICTION_BIAS_WEIGHT', 0.60),
                        calibration_shrink_weight=getattr(config, 'RBG_PREDICTION_SHRINK_WEIGHT', 0.55),
                        calibration_sigma_boost=getattr(config, 'RBG_PREDICTION_SIGMA_BOOST', 0.75),
                        calibration_min_scale=getattr(config, 'RBG_PREDICTION_MIN_SCALE', 0.55),
                        platform_task_weight=platform_task_weight,
                        platform_gap_weight=platform_gap_weight,
                        platform_release_credit_weight=platform_release_credit_weight,
                        platform_fairness_weight=float(current_slot_platform_transition.get('fairness_weight', 0.0)) if current_slot_platform_transition else 0.0,
                        platform_keep_scale=float(current_slot_platform_transition.get('keep_scale', 1.0)) if current_slot_platform_transition else 1.0,
                        platform_need_scale=float(current_slot_platform_transition.get('need_scale', 1.0)) if current_slot_platform_transition else 1.0,
                        center_local_task_weight=getattr(config, 'RBG_CENTER_LOCAL_TASK_WEIGHT', 1.0),
                        worker_completion_bonus=getattr(config, 'RBG_WORKER_COMPLETION_BONUS', 0.20),
                        worker_distance_penalty=getattr(config, 'RBG_WORKER_DISTANCE_PENALTY', 0.0),
                        same_worker_chain_bonus=getattr(config, 'RBG_WORKER_CHAIN_BONUS', 0.08),
                        min_buffer_workers=getattr(config, 'UABG_MIN_BUFFER_WORKERS', 1),
                        reserve_ratio=getattr(config, 'UABG_RESERVE_RATIO', 0.1),
                        bid_shortage_weight=getattr(config, 'UABG_BID_SHORTAGE_WEIGHT', 0.9),
                        bid_backlog_weight=getattr(config, 'UABG_BID_BACKLOG_WEIGHT', 0.45),
                        bid_debt_weight=getattr(config, 'UABG_BID_DEBT_WEIGHT', 0.85),
                        bid_burst_weight=getattr(config, 'UABG_BID_BURST_WEIGHT', 0.6),
                        ask_shortage_weight=getattr(config, 'UABG_ASK_SHORTAGE_WEIGHT', 0.85),
                        ask_uncertainty_weight=getattr(config, 'UABG_ASK_UNCERTAINTY_WEIGHT', 0.65),
                        dispatch_phase='slot_start',
                        hoard_discount_weight=getattr(config, 'RBG_HOARD_DISCOUNT_WEIGHT', 0.40),
                        move_cost_weight=getattr(config, 'RBG_MOVE_COST_WEIGHT', 0.02),
                        distance_penalty=getattr(config, 'UABG_DISTANCE_PENALTY', 0.004),
                        candidate_k=getattr(config, 'UABG_CANDIDATE_K', 16),
                        edge_epsilon=getattr(config, 'UABG_EDGE_EPSILON', 0.05),
                        record_transition=True,
                    )
                    current_slot_rbg_transitions = predispatch_result.get('transitions', {})
                    current_slot_rbg_hoard_penalty = predispatch_result.get('hoard_penalty', {})
                    current_slot_rbg_move_cost = predispatch_result.get('move_cost_by_region', {})
                    current_slot_rbg_moves = predispatch_result.get('moves', [])
                    current_slot_rbg_stackelberg_control = predispatch_result.get('stackelberg_control', {})
                    predict_label = 'Platform-RL-MCTGNet Predict' if algo_name.lower() in PREDICTIVE_PLATFORM_RL_ALGOS else 'RBG-MCTGNet Predict'
                elif algo_name.lower() in PREDICTIVE_UABG_ALGOS:
                    predispatch_result = uncertainty_aware_bilateral_predispatch_workers(
                        G=G,
                        worker_sim=worker_sim,
                        centers=centers,
                        predicted_demand=displayed_plan_demand,
                        state=uncertainty_dispatch_state,
                        slot_idx=slot_idx,
                        next_slot_start_seconds=slot_start_seconds,
                        predicted_distribution=predicted_distribution,
                        max_tasks_per_worker=getattr(config, 'MAX_TASKS_PER_WORKER', 4),
                        backlog_counts=backlog_counts,
                        backlog_weight=getattr(config, 'UABG_BACKLOG_WEIGHT', 1.0),
                        uncertainty_weight=getattr(config, 'UABG_UNCERTAINTY_WEIGHT', 0.45),
                        quantile_weight=getattr(config, 'UABG_QUANTILE_WEIGHT', 0.55),
                        burst_weight=getattr(config, 'UABG_BURST_WEIGHT', 1.2),
                        min_buffer_workers=getattr(config, 'UABG_MIN_BUFFER_WORKERS', 1),
                        reserve_ratio=getattr(config, 'UABG_RESERVE_RATIO', 0.1),
                        max_rebalance_share=getattr(config, 'UABG_MAX_SHARE_PER_DONOR', 0.6),
                        max_distance_km=getattr(config, 'UABG_MAX_DISTANCE_KM', getattr(config, 'PREDISPATCH_MAX_DISTANCE_KM', None)),
                        donor_sigma_buffer=getattr(config, 'UABG_DONOR_SIGMA_BUFFER', 0.3),
                        donor_tail_buffer=getattr(config, 'UABG_DONOR_TAIL_BUFFER', 0.4),
                        donor_debt_buffer=getattr(config, 'UABG_DONOR_DEBT_BUFFER', 0.35),
                        bid_shortage_weight=getattr(config, 'UABG_BID_SHORTAGE_WEIGHT', 0.9),
                        bid_service_weight=getattr(config, 'UABG_BID_SERVICE_WEIGHT', 0.7),
                        bid_backlog_weight=getattr(config, 'UABG_BID_BACKLOG_WEIGHT', 0.45),
                        bid_burst_weight=getattr(config, 'UABG_BID_BURST_WEIGHT', 0.6),
                        bid_debt_weight=getattr(config, 'UABG_BID_DEBT_WEIGHT', 0.85),
                        ask_shortage_weight=getattr(config, 'UABG_ASK_SHORTAGE_WEIGHT', 0.85),
                        ask_fairness_weight=getattr(config, 'UABG_ASK_FAIRNESS_WEIGHT', 0.7),
                        ask_uncertainty_weight=getattr(config, 'UABG_ASK_UNCERTAINTY_WEIGHT', 0.65),
                        distance_penalty=getattr(config, 'UABG_DISTANCE_PENALTY', 0.004),
                        opportunity_eta_weight=getattr(config, 'UABG_OPPORTUNITY_ETA_WEIGHT', 0.015),
                        opportunity_capture_weight=getattr(config, 'UABG_OPPORTUNITY_CAPTURE_WEIGHT', 0.9),
                        opportunity_return_weight=getattr(config, 'UABG_OPPORTUNITY_RETURN_WEIGHT', 0.06),
                        remote_worker_bonus=getattr(config, 'UABG_REMOTE_WORKER_BONUS', 0.05),
                        switch_cooldown_slots=getattr(config, 'UABG_SWITCH_COOLDOWN_SLOTS', 2),
                        switch_recent_penalty=getattr(config, 'UABG_SWITCH_RECENT_PENALTY', 0.6),
                        switch_repeat_penalty=getattr(config, 'UABG_SWITCH_REPEAT_PENALTY', 0.25),
                        switch_lookback_slots=getattr(config, 'UABG_SWITCH_LOOKBACK_SLOTS', 4),
                        candidate_k=getattr(config, 'UABG_CANDIDATE_K', 16),
                        edge_epsilon=getattr(config, 'UABG_EDGE_EPSILON', 0.05)
                    )
                    predict_label = 'UABG-MCTGNet Predict'
                elif algo_name.lower() in ['predictive_game_mctgnet', 'predictive_game_center_lstm']:
                    predispatch_result = game_theoretic_predispatch_workers(
                        G=G,
                        worker_sim=worker_sim,
                        centers=centers,
                        predicted_demand=displayed_plan_demand,
                        next_slot_start_seconds=slot_start_seconds,
                        max_tasks_per_worker=getattr(config, 'MAX_TASKS_PER_WORKER', 4),
                        backlog_counts=backlog_counts,
                        backlog_weight=getattr(config, 'PREDISPATCH_BACKLOG_WEIGHT', 1.0),
                        min_buffer_workers=getattr(config, 'PREDISPATCH_MIN_BUFFER_WORKERS', 3),
                        reserve_ratio=getattr(config, 'PREDISPATCH_RESERVE_RATIO', 0.15),
                        max_rebalance_share=getattr(config, 'PREDISPATCH_MAX_SHARE_PER_DONOR', 0.35),
                        max_distance_km=getattr(config, 'PREDISPATCH_MAX_DISTANCE_KM', None),
                        fairness_weight=getattr(config, 'GAME_DISPATCH_FAIRNESS_WEIGHT', 0.5),
                        distance_penalty=getattr(config, 'GAME_DISPATCH_DISTANCE_PENALTY', 0.015),
                        idle_penalty=getattr(config, 'PREDISPATCH_IDLE_PENALTY', 0.8),
                        congestion_penalty=getattr(config, 'PREDISPATCH_CONGESTION_PENALTY', 0.35),
                        remote_worker_bonus=getattr(config, 'PREDISPATCH_REMOTE_WORKER_BONUS', 0.03),
                        donor_max_utility_drop=getattr(config, 'GAME_DISPATCH_DONOR_MAX_UTILITY_DROP', 0.04),
                        receiver_min_utility_gain=getattr(config, 'GAME_DISPATCH_RECEIVER_MIN_GAIN', 0.01),
                        max_iterations=getattr(config, 'GAME_DISPATCH_MAX_ITERATIONS', 120),
                        burst_outbound_share=getattr(config, 'GAME_DISPATCH_BURST_OUTBOUND_SHARE', 0.6),
                        high_demand_multiplier=getattr(config, 'GAME_DISPATCH_HIGH_DEMAND_MULTIPLIER', 1.25),
                        high_demand_shortage_ratio=getattr(config, 'GAME_DISPATCH_HIGH_DEMAND_SHORTAGE_RATIO', 0.3),
                        candidate_k=getattr(config, 'GAME_DISPATCH_CANDIDATE_K', 12),
                        potential_gain_epsilon=getattr(config, 'GAME_DISPATCH_POTENTIAL_EPSILON', 1e-4)
                    )
                    predict_label = 'Game-CenterLSTM Predict' if algo_name.lower() == 'predictive_game_center_lstm' else 'Game-MCTGNet Predict'
                else:
                    predispatch_result = predispatch_workers_for_next_slot(
                        G=G,
                        worker_sim=worker_sim,
                        centers=centers,
                        predicted_demand=displayed_plan_demand,
                        next_slot_start_seconds=slot_start_seconds,
                        max_tasks_per_worker=getattr(config, 'MAX_TASKS_PER_WORKER', 4),
                        backlog_counts=backlog_counts,
                        backlog_weight=getattr(config, 'PREDISPATCH_BACKLOG_WEIGHT', 1.0),
                        min_buffer_workers=getattr(config, 'PREDICTIVE_PREDISPATCH_MIN_BUFFER_WORKERS', getattr(config, 'PREDISPATCH_MIN_BUFFER_WORKERS', 3)),
                        reserve_ratio=getattr(config, 'PREDICTIVE_PREDISPATCH_RESERVE_RATIO', getattr(config, 'PREDISPATCH_RESERVE_RATIO', 0.15)),
                        max_rebalance_share=getattr(config, 'PREDICTIVE_PREDISPATCH_MAX_SHARE_PER_DONOR', getattr(config, 'PREDISPATCH_MAX_SHARE_PER_DONOR', 0.35)),
                        max_distance_km=getattr(config, 'PREDISPATCH_MAX_DISTANCE_KM', None),
                        idle_penalty=getattr(config, 'PREDICTIVE_PREDISPATCH_IDLE_PENALTY', getattr(config, 'PREDISPATCH_IDLE_PENALTY', 0.8)),
                        congestion_penalty=getattr(config, 'PREDICTIVE_PREDISPATCH_CONGESTION_PENALTY', getattr(config, 'PREDISPATCH_CONGESTION_PENALTY', 0.35)),
                        distance_penalty=getattr(config, 'PREDICTIVE_PREDISPATCH_DISTANCE_PENALTY', getattr(config, 'GAME_DISPATCH_DISTANCE_PENALTY', 0.015)),
                        remote_worker_bonus=getattr(config, 'PREDICTIVE_PREDISPATCH_REMOTE_WORKER_BONUS', getattr(config, 'PREDISPATCH_REMOTE_WORKER_BONUS', 0.03))
                    )
                    predict_label = 'CenterLSTM Predict' if algo_name.lower() == 'predictive_center_lstm' else 'MCTGNet Predict'

                current_predict_label = predict_label
                if algo_name.lower() in [*PREDICTIVE_RBG_ALGOS, *PREDICTIVE_PLATFORM_RL_ALGOS]:
                    prediction_text = ", ".join(
                        [
                            f"R{rid}: mu={predispatch_result['demand_profile'][rid]['mu']:.1f}, "
                            f"sigma={predispatch_result['demand_profile'][rid]['sigma']:.1f}, "
                            f"q90={predispatch_result['demand_profile'][rid]['q90']:.1f}, "
                            f"hbias={predispatch_result['demand_profile'][rid].get('hist_bias', 0.0):.1f}, "
                            f"cbias={predispatch_result['demand_profile'][rid].get('combined_bias', 0.0):.1f}, "
                            f"bcap={predispatch_result['demand_profile'][rid].get('bias_cap', 0.0):.1f}, "
                            f"eff={predispatch_result['effective_demand'].get(rid, 0)}"
                            for rid in sorted(centers.keys())
                        ]
                    )
                elif algo_name.lower() in [*PREDICTIVE_UABG_ALGOS]:
                    prediction_text = ", ".join(
                        [
                            f"R{rid}: mu={predispatch_result['demand_profile'][rid]['mu']:.1f}, "
                            f"sigma={predispatch_result['demand_profile'][rid]['sigma']:.1f}, "
                            f"q90={predispatch_result['demand_profile'][rid]['q90']:.1f}, "
                            f"burst={predispatch_result['demand_profile'][rid]['burst_prob']:.2f}, "
                            f"eff={predispatch_result['effective_demand'].get(rid, 0)}"
                            for rid in sorted(centers.keys())
                        ]
                    )
                else:
                    prediction_text = ", ".join(
                        [
                            f"R{rid}: pred={one_step_predicted_demand.get(rid, 0)}, "
                            f"plan={displayed_plan_demand.get(rid, 0)}, "
                            f"eff={predispatch_result['effective_demand'].get(rid, 0)}"
                            for rid in sorted(centers.keys())
                        ]
                    )
                print(f"   [{predict_label}] current-slot forecast: {prediction_text}")
                if algo_name.lower() in [*PREDICTIVE_RBG_ALGOS, *PREDICTIVE_PLATFORM_RL_ALGOS]:
                    retain_text = ", ".join(
                        [
                            f"R{rid}: keep={predispatch_result['retain_count'].get(rid, 0)}, "
                            f"need={predispatch_result['desired_workers'].get(rid, 0)}, "
                            f"hoard={predispatch_result['hoard_penalty'].get(rid, 0):.1f}"
                            for rid in sorted(centers.keys())
                        ]
                    )
                    print(f"   [{predict_label}] retention policy: {retain_text}")

                if predispatch_result['moves']:
                    move_summary = ", ".join(
                        [f"{m['wid']}:{m['from_region']}->{m['to_region']}" for m in predispatch_result['moves'][:8]]
                    )
                    if len(predispatch_result['moves']) > 8:
                        move_summary += f", ... (+{len(predispatch_result['moves']) - 8} more)"
                    print(f"   [{predict_label}] pre-dispatched {len(predispatch_result['moves'])} workers: {move_summary}")
                    if uncertainty_dispatch_state is not None and algo_name.lower() in PREDICTIVE_UABG_ALGOS:
                        uncertainty_dispatch_state.record_moves(
                            slot_idx=slot_idx,
                            moved_workers=[m['wid'] for m in predispatch_result['moves']]
                        )
                else:
                    print(f"   [{predict_label}] no worker rebalancing needed")
            else:
                print("   [Predictive] insufficient history, skip pre-dispatch for this slot")

        if algo_name.lower() in NO_PRED_GAME_ALGOS:
            backlog_counts = {rid: len(unassigned_tasks_pool[rid]) for rid in centers.keys()}
            game_only_result = game_theoretic_predispatch_workers(
                G=G,
                worker_sim=worker_sim,
                centers=centers,
                predicted_demand={rid: 0 for rid in centers.keys()},
                next_slot_start_seconds=slot_start_seconds,
                max_tasks_per_worker=getattr(config, 'MAX_TASKS_PER_WORKER', 4),
                backlog_counts=backlog_counts,
                backlog_weight=getattr(config, 'PREDISPATCH_BACKLOG_WEIGHT', 1.0),
                min_buffer_workers=getattr(config, 'PREDISPATCH_MIN_BUFFER_WORKERS', 3),
                reserve_ratio=getattr(config, 'PREDISPATCH_RESERVE_RATIO', 0.15),
                max_rebalance_share=getattr(config, 'PREDISPATCH_MAX_SHARE_PER_DONOR', 0.35),
                max_distance_km=getattr(config, 'PREDISPATCH_MAX_DISTANCE_KM', None),
                fairness_weight=getattr(config, 'GAME_DISPATCH_FAIRNESS_WEIGHT', 0.5),
                distance_penalty=getattr(config, 'GAME_DISPATCH_DISTANCE_PENALTY', 0.015),
                idle_penalty=getattr(config, 'PREDISPATCH_IDLE_PENALTY', 0.8),
                congestion_penalty=getattr(config, 'PREDISPATCH_CONGESTION_PENALTY', 0.35),
                remote_worker_bonus=getattr(config, 'PREDISPATCH_REMOTE_WORKER_BONUS', 0.03),
                donor_max_utility_drop=getattr(config, 'GAME_DISPATCH_DONOR_MAX_UTILITY_DROP', 0.04),
                receiver_min_utility_gain=getattr(config, 'GAME_DISPATCH_RECEIVER_MIN_GAIN', 0.01),
                max_iterations=getattr(config, 'GAME_DISPATCH_MAX_ITERATIONS', 120),
                burst_outbound_share=getattr(config, 'GAME_DISPATCH_BURST_OUTBOUND_SHARE', 0.6),
                high_demand_multiplier=getattr(config, 'GAME_DISPATCH_HIGH_DEMAND_MULTIPLIER', 1.25),
                high_demand_shortage_ratio=getattr(config, 'GAME_DISPATCH_HIGH_DEMAND_SHORTAGE_RATIO', 0.3),
                candidate_k=getattr(config, 'GAME_DISPATCH_CANDIDATE_K', 12),
                potential_gain_epsilon=getattr(config, 'GAME_DISPATCH_POTENTIAL_EPSILON', 1e-4)
            )
            if game_only_result['moves']:
                move_summary = ", ".join(
                    [f"{m['wid']}:{m['from_region']}->{m['to_region']}" for m in game_only_result['moves'][:8]]
                )
                if len(game_only_result['moves']) > 8:
                    move_summary += f", ... (+{len(game_only_result['moves']) - 8} more)"
                print(f"   [NoPred-Game Dispatch] pre-dispatched {len(game_only_result['moves'])} workers: {move_summary}")
            else:
                print("   [NoPred-Game Dispatch] no worker rebalancing needed")

        slot_assignments = {}
        slot_details = []
        slot_profit = 0.0
        slot_dist_to_center = 0.0
        slot_dist_to_task = 0.0
        slot_new_tasks_per_center = {rid: 0 for rid in centers.keys()}
        slot_total_tasks_per_center = {rid: len(unassigned_tasks_pool[rid]) for rid in centers.keys()}
        slot_assigned_tasks_per_center = {rid: 0 for rid in centers.keys()}
        slot_worker_ids = set()
        slot_expired_count = 0
        last_micro_redispatch_idx = -999
        if algo_name.lower() in [*PREDICTIVE_RBG_ALGOS, *PREDICTIVE_PLATFORM_RL_ALGOS]:
            redispatch_gap = int(getattr(config, 'RBG_MICROBATCH_REDISPATCH_MIN_GAP', 1))
        else:
            redispatch_gap = int(getattr(config, 'MICROBATCH_REDISPATCH_MIN_GAP', 2))

        for micro_idx, micro_start_seconds in enumerate(range(slot_start_seconds, slot_end_seconds, micro_batch_seconds)):
            micro_end_seconds = min(slot_end_seconds, micro_start_seconds + micro_batch_seconds)
            worker_sim.advance_workers_to_time(centers, micro_start_seconds)

            micro_new_tasks = 0
            if not df_tasks.empty:
                mask = (df_tasks['seconds_of_day'] >= micro_start_seconds) & (df_tasks['seconds_of_day'] < micro_end_seconds)
                current_tasks = df_tasks[mask]
                for _, row in current_tasks.iterrows():
                    nearest_node = row['nearest_node']
                    if nearest_node not in rcc_partition:
                        continue
                    region_id = rcc_partition[nearest_node]
                    task = (
                        nearest_node,
                        row['task_id'],
                        config.TASK_BASE_REWARD,
                        row['seconds_of_day'] + config.TASK_EXPIRE_MINUTES * 60,
                        row['seconds_of_day'],
                    )
                    unassigned_tasks_pool[region_id].append(task)
                    slot_new_tasks_per_center[region_id] += 1
                    slot_total_tasks_per_center[region_id] += 1
                    micro_new_tasks += 1

            workers_per_center = {}
            for region_id in centers.keys():
                workers = worker_sim.get_available_workers_with_center_info(region_id, current_time=micro_start_seconds)
                workers_per_center[region_id] = [(w[0], w[1], w[2], w[3], centers[region_id]) for w in workers]

            if (
                micro_idx > 0
                and (micro_idx - last_micro_redispatch_idx) >= redispatch_gap
                and algo_name.lower() in [
                    'predictive_mctgnet', 'predictive_game_mctgnet', 'predictive_bstgcnet',
                    'predictive_center_lstm', 'predictive_game_center_lstm',
                    *PREDICTIVE_UABG_ALGOS, *PREDICTIVE_RBG_ALGOS, *PREDICTIVE_PLATFORM_RL_ALGOS, *NO_PRED_GAME_ALGOS
                ]
            ):
                micro_dispatch_result = _run_triggered_micro_predispatch(
                    algo_name=algo_name,
                    G=G,
                    worker_sim=worker_sim,
                    centers=centers,
                    current_time=micro_start_seconds,
                    slot_idx=slot_idx,
                    micro_idx=micro_idx,
                    slot_start_seconds=slot_start_seconds,
                    slot_end_seconds=slot_end_seconds,
                    current_slot_predicted_demand=current_slot_predicted_demand,
                    slot_new_tasks_per_center=slot_new_tasks_per_center,
                    unassigned_tasks_pool=unassigned_tasks_pool,
                    uncertainty_dispatch_state=uncertainty_dispatch_state,
                    dispatch_predictor=dispatch_predictor,
                    retention_game_state=retention_game_state,
                    platform_rl_state=platform_rl_state,
                )
                if micro_dispatch_result is not None:
                    last_micro_redispatch_idx = micro_idx
                    for region_id in centers.keys():
                        workers = worker_sim.get_available_workers_with_center_info(region_id, current_time=micro_start_seconds)
                        workers_per_center[region_id] = [(w[0], w[1], w[2], w[3], centers[region_id]) for w in workers]

            total_workers = sum(len(w) for w in workers_per_center.values())
            total_current_tasks = sum(len(unassigned_tasks_pool[rid]) for rid in centers.keys())
            if total_current_tasks > 0 or micro_new_tasks > 0:
                total_micro = len(range(slot_start_seconds, slot_end_seconds, micro_batch_seconds))
                progress_label = "批次内在线" if is_intrabatch_online else "微批"
                print(
                    f"   [{progress_label} {micro_idx + 1}/{total_micro}] 可用工人: {total_workers} 个 | "
                    f"新增订单: {micro_new_tasks} 个 | 池内总单量: {total_current_tasks} 个"
                )

            micro_assignments = {}
            micro_profit = 0.0
            micro_details = []
            if total_current_tasks > 0 and total_workers > 0:
                if is_intrabatch_online:
                    tasks_per_center = _build_intrabatch_online_tasks(
                        unassigned_tasks_pool=unassigned_tasks_pool,
                        current_time=micro_start_seconds,
                    )
                else:
                    tasks_per_center = _build_microbatch_candidate_tasks(
                        unassigned_tasks_pool=unassigned_tasks_pool,
                        workers_per_center=workers_per_center,
                        current_time=micro_start_seconds,
                        slot_end_seconds=slot_end_seconds,
                    )
                micro_assignments, micro_profit, micro_details = _run_assignment_for_window(
                    algo_name=algo_name,
                    G=G,
                    config=config,
                    centers=centers,
                    rcc_partition=rcc_partition,
                    workers_per_center=workers_per_center,
                    tasks_per_center=tasks_per_center,
                    slot_start_seconds=micro_start_seconds,
                    slot_end_seconds=slot_end_seconds,
                    stackelberg_control=current_slot_rbg_stackelberg_control,
                )
                dist_center_inc, dist_task_inc = _apply_assignment_results_to_workers(G, worker_sim, micro_details)
                slot_dist_to_center += dist_center_inc
                slot_dist_to_task += dist_task_inc
                slot_assignments.update(micro_assignments)
                slot_details.extend(micro_details)
                slot_profit += micro_profit
                for detail in micro_details:
                    slot_assigned_tasks_per_center[detail['region_id']] += 1
                    slot_worker_ids.add(detail['wid'])
                if micro_assignments:
                    print(
                        f"      分配结果: 成交 {len(micro_assignments)} 单, "
                        f"调度 {len(set(k[0] for k in micro_assignments.keys()))} 名工人"
                    )

            assigned_task_ids = {k[1] for k in micro_assignments.keys()}
            for rid in centers.keys():
                new_pool = []
                for t in unassigned_tasks_pool[rid]:
                    if t[1] in assigned_task_ids:
                        continue
                    if micro_end_seconds >= t[3]:
                        slot_expired_count += 1
                        continue
                    new_pool.append(t)
                unassigned_tasks_pool[rid] = new_pool

        total_expired_tasks_global += slot_expired_count
        leftover_count = sum(len(pool) for pool in unassigned_tasks_pool.values())

        if current_slot_predicted_demand is not None:
            actual_region_demand = {rid: int(slot_new_tasks_per_center[rid]) for rid in centers.keys()}
            actual_text = ", ".join(
                [
                    f"R{rid}: actual={slot_new_tasks_per_center[rid]}, "
                    f"abs={abs(current_slot_predicted_demand.get(rid, 0) - slot_new_tasks_per_center[rid])}"
                    for rid in sorted(centers.keys())
                ]
            )
            print(f"   [{current_predict_label}] actual arrivals: {actual_text}")
            if hasattr(dispatch_predictor, 'update_online'):
                dispatch_predictor.update_online(
                    slot_timestamp=slot_timestamp,
                    actual_region_demand=actual_region_demand,
                    predicted_region_demand=current_slot_predicted_demand
                )
            if uncertainty_dispatch_state is not None and algo_name.lower() in PREDICTIVE_UABG_ALGOS:
                uncertainty_dispatch_state.record_prediction_feedback(
                    predicted_region_demand=current_slot_predicted_demand,
                    actual_region_demand=actual_region_demand
                )
            if retention_game_state is not None and algo_name.lower() in [*PREDICTIVE_RBG_ALGOS, *PREDICTIVE_PLATFORM_RL_ALGOS]:
                retention_game_state.record_prediction_feedback(
                    predicted_region_demand=current_slot_predicted_demand,
                    actual_region_demand=actual_region_demand
                )
            for rid in centers.keys():
                err = float(current_slot_predicted_demand.get(rid, 0) - slot_new_tasks_per_center[rid])
                prediction_abs_errors.append(abs(err))
                prediction_sq_errors.append(err * err)

        if uncertainty_dispatch_state is not None and algo_name.lower() in PREDICTIVE_UABG_ALGOS:
            uncertainty_dispatch_state.record_service_outcome(
                total_tasks_by_region=slot_total_tasks_per_center,
                assigned_tasks_by_region=slot_assigned_tasks_per_center
            )
        if retention_game_state is not None and algo_name.lower() in [*PREDICTIVE_RBG_ALGOS, *PREDICTIVE_PLATFORM_RL_ALGOS] and current_slot_rbg_transitions:
                reward_by_region = update_rl_retention_bilateral_state(
                    state=retention_game_state,
                    transitions=current_slot_rbg_transitions,
                    assigned_tasks_by_region=slot_assigned_tasks_per_center,
                    total_tasks_by_region=slot_total_tasks_per_center,
                    hoard_penalty_by_region=current_slot_rbg_hoard_penalty,
                    move_cost_by_region=current_slot_rbg_move_cost,
                    moves=current_slot_rbg_moves,
                    hoard_penalty_weight=float(getattr(config, 'RBG_REWARD_HOARD_WEIGHT', 0.02)),
                    move_cost_weight=float(getattr(config, 'RBG_REWARD_MOVE_WEIGHT', 0.08)),
                    unfairness_weight=float(getattr(config, 'RBG_REWARD_UNFAIRNESS_WEIGHT', 1.0)),
                )
                reward_text = ", ".join(
                    [f"R{rid}: reward={reward_by_region.get(rid, 0.0):.2f}" for rid in sorted(centers.keys())]
                )
                reward_label = 'Platform-RL-MCTGNet Reward' if algo_name.lower() in PREDICTIVE_PLATFORM_RL_ALGOS else 'RBG-MCTGNet Reward'
                print(f"   [{reward_label}] {reward_text}")

        if platform_rl_state is not None and algo_name.lower() in PREDICTIVE_PLATFORM_RL_ALGOS and current_slot_platform_transition is not None:
            platform_stats = update_platform_task_first_state(
                state=platform_rl_state,
                transition=current_slot_platform_transition,
                assigned_tasks_by_region=slot_assigned_tasks_per_center,
                total_tasks_by_region=slot_total_tasks_per_center,
                fairness_secondary_weight=current_slot_platform_fairness_weight,
            )
            print(
                f"   [Platform-RL Policy] reward={platform_stats['platform_reward']:.2f}, "
                f"completion={platform_stats['completion_rate']:.4f}, "
                f"mean_unfairness={platform_stats['mean_unfairness']:.4f}"
            )

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

        print(f"分配结果: 成交 {len(slot_assignments)} 单, 调度 {len(slot_worker_ids)} 名工人")
        if slot_expired_count > 0:
            print(f"   ❌ 超时淘汰订单: {slot_expired_count} 个 (已自动取消该订单，不扣除利润)")
        if leftover_count > 0:
            print(f"   ⏳ 剩余积压订单: {leftover_count} 个 (安全范围内，自动滚入下一轮)")

        total_dist_to_center += slot_dist_to_center
        total_dist_to_task += slot_dist_to_task
        all_assignments.update(slot_assignments)
        all_details.extend(slot_details)
        total_profit += slot_profit

    return (
        all_assignments,
        all_details,
        total_profit,
        total_dist_to_center,
        total_dist_to_task,
        total_expired_tasks_global,
    )


def run_online_simulation_with_center_pickup(
        algo_name: str = 'greedy',
        test_date: str = DEFAULT_TEST_DATE,
        test_start_hour: int = DEFAULT_START_HOUR,
        test_end_hour: int = DEFAULT_END_HOUR,
        time_slot_minutes: int = DEFAULT_TIME_SLOT_MINUTES
):
    cpu_start = time.process_time()
    algo_key = algo_name.lower()
    execution_mode = "15分钟槽在线"
    effective_micro_batch_minutes = time_slot_minutes
    micro_batch_seconds = effective_micro_batch_minutes * 60
    compare_slots = max(1, DEFAULT_COMPARE_SLOT_COUNT)
    compare_end_seconds = test_start_hour * 3600 + compare_slots * time_slot_minutes * 60
    compare_end_hour = int(compare_end_seconds // 3600)
    compare_end_minute = int((compare_end_seconds % 3600) // 60)
    print("=" * 70)
    print(f"在线物流调度仿真实验 (算法: {algo_name.upper()})")
    print("=" * 70)
    print(
        f"测试日期：{test_date} | 时段：{test_start_hour}:00-"
        f"{compare_end_hour}:{compare_end_minute:02d} | 时间槽：{time_slot_minutes} 分钟 | 模式：{execution_mode} | 内部步长：{effective_micro_batch_minutes} 分钟"
    )
    print("=" * 70)

    context = _build_simulation_context(
        test_date,
        test_start_hour,
        test_end_hour,
        time_slot_minutes,
        compare_slots
    )
    G = context['G']
    rcc_partition = context['rcc_partition']
    centers = context['centers']
    df_tasks = context['df_tasks']
    total_tasks_per_center = copy.deepcopy(context['total_tasks_per_center'])
    worker_sim = _restore_worker_simulator(G, context['worker_state'])

    unassigned_tasks_pool = {region_id: [] for region_id in centers.keys()}
    total_expired_tasks_global = 0

    # ==================== 4. 在线仿真实验 ====================
    print(f"\n【阶段 5】开始{execution_mode}仿真...")
    all_assignments = {}
    all_details = []
    total_profit = 0
    total_dist_to_center = 0
    total_dist_to_task = 0

    num_slots = compare_slots
    observed_arrivals_history = []
    dispatch_predictor = None
    prediction_abs_errors = []
    prediction_sq_errors = []
    uncertainty_dispatch_state = None

    if algo_name.lower() in ['predictive_mctgnet', 'predictive_game_mctgnet', 'predictive_bstgcnet', *PREDICTIVE_UABG_ALGOS, *PREDICTIVE_RBG_ALGOS, *PREDICTIVE_PLATFORM_RL_ALGOS]:
        dispatch_predictor = _get_or_train_mctg_predictor(
            test_date=test_date,
            test_start_hour=test_start_hour,
            test_end_hour=test_end_hour,
            time_slot_minutes=time_slot_minutes,
            coords=context['coords'],
            nodes=context['nodes'],
            rcc_partition=rcc_partition,
            centers=centers
        )
    elif algo_name.lower() in ['predictive_center_lstm', 'predictive_game_center_lstm']:
        dispatch_predictor = _get_or_train_center_lstm_predictor(
            test_date=test_date,
            test_start_hour=test_start_hour,
            test_end_hour=test_end_hour,
            time_slot_minutes=time_slot_minutes,
            coords=context['coords'],
            nodes=context['nodes'],
            rcc_partition=rcc_partition,
            centers=centers
        )

    if dispatch_predictor is not None and hasattr(dispatch_predictor, 'reset_online_state'):
        dispatch_predictor.reset_online_state()
    if algo_name.lower() in PREDICTIVE_UABG_ALGOS:
        uncertainty_dispatch_state = UncertaintyAwareBilateralState(
            region_ids=sorted(centers.keys()),
            history_size=int(getattr(config, 'UABG_HISTORY_SIZE', 12)),
            service_debt_decay=float(getattr(config, 'UABG_SERVICE_DEBT_DECAY', 0.85)),
            max_service_debt=float(getattr(config, 'UABG_MAX_SERVICE_DEBT', 4.0)),
            move_history_size=int(getattr(config, 'UABG_MOVE_HISTORY_SIZE', 8))
        )
    retention_game_state = None
    if algo_name.lower() in [*PREDICTIVE_RBG_ALGOS, *PREDICTIVE_PLATFORM_RL_ALGOS]:
        retention_game_state = RLRetentionBilateralState(
            region_ids=sorted(centers.keys()),
            action_ratios=tuple(getattr(config, 'RBG_ACTION_RATIOS', (-0.30, -0.15, 0.0, 0.15, 0.30))),
            learning_rate=float(getattr(config, 'RBG_LEARNING_RATE', 0.03)),
            temperature=float(getattr(config, 'RBG_TEMPERATURE', 0.85)),
            exploration_prob=float(getattr(config, 'RBG_EXPLORATION_PROB', 0.12)),
            service_debt_decay=float(getattr(config, 'RBG_SERVICE_DEBT_DECAY', 0.85)),
            max_service_debt=float(getattr(config, 'RBG_MAX_SERVICE_DEBT', 4.0)),
            move_history_size=int(getattr(config, 'RBG_MOVE_HISTORY_SIZE', 8)),
            random_seed=int(getattr(config, 'RBG_RANDOM_SEED', 42)),
        )
        _offline_warm_start_rbg_state(
            state=retention_game_state,
            test_date=test_date,
            test_start_hour=test_start_hour,
            test_end_hour=test_end_hour,
            time_slot_minutes=time_slot_minutes,
            G=G,
            coords=context['coords'],
            nodes=context['nodes'],
            rcc_partition=rcc_partition,
            centers=centers,
            worker_sim=worker_sim,
        )
    platform_rl_state = None
    if algo_name.lower() in PREDICTIVE_PLATFORM_RL_ALGOS:
        platform_rl_state = PlatformTaskFirstRLState(
            region_ids=sorted(centers.keys()),
            learning_rate=float(getattr(config, 'PFRL_LEARNING_RATE', 0.03)),
            temperature=float(getattr(config, 'PFRL_TEMPERATURE', 0.90)),
            exploration_prob=float(getattr(config, 'PFRL_EXPLORATION_PROB', 0.10)),
            random_seed=int(getattr(config, 'PFRL_RANDOM_SEED', 7)),
        )

    all_assignments, all_details, total_profit, total_dist_to_center, total_dist_to_task, total_expired_tasks_global = _run_microbatch_simulation(
        algo_name=algo_name,
        test_date=test_date,
        test_start_hour=test_start_hour,
        time_slot_minutes=time_slot_minutes,
        micro_batch_seconds=micro_batch_seconds,
        num_slots=num_slots,
        G=G,
        centers=centers,
        rcc_partition=rcc_partition,
        df_tasks=df_tasks,
        worker_sim=worker_sim,
        unassigned_tasks_pool=unassigned_tasks_pool,
        dispatch_predictor=dispatch_predictor,
        prediction_abs_errors=prediction_abs_errors,
        prediction_sq_errors=prediction_sq_errors,
        observed_arrivals_history=observed_arrivals_history,
        uncertainty_dispatch_state=uncertainty_dispatch_state,
        retention_game_state=retention_game_state,
        platform_rl_state=platform_rl_state,
    )

    assigned_tasks_per_center = {region_id: 0 for region_id in centers.keys()}
    for detail in all_details:
        assigned_tasks_per_center[detail['region_id']] += 1

    rho, u_rho = calculate_collaboration_unfairness(total_tasks_per_center, assigned_tasks_per_center)
    cpu_time = time.process_time() - cpu_start
    total_tasks_in_scope = int(sum(total_tasks_per_center.values()))
    task_completion_rate = (len(all_assignments) / total_tasks_in_scope) if total_tasks_in_scope > 0 else 0.0

    print(f"\n[{algo_name.upper()}] 仿真完成:")
    print(f"  - #Assigned Tasks: {len(all_assignments)}")
    print(f"  - Task Completion Rate: {task_completion_rate:.4f}")
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
        'task_completion_rate': task_completion_rate,
        'rho': rho,
        'u_rho': u_rho,
        'cpu_time': cpu_time,
        'pred_mae': pred_mae,
        'pred_rmse': pred_rmse
    }

    return all_assignments, all_details, metrics

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
        current_slot_predicted_demand = None
        current_predict_label = None

        if algo_name.lower() in ['predictive_mctgnet', 'predictive_game_mctgnet', 'predictive_bstgcnet', 'predictive_center_lstm', 'predictive_game_center_lstm', *PREDICTIVE_UABG_ALGOS]:
            slot_timestamp = pd.Timestamp(test_date) + pd.Timedelta(seconds=slot_start_seconds)
            one_step_predicted_demand = dispatch_predictor.predict_region_demand(slot_timestamp)
            if one_step_predicted_demand is not None:
                current_slot_predicted_demand = one_step_predicted_demand
            if one_step_predicted_demand is not None:
                backlog_counts = {rid: len(unassigned_tasks_pool[rid]) for rid in centers.keys()}
                displayed_plan_demand = dict(one_step_predicted_demand)
                predicted_distribution = None
                if algo_name.lower() in PREDICTIVE_UABG_ALGOS and hasattr(dispatch_predictor, 'predict_region_distribution'):
                    predicted_distribution = dispatch_predictor.predict_region_distribution(slot_timestamp)
                if algo_name.lower() in PREDICTIVE_UABG_ALGOS:
                    predispatch_result = uncertainty_aware_bilateral_predispatch_workers(
                        G=G,
                        worker_sim=worker_sim,
                        centers=centers,
                        predicted_demand=displayed_plan_demand,
                        state=uncertainty_dispatch_state,
                        slot_idx=slot_idx,
                        next_slot_start_seconds=slot_start_seconds,
                        predicted_distribution=predicted_distribution,
                        max_tasks_per_worker=getattr(config, 'MAX_TASKS_PER_WORKER', 4),
                        backlog_counts=backlog_counts,
                        backlog_weight=getattr(config, 'UABG_BACKLOG_WEIGHT', 1.0),
                        uncertainty_weight=getattr(config, 'UABG_UNCERTAINTY_WEIGHT', 0.45),
                        quantile_weight=getattr(config, 'UABG_QUANTILE_WEIGHT', 0.55),
                        burst_weight=getattr(config, 'UABG_BURST_WEIGHT', 1.2),
                        min_buffer_workers=getattr(config, 'UABG_MIN_BUFFER_WORKERS', 1),
                        reserve_ratio=getattr(config, 'UABG_RESERVE_RATIO', 0.1),
                        max_rebalance_share=getattr(config, 'UABG_MAX_SHARE_PER_DONOR', 0.6),
                        max_distance_km=getattr(config, 'UABG_MAX_DISTANCE_KM', getattr(config, 'PREDISPATCH_MAX_DISTANCE_KM', None)),
                        donor_sigma_buffer=getattr(config, 'UABG_DONOR_SIGMA_BUFFER', 0.3),
                        donor_tail_buffer=getattr(config, 'UABG_DONOR_TAIL_BUFFER', 0.4),
                        donor_debt_buffer=getattr(config, 'UABG_DONOR_DEBT_BUFFER', 0.35),
                        bid_shortage_weight=getattr(config, 'UABG_BID_SHORTAGE_WEIGHT', 0.9),
                        bid_service_weight=getattr(config, 'UABG_BID_SERVICE_WEIGHT', 0.7),
                        bid_backlog_weight=getattr(config, 'UABG_BID_BACKLOG_WEIGHT', 0.45),
                        bid_burst_weight=getattr(config, 'UABG_BID_BURST_WEIGHT', 0.6),
                        bid_debt_weight=getattr(config, 'UABG_BID_DEBT_WEIGHT', 0.85),
                        ask_shortage_weight=getattr(config, 'UABG_ASK_SHORTAGE_WEIGHT', 0.85),
                        ask_fairness_weight=getattr(config, 'UABG_ASK_FAIRNESS_WEIGHT', 0.7),
                        ask_uncertainty_weight=getattr(config, 'UABG_ASK_UNCERTAINTY_WEIGHT', 0.65),
                        distance_penalty=getattr(config, 'UABG_DISTANCE_PENALTY', 0.004),
                        opportunity_eta_weight=getattr(config, 'UABG_OPPORTUNITY_ETA_WEIGHT', 0.015),
                        opportunity_capture_weight=getattr(config, 'UABG_OPPORTUNITY_CAPTURE_WEIGHT', 0.9),
                        opportunity_return_weight=getattr(config, 'UABG_OPPORTUNITY_RETURN_WEIGHT', 0.06),
                        remote_worker_bonus=getattr(config, 'UABG_REMOTE_WORKER_BONUS', 0.05),
                        switch_cooldown_slots=getattr(config, 'UABG_SWITCH_COOLDOWN_SLOTS', 2),
                        switch_recent_penalty=getattr(config, 'UABG_SWITCH_RECENT_PENALTY', 0.6),
                        switch_repeat_penalty=getattr(config, 'UABG_SWITCH_REPEAT_PENALTY', 0.25),
                        switch_lookback_slots=getattr(config, 'UABG_SWITCH_LOOKBACK_SLOTS', 4),
                        candidate_k=getattr(config, 'UABG_CANDIDATE_K', 16),
                        edge_epsilon=getattr(config, 'UABG_EDGE_EPSILON', 0.05)
                    )
                    predict_label = 'UABG-MCTGNet Predict'
                elif algo_name.lower() in ['predictive_game_mctgnet', 'predictive_game_center_lstm']:
                    predispatch_result = game_theoretic_predispatch_workers(
                        G=G,
                        worker_sim=worker_sim,
                        centers=centers,
                        predicted_demand=displayed_plan_demand,
                        next_slot_start_seconds=slot_start_seconds,
                        max_tasks_per_worker=getattr(config, 'MAX_TASKS_PER_WORKER', 4),
                        backlog_counts=backlog_counts,
                        backlog_weight=getattr(config, 'PREDISPATCH_BACKLOG_WEIGHT', 1.0),
                        min_buffer_workers=getattr(config, 'PREDISPATCH_MIN_BUFFER_WORKERS', 3),
                        reserve_ratio=getattr(config, 'PREDISPATCH_RESERVE_RATIO', 0.15),
                        max_rebalance_share=getattr(config, 'PREDISPATCH_MAX_SHARE_PER_DONOR', 0.35),
                        max_distance_km=getattr(config, 'PREDISPATCH_MAX_DISTANCE_KM', None),
                        fairness_weight=getattr(config, 'GAME_DISPATCH_FAIRNESS_WEIGHT', 0.5),
                        distance_penalty=getattr(config, 'GAME_DISPATCH_DISTANCE_PENALTY', 0.015),
                        idle_penalty=getattr(config, 'PREDISPATCH_IDLE_PENALTY', 0.8),
                        congestion_penalty=getattr(config, 'PREDISPATCH_CONGESTION_PENALTY', 0.35),
                        remote_worker_bonus=getattr(config, 'PREDISPATCH_REMOTE_WORKER_BONUS', 0.03),
                        donor_max_utility_drop=getattr(config, 'GAME_DISPATCH_DONOR_MAX_UTILITY_DROP', 0.04),
                        receiver_min_utility_gain=getattr(config, 'GAME_DISPATCH_RECEIVER_MIN_GAIN', 0.01),
                        max_iterations=getattr(config, 'GAME_DISPATCH_MAX_ITERATIONS', 120),
                        burst_outbound_share=getattr(config, 'GAME_DISPATCH_BURST_OUTBOUND_SHARE', 0.6),
                        high_demand_multiplier=getattr(config, 'GAME_DISPATCH_HIGH_DEMAND_MULTIPLIER', 1.25),
                        high_demand_shortage_ratio=getattr(config, 'GAME_DISPATCH_HIGH_DEMAND_SHORTAGE_RATIO', 0.3),
                        candidate_k=getattr(config, 'GAME_DISPATCH_CANDIDATE_K', 12),
                        potential_gain_epsilon=getattr(config, 'GAME_DISPATCH_POTENTIAL_EPSILON', 1e-4)
                    )
                    predict_label = 'Game-CenterLSTM Predict' if algo_name.lower() == 'predictive_game_center_lstm' else 'Game-MCTGNet Predict'
                else:
                    predispatch_result = predispatch_workers_for_next_slot(
                        G=G,
                        worker_sim=worker_sim,
                        centers=centers,
                        predicted_demand=displayed_plan_demand,
                        next_slot_start_seconds=slot_start_seconds,
                        max_tasks_per_worker=getattr(config, 'MAX_TASKS_PER_WORKER', 4),
                        backlog_counts=backlog_counts,
                        backlog_weight=getattr(config, 'PREDISPATCH_BACKLOG_WEIGHT', 1.0),
                        min_buffer_workers=getattr(config, 'PREDICTIVE_PREDISPATCH_MIN_BUFFER_WORKERS', getattr(config, 'PREDISPATCH_MIN_BUFFER_WORKERS', 3)),
                        reserve_ratio=getattr(config, 'PREDICTIVE_PREDISPATCH_RESERVE_RATIO', getattr(config, 'PREDISPATCH_RESERVE_RATIO', 0.15)),
                        max_rebalance_share=getattr(config, 'PREDICTIVE_PREDISPATCH_MAX_SHARE_PER_DONOR', getattr(config, 'PREDISPATCH_MAX_SHARE_PER_DONOR', 0.35)),
                        max_distance_km=getattr(config, 'PREDISPATCH_MAX_DISTANCE_KM', None),
                        idle_penalty=getattr(config, 'PREDICTIVE_PREDISPATCH_IDLE_PENALTY', getattr(config, 'PREDISPATCH_IDLE_PENALTY', 0.8)),
                        congestion_penalty=getattr(config, 'PREDICTIVE_PREDISPATCH_CONGESTION_PENALTY', getattr(config, 'PREDISPATCH_CONGESTION_PENALTY', 0.35)),
                        distance_penalty=getattr(config, 'PREDICTIVE_PREDISPATCH_DISTANCE_PENALTY', getattr(config, 'GAME_DISPATCH_DISTANCE_PENALTY', 0.015)),
                        remote_worker_bonus=getattr(config, 'PREDICTIVE_PREDISPATCH_REMOTE_WORKER_BONUS', getattr(config, 'PREDISPATCH_REMOTE_WORKER_BONUS', 0.03))
                    )
                    predict_label = 'CenterLSTM Predict' if algo_name.lower() == 'predictive_center_lstm' else 'MCTGNet Predict'
                current_predict_label = predict_label
                if algo_name.lower() in PREDICTIVE_UABG_ALGOS:
                    prediction_text = ", ".join(
                        [
                            f"R{rid}: mu={predispatch_result['demand_profile'][rid]['mu']:.1f}, "
                            f"sigma={predispatch_result['demand_profile'][rid]['sigma']:.1f}, "
                            f"q90={predispatch_result['demand_profile'][rid]['q90']:.1f}, "
                            f"burst={predispatch_result['demand_profile'][rid]['burst_prob']:.2f}, "
                            f"eff={predispatch_result['effective_demand'].get(rid, 0)}"
                            for rid in sorted(centers.keys())
                        ]
                    )
                else:
                    prediction_text = ", ".join(
                        [
                            f"R{rid}: pred={one_step_predicted_demand.get(rid, 0)}, "
                            f"plan={displayed_plan_demand.get(rid, 0)}, "
                            f"eff={predispatch_result['effective_demand'].get(rid, 0)}"
                            for rid in sorted(centers.keys())
                        ]
                    )
                print(f"   [{predict_label}] current-slot forecast: {prediction_text}")
                if algo_name.lower() in PREDICTIVE_UABG_ALGOS:
                    diag = predispatch_result.get('diagnostics', {})
                    print(
                        f"   [{predict_label}] diag: mode={'relative' if diag.get('global_shortage_mode') else 'absolute'}, "
                        f"donors={diag.get('active_donors', 0)}, receivers={diag.get('active_receivers', 0)}, "
                        f"candidates={diag.get('candidate_pairs', 0)}, positive={diag.get('positive_edges', 0)}"
                    )
                if predispatch_result['moves']:
                    move_summary = ", ".join(
                        [f"{m['wid']}:{m['from_region']}->{m['to_region']}" for m in predispatch_result['moves'][:8]]
                    )
                    if len(predispatch_result['moves']) > 8:
                        move_summary += f", ... (+{len(predispatch_result['moves']) - 8} more)"
                    print(f"   [{predict_label}] pre-dispatched {len(predispatch_result['moves'])} workers: {move_summary}")
                    if uncertainty_dispatch_state is not None and algo_name.lower() in PREDICTIVE_UABG_ALGOS:
                        uncertainty_dispatch_state.record_moves(
                            slot_idx=slot_idx,
                            moved_workers=[m['wid'] for m in predispatch_result['moves']]
                        )
                else:
                    if algo_name.lower() in PREDICTIVE_UABG_ALGOS:
                        diag = predispatch_result.get('diagnostics', {})
                        print(
                            f"   [{predict_label}] no worker rebalancing needed "
                            f"(max_gap={diag.get('max_receiver_gap', 0)}, max_supply={diag.get('max_donor_supply', 0)})"
                        )
                    else:
                        print(f"   [{predict_label}] no worker rebalancing needed")
            else:
                print("   [MCTGNet Predict] insufficient multi-batch history, skip pre-dispatch for this slot")

        print(f">> 按当前时刻推进工人位置与状态...")
        if algo_name.lower() in NO_PRED_GAME_ALGOS:
            backlog_counts = {rid: len(unassigned_tasks_pool[rid]) for rid in centers.keys()}
            displayed_plan_demand = {rid: 0 for rid in centers.keys()}
            game_only_result = game_theoretic_predispatch_workers(
                G=G,
                worker_sim=worker_sim,
                centers=centers,
                predicted_demand=displayed_plan_demand,
                next_slot_start_seconds=slot_start_seconds,
                max_tasks_per_worker=getattr(config, 'MAX_TASKS_PER_WORKER', 4),
                backlog_counts=backlog_counts,
                backlog_weight=getattr(config, 'PREDISPATCH_BACKLOG_WEIGHT', 1.0),
                min_buffer_workers=getattr(config, 'PREDISPATCH_MIN_BUFFER_WORKERS', 3),
                reserve_ratio=getattr(config, 'PREDISPATCH_RESERVE_RATIO', 0.15),
                max_rebalance_share=getattr(config, 'PREDISPATCH_MAX_SHARE_PER_DONOR', 0.35),
                max_distance_km=getattr(config, 'PREDISPATCH_MAX_DISTANCE_KM', None),
                fairness_weight=getattr(config, 'GAME_DISPATCH_FAIRNESS_WEIGHT', 0.5),
                distance_penalty=getattr(config, 'GAME_DISPATCH_DISTANCE_PENALTY', 0.015),
                idle_penalty=getattr(config, 'PREDISPATCH_IDLE_PENALTY', 0.8),
                congestion_penalty=getattr(config, 'PREDISPATCH_CONGESTION_PENALTY', 0.35),
                remote_worker_bonus=getattr(config, 'PREDISPATCH_REMOTE_WORKER_BONUS', 0.03),
                donor_max_utility_drop=getattr(config, 'GAME_DISPATCH_DONOR_MAX_UTILITY_DROP', 0.04),
                receiver_min_utility_gain=getattr(config, 'GAME_DISPATCH_RECEIVER_MIN_GAIN', 0.01),
                max_iterations=getattr(config, 'GAME_DISPATCH_MAX_ITERATIONS', 120),
                burst_outbound_share=getattr(config, 'GAME_DISPATCH_BURST_OUTBOUND_SHARE', 0.6),
                high_demand_multiplier=getattr(config, 'GAME_DISPATCH_HIGH_DEMAND_MULTIPLIER', 1.25),
                high_demand_shortage_ratio=getattr(config, 'GAME_DISPATCH_HIGH_DEMAND_SHORTAGE_RATIO', 0.3),
                candidate_k=getattr(config, 'GAME_DISPATCH_CANDIDATE_K', 12),
                potential_gain_epsilon=getattr(config, 'GAME_DISPATCH_POTENTIAL_EPSILON', 1e-4)
            )
            game_only_text = ", ".join(
                [
                    f"R{rid}: backlog={backlog_counts.get(rid, 0)}, eff={game_only_result['effective_demand'].get(rid, 0)}"
                    for rid in sorted(centers.keys())
                ]
            )
            print(f"   [NoPred-Game Dispatch] backlog-only plan: {game_only_text}")
            if game_only_result['moves']:
                move_summary = ", ".join(
                    [f"{m['wid']}:{m['from_region']}->{m['to_region']}" for m in game_only_result['moves'][:8]]
                )
                if len(game_only_result['moves']) > 8:
                    move_summary += f", ... (+{len(game_only_result['moves']) - 8} more)"
                print(f"   [NoPred-Game Dispatch] pre-dispatched {len(game_only_result['moves'])} workers: {move_summary}")
            else:
                print("   [NoPred-Game Dispatch] no worker rebalancing needed")

        worker_sim.advance_workers_to_time(centers, slot_start_seconds)

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
            actual_region_demand = {
                rid: int(slot_new_tasks_per_center[rid])
                for rid in centers.keys()
            }
            actual_text = ", ".join(
                [
                    f"R{rid}: actual={slot_new_tasks_per_center[rid]}, "
                    f"abs={abs(current_slot_predicted_demand.get(rid, 0) - slot_new_tasks_per_center[rid])}"
                    for rid in sorted(centers.keys())
                ]
            )
            print(f"   [{current_predict_label}] actual arrivals: {actual_text}")
            if hasattr(dispatch_predictor, 'update_online'):
                dispatch_predictor.update_online(
                    slot_timestamp=slot_timestamp,
                    actual_region_demand=actual_region_demand,
                    predicted_region_demand=current_slot_predicted_demand
                )
            if uncertainty_dispatch_state is not None and algo_name.lower() in PREDICTIVE_UABG_ALGOS:
                uncertainty_dispatch_state.record_prediction_feedback(
                    predicted_region_demand=current_slot_predicted_demand,
                    actual_region_demand=actual_region_demand
                )
            for rid in centers.keys():
                err = float(current_slot_predicted_demand.get(rid, 0) - slot_new_tasks_per_center[rid])
                prediction_abs_errors.append(abs(err))
                prediction_sq_errors.append(err * err)

        if False and algo_name.lower() in ['game_only_dispatch', 'game_only', 'no_pred_game']:
            backlog_counts = {rid: len(unassigned_tasks_pool[rid]) for rid in centers.keys()}
            displayed_plan_demand = {
                rid: int(slot_new_tasks_per_center[rid])
                for rid in centers.keys()
            }
            game_only_result = game_theoretic_predispatch_workers(
                G=G,
                worker_sim=worker_sim,
                centers=centers,
                predicted_demand=displayed_plan_demand,
                next_slot_start_seconds=slot_start_seconds,
                max_tasks_per_worker=getattr(config, 'MAX_TASKS_PER_WORKER', 4),
                backlog_counts=backlog_counts,
                backlog_weight=getattr(config, 'PREDISPATCH_BACKLOG_WEIGHT', 1.0),
                min_buffer_workers=getattr(config, 'PREDISPATCH_MIN_BUFFER_WORKERS', 3),
                reserve_ratio=getattr(config, 'PREDISPATCH_RESERVE_RATIO', 0.15),
                max_rebalance_share=getattr(config, 'PREDISPATCH_MAX_SHARE_PER_DONOR', 0.35),
                max_distance_km=getattr(config, 'PREDISPATCH_MAX_DISTANCE_KM', None),
                fairness_weight=getattr(config, 'GAME_DISPATCH_FAIRNESS_WEIGHT', 0.5),
                distance_penalty=getattr(config, 'GAME_DISPATCH_DISTANCE_PENALTY', 0.015),
                idle_penalty=getattr(config, 'PREDISPATCH_IDLE_PENALTY', 0.8),
                congestion_penalty=getattr(config, 'PREDISPATCH_CONGESTION_PENALTY', 0.35),
                remote_worker_bonus=getattr(config, 'PREDISPATCH_REMOTE_WORKER_BONUS', 0.03),
                donor_max_utility_drop=getattr(config, 'GAME_DISPATCH_DONOR_MAX_UTILITY_DROP', 0.04),
                receiver_min_utility_gain=getattr(config, 'GAME_DISPATCH_RECEIVER_MIN_GAIN', 0.01),
                max_iterations=getattr(config, 'GAME_DISPATCH_MAX_ITERATIONS', 120),
                burst_outbound_share=getattr(config, 'GAME_DISPATCH_BURST_OUTBOUND_SHARE', 0.6),
                high_demand_multiplier=getattr(config, 'GAME_DISPATCH_HIGH_DEMAND_MULTIPLIER', 1.25),
                high_demand_shortage_ratio=getattr(config, 'GAME_DISPATCH_HIGH_DEMAND_SHORTAGE_RATIO', 0.3),
                candidate_k=getattr(config, 'GAME_DISPATCH_CANDIDATE_K', 12),
                potential_gain_epsilon=getattr(config, 'GAME_DISPATCH_POTENTIAL_EPSILON', 1e-4)
            )
            game_only_text = ", ".join(
                [
                    f"R{rid}: current={displayed_plan_demand.get(rid, 0)}, eff={game_only_result['effective_demand'].get(rid, 0)}"
                    for rid in sorted(centers.keys())
                ]
            )
            print(f"   [Game-Only Dispatch] current-slot demand: {game_only_text}")
            if game_only_result['moves']:
                move_summary = ", ".join(
                    [f"{m['wid']}:{m['from_region']}->{m['to_region']}" for m in game_only_result['moves'][:8]]
                )
                if len(game_only_result['moves']) > 8:
                    move_summary += f", ... (+{len(game_only_result['moves']) - 8} more)"
                print(f"   [Game-Only Dispatch] pre-dispatched {len(game_only_result['moves'])} workers: {move_summary}")
            else:
                print("   [Game-Only Dispatch] no worker rebalancing needed")

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
        total_slot_tasks_per_center = {rid: len(tasks_per_center[rid]) for rid in centers.keys()}

        if total_current_tasks == 0:
            if uncertainty_dispatch_state is not None and algo_name.lower() in PREDICTIVE_UABG_ALGOS:
                uncertainty_dispatch_state.record_service_outcome(
                    total_tasks_by_region=total_slot_tasks_per_center,
                    assigned_tasks_by_region={rid: 0 for rid in centers.keys()}
                )
            print("本时段无订单，跳过分配。")
            continue

        # 4.3 执行调度分配算法
        if algo_name.lower() in ['greedy', 'predictive_greedy']:
            slot_assignments, slot_profit, slot_details = greedy_assignment_with_center_pickup(
                G=G, config=config, centers=centers, partition=rcc_partition,
                workers_per_center=workers_per_center, tasks_per_center=tasks_per_center,
                slot_start_seconds=slot_start_seconds  # 💡 补全了这个漏掉的参数！
            )
        elif algo_name.lower() in ROUTE_ILP_ASSIGNMENT_ALGOS:
            slot_assignments, slot_profit, slot_details = enhanced_imtao_assignment_with_center_pickup(
                G=G, config=config, centers_dict=centers,
                workers_per_center=workers_per_center, tasks_per_center=tasks_per_center,
                slot_start_seconds=slot_start_seconds,
                slot_end_seconds=slot_end_seconds,
            )
        elif algo_name.lower() in ['imtao', 'imtao_seq_bdc', 'seq_bdc']:
            slot_assignments, slot_profit, slot_details = run_imtao_for_slot(
                G=G, config=config, centers_dict=centers,
                workers_per_center=workers_per_center, tasks_per_center=tasks_per_center,
                slot_start_seconds=slot_start_seconds,
                collaboration_mode=IMTAO_MODE_BDC,
                center_selection=IMTAO_SELECT_LOWEST_RHO
            )
        elif algo_name.lower() in ['imtao_seq_rbdc', 'seq_rbdc', 'imtao_rbdc']:
            slot_assignments, slot_profit, slot_details = run_imtao_for_slot(
                G=G, config=config, centers_dict=centers,
                workers_per_center=workers_per_center, tasks_per_center=tasks_per_center,
                slot_start_seconds=slot_start_seconds,
                collaboration_mode=IMTAO_MODE_BDC,
                center_selection=IMTAO_SELECT_RANDOM
            )
        elif algo_name.lower() in ['imtao_seq_dc', 'seq_dc', 'imtao_dc']:
            slot_assignments, slot_profit, slot_details = run_imtao_for_slot(
                G=G, config=config, centers_dict=centers,
                workers_per_center=workers_per_center, tasks_per_center=tasks_per_center,
                slot_start_seconds=slot_start_seconds,
                collaboration_mode=IMTAO_MODE_DC,
                center_selection=IMTAO_SELECT_LOWEST_RHO
            )
        elif algo_name.lower() in ['imtao_seq_wo_c', 'seq_wo_c', 'imtao_wo_c', 'imtao_no_collab']:
            slot_assignments, slot_profit, slot_details = run_imtao_for_slot(
                G=G, config=config, centers_dict=centers,
                workers_per_center=workers_per_center, tasks_per_center=tasks_per_center,
                slot_start_seconds=slot_start_seconds,
                collaboration_mode=IMTAO_MODE_WO_C,
                center_selection=IMTAO_SELECT_LOWEST_RHO
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
        slot_assigned_tasks_per_center = {rid: 0 for rid in centers.keys()}
        for detail in slot_details:
            slot_assigned_tasks_per_center[detail['region_id']] += 1
        if uncertainty_dispatch_state is not None and algo_name.lower() in PREDICTIVE_UABG_ALGOS:
            uncertainty_dispatch_state.record_service_outcome(
                total_tasks_by_region=total_slot_tasks_per_center,
                assigned_tasks_by_region=slot_assigned_tasks_per_center
            )

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
    total_tasks_in_scope = int(sum(total_tasks_per_center.values()))
    task_completion_rate = (len(all_assignments) / total_tasks_in_scope) if total_tasks_in_scope > 0 else 0.0

    print(f"\n[{algo_name.upper()}] 仿真完成:")
    print(f"  - #Assigned Tasks: {len(all_assignments)}")
    print(f"  - Task Completion Rate: {task_completion_rate:.4f}")
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
        'task_completion_rate': task_completion_rate,
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

    game_only_assignments, game_only_details, game_only_metrics = run_online_simulation_with_center_pickup(
        algo_name='game_only_dispatch',
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
    uabg_assignments, uabg_details, uabg_metrics = run_online_simulation_with_center_pickup(
        algo_name='predictive_uabg_mctgnet',
        test_date=DEFAULT_TEST_DATE,
        test_start_hour=DEFAULT_START_HOUR,
        test_end_hour=DEFAULT_END_HOUR,
        time_slot_minutes=DEFAULT_TIME_SLOT_MINUTES
    )

    # 3. 打印与论文一致的评价指标
    print("\n\n" + "=" * 142)
    print("论文指标对齐：多中心调度算法横向对比")
    print("=" * 142)
    def _fmt_optional_metric(value):
        return f"{value:.4f}" if value is not None else "-"

    print(
        f"{'指标':<25} | {'Greedy':<12} | {'IMTAO':<12} | {'Game-Only':<12} | "
        f"{'Predictive-MCTGNet':<20} | {'Game-MCTGNet':<16} | {'UABG-MCTGNet':<16}"
    )
    print("-" * 142)
    print(
        f"{'#Assigned Tasks':<25} | {greedy_metrics['assigned_tasks']:<12} | "
        f"{imtao_metrics['assigned_tasks']:<12} | {game_only_metrics['assigned_tasks']:<12} | {predictive_metrics['assigned_tasks']:<20} | "
        f"{game_predictive_metrics['assigned_tasks']:<16} | {uabg_metrics['assigned_tasks']:<16}"
    )
    print(
        f"{'Task Completion Rate':<25} | {greedy_metrics['task_completion_rate']:<12.4f} | "
        f"{imtao_metrics['task_completion_rate']:<12.4f} | {game_only_metrics['task_completion_rate']:<12.4f} | "
        f"{predictive_metrics['task_completion_rate']:<20.4f} | {game_predictive_metrics['task_completion_rate']:<16.4f} | "
        f"{uabg_metrics['task_completion_rate']:<16.4f}"
    )
    print(
        f"{'Collaboration Unfairness':<25} | {greedy_metrics['u_rho']:<12.4f} | "
        f"{imtao_metrics['u_rho']:<12.4f} | {game_only_metrics['u_rho']:<12.4f} | {predictive_metrics['u_rho']:<20.4f} | "
        f"{game_predictive_metrics['u_rho']:<16.4f} | {uabg_metrics['u_rho']:<16.4f}"
    )
    print(
        f"{'CPU Time (s)':<25} | {greedy_metrics['cpu_time']:<12.4f} | "
        f"{imtao_metrics['cpu_time']:<12.4f} | {game_only_metrics['cpu_time']:<12.4f} | {predictive_metrics['cpu_time']:<20.4f} | "
        f"{game_predictive_metrics['cpu_time']:<16.4f} | {uabg_metrics['cpu_time']:<16.4f}"
    )
    print(
        f"{'Prediction MAE':<25} | {_fmt_optional_metric(greedy_metrics.get('pred_mae')):<12} | "
        f"{_fmt_optional_metric(imtao_metrics.get('pred_mae')):<12} | {_fmt_optional_metric(game_only_metrics.get('pred_mae')):<12} | "
        f"{_fmt_optional_metric(predictive_metrics.get('pred_mae')):<20} | "
        f"{_fmt_optional_metric(game_predictive_metrics.get('pred_mae')):<16} | "
        f"{_fmt_optional_metric(uabg_metrics.get('pred_mae')):<16}"
    )
    print(
        f"{'Prediction RMSE':<25} | {_fmt_optional_metric(greedy_metrics.get('pred_rmse')):<12} | "
        f"{_fmt_optional_metric(imtao_metrics.get('pred_rmse')):<12} | {_fmt_optional_metric(game_only_metrics.get('pred_rmse')):<12} | "
        f"{_fmt_optional_metric(predictive_metrics.get('pred_rmse')):<20} | "
        f"{_fmt_optional_metric(game_predictive_metrics.get('pred_rmse')):<16} | "
        f"{_fmt_optional_metric(uabg_metrics.get('pred_rmse')):<16}"
    )
    print("=" * 142)
