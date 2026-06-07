import time
import copy
import os
import math
import contextlib
import io
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import networkx as nx
import torch
from scipy.spatial import KDTree

import config
from algorithm.Greedy import greedy_assignment_with_center_pickup
from algorithm.CenterPrepackedAssignment import center_prepacked_assignment_with_center_pickup
from algorithm.EventTaskRLGameDispatch import (
    CenterTaskAllocationRLState,
    event_task_rl_game_predispatch_workers,
    update_center_task_allocation_state,
)
from algorithm.PredictiveDispatch import predict_next_slot_demand, predispatch_workers_for_next_slot
from algorithm.GameTheoreticPredictiveDispatch import game_theoretic_predispatch_workers
from algorithm.UncertaintyAwareBilateralDispatch import (
    UncertaintyAwareBilateralState,
    uncertainty_aware_bilateral_predispatch_workers,
)
from algorithm.RLRetentionGameDispatch import (
    PlatformTaskFirstRLState,
    RLRetentionBilateralState,
    hydrate_platform_transition_with_next_state,
    hydrate_retention_transitions_with_next_state,
    offline_warm_start_retention_policy,
    rl_retention_bilateral_predispatch_workers,
    sample_platform_task_first_control,
    update_platform_task_first_state,
    update_rl_retention_bilateral_state,
)
from predicate.MCTGNetDispatchPredictor import MCTGNetDispatchPredictor
from predicate.CenterPatternLSTMDispatchPredictor import CenterPatternLSTMDispatchPredictor
from predicate.EventTaskDispatchPredictor import EventTaskDispatchPredictor
from tool.TaskWorkerToMap import WorkerSimulator
from tool.data_loader import get_real_road_network
from tool.map_algorithms import run_kmeans_baseline, run_rcc_algorithm, find_region_centers

from algorithm.IMTAO import (
    Center as IMTAOCenter,
    IMTAO_Framework,
    IMTAO_MODE_BDC,
    IMTAO_MODE_RBDC,
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
_EVENT_TASK_PREDICTOR_CACHE = {}
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
NO_PRED_RBG_ALGOS = {
    'no_pred_rl_game',
    'game_only_rl_dispatch',
    'game_only_rl',
}
PREDICTIVE_PLATFORM_RL_ALGOS = {
    'predictive_platform_rl_mctgnet',
    'platform_rl_mctgnet',
    'predictive_taskfirst_platform_mctgnet',
    'predictive_event_rl_game',
    'event_task_rl_game',
}
PREDICTIVE_EVENT_RL_GAME_ALGOS = {
    'predictive_event_rl_game',
    'event_task_rl_game',
}
NO_PRED_GAME_ALGOS = {'game_only_dispatch', 'game_only', 'no_pred_game'}
RETENTION_RL_ALGOS = {
    *PREDICTIVE_RBG_ALGOS,
    *NO_PRED_RBG_ALGOS,
    *PREDICTIVE_PLATFORM_RL_ALGOS,
}
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
    *NO_PRED_RBG_ALGOS,
    *PREDICTIVE_PLATFORM_RL_ALGOS,
    *NO_PRED_GAME_ALGOS,
}


def _should_force_mctgnet_cpu() -> bool:
    env_value = os.environ.get("MCTGNET_DISPATCH_FORCE_CPU")
    if env_value is not None:
        return env_value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(getattr(config, 'MCTGNET_DISPATCH_FORCE_CPU', False))


def _get_preferred_torch_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    preferred = str(getattr(config, "TORCH_DEVICE_PREFERENCE", "cuda")).strip().lower()
    if preferred.startswith("cuda") and torch.cuda.is_available():
        return preferred
    return "cpu"


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


def _current_scope_metadata():
    download_dist_m = int(config.DOWNLOAD_DIST)
    worker_speed_ms = float(getattr(config, 'WORKER_SPEED_MS', 0.0))
    worker_speed_kmh = float(getattr(config, 'WORKER_SPEED_KMH', worker_speed_ms * 3.6))
    return {
        'map_center_lat': float(config.CHENGDU_CENTER[0]),
        'map_center_lon': float(config.CHENGDU_CENTER[1]),
        'download_dist_m': download_dist_m,
        'map_download_dist_m': download_dist_m,
        'map_size_km': 2.0 * download_dist_m / 1000.0,
        'map_side_km': 2.0 * download_dist_m / 1000.0,
        'center_count': int(config.NUM_ZONES),
        'worker_speed_kmh': worker_speed_kmh,
        'worker_speed_ms': worker_speed_ms,
    }


def _print_scope_metadata(scope):
    expected_speed_ms = scope['worker_speed_kmh'] / 3.6
    speed_note = ""
    if abs(expected_speed_ms - scope['worker_speed_ms']) > 1e-6:
        speed_note = " [WARNING: WORKER_SPEED_KMH and WORKER_SPEED_MS differ]"
    print(
        "   - Scope: "
        f"center=({scope['map_center_lat']:.5f}, {scope['map_center_lon']:.5f}), "
        f"dist={scope['download_dist_m']}m, "
        f"approx_map={scope['map_size_km']:.1f}km x {scope['map_size_km']:.1f}km, "
        f"centers={scope['center_count']}, "
        f"worker_speed={scope['worker_speed_kmh']:.2f}km/h ({scope['worker_speed_ms']:.4f}m/s)"
        f"{speed_note}"
    )


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


def _record_prefix_action_stats(
        stats_by_action: Dict[int, Dict[str, float]],
        action_idx: int,
        reward: float,
        served: float,
):
    bucket = stats_by_action.setdefault(
        int(action_idx),
        {
            'count': 0,
            'reward_sum': 0.0,
            'served_sum': 0.0,
        }
    )
    bucket['count'] += 1
    bucket['reward_sum'] += float(reward)
    bucket['served_sum'] += float(served)


def _summarize_prefix_action_stats(
        action_ratios: Tuple[float, ...],
        global_action_stats: Dict[int, Dict[str, float]],
        region_action_stats: Dict[int, Dict[int, Dict[str, float]]],
        action_labels: Optional[Tuple[str, ...]] = None,
):
    global_summary = []
    for action_idx, action_ratio in enumerate(action_ratios):
        bucket = global_action_stats.get(action_idx, {})
        count = int(bucket.get('count', 0))
        if count <= 0:
            continue
        global_summary.append(
            {
                'action_idx': int(action_idx),
                'action_ratio': float(action_ratio),
                'action_label': (
                    str(action_labels[action_idx])
                    if action_labels is not None and action_idx < len(action_labels)
                    else f"{float(action_ratio):+.2f}"
                ),
                'count': count,
                'avg_reward': float(bucket.get('reward_sum', 0.0)) / count,
                'avg_served': float(bucket.get('served_sum', 0.0)) / count,
            }
        )

    dominant_region_actions = {}
    for rid, action_map in region_action_stats.items():
        best_entry = None
        for action_idx, bucket in action_map.items():
            count = int(bucket.get('count', 0))
            if count <= 0:
                continue
            avg_reward = float(bucket.get('reward_sum', 0.0)) / count
            candidate = (
                count,
                avg_reward,
                -abs(float(action_ratios[action_idx])),
                int(action_idx),
            )
            if best_entry is None or candidate > best_entry[0]:
                best_entry = (
                    candidate,
                    {
                        'action_idx': int(action_idx),
                        'action_ratio': float(action_ratios[action_idx]),
                        'action_label': (
                            str(action_labels[action_idx])
                            if action_labels is not None and action_idx < len(action_labels)
                            else f"{float(action_ratios[action_idx]):+.2f}"
                        ),
                        'count': count,
                        'avg_reward': avg_reward,
                        'avg_served': float(bucket.get('served_sum', 0.0)) / count,
                    }
                )
        if best_entry is not None:
            dominant_region_actions[int(rid)] = best_entry[1]

    return global_summary, dominant_region_actions


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
        platform_state=None,
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
    prefix_micro_batch_minutes = int(getattr(config, 'RBG_PREFIX_MICRO_BATCH_MINUTES', time_slot_minutes))
    prefix_micro_batch_minutes = max(1, min(int(time_slot_minutes), prefix_micro_batch_minutes))
    micro_batch_seconds = prefix_micro_batch_minutes * 60
    slot_duration_seconds = int(time_slot_minutes) * 60
    commit_next_step_only = bool(getattr(config, 'ONLINE_COMMIT_ONE_TASK_AT_A_TIME', True))
    batch_fine_tune_enabled = bool(getattr(config, 'RBG_BATCH_ONLINE_FINE_TUNE', True))
    fast_proxy_enabled = bool(getattr(config, 'RBG_PREFIX_FAST_PROXY', True))
    global_action_stats = {}
    region_action_stats = {}

    def _record_retention_update(transitions, reward_by_region, served_by_region):
        return None

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

        slot_total_tasks_per_center = {rid: len(unassigned_tasks_pool[rid]) for rid in centers.keys()}
        slot_new_tasks_per_center = {rid: 0 for rid in centers.keys()}
        slot_assigned_tasks_per_center = {rid: 0 for rid in centers.keys()}
        last_rbg_reward_by_region = None
        last_platform_stats = None

        current_slot_platform_transition = None
        current_slot_platform_fairness_weight = 0.0
        platform_task_weight = getattr(config, 'RBG_PLATFORM_TASK_WEIGHT', 0.30)
        platform_gap_weight = getattr(config, 'RBG_PLATFORM_GAP_WEIGHT', 0.55)
        platform_release_credit_weight = getattr(config, 'RBG_PLATFORM_RELEASE_CREDIT_WEIGHT', 0.35)
        platform_keep_scale = 1.0
        platform_need_scale = 1.0
        platform_move_share_scale = 1.0
        platform_slot_start_blend_scale = 1.0

        slot_start_available_workers = {
            rid: len(prefix_worker_sim.get_available_workers_with_center_info(rid, current_time=slot_start_seconds))
            for rid in centers.keys()
        }
        if platform_state is not None:
            current_slot_platform_transition = sample_platform_task_first_control(
                region_ids=sorted(centers.keys()),
                predicted_demand=slot_actual_counts,
                backlog_counts=backlog_counts,
                available_workers=slot_start_available_workers,
                max_tasks_per_worker=getattr(config, 'MAX_TASKS_PER_WORKER', 4),
                retention_state=state,
                platform_state=platform_state,
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
            current_slot_platform_fairness_weight = float(current_slot_platform_transition.get('fairness_weight', 0.0))
            platform_keep_scale = float(current_slot_platform_transition.get('keep_scale', 1.0))
            platform_need_scale = float(current_slot_platform_transition.get('need_scale', 1.0))
            platform_move_share_scale = float(current_slot_platform_transition.get('move_share_scale', 1.0))
            platform_slot_start_blend_scale = float(current_slot_platform_transition.get('slot_start_blend_scale', 1.0))

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
            platform_task_weight=platform_task_weight,
            platform_gap_weight=platform_gap_weight,
            platform_release_credit_weight=platform_release_credit_weight,
            platform_fairness_weight=current_slot_platform_fairness_weight,
            platform_keep_scale=platform_keep_scale,
            platform_need_scale=platform_need_scale,
            platform_move_share_scale=platform_move_share_scale,
            platform_slot_start_blend_scale=platform_slot_start_blend_scale,
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
        current_slot_rbg_demand_profile = predispatch_result.get('demand_profile', {})
        current_slot_rbg_desired_workers = predispatch_result.get('desired_workers', {})
        _prepare_and_log_opportunistic_support_tasks(
            G=G,
            worker_sim=prefix_worker_sim,
            centers=centers,
            unassigned_tasks_pool=unassigned_tasks_pool,
            moves=current_slot_rbg_moves,
            current_time=slot_start_seconds,
            stackelberg_control=current_slot_rbg_stackelberg_control,
            label='RBG-Prefix 顺路支援',
        )

        if fast_proxy_enabled:
            max_tasks = max(1, int(getattr(config, 'MAX_TASKS_PER_WORKER', 4)))
            proxy_total_tasks = {
                rid: int(backlog_counts.get(rid, 0)) + int(slot_actual_counts.get(rid, 0))
                for rid in centers.keys()
            }
            retain_count = predispatch_result.get('retain_count', {})
            proxy_assigned_tasks = {
                rid: min(
                    proxy_total_tasks[rid],
                    max(0, int(retain_count.get(rid, 0))) * max_tasks,
                )
                for rid in centers.keys()
            }
            backlog_counts = {
                rid: max(0, proxy_total_tasks[rid] - proxy_assigned_tasks[rid])
                for rid in centers.keys()
            }
            next_available_workers = {
                rid: max(0, int(retain_count.get(rid, 0)))
                for rid in centers.keys()
            }
            if current_slot_rbg_transitions:
                hydrate_retention_transitions_with_next_state(
                    state=state,
                    transitions=current_slot_rbg_transitions,
                    demand_profile=current_slot_rbg_demand_profile,
                    desired_workers=current_slot_rbg_desired_workers,
                    available_workers=next_available_workers,
                    backlog_counts=backlog_counts,
                    max_tasks_per_worker=max_tasks,
                    min_buffer_workers=int(getattr(config, 'UABG_MIN_BUFFER_WORKERS', 1)),
                    backlog_weight=float(getattr(config, 'UABG_BACKLOG_WEIGHT', 1.0)),
                    done=float(local_slot_idx == slot_count - 1),
                )
                update_rl_retention_bilateral_state(
                    state=state,
                    transitions=current_slot_rbg_transitions,
                    assigned_tasks_by_region=proxy_assigned_tasks,
                    total_tasks_by_region=proxy_total_tasks,
                    hoard_penalty_by_region=current_slot_rbg_hoard_penalty,
                    move_cost_by_region=current_slot_rbg_move_cost,
                    moves=current_slot_rbg_moves,
                    hoard_penalty_weight=float(getattr(config, 'RBG_REWARD_HOARD_WEIGHT', 0.02)),
                    move_cost_weight=float(getattr(config, 'RBG_REWARD_MOVE_WEIGHT', 0.08)),
                    unfairness_weight=float(getattr(config, 'RBG_REWARD_UNFAIRNESS_WEIGHT', 1.0)),
                )
            if platform_state is not None and current_slot_platform_transition is not None:
                hydrate_platform_transition_with_next_state(
                    state=platform_state,
                    transition=current_slot_platform_transition,
                    available_workers=next_available_workers,
                    backlog_counts=backlog_counts,
                    max_tasks_per_worker=max_tasks,
                    backlog_weight=float(getattr(config, 'UABG_BACKLOG_WEIGHT', 1.0)),
                    done=float(local_slot_idx == slot_count - 1),
                )
                update_platform_task_first_state(
                    state=platform_state,
                    transition=current_slot_platform_transition,
                    assigned_tasks_by_region=proxy_assigned_tasks,
                    total_tasks_by_region=proxy_total_tasks,
                    fairness_secondary_weight=current_slot_platform_fairness_weight,
                )
            for rid in centers.keys():
                rolling_history[rid].append(slot_actual_counts[rid])
            state.record_prediction_feedback(
                predicted_region_demand=slot_actual_counts,
                actual_region_demand=slot_actual_counts,
            )
            continue

        algo_name = 'predictive_platform_rl_mctgnet' if platform_state is not None else 'predictive_rl_game_mctgnet'
        last_micro_redispatch_idx = 0
        redispatch_gap = _resolve_micro_redispatch_gap_batches(
            algo_name='predictive_rl_game_mctgnet',
            micro_batch_seconds=micro_batch_seconds,
        )
        total_micro = len(range(slot_start_seconds, slot_end_seconds, micro_batch_seconds))

        for micro_idx, micro_start_seconds in enumerate(range(slot_start_seconds, slot_end_seconds, micro_batch_seconds)):
            micro_end_seconds = min(slot_end_seconds, micro_start_seconds + micro_batch_seconds)
            prefix_worker_sim.advance_workers_to_time(centers, micro_start_seconds)

            micro_new_tasks = 0
            micro_assignments = {}
            micro_details = []
            micro_assigned_tasks_per_center = {rid: 0 for rid in centers.keys()}
            micro_total_tasks_per_center = {rid: len(unassigned_tasks_pool[rid]) for rid in centers.keys()}

            workers_per_center = {}
            for rid in centers.keys():
                workers = prefix_worker_sim.get_available_workers_with_center_info(rid, current_time=micro_start_seconds)
                workers_per_center[rid] = [(w[0], w[1], w[2], w[3], centers[rid]) for w in workers]

            if (
                total_micro > 1
                and micro_idx > 0
                and (micro_idx - last_micro_redispatch_idx) >= redispatch_gap
            ):
                micro_dispatch_result = _run_triggered_micro_predispatch(
                    algo_name=algo_name,
                    G=G,
                    worker_sim=prefix_worker_sim,
                    centers=centers,
                    current_time=micro_start_seconds,
                    slot_idx=local_slot_idx,
                    micro_idx=micro_idx,
                    slot_start_seconds=slot_start_seconds,
                    slot_end_seconds=slot_end_seconds,
                    current_slot_predicted_demand=slot_actual_counts,
                    slot_new_tasks_per_center=slot_new_tasks_per_center,
                    unassigned_tasks_pool=unassigned_tasks_pool,
                    uncertainty_dispatch_state=None,
                    dispatch_predictor=None,
                    retention_game_state=state,
                    platform_rl_state=platform_state,
                )
                if micro_dispatch_result is not None:
                    last_micro_redispatch_idx = micro_idx
                    current_slot_rbg_transitions = micro_dispatch_result.get('transitions', current_slot_rbg_transitions)
                    current_slot_rbg_hoard_penalty = micro_dispatch_result.get('hoard_penalty', current_slot_rbg_hoard_penalty)
                    current_slot_rbg_move_cost = micro_dispatch_result.get('move_cost_by_region', current_slot_rbg_move_cost)
                    current_slot_rbg_moves = micro_dispatch_result.get('moves', current_slot_rbg_moves)
                    current_slot_rbg_stackelberg_control = micro_dispatch_result.get('stackelberg_control', current_slot_rbg_stackelberg_control)
                    current_slot_rbg_demand_profile = micro_dispatch_result.get('demand_profile', current_slot_rbg_demand_profile)
                    current_slot_rbg_desired_workers = micro_dispatch_result.get('desired_workers', current_slot_rbg_desired_workers)
                    if platform_state is not None:
                        current_slot_platform_transition = micro_dispatch_result.get('platform_transition', current_slot_platform_transition)
                    _prepare_and_log_opportunistic_support_tasks(
                        G=G,
                        worker_sim=prefix_worker_sim,
                        centers=centers,
                        unassigned_tasks_pool=unassigned_tasks_pool,
                        moves=current_slot_rbg_moves,
                        current_time=micro_start_seconds,
                        stackelberg_control=current_slot_rbg_stackelberg_control,
                        label='RBG-Prefix-Micro 顺路支援',
                    )
                    for rid in centers.keys():
                        workers = prefix_worker_sim.get_available_workers_with_center_info(rid, current_time=micro_start_seconds)
                        workers_per_center[rid] = [(w[0], w[1], w[2], w[3], centers[rid]) for w in workers]

            def run_prefix_visible_assignment(decision_time, force=False):
                _release_stale_pending_support_transfers(
                    prefix_worker_sim,
                    unassigned_tasks_pool,
                    decision_time,
                )
                workers_snapshot = {}
                total_workers = 0
                for rid in centers.keys():
                    workers = prefix_worker_sim.get_available_workers_with_center_info(rid, current_time=decision_time)
                    workers_snapshot[rid] = [(w[0], w[1], w[2], w[3], centers[rid]) for w in workers]
                    total_workers += len(workers)

                total_current_tasks = sum(len(unassigned_tasks_pool[rid]) for rid in centers.keys())
                if total_current_tasks <= 0 or total_workers <= 0:
                    return 0
                if (
                    not force
                    and _should_defer_online_dispatch(
                        algo_key=algo_name,
                        unassigned_tasks_pool=unassigned_tasks_pool,
                        total_workers=total_workers,
                        current_time=decision_time,
                    )
                ):
                    return 0

                tasks_per_center = _build_microbatch_candidate_tasks(
                    unassigned_tasks_pool=unassigned_tasks_pool,
                    workers_per_center=workers_snapshot,
                    current_time=decision_time,
                    slot_end_seconds=slot_end_seconds,
                )
                assignment_kwargs = dict(
                    algo_name=algo_name,
                    G=G,
                    config=config,
                    centers=centers,
                    rcc_partition=rcc_partition,
                    workers_per_center=workers_snapshot,
                    tasks_per_center=tasks_per_center,
                    slot_start_seconds=decision_time,
                    slot_end_seconds=slot_end_seconds,
                    stackelberg_control=current_slot_rbg_stackelberg_control,
                    force_center_pickup_on_first_departure=_online_force_center_pickup(commit_next_step_only),
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    event_assignments, _, event_details = _run_assignment_for_window(**assignment_kwargs)
                if commit_next_step_only:
                    event_assignments, _, event_details = _reduce_microbatch_results_for_online_replanning(
                        micro_assignments=event_assignments,
                        micro_details=event_details,
                        commit_horizon_seconds=decision_time,
                    )
                _apply_assignment_results_to_workers(
                    G,
                    prefix_worker_sim,
                    event_details,
                    commit_service_only=commit_next_step_only,
                )
                micro_assignments.update(event_assignments)
                micro_details.extend(event_details)
                for detail in event_details:
                    slot_assigned_tasks_per_center[detail['region_id']] += 1
                    micro_assigned_tasks_per_center[detail['region_id']] += 1

                assigned_task_ids = {k[1] for k in event_assignments.keys()}
                if not assigned_task_ids:
                    return 0
                for rid in centers.keys():
                    new_pool = []
                    for t in unassigned_tasks_pool[rid]:
                        if t[1] in assigned_task_ids:
                            continue
                        if decision_time >= t[3]:
                            continue
                        new_pool.append(t)
                    unassigned_tasks_pool[rid] = new_pool
                    backlog_counts[rid] = len(new_pool)
                return len(assigned_task_ids)

            run_prefix_visible_assignment(micro_start_seconds)

            if not slot_df.empty:
                current_tasks = slot_df[
                    (slot_df['seconds_of_day'] >= micro_start_seconds)
                    & (slot_df['seconds_of_day'] < micro_end_seconds)
                ].sort_values(['seconds_of_day', 'task_id'])
                for release_time, event_rows in current_tasks.groupby('seconds_of_day', sort=True):
                    decision_time = float(release_time)
                    prefix_worker_sim.advance_workers_to_time(centers, decision_time)
                    event_new_tasks = 0
                    for _, row in event_rows.iterrows():
                        rid = int(row['region_id'])
                        task = (
                            row['nearest_node'],
                            row['task_id'],
                            config.TASK_BASE_REWARD,
                            decision_time + config.TASK_EXPIRE_MINUTES * 60,
                            decision_time,
                        )
                        unassigned_tasks_pool[rid].append(task)
                        slot_new_tasks_per_center[rid] += 1
                        slot_total_tasks_per_center[rid] += 1
                        micro_total_tasks_per_center[rid] += 1
                        micro_new_tasks += 1
                        event_new_tasks += 1
                    if event_new_tasks > 0:
                        run_prefix_visible_assignment(decision_time)

            prefix_worker_sim.advance_workers_to_time(centers, micro_end_seconds)
            run_prefix_visible_assignment(micro_end_seconds)

            for rid in centers.keys():
                new_pool = []
                for t in unassigned_tasks_pool[rid]:
                    if micro_end_seconds >= t[3]:
                        continue
                    new_pool.append(t)
                unassigned_tasks_pool[rid] = new_pool
                backlog_counts[rid] = len(new_pool)

            if (
                batch_fine_tune_enabled
                and total_micro > 1
                and current_slot_rbg_transitions
                and sum(micro_total_tasks_per_center.values()) > 0
            ):
                next_available_workers = {
                    rid: len(prefix_worker_sim.get_available_workers_with_center_info(rid, current_time=micro_end_seconds))
                    for rid in centers.keys()
                }
                hydrate_retention_transitions_with_next_state(
                    state=state,
                    transitions=current_slot_rbg_transitions,
                    demand_profile=current_slot_rbg_demand_profile,
                    desired_workers=current_slot_rbg_desired_workers,
                    available_workers=next_available_workers,
                    backlog_counts={rid: len(unassigned_tasks_pool[rid]) for rid in centers.keys()},
                    max_tasks_per_worker=int(getattr(config, 'MAX_TASKS_PER_WORKER', 4)),
                    min_buffer_workers=int(getattr(config, 'UABG_MIN_BUFFER_WORKERS', 1)),
                    backlog_weight=float(getattr(config, 'UABG_BACKLOG_WEIGHT', 1.0)),
                    done=float(local_slot_idx == slot_count - 1 and micro_idx == total_micro - 1),
                )
                last_rbg_reward_by_region = update_rl_retention_bilateral_state(
                    state=state,
                    transitions=current_slot_rbg_transitions,
                    assigned_tasks_by_region=micro_assigned_tasks_per_center,
                    total_tasks_by_region=micro_total_tasks_per_center,
                    hoard_penalty_by_region=current_slot_rbg_hoard_penalty,
                    move_cost_by_region=current_slot_rbg_move_cost,
                    moves=current_slot_rbg_moves,
                    hoard_penalty_weight=float(getattr(config, 'RBG_REWARD_HOARD_WEIGHT', 0.02)),
                    move_cost_weight=float(getattr(config, 'RBG_REWARD_MOVE_WEIGHT', 0.08)),
                    unfairness_weight=float(getattr(config, 'RBG_REWARD_UNFAIRNESS_WEIGHT', 1.0)),
                )
                _record_retention_update(
                    transitions=current_slot_rbg_transitions,
                    reward_by_region=last_rbg_reward_by_region,
                    served_by_region=micro_assigned_tasks_per_center,
                )
                if platform_state is not None and current_slot_platform_transition is not None:
                    hydrate_platform_transition_with_next_state(
                        state=platform_state,
                        transition=current_slot_platform_transition,
                        available_workers=next_available_workers,
                        backlog_counts={rid: len(unassigned_tasks_pool[rid]) for rid in centers.keys()},
                        max_tasks_per_worker=int(getattr(config, 'MAX_TASKS_PER_WORKER', 4)),
                        backlog_weight=float(getattr(config, 'UABG_BACKLOG_WEIGHT', 1.0)),
                        done=float(local_slot_idx == slot_count - 1 and micro_idx == total_micro - 1),
                    )
                    last_platform_stats = update_platform_task_first_state(
                        state=platform_state,
                        transition=current_slot_platform_transition,
                        assigned_tasks_by_region=micro_assigned_tasks_per_center,
                        total_tasks_by_region=micro_total_tasks_per_center,
                        fairness_secondary_weight=current_slot_platform_fairness_weight,
                    )

        for rid in centers.keys():
            rolling_history[rid].append(slot_actual_counts[rid])

        state.record_prediction_feedback(
            predicted_region_demand=slot_actual_counts,
            actual_region_demand=slot_actual_counts,
        )
        if current_slot_rbg_transitions and not (batch_fine_tune_enabled and total_micro > 1):
            next_available_workers = {
                rid: len(prefix_worker_sim.get_available_workers_with_center_info(rid, current_time=slot_end_seconds))
                for rid in centers.keys()
            }
            hydrate_retention_transitions_with_next_state(
                state=state,
                transitions=current_slot_rbg_transitions,
                demand_profile=current_slot_rbg_demand_profile,
                desired_workers=current_slot_rbg_desired_workers,
                available_workers=next_available_workers,
                backlog_counts={rid: len(unassigned_tasks_pool[rid]) for rid in centers.keys()},
                max_tasks_per_worker=int(getattr(config, 'MAX_TASKS_PER_WORKER', 4)),
                min_buffer_workers=int(getattr(config, 'UABG_MIN_BUFFER_WORKERS', 1)),
                backlog_weight=float(getattr(config, 'UABG_BACKLOG_WEIGHT', 1.0)),
                done=float(local_slot_idx == slot_count - 1),
            )
            last_rbg_reward_by_region = update_rl_retention_bilateral_state(
                state=state,
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
            _record_retention_update(
                transitions=current_slot_rbg_transitions,
                reward_by_region=last_rbg_reward_by_region,
                served_by_region=slot_assigned_tasks_per_center,
            )
        if platform_state is not None and current_slot_platform_transition is not None and not (batch_fine_tune_enabled and total_micro > 1):
            next_available_workers = {
                rid: len(prefix_worker_sim.get_available_workers_with_center_info(rid, current_time=slot_end_seconds))
                for rid in centers.keys()
            }
            hydrate_platform_transition_with_next_state(
                state=platform_state,
                transition=current_slot_platform_transition,
                available_workers=next_available_workers,
                backlog_counts={rid: len(unassigned_tasks_pool[rid]) for rid in centers.keys()},
                max_tasks_per_worker=int(getattr(config, 'MAX_TASKS_PER_WORKER', 4)),
                backlog_weight=float(getattr(config, 'UABG_BACKLOG_WEIGHT', 1.0)),
                done=float(local_slot_idx == slot_count - 1),
            )
            last_platform_stats = update_platform_task_first_state(
                state=platform_state,
                transition=current_slot_platform_transition,
                assigned_tasks_by_region=slot_assigned_tasks_per_center,
                total_tasks_by_region=slot_total_tasks_per_center,
                fairness_secondary_weight=current_slot_platform_fairness_weight,
            )

    extra_replay_stats = state.offline_replay_train(
        epochs=int(getattr(config, 'RBG_PREFIX_EXTRA_REPLAY_EPOCHS', 0)),
        updates_per_region=int(getattr(config, 'RBG_PREFIX_REPLAY_UPDATES_PER_REGION', 1)),
    )
    platform_extra_replay_stats = {}
    if platform_state is not None:
        platform_extra_replay_stats = platform_state.offline_replay_train(
            epochs=int(getattr(config, 'PFRL_PREFIX_EXTRA_REPLAY_EPOCHS', 0)),
            updates_per_epoch=int(getattr(config, 'PFRL_PREFIX_REPLAY_UPDATES', 1)),
        )
    global_action_summary, dominant_region_actions = [], {}

    return {
        'source': 'same_day_prefix_proxy' if fast_proxy_enabled else 'same_day_prefix_simulation',
        'slot_count': slot_count,
        'worker_count': prefix_worker_count,
        'micro_batch_minutes': prefix_micro_batch_minutes,
        'extra_replay': extra_replay_stats,
        'platform_extra_replay': platform_extra_replay_stats,
        'global_action_summary': global_action_summary,
        'dominant_region_actions': dominant_region_actions,
    }


def _load_event_rl_warm_start_tasks(
        history_date: str,
        test_start_hour: int,
        test_end_hour: int,
        coords,
        nodes,
        rcc_partition,
):
    task_file = os.path.join(config.TASK_DATA_DIR, f"tasks_{history_date}.csv")
    if not os.path.exists(task_file):
        return None

    df = pd.read_csv(
        task_file,
        usecols=['task_id', 'first_time', 'first_lon', 'first_lat'],
    )
    if df.empty:
        return None
    df, _ = _filter_df_by_map_bbox(df, 'first_lon', 'first_lat')
    if df.empty:
        return None

    df['first_time'] = pd.to_datetime(df['first_time'])
    history_day = pd.Timestamp(history_date).normalize()
    df = df[df['first_time'].dt.normalize() == history_day].copy()
    if df.empty:
        return None
    df['seconds_of_day'] = (
        df['first_time'].dt.hour * 3600
        + df['first_time'].dt.minute * 60
        + df['first_time'].dt.second
    )
    start_seconds = int(test_start_hour) * 3600
    end_seconds = int(test_end_hour) * 3600
    df = df[
        (df['seconds_of_day'] >= start_seconds)
        & (df['seconds_of_day'] < end_seconds)
    ].copy()
    if df.empty:
        return None

    _, idxs = KDTree(coords).query(df[['first_lon', 'first_lat']].values)
    df['nearest_node'] = [nodes[i] for i in idxs]
    df = df[df['nearest_node'].isin(rcc_partition)].copy()
    if df.empty:
        return None
    df['region_id'] = df['nearest_node'].map(rcc_partition)
    return df


def _simulate_historical_event_task_rl_warm_start(
        state: RLRetentionBilateralState,
        platform_state,
        center_task_state,
        test_date: str,
        test_start_hour: int,
        test_end_hour: int,
        time_slot_minutes: int,
        G,
        coords,
        nodes,
        rcc_partition,
        centers,
        reference_worker_sim,
):
    if not bool(getattr(config, 'EVENT_RL_WARM_START_USE_TASK_EVENTS', True)):
        return None

    try:
        train_dates, val_dates = build_prediction_date_split(test_date, config.TASK_DATA_DIR)
    except ValueError:
        return None
    history_dates = train_dates + val_dates
    warm_days = max(1, int(getattr(config, 'EVENT_RL_WARM_START_DAYS', 3)))
    history_dates = history_dates[-warm_days:]
    if not history_dates:
        return None

    region_ids = sorted(centers.keys())
    worker_count = len(reference_worker_sim.worker_positions)
    slot_duration_seconds = int(time_slot_minutes) * 60
    start_seconds = int(test_start_hour) * 3600
    slot_count_per_day = max(
        1,
        ((int(test_end_hour) - int(test_start_hour)) * 60) // int(time_slot_minutes),
    )
    completed_days = 0
    completed_slots = 0
    replayed_tasks = 0
    precommit_tasks = 0

    for day_idx, history_date in enumerate(history_dates):
        df = _load_event_rl_warm_start_tasks(
            history_date=history_date,
            test_start_hour=test_start_hour,
            test_end_hour=test_end_hour,
            coords=coords,
            nodes=nodes,
            rcc_partition=rcc_partition,
        )
        if df is None or df.empty:
            continue

        warm_worker_sim = WorkerSimulator(G, config)
        try:
            warm_worker_sim.initialize_from_real_data(
                date=history_date,
                test_start_hour=test_start_hour,
                prep_minutes=FIXED_INIT_PREP_MINUTES,
                coords=coords,
                nodes=nodes,
                partition=rcc_partition,
                centers=centers,
                max_workers=worker_count,
                sampling_mode=DEFAULT_WORKER_SAMPLING_MODE,
                random_seed=DEFAULT_WORKER_SAMPLE_SEED,
            )
        except (FileNotFoundError, ValueError):
            continue

        unassigned_tasks_pool = {rid: [] for rid in region_ids}
        for slot_idx in range(slot_count_per_day):
            slot_start_seconds = start_seconds + slot_idx * slot_duration_seconds
            slot_end_seconds = slot_start_seconds + slot_duration_seconds
            warm_worker_sim.advance_workers_to_time(centers, slot_start_seconds)
            slot_df = df[
                (df['seconds_of_day'] >= slot_start_seconds)
                & (df['seconds_of_day'] < slot_end_seconds)
            ]

            predicted_tasks = {rid: [] for rid in region_ids}
            actual_tasks = {rid: [] for rid in region_ids}
            for _, row in slot_df.iterrows():
                rid = int(row['region_id'])
                release_time = float(row['seconds_of_day'])
                expiry_time = release_time + float(config.TASK_EXPIRE_MINUTES) * 60
                task_id = str(row['task_id'])
                predicted_tasks[rid].append(
                    (
                        row['nearest_node'],
                        f"warm_pred:{history_date}:{task_id}",
                        1.0,
                        expiry_time,
                        release_time,
                    )
                )
                actual_tasks[rid].append(
                    (
                        row['nearest_node'],
                        f"warm_actual:{history_date}:{task_id}",
                        config.TASK_BASE_REWARD,
                        expiry_time,
                        release_time,
                    )
                )

            actual_counts = {rid: len(actual_tasks[rid]) for rid in region_ids}
            predicted_distribution = {
                rid: {
                    'mu': float(actual_counts[rid]),
                    'sigma': 0.0,
                    'q90': float(actual_counts[rid]),
                    'burst_prob': 0.0,
                }
                for rid in region_ids
            }
            dispatch_result = event_task_rl_game_predispatch_workers(
                G=G,
                worker_sim=warm_worker_sim,
                centers=centers,
                predicted_tasks=predicted_tasks,
                backlog_tasks=unassigned_tasks_pool,
                predicted_distribution=predicted_distribution,
                state=state,
                platform_state=platform_state,
                center_task_state=center_task_state,
                slot_idx=day_idx * slot_count_per_day + slot_idx,
                slot_start_seconds=slot_start_seconds,
                slot_end_seconds=slot_end_seconds,
            )
            precommit_tasks += sum(dispatch_result.get('precommit_planned_by_region', {}).values())
            replayed_tasks += sum(actual_counts.values())

            slot_total_tasks = {
                rid: len(unassigned_tasks_pool[rid]) + actual_counts[rid]
                for rid in region_ids
            }
            if bool(getattr(config, 'EVENT_RL_WARM_START_FAST_PROXY', True)):
                max_tasks = max(1, int(getattr(config, 'MAX_TASKS_PER_WORKER', 4)))
                assigned_counts = {rid: 0 for rid in region_ids}
                retain_count = dispatch_result.get('retain_count', {})
                for rid in region_ids:
                    visible_tasks = sorted(
                        list(unassigned_tasks_pool[rid]) + list(actual_tasks[rid]),
                        key=lambda task: (float(task[3]), float(task[4]), str(task[1])),
                    )
                    worker_capacity = max(0, int(retain_count.get(rid, 0))) * max_tasks
                    assigned_counts[rid] = min(len(visible_tasks), worker_capacity)
                    unassigned_tasks_pool[rid] = [
                        task for task in visible_tasks[assigned_counts[rid]:]
                        if slot_end_seconds < task[3]
                    ]

                transitions = dispatch_result.get('transitions', {})
                next_available_workers = {
                    rid: max(0, int(retain_count.get(rid, 0)))
                    for rid in region_ids
                }
                if transitions:
                    hydrate_retention_transitions_with_next_state(
                        state=state,
                        transitions=transitions,
                        demand_profile=dispatch_result.get('demand_profile', {}),
                        desired_workers=dispatch_result.get('desired_workers', {}),
                        available_workers=next_available_workers,
                        backlog_counts={rid: len(unassigned_tasks_pool[rid]) for rid in region_ids},
                        max_tasks_per_worker=max_tasks,
                        min_buffer_workers=int(getattr(config, 'UABG_MIN_BUFFER_WORKERS', 1)),
                        backlog_weight=float(getattr(config, 'UABG_BACKLOG_WEIGHT', 1.0)),
                        done=float(slot_idx == slot_count_per_day - 1),
                    )
                    update_rl_retention_bilateral_state(
                        state=state,
                        transitions=transitions,
                        assigned_tasks_by_region=assigned_counts,
                        total_tasks_by_region=slot_total_tasks,
                        hoard_penalty_by_region=dispatch_result.get('hoard_penalty', {}),
                        move_cost_by_region=dispatch_result.get('move_cost_by_region', {}),
                        moves=dispatch_result.get('moves', []),
                        hoard_penalty_weight=float(getattr(config, 'RBG_REWARD_HOARD_WEIGHT', 0.02)),
                        move_cost_weight=float(getattr(config, 'RBG_REWARD_MOVE_WEIGHT', 0.08)),
                        unfairness_weight=float(getattr(config, 'RBG_REWARD_UNFAIRNESS_WEIGHT', 1.0)),
                    )
                center_transitions = dispatch_result.get('center_transitions', {})
                if center_task_state is not None and center_transitions:
                    update_center_task_allocation_state(
                        state=center_task_state,
                        transitions=center_transitions,
                        assigned_tasks_by_region=assigned_counts,
                        total_tasks_by_region=slot_total_tasks,
                        remaining_tasks_by_region={rid: len(unassigned_tasks_pool[rid]) for rid in region_ids},
                        actual_arrivals_by_region=actual_counts,
                        predicted_demand_by_region={rid: len(predicted_tasks.get(rid, [])) for rid in region_ids},
                    )
                platform_transition = dispatch_result.get('platform_transition')
                if platform_state is not None and platform_transition is not None:
                    hydrate_platform_transition_with_next_state(
                        state=platform_state,
                        transition=platform_transition,
                        available_workers=next_available_workers,
                        backlog_counts={rid: len(unassigned_tasks_pool[rid]) for rid in region_ids},
                        max_tasks_per_worker=max_tasks,
                        backlog_weight=float(getattr(config, 'UABG_BACKLOG_WEIGHT', 1.0)),
                        done=float(slot_idx == slot_count_per_day - 1),
                    )
                    update_platform_task_first_state(
                        state=platform_state,
                        transition=platform_transition,
                        assigned_tasks_by_region=assigned_counts,
                        total_tasks_by_region=slot_total_tasks,
                        fairness_secondary_weight=float(
                            getattr(config, 'PFRL_FAIRNESS_SECONDARY_WEIGHT', 0.20)
                        ),
                    )
                state.record_prediction_feedback(
                    predicted_region_demand=actual_counts,
                    actual_region_demand=actual_counts,
                )
                completed_slots += 1
                continue

            assignments = {}
            details = []

            def run_warm_visible_assignment(decision_time):
                workers_per_center = {}
                for rid in region_ids:
                    workers = warm_worker_sim.get_available_workers_with_center_info(
                        rid,
                        current_time=decision_time,
                    )
                    workers_per_center[rid] = [
                        (w[0], w[1], w[2], w[3], centers[rid]) for w in workers
                    ]
                if (
                    sum(len(unassigned_tasks_pool[rid]) for rid in region_ids) <= 0
                    or sum(len(v) for v in workers_per_center.values()) <= 0
                ):
                    return
                if _should_defer_online_dispatch(
                    algo_key='predictive_event_rl_game',
                    unassigned_tasks_pool=unassigned_tasks_pool,
                    total_workers=sum(len(v) for v in workers_per_center.values()),
                    current_time=decision_time,
                ):
                    return
                tasks_per_center = _build_microbatch_candidate_tasks(
                    unassigned_tasks_pool=unassigned_tasks_pool,
                    workers_per_center=workers_per_center,
                    current_time=decision_time,
                    slot_end_seconds=slot_end_seconds,
                )
                assignment_kwargs = dict(
                    algo_name='predictive_event_rl_game',
                    G=G,
                    config=config,
                    centers=centers,
                    rcc_partition=rcc_partition,
                    workers_per_center=workers_per_center,
                    tasks_per_center=tasks_per_center,
                    slot_start_seconds=decision_time,
                    slot_end_seconds=slot_end_seconds,
                    stackelberg_control=dispatch_result.get('stackelberg_control', {}),
                    force_center_pickup_on_first_departure=True,
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    event_assignments, _, event_details = _run_assignment_for_window(**assignment_kwargs)
                _apply_assignment_results_to_workers(G, warm_worker_sim, event_details)
                assignments.update(event_assignments)
                details.extend(event_details)

                assigned_task_ids = {key[1] for key in event_assignments.keys()}
                if not assigned_task_ids:
                    return
                for rid in region_ids:
                    unassigned_tasks_pool[rid] = [
                        task for task in unassigned_tasks_pool[rid]
                        if task[1] not in assigned_task_ids and decision_time < task[3]
                    ]

            run_warm_visible_assignment(slot_start_seconds)

            slot_events = slot_df.sort_values(['seconds_of_day', 'task_id'])
            for release_time, event_rows in slot_events.groupby('seconds_of_day', sort=True):
                decision_time = float(release_time)
                warm_worker_sim.advance_workers_to_time(centers, decision_time)
                for _, row in event_rows.iterrows():
                    rid = int(row['region_id'])
                    task_id = str(row['task_id'])
                    unassigned_tasks_pool[rid].append(
                        (
                            row['nearest_node'],
                            f"warm_actual:{history_date}:{task_id}",
                            config.TASK_BASE_REWARD,
                            decision_time + float(config.TASK_EXPIRE_MINUTES) * 60,
                            decision_time,
                        )
                    )
                run_warm_visible_assignment(decision_time)

            warm_worker_sim.advance_workers_to_time(centers, slot_end_seconds)
            run_warm_visible_assignment(slot_end_seconds)

            assigned_counts = {rid: 0 for rid in region_ids}
            for detail in details:
                assigned_counts[int(detail['region_id'])] += 1
            assigned_task_ids = {key[1] for key in assignments.keys()}
            for rid in region_ids:
                unassigned_tasks_pool[rid] = [
                    task for task in unassigned_tasks_pool[rid]
                    if task[1] not in assigned_task_ids and slot_end_seconds < task[3]
                ]

            transitions = dispatch_result.get('transitions', {})
            next_available_workers = {
                rid: len(
                    warm_worker_sim.get_available_workers_with_center_info(
                        rid,
                        current_time=slot_end_seconds,
                    )
                )
                for rid in region_ids
            }
            if transitions:
                hydrate_retention_transitions_with_next_state(
                    state=state,
                    transitions=transitions,
                    demand_profile=dispatch_result.get('demand_profile', {}),
                    desired_workers=dispatch_result.get('desired_workers', {}),
                    available_workers=next_available_workers,
                    backlog_counts={rid: len(unassigned_tasks_pool[rid]) for rid in region_ids},
                    max_tasks_per_worker=int(getattr(config, 'MAX_TASKS_PER_WORKER', 4)),
                    min_buffer_workers=int(getattr(config, 'UABG_MIN_BUFFER_WORKERS', 1)),
                    backlog_weight=float(getattr(config, 'UABG_BACKLOG_WEIGHT', 1.0)),
                    done=float(slot_idx == slot_count_per_day - 1),
                )
                update_rl_retention_bilateral_state(
                    state=state,
                    transitions=transitions,
                    assigned_tasks_by_region=assigned_counts,
                    total_tasks_by_region=slot_total_tasks,
                    hoard_penalty_by_region=dispatch_result.get('hoard_penalty', {}),
                    move_cost_by_region=dispatch_result.get('move_cost_by_region', {}),
                    moves=dispatch_result.get('moves', []),
                    hoard_penalty_weight=float(getattr(config, 'RBG_REWARD_HOARD_WEIGHT', 0.02)),
                    move_cost_weight=float(getattr(config, 'RBG_REWARD_MOVE_WEIGHT', 0.08)),
                    unfairness_weight=float(getattr(config, 'RBG_REWARD_UNFAIRNESS_WEIGHT', 1.0)),
                )
            center_transitions = dispatch_result.get('center_transitions', {})
            if center_task_state is not None and center_transitions:
                update_center_task_allocation_state(
                    state=center_task_state,
                    transitions=center_transitions,
                    assigned_tasks_by_region=assigned_counts,
                    total_tasks_by_region=slot_total_tasks,
                    remaining_tasks_by_region={rid: len(unassigned_tasks_pool[rid]) for rid in region_ids},
                    actual_arrivals_by_region=actual_counts,
                    predicted_demand_by_region={rid: len(predicted_tasks.get(rid, [])) for rid in region_ids},
                )
            platform_transition = dispatch_result.get('platform_transition')
            if platform_state is not None and platform_transition is not None:
                hydrate_platform_transition_with_next_state(
                    state=platform_state,
                    transition=platform_transition,
                    available_workers=next_available_workers,
                    backlog_counts={rid: len(unassigned_tasks_pool[rid]) for rid in region_ids},
                    max_tasks_per_worker=int(getattr(config, 'MAX_TASKS_PER_WORKER', 4)),
                    backlog_weight=float(getattr(config, 'UABG_BACKLOG_WEIGHT', 1.0)),
                    done=float(slot_idx == slot_count_per_day - 1),
                )
                update_platform_task_first_state(
                    state=platform_state,
                    transition=platform_transition,
                    assigned_tasks_by_region=assigned_counts,
                    total_tasks_by_region=slot_total_tasks,
                    fairness_secondary_weight=float(
                        getattr(config, 'PFRL_FAIRNESS_SECONDARY_WEIGHT', 0.20)
                    ),
                )
            state.record_prediction_feedback(
                predicted_region_demand=actual_counts,
                actual_region_demand=actual_counts,
            )
            completed_slots += 1
        completed_days += 1

    if completed_days <= 0 or completed_slots <= 0:
        return None
    extra_replay = state.offline_replay_train(
        epochs=int(getattr(config, 'RBG_PREFIX_EXTRA_REPLAY_EPOCHS', 0)),
        updates_per_region=int(getattr(config, 'RBG_PREFIX_REPLAY_UPDATES_PER_REGION', 1)),
    )
    platform_extra_replay = {}
    if platform_state is not None:
        platform_extra_replay = platform_state.offline_replay_train(
            epochs=int(getattr(config, 'PFRL_PREFIX_EXTRA_REPLAY_EPOCHS', 0)),
            updates_per_epoch=int(getattr(config, 'PFRL_PREFIX_REPLAY_UPDATES', 1)),
        )
    center_extra_replay = {}
    if center_task_state is not None:
        center_extra_replay = center_task_state.offline_replay_train(
            epochs=int(getattr(config, 'EVENT_CENTER_PREFIX_EXTRA_REPLAY_EPOCHS', 0)),
            updates_per_region=int(getattr(config, 'EVENT_CENTER_PREFIX_REPLAY_UPDATES_PER_REGION', 1)),
        )
    return {
        'source': (
            'historical_event_task_proxy'
            if bool(getattr(config, 'EVENT_RL_WARM_START_FAST_PROXY', True))
            else 'historical_event_task_replay'
        ),
        'history_days': completed_days,
        'slot_count': completed_slots,
        'worker_count': worker_count,
        'replayed_tasks': replayed_tasks,
        'precommit_tasks': precommit_tasks,
        'extra_replay': extra_replay,
        'platform_extra_replay': platform_extra_replay,
        'center_extra_replay': center_extra_replay,
    }


def _offline_warm_start_rbg_state(
        state: RLRetentionBilateralState,
        platform_state,
        center_task_state,
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
        algo_name=None,
):
    if not bool(getattr(config, 'RBG_OFFLINE_WARM_START', True)):
        state.set_exploration_prob(float(getattr(config, 'RBG_ONLINE_EXPLORATION_PROB', state.exploration_prob)))
        return None

    if str(algo_name).lower() in PREDICTIVE_EVENT_RL_GAME_ALGOS:
        event_task_stats = _simulate_historical_event_task_rl_warm_start(
            state=state,
            platform_state=platform_state,
            center_task_state=center_task_state,
            test_date=test_date,
            test_start_hour=test_start_hour,
            test_end_hour=test_end_hour,
            time_slot_minutes=time_slot_minutes,
            G=G,
            coords=coords,
            nodes=nodes,
            rcc_partition=rcc_partition,
            centers=centers,
            reference_worker_sim=worker_sim,
        )
        if event_task_stats is not None:
            print(
                f"   - Event-task RL warm-start ready: days={event_task_stats['history_days']}, "
                f"slots={event_task_stats['slot_count']}, tasks={event_task_stats['replayed_tasks']}, "
                f"precommit={event_task_stats['precommit_tasks']}, source={event_task_stats['source']}"
            )
            extra_replay = event_task_stats.get('extra_replay', {})
            platform_extra_replay = event_task_stats.get('platform_extra_replay', {})
            center_extra_replay = event_task_stats.get('center_extra_replay', {})
            print(
                f"   - Continuous SAC warm-start optimization: "
                f"retention={extra_replay.get('optimization_steps', 0)}, "
                f"platform={platform_extra_replay.get('optimization_steps', 0)}, "
                f"center={center_extra_replay.get('optimization_steps', 0)} steps"
            )
            state.set_exploration_prob(float(getattr(config, 'RBG_ONLINE_EXPLORATION_PROB', state.exploration_prob)))
            return event_task_stats

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
        platform_state=platform_state,
    )
    if prefix_sim_stats is not None:
        print(
            f"   - RBG continuous replay warm-start ready: slots={prefix_sim_stats['slot_count']}, "
            f"workers={prefix_sim_stats['worker_count']}, source={prefix_sim_stats['source']}, "
            f"micro={prefix_sim_stats.get('micro_batch_minutes', time_slot_minutes)}min"
        )
        extra_replay = prefix_sim_stats.get('extra_replay', {})
        if int(extra_replay.get('optimization_steps', 0)) > 0:
            print(
                f"   - RBG prefix replay refinement: epochs={extra_replay.get('epochs', 0)}, "
                f"updates/region={extra_replay.get('updates_per_region', 0)}, "
                f"opt_steps={extra_replay.get('optimization_steps', 0)}"
            )
        platform_extra_replay = prefix_sim_stats.get('platform_extra_replay', {})
        if int(platform_extra_replay.get('optimization_steps', 0)) > 0:
            print(
                f"   - Platform prefix replay refinement: epochs={platform_extra_replay.get('epochs', 0)}, "
                f"updates={platform_extra_replay.get('updates_per_epoch', 0)}, "
                f"opt_steps={platform_extra_replay.get('optimization_steps', 0)}"
            )
        global_action_summary = prefix_sim_stats.get('global_action_summary', [])
        if global_action_summary:
            action_text = ", ".join(
                f"{item['action_label']}({item['action_ratio']:+.2f}): n={item['count']}, "
                f"avg_reward={item['avg_reward']:.1f}, avg_served={item['avg_served']:.1f}"
                for item in global_action_summary
            )
            print(f"   - RBG prefix action reward: {action_text}")
        dominant_region_actions = prefix_sim_stats.get('dominant_region_actions', {})
        if dominant_region_actions:
            region_text = ", ".join(
                f"R{rid}: {info['action_label']}({info['action_ratio']:+.2f}) "
                f"(n={info['count']}, avg_reward={info['avg_reward']:.1f})"
                for rid, info in sorted(dominant_region_actions.items())
            )
            print(f"   - RBG prefix dominant actions: {region_text}")
        state.set_exploration_prob(float(getattr(config, 'RBG_ONLINE_EXPLORATION_PROB', state.exploration_prob)))
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
        state.set_exploration_prob(float(getattr(config, 'RBG_ONLINE_EXPLORATION_PROB', state.exploration_prob)))
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
    if bool(stats.get('continuous_sac')) and int(stats.get('epochs', 0)) <= 0:
        print(
            f"   - Continuous SAC offline fallback skipped: "
            f"{stats.get('reason', 'no replay transitions')}"
        )
        state.set_exploration_prob(float(getattr(config, 'RBG_ONLINE_EXPLORATION_PROB', state.exploration_prob)))
        return stats
    print(
        f"   - RBG offline warm-start ready: samples={stats['sample_count']}, "
        f"epochs={stats['epochs']}, source={historical_samples[0].get('source', 'unknown')}"
    )
    state.set_exploration_prob(float(getattr(config, 'RBG_ONLINE_EXPLORATION_PROB', state.exploration_prob)))
    return stats

def _build_simulation_context(
        test_date: str,
        test_start_hour: int,
        test_end_hour: int,
        time_slot_minutes: int,
        slots_to_run: int
):
    compare_end_seconds = test_start_hour * 3600 + slots_to_run * time_slot_minutes * 60
    scope = _current_scope_metadata()
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
        _print_scope_metadata(_SIMULATION_CONTEXT_CACHE[cache_key].get('scope', scope))
        return _SIMULATION_CONTEXT_CACHE[cache_key]

    print("\n【阶段 1-3】加载路网数据与中心划分...")
    _print_scope_metadata(scope)
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
        eval_slot_counts = []
        for slot_idx in range(slots_to_run):
            slot_start = test_start_hour * 3600 + slot_idx * time_slot_minutes * 60
            slot_end = slot_start + time_slot_minutes * 60
            eval_slot_counts.append(
                int(((eval_tasks['seconds_of_day'] >= slot_start) & (eval_tasks['seconds_of_day'] < slot_end)).sum())
            )
        print(
            f"   - Evaluation window tasks: {len(eval_tasks)} "
            f"({test_start_hour:02d}:00-{compare_end_seconds // 3600:02d}:{(compare_end_seconds % 3600) // 60:02d}); "
            f"per-slot={eval_slot_counts}"
        )
        df_tasks = eval_tasks.copy()
    else:
        print(f"⚠️ 未找到任务文件: {task_file}")
        df_tasks = pd.DataFrame()
        eval_tasks = pd.DataFrame(columns=['region_id'])

    total_tasks_per_center = {region_id: 0 for region_id in centers.keys()}
    if not eval_tasks.empty:
        for region_id, count in eval_tasks['region_id'].value_counts().items():
            total_tasks_per_center[region_id] = int(count)
    counted_eval_tasks = int(sum(total_tasks_per_center.values()))
    if counted_eval_tasks != int(len(eval_tasks)):
        print(
            f"   - WARNING: center task count mismatch: "
            f"by_center={counted_eval_tasks}, eval_rows={len(eval_tasks)}"
        )

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
        'scope': scope,
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
    force_cpu = _should_force_mctgnet_cpu()
    predictor_device = _get_preferred_torch_device(force_cpu=force_cpu)
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
        predictor_device,
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
    if force_cpu:
        print("   - MCTGNet dispatch predictor device: CPU-only (forced for this run)")
    else:
        print(f"   - MCTGNet dispatch predictor device: {predictor_device}")

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
        device=predictor_device,
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
    predictor_device = _get_preferred_torch_device()
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
        predictor_device,
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
    print(f"   - CenterPatternLSTM predictor device: {predictor_device}")

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
        device=predictor_device,
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


def _get_or_train_event_task_predictor(
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
        getattr(config, 'EVENT_PRED_SLOT_MINUTES', 5),
        config.DOWNLOAD_DIST,
        config.NUM_ZONES,
    )
    if predictor_key in _EVENT_TASK_PREDICTOR_CACHE:
        print("   - Reusing cached event-task predictor")
        return _EVENT_TASK_PREDICTOR_CACHE[predictor_key]

    print(f"\n[阶段 4.5] 训练事件级预测器并准备 {test_date} 的任务级预承诺计划...")
    predictor = EventTaskDispatchPredictor(
        data_dir=config.TASK_DATA_DIR,
        coords=coords,
        nodes=nodes,
        partition=rcc_partition,
        centers=centers,
        time_interval=time_slot_minutes,
        event_slot_minutes=int(getattr(config, 'EVENT_PRED_SLOT_MINUTES', 5)),
        random_state=int(getattr(config, 'RBG_RANDOM_SEED', 42)),
    )
    predictor.fit(
        target_date=test_date,
        start_hour=float(test_start_hour),
        end_hour=float(test_end_hour),
    )
    print(f"   - Event forecast tasks ready: {len(predictor.forecast)}")
    _EVENT_TASK_PREDICTOR_CACHE[predictor_key] = predictor
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
        repartition=False,
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
        repartition=repartition,
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


def _summarize_unique_assigned_tasks(details, region_ids):
    task_region = {}
    duplicate_detail_count = 0
    for detail in details:
        task_id = detail.get('task_id')
        if task_id is None:
            continue
        if task_id in task_region:
            duplicate_detail_count += 1
            continue
        task_region[task_id] = detail.get('region_id')

    assigned_tasks_per_center = {region_id: 0 for region_id in region_ids}
    for region_id in task_region.values():
        if region_id in assigned_tasks_per_center:
            assigned_tasks_per_center[region_id] += 1
    return task_region, assigned_tasks_per_center, duplicate_detail_count


def _count_unique_assigned_task_ids(assignments):
    return len({task_id for _, task_id in assignments.keys()})


def _count_unassigned_pool_tasks(unassigned_tasks_pool):
    return sum(len(pool) for pool in unassigned_tasks_pool.values())


def _expire_unassigned_tasks(unassigned_tasks_pool, current_time):
    expired_count = 0
    for rid, pool in unassigned_tasks_pool.items():
        kept = []
        for task in pool:
            if float(current_time) >= float(task[3]):
                expired_count += 1
                continue
            kept.append(task)
        unassigned_tasks_pool[rid] = kept
    return expired_count


def _next_worker_service_completion_time(worker_sim, current_time, horizon_time):
    eps = 1e-6
    completion_times = [
        float(busy_until)
        for wid, busy_until in worker_sim.worker_busy_until.items()
        if worker_sim.worker_status.get(wid) == 'en_route_to_task'
        and busy_until is not None
        and float(busy_until) > float(current_time) + eps
        and float(busy_until) <= float(horizon_time) + eps
    ]
    return min(completion_times) if completion_times else None


def _resolve_assignment_service_horizon(tasks_per_center, slot_start_seconds, slot_end_seconds):
    if not bool(getattr(config, 'ONLINE_ALLOW_SERVICE_AFTER_BATCH_END', False)):
        return slot_end_seconds
    max_expire = None
    for tasks in tasks_per_center.values():
        for task in tasks:
            expire_time = float(task[3])
            max_expire = expire_time if max_expire is None else max(max_expire, expire_time)
    if max_expire is None:
        return slot_end_seconds
    max_overtime = max(0.0, float(getattr(config, 'ONLINE_DRAIN_MAX_SECONDS', 0.0)))
    capped_horizon = float(slot_end_seconds) + max_overtime if max_overtime > 0.0 else max_expire
    return max(float(slot_end_seconds), min(max_expire, capped_horizon))


def _online_force_center_pickup(commit_next_step_only):
    return (
        not commit_next_step_only
        or bool(getattr(config, 'ONLINE_REPLAN_REQUIRES_CENTER_PICKUP', True))
    )


def _get_worker_pending_center_transfers(worker_sim):
    pending = getattr(worker_sim, 'worker_pending_center_transfers', None)
    if pending is None:
        pending = {}
        setattr(worker_sim, 'worker_pending_center_transfers', pending)
    return pending


def _shortest_distance_m(G, src_node, dst_node, distance_cache):
    if src_node == dst_node:
        return 0.0
    key = (src_node, dst_node)
    if key not in distance_cache:
        try:
            distance_cache[key] = float(nx.shortest_path_length(G, src_node, dst_node, weight='length'))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            distance_cache[key] = float('inf')
    return distance_cache[key]


def _find_opportunistic_support_task(
        G,
        worker_node,
        donor_center_node,
        receiver_center_node,
        donor_tasks,
        current_time,
):
    if not donor_tasks:
        return None

    speed = max(1e-6, float(getattr(config, 'WORKER_SPEED_MS', 1.0)))
    max_detour_m = max(0.0, float(getattr(config, 'SUPPORT_OPPORTUNISTIC_MAX_DETOUR_M', 800.0)))
    max_detour_ratio = max(0.0, float(getattr(config, 'SUPPORT_OPPORTUNISTIC_MAX_DETOUR_RATIO', 0.35)))
    candidate_limit = max(1, int(getattr(config, 'SUPPORT_OPPORTUNISTIC_CANDIDATE_LIMIT', 32)))
    distance_cache = {}

    direct_to_receiver = _shortest_distance_m(G, worker_node, receiver_center_node, distance_cache)
    if not np.isfinite(direct_to_receiver):
        return None

    worker_to_donor = _shortest_distance_m(G, worker_node, donor_center_node, distance_cache)
    if not np.isfinite(worker_to_donor):
        return None

    best = None
    prioritized_tasks = sorted(
        donor_tasks,
        key=lambda task: (
            max(0.0, float(task[3]) - float(current_time)),
            float(task[4]) if len(task) > 4 else float(current_time),
            str(task[1]),
        ),
    )[:candidate_limit]
    for task in prioritized_tasks:
        task_node = task[0]
        donor_to_task = _shortest_distance_m(G, donor_center_node, task_node, distance_cache)
        task_to_receiver = _shortest_distance_m(G, task_node, receiver_center_node, distance_cache)
        if not np.isfinite(donor_to_task) or not np.isfinite(task_to_receiver):
            continue

        release_time = float(task[4]) if len(task) > 4 else float(current_time)
        service_departure = max(float(current_time) + worker_to_donor / speed, release_time)
        service_finish = service_departure + donor_to_task / speed
        if service_finish > float(task[3]):
            continue

        via_distance = worker_to_donor + donor_to_task + task_to_receiver
        detour_m = max(0.0, via_distance - direct_to_receiver)
        detour_ratio = detour_m / max(1.0, direct_to_receiver)
        if detour_m > max_detour_m and detour_ratio > max_detour_ratio:
            continue

        score = (
            detour_m,
            service_finish,
            max(0.0, float(task[3]) - service_finish),
            str(task[1]),
        )
        if best is None or score < best[0]:
            best = (
                score,
                {
                    'task': task,
                    'detour_m': detour_m,
                    'detour_ratio': detour_ratio,
                    'direct_distance_m': direct_to_receiver,
                    'via_distance_m': via_distance,
                    'service_finish': service_finish,
                },
            )

    return None if best is None else best[1]


def _prepare_opportunistic_support_tasks(
        G,
        worker_sim,
        centers,
        unassigned_tasks_pool,
        moves,
        current_time,
        stackelberg_control=None,
):
    if not bool(getattr(config, 'SUPPORT_OPPORTUNISTIC_LOCAL_TASK_ENABLED', True)):
        return 0
    if not moves:
        return 0

    pending = _get_worker_pending_center_transfers(worker_sim)
    worker_task_priority_map = None
    if isinstance(stackelberg_control, dict):
        worker_task_priority_map = stackelberg_control.setdefault('worker_task_priority_map', {})

    prepared = 0
    wait_seconds = max(0.0, float(getattr(config, 'SUPPORT_OPPORTUNISTIC_WAIT_SECONDS', 60.0)))
    priority_bonus = max(0.0, float(getattr(config, 'SUPPORT_OPPORTUNISTIC_PRIORITY_BONUS', 0.60)))
    for move in moves:
        try:
            donor = int(move.get('from_region'))
            receiver = int(move.get('to_region'))
        except (TypeError, ValueError):
            continue
        wid = str(move.get('wid'))
        if donor == receiver or donor not in centers or receiver not in centers:
            continue
        if wid not in worker_sim.worker_positions:
            continue
        if wid in pending:
            continue
        if worker_sim.worker_status.get(wid, 'idle') == 'en_route_to_task':
            continue

        donor_tasks = [
            task for task in unassigned_tasks_pool.get(donor, [])
            if float(task[3]) > float(current_time)
        ]
        match = _find_opportunistic_support_task(
            G=G,
            worker_node=worker_sim.worker_positions[wid][0],
            donor_center_node=centers[donor],
            receiver_center_node=centers[receiver],
            donor_tasks=donor_tasks,
            current_time=current_time,
        )
        if match is None:
            continue

        task = match['task']
        task_id = str(task[1])
        worker_sim.worker_center_map[wid] = donor
        pending[wid] = {
            'from_region': donor,
            'to_region': receiver,
            'task_id': task_id,
            'created_at': float(current_time),
            'release_after': float(current_time) + wait_seconds,
        }
        move['opportunistic_task_id'] = task_id
        move['opportunistic_detour_m'] = float(match['detour_m'])
        move['opportunistic_detour_ratio'] = float(match['detour_ratio'])
        move['deferred_until_local_task'] = True
        if worker_task_priority_map is not None:
            pair_key = f"{wid}|{task_id}"
            worker_task_priority_map[pair_key] = max(
                float(worker_task_priority_map.get(pair_key, 1.0)),
                1.0 + priority_bonus,
            )
        prepared += 1

    return prepared


def _prepare_and_log_opportunistic_support_tasks(
        G,
        worker_sim,
        centers,
        unassigned_tasks_pool,
        moves,
        current_time,
        stackelberg_control=None,
        label='顺路支援',
):
    prepared = _prepare_opportunistic_support_tasks(
        G=G,
        worker_sim=worker_sim,
        centers=centers,
        unassigned_tasks_pool=unassigned_tasks_pool,
        moves=moves,
        current_time=current_time,
        stackelberg_control=stackelberg_control,
    )
    if prepared > 0:
        print(
            f"   [{label}] {prepared} 名支援工人先在原中心接顺路单，完成后继续支援目标中心"
        )
    return prepared


def _release_stale_pending_support_transfers(worker_sim, unassigned_tasks_pool, current_time):
    pending = _get_worker_pending_center_transfers(worker_sim)
    if not pending:
        return 0

    released = 0
    for wid, transfer in list(pending.items()):
        if worker_sim.worker_status.get(wid, 'idle') == 'en_route_to_task':
            continue
        target_region = int(transfer.get('to_region'))
        task_id = str(transfer.get('task_id'))
        release_after = float(transfer.get('release_after', current_time))
        donor = int(transfer.get('from_region', worker_sim.worker_center_map.get(wid, target_region)))
        donor_tasks = unassigned_tasks_pool.get(donor, [])
        task_still_waiting = any(str(task[1]) == task_id and float(task[3]) > float(current_time) for task in donor_tasks)
        if current_time < release_after and task_still_waiting:
            continue
        worker_sim.worker_center_map[wid] = target_region
        pending.pop(wid, None)
        released += 1
    return released


def _node_rough_distance_m(G, src_node, dst_node):
    if src_node == dst_node:
        return 0.0
    try:
        src = G.nodes[src_node]
        dst = G.nodes[dst_node]
        src_lon = float(src.get('x', src.get('lon')))
        src_lat = float(src.get('y', src.get('lat')))
        dst_lon = float(dst.get('x', dst.get('lon')))
        dst_lat = float(dst.get('y', dst.get('lat')))
    except (KeyError, TypeError, ValueError):
        return 0.0
    mean_lat = math.radians((src_lat + dst_lat) * 0.5)
    lon_scale = 111320.0 * max(0.1, math.cos(mean_lat))
    lat_scale = 111320.0
    return math.hypot((src_lon - dst_lon) * lon_scale, (src_lat - dst_lat) * lat_scale)


def _match_precommit_bundles_to_visible_tasks(
        G,
        tasks_per_center,
        control,
        current_time,
        worker_task_priority_map,
        task_priority_weight_map,
):
    bundles_by_worker = control.get('precommit_bundles_by_worker', {}) or {}
    if not bundles_by_worker:
        return worker_task_priority_map, control.get('task_bundle_map') or {}

    max_match_distance_m = float(getattr(config, 'EVENT_PRECOMMIT_MATCH_MAX_DISTANCE_M', 500.0))
    max_time_delta = float(getattr(config, 'EVENT_PRECOMMIT_MATCH_TIME_TOLERANCE_SECONDS', 180.0))
    candidate_limit = max(1, int(getattr(config, 'EVENT_PRECOMMIT_MATCH_CANDIDATE_LIMIT', 4)))
    route_weight = max(0.0, float(getattr(config, 'EVENT_PRECOMMIT_MATCH_ROUTE_INSERTION_WEIGHT', 1.0)))
    time_weight = max(0.0, float(getattr(config, 'EVENT_PRECOMMIT_MATCH_TIME_WEIGHT', 0.5)))
    expire_weight = max(0.0, float(getattr(config, 'EVENT_PRECOMMIT_MATCH_EXPIRE_WEIGHT', 0.1)))
    priority_bonus = max(0.0, float(getattr(config, 'EVENT_PRECOMMIT_MATCH_PRIORITY_BONUS', 0.25)))

    task_bundle_map = {
        str(task_id): str(bundle_id)
        for task_id, bundle_id in (control.get('task_bundle_map') or {}).items()
    }
    used_task_ids = set()
    distance_cache = {}

    def network_distance(src_node, dst_node):
        if src_node == dst_node:
            return 0.0
        key = (src_node, dst_node)
        if key not in distance_cache:
            try:
                distance_cache[key] = float(nx.shortest_path_length(G, src_node, dst_node, weight='length'))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                distance_cache[key] = float('inf')
        return distance_cache[key]

    for wid, bundle in bundles_by_worker.items():
        anchors = list(bundle.get('anchors', []) or [])
        anchors.sort(key=lambda item: (int(item.get('round_id', 0)), int(item.get('sequence_index', 0))))
        for anchor_idx, anchor in enumerate(anchors):
            rid = int(anchor.get('region_id', -1))
            candidates = tasks_per_center.get(rid, [])
            if not candidates:
                continue
            anchor_node = anchor.get('node')
            anchor_release = float(anchor.get('release_time', current_time))
            anchor_expire = float(anchor.get('expire_time', current_time))
            rough_candidates = []
            for task in candidates:
                task_id = str(task[1])
                if task_id in used_task_ids:
                    continue
                release_delta = abs((float(task[4]) if len(task) > 4 else float(current_time)) - anchor_release)
                expire_delta = abs(float(task[3]) - anchor_expire)
                if release_delta > max_time_delta and expire_delta > max_time_delta:
                    continue
                rough_distance = _node_rough_distance_m(G, anchor_node, task[0])
                if rough_distance > max_match_distance_m * 2.0:
                    continue
                rough_candidates.append((rough_distance, release_delta, expire_delta, task))

            best_task = None
            best_score = None
            for rough_distance, release_delta, expire_delta, task in sorted(rough_candidates)[:candidate_limit]:
                distance_m = network_distance(anchor_node, task[0])
                if not np.isfinite(distance_m) or distance_m > max_match_distance_m:
                    continue

                insertion_penalty = 0.0
                if anchor_idx + 1 < len(anchors):
                    next_anchor = anchors[anchor_idx + 1]
                    if str(next_anchor.get('bundle_id')) == str(anchor.get('bundle_id')):
                        next_node = next_anchor.get('node')
                        direct_next = network_distance(anchor_node, next_node)
                        via_next = distance_m + network_distance(task[0], next_node)
                        if np.isfinite(direct_next) and np.isfinite(via_next):
                            insertion_penalty = max(0.0, via_next - direct_next)

                score = (
                    distance_m
                    + route_weight * insertion_penalty
                    + time_weight * release_delta
                    + expire_weight * expire_delta
                    + 0.001 * rough_distance
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_task = task

            if best_task is None:
                continue

            task_id = str(best_task[1])
            used_task_ids.add(task_id)
            base_weight = max(1.0, float(anchor.get('priority_weight', 1.0)))
            matched_weight = base_weight + priority_bonus
            pair_key = f"{wid}|{task_id}"
            worker_task_priority_map[pair_key] = max(
                worker_task_priority_map.get(pair_key, 1.0),
                matched_weight,
            )
            task_priority_weight_map[task_id] = max(
                task_priority_weight_map.get(task_id, 1.0),
                1.0 + 0.5 * priority_bonus,
            )
            task_bundle_map[task_id] = str(anchor.get('bundle_id', f"{wid}:0"))

    return worker_task_priority_map, task_bundle_map


def _with_event_center_task_priorities(
        algo_key,
        G,
        tasks_per_center,
        stackelberg_control,
        current_time,
):
    if algo_key not in PREDICTIVE_EVENT_RL_GAME_ALGOS or not stackelberg_control:
        return stackelberg_control
    profiles = stackelberg_control.get('event_center_task_action_profile', {})
    if not profiles:
        return stackelberg_control

    control = dict(stackelberg_control)
    task_priority_weight_map = {}
    current_task_ids_by_region = {
        rid: {str(task[1]) for task in tasks}
        for rid, tasks in tasks_per_center.items()
    }
    all_current_task_ids = set()
    for task_ids in current_task_ids_by_region.values():
        all_current_task_ids.update(task_ids)
    worker_task_priority_map = {
        key: value
        for key, value in (control.get('worker_task_priority_map') or {}).items()
        if len(str(key).split('|', 1)) == 2
        and str(key).split('|', 1)[1] in all_current_task_ids
    }
    match_precommit_tasks = bool(getattr(config, 'EVENT_PRECOMMIT_MATCH_ENABLED', False))
    for rid, tasks in tasks_per_center.items():
        action = profiles.get(rid, {})
        backlog_weight = float(action.get('backlog_weight', 1.0))
        arrival_weight = float(action.get('predicted_weight', 1.0))
        urgency_weight = float(action.get('urgency_weight', 1.0))
        for task in tasks:
            release_seconds = float(task[4]) if len(task) > 4 else float(current_time)
            is_backlog = release_seconds < float(current_time)
            remaining = max(1.0, float(task[3]) - float(current_time))
            urgency = 1.0 + 60.0 / remaining
            source_weight = backlog_weight if is_backlog else arrival_weight
            task_priority_weight_map[str(task[1])] = max(0.05, source_weight + urgency_weight * urgency)
    if match_precommit_tasks:
        worker_task_priority_map, task_bundle_map = _match_precommit_bundles_to_visible_tasks(
            G=G,
            tasks_per_center=tasks_per_center,
            control=control,
            current_time=current_time,
            worker_task_priority_map=worker_task_priority_map,
            task_priority_weight_map=task_priority_weight_map,
        )
        control['task_bundle_map'] = task_bundle_map
    control['task_priority_weight_map'] = task_priority_weight_map
    control['worker_task_priority_map'] = worker_task_priority_map
    return control


def _event_precommit_has_matching_actual_task(
        G,
        planned,
        candidate_tasks,
        current_time,
):
    max_match_distance_m = float(getattr(config, 'EVENT_PRECOMMIT_MATCH_MAX_DISTANCE_M', 500.0))
    max_time_delta = float(getattr(config, 'EVENT_PRECOMMIT_MATCH_TIME_TOLERANCE_SECONDS', 180.0))
    planned_node = planned.get('node')
    planned_release = float(planned.get('release_time', current_time))
    planned_expire = float(planned.get('expire_time', current_time))
    for task in candidate_tasks:
        if str(task[1]) == str(planned.get('task_id')):
            distance_m = 0.0
        else:
            try:
                distance_m = float(nx.shortest_path_length(G, planned_node, task[0], weight='length'))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        if distance_m > max_match_distance_m:
            continue
        release_delta = abs((float(task[4]) if len(task) > 4 else float(current_time)) - planned_release)
        expire_delta = abs(float(task[3]) - planned_expire)
        if release_delta <= max_time_delta or expire_delta <= max_time_delta:
            return True
    return False


def _filter_event_precommit_waiting_workers(
        algo_key,
        G,
        centers,
        workers_per_center,
        tasks_per_center,
        stackelberg_control,
        current_time,
):
    if algo_key not in PREDICTIVE_EVENT_RL_GAME_ALGOS or not stackelberg_control:
        return workers_per_center
    if not bool(getattr(config, 'EVENT_PRECOMMIT_WAIT_ENABLED', False)):
        return workers_per_center
    precommit_records = stackelberg_control.get('precommit_task_records_by_worker', {}) or {}
    if not precommit_records:
        return workers_per_center

    wait_before = max(0.0, float(getattr(config, 'EVENT_PRECOMMIT_WAIT_BEFORE_SECONDS', 60.0)))
    wait_after = max(0.0, float(getattr(config, 'EVENT_PRECOMMIT_WAIT_AFTER_SECONDS', 120.0)))
    speed = max(1e-6, float(getattr(config, 'WORKER_SPEED_MS', 1.0)))
    filtered = {}
    for rid, workers in workers_per_center.items():
        kept_workers = []
        center_node = centers.get(rid)
        candidate_tasks = tasks_per_center.get(rid, [])
        urgent_slack = max(0.0, float(getattr(config, 'EVENT_PRECOMMIT_WAIT_URGENT_SLACK_SECONDS', 120.0)))
        backlog_release_ratio = max(
            0.0,
            float(getattr(config, 'EVENT_PRECOMMIT_WAIT_BACKLOG_RELEASE_RATIO', 0.35)),
        )
        urgent_backlog = any(float(task[3]) - float(current_time) <= urgent_slack for task in candidate_tasks)
        backlog_pressure = len(candidate_tasks) >= max(1, int(math.ceil(len(workers) * backlog_release_ratio)))
        if urgent_backlog or backlog_pressure:
            filtered[rid] = list(workers)
            continue
        for worker in workers:
            worker_node, wid = worker[0], str(worker[1])
            should_wait = False
            for planned in precommit_records.get(wid, []):
                if int(planned.get('region_id', rid)) != int(rid):
                    continue
                planned_release = float(planned.get('release_time', current_time))
                if current_time < planned_release - wait_before or current_time > planned_release + wait_after:
                    continue
                if _event_precommit_has_matching_actual_task(G, planned, candidate_tasks, current_time):
                    continue
                try:
                    center_eta = float(nx.shortest_path_length(G, worker_node, center_node, weight='length')) / speed
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    center_eta = float('inf')
                if current_time + center_eta <= planned_release + wait_after:
                    should_wait = True
                    break
            if not should_wait:
                kept_workers.append(worker)
        filtered[rid] = kept_workers
    return filtered


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
        force_center_pickup_on_first_departure=True,
):
    algo_key = algo_name.lower()
    stackelberg_control = _with_event_center_task_priorities(
        algo_key=algo_key,
        G=G,
        tasks_per_center=tasks_per_center,
        stackelberg_control=stackelberg_control,
        current_time=slot_start_seconds,
    )
    workers_per_center = _filter_event_precommit_waiting_workers(
        algo_key=algo_key,
        G=G,
        centers=centers,
        workers_per_center=workers_per_center,
        tasks_per_center=tasks_per_center,
        stackelberg_control=stackelberg_control,
        current_time=slot_start_seconds,
    )
    assignment_end_seconds = _resolve_assignment_service_horizon(
        tasks_per_center=tasks_per_center,
        slot_start_seconds=slot_start_seconds,
        slot_end_seconds=slot_end_seconds,
    )
    if algo_key in ['greedy', 'predictive_greedy']:
        return greedy_assignment_with_center_pickup(
            G=G,
            config=config,
            centers=centers,
            partition=rcc_partition,
            workers_per_center=workers_per_center,
            tasks_per_center=tasks_per_center,
            slot_start_seconds=slot_start_seconds,
            slot_end_seconds=assignment_end_seconds,
        )
    if algo_key in ROUTE_ILP_ASSIGNMENT_ALGOS:
        return center_prepacked_assignment_with_center_pickup(
            G=G,
            config=config,
            centers_dict=centers,
            workers_per_center=workers_per_center,
            tasks_per_center=tasks_per_center,
            slot_start_seconds=slot_start_seconds,
            slot_end_seconds=assignment_end_seconds,
            stackelberg_control=stackelberg_control if algo_key in RETENTION_RL_ALGOS else None,
            force_center_pickup_on_first_departure=force_center_pickup_on_first_departure,
        )
    if algo_key in ['imtao', 'imtao_seq_bdc', 'seq_bdc']:
        return run_imtao_for_slot(
            G=G,
            config=config,
            centers_dict=centers,
            workers_per_center=workers_per_center,
            tasks_per_center=tasks_per_center,
            slot_start_seconds=slot_start_seconds,
            slot_end_seconds=assignment_end_seconds,
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
            slot_end_seconds=assignment_end_seconds,
            collaboration_mode=IMTAO_MODE_RBDC,
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
            slot_end_seconds=assignment_end_seconds,
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
            slot_end_seconds=assignment_end_seconds,
            collaboration_mode=IMTAO_MODE_WO_C,
            center_selection=IMTAO_SELECT_LOWEST_RHO,
        )
    raise ValueError(f"Unsupported algorithm: {algo_name}")


def _apply_assignment_results_to_workers(
        G,
        worker_sim,
        slot_details,
        commit_service_only=False,
):
    slot_dist_to_center = 0.0
    slot_dist_to_task = 0.0
    worker_final_state = {}

    time_key = 'service_finish_time' if commit_service_only else 'finish_time'
    node_key = 'service_node' if commit_service_only else 'task_node'

    for detail in slot_details:
        wid = detail['wid']
        slot_dist_to_center += detail['dist_to_center']
        slot_dist_to_task += detail['dist_to_task']

        prev_detail = worker_final_state.get(wid)
        if prev_detail is None:
            worker_final_state[wid] = detail
            continue

        if commit_service_only:
            if float(detail.get(time_key, float('inf'))) < float(prev_detail.get(time_key, float('inf'))):
                worker_final_state[wid] = detail
        else:
            if float(detail.get(time_key, 0.0)) > float(prev_detail.get(time_key, 0.0)):
                worker_final_state[wid] = detail

    for wid, detail in worker_final_state.items():
        task_node = detail[node_key]
        if task_node not in G.nodes:
            continue
        task_lon = G.nodes[task_node].get('x', G.nodes[task_node].get('lon'))
        task_lat = G.nodes[task_node].get('y', G.nodes[task_node].get('lat'))
        worker_sim.update_worker_position(wid, task_node, task_lon, task_lat)
        pending_transfers = _get_worker_pending_center_transfers(worker_sim)
        pending_transfer = pending_transfers.get(str(wid))
        if pending_transfer is not None and str(detail.get('task_id')) == str(pending_transfer.get('task_id')):
            worker_sim.worker_center_map[wid] = int(pending_transfer['to_region'])
            pending_transfers.pop(str(wid), None)
        worker_sim.set_worker_en_route_to_task(wid, float(detail.get(time_key, 0.0)))

    return slot_dist_to_center, slot_dist_to_task


def _reduce_microbatch_results_for_online_replanning(
        micro_assignments,
        micro_details,
        commit_horizon_seconds=None,
):
    """
    In sub-slot rolling dispatch, only commit work that must leave in the current micro-batch.
    This keeps not-yet-urgent tasks available for later re-packing.
    """
    if not micro_details:
        return micro_assignments, 0.0, micro_details

    latest_departure_commit = (
        bool(getattr(config, 'MICROBATCH_LATEST_DEPARTURE_COMMIT', False))
        and not bool(getattr(config, 'ONLINE_COMMIT_ONE_TASK_AT_A_TIME', False))
    )
    if latest_departure_commit and any('round_departure_time' in detail for detail in micro_details):
        rounds_by_worker = {}
        for detail in micro_details:
            wid = detail['wid']
            round_id = int(detail.get('round_id', 0))
            rounds_by_worker.setdefault(wid, {}).setdefault(round_id, []).append(detail)

        committed_details = []
        for wid, worker_rounds in rounds_by_worker.items():
            ordered_rounds = sorted(
                worker_rounds.items(),
                key=lambda item: (
                    float(item[1][0].get('round_departure_time', item[1][0].get('service_finish_time', float('inf')))),
                    int(item[0]),
                    str(wid),
                )
            )
            for _, round_details in ordered_rounds:
                round_departure_time = float(
                    round_details[0].get(
                        'round_departure_time',
                        round_details[0].get('service_finish_time', float('inf'))
                    )
                )
                if commit_horizon_seconds is not None and round_departure_time > float(commit_horizon_seconds):
                    continue
                committed_details.extend(
                    sorted(
                        round_details,
                        key=lambda d: (
                            float(d.get('service_finish_time', d.get('finish_time', float('inf')))),
                            str(d.get('task_id')),
                        )
                    )
                )
                break

        if not committed_details:
            return {}, 0.0, []

        committed_assignments = {}
        committed_score = 0.0
        for detail in committed_details:
            pair = (detail['wid'], detail['task_id'])
            committed_assignments[pair] = micro_assignments.get(
                pair,
                detail.get('profit', detail.get('objective_score', 1.0))
            )
            if 'objective_score' in detail:
                committed_score += float(detail.get('objective_score', 0.0))
            else:
                committed_score += float(detail.get('profit', committed_assignments[pair]))
        return committed_assignments, committed_score, committed_details

    earliest_detail_by_worker = {}
    for detail in micro_details:
        wid = detail['wid']
        candidate_key = (
            float(detail.get('service_finish_time', detail.get('finish_time', float('inf')))),
            float(detail.get('finish_time', float('inf'))),
            str(detail.get('task_id')),
        )
        prev = earliest_detail_by_worker.get(wid)
        if prev is None:
            earliest_detail_by_worker[wid] = detail
            continue

        prev_key = (
            float(prev.get('service_finish_time', prev.get('finish_time', float('inf')))),
            float(prev.get('finish_time', float('inf'))),
            str(prev.get('task_id')),
        )
        if candidate_key < prev_key:
            earliest_detail_by_worker[wid] = detail

    committed_details = sorted(
        earliest_detail_by_worker.values(),
        key=lambda d: (
            float(d.get('service_finish_time', d.get('finish_time', float('inf')))),
            str(d.get('wid')),
            str(d.get('task_id')),
        )
    )
    committed_assignments = {}
    committed_score = 0.0

    for detail in committed_details:
        pair = (detail['wid'], detail['task_id'])
        committed_assignments[pair] = micro_assignments.get(
            pair,
            detail.get('profit', detail.get('objective_score', 1.0))
        )
        if 'objective_score' in detail:
            committed_score += float(detail.get('objective_score', 0.0))
        else:
            committed_score += float(detail.get('profit', committed_assignments[pair]))

    return committed_assignments, committed_score, committed_details


_MICROBATCH_CANDIDATE_DEFAULT = object()


def _build_microbatch_candidate_tasks(
        unassigned_tasks_pool,
        workers_per_center,
        current_time,
        slot_end_seconds,
        candidate_factor=_MICROBATCH_CANDIDATE_DEFAULT,
        candidate_floor=_MICROBATCH_CANDIDATE_DEFAULT,
        candidate_cap=_MICROBATCH_CANDIDATE_DEFAULT,
):
    raw_candidate_factor = (
        getattr(config, 'MICROBATCH_TASK_CANDIDATE_FACTOR', 3.0)
        if candidate_factor is _MICROBATCH_CANDIDATE_DEFAULT
        else candidate_factor
    )
    candidate_factor = None if raw_candidate_factor is None else float(raw_candidate_factor)
    candidate_floor = int(
        getattr(config, 'MICROBATCH_TASK_CANDIDATE_FLOOR', 48)
        if candidate_floor is _MICROBATCH_CANDIDATE_DEFAULT
        else candidate_floor
    )
    raw_candidate_cap = (
        getattr(config, 'MICROBATCH_TASK_CANDIDATE_CAP', 240)
        if candidate_cap is _MICROBATCH_CANDIDATE_DEFAULT
        else candidate_cap
    )
    candidate_cap = None if raw_candidate_cap is None else int(raw_candidate_cap)
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

        if candidate_factor is None:
            limit = len(pool)
        else:
            limit = int(round(worker_count * max_tasks_per_worker * candidate_factor))
            limit = max(candidate_floor, limit)
        if candidate_cap is not None and candidate_cap > 0:
            limit = min(candidate_cap, limit)
        limit = min(limit, len(pool))

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


def _should_defer_online_dispatch(algo_key, unassigned_tasks_pool, total_workers, current_time):
    if not bool(getattr(config, 'ONLINE_DISPATCH_HOLD_ENABLED', False)):
        return False

    hold_algos = {
        str(item).lower()
        for item in getattr(config, 'ONLINE_DISPATCH_HOLD_ALGOS', ())
    }
    if hold_algos and str(algo_key).lower() not in hold_algos:
        return False

    visible_tasks = [
        task
        for pool in unassigned_tasks_pool.values()
        for task in pool
    ]
    if not visible_tasks or total_workers <= 0:
        return False

    urgent_slack = max(0.0, float(getattr(config, 'ONLINE_DISPATCH_HOLD_URGENT_SLACK_SECONDS', 120.0)))
    if any(float(task[3]) - float(current_time) <= urgent_slack for task in visible_tasks):
        return False

    oldest_release = min(float(task[4]) if len(task) > 4 else float(current_time) for task in visible_tasks)
    max_wait = max(0.0, float(getattr(config, 'ONLINE_DISPATCH_HOLD_MAX_SECONDS', 60.0)))
    if float(current_time) - oldest_release >= max_wait:
        return False

    min_tasks = max(1, int(getattr(config, 'ONLINE_DISPATCH_HOLD_MIN_TASKS', 16)))
    worker_ratio = max(0.0, float(getattr(config, 'ONLINE_DISPATCH_HOLD_WORKER_RATIO', 0.35)))
    max_trigger = max(min_tasks, int(getattr(config, 'ONLINE_DISPATCH_HOLD_MAX_TRIGGER_TASKS', 48)))
    worker_scaled = int(math.ceil(float(total_workers) * worker_ratio))
    trigger_tasks = max(min_tasks, min(max_trigger, worker_scaled))
    return len(visible_tasks) < trigger_tasks


def _resolve_micro_redispatch_gap_batches(algo_name, micro_batch_seconds):
    algo_key = str(algo_name).lower()
    if algo_key in RETENTION_RL_ALGOS:
        base_gap = int(getattr(config, 'RBG_MICROBATCH_REDISPATCH_MIN_GAP', 1))
        min_interval_minutes = float(getattr(config, 'RBG_MICROBATCH_REDISPATCH_MIN_INTERVAL_MINUTES', 0.0))
    else:
        base_gap = int(getattr(config, 'MICROBATCH_REDISPATCH_MIN_GAP', 2))
        min_interval_minutes = float(getattr(config, 'MICROBATCH_REDISPATCH_MIN_INTERVAL_MINUTES', 0.0))

    gap = max(1, base_gap)
    if micro_batch_seconds > 0 and min_interval_minutes > 0:
        interval_gap = int(math.ceil((min_interval_minutes * 60.0) / float(micro_batch_seconds)))
        gap = max(gap, interval_gap)
    return gap


def _build_micro_correction_profile(
        algo_name,
        current_time,
        slot_start_seconds,
        slot_end_seconds,
        available_workers_per_center,
        backlog_counts,
        observed_arrivals_so_far,
        predicted_total_demand,
):
    slot_duration = max(1.0, float(slot_end_seconds - slot_start_seconds))
    elapsed_ratio = min(1.0, max(0.0, (current_time - slot_start_seconds) / slot_duration))
    max_tasks_per_worker = int(getattr(config, 'MAX_TASKS_PER_WORKER', 4))
    correction_only = bool(
        algo_name.lower() in RETENTION_RL_ALGOS
        and getattr(config, 'RBG_MICRO_CORRECTION_ONLY', False)
    )
    trigger_abs_tasks = int(getattr(config, 'RBG_MICRO_CORRECTION_TRIGGER_ABS_TASKS', 10))
    trigger_ratio = float(getattr(config, 'RBG_MICRO_CORRECTION_TRIGGER_RATIO', 0.12))
    correction_gain = float(getattr(config, 'RBG_MICRO_CORRECTION_GAIN', 1.0))
    min_elapsed_ratio = float(getattr(config, 'RBG_MICRO_CORRECTION_MIN_ELAPSED_RATIO', 0.20))
    backlog_multiplier = float(getattr(config, 'RBG_MICRO_CORRECTION_BACKLOG_MULTIPLIER', 1.10))
    min_move_scale = float(getattr(config, 'RBG_MICRO_CORRECTION_MOVE_SHARE_MIN_SCALE', 0.35))

    dispatch_demand = {}
    diagnostics = {}
    should_trigger = False
    severity = 0.0

    for rid in available_workers_per_center.keys():
        predicted_total = 0 if predicted_total_demand is None else int(predicted_total_demand.get(rid, 0))
        observed_cumulative = int(observed_arrivals_so_far.get(rid, 0))
        expected_cumulative = float(predicted_total) * elapsed_ratio
        arrival_gap = float(observed_cumulative) - expected_cumulative
        remaining_predicted = max(0.0, float(predicted_total - observed_cumulative))
        backlog = max(0.0, float(backlog_counts.get(rid, 0)))
        available_workers = max(0, int(available_workers_per_center.get(rid, 0)))
        local_capacity = max(1.0, float(available_workers * max_tasks_per_worker))
        backlog_excess = max(0.0, backlog - local_capacity * backlog_multiplier)

        if correction_only:
            dispatch_value = max(0.0, backlog_excess + correction_gain * max(0.0, arrival_gap))
        else:
            dispatch_value = remaining_predicted
        dispatch_demand[rid] = int(math.ceil(dispatch_value))

        gap_threshold = max(float(trigger_abs_tasks), float(predicted_total) * trigger_ratio)
        region_severity = max(
            backlog_excess / max(1.0, gap_threshold),
            abs(arrival_gap) / max(1.0, gap_threshold),
        )
        severity = max(severity, region_severity)

        if correction_only and elapsed_ratio >= min_elapsed_ratio:
            if backlog_excess >= gap_threshold or abs(arrival_gap) >= gap_threshold:
                should_trigger = True

        diagnostics[rid] = {
            'predicted_total': predicted_total,
            'observed_cumulative': observed_cumulative,
            'expected_cumulative': expected_cumulative,
            'arrival_gap': arrival_gap,
            'remaining_predicted': remaining_predicted,
            'backlog': backlog,
            'local_capacity': local_capacity,
            'backlog_excess': backlog_excess,
            'dispatch_demand': dispatch_demand[rid],
        }

    move_share_scale = 1.0
    if correction_only:
        move_share_scale = float(np.clip(max(min_move_scale, severity), min_move_scale, 1.0))

    return {
        'correction_only': correction_only,
        'elapsed_ratio': elapsed_ratio,
        'should_trigger': should_trigger,
        'dispatch_demand': dispatch_demand,
        'move_share_scale': move_share_scale,
        'diagnostics': diagnostics,
    }


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
    if (
        algo_name.lower() in RETENTION_RL_ALGOS
        and getattr(config, 'RBG_MICRO_CORRECTION_ONLY', False)
    ):
        correction_profile = _build_micro_correction_profile(
            algo_name=algo_name,
            current_time=current_time,
            slot_start_seconds=slot_start_seconds,
            slot_end_seconds=slot_end_seconds,
            available_workers_per_center=available_workers_per_center,
            backlog_counts=backlog_counts,
            observed_arrivals_so_far=observed_arrivals_so_far,
            predicted_total_demand=predicted_total_demand,
        )
        return bool(correction_profile['should_trigger'])

    if algo_name.lower() in RETENTION_RL_ALGOS:
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
    correction_profile = None
    if (
        algo_name.lower() in RETENTION_RL_ALGOS
        and getattr(config, 'RBG_MICRO_CORRECTION_ONLY', False)
    ):
        correction_profile = _build_micro_correction_profile(
            algo_name=algo_name,
            current_time=current_time,
            slot_start_seconds=slot_start_seconds,
            slot_end_seconds=slot_end_seconds,
            available_workers_per_center=available_workers,
            backlog_counts=backlog_counts,
            observed_arrivals_so_far=slot_new_tasks_per_center,
            predicted_total_demand=current_slot_predicted_demand,
        )

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
    redispatch_target_demand = remaining_predicted
    correction_move_share_scale = 1.0
    if correction_profile is not None:
        redispatch_target_demand = correction_profile['dispatch_demand']
        correction_move_share_scale = float(correction_profile.get('move_share_scale', 1.0))

    if algo_name.lower() in RETENTION_RL_ALGOS:
        batch_fine_tune = bool(getattr(config, 'RBG_BATCH_ONLINE_FINE_TUNE', True))
        current_slot_platform_transition = None
        platform_task_weight = getattr(config, 'RBG_PLATFORM_TASK_WEIGHT', 0.30)
        platform_gap_weight = getattr(config, 'RBG_PLATFORM_GAP_WEIGHT', 0.55)
        platform_release_credit_weight = getattr(config, 'RBG_PLATFORM_RELEASE_CREDIT_WEIGHT', 0.35)
        predicted_distribution = None
        if hasattr(dispatch_predictor, 'predict_region_distribution'):
            dispatch_base_date = pd.Timestamp(DEFAULT_TEST_DATE).date()
            predicted_distribution = dispatch_predictor.predict_region_distribution(
                datetime.combine(
                    dispatch_base_date,
                    datetime.min.time(),
                ) + timedelta(seconds=float(current_time))
            )
        if algo_name.lower() in PREDICTIVE_PLATFORM_RL_ALGOS and platform_rl_state is not None:
            current_slot_platform_transition = sample_platform_task_first_control(
                region_ids=sorted(centers.keys()),
                predicted_demand=redispatch_target_demand,
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
            predicted_demand=redispatch_target_demand,
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
            platform_move_share_scale=(
                float(current_slot_platform_transition.get('move_share_scale', 1.0)) * correction_move_share_scale
                if current_slot_platform_transition else correction_move_share_scale
            ),
            platform_slot_start_blend_scale=float(current_slot_platform_transition.get('slot_start_blend_scale', 1.0)) if current_slot_platform_transition else 1.0,
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
            record_transition=batch_fine_tune,
        )
        if current_slot_platform_transition is not None:
            result['platform_transition'] = current_slot_platform_transition
        if correction_profile is not None:
            result['micro_correction'] = correction_profile
        if algo_name.lower() in PREDICTIVE_PLATFORM_RL_ALGOS:
            label = 'Platform-RL-Micro'
        elif algo_name.lower() in NO_PRED_RBG_ALGOS:
            label = 'NoPred-RL-Micro'
        else:
            label = 'RBG-Micro'
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
        center_task_rl_state,
):
    algo_key = algo_name.lower()
    slot_duration_seconds = int(time_slot_minutes) * 60
    rbg_realtime_dispatch = (
        algo_key in RETENTION_RL_ALGOS
        and algo_key not in PREDICTIVE_EVENT_RL_GAME_ALGOS
        and bool(getattr(config, 'RBG_INTRABATCH_REALTIME_DISPATCH', False))
    )
    is_intrabatch_online = (
        algo_key in INTRABATCH_ONLINE_ALGOS
        or algo_key in PREDICTIVE_EVENT_RL_GAME_ALGOS
        or rbg_realtime_dispatch
        or micro_batch_seconds >= slot_duration_seconds
    )
    commit_next_step_only = bool(getattr(config, 'ONLINE_COMMIT_ONE_TASK_AT_A_TIME', True))
    all_assignments = {}
    all_details = []
    total_profit = 0.0
    total_dist_to_center = 0.0
    total_dist_to_task = 0.0
    total_expired_tasks_global = 0
    batch_fine_tune_enabled = bool(getattr(config, 'RBG_BATCH_ONLINE_FINE_TUNE', True))

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
        current_slot_center_transitions = {}
        current_slot_rbg_hoard_penalty = {}
        current_slot_rbg_move_cost = {}
        current_slot_rbg_moves = []
        current_slot_rbg_stackelberg_control = {}
        current_slot_rbg_demand_profile = {}
        current_slot_rbg_desired_workers = {}
        current_slot_platform_transition = None
        current_slot_platform_fairness_weight = float(getattr(config, 'PFRL_FAIRNESS_SECONDARY_WEIGHT', 0.20))
        last_rbg_reward_by_region = None
        last_platform_stats = None

        if algo_name.lower() in ['predictive_mctgnet', 'predictive_game_mctgnet', 'predictive_bstgcnet', 'predictive_center_lstm', 'predictive_game_center_lstm', *PREDICTIVE_UABG_ALGOS, *PREDICTIVE_RBG_ALGOS, *PREDICTIVE_PLATFORM_RL_ALGOS]:
            one_step_predicted_demand = dispatch_predictor.predict_region_demand(slot_timestamp)
            if one_step_predicted_demand is not None:
                current_slot_predicted_demand = one_step_predicted_demand
                backlog_counts = {rid: len(unassigned_tasks_pool[rid]) for rid in centers.keys()}
                displayed_plan_demand = dict(one_step_predicted_demand)
                predicted_distribution = None
                if algo_name.lower() in [*PREDICTIVE_UABG_ALGOS, *PREDICTIVE_RBG_ALGOS, *PREDICTIVE_PLATFORM_RL_ALGOS] and hasattr(dispatch_predictor, 'predict_region_distribution'):
                    predicted_distribution = dispatch_predictor.predict_region_distribution(slot_timestamp)

                if algo_name.lower() in PREDICTIVE_EVENT_RL_GAME_ALGOS:
                    predicted_tasks = dispatch_predictor.predict_dispatch_tasks(slot_timestamp)
                    predispatch_result = event_task_rl_game_predispatch_workers(
                        G=G,
                        worker_sim=worker_sim,
                        centers=centers,
                        predicted_tasks=predicted_tasks,
                        backlog_tasks=unassigned_tasks_pool,
                        predicted_distribution=predicted_distribution or {},
                        state=retention_game_state,
                        platform_state=platform_rl_state,
                        center_task_state=center_task_rl_state,
                        slot_idx=slot_idx,
                        slot_start_seconds=slot_start_seconds,
                        slot_end_seconds=slot_end_seconds,
                    )
                    current_slot_rbg_transitions = predispatch_result.get('transitions', {})
                    current_slot_center_transitions = predispatch_result.get('center_transitions', {})
                    current_slot_rbg_hoard_penalty = predispatch_result.get('hoard_penalty', {})
                    current_slot_rbg_move_cost = predispatch_result.get('move_cost_by_region', {})
                    current_slot_rbg_moves = predispatch_result.get('moves', [])
                    current_slot_rbg_stackelberg_control = predispatch_result.get('stackelberg_control', {})
                    current_slot_rbg_demand_profile = predispatch_result.get('demand_profile', {})
                    current_slot_rbg_desired_workers = predispatch_result.get('desired_workers', {})
                    current_slot_platform_transition = predispatch_result.get('platform_transition')
                    predict_label = 'Event-RL-Game Predict'
                elif algo_name.lower() in RETENTION_RL_ALGOS:
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
                        platform_move_share_scale=float(current_slot_platform_transition.get('move_share_scale', 1.0)) if current_slot_platform_transition else 1.0,
                        platform_slot_start_blend_scale=float(current_slot_platform_transition.get('slot_start_blend_scale', 1.0)) if current_slot_platform_transition else 1.0,
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
                    current_slot_rbg_demand_profile = predispatch_result.get('demand_profile', {})
                    current_slot_rbg_desired_workers = predispatch_result.get('desired_workers', {})
                    if algo_name.lower() in PREDICTIVE_PLATFORM_RL_ALGOS:
                        predict_label = 'Platform-RL-MCTGNet Predict' if platform_rl_state is not None else 'Platform-Fixed-MCTGNet Predict'
                    else:
                        predict_label = 'RBG-MCTGNet Predict'
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
                if algo_name.lower() in RETENTION_RL_ALGOS:
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
                if algo_name.lower() in PREDICTIVE_EVENT_RL_GAME_ALGOS:
                    retain_text = ", ".join(
                        [
                            f"R{rid}: planned={predispatch_result.get('precommit_planned_by_region', {}).get(rid, 0)}, "
                            f"bw={predispatch_result.get('center_action_profile', {}).get(rid, {}).get('backlog_weight', 1.0):.2f}, "
                            f"pw={predispatch_result.get('center_action_profile', {}).get(rid, {}).get('predicted_weight', 1.0):.2f}, "
                            f"uw={predispatch_result.get('center_action_profile', {}).get(rid, {}).get('urgency_weight', 1.0):.2f}, "
                            f"trust={predispatch_result.get('center_prediction_trust', {}).get(rid, 1.0):.2f}, "
                            f"workers_after_loan={max(0, predispatch_result['retain_count'].get(rid, 0))}"
                            for rid in sorted(centers.keys())
                        ]
                    )
                    print(f"   [{predict_label}] center task policy: {retain_text}")
                elif algo_name.lower() in RETENTION_RL_ALGOS:
                    retain_text = ", ".join(
                        [
                            f"R{rid}: keep={predispatch_result['retain_count'].get(rid, 0)}, "
                            f"need={predispatch_result['desired_workers'].get(rid, 0)}, "
                            f"rr={predispatch_result.get('action_profile', {}).get(rid, {}).get('retention_ratio', 0.0):+.2f}, "
                            f"bid={predispatch_result.get('action_profile', {}).get(rid, {}).get('receive_bid_scale', 1.0):.2f}, "
                            f"ask={predispatch_result.get('action_profile', {}).get(rid, {}).get('release_ask_scale', 1.0):.2f}, "
                            f"hoard={predispatch_result['hoard_penalty'].get(rid, 0):.1f}"
                            for rid in sorted(centers.keys())
                        ]
                    )
                    print(f"   [{predict_label}] retention policy: {retain_text}")
                if algo_name.lower() in PREDICTIVE_EVENT_RL_GAME_ALGOS:
                    planned_count = sum(
                        len(task_ids)
                        for task_ids in predispatch_result.get('precommit_plan_by_worker', {}).values()
                    )
                    planned_workers = len(predispatch_result.get('precommit_plan_by_worker', {}))
                    uncovered_text = ", ".join(
                        [
                            f"R{rid}: uncovered={predispatch_result.get('precommit_uncovered_predicted', {}).get(rid, 0)}"
                            for rid in sorted(centers.keys())
                        ]
                    )
                    print(
                        f"   [{predict_label}] precommitment: "
                        f"{planned_workers} workers / {planned_count} planned tasks | {uncovered_text}"
                    )
                    plan_preview = []
                    for wid, task_ids in list(predispatch_result.get('precommit_plan_by_worker', {}).items())[:3]:
                        shown_tasks = ", ".join(str(tid) for tid in task_ids[:4])
                        if len(task_ids) > 4:
                            shown_tasks += ", ..."
                        plan_preview.append(f"{wid}->[{shown_tasks}]")
                    if plan_preview:
                        print(f"   [{predict_label}] worker-task precommit sample: " + "; ".join(plan_preview))

                if predispatch_result['moves']:
                    move_summary = ", ".join(
                        [f"{m['wid']}:{m['from_region']}->{m['to_region']}" for m in predispatch_result['moves'][:8]]
                    )
                    if len(predispatch_result['moves']) > 8:
                        move_summary += f", ... (+{len(predispatch_result['moves']) - 8} more)"
                    post_service_count = sum(1 for m in predispatch_result['moves'] if m.get('post_service'))
                    post_service_note = f" (完成后支援 {post_service_count} 名)" if post_service_count else ""
                    print(
                        f"   [{predict_label}] pre-dispatched {len(predispatch_result['moves'])} workers"
                        f"{post_service_note}: {move_summary}"
                    )
                    if uncertainty_dispatch_state is not None and algo_name.lower() in PREDICTIVE_UABG_ALGOS:
                        uncertainty_dispatch_state.record_moves(
                            slot_idx=slot_idx,
                            moved_workers=[m['wid'] for m in predispatch_result['moves']]
                        )
                else:
                    print(f"   [{predict_label}] no worker rebalancing needed")
            else:
                print("   [Predictive] insufficient history, skip pre-dispatch for this slot")

        if algo_name.lower() in NO_PRED_RBG_ALGOS:
            backlog_counts = {rid: len(unassigned_tasks_pool[rid]) for rid in centers.keys()}
            current_slot_predicted_demand = {rid: 0 for rid in centers.keys()}
            predispatch_result = rl_retention_bilateral_predispatch_workers(
                G=G,
                worker_sim=worker_sim,
                centers=centers,
                predicted_demand=current_slot_predicted_demand,
                state=retention_game_state,
                slot_idx=slot_idx,
                next_slot_start_seconds=slot_start_seconds,
                predicted_distribution=None,
                max_tasks_per_worker=getattr(config, 'MAX_TASKS_PER_WORKER', 4),
                backlog_counts=backlog_counts,
                backlog_weight=getattr(config, 'UABG_BACKLOG_WEIGHT', 1.0),
                uncertainty_weight=getattr(config, 'UABG_UNCERTAINTY_WEIGHT', 0.45),
                quantile_weight=getattr(config, 'UABG_QUANTILE_WEIGHT', 0.55),
                burst_weight=getattr(config, 'UABG_BURST_WEIGHT', 1.2),
                calibration_bias_weight=0.0,
                calibration_shrink_weight=0.0,
                calibration_sigma_boost=0.0,
                calibration_min_scale=1.0,
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
            current_slot_rbg_transitions = predispatch_result.get('transitions', {})
            current_slot_rbg_hoard_penalty = predispatch_result.get('hoard_penalty', {})
            current_slot_rbg_move_cost = predispatch_result.get('move_cost_by_region', {})
            current_slot_rbg_moves = predispatch_result.get('moves', [])
            current_slot_rbg_stackelberg_control = predispatch_result.get('stackelberg_control', {})
            current_slot_rbg_demand_profile = predispatch_result.get('demand_profile', {})
            current_slot_rbg_desired_workers = predispatch_result.get('desired_workers', {})
            current_predict_label = 'NoPred-RL-Game Dispatch'
            retain_text = ", ".join(
                [
                    f"R{rid}: keep={predispatch_result['retain_count'].get(rid, 0)}, "
                    f"need={predispatch_result['desired_workers'].get(rid, 0)}, "
                    f"rr={predispatch_result.get('action_profile', {}).get(rid, {}).get('retention_ratio', 0.0):+.2f}, "
                    f"bid={predispatch_result.get('action_profile', {}).get(rid, {}).get('receive_bid_scale', 1.0):.2f}, "
                    f"ask={predispatch_result.get('action_profile', {}).get(rid, {}).get('release_ask_scale', 1.0):.2f}, "
                    f"hoard={predispatch_result['hoard_penalty'].get(rid, 0):.1f}"
                    for rid in sorted(centers.keys())
                ]
            )
            print(f"   [{current_predict_label}] retention policy: {retain_text}")
            if predispatch_result['moves']:
                move_summary = ", ".join(
                    [f"{m['wid']}:{m['from_region']}->{m['to_region']}" for m in predispatch_result['moves'][:8]]
                )
                if len(predispatch_result['moves']) > 8:
                    move_summary += f", ... (+{len(predispatch_result['moves']) - 8} more)"
                print(f"   [{current_predict_label}] pre-dispatched {len(predispatch_result['moves'])} workers: {move_summary}")
            else:
                print(f"   [{current_predict_label}] no worker rebalancing needed")

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

        _prepare_and_log_opportunistic_support_tasks(
            G=G,
            worker_sim=worker_sim,
            centers=centers,
            unassigned_tasks_pool=unassigned_tasks_pool,
            moves=current_slot_rbg_moves,
            current_time=slot_start_seconds,
            stackelberg_control=current_slot_rbg_stackelberg_control,
            label=f"{current_predict_label or 'RBG Dispatch'} 顺路支援",
        )

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
        total_micro = len(range(slot_start_seconds, slot_end_seconds, micro_batch_seconds))
        redispatch_gap = _resolve_micro_redispatch_gap_batches(
            algo_name=algo_name,
            micro_batch_seconds=micro_batch_seconds,
        )

        for micro_idx, micro_start_seconds in enumerate(range(slot_start_seconds, slot_end_seconds, micro_batch_seconds)):
            micro_end_seconds = min(slot_end_seconds, micro_start_seconds + micro_batch_seconds)
            worker_sim.advance_workers_to_time(centers, micro_start_seconds)

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
                and algo_name.lower() not in PREDICTIVE_EVENT_RL_GAME_ALGOS
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
                    if algo_name.lower() in RETENTION_RL_ALGOS:
                        current_slot_rbg_transitions = micro_dispatch_result.get('transitions', current_slot_rbg_transitions)
                        current_slot_rbg_hoard_penalty = micro_dispatch_result.get('hoard_penalty', current_slot_rbg_hoard_penalty)
                        current_slot_rbg_move_cost = micro_dispatch_result.get('move_cost_by_region', current_slot_rbg_move_cost)
                        current_slot_rbg_moves = micro_dispatch_result.get('moves', current_slot_rbg_moves)
                        current_slot_rbg_stackelberg_control = micro_dispatch_result.get('stackelberg_control', current_slot_rbg_stackelberg_control)
                        current_slot_rbg_demand_profile = micro_dispatch_result.get('demand_profile', current_slot_rbg_demand_profile)
                        current_slot_rbg_desired_workers = micro_dispatch_result.get('desired_workers', current_slot_rbg_desired_workers)
                        if algo_name.lower() in PREDICTIVE_PLATFORM_RL_ALGOS:
                            current_slot_platform_transition = micro_dispatch_result.get('platform_transition', current_slot_platform_transition)
                        _prepare_and_log_opportunistic_support_tasks(
                            G=G,
                            worker_sim=worker_sim,
                            centers=centers,
                            unassigned_tasks_pool=unassigned_tasks_pool,
                            moves=current_slot_rbg_moves,
                            current_time=micro_start_seconds,
                            stackelberg_control=current_slot_rbg_stackelberg_control,
                            label='RBG-Micro 顺路支援',
                        )
                    for region_id in centers.keys():
                        workers = worker_sim.get_available_workers_with_center_info(region_id, current_time=micro_start_seconds)
                        workers_per_center[region_id] = [(w[0], w[1], w[2], w[3], centers[region_id]) for w in workers]

            micro_new_tasks = 0
            micro_assignments = {}
            micro_profit = 0.0
            micro_details = []
            micro_assigned_tasks_per_center = {rid: 0 for rid in centers.keys()}
            micro_total_tasks_per_center = {rid: len(unassigned_tasks_pool[rid]) for rid in centers.keys()}
            micro_event_count = 0

            def run_visible_assignment(decision_time, force=False):
                nonlocal micro_profit, slot_profit, slot_dist_to_center, slot_dist_to_task
                nonlocal slot_expired_count
                _release_stale_pending_support_transfers(
                    worker_sim,
                    unassigned_tasks_pool,
                    decision_time,
                )
                total_workers_at_time = 0
                workers_snapshot = {}
                for region_id in centers.keys():
                    workers = worker_sim.get_available_workers_with_center_info(
                        region_id,
                        current_time=decision_time,
                    )
                    workers_snapshot[region_id] = [(w[0], w[1], w[2], w[3], centers[region_id]) for w in workers]
                    total_workers_at_time += len(workers)

                total_tasks_at_time = sum(len(unassigned_tasks_pool[rid]) for rid in centers.keys())
                if total_tasks_at_time <= 0 or total_workers_at_time <= 0:
                    return 0

                if (
                    not force
                    and _should_defer_online_dispatch(
                        algo_key=algo_key,
                        unassigned_tasks_pool=unassigned_tasks_pool,
                        total_workers=total_workers_at_time,
                        current_time=decision_time,
                    )
                ):
                    return 0

                if is_intrabatch_online and algo_key not in ROUTE_ILP_ASSIGNMENT_ALGOS:
                    tasks_snapshot = _build_intrabatch_online_tasks(
                        unassigned_tasks_pool=unassigned_tasks_pool,
                        current_time=decision_time,
                    )
                else:
                    route_candidate_kwargs = {}
                    if algo_key in ROUTE_ILP_ASSIGNMENT_ALGOS:
                        route_candidate_kwargs = {
                            'candidate_factor': getattr(config, 'ROUTE_ILP_MICROBATCH_TASK_CANDIDATE_FACTOR', None),
                            'candidate_floor': getattr(config, 'ROUTE_ILP_MICROBATCH_TASK_CANDIDATE_FLOOR', None),
                            'candidate_cap': getattr(config, 'ROUTE_ILP_MICROBATCH_TASK_CANDIDATE_CAP', None),
                        }
                    tasks_snapshot = _build_microbatch_candidate_tasks(
                        unassigned_tasks_pool=unassigned_tasks_pool,
                        workers_per_center=workers_snapshot,
                        current_time=decision_time,
                        slot_end_seconds=slot_end_seconds,
                        **route_candidate_kwargs,
                    )

                assignment_kwargs = dict(
                    algo_name=algo_name,
                    G=G,
                    config=config,
                    centers=centers,
                    rcc_partition=rcc_partition,
                    workers_per_center=workers_snapshot,
                    tasks_per_center=tasks_snapshot,
                    slot_start_seconds=decision_time,
                    slot_end_seconds=slot_end_seconds,
                    stackelberg_control=current_slot_rbg_stackelberg_control,
                    force_center_pickup_on_first_departure=_online_force_center_pickup(commit_next_step_only),
                )
                if bool(getattr(config, 'EVENT_ONLINE_ASSIGNMENT_VERBOSE', False)):
                    event_assignments, event_profit, event_details = _run_assignment_for_window(**assignment_kwargs)
                else:
                    with contextlib.redirect_stdout(io.StringIO()):
                        event_assignments, event_profit, event_details = _run_assignment_for_window(**assignment_kwargs)
                if commit_next_step_only:
                    event_assignments, event_profit, event_details = _reduce_microbatch_results_for_online_replanning(
                        micro_assignments=event_assignments,
                        micro_details=event_details,
                        commit_horizon_seconds=decision_time,
                    )

                dist_center_inc, dist_task_inc = _apply_assignment_results_to_workers(
                    G,
                    worker_sim,
                    event_details,
                    commit_service_only=commit_next_step_only,
                )
                slot_dist_to_center += dist_center_inc
                slot_dist_to_task += dist_task_inc
                micro_assignments.update(event_assignments)
                micro_profit += event_profit
                micro_details.extend(event_details)
                slot_assignments.update(event_assignments)
                slot_details.extend(event_details)
                slot_profit += event_profit

                for detail in event_details:
                    slot_assigned_tasks_per_center[detail['region_id']] += 1
                    micro_assigned_tasks_per_center[detail['region_id']] += 1
                    slot_worker_ids.add(detail['wid'])

                assigned_task_ids = {k[1] for k in event_assignments.keys()}
                if not assigned_task_ids:
                    return 0

                for rid in centers.keys():
                    new_pool = []
                    for t in unassigned_tasks_pool[rid]:
                        if t[1] in assigned_task_ids:
                            continue
                        if decision_time >= t[3]:
                            slot_expired_count += 1
                            continue
                        new_pool.append(t)
                    unassigned_tasks_pool[rid] = new_pool
                return len(assigned_task_ids)

            # Carry-over orders are already known at the tick boundary; handling them
            # here does not reveal any future arrivals inside the current minute.
            run_visible_assignment(micro_start_seconds)

            if not df_tasks.empty:
                mask = (df_tasks['seconds_of_day'] >= micro_start_seconds) & (df_tasks['seconds_of_day'] < micro_end_seconds)
                current_tasks = df_tasks[mask].sort_values(['seconds_of_day', 'task_id'])
                for release_time, event_rows in current_tasks.groupby('seconds_of_day', sort=True):
                    decision_time = float(release_time)
                    worker_sim.advance_workers_to_time(centers, decision_time)
                    event_new_tasks = 0
                    for _, row in event_rows.iterrows():
                        nearest_node = row['nearest_node']
                        if nearest_node not in rcc_partition:
                            continue
                        region_id = rcc_partition[nearest_node]
                        task = (
                            nearest_node,
                            row['task_id'],
                            config.TASK_BASE_REWARD,
                            decision_time + config.TASK_EXPIRE_MINUTES * 60,
                            decision_time,
                        )
                        unassigned_tasks_pool[region_id].append(task)
                        slot_new_tasks_per_center[region_id] += 1
                        slot_total_tasks_per_center[region_id] += 1
                        micro_total_tasks_per_center[region_id] += 1
                        micro_new_tasks += 1
                        event_new_tasks += 1

                    if event_new_tasks > 0:
                        micro_event_count += 1
                        run_visible_assignment(decision_time)

            worker_sim.advance_workers_to_time(centers, micro_end_seconds)

            # At the tick boundary, workers that just became idle may process only the
            # already-arrived backlog. No tasks from the next minute are visible here.
            force_final_dispatch = (slot_idx == num_slots - 1 and micro_end_seconds >= slot_end_seconds)
            run_visible_assignment(micro_end_seconds, force=force_final_dispatch)

            cleanup_time = float(micro_end_seconds)
            if (
                slot_idx == num_slots - 1
                and micro_end_seconds >= slot_end_seconds
                and bool(getattr(config, 'ONLINE_DRAIN_AFTER_BATCH_END', True))
                and _count_unassigned_pool_tasks(unassigned_tasks_pool) > 0
            ):
                drain_until = _resolve_assignment_service_horizon(
                    tasks_per_center=unassigned_tasks_pool,
                    slot_start_seconds=slot_start_seconds,
                    slot_end_seconds=slot_end_seconds,
                )
                drain_time = float(micro_end_seconds)
                drain_assigned = 0
                drain_steps = 0
                max_drain_steps = max(1, int(getattr(config, 'ONLINE_DRAIN_MAX_STEPS', 2000)))
                while (
                    drain_steps < max_drain_steps
                    and drain_time <= drain_until + 1e-6
                    and _count_unassigned_pool_tasks(unassigned_tasks_pool) > 0
                ):
                    worker_sim.advance_workers_to_time(centers, drain_time)
                    slot_expired_count += _expire_unassigned_tasks(unassigned_tasks_pool, drain_time)
                    if _count_unassigned_pool_tasks(unassigned_tasks_pool) <= 0:
                        break

                    assigned_now = run_visible_assignment(drain_time, force=True)
                    drain_steps += 1
                    if assigned_now:
                        drain_assigned += assigned_now
                        continue

                    next_ready_time = _next_worker_service_completion_time(
                        worker_sim,
                        current_time=drain_time,
                        horizon_time=drain_until,
                    )
                    future_expire_times = [
                        float(task[3])
                        for pool in unassigned_tasks_pool.values()
                        for task in pool
                        if float(task[3]) > drain_time + 1e-6
                    ]
                    idle_tick = max(1.0, float(getattr(config, 'ONLINE_DRAIN_IDLE_TICK_SECONDS', 30.0)))
                    next_tick_time = min(drain_time + idle_tick, drain_until)
                    if future_expire_times:
                        next_tick_time = min(next_tick_time, min(future_expire_times))
                    next_candidates = [next_tick_time]
                    if next_ready_time is not None:
                        next_candidates.append(next_ready_time)
                    valid_next_times = [t for t in next_candidates if t > drain_time + 1e-6]
                    if not valid_next_times:
                        break
                    next_decision_time = min(valid_next_times)
                    drain_time = next_decision_time

                if drain_assigned > 0:
                    print(
                        f"   [批后续跑] 继续到 {int(drain_time // 3600):02d}:{int((drain_time % 3600) // 60):02d}:{int(drain_time % 60):02d}, "
                        f"补充分配 {drain_assigned} 单"
                    )
                cleanup_time = max(cleanup_time, drain_time)

            for rid in centers.keys():
                new_pool = []
                for t in unassigned_tasks_pool[rid]:
                    if cleanup_time >= t[3]:
                        slot_expired_count += 1
                        continue
                    new_pool.append(t)
                unassigned_tasks_pool[rid] = new_pool

            total_workers = sum(
                len(worker_sim.get_available_workers_with_center_info(rid, current_time=micro_end_seconds))
                for rid in centers.keys()
            )
            total_current_tasks = sum(len(unassigned_tasks_pool[rid]) for rid in centers.keys())
            if total_current_tasks > 0 or micro_new_tasks > 0 or micro_assignments:
                total_micro = len(range(slot_start_seconds, slot_end_seconds, micro_batch_seconds))
                progress_label = "事件在线" if is_intrabatch_online else "事件微调"
                print(
                    f"   [{progress_label} {micro_idx + 1}/{total_micro}] 可用工人: {total_workers} 个 | "
                    f"秒级到达: {micro_new_tasks} 个 | 触发时刻: {micro_event_count} 个 | 池内总单量: {total_current_tasks} 个"
                )
                if micro_assignments:
                    print(
                        f"      分配结果: 成交 {_count_unique_assigned_task_ids(micro_assignments)} 单, "
                        f"调度 {len(set(k[0] for k in micro_assignments.keys()))} 名工人"
                    )

            if (
                batch_fine_tune_enabled
                and total_micro > 1
                and retention_game_state is not None
                and algo_name.lower() in RETENTION_RL_ALGOS
                and current_slot_rbg_transitions
                and sum(micro_total_tasks_per_center.values()) > 0
            ):
                next_available_workers = {
                    rid: len(worker_sim.get_available_workers_with_center_info(rid, current_time=micro_end_seconds))
                    for rid in centers.keys()
                }
                hydrate_retention_transitions_with_next_state(
                    state=retention_game_state,
                    transitions=current_slot_rbg_transitions,
                    demand_profile=current_slot_rbg_demand_profile,
                    desired_workers=current_slot_rbg_desired_workers,
                    available_workers=next_available_workers,
                    backlog_counts={rid: len(unassigned_tasks_pool[rid]) for rid in centers.keys()},
                    max_tasks_per_worker=int(getattr(config, 'MAX_TASKS_PER_WORKER', 4)),
                    min_buffer_workers=int(getattr(config, 'UABG_MIN_BUFFER_WORKERS', 1)),
                    backlog_weight=float(getattr(config, 'UABG_BACKLOG_WEIGHT', 1.0)),
                    done=float(slot_idx == num_slots - 1 and micro_idx == total_micro - 1),
                )
                last_rbg_reward_by_region = update_rl_retention_bilateral_state(
                    state=retention_game_state,
                    transitions=current_slot_rbg_transitions,
                    assigned_tasks_by_region=micro_assigned_tasks_per_center,
                    total_tasks_by_region=micro_total_tasks_per_center,
                    hoard_penalty_by_region=current_slot_rbg_hoard_penalty,
                    move_cost_by_region=current_slot_rbg_move_cost,
                    moves=current_slot_rbg_moves,
                    hoard_penalty_weight=float(getattr(config, 'RBG_REWARD_HOARD_WEIGHT', 0.02)),
                    move_cost_weight=float(getattr(config, 'RBG_REWARD_MOVE_WEIGHT', 0.08)),
                    unfairness_weight=float(getattr(config, 'RBG_REWARD_UNFAIRNESS_WEIGHT', 1.0)),
                )
                if total_micro > 1:
                    rl_batch_label = 'Platform-RL-Batch' if algo_name.lower() in PREDICTIVE_PLATFORM_RL_ALGOS else 'RBG-Batch'
                    print(f"   [{rl_batch_label}] updated after batch {micro_idx + 1}/{total_micro}")

            if (
                batch_fine_tune_enabled
                and total_micro > 1
                and platform_rl_state is not None
                and algo_name.lower() in PREDICTIVE_PLATFORM_RL_ALGOS
                and current_slot_platform_transition is not None
                and sum(micro_total_tasks_per_center.values()) > 0
            ):
                next_available_workers = {
                    rid: len(worker_sim.get_available_workers_with_center_info(rid, current_time=micro_end_seconds))
                    for rid in centers.keys()
                }
                hydrate_platform_transition_with_next_state(
                    state=platform_rl_state,
                    transition=current_slot_platform_transition,
                    available_workers=next_available_workers,
                    backlog_counts={rid: len(unassigned_tasks_pool[rid]) for rid in centers.keys()},
                    max_tasks_per_worker=int(getattr(config, 'MAX_TASKS_PER_WORKER', 4)),
                    backlog_weight=float(getattr(config, 'UABG_BACKLOG_WEIGHT', 1.0)),
                    done=float(slot_idx == num_slots - 1 and micro_idx == total_micro - 1),
                )
                last_platform_stats = update_platform_task_first_state(
                    state=platform_rl_state,
                    transition=current_slot_platform_transition,
                    assigned_tasks_by_region=micro_assigned_tasks_per_center,
                    total_tasks_by_region=micro_total_tasks_per_center,
                    fairness_secondary_weight=current_slot_platform_fairness_weight,
                )
                if total_micro > 1:
                    print(f"   [Platform-RL-Batch] policy updated after batch {micro_idx + 1}/{total_micro}")

        total_expired_tasks_global += slot_expired_count
        leftover_count = sum(len(pool) for pool in unassigned_tasks_pool.values())

        if current_slot_predicted_demand is not None and algo_name.lower() not in [*NO_PRED_RBG_ALGOS, *NO_PRED_GAME_ALGOS]:
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
            if retention_game_state is not None and algo_name.lower() in RETENTION_RL_ALGOS:
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
        if center_task_rl_state is not None and current_slot_center_transitions:
            center_rewards = update_center_task_allocation_state(
                state=center_task_rl_state,
                transitions=current_slot_center_transitions,
                assigned_tasks_by_region=slot_assigned_tasks_per_center,
                total_tasks_by_region=slot_total_tasks_per_center,
                remaining_tasks_by_region={rid: len(unassigned_tasks_pool[rid]) for rid in centers.keys()},
                actual_arrivals_by_region=slot_new_tasks_per_center,
                predicted_demand_by_region=current_slot_predicted_demand,
            )
            reward_text = ", ".join(
                f"R{rid}: reward={center_rewards.get(rid, 0.0):.2f}"
                for rid in sorted(centers.keys())
            )
            print(f"   [Event-Center-RL Reward] {reward_text}")
        if retention_game_state is not None and algo_name.lower() in RETENTION_RL_ALGOS and current_slot_rbg_transitions:
                next_available_workers = {
                    rid: len(worker_sim.get_available_workers_with_center_info(rid, current_time=slot_end_seconds))
                    for rid in centers.keys()
                }
                hydrate_retention_transitions_with_next_state(
                    state=retention_game_state,
                    transitions=current_slot_rbg_transitions,
                    demand_profile=current_slot_rbg_demand_profile,
                    desired_workers=current_slot_rbg_desired_workers,
                    available_workers=next_available_workers,
                    backlog_counts={rid: len(unassigned_tasks_pool[rid]) for rid in centers.keys()},
                    max_tasks_per_worker=int(getattr(config, 'MAX_TASKS_PER_WORKER', 4)),
                    min_buffer_workers=int(getattr(config, 'UABG_MIN_BUFFER_WORKERS', 1)),
                    backlog_weight=float(getattr(config, 'UABG_BACKLOG_WEIGHT', 1.0)),
                    done=float(slot_idx == num_slots - 1),
                )
                if batch_fine_tune_enabled and total_micro > 1:
                    reward_by_region = last_rbg_reward_by_region or {}
                else:
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
                if algo_name.lower() in PREDICTIVE_EVENT_RL_GAME_ALGOS:
                    reward_label = 'Event-RL-Game Reward'
                elif algo_name.lower() in PREDICTIVE_PLATFORM_RL_ALGOS:
                    reward_label = 'Platform-RL-MCTGNet Reward'
                elif algo_name.lower() in NO_PRED_RBG_ALGOS:
                    reward_label = 'NoPred-RL-Game Reward'
                else:
                    reward_label = 'RBG-MCTGNet Reward'
                print(f"   [{reward_label}] {reward_text}")

        if platform_rl_state is not None and algo_name.lower() in PREDICTIVE_PLATFORM_RL_ALGOS and current_slot_platform_transition is not None:
            next_available_workers = {
                rid: len(worker_sim.get_available_workers_with_center_info(rid, current_time=slot_end_seconds))
                for rid in centers.keys()
            }
            hydrate_platform_transition_with_next_state(
                state=platform_rl_state,
                transition=current_slot_platform_transition,
                available_workers=next_available_workers,
                backlog_counts={rid: len(unassigned_tasks_pool[rid]) for rid in centers.keys()},
                max_tasks_per_worker=int(getattr(config, 'MAX_TASKS_PER_WORKER', 4)),
                backlog_weight=float(getattr(config, 'UABG_BACKLOG_WEIGHT', 1.0)),
                done=float(slot_idx == num_slots - 1),
            )
            if batch_fine_tune_enabled and total_micro > 1:
                platform_stats = last_platform_stats or {
                    "platform_reward": 0.0,
                    "completion_rate": 0.0,
                    "mean_unfairness": 0.0,
                }
            else:
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

        print(f"分配结果: 成交 {_count_unique_assigned_task_ids(slot_assignments)} 单, 调度 {len(slot_worker_ids)} 名工人")
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
    algo_key = algo_name.lower()
    execution_mode = "事件级在线"
    effective_micro_batch_minutes = int(
        getattr(config, 'EXPERIMENT_MICRO_BATCH_MINUTES', time_slot_minutes)
    )
    if algo_key in RETENTION_RL_ALGOS:
        effective_micro_batch_minutes = int(
            getattr(config, 'RBG_INTERNAL_MICRO_BATCH_MINUTES', time_slot_minutes)
        )
    effective_micro_batch_minutes = max(1, min(int(time_slot_minutes), int(effective_micro_batch_minutes)))
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
        f"{compare_end_hour}:{compare_end_minute:02d} | 时间槽：{time_slot_minutes} 分钟 | 模式：{execution_mode} | 工人调分布步长：{effective_micro_batch_minutes} 分钟"
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
    scope = context.get('scope', _current_scope_metadata())
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

    if algo_name.lower() in PREDICTIVE_EVENT_RL_GAME_ALGOS:
        dispatch_predictor = _get_or_train_event_task_predictor(
            test_date=test_date,
            test_start_hour=test_start_hour,
            test_end_hour=test_end_hour,
            time_slot_minutes=time_slot_minutes,
            coords=context['coords'],
            nodes=context['nodes'],
            rcc_partition=rcc_partition,
            centers=centers
        )
    elif algo_name.lower() in ['predictive_mctgnet', 'predictive_game_mctgnet', 'predictive_bstgcnet', *PREDICTIVE_UABG_ALGOS, *PREDICTIVE_RBG_ALGOS, *PREDICTIVE_PLATFORM_RL_ALGOS]:
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
    if algo_name.lower() in RETENTION_RL_ALGOS:
        rl_device = _get_preferred_torch_device()
        state_label = 'Demand calibration continuous SAC' if algo_name.lower() in PREDICTIVE_EVENT_RL_GAME_ALGOS else 'Retention continuous SAC'
        print(f"   - {state_label} device: {rl_device}")
        retention_game_state = RLRetentionBilateralState(
            region_ids=sorted(centers.keys()),
            action_bounds=tuple(getattr(config, 'RBG_CONTINUOUS_ACTION_BOUNDS', ())),
            learning_rate=float(getattr(config, 'RBG_LEARNING_RATE', 0.03)),
            exploration_prob=float(getattr(config, 'RBG_EXPLORATION_PROB', 0.12)),
            gamma=float(getattr(config, 'RBG_SAC_GAMMA', 0.6)),
            replay_capacity=int(getattr(config, 'RBG_SAC_REPLAY_CAPACITY', 512)),
            batch_size=int(getattr(config, 'RBG_SAC_BATCH_SIZE', 32)),
            hidden_dim=int(getattr(config, 'RBG_POLICY_HIDDEN_DIM', 32)),
            critic_learning_rate=float(getattr(config, 'RBG_SAC_CRITIC_LEARNING_RATE', getattr(config, 'RBG_LEARNING_RATE', 0.001))),
            alpha_learning_rate=float(getattr(config, 'RBG_SAC_ALPHA_LEARNING_RATE', 0.001)),
            sac_tau=float(getattr(config, 'RBG_SAC_TAU', 0.01)),
            sac_initial_alpha=float(getattr(config, 'RBG_SAC_INITIAL_ALPHA', 0.10)),
            sac_auto_entropy=bool(getattr(config, 'RBG_SAC_AUTO_ENTROPY', True)),
            sac_target_entropy_scale=float(getattr(config, 'RBG_SAC_TARGET_ENTROPY_SCALE', 0.60)),
            service_debt_decay=float(getattr(config, 'RBG_SERVICE_DEBT_DECAY', 0.85)),
            max_service_debt=float(getattr(config, 'RBG_MAX_SERVICE_DEBT', 4.0)),
            move_history_size=int(getattr(config, 'RBG_MOVE_HISTORY_SIZE', 8)),
            random_seed=int(getattr(config, 'RBG_RANDOM_SEED', 42)),
            device=rl_device,
        )
    platform_rl_state = None
    if algo_name.lower() in PREDICTIVE_PLATFORM_RL_ALGOS and bool(getattr(config, 'PFRL_ENABLE_LEARNING', False)):
        platform_rl_device = _get_preferred_torch_device()
        print(f"   - Platform continuous SAC device: {platform_rl_device}")
        platform_rl_state = PlatformTaskFirstRLState(
            region_ids=sorted(centers.keys()),
            action_bounds=tuple(getattr(config, 'PFRL_CONTINUOUS_ACTION_BOUNDS', ())),
            learning_rate=float(getattr(config, 'PFRL_LEARNING_RATE', 0.03)),
            exploration_prob=float(getattr(config, 'PFRL_EXPLORATION_PROB', 0.10)),
            gamma=float(getattr(config, 'PFRL_SAC_GAMMA', 0.6)),
            replay_capacity=int(getattr(config, 'PFRL_SAC_REPLAY_CAPACITY', 256)),
            batch_size=int(getattr(config, 'PFRL_SAC_BATCH_SIZE', 32)),
            hidden_dim=int(getattr(config, 'PFRL_POLICY_HIDDEN_DIM', getattr(config, 'RBG_POLICY_HIDDEN_DIM', 32))),
            critic_learning_rate=float(getattr(config, 'PFRL_SAC_CRITIC_LEARNING_RATE', getattr(config, 'PFRL_LEARNING_RATE', 0.001))),
            alpha_learning_rate=float(getattr(config, 'PFRL_SAC_ALPHA_LEARNING_RATE', 0.001)),
            sac_tau=float(getattr(config, 'PFRL_SAC_TAU', 0.01)),
            sac_initial_alpha=float(getattr(config, 'PFRL_SAC_INITIAL_ALPHA', 0.10)),
            sac_auto_entropy=bool(getattr(config, 'PFRL_SAC_AUTO_ENTROPY', True)),
            sac_target_entropy_scale=float(getattr(config, 'PFRL_SAC_TARGET_ENTROPY_SCALE', 0.60)),
            random_seed=int(getattr(config, 'PFRL_RANDOM_SEED', 7)),
            device=platform_rl_device,
        )
    center_task_rl_state = None
    if algo_name.lower() in PREDICTIVE_EVENT_RL_GAME_ALGOS:
        center_rl_device = _get_preferred_torch_device()
        print(f"   - Center task-allocation continuous SAC device: {center_rl_device}")
        center_task_rl_state = CenterTaskAllocationRLState(
            region_ids=sorted(centers.keys()),
            action_bounds=tuple(getattr(config, 'EVENT_CENTER_CONTINUOUS_ACTION_BOUNDS', ())),
            learning_rate=float(getattr(config, 'EVENT_CENTER_RL_LEARNING_RATE', 0.03)),
            exploration_prob=float(getattr(config, 'EVENT_CENTER_RL_EXPLORATION_PROB', 0.02)),
            gamma=float(getattr(config, 'EVENT_CENTER_SAC_GAMMA', 0.60)),
            replay_capacity=int(getattr(config, 'EVENT_CENTER_SAC_REPLAY_CAPACITY', 512)),
            batch_size=int(getattr(config, 'EVENT_CENTER_SAC_BATCH_SIZE', 32)),
            hidden_dim=int(getattr(config, 'RBG_POLICY_HIDDEN_DIM', 32)),
            critic_learning_rate=float(getattr(config, 'EVENT_CENTER_SAC_CRITIC_LEARNING_RATE', getattr(config, 'EVENT_CENTER_RL_LEARNING_RATE', 0.001))),
            alpha_learning_rate=float(getattr(config, 'EVENT_CENTER_SAC_ALPHA_LEARNING_RATE', 0.001)),
            sac_tau=float(getattr(config, 'EVENT_CENTER_SAC_TAU', 0.01)),
            sac_initial_alpha=float(getattr(config, 'EVENT_CENTER_SAC_INITIAL_ALPHA', 0.10)),
            sac_auto_entropy=bool(getattr(config, 'EVENT_CENTER_SAC_AUTO_ENTROPY', True)),
            sac_target_entropy_scale=float(getattr(config, 'EVENT_CENTER_SAC_TARGET_ENTROPY_SCALE', 0.60)),
            random_seed=int(getattr(config, 'RBG_RANDOM_SEED', 42)),
            device=center_rl_device,
        )
    if retention_game_state is not None:
        _offline_warm_start_rbg_state(
            state=retention_game_state,
            platform_state=platform_rl_state,
            center_task_state=center_task_rl_state,
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
            algo_name=algo_name,
        )

    cpu_start = time.process_time()
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
        center_task_rl_state=center_task_rl_state,
    )

    unique_assigned_tasks, assigned_tasks_per_center, duplicate_assigned_details = _summarize_unique_assigned_tasks(
        all_details,
        centers.keys(),
    )

    rho, u_rho = calculate_collaboration_unfairness(total_tasks_per_center, assigned_tasks_per_center)
    cpu_time = time.process_time() - cpu_start
    total_tasks_in_scope = int(sum(total_tasks_per_center.values()))
    assigned_task_count = len(unique_assigned_tasks)
    assignment_pair_count = len(all_assignments)
    task_completion_rate = (assigned_task_count / total_tasks_in_scope) if total_tasks_in_scope > 0 else 0.0

    print(f"\n[{algo_name.upper()}] 仿真完成:")
    print(f"  - #Assigned Tasks: {assigned_task_count}")
    print(f"  - #Total Tasks: {total_tasks_in_scope}")
    if duplicate_assigned_details or assignment_pair_count != assigned_task_count:
        print(
            f"  - Assignment de-dup: pairs={assignment_pair_count}, "
            f"duplicate_details={duplicate_assigned_details}"
        )
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
        'assigned_tasks': assigned_task_count,
        'total_tasks': total_tasks_in_scope,
        'map_center_lat': scope.get('map_center_lat'),
        'map_center_lon': scope.get('map_center_lon'),
        'download_dist_m': scope.get('download_dist_m'),
        'map_download_dist_m': scope.get('map_download_dist_m'),
        'map_size_km': scope.get('map_size_km'),
        'map_side_km': scope.get('map_side_km'),
        'center_count': scope.get('center_count'),
        'worker_speed_kmh': scope.get('worker_speed_kmh'),
        'worker_speed_ms': scope.get('worker_speed_ms'),
        'assignment_pairs': assignment_pair_count,
        'duplicate_assigned_details': duplicate_assigned_details,
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
                    post_service_count = sum(1 for m in predispatch_result['moves'] if m.get('post_service'))
                    post_service_note = f" (完成后支援 {post_service_count} 名)" if post_service_count else ""
                    print(
                        f"   [{predict_label}] pre-dispatched {len(predispatch_result['moves'])} workers"
                        f"{post_service_note}: {move_summary}"
                    )
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
            slot_assignments, slot_profit, slot_details = center_prepacked_assignment_with_center_pickup(
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
                collaboration_mode=IMTAO_MODE_RBDC,
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
            f"分配结果: 成交 {_count_unique_assigned_task_ids(slot_assignments)} 单, 调度 {len(set(k[0] for k in slot_assignments.keys()))} 名工人")

        if slot_expired_count > 0:
            print(f"   ❌ 超时淘汰订单: {slot_expired_count} 个 (已自动取消该订单，不扣除利润)")

        if leftover_count > 0:
            print(f"   ⏳ 剩余积压订单: {leftover_count} 个 (安全范围内，自动滚入下一轮)")

        total_dist_to_center += slot_dist_to_center
        total_dist_to_task += slot_dist_to_task
        all_assignments.update(slot_assignments)
        all_details.extend(slot_details)
        total_profit += slot_profit

    unique_assigned_tasks, assigned_tasks_per_center, duplicate_assigned_details = _summarize_unique_assigned_tasks(
        all_details,
        centers.keys(),
    )

    rho, u_rho = calculate_collaboration_unfairness(total_tasks_per_center, assigned_tasks_per_center)
    cpu_time = time.process_time() - cpu_start
    total_tasks_in_scope = int(sum(total_tasks_per_center.values()))
    assigned_task_count = len(unique_assigned_tasks)
    assignment_pair_count = len(all_assignments)
    task_completion_rate = (assigned_task_count / total_tasks_in_scope) if total_tasks_in_scope > 0 else 0.0

    print(f"\n[{algo_name.upper()}] 仿真完成:")
    print(f"  - #Assigned Tasks: {assigned_task_count}")
    print(f"  - #Total Tasks: {total_tasks_in_scope}")
    if duplicate_assigned_details or assignment_pair_count != assigned_task_count:
        print(
            f"  - Assignment de-dup: pairs={assignment_pair_count}, "
            f"duplicate_details={duplicate_assigned_details}"
        )
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
        'assigned_tasks': assigned_task_count,
        'total_tasks': total_tasks_in_scope,
        'map_center_lat': scope.get('map_center_lat'),
        'map_center_lon': scope.get('map_center_lon'),
        'download_dist_m': scope.get('download_dist_m'),
        'map_download_dist_m': scope.get('map_download_dist_m'),
        'map_size_km': scope.get('map_size_km'),
        'map_side_km': scope.get('map_side_km'),
        'center_count': scope.get('center_count'),
        'worker_speed_kmh': scope.get('worker_speed_kmh'),
        'worker_speed_ms': scope.get('worker_speed_ms'),
        'assignment_pairs': assignment_pair_count,
        'duplicate_assigned_details': duplicate_assigned_details,
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
        algo_name='no_pred_rl_game',
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
        algo_name='predictive_platform_rl_mctgnet',
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
