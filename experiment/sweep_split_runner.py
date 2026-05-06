import csv
import os
from typing import Dict

import batch as batch_exp


WORKER_COUNTS = [500, 600, 700, 800, 900]
BATCH_COUNTS = [2, 4, 6, 8, 10]
CENTER_COUNTS = [3, 4, 5, 6, 7]
FIXED_WORKER_COUNT = 500
ALGORITHMS = [
    # ("greedy", "Greedy"),
    # ("imtao", "IMTAO (Seq-BDC)"),
    # ("game_only_dispatch", "NoPred-Game"),
    # ("predictive_mctgnet", "Predictive-MCTGNet"),
    # ("predictive_game_mctgnet", "Game-MCTGNet"),
    # ("predictive_uabg_mctgnet", "UABG-MCTGNet"),
    # ("predictive_rl_game_mctgnet", "RBG-MCTGNet"),
    ("predictive_platform_rl_mctgnet", "Platform-RL-MCTGNet"),
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_ROOT_DIR = os.path.join(PROJECT_ROOT, "result")
WORKER_SPLIT_RESULTS_DIR = os.path.join(RESULT_ROOT_DIR, "worker_count_split")
BATCH_SPLIT_RESULTS_DIR = os.path.join(RESULT_ROOT_DIR, "batch_count_split")
CENTER_SPLIT_RESULTS_DIR = os.path.join(RESULT_ROOT_DIR, "center_count_split")


def _fmt_optional(value):
    return f"{value:.4f}" if value is not None else "-"


def _ensure_output_dir(output_dir: str = None) -> str:
    resolved_dir = output_dir or RESULT_ROOT_DIR
    os.makedirs(resolved_dir, exist_ok=True)
    return resolved_dir


def _run_algorithms(
    worker_limit: int,
    batch_count: int,
    center_count: int = None,
    download_dist: int = None,
) -> Dict[str, dict]:
    original_worker_limit = batch_exp.DEFAULT_WORKER_LIMIT
    original_compare_slot_count = batch_exp.DEFAULT_COMPARE_SLOT_COUNT
    original_num_zones = batch_exp.config.NUM_ZONES
    original_download_dist = batch_exp.config.DOWNLOAD_DIST

    resolved_center_count = (
        int(center_count)
        if center_count is not None
        else int(getattr(batch_exp.config, "DEFAULT_NUM_ZONES", original_num_zones))
    )
    resolved_download_dist = (
        int(download_dist)
        if download_dist is not None
        else int(getattr(batch_exp.config, "DEFAULT_DOWNLOAD_DIST", original_download_dist))
    )

    batch_exp._MCTG_PREDICTOR_CACHE.clear()
    batch_exp.DEFAULT_WORKER_LIMIT = worker_limit
    batch_exp.DEFAULT_COMPARE_SLOT_COUNT = batch_count
    batch_exp.config.NUM_ZONES = resolved_center_count
    batch_exp.config.DOWNLOAD_DIST = resolved_download_dist
    # Important: the simulation context cache key does not include worker_limit.
    batch_exp._SIMULATION_CONTEXT_CACHE.clear()

    try:
        results = {}
        for algo_name, display_name in ALGORITHMS:
            _, _, metrics = batch_exp.run_online_simulation_with_center_pickup(
                algo_name=algo_name,
                test_date=batch_exp.DEFAULT_TEST_DATE,
                test_start_hour=batch_exp.DEFAULT_START_HOUR,
                test_end_hour=batch_exp.DEFAULT_END_HOUR,
                time_slot_minutes=batch_exp.DEFAULT_TIME_SLOT_MINUTES,
            )
            results[display_name] = metrics
        return results
    finally:
        batch_exp.DEFAULT_WORKER_LIMIT = original_worker_limit
        batch_exp.DEFAULT_COMPARE_SLOT_COUNT = original_compare_slot_count
        batch_exp.config.NUM_ZONES = original_num_zones
        batch_exp.config.DOWNLOAD_DIST = original_download_dist


def _write_single_setting_csv(
    output_path: str,
    worker_count: int,
    batch_count: int,
    algo_results: Dict[str, dict],
    center_count: int = None,
):
    fieldnames = [
        "worker_count",
        "batch_count",
        "center_count",
        "algorithm",
        "assigned_tasks",
        "task_completion_rate",
        "u_rho",
        "cpu_time",
        "pred_mae",
        "pred_rmse",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for algorithm, metrics in algo_results.items():
            writer.writerow(
                {
                    "worker_count": worker_count,
                    "batch_count": batch_count,
                    "center_count": center_count,
                    "algorithm": algorithm,
                    "assigned_tasks": metrics["assigned_tasks"],
                    "task_completion_rate": metrics["task_completion_rate"],
                    "u_rho": metrics["u_rho"],
                    "cpu_time": metrics["cpu_time"],
                    "pred_mae": metrics.get("pred_mae"),
                    "pred_rmse": metrics.get("pred_rmse"),
                }
            )


def _print_summary(title: str, worker_count: int, batch_count: int, algo_results: Dict[str, dict]):
    print("\n" + "=" * 130)
    print(title)
    print(f"Worker Count: {worker_count} | Batch Count: {batch_count}")
    print("=" * 130)
    print(
        f"{'Algorithm':<22} | {'#Assigned Tasks':<16} | {'Task Completion Rate':<22} | {'Collaboration Unfairness':<26} | "
        f"{'CPU Time (s)':<14} | {'Prediction MAE':<14} | {'Prediction RMSE':<14}"
    )
    print("-" * 130)
    for algorithm, metrics in algo_results.items():
        print(
            f"{algorithm:<22} | "
            f"{metrics['assigned_tasks']:<16} | "
            f"{metrics['task_completion_rate']:<22.4f} | "
            f"{metrics['u_rho']:<26.4f} | "
            f"{metrics['cpu_time']:<14.4f} | "
            f"{_fmt_optional(metrics.get('pred_mae')):<14} | "
            f"{_fmt_optional(metrics.get('pred_rmse')):<14}"
        )
    print("=" * 130)


def run_worker_setting(
    worker_count: int,
    batch_count: int = None,
    output_dir: str = None,
) -> str:
    resolved_batch_count = batch_count if batch_count is not None else batch_exp.DEFAULT_COMPARE_SLOT_COUNT
    algo_results = _run_algorithms(worker_count, resolved_batch_count)
    resolved_output_dir = _ensure_output_dir(output_dir or WORKER_SPLIT_RESULTS_DIR)
    output_path = os.path.join(resolved_output_dir, f"worker_count_{worker_count}.csv")
    _write_single_setting_csv(output_path, worker_count, resolved_batch_count, algo_results)
    _print_summary("Worker Count Split Run", worker_count, resolved_batch_count, algo_results)
    return output_path


def run_batch_setting(
    batch_count: int,
    worker_count: int = FIXED_WORKER_COUNT,
    output_dir: str = None,
) -> str:
    algo_results = _run_algorithms(worker_count, batch_count)
    resolved_output_dir = _ensure_output_dir(output_dir or BATCH_SPLIT_RESULTS_DIR)
    output_path = os.path.join(resolved_output_dir, f"batch_count_{batch_count}.csv")
    _write_single_setting_csv(output_path, worker_count, batch_count, algo_results)
    _print_summary("Batch Count Split Run", worker_count, batch_count, algo_results)
    return output_path


def run_center_count_setting(
    center_count: int,
    worker_count: int = FIXED_WORKER_COUNT,
    batch_count: int = None,
    output_dir: str = None,
) -> str:
    resolved_batch_count = batch_count if batch_count is not None else batch_exp.DEFAULT_COMPARE_SLOT_COUNT
    resolved_output_dir = _ensure_output_dir(output_dir or CENTER_SPLIT_RESULTS_DIR)
    algo_results = _run_algorithms(
        worker_count,
        resolved_batch_count,
        center_count=int(center_count),
    )

    output_path = os.path.join(resolved_output_dir, f"center_count_{int(center_count)}.csv")
    _write_single_setting_csv(
        output_path,
        worker_count,
        resolved_batch_count,
        algo_results,
        center_count=int(center_count),
    )
    _print_summary(
        f"Center Count Split Run (centers={int(center_count)})",
        worker_count,
        resolved_batch_count,
        algo_results,
    )
    return output_path
