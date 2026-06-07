import csv
import os

import batch as batch_exp
from small_scale_config import (
    SMALL_SCALE_CENTER_COUNT,
    SMALL_SCALE_DOWNLOAD_DIST_M,
    SMALL_SCALE_SIDE_LENGTH_KM,
    SMALL_SCALE_WORKER_COUNTS,
    WORKER_RESULT_DIR,
)


ALGORITHMS = [
    ("greedy", "Greedy"),
    ("imtao", "IMTAO (Seq-BDC)"),
    ("no_pred_rl_game", "NoPred-RL-Game"),
    ("predictive_mctgnet", "Predictive-MCTGNet"),
    ("predictive_platform_rl_mctgnet", "Platform-RL-MCTGNet"),
    ("predictive_event_rl_game", "Event-RL-Game"),
]


def _fmt_optional(value):
    return f"{value:.4f}" if value is not None else "-"


def run_single_setting(worker_count: int):
    batch_exp._MCTG_PREDICTOR_CACHE.clear()
    original_worker_limit = batch_exp.DEFAULT_WORKER_LIMIT
    batch_exp.DEFAULT_WORKER_LIMIT = worker_count
    batch_exp._SIMULATION_CONTEXT_CACHE.clear()

    original_download_dist = batch_exp.config.DOWNLOAD_DIST
    original_num_zones = batch_exp.config.NUM_ZONES
    batch_exp.config.DOWNLOAD_DIST = SMALL_SCALE_DOWNLOAD_DIST_M
    batch_exp.config.NUM_ZONES = SMALL_SCALE_CENTER_COUNT
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
        batch_exp.config.DOWNLOAD_DIST = original_download_dist
        batch_exp.config.NUM_ZONES = original_num_zones
        batch_exp._SIMULATION_CONTEXT_CACHE.clear()


def write_csv(results_by_worker_count, output_path: str):
    fieldnames = [
        "map_size_km",
        "download_dist_m",
        "center_count",
        "worker_count",
        "batch_count",
        "worker_speed_kmh",
        "worker_speed_ms",
        "algorithm",
        "assigned_tasks",
        "total_tasks",
        "task_completion_rate",
        "u_rho",
        "cpu_time",
        "pred_mae",
        "pred_rmse",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for worker_count, algo_results in results_by_worker_count.items():
            for algorithm, metrics in algo_results.items():
                writer.writerow(
                    {
                        "map_size_km": SMALL_SCALE_SIDE_LENGTH_KM,
                        "download_dist_m": metrics.get("download_dist_m", SMALL_SCALE_DOWNLOAD_DIST_M),
                        "center_count": metrics.get("center_count", SMALL_SCALE_CENTER_COUNT),
                        "worker_count": worker_count,
                        "batch_count": batch_exp.DEFAULT_COMPARE_SLOT_COUNT,
                        "worker_speed_kmh": metrics.get("worker_speed_kmh"),
                        "worker_speed_ms": metrics.get("worker_speed_ms"),
                        "algorithm": algorithm,
                        "assigned_tasks": metrics["assigned_tasks"],
                        "total_tasks": metrics.get("total_tasks"),
                        "task_completion_rate": metrics["task_completion_rate"],
                        "u_rho": metrics["u_rho"],
                        "cpu_time": metrics["cpu_time"],
                        "pred_mae": metrics.get("pred_mae"),
                        "pred_rmse": metrics.get("pred_rmse"),
                    }
                )


def print_summary(results_by_worker_count):
    print("\n" + "=" * 150)
    print(
        f"Small-Scale Worker Sweep | Map Size = {SMALL_SCALE_SIDE_LENGTH_KM}km x {SMALL_SCALE_SIDE_LENGTH_KM}km "
        f"| dist = {SMALL_SCALE_DOWNLOAD_DIST_M}m | centers = {SMALL_SCALE_CENTER_COUNT}"
    )
    print("=" * 150)
    for worker_count, algo_results in results_by_worker_count.items():
        print(f"\n[Worker Count = {worker_count}]")
        print("-" * 150)
        print(
            f"{'Algorithm':<22} | {'#Assigned Tasks':<16} | {'Task Completion Rate':<22} | "
            f"{'Collaboration Unfairness':<26} | {'CPU Time (s)':<14} | {'Prediction MAE':<14} | {'Prediction RMSE':<14}"
        )
        print("-" * 150)
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
    print("=" * 150)


def main():
    results_by_worker_count = {}
    os.makedirs(WORKER_RESULT_DIR, exist_ok=True)
    for worker_count in SMALL_SCALE_WORKER_COUNTS:
        print("\n" + "#" * 90)
        print(
            f"Running small-scale worker sweep | map={SMALL_SCALE_SIDE_LENGTH_KM}km x {SMALL_SCALE_SIDE_LENGTH_KM}km "
            f"| centers={SMALL_SCALE_CENTER_COUNT} | workers={worker_count}"
        )
        print("#" * 90)
        results_by_worker_count[worker_count] = run_single_setting(worker_count)

    output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_small_scale_worker_sweep_results.csv")
    write_csv(results_by_worker_count, output_csv)
    print_summary(results_by_worker_count)
    print(f"\nCSV results saved to: {output_csv}")


if __name__ == "__main__":
    main()
