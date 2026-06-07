import argparse
import csv
import os

import batch as batch_exp
from small_scale_config import (
    SMALL_SCALE_CENTER_COUNT,
    SMALL_SCALE_DOWNLOAD_DIST_M,
    SMALL_SCALE_SIDE_LENGTH_KM,
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


def main_for_worker_count(worker_count: int, output_dir: str = None):
    batch_exp._MCTG_PREDICTOR_CACHE.clear()
    original_worker_limit = batch_exp.DEFAULT_WORKER_LIMIT
    batch_exp.DEFAULT_WORKER_LIMIT = worker_count
    batch_exp._SIMULATION_CONTEXT_CACHE.clear()

    original_download_dist = batch_exp.config.DOWNLOAD_DIST
    original_num_zones = batch_exp.config.NUM_ZONES
    try:
        batch_exp.config.DOWNLOAD_DIST = SMALL_SCALE_DOWNLOAD_DIST_M
        batch_exp.config.NUM_ZONES = SMALL_SCALE_CENTER_COUNT
        algo_results = {}
        for algo_name, display_name in ALGORITHMS:
            _, _, metrics = batch_exp.run_online_simulation_with_center_pickup(
                algo_name=algo_name,
                test_date=batch_exp.DEFAULT_TEST_DATE,
                test_start_hour=batch_exp.DEFAULT_START_HOUR,
                test_end_hour=batch_exp.DEFAULT_END_HOUR,
                time_slot_minutes=batch_exp.DEFAULT_TIME_SLOT_MINUTES,
            )
            algo_results[display_name] = metrics
    finally:
        batch_exp.DEFAULT_WORKER_LIMIT = original_worker_limit
        batch_exp.config.DOWNLOAD_DIST = original_download_dist
        batch_exp.config.NUM_ZONES = original_num_zones
        batch_exp._SIMULATION_CONTEXT_CACHE.clear()

    resolved_output_dir = output_dir or WORKER_RESULT_DIR
    os.makedirs(resolved_output_dir, exist_ok=True)
    output_path = os.path.join(resolved_output_dir, f"worker_count_{worker_count}.csv")

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

    print(f"\nCSV results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run one 3x3km small-scale worker-count setting.")
    parser.add_argument("worker_count", type=int, help="Worker count to evaluate.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store the per-setting CSV file.",
    )
    args = parser.parse_args()
    main_for_worker_count(args.worker_count, args.output_dir)


if __name__ == "__main__":
    main()
