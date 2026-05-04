import argparse
import csv
import os

import batch as batch_exp
from small_scale_config import (
    BATCH_RESULT_DIR,
    SMALL_SCALE_BATCH_COUNTS,
    SMALL_SCALE_DOWNLOAD_DIST_M,
    SMALL_SCALE_FIXED_WORKER_COUNT,
    SMALL_SCALE_SIDE_LENGTH_KM,
)


ALGORITHMS = [
    ("greedy", "Greedy"),
    ("imtao", "IMTAO (Seq-BDC)"),
    ("game_only_dispatch", "NoPred-Game"),
    ("predictive_mctgnet", "Predictive-MCTGNet"),
    ("predictive_game_mctgnet", "Game-MCTGNet"),
    ("predictive_uabg_mctgnet", "UABG-MCTGNet"),
    ("predictive_rl_game_mctgnet", "RBG-MCTGNet"),
]


def main_for_batch_count(batch_count: int, output_dir: str = None):
    batch_exp._MCTG_PREDICTOR_CACHE.clear()
    batch_exp.DEFAULT_WORKER_LIMIT = SMALL_SCALE_FIXED_WORKER_COUNT
    batch_exp.DEFAULT_COMPARE_SLOT_COUNT = batch_count
    batch_exp._SIMULATION_CONTEXT_CACHE.clear()

    original_download_dist = batch_exp.config.DOWNLOAD_DIST
    try:
        batch_exp.config.DOWNLOAD_DIST = SMALL_SCALE_DOWNLOAD_DIST_M
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
        batch_exp.config.DOWNLOAD_DIST = original_download_dist

    resolved_output_dir = output_dir or BATCH_RESULT_DIR
    os.makedirs(resolved_output_dir, exist_ok=True)
    output_path = os.path.join(resolved_output_dir, f"batch_count_{batch_count}.csv")

    fieldnames = [
        "map_size_km",
        "download_dist_m",
        "worker_count",
        "batch_count",
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
                    "map_size_km": SMALL_SCALE_SIDE_LENGTH_KM,
                    "download_dist_m": SMALL_SCALE_DOWNLOAD_DIST_M,
                    "worker_count": SMALL_SCALE_FIXED_WORKER_COUNT,
                    "batch_count": batch_count,
                    "algorithm": algorithm,
                    "assigned_tasks": metrics["assigned_tasks"],
                    "task_completion_rate": metrics["task_completion_rate"],
                    "u_rho": metrics["u_rho"],
                    "cpu_time": metrics["cpu_time"],
                    "pred_mae": metrics.get("pred_mae"),
                    "pred_rmse": metrics.get("pred_rmse"),
                }
            )

    print(f"\nCSV results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run one 3x3km small-scale batch-count setting.")
    parser.add_argument("batch_count", type=int, choices=SMALL_SCALE_BATCH_COUNTS, help="Batch count to evaluate.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store the per-setting CSV file.",
    )
    args = parser.parse_args()
    main_for_batch_count(args.batch_count, args.output_dir)


if __name__ == "__main__":
    main()
