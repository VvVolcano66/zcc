import csv
import os

import batch as batch_exp


WORKER_COUNTS = [500, 600, 700, 800, 900]
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


def run_single_setting(worker_limit: int):
    original_worker_limit = batch_exp.DEFAULT_WORKER_LIMIT
    batch_exp._MCTG_PREDICTOR_CACHE.clear()
    batch_exp.DEFAULT_WORKER_LIMIT = worker_limit
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


def write_csv(results_by_worker_count, output_path: str):
    fieldnames = [
        "worker_count",
        "batch_count",
        "center_count",
        "map_size_km",
        "download_dist_m",
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
                        "worker_count": worker_count,
                        "batch_count": batch_exp.DEFAULT_COMPARE_SLOT_COUNT,
                        "center_count": metrics.get("center_count"),
                        "map_size_km": metrics.get("map_size_km"),
                        "download_dist_m": metrics.get("download_dist_m"),
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
    print("\n" + "=" * 130)
    print("不同工人数量下的多算法批量实验结果")
    print("=" * 130)
    for worker_count, algo_results in results_by_worker_count.items():
        print(f"\n[Worker Count = {worker_count}]")
        print("-" * 130)
        print(
            f"{'Algorithm':<22} | {'#Assigned Tasks':<16} | {'#Total Tasks':<13} | {'Task Completion Rate':<22} | {'Collaboration Unfairness':<26} | "
            f"{'CPU Time (s)':<14} | {'Prediction MAE':<14} | {'Prediction RMSE':<14}"
        )
        print("-" * 130)
        for algorithm, metrics in algo_results.items():
            print(
                f"{algorithm:<22} | "
                f"{metrics['assigned_tasks']:<16} | "
                f"{str(metrics.get('total_tasks', '-')):<13} | "
                f"{metrics['task_completion_rate']:<22.4f} | "
                f"{metrics['u_rho']:<26.4f} | "
                f"{metrics['cpu_time']:<14.4f} | "
                f"{_fmt_optional(metrics.get('pred_mae')):<14} | "
                f"{_fmt_optional(metrics.get('pred_rmse')):<14}"
            )
    print("=" * 130)


def main():
    results_by_worker_count = {}
    for worker_count in WORKER_COUNTS:
        print("\n" + "#" * 90)
        print(f"开始工人数实验: {worker_count}")
        print("#" * 90)
        results_by_worker_count[worker_count] = run_single_setting(worker_count)

    output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_worker_sweep_results.csv")
    write_csv(results_by_worker_count, output_csv)
    print_summary(results_by_worker_count)
    print(f"\nCSV results saved to: {output_csv}")


if __name__ == "__main__":
    main()
