import csv
import os

os.environ["MCTGNET_DISPATCH_FORCE_CPU"] = "1"

import batch as batch_exp


FIXED_WORKER_COUNT = 500
BATCH_COUNTS = [2, 4, 6, 8, 10]
ALGORITHMS = [
    ("greedy", "Greedy"),
    ("imtao", "IMTAO (Seq-BDC)"),
    ("game_only_dispatch", "NoPred-Game"),
    ("predictive_mctgnet", "Predictive-MCTGNet"),
    ("predictive_game_mctgnet", "Game-MCTGNet"),
    ("predictive_uabg_mctgnet", "UABG-MCTGNet"),
    ("predictive_rl_game_mctgnet", "RBG-MCTGNet"),
]


def _fmt_optional(value):
    return f"{value:.4f}" if value is not None else "-"


def run_single_setting(batch_count: int):
    batch_exp._MCTG_PREDICTOR_CACHE.clear()
    batch_exp.DEFAULT_WORKER_LIMIT = FIXED_WORKER_COUNT
    batch_exp.DEFAULT_COMPARE_SLOT_COUNT = batch_count
    # Important: the simulation context cache key does not include worker_limit.
    batch_exp._SIMULATION_CONTEXT_CACHE.clear()

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


def write_csv(results_by_batch_count, output_path: str):
    fieldnames = [
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
        for batch_count, algo_results in results_by_batch_count.items():
            for algorithm, metrics in algo_results.items():
                writer.writerow(
                    {
                        "worker_count": FIXED_WORKER_COUNT,
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


def print_summary(results_by_batch_count):
    print("\n" + "=" * 130)
    print(f"固定工人数 {FIXED_WORKER_COUNT} 下的 batch 数量敏感性实验结果")
    print("=" * 130)
    for batch_count, algo_results in results_by_batch_count.items():
        print(f"\n[Batch Count = {batch_count}]")
        print("-" * 130)
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


def main():
    results_by_batch_count = {}
    for batch_count in BATCH_COUNTS:
        print("\n" + "#" * 90)
        print(f"开始 batch 数实验: {batch_count} | 固定工人数: {FIXED_WORKER_COUNT}")
        print("#" * 90)
        results_by_batch_count[batch_count] = run_single_setting(batch_count)

    output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_count_sweep_results.csv")
    write_csv(results_by_batch_count, output_csv)
    print_summary(results_by_batch_count)
    print(f"\nCSV results saved to: {output_csv}")


if __name__ == "__main__":
    main()
