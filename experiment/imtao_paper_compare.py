try:
    from . import batch as batch_exp
except ImportError:
    import batch as batch_exp


IMTAO_PAPER_ALGOS = [
    ("imtao_seq_bdc", "Seq-BDC"),
    ("imtao_seq_rbdc", "Seq-RBDC"),
    ("imtao_seq_dc", "Seq-DC"),
    ("imtao_seq_wo_c", "Seq-w/o-C"),
]


def _fmt_optional(value):
    return f"{value:.4f}" if value is not None else "-"


def main():
    results = {}
    for algo_name, display_name in IMTAO_PAPER_ALGOS:
        _, _, metrics = batch_exp.run_online_simulation_with_center_pickup(
            algo_name=algo_name,
            test_date=batch_exp.DEFAULT_TEST_DATE,
            test_start_hour=batch_exp.DEFAULT_START_HOUR,
            test_end_hour=batch_exp.DEFAULT_END_HOUR,
            time_slot_minutes=batch_exp.DEFAULT_TIME_SLOT_MINUTES,
        )
        results[display_name] = metrics

    print("\n" + "=" * 110)
    print("IMTAO paper-style comparison on the project partition")
    print("=" * 110)
    print(
        f"{'Algorithm':<16} | {'#Assigned Tasks':<16} | {'Collaboration Unfairness':<26} | "
        f"{'CPU Time (s)':<14} | {'Prediction MAE':<14} | {'Prediction RMSE':<14}"
    )
    print("-" * 110)
    for algorithm, metrics in results.items():
        print(
            f"{algorithm:<16} | "
            f"{metrics['assigned_tasks']:<16} | "
            f"{metrics['u_rho']:<26.4f} | "
            f"{metrics['cpu_time']:<14.4f} | "
            f"{_fmt_optional(metrics.get('pred_mae')):<14} | "
            f"{_fmt_optional(metrics.get('pred_rmse')):<14}"
        )
    print("=" * 110)


if __name__ == "__main__":
    main()
