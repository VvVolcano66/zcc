import argparse

from sweep_split_runner import FIXED_WORKER_COUNT, run_center_count_setting


def main_for_center_count(center_count: int, worker_count: int = FIXED_WORKER_COUNT, batch_count: int = None, output_dir: str = None):
    output_path = run_center_count_setting(
        center_count=center_count,
        worker_count=worker_count,
        batch_count=batch_count,
        output_dir=output_dir,
    )
    print(f"\nCSV results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run one center-count experiment setting.")
    parser.add_argument("center_count", type=int, help="Center count to evaluate, e.g. 3, 4, 5, 6, 7.")
    parser.add_argument(
        "--worker-count",
        type=int,
        default=FIXED_WORKER_COUNT,
        help="Worker count to use for the run.",
    )
    parser.add_argument(
        "--batch-count",
        type=int,
        default=None,
        help="Optional batch count override. Defaults to batch.py DEFAULT_COMPARE_SLOT_COUNT.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store the per-setting CSV file.",
    )
    args = parser.parse_args()
    main_for_center_count(
        center_count=args.center_count,
        worker_count=args.worker_count,
        batch_count=args.batch_count,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
