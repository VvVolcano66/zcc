import argparse

from sweep_split_runner import FIXED_WORKER_COUNT, run_batch_setting


def main_for_batch_count(batch_count: int, worker_count: int = FIXED_WORKER_COUNT, output_dir: str = None):
    output_path = run_batch_setting(
        batch_count=batch_count,
        worker_count=worker_count,
        output_dir=output_dir,
    )
    print(f"\nCSV results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run one batch-count experiment setting.")
    parser.add_argument("batch_count", type=int, help="Batch count to evaluate.")
    parser.add_argument(
        "--worker-count",
        type=int,
        default=FIXED_WORKER_COUNT,
        help="Worker count to use for the run.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store the per-setting CSV file.",
    )
    args = parser.parse_args()
    main_for_batch_count(
        batch_count=args.batch_count,
        worker_count=args.worker_count,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
