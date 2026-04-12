import argparse

from sweep_split_runner import run_worker_setting


def main_for_worker_count(worker_count: int, batch_count: int = None, output_dir: str = None):
    output_path = run_worker_setting(
        worker_count=worker_count,
        batch_count=batch_count,
        output_dir=output_dir,
    )
    print(f"\nCSV results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run one worker-count experiment setting.")
    parser.add_argument("worker_count", type=int, help="Worker count to evaluate.")
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
    main_for_worker_count(
        worker_count=args.worker_count,
        batch_count=args.batch_count,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
