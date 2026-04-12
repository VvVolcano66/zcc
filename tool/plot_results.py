from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_RESULT_DIR = Path(r"D:\biyelunwen\result")
DEFAULT_SWEEP_CSV = Path(r"D:\biyelunwen\experiment\batch_worker_sweep_results.csv")
DEFAULT_BATCH_SWEEP_CSV = Path(r"D:\biyelunwen\experiment\batch_count_sweep_results.csv")
DEFAULT_WORKER_COUNTS = [200, 300, 400, 500, 600, 700, 800, 900]
DEFAULT_BATCH_COUNTS = [2, 4, 6, 8]


def ensure_result_dir(result_dir: Optional[Path] = None) -> Path:
    output_dir = Path(result_dir) if result_dir is not None else DEFAULT_RESULT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _plot_metric(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
    algorithms: Optional[Iterable[str]] = None,
    x_col: str = "worker_count",
    x_label: str = "Worker Count",
    title_suffix: str = "Worker Count",
) -> None:
    metric_df = df.copy()
    metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce")
    metric_df = metric_df.dropna(subset=[metric])
    if metric_df.empty:
        return

    if algorithms is None:
        algorithms = metric_df["algorithm"].drop_duplicates().tolist()

    plt.figure(figsize=(10, 6), dpi=180)
    for algorithm in algorithms:
        algo_df = metric_df[metric_df["algorithm"] == algorithm].sort_values(x_col)
        if algo_df.empty:
            continue
        plt.plot(
            algo_df[x_col],
            algo_df[metric],
            marker="o",
            linewidth=2,
            markersize=5,
            label=algorithm,
        )

    plt.xlabel(x_label)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs {title_suffix}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    x_values = sorted(metric_df[x_col].drop_duplicates().astype(int).tolist())
    plt.xticks(x_values)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_worker_sweep_results(
    csv_path: Optional[str] = None,
    result_dir: Optional[str] = None,
    worker_counts: Optional[Iterable[int]] = None,
) -> Path:
    _configure_matplotlib()

    csv_file = Path(csv_path) if csv_path is not None else DEFAULT_SWEEP_CSV
    if not csv_file.exists():
        raise FileNotFoundError(f"Result CSV not found: {csv_file}")

    output_dir = ensure_result_dir(Path(result_dir) if result_dir is not None else None)
    df = pd.read_csv(csv_file)
    df["worker_count"] = pd.to_numeric(df["worker_count"], errors="coerce")
    df = df.dropna(subset=["worker_count"])
    df["worker_count"] = df["worker_count"].astype(int)
    allowed_counts = set(df["worker_count"].drop_duplicates().astype(int).tolist()) if worker_counts is None else {
        int(x) for x in worker_counts
    }
    df = df[df["worker_count"].isin(allowed_counts)].copy()

    plots = {
        "assigned_tasks": "Assigned Tasks",
        "u_rho": "Collaboration Unfairness",
        "cpu_time": "CPU Time (s)",
        "pred_mae": "Prediction MAE",
        "pred_rmse": "Prediction RMSE",
    }

    for metric, ylabel in plots.items():
        output_path = output_dir / f"{metric}.png"
        _plot_metric(
            df=df,
            metric=metric,
            ylabel=ylabel,
            output_path=output_path,
            x_col="worker_count",
            x_label="Worker Count",
            title_suffix="Worker Count",
        )

    return output_dir


def plot_batch_count_sweep_results(
    csv_path: Optional[str] = None,
    result_dir: Optional[str] = None,
    batch_counts: Optional[Iterable[int]] = None,
) -> Path:
    _configure_matplotlib()

    csv_file = Path(csv_path) if csv_path is not None else DEFAULT_BATCH_SWEEP_CSV
    if not csv_file.exists():
        raise FileNotFoundError(f"Result CSV not found: {csv_file}")

    output_dir = ensure_result_dir(Path(result_dir) if result_dir is not None else None) / "batch_count_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_file)
    df["batch_count"] = pd.to_numeric(df["batch_count"], errors="coerce")
    df = df.dropna(subset=["batch_count"])
    df["batch_count"] = df["batch_count"].astype(int)
    allowed_counts = set(df["batch_count"].drop_duplicates().astype(int).tolist()) if batch_counts is None else {
        int(x) for x in batch_counts
    }
    df = df[df["batch_count"].isin(allowed_counts)].copy()

    plots = {
        "assigned_tasks": "Assigned Tasks",
        "u_rho": "Collaboration Unfairness",
        "cpu_time": "CPU Time (s)",
        "pred_mae": "Prediction MAE",
        "pred_rmse": "Prediction RMSE",
    }

    for metric, ylabel in plots.items():
        output_path = output_dir / f"{metric}.png"
        _plot_metric(
            df=df,
            metric=metric,
            ylabel=ylabel,
            output_path=output_path,
            x_col="batch_count",
            x_label="Batch Count",
            title_suffix="Batch Count",
        )

    return output_dir


if __name__ == "__main__":
    saved_dir = plot_batch_count_sweep_results()
    print(f"Plots saved to: {saved_dir}")
