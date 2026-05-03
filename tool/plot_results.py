from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_RESULT_DIR = Path(r"D:\biyelunwen\result")
DEFAULT_SWEEP_CSV = Path(r"D:\biyelunwen\experiment\batch_worker_sweep_results.csv")
DEFAULT_BATCH_SWEEP_CSV = Path(r"D:\biyelunwen\experiment\batch_count_sweep_results.csv")
DEFAULT_CENTER_SWEEP_CSV = Path(r"D:\biyelunwen\experiment\batch_center_count_sweep_results.csv")
DEFAULT_CENTER_SPLIT_DIR = Path(r"D:\biyelunwen\result\center_count_split")
DEFAULT_SMALL_SCALE_WORKER_SWEEP_CSV = Path(r"D:\biyelunwen\experiment\batch_small_scale_worker_sweep_results.csv")
DEFAULT_SMALL_SCALE_BATCH_SWEEP_CSV = Path(r"D:\biyelunwen\experiment\batch_small_scale_batch_sweep_results.csv")
DEFAULT_SMALL_SCALE_WORKER_SPLIT_DIR = Path(r"D:\biyelunwen\result\small_scale_3x3\worker_count_sweep")
DEFAULT_SMALL_SCALE_BATCH_SPLIT_DIR = Path(r"D:\biyelunwen\result\small_scale_3x3\batch_count_sweep")
DEFAULT_WORKER_COUNTS = [200, 300, 400, 500, 600, 700, 800, 900]
DEFAULT_BATCH_COUNTS = [2, 4, 6, 8]
DEFAULT_CENTER_COUNTS = [3, 4, 5, 6, 7]
DEFAULT_ALGORITHM_ORDER = [
    "Greedy",
    "IMTAO",
    "IMTAO (Seq-BDC)",
    "NoPred-Game",
    "Game-Only",
    "Predictive-MCTGNet",
    "Game-MCTGNet",
    "UABG-MCTGNet",
    "RBG-MCTGNet",
    "Platform-RL-MCTGNet",
]


def ensure_result_dir(result_dir: Optional[Path] = None) -> Path:
    output_dir = Path(result_dir) if result_dir is not None else DEFAULT_RESULT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _ordered_algorithms(df: pd.DataFrame, series_col: str = "algorithm") -> list:
    if series_col not in df.columns:
        return []
    observed = [str(value) for value in df[series_col].dropna().drop_duplicates().tolist()]
    ordered = [name for name in DEFAULT_ALGORITHM_ORDER if name in observed]
    ordered.extend([name for name in observed if name not in ordered])
    return ordered


def _plot_metric(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
    algorithms: Optional[Iterable[str]] = None,
    series_col: str = "algorithm",
    x_col: str = "worker_count",
    x_label: str = "Worker Count",
    title_suffix: str = "Worker Count",
) -> None:
    if metric not in df.columns or series_col not in df.columns or x_col not in df.columns:
        return

    metric_df = df.copy()
    metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce")
    metric_df = metric_df.dropna(subset=[metric])
    if metric_df.empty:
        return

    if algorithms is None:
        algorithms = _ordered_algorithms(metric_df, series_col=series_col)

    plt.figure(figsize=(10, 6), dpi=180)
    for algorithm in algorithms:
        algo_df = metric_df[metric_df[series_col] == algorithm].sort_values(x_col)
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
    x_values = sorted(pd.to_numeric(metric_df[x_col], errors="coerce").dropna().tolist())
    if all(float(x).is_integer() for x in x_values):
        x_values = [int(x) for x in x_values]
    plt.xticks(x_values)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _plot_single_setting_metric(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
    title_suffix: str = "Single Setting",
) -> None:
    if metric not in df.columns or "algorithm" not in df.columns:
        return

    metric_df = df.copy()
    metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce")
    metric_df = metric_df.dropna(subset=[metric])
    if metric_df.empty:
        return

    order_map = {name: idx for idx, name in enumerate(DEFAULT_ALGORITHM_ORDER)}
    metric_df["_algorithm_order"] = metric_df["algorithm"].map(lambda value: order_map.get(str(value), len(order_map)))
    metric_df = metric_df.sort_values(["_algorithm_order", "algorithm"])

    plt.figure(figsize=(10, 6), dpi=180)
    plt.bar(metric_df["algorithm"], metric_df[metric])
    plt.xlabel("Algorithm")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} - {title_suffix}")
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.xticks(rotation=20, ha="right")
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
        "task_completion_rate": "Task Completion Rate",
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
            series_col="algorithm",
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
        "task_completion_rate": "Task Completion Rate",
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
            series_col="algorithm",
            x_col="batch_count",
            x_label="Batch Count",
            title_suffix="Batch Count",
        )

    return output_dir


def plot_center_count_sweep_results(
    csv_path: Optional[str] = None,
    result_dir: Optional[str] = None,
    center_counts: Optional[Iterable[int]] = None,
) -> Path:
    _configure_matplotlib()

    csv_file = Path(csv_path) if csv_path is not None else DEFAULT_CENTER_SWEEP_CSV
    if not csv_file.exists():
        raise FileNotFoundError(f"Result CSV not found: {csv_file}")

    output_dir = ensure_result_dir(Path(result_dir) if result_dir is not None else None) / "center_count_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_file)
    df["center_count"] = pd.to_numeric(df["center_count"], errors="coerce")
    df = df.dropna(subset=["center_count"])
    df["center_count"] = df["center_count"].astype(int)
    allowed_counts = set(df["center_count"].drop_duplicates().astype(int).tolist()) if center_counts is None else {
        int(x) for x in center_counts
    }
    df = df[df["center_count"].isin(allowed_counts)].copy()

    plots = {
        "assigned_tasks": "Assigned Tasks",
        "task_completion_rate": "Task Completion Rate",
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
            series_col="algorithm",
            x_col="center_count",
            x_label="Center Count",
            title_suffix="Center Count",
        )

    return output_dir


def plot_center_count_split_results(
    split_dir: Optional[str] = None,
    result_dir: Optional[str] = None,
    center_counts: Optional[Iterable[int]] = None,
) -> Path:
    _configure_matplotlib()

    input_dir = Path(split_dir) if split_dir is not None else DEFAULT_CENTER_SPLIT_DIR
    if not input_dir.exists():
        raise FileNotFoundError(f"Split result directory not found: {input_dir}")

    csv_files = sorted(input_dir.glob("center_count_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No center_count_*.csv files found under: {input_dir}")

    frames = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if "center_count" not in df.columns or df["center_count"].isna().all():
            inferred_value = csv_file.stem.rsplit("_", 1)[-1]
            df["center_count"] = inferred_value
        frames.append(df)

    merged_df = pd.concat(frames, ignore_index=True)
    merged_df["center_count"] = pd.to_numeric(merged_df["center_count"], errors="coerce")
    merged_df = merged_df.dropna(subset=["center_count"])
    merged_df["center_count"] = merged_df["center_count"].astype(int)

    allowed_counts = (
        set(merged_df["center_count"].drop_duplicates().astype(int).tolist())
        if center_counts is None
        else {int(x) for x in center_counts}
    )
    merged_df = merged_df[merged_df["center_count"].isin(allowed_counts)].copy()

    output_dir = ensure_result_dir(Path(result_dir) if result_dir is not None else None) / "center_count_split"
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_csv_path = output_dir / "merged_center_count_results.csv"
    merged_df.sort_values(["center_count", "algorithm"]).to_csv(merged_csv_path, index=False, encoding="utf-8-sig")

    plots = {
        "assigned_tasks": "Assigned Tasks",
        "task_completion_rate": "Task Completion Rate",
        "u_rho": "Collaboration Unfairness",
        "cpu_time": "CPU Time (s)",
        "pred_mae": "Prediction MAE",
        "pred_rmse": "Prediction RMSE",
    }

    for metric, ylabel in plots.items():
        output_path = output_dir / f"{metric}.png"
        _plot_metric(
            df=merged_df,
            metric=metric,
            ylabel=ylabel,
            output_path=output_path,
            series_col="algorithm",
            x_col="center_count",
            x_label="Center Count",
            title_suffix="Center Count",
        )

    return output_dir


def plot_small_scale_worker_sweep_results(
    csv_path: Optional[str] = None,
    result_dir: Optional[str] = None,
    worker_counts: Optional[Iterable[int]] = None,
) -> Path:
    _configure_matplotlib()

    csv_file = Path(csv_path) if csv_path is not None else DEFAULT_SMALL_SCALE_WORKER_SWEEP_CSV
    if not csv_file.exists():
        raise FileNotFoundError(f"Result CSV not found: {csv_file}")

    output_dir = ensure_result_dir(Path(result_dir) if result_dir is not None else None) / "small_scale_3x3" / "worker_count_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

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
        "task_completion_rate": "Task Completion Rate",
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
            series_col="algorithm",
            x_col="worker_count",
            x_label="Worker Count",
            title_suffix="Small-Scale Worker Count",
        )

    return output_dir


def plot_small_scale_worker_split_results(
    split_dir: Optional[str] = None,
    result_dir: Optional[str] = None,
    worker_counts: Optional[Iterable[int]] = None,
) -> Path:
    _configure_matplotlib()

    input_dir = Path(split_dir) if split_dir is not None else DEFAULT_SMALL_SCALE_WORKER_SPLIT_DIR
    if not input_dir.exists():
        raise FileNotFoundError(f"Split result directory not found: {input_dir}")

    csv_files = sorted(input_dir.glob("worker_count_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No worker_count_*.csv files found under: {input_dir}")

    frames = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if "worker_count" not in df.columns or df["worker_count"].isna().all():
            inferred_value = csv_file.stem.rsplit("_", 1)[-1]
            df["worker_count"] = inferred_value
        frames.append(df)

    merged_df = pd.concat(frames, ignore_index=True)
    merged_df["worker_count"] = pd.to_numeric(merged_df["worker_count"], errors="coerce")
    merged_df = merged_df.dropna(subset=["worker_count"])
    merged_df["worker_count"] = merged_df["worker_count"].astype(int)

    allowed_counts = (
        set(merged_df["worker_count"].drop_duplicates().astype(int).tolist())
        if worker_counts is None
        else {int(x) for x in worker_counts}
    )
    merged_df = merged_df[merged_df["worker_count"].isin(allowed_counts)].copy()

    output_dir = ensure_result_dir(Path(result_dir) if result_dir is not None else None) / "small_scale_3x3" / "worker_count_split_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_csv_path = output_dir / "merged_small_scale_worker_results.csv"
    merged_df.sort_values(["worker_count", "algorithm"]).to_csv(merged_csv_path, index=False, encoding="utf-8-sig")

    plots = {
        "assigned_tasks": "Assigned Tasks",
        "task_completion_rate": "Task Completion Rate",
        "u_rho": "Collaboration Unfairness",
        "cpu_time": "CPU Time (s)",
        "pred_mae": "Prediction MAE",
        "pred_rmse": "Prediction RMSE",
    }

    for metric, ylabel in plots.items():
        output_path = output_dir / f"{metric}.png"
        _plot_metric(
            df=merged_df,
            metric=metric,
            ylabel=ylabel,
            output_path=output_path,
            series_col="algorithm",
            x_col="worker_count",
            x_label="Worker Count",
            title_suffix="Small-Scale Worker Count",
        )

    return output_dir


def plot_small_scale_batch_sweep_results(
    csv_path: Optional[str] = None,
    result_dir: Optional[str] = None,
    batch_counts: Optional[Iterable[int]] = None,
) -> Path:
    _configure_matplotlib()

    csv_file = Path(csv_path) if csv_path is not None else DEFAULT_SMALL_SCALE_BATCH_SWEEP_CSV
    if not csv_file.exists():
        raise FileNotFoundError(f"Result CSV not found: {csv_file}")

    output_dir = ensure_result_dir(Path(result_dir) if result_dir is not None else None) / "small_scale_3x3" / "batch_count_sweep"
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
        "task_completion_rate": "Task Completion Rate",
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
            series_col="algorithm",
            x_col="batch_count",
            x_label="Batch Count",
            title_suffix="Small-Scale Batch Count",
        )

    return output_dir


def plot_small_scale_batch_split_results(
    split_dir: Optional[str] = None,
    result_dir: Optional[str] = None,
    batch_counts: Optional[Iterable[int]] = None,
) -> Path:
    _configure_matplotlib()

    input_dir = Path(split_dir) if split_dir is not None else DEFAULT_SMALL_SCALE_BATCH_SPLIT_DIR
    if not input_dir.exists():
        raise FileNotFoundError(f"Split result directory not found: {input_dir}")

    csv_files = sorted(input_dir.glob("batch_count_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No batch_count_*.csv files found under: {input_dir}")

    frames = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if "batch_count" not in df.columns or df["batch_count"].isna().all():
            inferred_value = csv_file.stem.rsplit("_", 1)[-1]
            df["batch_count"] = inferred_value
        frames.append(df)

    merged_df = pd.concat(frames, ignore_index=True)
    merged_df["batch_count"] = pd.to_numeric(merged_df["batch_count"], errors="coerce")
    merged_df = merged_df.dropna(subset=["batch_count"])
    merged_df["batch_count"] = merged_df["batch_count"].astype(int)

    allowed_counts = (
        set(merged_df["batch_count"].drop_duplicates().astype(int).tolist())
        if batch_counts is None
        else {int(x) for x in batch_counts}
    )
    merged_df = merged_df[merged_df["batch_count"].isin(allowed_counts)].copy()

    output_dir = ensure_result_dir(Path(result_dir) if result_dir is not None else None) / "small_scale_3x3" / "batch_count_split_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_csv_path = output_dir / "merged_small_scale_batch_results.csv"
    merged_df.sort_values(["batch_count", "algorithm"]).to_csv(merged_csv_path, index=False, encoding="utf-8-sig")

    plots = {
        "assigned_tasks": "Assigned Tasks",
        "task_completion_rate": "Task Completion Rate",
        "u_rho": "Collaboration Unfairness",
        "cpu_time": "CPU Time (s)",
        "pred_mae": "Prediction MAE",
        "pred_rmse": "Prediction RMSE",
    }

    for metric, ylabel in plots.items():
        output_path = output_dir / f"{metric}.png"
        _plot_metric(
            df=merged_df,
            metric=metric,
            ylabel=ylabel,
            output_path=output_path,
            series_col="algorithm",
            x_col="batch_count",
            x_label="Batch Count",
            title_suffix="Small-Scale Batch Count",
        )

    return output_dir


def plot_single_setting_results(
    csv_path: str,
    result_dir: Optional[str] = None,
) -> Path:
    _configure_matplotlib()

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Result CSV not found: {csv_file}")

    output_dir = ensure_result_dir(Path(result_dir) if result_dir is not None else csv_file.parent) / csv_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_file)
    if "algorithm" not in df.columns:
        raise ValueError(f"CSV does not contain an 'algorithm' column: {csv_file}")

    excluded_columns = {"algorithm"}
    numeric_candidates = []
    constant_metadata = {}
    for column in df.columns:
        if column in excluded_columns:
            continue
        numeric_series = pd.to_numeric(df[column], errors="coerce")
        non_null = numeric_series.dropna()
        if non_null.empty:
            continue
        if non_null.nunique() <= 1:
            constant_metadata[column] = non_null.iloc[0]
            continue
        numeric_candidates.append(column)

    default_labels = {
        "assigned_tasks": "Assigned Tasks",
        "task_completion_rate": "Task Completion Rate",
        "u_rho": "Collaboration Unfairness",
        "cpu_time": "CPU Time (s)",
        "pred_mae": "Prediction MAE",
        "pred_rmse": "Prediction RMSE",
    }

    plots = {column: default_labels.get(column, column.replace("_", " ").title()) for column in numeric_candidates}

    for metric, ylabel in plots.items():
        output_path = output_dir / f"{metric}.png"
        _plot_single_setting_metric(
            df=df,
            metric=metric,
            ylabel=ylabel,
            output_path=output_path,
            title_suffix=csv_file.stem,
        )

    if constant_metadata:
        metadata_path = output_dir / "metadata.txt"
        with open(metadata_path, "w", encoding="utf-8") as f:
            for key, value in constant_metadata.items():
                f.write(f"{key}: {value}\n")

    return output_dir


if __name__ == "__main__":
    saved_dirs = []

    if DEFAULT_SWEEP_CSV.exists():
        saved_dirs.append(plot_worker_sweep_results())

    if DEFAULT_BATCH_SWEEP_CSV.exists():
        saved_dirs.append(plot_batch_count_sweep_results())

    if DEFAULT_CENTER_SWEEP_CSV.exists():
        saved_dirs.append(plot_center_count_sweep_results())

    if DEFAULT_CENTER_SPLIT_DIR.exists() and any(DEFAULT_CENTER_SPLIT_DIR.glob("center_count_*.csv")):
        saved_dirs.append(plot_center_count_split_results())

    if DEFAULT_SMALL_SCALE_WORKER_SWEEP_CSV.exists():
        saved_dirs.append(plot_small_scale_worker_sweep_results())

    if DEFAULT_SMALL_SCALE_WORKER_SPLIT_DIR.exists() and any(DEFAULT_SMALL_SCALE_WORKER_SPLIT_DIR.glob("worker_count_*.csv")):
        saved_dirs.append(plot_small_scale_worker_split_results())

    if DEFAULT_SMALL_SCALE_BATCH_SWEEP_CSV.exists():
        saved_dirs.append(plot_small_scale_batch_sweep_results())

    if DEFAULT_SMALL_SCALE_BATCH_SPLIT_DIR.exists() and any(DEFAULT_SMALL_SCALE_BATCH_SPLIT_DIR.glob("batch_count_*.csv")):
        saved_dirs.append(plot_small_scale_batch_split_results())

    if not saved_dirs:
        raise FileNotFoundError(
            "No known result CSV files were found under D:\\biyelunwen\\experiment."
        )

    for saved_dir in saved_dirs:
        print(f"Plots saved to: {saved_dir}")
