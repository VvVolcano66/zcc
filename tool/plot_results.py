from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import importlib.util


DEFAULT_RESULT_DIR = Path(r"D:\biyelunwen\result")
DEFAULT_SWEEP_CSV = Path(r"D:\biyelunwen\experiment\batch_worker_sweep_results.csv")
DEFAULT_BATCH_SWEEP_CSV = Path(r"D:\biyelunwen\experiment\batch_count_sweep_results.csv")
DEFAULT_CENTER_SWEEP_CSV = Path(r"D:\biyelunwen\experiment\batch_center_count_sweep_results.csv")
DEFAULT_WORKER_SPLIT_DIR = Path(r"D:\biyelunwen\result\worker_count_split")
DEFAULT_BATCH_SPLIT_DIR = Path(r"D:\biyelunwen\result\batch_count_split")
DEFAULT_CENTER_SPLIT_DIR = Path(r"D:\biyelunwen\result\center_count_split")
DEFAULT_SMALL_SCALE_WORKER_SWEEP_CSV = Path(r"D:\biyelunwen\experiment\batch_small_scale_worker_sweep_results.csv")
DEFAULT_SMALL_SCALE_BATCH_SWEEP_CSV = Path(r"D:\biyelunwen\experiment\batch_small_scale_batch_sweep_results.csv")
DEFAULT_SMALL_SCALE_WORKER_SPLIT_DIR = Path(r"D:\biyelunwen\result\small_scale_3x3\worker_count_sweep")
DEFAULT_SMALL_SCALE_BATCH_SPLIT_DIR = Path(r"D:\biyelunwen\result\small_scale_3x3\batch_count_sweep")
DEFAULT_WORKER_COUNTS = [200, 300, 400, 500, 600, 700, 800, 900]
DEFAULT_BATCH_COUNTS = [2, 4, 6, 8]
DEFAULT_CENTER_COUNTS = [3, 4, 5, 6, 7]
CURRENT_ALGORITHM_ORDER = [
    "Greedy",
    "IMTAO (Seq-BDC)",
    "Predictive-MCTGNet",
    "NoPred-RL-Game",
    "Platform-RL-MCTGNet",
]
ALGORITHM_STYLE_MAP = {
    "Greedy": {"marker": "o", "color": "#4E79A7"},
    "IMTAO (Seq-BDC)": {"marker": "s", "color": "#F28E2B"},
    "Predictive-MCTGNet": {"marker": "^", "color": "#59A14F"},
    "NoPred-RL-Game": {"marker": "D", "color": "#E15759"},
    "Platform-RL-MCTGNet": {"marker": "P", "color": "#76B7B2"},
}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_CONFIG_PATH = PROJECT_ROOT / "config.py"


def _load_project_utility_weights() -> tuple[float, float]:
    task_weight = 1.0
    unfairness_weight = 0.0
    if not PROJECT_CONFIG_PATH.exists():
        return task_weight, unfairness_weight

    spec = importlib.util.spec_from_file_location("biyelunwen_config_for_plots", PROJECT_CONFIG_PATH)
    if spec is None or spec.loader is None:
        return task_weight, unfairness_weight

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    task_weight = float(getattr(module, "RBG_PLATFORM_TASK_WEIGHT", task_weight))
    unfairness_weight = float(getattr(module, "PFRL_FAIRNESS_SECONDARY_WEIGHT", unfairness_weight))
    return task_weight, unfairness_weight


UTILITY_TASK_WEIGHT, UTILITY_UNFAIRNESS_WEIGHT = _load_project_utility_weights()
UTILITY_METRIC = "utility"
UTILITY_LABEL = (
    f"Utility ({UTILITY_TASK_WEIGHT:.2f}*Assigned Tasks - "
    f"{UTILITY_UNFAIRNESS_WEIGHT:.2f}*Unfairness)"
)
DEFAULT_PLOT_LABELS = {
    "assigned_tasks": "Assigned Tasks",
    "task_completion_rate": "Task Completion Rate",
    "u_rho": "Collaboration Unfairness",
    "cpu_time": "CPU Time (s)",
    "pred_mae": "Prediction MAE",
    "pred_rmse": "Prediction RMSE",
    UTILITY_METRIC: UTILITY_LABEL,
}


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
    ordered = [name for name in CURRENT_ALGORITHM_ORDER if name in observed]
    return ordered


def _filter_current_algorithms(df: pd.DataFrame, series_col: str = "algorithm") -> pd.DataFrame:
    if series_col not in df.columns:
        return df.copy()
    filtered = df[df[series_col].astype(str).isin(CURRENT_ALGORITHM_ORDER)].copy()
    return filtered


def _sort_by_algorithm_order(
    df: pd.DataFrame,
    primary_cols: Optional[Iterable[str]] = None,
    series_col: str = "algorithm",
) -> pd.DataFrame:
    sorted_df = df.copy()
    if series_col not in sorted_df.columns:
        return sorted_df
    order_map = {name: idx for idx, name in enumerate(CURRENT_ALGORITHM_ORDER)}
    sorted_df["_algorithm_order"] = sorted_df[series_col].map(
        lambda value: order_map.get(str(value), len(order_map))
    )
    sort_cols = list(primary_cols or [])
    sort_cols.extend(["_algorithm_order", series_col])
    sorted_df = sorted_df.sort_values(sort_cols).drop(columns=["_algorithm_order"])
    return sorted_df


def _attach_utility_metric(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    if "assigned_tasks" not in enriched.columns or "u_rho" not in enriched.columns:
        return enriched

    enriched["assigned_tasks"] = pd.to_numeric(enriched["assigned_tasks"], errors="coerce")
    enriched["u_rho"] = pd.to_numeric(enriched["u_rho"], errors="coerce")
    enriched[UTILITY_METRIC] = (
        UTILITY_TASK_WEIGHT * enriched["assigned_tasks"]
        - UTILITY_UNFAIRNESS_WEIGHT * enriched["u_rho"]
    )
    return enriched


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
        style = ALGORITHM_STYLE_MAP.get(str(algorithm), {})
        plt.plot(
            algo_df[x_col],
            algo_df[metric],
            marker=style.get("marker", "o"),
            linewidth=2,
            markersize=5,
            color=style.get("color"),
            label=algorithm,
        )

    plt.xlabel(x_label)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=max(1, min(5, len(list(algorithms)))),
        frameon=False,
    )
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

    order_map = {name: idx for idx, name in enumerate(CURRENT_ALGORITHM_ORDER)}
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
    df = _filter_current_algorithms(_attach_utility_metric(pd.read_csv(csv_file)))
    df["worker_count"] = pd.to_numeric(df["worker_count"], errors="coerce")
    df = df.dropna(subset=["worker_count"])
    df["worker_count"] = df["worker_count"].astype(int)
    allowed_counts = set(df["worker_count"].drop_duplicates().astype(int).tolist()) if worker_counts is None else {
        int(x) for x in worker_counts
    }
    df = df[df["worker_count"].isin(allowed_counts)].copy()

    plots = DEFAULT_PLOT_LABELS

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

    df = _filter_current_algorithms(_attach_utility_metric(pd.read_csv(csv_file)))
    df["batch_count"] = pd.to_numeric(df["batch_count"], errors="coerce")
    df = df.dropna(subset=["batch_count"])
    df["batch_count"] = df["batch_count"].astype(int)
    allowed_counts = set(df["batch_count"].drop_duplicates().astype(int).tolist()) if batch_counts is None else {
        int(x) for x in batch_counts
    }
    df = df[df["batch_count"].isin(allowed_counts)].copy()

    plots = DEFAULT_PLOT_LABELS

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

    df = _filter_current_algorithms(_attach_utility_metric(pd.read_csv(csv_file)))
    df["center_count"] = pd.to_numeric(df["center_count"], errors="coerce")
    df = df.dropna(subset=["center_count"])
    df["center_count"] = df["center_count"].astype(int)
    allowed_counts = set(df["center_count"].drop_duplicates().astype(int).tolist()) if center_counts is None else {
        int(x) for x in center_counts
    }
    df = df[df["center_count"].isin(allowed_counts)].copy()

    plots = DEFAULT_PLOT_LABELS

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
        df = _filter_current_algorithms(_attach_utility_metric(pd.read_csv(csv_file)))
        if "center_count" not in df.columns or df["center_count"].isna().all():
            inferred_value = csv_file.stem.rsplit("_", 1)[-1]
            df["center_count"] = inferred_value
        frames.append(df)

    merged_df = _filter_current_algorithms(pd.concat(frames, ignore_index=True))
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
    _sort_by_algorithm_order(merged_df, primary_cols=["center_count"]).to_csv(
        merged_csv_path, index=False, encoding="utf-8-sig"
    )

    plots = DEFAULT_PLOT_LABELS

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


def plot_worker_count_split_results(
    split_dir: Optional[str] = None,
    result_dir: Optional[str] = None,
    worker_counts: Optional[Iterable[int]] = None,
) -> Path:
    _configure_matplotlib()

    input_dir = Path(split_dir) if split_dir is not None else DEFAULT_WORKER_SPLIT_DIR
    if not input_dir.exists():
        raise FileNotFoundError(f"Split result directory not found: {input_dir}")

    csv_files = sorted(input_dir.glob("worker_count_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No worker_count_*.csv files found under: {input_dir}")

    frames = []
    for csv_file in csv_files:
        df = _filter_current_algorithms(_attach_utility_metric(pd.read_csv(csv_file)))
        if "worker_count" not in df.columns or df["worker_count"].isna().all():
            inferred_value = csv_file.stem.rsplit("_", 1)[-1]
            df["worker_count"] = inferred_value
        frames.append(df)

    merged_df = _filter_current_algorithms(pd.concat(frames, ignore_index=True))
    merged_df["worker_count"] = pd.to_numeric(merged_df["worker_count"], errors="coerce")
    merged_df = merged_df.dropna(subset=["worker_count"])
    merged_df["worker_count"] = merged_df["worker_count"].astype(int)

    allowed_counts = (
        set(merged_df["worker_count"].drop_duplicates().astype(int).tolist())
        if worker_counts is None
        else {int(x) for x in worker_counts}
    )
    merged_df = merged_df[merged_df["worker_count"].isin(allowed_counts)].copy()

    output_dir = ensure_result_dir(Path(result_dir) if result_dir is not None else None) / "worker_count_split_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_csv_path = output_dir / "merged_worker_count_results.csv"
    _sort_by_algorithm_order(merged_df, primary_cols=["worker_count"]).to_csv(
        merged_csv_path, index=False, encoding="utf-8-sig"
    )

    plots = DEFAULT_PLOT_LABELS

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
            title_suffix="Worker Count",
        )

    return output_dir


def plot_worker_count_grouped_results(
    split_dir: Optional[str] = None,
    result_dir: Optional[str] = None,
    worker_groups: Optional[dict] = None,
) -> Path:
    _configure_matplotlib()

    input_dir = Path(split_dir) if split_dir is not None else DEFAULT_WORKER_SPLIT_DIR
    if not input_dir.exists():
        raise FileNotFoundError(f"Split result directory not found: {input_dir}")

    csv_files = sorted(input_dir.glob("worker_count_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No worker_count_*.csv files found under: {input_dir}")

    frames = []
    for csv_file in csv_files:
        df = _filter_current_algorithms(_attach_utility_metric(pd.read_csv(csv_file)))
        if "worker_count" not in df.columns or df["worker_count"].isna().all():
            inferred_value = csv_file.stem.rsplit("_", 1)[-1]
            df["worker_count"] = inferred_value
        frames.append(df)

    merged_df = _filter_current_algorithms(pd.concat(frames, ignore_index=True))
    merged_df["worker_count"] = pd.to_numeric(merged_df["worker_count"], errors="coerce")
    merged_df = merged_df.dropna(subset=["worker_count"])
    merged_df["worker_count"] = merged_df["worker_count"].astype(int)

    if worker_groups is None:
        worker_groups = {
            "workers_500_900": [500, 600, 700, 800, 900],
            "workers_500_2500": [500, 1000, 1500, 2000, 2500],
        }

    output_root = ensure_result_dir(Path(result_dir) if result_dir is not None else None) / "worker_count_grouped_plots"
    output_root.mkdir(parents=True, exist_ok=True)

    plots = DEFAULT_PLOT_LABELS

    for group_name, worker_counts in worker_groups.items():
        allowed_counts = {int(x) for x in worker_counts}
        group_df = merged_df[merged_df["worker_count"].isin(allowed_counts)].copy()
        if group_df.empty:
            continue

        group_dir = output_root / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        merged_csv_path = group_dir / f"{group_name}.csv"
        _sort_by_algorithm_order(group_df, primary_cols=["worker_count"]).to_csv(
            merged_csv_path, index=False, encoding="utf-8-sig"
        )

        for metric, ylabel in plots.items():
            output_path = group_dir / f"{metric}.png"
            _plot_metric(
                df=group_df,
                metric=metric,
                ylabel=ylabel,
                output_path=output_path,
                series_col="algorithm",
                x_col="worker_count",
                x_label="Worker Count",
                title_suffix=group_name,
            )

    return output_root


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

    df = _filter_current_algorithms(_attach_utility_metric(pd.read_csv(csv_file)))
    df["worker_count"] = pd.to_numeric(df["worker_count"], errors="coerce")
    df = df.dropna(subset=["worker_count"])
    df["worker_count"] = df["worker_count"].astype(int)
    allowed_counts = set(df["worker_count"].drop_duplicates().astype(int).tolist()) if worker_counts is None else {
        int(x) for x in worker_counts
    }
    df = df[df["worker_count"].isin(allowed_counts)].copy()

    plots = DEFAULT_PLOT_LABELS

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
        df = _filter_current_algorithms(_attach_utility_metric(pd.read_csv(csv_file)))
        if "worker_count" not in df.columns or df["worker_count"].isna().all():
            inferred_value = csv_file.stem.rsplit("_", 1)[-1]
            df["worker_count"] = inferred_value
        frames.append(df)

    merged_df = _filter_current_algorithms(pd.concat(frames, ignore_index=True))
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
    _sort_by_algorithm_order(merged_df, primary_cols=["worker_count"]).to_csv(
        merged_csv_path, index=False, encoding="utf-8-sig"
    )

    plots = DEFAULT_PLOT_LABELS

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

    df = _filter_current_algorithms(_attach_utility_metric(pd.read_csv(csv_file)))
    df["batch_count"] = pd.to_numeric(df["batch_count"], errors="coerce")
    df = df.dropna(subset=["batch_count"])
    df["batch_count"] = df["batch_count"].astype(int)
    allowed_counts = set(df["batch_count"].drop_duplicates().astype(int).tolist()) if batch_counts is None else {
        int(x) for x in batch_counts
    }
    df = df[df["batch_count"].isin(allowed_counts)].copy()

    plots = DEFAULT_PLOT_LABELS

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
        df = _filter_current_algorithms(_attach_utility_metric(pd.read_csv(csv_file)))
        if "batch_count" not in df.columns or df["batch_count"].isna().all():
            inferred_value = csv_file.stem.rsplit("_", 1)[-1]
            df["batch_count"] = inferred_value
        frames.append(df)

    merged_df = _filter_current_algorithms(pd.concat(frames, ignore_index=True))
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
    _sort_by_algorithm_order(merged_df, primary_cols=["batch_count"]).to_csv(
        merged_csv_path, index=False, encoding="utf-8-sig"
    )

    plots = DEFAULT_PLOT_LABELS

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

    df = _filter_current_algorithms(_attach_utility_metric(pd.read_csv(csv_file)))
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

    plots = {
        column: DEFAULT_PLOT_LABELS.get(column, column.replace("_", " ").title())
        for column in numeric_candidates
    }

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


def plot_grouped_experiment_folder(
    folder_path: str,
    x_col: str,
    x_label: str,
    title_suffix: str,
    csv_name: Optional[str] = None,
) -> Path:
    _configure_matplotlib()

    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Experiment folder not found: {folder}")

    if csv_name is not None:
        csv_file = folder / csv_name
    else:
        csv_candidates = sorted(folder.glob("*.csv"))
        if not csv_candidates:
            raise FileNotFoundError(f"No CSV files found under: {folder}")
        csv_file = csv_candidates[0]

    df = _filter_current_algorithms(_attach_utility_metric(pd.read_csv(csv_file)))
    if x_col not in df.columns:
        raise ValueError(f"CSV does not contain x-axis column '{x_col}': {csv_file}")

    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df = df.dropna(subset=[x_col])
    if df.empty:
        raise ValueError(f"No valid rows found in: {csv_file}")

    if all(float(value).is_integer() for value in df[x_col].tolist()):
        df[x_col] = df[x_col].astype(int)

    sorted_df = _sort_by_algorithm_order(df, primary_cols=[x_col])
    sorted_df.to_csv(csv_file, index=False, encoding="utf-8-sig")

    for metric, ylabel in DEFAULT_PLOT_LABELS.items():
        output_path = folder / f"{metric}.png"
        _plot_metric(
            df=sorted_df,
            metric=metric,
            ylabel=ylabel,
            output_path=output_path,
            series_col="algorithm",
            x_col=x_col,
            x_label=x_label,
            title_suffix=title_suffix,
        )

    return folder


def plot_batch_change_experiment_results(
    split_dir: Optional[str] = None,
    result_dir: Optional[str] = None,
    batch_counts: Optional[Iterable[int]] = None,
) -> Path:
    _configure_matplotlib()

    input_dir = Path(split_dir) if split_dir is not None else DEFAULT_BATCH_SPLIT_DIR
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

    merged_df = _filter_current_algorithms(pd.concat(frames, ignore_index=True))
    merged_df["batch_count"] = pd.to_numeric(merged_df["batch_count"], errors="coerce")
    merged_df = merged_df.dropna(subset=["batch_count"])
    merged_df["batch_count"] = merged_df["batch_count"].astype(int)
    merged_df = _attach_utility_metric(merged_df)

    allowed_counts = (
        set(merged_df["batch_count"].drop_duplicates().astype(int).tolist())
        if batch_counts is None
        else {int(x) for x in batch_counts}
    )
    merged_df = merged_df[merged_df["batch_count"].isin(allowed_counts)].copy()

    output_dir = ensure_result_dir(Path(result_dir) if result_dir is not None else None) / "batch_change_experiment_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_csv_path = output_dir / "merged_batch_change_experiment_results.csv"
    _sort_by_algorithm_order(merged_df, primary_cols=["batch_count"]).to_csv(
        merged_csv_path, index=False, encoding="utf-8-sig"
    )

    for metric, ylabel in DEFAULT_PLOT_LABELS.items():
        output_path = output_dir / f"{metric}.png"
        _plot_metric(
            df=merged_df,
            metric=metric,
            ylabel=ylabel,
            output_path=output_path,
            series_col="algorithm",
            x_col="batch_count",
            x_label="Batch Count",
            title_suffix="Batch Change Experiment",
        )

    return output_dir


if __name__ == "__main__":
    saved_dirs = []

    if DEFAULT_SWEEP_CSV.exists():
        saved_dirs.append(plot_worker_sweep_results())

    if DEFAULT_CENTER_SWEEP_CSV.exists():
        saved_dirs.append(plot_center_count_sweep_results())

    if DEFAULT_CENTER_SPLIT_DIR.exists() and any(DEFAULT_CENTER_SPLIT_DIR.glob("center_count_*.csv")):
        saved_dirs.append(plot_center_count_split_results())

    if DEFAULT_WORKER_SPLIT_DIR.exists() and any(DEFAULT_WORKER_SPLIT_DIR.glob("worker_count_*.csv")):
        saved_dirs.append(plot_worker_count_split_results())

    if DEFAULT_SMALL_SCALE_WORKER_SWEEP_CSV.exists():
        saved_dirs.append(plot_small_scale_worker_sweep_results())

    if DEFAULT_SMALL_SCALE_WORKER_SPLIT_DIR.exists() and any(DEFAULT_SMALL_SCALE_WORKER_SPLIT_DIR.glob("worker_count_*.csv")):
        saved_dirs.append(plot_small_scale_worker_split_results())

    if DEFAULT_SMALL_SCALE_BATCH_SWEEP_CSV.exists():
        saved_dirs.append(plot_small_scale_batch_sweep_results())

    if DEFAULT_BATCH_SPLIT_DIR.exists() and any(DEFAULT_BATCH_SPLIT_DIR.glob("batch_count_*.csv")):
        saved_dirs.append(plot_batch_change_experiment_results())

    if DEFAULT_SMALL_SCALE_BATCH_SPLIT_DIR.exists() and any(DEFAULT_SMALL_SCALE_BATCH_SPLIT_DIR.glob("batch_count_*.csv")):
        saved_dirs.append(plot_small_scale_batch_split_results())

    if not saved_dirs:
        raise FileNotFoundError(
            "No known result CSV files were found under D:\\biyelunwen\\experiment."
        )

    for saved_dir in saved_dirs:
        print(f"Plots saved to: {saved_dir}")
