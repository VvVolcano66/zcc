import argparse
import json
import os
import sys

import pandas as pd


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import config
from predicate.EventSTPPredictor import EventSTPPredictor


def load_actual_events(data_dir: str, target_date: str, start_hour: float, end_hour: float) -> pd.DataFrame:
    path = os.path.join(data_dir, f"tasks_{target_date}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Target task file does not exist: {path}")
    events = pd.read_csv(path)
    events["first_time"] = pd.to_datetime(events["first_time"])
    target_ts = pd.Timestamp(target_date).normalize()
    return events.loc[
        (events["first_time"] >= target_ts + pd.Timedelta(hours=start_hour))
        & (events["first_time"] < target_ts + pd.Timedelta(hours=end_hour))
    ].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest event-level time and location forecasts on a target task day."
    )
    parser.add_argument("--data-dir", default=config.TASK_DATA_DIR)
    parser.add_argument("--target-date", default=config.EXPERIMENT_TEST_DATE)
    parser.add_argument("--start-hour", type=float, default=float(config.EXPERIMENT_START_HOUR))
    parser.add_argument("--end-hour", type=float, default=float(config.EXPERIMENT_END_HOUR))
    parser.add_argument("--slot-minutes", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(project_root, "result", "event_prediction"),
    )
    args = parser.parse_args()

    predictor = EventSTPPredictor(slot_minutes=args.slot_minutes)
    predictor.fit_from_directory(
        args.data_dir,
        before_date=args.target_date,
        start_hour=args.start_hour,
        end_hour=args.end_hour,
    )
    predictions = predictor.predict(args.target_date, args.start_hour, args.end_hour)
    actual = load_actual_events(args.data_dir, args.target_date, args.start_hour, args.end_hour)
    metrics = predictor.evaluate(actual, predictions).as_dict()

    os.makedirs(args.output_dir, exist_ok=True)
    forecast_path = os.path.join(args.output_dir, f"event_forecast_{args.target_date}.csv")
    metric_path = os.path.join(args.output_dir, f"event_forecast_metrics_{args.target_date}.json")
    predictions.to_csv(forecast_path, index=False)
    with open(metric_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(f"Training history ends before: {args.target_date}")
    print(
        "Calibrated intensity weights "
        f"(learned, analogue, persistence): {predictor.intensity_weights.tolist()}"
    )
    print(f"Calibrated intensity scale: {predictor.intensity_scale:.4f}")
    print(f"Forecast saved to: {forecast_path}")
    print(f"Metrics saved to: {metric_path}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
