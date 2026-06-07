import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingRegressor


REQUIRED_COLUMNS = ("task_id", "first_time", "first_lon", "first_lat")


@dataclass
class EventForecastMetrics:
    actual_events: int
    predicted_events: int
    matched_events: int
    count_absolute_error: int
    slot_count_mae: float
    slot_count_rmse: float
    time_mae_minutes: float
    location_mae_meters: float
    joint_hit_rate: float
    precision_by_count: float
    recall_by_count: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "actual_events": self.actual_events,
            "predicted_events": self.predicted_events,
            "matched_events": self.matched_events,
            "count_absolute_error": self.count_absolute_error,
            "slot_count_mae": self.slot_count_mae,
            "slot_count_rmse": self.slot_count_rmse,
            "time_mae_minutes": self.time_mae_minutes,
            "location_mae_meters": self.location_mae_meters,
            "joint_hit_rate": self.joint_hit_rate,
            "precision_by_count": self.precision_by_count,
            "recall_by_count": self.recall_by_count,
        }


class EventSTPPredictor:
    """
    Event-level spatio-temporal point prediction for crowdsourced tasks.

    The model separates a task stream into:
    1. Intensity: number of new tasks expected in each fine time slot.
    2. Marks: concrete second-level time and geographic position for every
       expected event, sampled deterministically from relevant historic marks.

    This is intentionally distinct from dispatch predictors that aggregate
    demand into regions; ``predict`` returns one row per future task.
    """

    def __init__(
        self,
        slot_minutes: int = 5,
        recency_half_life_days: float = 28.0,
        same_weekday_multiplier: float = 3.0,
        ml_blend: float = 0.65,
        max_analogue_days: int = 21,
        calibration_days: int = 7,
        random_state: int = 42,
    ) -> None:
        if slot_minutes <= 0 or (24 * 60) % slot_minutes:
            raise ValueError("slot_minutes must be a positive divisor of 1440.")
        self.slot_minutes = int(slot_minutes)
        self.slots_per_day = (24 * 60) // self.slot_minutes
        self.recency_half_life_days = float(recency_half_life_days)
        self.same_weekday_multiplier = float(same_weekday_multiplier)
        self.ml_blend = float(np.clip(ml_blend, 0.0, 1.0))
        self.max_analogue_days = max(1, int(max_analogue_days))
        self.calibration_days = max(0, int(calibration_days))
        self.random_state = int(random_state)
        self.events: Optional[pd.DataFrame] = None
        self.daily_counts: Optional[pd.DataFrame] = None
        self.count_model: Optional[HistGradientBoostingRegressor] = None
        self.train_dates: Optional[pd.DatetimeIndex] = None
        self.preselected_target_date: Optional[pd.Timestamp] = None
        self.modeled_slots = np.arange(self.slots_per_day, dtype=int)
        self.calibrated_ml_blend = self.ml_blend
        self.intensity_weights = np.array([self.ml_blend, 1.0 - self.ml_blend, 0.0])
        self.intensity_scale = 1.0

    @staticmethod
    def _check_columns(df: pd.DataFrame) -> None:
        missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"Task data missing required columns: {missing}")

    def _prepare_events(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_columns(df)
        prepared = df.loc[:, REQUIRED_COLUMNS].copy()
        prepared["first_time"] = pd.to_datetime(prepared["first_time"], errors="raise")
        prepared["first_lon"] = pd.to_numeric(prepared["first_lon"], errors="raise")
        prepared["first_lat"] = pd.to_numeric(prepared["first_lat"], errors="raise")
        prepared = prepared.dropna().sort_values("first_time").reset_index(drop=True)
        prepared["event_date"] = prepared["first_time"].dt.normalize()
        minute_of_day = prepared["first_time"].dt.hour * 60 + prepared["first_time"].dt.minute
        prepared["slot_id"] = (minute_of_day // self.slot_minutes).astype(int)
        slot_start = prepared["event_date"] + pd.to_timedelta(
            prepared["slot_id"] * self.slot_minutes, unit="m"
        )
        prepared["offset_seconds"] = (
            prepared["first_time"] - slot_start
        ).dt.total_seconds().astype(float)
        return prepared

    def fit(
        self, events: pd.DataFrame, modeled_slots: Optional[Iterable[int]] = None
    ) -> "EventSTPPredictor":
        prepared = self._prepare_events(events)
        if prepared["event_date"].nunique() < 8:
            raise ValueError("At least 8 historic days are needed for event prediction.")
        if modeled_slots is None:
            self.modeled_slots = np.arange(self.slots_per_day, dtype=int)
        else:
            self.modeled_slots = np.asarray(sorted(set(int(slot) for slot in modeled_slots)), dtype=int)
            if not len(self.modeled_slots):
                raise ValueError("modeled_slots cannot be empty.")
        self.events = prepared
        self.preselected_target_date = None
        self.train_dates = pd.DatetimeIndex(sorted(prepared["event_date"].unique()))
        count_table = (
            prepared.groupby(["event_date", "slot_id"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=self.modeled_slots, fill_value=0)
            .sort_index()
        )
        self.daily_counts = count_table.astype(float)
        self._fit_count_model()
        return self

    def fit_from_directory(
        self,
        data_dir: str,
        before_date: Optional[str] = None,
        start_hour: float = 0.0,
        end_hour: float = 24.0,
        spatial_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> "EventSTPPredictor":
        if not 0.0 <= start_hour < end_hour <= 24.0:
            raise ValueError("Training hours must satisfy 0 <= start_hour < end_hour <= 24.")
        before_ts = pd.Timestamp(before_date).normalize() if before_date else None
        start_minute = int(round(start_hour * 60))
        end_minute = int(round(end_hour * 60))
        first_slot = start_minute // self.slot_minutes
        final_slot = int(math.ceil(end_minute / self.slot_minutes))
        modeled_slots = np.asarray(list(range(first_slot, final_slot)), dtype=int)
        daily_count_parts = []
        eligible_files = []
        for file_name in sorted(os.listdir(data_dir)):
            if not (file_name.startswith("tasks_") and file_name.endswith(".csv")):
                continue
            date_string = file_name[len("tasks_") : -len(".csv")]
            try:
                file_date = pd.Timestamp(date_string).normalize()
            except ValueError:
                continue
            if before_ts is not None and file_date >= before_ts:
                continue
            file_path = os.path.join(data_dir, file_name)
            eligible_files.append(file_path)
            frame = pd.read_csv(
                file_path,
                usecols=["first_time", "first_lon", "first_lat"],
                parse_dates=["first_time"],
            )
            minute_of_day = frame["first_time"].dt.hour * 60 + frame["first_time"].dt.minute
            keep = (minute_of_day >= start_minute) & (minute_of_day < end_minute)
            if spatial_bounds is not None:
                min_lon, max_lon, min_lat, max_lat = spatial_bounds
                keep &= (
                    frame["first_lon"].between(min_lon, max_lon)
                    & frame["first_lat"].between(min_lat, max_lat)
                )
            if before_ts is not None:
                keep &= frame["first_time"] < before_ts
            selected_times = frame.loc[keep, "first_time"]
            if selected_times.empty:
                continue
            selected_dates = selected_times.dt.normalize()
            selected_slots = (
                (selected_times.dt.hour * 60 + selected_times.dt.minute) // self.slot_minutes
            ).astype(int)
            daily_count_parts.append(
                pd.DataFrame({"event_date": selected_dates, "slot_id": selected_slots})
                .groupby(["event_date", "slot_id"])
                .size()
            )
        if not daily_count_parts:
            raise FileNotFoundError(f"No historic task CSV files found in {data_dir}")
        counts = (
            pd.concat(daily_count_parts, axis=1)
            .fillna(0)
            .sum(axis=1)
            .unstack(fill_value=0)
            .reindex(columns=modeled_slots, fill_value=0)
            .sort_index()
        )
        if len(counts.index) < 8:
            raise ValueError("At least 8 historic days are needed for event prediction.")
        target_for_weights = before_ts or (pd.Timestamp(counts.index.max()) + pd.Timedelta(days=1))
        candidate_dates = pd.DatetimeIndex(counts.index)
        candidate_weights = self._day_weights(target_for_weights, candidate_dates)
        analogue_dates = set(
            candidate_dates[np.argsort(-candidate_weights)[: self.max_analogue_days]]
        )
        analogue_frames = []
        for file_path in eligible_files:
            frame = pd.read_csv(
                file_path,
                usecols=list(REQUIRED_COLUMNS),
                parse_dates=["first_time"],
            )
            minute_of_day = frame["first_time"].dt.hour * 60 + frame["first_time"].dt.minute
            event_dates = frame["first_time"].dt.normalize()
            keep = (
                (minute_of_day >= start_minute)
                & (minute_of_day < end_minute)
                & event_dates.isin(analogue_dates)
            )
            if spatial_bounds is not None:
                min_lon, max_lon, min_lat, max_lat = spatial_bounds
                keep &= (
                    frame["first_lon"].between(min_lon, max_lon)
                    & frame["first_lat"].between(min_lat, max_lat)
                )
            if before_ts is not None:
                keep &= frame["first_time"] < before_ts
            if keep.any():
                analogue_frames.append(frame.loc[keep])
        if not analogue_frames:
            raise FileNotFoundError("No analogue task events remain for location generation.")
        self.modeled_slots = modeled_slots
        self.events = self._prepare_events(pd.concat(analogue_frames, ignore_index=True))
        self.preselected_target_date = pd.Timestamp(target_for_weights).normalize()
        self.train_dates = pd.DatetimeIndex(counts.index)
        self.daily_counts = counts.astype(float)
        self._fit_count_model()
        return self

    def _day_weights(self, target_date: pd.Timestamp, dates: Iterable[pd.Timestamp]) -> np.ndarray:
        date_index = pd.DatetimeIndex(dates)
        ages = np.maximum((target_date - date_index).days.astype(float), 1.0)
        decay = np.exp(-math.log(2.0) * ages / max(self.recency_half_life_days, 1.0))
        weekday_weight = np.where(
            date_index.dayofweek == target_date.dayofweek,
            self.same_weekday_multiplier,
            np.where(
                (date_index.dayofweek >= 5) == (target_date.dayofweek >= 5),
                1.0,
                0.45,
            ),
        )
        return decay * weekday_weight

    def _history_before(self, target_date: pd.Timestamp) -> pd.DataFrame:
        if self.daily_counts is None:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.daily_counts.loc[self.daily_counts.index < target_date]

    def _feature_vector(
        self, target_date: pd.Timestamp, slot_id: int, history: pd.DataFrame
    ) -> np.ndarray:
        if history.empty:
            raise ValueError("Historic counts are empty for target date.")
        values = history[slot_id].to_numpy(dtype=float)
        weights = self._day_weights(target_date, history.index)
        weighted_mean = float(np.average(values, weights=weights))
        weighted_var = float(np.average((values - weighted_mean) ** 2, weights=weights))
        same_weekday = history.loc[history.index.dayofweek == target_date.dayofweek, slot_id]
        same_weekday_mean = (
            float(same_weekday.tail(8).mean()) if len(same_weekday) else weighted_mean
        )
        slot_angle = 2.0 * math.pi * slot_id / self.slots_per_day
        weekday_angle = 2.0 * math.pi * target_date.dayofweek / 7.0
        lag_1 = float(values[-1])
        lag_7 = float(values[-7]) if len(values) >= 7 else weighted_mean
        return np.array(
            [
                math.sin(slot_angle),
                math.cos(slot_angle),
                math.sin(weekday_angle),
                math.cos(weekday_angle),
                float(target_date.dayofweek >= 5),
                lag_1,
                lag_7,
                float(np.mean(values[-3:])),
                float(np.mean(values[-7:])),
                same_weekday_mean,
                weighted_mean,
                math.sqrt(weighted_var),
            ],
            dtype=float,
        )

    def _train_count_model(self, daily_counts: pd.DataFrame) -> HistGradientBoostingRegressor:
        rows = []
        targets = []
        dates = daily_counts.index
        for day_index in range(7, len(dates)):
            target_date = pd.Timestamp(dates[day_index])
            history = daily_counts.iloc[:day_index]
            for slot_id in self.modeled_slots:
                rows.append(self._feature_vector(target_date, slot_id, history))
                targets.append(float(daily_counts.loc[target_date, slot_id]))
        model = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.07,
            max_iter=180,
            max_leaf_nodes=24,
            min_samples_leaf=25,
            l2_regularization=1.0,
            random_state=self.random_state,
        )
        model.fit(np.asarray(rows), np.asarray(targets))
        return model

    def _fit_count_model(self) -> None:
        if self.daily_counts is None:
            raise RuntimeError("Daily counts must be initialized first.")
        self.count_model = self._train_count_model(self.daily_counts)
        self._calibrate_intensity()

    def _calibrate_intensity(self) -> None:
        if self.daily_counts is None or self.calibration_days == 0:
            return
        dates = self.daily_counts.index
        validation_count = min(self.calibration_days, max(0, len(dates) - 15))
        if validation_count == 0:
            return
        weight_options = []
        for learned_units in range(11):
            for analogue_units in range(11 - learned_units):
                persistence_units = 10 - learned_units - analogue_units
                weight_options.append(
                    (learned_units / 10.0, analogue_units / 10.0, persistence_units / 10.0)
                )
        errors = {weights: [] for weights in weight_options}
        predicted_totals = {weights: 0.0 for weights in weight_options}
        actual_total = 0.0
        for target_date in dates[-validation_count:]:
            history = self.daily_counts.loc[self.daily_counts.index < target_date]
            model = self._train_count_model(history)
            features = np.vstack(
                [self._feature_vector(pd.Timestamp(target_date), int(slot_id), history)
                 for slot_id in self.modeled_slots]
            )
            learned = np.clip(model.predict(features), 0.0, None)
            analogue = features[:, 10]
            persistence = features[:, 5]
            actual = self.daily_counts.loc[target_date, self.modeled_slots].to_numpy(dtype=float)
            actual_total += float(actual.sum())
            for weights in weight_options:
                expected = (
                    weights[0] * learned
                    + weights[1] * analogue
                    + weights[2] * persistence
                )
                forecast = self._integerize_counts(expected).astype(float)
                errors[weights].extend(np.abs(actual - forecast).tolist())
                predicted_totals[weights] += float(forecast.sum())
        best_weights = min(errors, key=lambda weights: float(np.mean(errors[weights])))
        self.intensity_weights = np.asarray(best_weights, dtype=float)
        self.calibrated_ml_blend = float(self.intensity_weights[0])
        predicted_total = predicted_totals[best_weights]
        if predicted_total > 0.0:
            self.intensity_scale = float(np.clip(actual_total / predicted_total, 0.85, 1.15))

    def predict_intensity(self, target_date: str) -> pd.DataFrame:
        target_ts = pd.Timestamp(target_date).normalize()
        if self.events is not None and self.events["event_date"].max() >= target_ts:
            raise ValueError(
                "Fitted events include target_date or future events. "
                "Fit only on history strictly before the forecast date."
            )
        history = self._history_before(target_ts)
        if len(history) < 7 or self.count_model is None:
            raise ValueError("At least 7 days before target_date are required.")
        features = np.vstack(
            [self._feature_vector(target_ts, int(slot_id), history) for slot_id in self.modeled_slots]
        )
        learned = np.clip(self.count_model.predict(features), 0.0, None)
        analogue = features[:, 10]
        persistence = features[:, 5]
        expected = (
            self.intensity_weights[0] * learned
            + self.intensity_weights[1] * analogue
            + self.intensity_weights[2] * persistence
        ) * self.intensity_scale
        confidence = 1.0 / (1.0 + features[:, 11] / np.maximum(expected + 1.0, 1.0))
        return pd.DataFrame(
            {
                "slot_id": self.modeled_slots,
                "time_slot": target_ts
                + pd.to_timedelta(self.modeled_slots * self.slot_minutes, unit="m"),
                "expected_count": np.maximum(expected, 0.0),
                "confidence": np.clip(confidence, 0.05, 0.99),
            }
        )

    @staticmethod
    def _integerize_counts(expected: np.ndarray) -> np.ndarray:
        expected = np.maximum(np.asarray(expected, dtype=float), 0.0)
        integer_counts = np.floor(expected).astype(int)
        additions = int(round(float(expected.sum()))) - int(integer_counts.sum())
        if additions > 0:
            order = np.argsort(-(expected - integer_counts), kind="stable")
            integer_counts[order[:additions]] += 1
        return integer_counts

    def _candidate_events(
        self, target_date: pd.Timestamp, slot_id: int, minimum_events: int
    ) -> pd.DataFrame:
        if self.events is None:
            raise RuntimeError("Model must be fitted before prediction.")
        if self.preselected_target_date is not None and target_date == self.preselected_target_date:
            pool = self.events
        else:
            historical = self.events.loc[self.events["event_date"] < target_date]
            if historical.empty:
                raise ValueError("No events occur before target_date.")
            historical_dates = pd.DatetimeIndex(sorted(historical["event_date"].unique()))
            weights = self._day_weights(target_date, historical_dates)
            best_dates = historical_dates[np.argsort(-weights)[: self.max_analogue_days]]
            pool = historical.loc[historical["event_date"].isin(best_dates)]
        collected = pool.loc[pool["slot_id"] == slot_id].copy()
        radius = 1
        while len(collected) < minimum_events and radius < self.slots_per_day:
            neighbor_slots = {
                (slot_id - radius) % self.slots_per_day,
                (slot_id + radius) % self.slots_per_day,
            }
            neighbor_rows = pool.loc[pool["slot_id"].isin(neighbor_slots)].copy()
            if not neighbor_rows.empty:
                neighbor_rows["neighbor_distance"] = radius
                collected["neighbor_distance"] = collected.get("neighbor_distance", 0)
                collected = pd.concat([collected, neighbor_rows], ignore_index=True)
            radius += 1
        if collected.empty:
            raise ValueError(f"No analogue events available for slot {slot_id}.")
        if "neighbor_distance" not in collected:
            collected["neighbor_distance"] = 0
        event_weights = self._day_weights(target_date, collected["event_date"])
        collected["sample_weight"] = event_weights * np.power(0.55, collected["neighbor_distance"])
        return collected.reset_index(drop=True)

    def _sample_marks(
        self,
        target_date: pd.Timestamp,
        slot_id: int,
        event_count: int,
    ) -> pd.DataFrame:
        if event_count <= 0:
            return pd.DataFrame()
        candidates = self._candidate_events(target_date, slot_id, minimum_events=event_count * 2)
        weights = candidates["sample_weight"].to_numpy(dtype=float).copy()
        weights /= weights.sum()
        seed = self.random_state + target_date.toordinal() * self.slots_per_day + slot_id
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(candidates))
        shuffled = candidates.iloc[order].reset_index(drop=True)
        cumulative = np.cumsum(weights[order])
        start = rng.random() / event_count
        positions = start + np.arange(event_count) / event_count
        selected_indices = np.searchsorted(cumulative, positions, side="right")
        selected_indices = np.clip(selected_indices, 0, len(shuffled) - 1)
        selected = shuffled.iloc[selected_indices].copy()
        selected["first_time"] = (
            target_date
            + pd.to_timedelta(slot_id * self.slot_minutes, unit="m")
            + pd.to_timedelta(selected["offset_seconds"].to_numpy(), unit="s")
        )
        return selected

    def predict(
        self,
        target_date: str,
        start_hour: float = 0.0,
        end_hour: float = 24.0,
    ) -> pd.DataFrame:
        target_ts = pd.Timestamp(target_date).normalize()
        if not 0.0 <= start_hour < end_hour <= 24.0:
            raise ValueError("Prediction hours must satisfy 0 <= start_hour < end_hour <= 24.")
        intensity = self.predict_intensity(str(target_ts.date()))
        start_minute = int(round(start_hour * 60))
        end_minute = int(round(end_hour * 60))
        first_slot = start_minute // self.slot_minutes
        final_slot = int(math.ceil(end_minute / self.slot_minutes))
        required_slots = set(range(first_slot, final_slot))
        available_slots = set(int(slot_id) for slot_id in intensity["slot_id"])
        if not required_slots.issubset(available_slots):
            raise ValueError(
                "Requested prediction hours were not included when this predictor was fitted."
            )
        selected_intensity = intensity.loc[intensity["slot_id"].isin(required_slots)].copy()
        counts = self._integerize_counts(selected_intensity["expected_count"].to_numpy())
        predicted_frames = []
        sequence = 0
        for local_index, (_, row) in enumerate(selected_intensity.iterrows()):
            event_count = int(counts[local_index])
            marks = self._sample_marks(target_ts, int(row["slot_id"]), event_count)
            if marks.empty:
                continue
            mark_count = len(marks)
            marks["source_task_id"] = marks["task_id"].astype(str)
            marks["task_id"] = [
                f"pred_{target_ts.strftime('%Y%m%d')}_{sequence + index:07d}"
                for index in range(mark_count)
            ]
            sequence += mark_count
            marks["predicted_count_in_slot"] = event_count
            marks["expected_count_in_slot"] = float(row["expected_count"])
            marks["confidence"] = float(row["confidence"])
            marks["time_slot"] = row["time_slot"]
            predicted_frames.append(marks)
        if not predicted_frames:
            return pd.DataFrame(
                columns=[
                    "task_id",
                    "first_time",
                    "first_lon",
                    "first_lat",
                    "time_slot",
                    "predicted_count_in_slot",
                    "expected_count_in_slot",
                    "confidence",
                    "source_task_id",
                ]
            )
        result = pd.concat(predicted_frames, ignore_index=True)
        result = result.loc[
            (result["first_time"] >= target_ts + pd.Timedelta(minutes=start_minute))
            & (result["first_time"] < target_ts + pd.Timedelta(minutes=end_minute))
        ].copy()
        result = result[
            [
                "task_id",
                "first_time",
                "first_lon",
                "first_lat",
                "time_slot",
                "predicted_count_in_slot",
                "expected_count_in_slot",
                "confidence",
                "source_task_id",
            ]
        ].sort_values("first_time")
        return result.reset_index(drop=True)

    @staticmethod
    def _pairwise_distance_meters(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        actual = np.asarray(actual, dtype=np.float32)
        predicted = np.asarray(predicted, dtype=np.float32)
        reference_lat = np.radians(float(np.mean(actual[:, 1])))
        lon_scale = np.float32(111320.0 * math.cos(reference_lat))
        lat_scale = np.float32(111320.0)
        actual_xy = np.column_stack((actual[:, 0] * lon_scale, actual[:, 1] * lat_scale))
        predicted_xy = np.column_stack((predicted[:, 0] * lon_scale, predicted[:, 1] * lat_scale))
        delta = actual_xy[:, None, :] - predicted_xy[None, :, :]
        return np.sqrt(np.sum(delta * delta, axis=2, dtype=np.float32))

    def evaluate(
        self,
        actual_events: pd.DataFrame,
        predicted_events: pd.DataFrame,
        time_threshold_minutes: float = 2.5,
        distance_threshold_meters: float = 500.0,
        time_cost_meters_per_minute: float = 150.0,
    ) -> EventForecastMetrics:
        actual = self._prepare_events(actual_events)
        predicted = self._prepare_events(predicted_events)
        evaluated_slots = sorted(set(actual["slot_id"]) | set(predicted["slot_id"]))
        actual_count_by_slot = actual.groupby("slot_id").size().reindex(evaluated_slots, fill_value=0)
        pred_count_by_slot = predicted.groupby("slot_id").size().reindex(evaluated_slots, fill_value=0)
        count_delta = actual_count_by_slot.to_numpy() - pred_count_by_slot.to_numpy()
        distances = []
        time_errors = []
        for slot_id in sorted(set(actual["slot_id"]) & set(predicted["slot_id"])):
            actual_slot = actual.loc[actual["slot_id"] == slot_id]
            predicted_slot = predicted.loc[predicted["slot_id"] == slot_id]
            spatial_cost = self._pairwise_distance_meters(
                actual_slot[["first_lon", "first_lat"]].to_numpy(dtype=float),
                predicted_slot[["first_lon", "first_lat"]].to_numpy(dtype=float),
            )
            time_delta = np.abs(
                actual_slot["offset_seconds"].to_numpy(dtype=np.float32)[:, None]
                - predicted_slot["offset_seconds"].to_numpy(dtype=np.float32)[None, :]
            ) / np.float32(60.0)
            row_idx, col_idx = linear_sum_assignment(
                spatial_cost + np.float32(time_cost_meters_per_minute) * time_delta
            )
            distances.extend(spatial_cost[row_idx, col_idx].tolist())
            time_errors.extend(time_delta[row_idx, col_idx].tolist())
        distance_array = np.asarray(distances, dtype=float)
        time_array = np.asarray(time_errors, dtype=float)
        matched = int(len(distance_array))
        if matched:
            hits = (distance_array <= distance_threshold_meters) & (
                time_array <= time_threshold_minutes
            )
            time_mae = float(time_array.mean())
            location_mae = float(distance_array.mean())
            hit_rate = float(hits.mean())
        else:
            time_mae = float("nan")
            location_mae = float("nan")
            hit_rate = 0.0
        return EventForecastMetrics(
            actual_events=len(actual),
            predicted_events=len(predicted),
            matched_events=matched,
            count_absolute_error=abs(len(actual) - len(predicted)),
            slot_count_mae=float(np.mean(np.abs(count_delta))),
            slot_count_rmse=float(np.sqrt(np.mean(count_delta.astype(float) ** 2))),
            time_mae_minutes=time_mae,
            location_mae_meters=location_mae,
            joint_hit_rate=hit_rate,
            precision_by_count=matched / len(predicted) if len(predicted) else 0.0,
            recall_by_count=matched / len(actual) if len(actual) else 0.0,
        )
