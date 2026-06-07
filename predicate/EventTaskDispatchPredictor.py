from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

import config
from predicate.EventSTPPredictor import EventSTPPredictor


class EventTaskDispatchPredictor:
    """Adapt event-level time/location forecasts to the online dispatch interface."""

    def __init__(
        self,
        data_dir: str,
        coords: np.ndarray,
        nodes: List[Any],
        partition: Dict[Any, int],
        centers: Dict[int, Any],
        time_interval: int = 15,
        event_slot_minutes: int = 5,
        random_state: int = 42,
    ) -> None:
        self.data_dir = data_dir
        self.coords = coords
        self.nodes = nodes
        self.partition = partition
        self.centers = centers
        self.time_interval = int(time_interval)
        self.event_predictor = EventSTPPredictor(
            slot_minutes=event_slot_minutes,
            random_state=random_state,
        )
        self.forecast = pd.DataFrame()
        self.target_date: Optional[pd.Timestamp] = None

    @staticmethod
    def _current_map_bounds() -> tuple:
        center_lat, center_lon = config.CHENGDU_CENTER
        lat_delta = float(config.DOWNLOAD_DIST) / 111320.0
        lon_scale = max(np.cos(np.radians(center_lat)), 1e-6)
        lon_delta = float(config.DOWNLOAD_DIST) / (111320.0 * lon_scale)
        return (
            center_lon - lon_delta,
            center_lon + lon_delta,
            center_lat - lat_delta,
            center_lat + lat_delta,
        )

    def fit(self, target_date: str, start_hour: float, end_hour: float) -> "EventTaskDispatchPredictor":
        self.target_date = pd.Timestamp(target_date).normalize()
        spatial_bounds = self._current_map_bounds()
        self.event_predictor.fit_from_directory(
            self.data_dir,
            before_date=target_date,
            start_hour=start_hour,
            end_hour=end_hour,
            spatial_bounds=spatial_bounds,
        )
        forecast = self.event_predictor.predict(target_date, start_hour, end_hour)
        if forecast.empty:
            self.forecast = forecast
            return self
        min_lon, max_lon, min_lat, max_lat = spatial_bounds
        forecast = forecast[
            forecast["first_lon"].between(min_lon, max_lon)
            & forecast["first_lat"].between(min_lat, max_lat)
        ].copy()
        if forecast.empty:
            self.forecast = forecast
            return self
        tree = KDTree(self.coords)
        _, nearest = tree.query(forecast[["first_lon", "first_lat"]].to_numpy(dtype=float))
        forecast = forecast.copy()
        forecast["nearest_node"] = [self.nodes[int(index)] for index in nearest]
        forecast["region_id"] = forecast["nearest_node"].map(self.partition)
        forecast = forecast.dropna(subset=["region_id"]).copy()
        forecast["region_id"] = forecast["region_id"].astype(int)
        forecast["seconds_of_day"] = (
            forecast["first_time"].dt.hour * 3600
            + forecast["first_time"].dt.minute * 60
            + forecast["first_time"].dt.second
        )
        self.forecast = forecast.sort_values("first_time").reset_index(drop=True)
        return self

    def reset_online_state(self) -> None:
        return None

    def predict_tasks(self, slot_timestamp: pd.Timestamp) -> pd.DataFrame:
        slot_start = pd.Timestamp(slot_timestamp)
        slot_end = slot_start + pd.Timedelta(minutes=self.time_interval)
        return self.forecast.loc[
            (self.forecast["first_time"] >= slot_start)
            & (self.forecast["first_time"] < slot_end)
        ].copy()

    def predict_dispatch_tasks(self, slot_timestamp: pd.Timestamp) -> Dict[int, list]:
        tasks = {rid: [] for rid in self.centers.keys()}
        for _, row in self.predict_tasks(slot_timestamp).iterrows():
            rid = int(row["region_id"])
            tasks[rid].append(
                (
                    row["nearest_node"],
                    row["task_id"],
                    float(row["confidence"]),
                    float(row["seconds_of_day"]) + float(config.TASK_EXPIRE_MINUTES) * 60,
                    float(row["seconds_of_day"]),
                )
            )
        return tasks

    def predict_region_demand(self, slot_timestamp: pd.Timestamp) -> Dict[int, int]:
        tasks = self.predict_tasks(slot_timestamp)
        counts = tasks.groupby("region_id").size() if not tasks.empty else {}
        return {rid: int(counts.get(rid, 0)) for rid in self.centers.keys()}

    def predict_region_distribution(self, slot_timestamp: pd.Timestamp) -> Dict[int, Dict[str, float]]:
        tasks = self.predict_tasks(slot_timestamp)
        distribution = {}
        for rid in self.centers.keys():
            local = tasks.loc[tasks["region_id"] == rid] if not tasks.empty else tasks
            mu = float(len(local))
            confidence = float(local["confidence"].mean()) if len(local) else 0.5
            sigma = max(1.0, np.sqrt(max(mu, 1.0)) * (1.0 + (1.0 - confidence)))
            distribution[rid] = {
                "mu": mu,
                "sigma": float(sigma),
                "q90": float(mu + 1.2816 * sigma),
                "burst_prob": float(np.clip(1.0 - confidence, 0.0, 1.0)),
                "hist_bias": 0.0,
                "hist_abs_bias": sigma,
            }
        return distribution

    def update_online(
        self,
        slot_timestamp: pd.Timestamp,
        actual_region_demand: Dict[int, int],
        predicted_region_demand: Dict[int, int],
    ) -> None:
        return None
