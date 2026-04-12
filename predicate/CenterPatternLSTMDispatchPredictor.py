import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.spatial import KDTree

from predicate.model.CenterTaskPatternNet import CenterTaskPatternNet


class CenterPatternLSTMDispatchPredictor:
    def __init__(
        self,
        data_dir: str,
        coords: np.ndarray,
        nodes: List[Any],
        partition: Dict[Any, int],
        centers: Dict[int, Any],
        time_interval: int = 15,
        seq_len: int = 32,
        pre_len: int = 1,
        max_epochs: int = 400,
        patience: int = 60,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        hidden_dim: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.15,
        log_interval: int = 20,
        hotspot_alpha: float = 2.5,
        underpredict_alpha: float = 3.0,
        underpredict_power: float = 1.0,
        refit_on_all_pretarget: bool = True,
        use_online_adaptation: bool = True,
        online_bias_alpha: float = 0.30,
        online_slot_bias_alpha: float = 0.40,
        online_scale_alpha: float = 0.15,
        use_log1p: bool = True,
        device: Optional[str] = None,
    ):
        if pre_len != 1:
            raise ValueError("CenterPatternLSTMDispatchPredictor currently supports pre_len=1 only.")

        self.data_dir = data_dir
        self.coords = coords
        self.nodes = nodes
        self.partition = partition
        self.centers = centers
        self.time_interval = time_interval
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr = lr
        self.weight_decay = weight_decay
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.log_interval = max(1, log_interval)
        self.hotspot_alpha = hotspot_alpha
        self.underpredict_alpha = underpredict_alpha
        self.underpredict_power = underpredict_power
        self.refit_on_all_pretarget = refit_on_all_pretarget
        self.use_online_adaptation = use_online_adaptation
        self.online_bias_alpha = float(online_bias_alpha)
        self.online_slot_bias_alpha = float(online_slot_bias_alpha)
        self.online_scale_alpha = float(online_scale_alpha)
        self.use_log1p = use_log1p
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = None
        self.sorted_region_ids = sorted(centers.keys())
        self.num_centers = len(self.sorted_region_ids)
        self.context_start_hour = 0
        self.context_end_hour = 24
        self.num_time_slots = max(1, (24 * 60) // self.time_interval)
        self.full_demand_matrix = None
        self.full_slots = None
        self.full_slot_to_idx = None
        self.slot_means = None
        self.slot_stds = None
        self.slot_maxs = None
        self.weekday_slot_means = None
        self.weekday_slot_stds = None
        self.weekday_slot_maxs = None
        self.center_linear_models = {
            rid: np.array([1.0, 0.25, 0.25, 0.15, 0.0], dtype=np.float32)
            for rid in self.sorted_region_ids
        }
        self.region_scale_factors = {rid: 1.0 for rid in self.sorted_region_ids}
        self.reset_online_state()

    @staticmethod
    def _clone_state(model):
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def _transform(self, array: np.ndarray) -> np.ndarray:
        array = np.asarray(array, dtype=np.float32)
        if self.use_log1p:
            return np.log1p(np.clip(array, 0.0, None))
        return array

    def _inverse(self, array: np.ndarray) -> np.ndarray:
        array = np.asarray(array, dtype=np.float32)
        if self.use_log1p:
            return np.expm1(array)
        return array

    def _weighted_center_loss(self, pred: torch.Tensor, raw_target: torch.Tensor) -> torch.Tensor:
        pred_raw = torch.expm1(pred) if self.use_log1p else pred
        pred_raw = torch.clamp(pred_raw, min=0.0)
        residual = pred_raw - raw_target
        hotspot_weights = 1.0 + self.hotspot_alpha * torch.log1p(torch.clamp(raw_target, min=0.0))
        under_gap = torch.clamp(raw_target - pred_raw, min=0.0)
        relative_under_gap = under_gap / torch.clamp(raw_target + 1.0, min=1.0)
        under_weights = 1.0 + self.underpredict_alpha * torch.pow(relative_under_gap, self.underpredict_power)
        return ((residual ** 2) * hotspot_weights * under_weights).mean()

    def _build_model(self, aux_dim: int):
        return CenterTaskPatternNet(
            num_centers=self.num_centers,
            seq_len=self.seq_len,
            aux_dim=aux_dim,
            hidden_dim=self.hidden_dim,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout,
            slot_vocab_size=self.num_time_slots,
            weekday_vocab_size=7,
        ).to(self.device)

    def _train_fixed_epochs(
        self,
        X_short,
        X_aux,
        X_base,
        y_raw,
        slot_ids,
        weekday_ids,
        is_weekend,
        epochs: int,
    ) -> None:
        if epochs <= 0:
            return

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        for _ in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            pred = self.model(
                X_short,
                X_aux,
                X_base,
                slot_ids,
                weekday_ids,
                is_weekend,
            )
            loss = self._weighted_center_loss(pred, y_raw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            optimizer.step()
        self.model.eval()

    def _load_dates(self, dates: Sequence[str]) -> pd.DataFrame:
        frames = []
        for date_str in dates:
            file_path = os.path.join(self.data_dir, f"tasks_{date_str}.csv")
            if not os.path.exists(file_path):
                continue
            frames.append(pd.read_csv(file_path))
        if not frames:
            raise FileNotFoundError(f"No task files found for dates: {dates}")
        df = pd.concat(frames, ignore_index=True)
        df["first_time"] = pd.to_datetime(df["first_time"])
        return df

    def _build_slots_for_dates(self, dates: Sequence[str], start_hour: int, end_hour: int) -> pd.DatetimeIndex:
        normalized_dates = sorted(pd.to_datetime(pd.Series(list(dates))).dt.normalize().unique())
        all_slots = []
        for date_value in normalized_dates:
            day_start = pd.Timestamp(date_value) + pd.Timedelta(hours=start_hour)
            day_end = pd.Timestamp(date_value) + pd.Timedelta(hours=end_hour)
            if end_hour >= 24:
                day_end = pd.Timestamp(date_value) + pd.Timedelta(days=1)
            day_slots = pd.date_range(
                day_start,
                day_end - pd.Timedelta(minutes=self.time_interval),
                freq=f"{self.time_interval}min",
            )
            all_slots.extend(day_slots)
        return pd.DatetimeIndex(all_slots)

    def _aggregate_tasks_to_centers(
        self,
        df: pd.DataFrame,
        dates: Sequence[str],
        start_hour: int,
        end_hour: int,
    ) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        all_slots = self._build_slots_for_dates(dates, start_hour, end_hour)
        demand_matrix = np.zeros((len(all_slots), self.num_centers), dtype=np.float32)
        if len(df) == 0:
            return demand_matrix, all_slots

        df = df.copy()
        df["date"] = df["first_time"].dt.normalize()
        allowed_dates = set(pd.to_datetime(pd.Series(list(dates))).dt.normalize())
        df = df[df["date"].isin(allowed_dates)].copy()
        if len(df) == 0:
            return demand_matrix, all_slots

        minutes_of_day = df["first_time"].dt.hour * 60 + df["first_time"].dt.minute
        start_minute = start_hour * 60
        end_minute = end_hour * 60 if end_hour < 24 else 24 * 60
        mask = (minutes_of_day >= start_minute) & (minutes_of_day < end_minute)
        df = df[mask].copy()
        if len(df) == 0:
            return demand_matrix, all_slots

        task_coords = df[["first_lon", "first_lat"]].values
        tree = KDTree(self.coords)
        _, idxs = tree.query(task_coords)
        df["nearest_node"] = [self.nodes[i] for i in idxs]
        df["region_id"] = df["nearest_node"].map(self.partition)
        df = df[df["region_id"].isin(self.sorted_region_ids)].copy()
        if len(df) == 0:
            return demand_matrix, all_slots

        df["time_slot"] = df["first_time"].dt.floor(f"{self.time_interval}min")
        grouped = df.groupby(["time_slot", "region_id"]).size().reset_index(name="count")
        slot_to_idx = {pd.Timestamp(slot): idx for idx, slot in enumerate(all_slots)}
        region_to_idx = {rid: idx for idx, rid in enumerate(self.sorted_region_ids)}

        for _, row in grouped.iterrows():
            slot_idx = slot_to_idx.get(pd.Timestamp(row["time_slot"]))
            region_idx = region_to_idx.get(int(row["region_id"]))
            if slot_idx is None or region_idx is None:
                continue
            demand_matrix[slot_idx, region_idx] = float(row["count"])

        return demand_matrix, all_slots

    def _slot_id(self, ts: pd.Timestamp) -> int:
        return int((ts.hour * 60 + ts.minute) // self.time_interval)

    def _compute_slot_statistics(self, train_matrix: np.ndarray, train_slots: pd.DatetimeIndex) -> None:
        slot_buckets = {slot_id: [] for slot_id in range(self.num_time_slots)}
        weekday_slot_buckets = {(weekday, slot_id): [] for weekday in range(7) for slot_id in range(self.num_time_slots)}

        for idx, ts in enumerate(train_slots):
            slot_id = self._slot_id(pd.Timestamp(ts))
            weekday = int(pd.Timestamp(ts).dayofweek)
            slot_buckets[slot_id].append(train_matrix[idx])
            weekday_slot_buckets[(weekday, slot_id)].append(train_matrix[idx])

    def reset_online_state(self) -> None:
        self.online_region_bias = {rid: 0.0 for rid in self.sorted_region_ids}
        self.online_region_scale = {rid: 1.0 for rid in self.sorted_region_ids}
        self.online_slot_bias = {rid: {} for rid in self.sorted_region_ids}
        self.online_weekday_slot_bias = {rid: {} for rid in self.sorted_region_ids}

        self.slot_means = np.zeros((self.num_time_slots, self.num_centers), dtype=np.float32)
        self.slot_stds = np.zeros((self.num_time_slots, self.num_centers), dtype=np.float32)
        self.slot_maxs = np.zeros((self.num_time_slots, self.num_centers), dtype=np.float32)
        self.weekday_slot_means = np.zeros((7, self.num_time_slots, self.num_centers), dtype=np.float32)
        self.weekday_slot_stds = np.zeros((7, self.num_time_slots, self.num_centers), dtype=np.float32)
        self.weekday_slot_maxs = np.zeros((7, self.num_time_slots, self.num_centers), dtype=np.float32)

        global_mean = np.mean(train_matrix, axis=0).astype(np.float32)
        global_std = np.std(train_matrix, axis=0).astype(np.float32)
        global_max = np.max(train_matrix, axis=0).astype(np.float32)

        for slot_id in range(self.num_time_slots):
            samples = np.asarray(slot_buckets[slot_id], dtype=np.float32)
            if len(samples) == 0:
                self.slot_means[slot_id] = global_mean
                self.slot_stds[slot_id] = global_std
                self.slot_maxs[slot_id] = global_max
            else:
                self.slot_means[slot_id] = samples.mean(axis=0)
                self.slot_stds[slot_id] = samples.std(axis=0)
                self.slot_maxs[slot_id] = samples.max(axis=0)

        for weekday in range(7):
            for slot_id in range(self.num_time_slots):
                samples = np.asarray(weekday_slot_buckets[(weekday, slot_id)], dtype=np.float32)
                if len(samples) == 0:
                    self.weekday_slot_means[weekday, slot_id] = self.slot_means[slot_id]
                    self.weekday_slot_stds[weekday, slot_id] = self.slot_stds[slot_id]
                    self.weekday_slot_maxs[weekday, slot_id] = self.slot_maxs[slot_id]
                else:
                    self.weekday_slot_means[weekday, slot_id] = samples.mean(axis=0)
                    self.weekday_slot_stds[weekday, slot_id] = samples.std(axis=0)
                    self.weekday_slot_maxs[weekday, slot_id] = samples.max(axis=0)

    def _window_mean(self, matrix: np.ndarray, center_idx: int, target_idx: Optional[int], radius: int = 1) -> float:
        if target_idx is None:
            return 0.0
        lo = max(0, target_idx - radius)
        hi = min(len(matrix), target_idx + radius + 1)
        if lo >= hi:
            return 0.0
        return float(np.mean(matrix[lo:hi, center_idx]))

    def _build_single_sample(
        self,
        matrix: np.ndarray,
        slots: pd.DatetimeIndex,
        slot_to_idx: Dict[pd.Timestamp, int],
        target_idx: int,
    ) -> Dict[str, np.ndarray]:
        target_ts = pd.Timestamp(slots[target_idx])
        slot_id = self._slot_id(target_ts)
        weekday_id = int(target_ts.dayofweek)

        short_seq = matrix[target_idx - self.seq_len:target_idx]
        last_values = matrix[target_idx - 1]
        recent_4 = matrix[max(0, target_idx - 4):target_idx]
        recent_8 = matrix[max(0, target_idx - 8):target_idx]

        prev_day_idx = slot_to_idx.get(target_ts - pd.Timedelta(days=1))
        prev_week_idx = slot_to_idx.get(target_ts - pd.Timedelta(days=7))

        prev_day_same = matrix[prev_day_idx] if prev_day_idx is not None else self.slot_means[slot_id]
        prev_week_same = matrix[prev_week_idx] if prev_week_idx is not None else self.weekday_slot_means[weekday_id, slot_id]

        center_aux = []
        base_components = []
        for center_idx in range(self.num_centers):
            recent_mean_4 = float(np.mean(recent_4[:, center_idx])) if len(recent_4) > 0 else 0.0
            recent_mean_8 = float(np.mean(recent_8[:, center_idx])) if len(recent_8) > 0 else 0.0
            recent_max_4 = float(np.max(recent_4[:, center_idx])) if len(recent_4) > 0 else 0.0
            if len(recent_4) >= 2:
                recent_slope_4 = float(recent_4[-1, center_idx] - recent_4[0, center_idx]) / max(len(recent_4) - 1, 1)
            else:
                recent_slope_4 = 0.0

            prev_day_window_mean = self._window_mean(matrix, center_idx, prev_day_idx, radius=1)

            aux_values = np.array(
                [
                    last_values[center_idx],
                    recent_mean_4,
                    recent_mean_8,
                    recent_max_4,
                    recent_slope_4,
                    prev_day_same[center_idx],
                    prev_day_window_mean,
                    prev_week_same[center_idx],
                    self.slot_means[slot_id, center_idx],
                    self.slot_stds[slot_id, center_idx],
                    self.slot_maxs[slot_id, center_idx],
                    self.weekday_slot_means[weekday_id, slot_id, center_idx],
                    self.weekday_slot_stds[weekday_id, slot_id, center_idx],
                    self.weekday_slot_maxs[weekday_id, slot_id, center_idx],
                ],
                dtype=np.float32,
            )
            center_aux.append(aux_values)
            base_components.append(
                np.array(
                    [
                        last_values[center_idx],
                        prev_day_same[center_idx],
                        self.weekday_slot_means[weekday_id, slot_id, center_idx],
                    ],
                    dtype=np.float32,
                )
            )

        return {
            "short_seq": self._transform(short_seq),
            "center_aux": self._transform(np.stack(center_aux, axis=0)),
            "base_components": self._transform(np.stack(base_components, axis=0)),
            "target_raw": matrix[target_idx].astype(np.float32),
            "target_transformed": self._transform(matrix[target_idx]),
            "slot_id": np.int64(slot_id),
            "weekday_id": np.int64(weekday_id),
            "is_weekend": np.float32(1.0 if weekday_id >= 5 else 0.0),
            "target_ts": target_ts,
        }

    def _build_dataset(
        self,
        matrix: np.ndarray,
        slots: pd.DatetimeIndex,
        target_date_set: Sequence[pd.Timestamp],
    ) -> Dict[str, np.ndarray]:
        slot_to_idx = {pd.Timestamp(slot): idx for idx, slot in enumerate(slots)}
        normalized_dates = {pd.Timestamp(ts).normalize() for ts in target_date_set}

        samples = []
        for target_idx in range(self.seq_len, len(slots)):
            target_ts = pd.Timestamp(slots[target_idx])
            if target_ts.normalize() not in normalized_dates:
                continue
            samples.append(self._build_single_sample(matrix, slots, slot_to_idx, target_idx))

        if not samples:
            raise ValueError("No valid samples were built for the requested dates.")

        return {
            "short_seq": np.stack([sample["short_seq"] for sample in samples], axis=0).astype(np.float32),
            "center_aux": np.stack([sample["center_aux"] for sample in samples], axis=0).astype(np.float32),
            "base_components": np.stack([sample["base_components"] for sample in samples], axis=0).astype(np.float32),
            "target_raw": np.stack([sample["target_raw"] for sample in samples], axis=0).astype(np.float32),
            "target_transformed": np.stack([sample["target_transformed"] for sample in samples], axis=0).astype(np.float32),
            "slot_ids": np.array([sample["slot_id"] for sample in samples], dtype=np.int64),
            "weekday_ids": np.array([sample["weekday_id"] for sample in samples], dtype=np.int64),
            "is_weekend": np.array([sample["is_weekend"] for sample in samples], dtype=np.float32),
            "target_timestamps": [sample["target_ts"] for sample in samples],
        }

    def _fit_center_calibration(
        self,
        pred_raw: np.ndarray,
        val_data: Dict[str, np.ndarray],
    ) -> None:
        target_raw = val_data["target_raw"]
        base_components = self._inverse(val_data["base_components"])
        slot_ids = val_data["slot_ids"]
        weekday_ids = val_data["weekday_ids"]

        for center_idx, rid in enumerate(self.sorted_region_ids):
            features = np.column_stack(
                [
                    pred_raw[:, center_idx],
                    base_components[:, center_idx, 0],
                    base_components[:, center_idx, 1],
                    self.weekday_slot_means[weekday_ids, slot_ids, center_idx],
                    np.ones(len(pred_raw), dtype=np.float32),
                ]
            )
            target = target_raw[:, center_idx]
            coef, *_ = np.linalg.lstsq(features, target, rcond=None)
            coef = coef.astype(np.float32)
            coef[:4] = np.clip(coef[:4], -1.0, 3.0)
            coef[4] = float(np.clip(coef[4], -50.0, 50.0))
            self.center_linear_models[rid] = coef

        pred_totals = np.maximum(pred_raw.sum(axis=0), 1e-6)
        actual_totals = np.maximum(target_raw.sum(axis=0), 0.0)
        global_scale = float(np.clip(np.sum(actual_totals) / max(np.sum(pred_totals), 1e-6), 0.7, 2.5))
        for center_idx, rid in enumerate(self.sorted_region_ids):
            center_scale = actual_totals[center_idx] / pred_totals[center_idx]
            self.region_scale_factors[rid] = float(np.clip(0.5 * center_scale + 0.5 * global_scale, 0.7, 2.5))

    def fit(
        self,
        train_dates: List[str],
        val_dates: List[str],
        target_date: str,
        history_start_hour: int,
        end_hour: int,
    ) -> None:
        self.context_start_hour = 0
        self.context_end_hour = 24
        self.num_time_slots = max(1, (24 * 60) // self.time_interval)

        all_context_dates = sorted(set(train_dates + val_dates + [target_date]))
        full_df = self._load_dates(all_context_dates)
        self.full_demand_matrix, self.full_slots = self._aggregate_tasks_to_centers(
            full_df,
            dates=all_context_dates,
            start_hour=self.context_start_hour,
            end_hour=self.context_end_hour,
        )
        self.full_slot_to_idx = {pd.Timestamp(slot): idx for idx, slot in enumerate(self.full_slots)}

        train_dates_ts = pd.to_datetime(pd.Series(train_dates)).dt.normalize().tolist()
        val_dates_ts = pd.to_datetime(pd.Series(val_dates)).dt.normalize().tolist()

        train_mask = pd.Series(self.full_slots).dt.normalize().isin(train_dates_ts).to_numpy()
        train_matrix = self.full_demand_matrix[train_mask]
        train_slots = self.full_slots[train_mask]
        self._compute_slot_statistics(train_matrix, train_slots)

        train_data = self._build_dataset(self.full_demand_matrix, self.full_slots, train_dates_ts)
        val_data = self._build_dataset(self.full_demand_matrix, self.full_slots, val_dates_ts)

        aux_dim = train_data["center_aux"].shape[-1]
        self.model = self._build_model(aux_dim)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(10, self.patience // 4)
        )

        X_train_short = torch.FloatTensor(train_data["short_seq"]).to(self.device)
        X_train_aux = torch.FloatTensor(train_data["center_aux"]).to(self.device)
        X_train_base = torch.FloatTensor(train_data["base_components"]).to(self.device)
        y_train_raw = torch.FloatTensor(train_data["target_raw"]).to(self.device)
        slot_train = torch.LongTensor(train_data["slot_ids"]).to(self.device)
        weekday_train = torch.LongTensor(train_data["weekday_ids"]).to(self.device)
        weekend_train = torch.FloatTensor(train_data["is_weekend"]).to(self.device)

        X_val_short = torch.FloatTensor(val_data["short_seq"]).to(self.device)
        X_val_aux = torch.FloatTensor(val_data["center_aux"]).to(self.device)
        X_val_base = torch.FloatTensor(val_data["base_components"]).to(self.device)
        y_val_raw = torch.FloatTensor(val_data["target_raw"]).to(self.device)
        slot_val = torch.LongTensor(val_data["slot_ids"]).to(self.device)
        weekday_val = torch.LongTensor(val_data["weekday_ids"]).to(self.device)
        weekend_val = torch.FloatTensor(val_data["is_weekend"]).to(self.device)

        best_val_loss = float("inf")
        best_state = self._clone_state(self.model)
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            pred_train = self.model(
                X_train_short,
                X_train_aux,
                X_train_base,
                slot_train,
                weekday_train,
                weekend_train,
            )
            train_loss = self._weighted_center_loss(pred_train, y_train_raw)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                pred_val = self.model(
                    X_val_short,
                    X_val_aux,
                    X_val_base,
                    slot_val,
                    weekday_val,
                    weekend_val,
                )
                val_loss = self._weighted_center_loss(pred_val, y_val_raw).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self._clone_state(self.model)
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % self.log_interval == 0 or epoch == 1:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"   [CenterPatternLSTM] Epoch [{epoch:04d}/{self.max_epochs}], "
                    f"Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}"
                )

            if patience_counter >= self.patience:
                print(
                    f"   [CenterPatternLSTM] Early stop at epoch {epoch}, "
                    f"best epoch = {best_epoch}, best val loss = {best_val_loss:.4f}"
                )
                break

        self.model.load_state_dict(best_state)
        self.model.to(self.device)
        self.model.eval()
        print(f"   [CenterPatternLSTM] Best Epoch: {best_epoch}, Best Val Loss: {best_val_loss:.4f}")

        if self.refit_on_all_pretarget and best_epoch > 0:
            print(f"   [CenterPatternLSTM] Refit on train+val for {best_epoch} epochs...")
            refit_short = np.concatenate([train_data["short_seq"], val_data["short_seq"]], axis=0)
            refit_aux = np.concatenate([train_data["center_aux"], val_data["center_aux"]], axis=0)
            refit_base = np.concatenate([train_data["base_components"], val_data["base_components"]], axis=0)
            refit_target = np.concatenate([train_data["target_raw"], val_data["target_raw"]], axis=0)
            refit_slot_ids = np.concatenate([train_data["slot_ids"], val_data["slot_ids"]], axis=0)
            refit_weekday_ids = np.concatenate([train_data["weekday_ids"], val_data["weekday_ids"]], axis=0)
            refit_is_weekend = np.concatenate([train_data["is_weekend"], val_data["is_weekend"]], axis=0)

            self.model = self._build_model(aux_dim)
            self._train_fixed_epochs(
                X_short=torch.FloatTensor(refit_short).to(self.device),
                X_aux=torch.FloatTensor(refit_aux).to(self.device),
                X_base=torch.FloatTensor(refit_base).to(self.device),
                y_raw=torch.FloatTensor(refit_target).to(self.device),
                slot_ids=torch.LongTensor(refit_slot_ids).to(self.device),
                weekday_ids=torch.LongTensor(refit_weekday_ids).to(self.device),
                is_weekend=torch.FloatTensor(refit_is_weekend).to(self.device),
                epochs=best_epoch,
            )

        with torch.no_grad():
            pred_val = self.model(
                X_val_short,
                X_val_aux,
                X_val_base,
                slot_val,
                weekday_val,
                weekend_val,
            ).detach().cpu().numpy()
        pred_val_raw = np.clip(self._inverse(pred_val), 0.0, None)
        self._fit_center_calibration(pred_val_raw, val_data)

    def predict_region_demand(self, slot_timestamp: pd.Timestamp) -> Optional[Dict[int, int]]:
        if self.model is None or self.full_demand_matrix is None or self.full_slots is None:
            raise RuntimeError("CenterPatternLSTMDispatchPredictor must be fitted before prediction.")

        target_slot = pd.Timestamp(slot_timestamp)
        target_idx = self.full_slot_to_idx.get(target_slot)
        if target_idx is None or target_idx < self.seq_len:
            return None

        sample = self._build_single_sample(
            self.full_demand_matrix,
            self.full_slots,
            self.full_slot_to_idx,
            target_idx,
        )

        short_x = torch.FloatTensor(sample["short_seq"]).unsqueeze(0).to(self.device)
        aux_x = torch.FloatTensor(sample["center_aux"]).unsqueeze(0).to(self.device)
        base_x = torch.FloatTensor(sample["base_components"]).unsqueeze(0).to(self.device)
        slot_ids = torch.LongTensor([sample["slot_id"]]).to(self.device)
        weekday_ids = torch.LongTensor([sample["weekday_id"]]).to(self.device)
        is_weekend = torch.FloatTensor([sample["is_weekend"]]).to(self.device)

        with torch.no_grad():
            pred = self.model(
                short_x,
                aux_x,
                base_x,
                slot_ids,
                weekday_ids,
                is_weekend,
            ).detach().cpu().numpy()[0]

        pred_raw = np.clip(self._inverse(pred), 0.0, None)
        base_raw = self._inverse(sample["base_components"])
        calibrated = {}
        for center_idx, rid in enumerate(self.sorted_region_ids):
            coef = self.center_linear_models.get(rid)
            if coef is None:
                adjusted = pred_raw[center_idx]
            else:
                adjusted = (
                    coef[0] * pred_raw[center_idx]
                    + coef[1] * base_raw[center_idx, 0]
                    + coef[2] * base_raw[center_idx, 1]
                    + coef[3] * self.weekday_slot_means[sample["weekday_id"], sample["slot_id"], center_idx]
                    + coef[4]
                )
            adjusted *= self.region_scale_factors.get(rid, 1.0)
            if self.use_online_adaptation:
                slot_bias = self.online_slot_bias.get(rid, {}).get(sample["slot_id"], 0.0)
                weekday_slot_bias = self.online_weekday_slot_bias.get(rid, {}).get((sample["weekday_id"], sample["slot_id"]), 0.0)
                online_scale = self.online_region_scale.get(rid, 1.0)
                online_bias = self.online_region_bias.get(rid, 0.0)
                adjusted = adjusted * online_scale + online_bias + slot_bias + weekday_slot_bias
            calibrated[rid] = int(round(max(0.0, adjusted)))
        return calibrated

    def update_online(
        self,
        slot_timestamp: pd.Timestamp,
        actual_region_demand: Dict[int, int],
        predicted_region_demand: Optional[Dict[int, int]] = None,
    ) -> None:
        if not self.use_online_adaptation:
            return

        ts = pd.Timestamp(slot_timestamp)
        slot_id = self._slot_id(ts)
        weekday_id = int(ts.dayofweek)

        for rid in self.sorted_region_ids:
            actual = float(actual_region_demand.get(rid, 0.0))
            pred = float(predicted_region_demand.get(rid, actual)) if predicted_region_demand is not None else actual
            error = actual - pred
            target_scale = float(np.clip(actual / max(pred, 1.0), 0.5, 2.5))

            self.online_region_bias[rid] = (
                (1.0 - self.online_bias_alpha) * self.online_region_bias.get(rid, 0.0)
                + self.online_bias_alpha * error
            )
            self.online_region_scale[rid] = float(np.clip(
                (1.0 - self.online_scale_alpha) * self.online_region_scale.get(rid, 1.0)
                + self.online_scale_alpha * target_scale,
                0.5,
                2.5
            ))

            prev_slot_bias = self.online_slot_bias.setdefault(rid, {}).get(slot_id, 0.0)
            self.online_slot_bias[rid][slot_id] = (
                (1.0 - self.online_slot_bias_alpha) * prev_slot_bias
                + self.online_slot_bias_alpha * error
            )

            weekday_slot_key = (weekday_id, slot_id)
            prev_weekday_slot_bias = self.online_weekday_slot_bias.setdefault(rid, {}).get(weekday_slot_key, 0.0)
            self.online_weekday_slot_bias[rid][weekday_slot_key] = (
                (1.0 - self.online_slot_bias_alpha) * prev_weekday_slot_bias
                + self.online_slot_bias_alpha * error
            )
