import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.spatial import KDTree

from predicate.data_pipeline import SpatioTemporalDataset
from predicate.model.MCTGNet import MCTGNet


class MCTGNetDispatchPredictor:
    def __init__(
            self,
            data_dir: str,
            coords: np.ndarray,
            nodes: List[Any],
            partition: Dict[Any, int],
            centers: Dict[int, Any],
            time_interval: int = 15,
            seq_len: int = 4,
            pre_len: int = 1,
            max_epochs: int = 300,
            patience: int = 50,
            lr: float = 5e-4,
            log_interval: int = 20,
            weight_decay: float = 1e-4,
            hotspot_alpha: float = 2.5,
            center_loss_weight: float = 0.35,
            center_hotspot_alpha: float = 1.5,
            center_underpredict_alpha: float = 2.0,
            center_underpredict_power: float = 1.0,
            use_lstm_branch: bool = True,
            lstm_layers: int = 1,
            lstm_dropout: float = 0.1,
            refit_on_all_pretarget: bool = True,
            use_online_adaptation: bool = True,
            online_bias_alpha: float = 0.30,
            online_slot_bias_alpha: float = 0.40,
            online_scale_alpha: float = 0.15,
            uncertainty_quantile: float = 0.90,
            uncertainty_slot_blend: float = 0.65,
            online_uncertainty_alpha: float = 0.20,
            min_sigma_ratio: float = 0.15,
            use_log1p: bool = True,
            device: Optional[str] = None
    ):
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
        self.log_interval = max(1, log_interval)
        self.weight_decay = weight_decay
        self.hotspot_alpha = hotspot_alpha
        self.center_loss_weight = center_loss_weight
        self.center_hotspot_alpha = center_hotspot_alpha
        self.center_underpredict_alpha = center_underpredict_alpha
        self.center_underpredict_power = center_underpredict_power
        self.use_lstm_branch = use_lstm_branch
        self.lstm_layers = max(1, int(lstm_layers))
        self.lstm_dropout = float(lstm_dropout)
        self.refit_on_all_pretarget = refit_on_all_pretarget
        self.use_online_adaptation = use_online_adaptation
        self.online_bias_alpha = float(online_bias_alpha)
        self.online_slot_bias_alpha = float(online_slot_bias_alpha)
        self.online_scale_alpha = float(online_scale_alpha)
        self.uncertainty_quantile = float(np.clip(uncertainty_quantile, 0.5, 0.99))
        self.uncertainty_slot_blend = float(np.clip(uncertainty_slot_blend, 0.0, 1.0))
        self.online_uncertainty_alpha = float(np.clip(online_uncertainty_alpha, 0.0, 1.0))
        self.min_sigma_ratio = float(max(0.01, min_sigma_ratio))
        self.use_log1p = use_log1p
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.dataset = SpatioTemporalDataset(data_dir=data_dir, time_interval=time_interval)
        self.model = None
        self.grid_size = None
        self.region_cell_index = None
        self.target_tensor = None
        self.target_slots = None
        self.target_slot_to_idx = None
        self.history_start_hour = None
        self.end_hour = None
        self.num_time_slots = None
        self.region_scale_factors = {rid: 1.0 for rid in centers.keys()}
        self.sorted_region_ids = sorted(centers.keys())
        self.region_slot_means = {rid: {} for rid in centers.keys()}
        self.region_linear_models = {
            rid: np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            for rid in centers.keys()
        }
        self.region_uncertainty_stats = {
            rid: {'sigma': 1.0, 'q_resid': 1.0, 'under_rate': 0.0}
            for rid in centers.keys()
        }
        self.region_slot_uncertainty = {rid: {} for rid in centers.keys()}
        self.region_masks_t = None
        self._cuda_fallback_triggered = False
        self.reset_online_state()

    @staticmethod
    def _clone_state(model):
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    @staticmethod
    def _is_cuda_runtime_error(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return "cuda" in message or "cublas" in message or "cudnn" in message

    def _fallback_to_cpu(self, reason: str) -> None:
        if self.device.type == "cpu":
            return
        if not self._cuda_fallback_triggered:
            print(f"   [MCTGNet Dispatch] CUDA runtime failed, falling back to CPU. Reason: {reason}")
            self._cuda_fallback_triggered = True
        self.device = torch.device("cpu")
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()
        if self.region_masks_t is not None:
            self.region_masks_t = self.region_masks_t.to(self.device)
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _transform(self, array):
        if self.use_log1p:
            return np.log1p(np.clip(array, 0.0, None))
        return array.astype(np.float32, copy=False)

    def _inverse(self, array):
        if self.use_log1p:
            return np.expm1(array)
        return array

    def _weighted_mse(self, pred, target, raw_target):
        weights = 1.0 + self.hotspot_alpha * torch.log1p(torch.clamp(raw_target, min=0.0))
        return ((pred - target) ** 2 * weights).mean()

    def _build_region_masks(self) -> None:
        masks = np.zeros((len(self.sorted_region_ids), self.grid_size[0], self.grid_size[1]), dtype=np.float32)
        for rid_idx, rid in enumerate(self.sorted_region_ids):
            for y_idx, x_idx in self.region_cell_index.get(rid, []):
                masks[rid_idx, y_idx, x_idx] = 1.0
        self.region_masks_t = torch.FloatTensor(masks).to(self.device)

    def _aggregate_grid_tensor_to_regions(self, grid_tensor: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bhw,khw->bk', grid_tensor, self.region_masks_t)

    def _center_weighted_mse(self, pred_grid, target_region_raw):
        pred_grid_raw = torch.expm1(pred_grid) if self.use_log1p else pred_grid
        pred_grid_raw = torch.clamp(pred_grid_raw, min=0.0)
        pred_region_raw = self._aggregate_grid_tensor_to_regions(pred_grid_raw)
        weights = 1.0 + self.center_hotspot_alpha * torch.log1p(torch.clamp(target_region_raw, min=0.0))
        residual = pred_region_raw - target_region_raw
        under_gap = torch.clamp(target_region_raw - pred_region_raw, min=0.0)
        relative_under_gap = under_gap / torch.clamp(target_region_raw + 1.0, min=1.0)
        under_weights = 1.0 + self.center_underpredict_alpha * torch.pow(
            relative_under_gap,
            self.center_underpredict_power
        )
        return ((residual ** 2) * weights * under_weights).mean()

    def _build_model(self):
        return MCTGNet(
            seq_len=self.seq_len,
            grid_size=self.grid_size,
            hidden_dim=64,
            num_centers=len(self.centers),
            num_time_slots=self.num_time_slots,
            num_weekdays=7,
            use_lstm_branch=self.use_lstm_branch,
            lstm_layers=self.lstm_layers,
            lstm_dropout=self.lstm_dropout,
        ).to(self.device)

    def _train_fixed_epochs(
            self,
            X_t,
            Y_t,
            Y_raw_t,
            Y_region_raw_t,
            X_periodic_t,
            X_weekly_t,
            slot_ids_t,
            weekday_ids_t,
            epochs: int,
    ) -> None:
        if epochs <= 0:
            return

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        for _ in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            pred = self.model(
                X_t,
                periodic_x=X_periodic_t,
                weekly_x=X_weekly_t,
                slot_ids=slot_ids_t,
                weekday_ids=weekday_ids_t
            )
            grid_loss = self._weighted_mse(pred, Y_t, Y_raw_t)
            center_loss = self._center_weighted_mse(pred, Y_region_raw_t)
            loss = grid_loss + self.center_loss_weight * center_loss
            loss.backward()
            optimizer.step()

        self.model.eval()

    def _load_dates(self, dates: List[str]) -> pd.DataFrame:
        df_list = []
        for date_str in dates:
            file_path = os.path.join(self.data_dir, f'tasks_{date_str}.csv')
            if not os.path.exists(file_path):
                continue
            df_list.append(pd.read_csv(file_path))

        if not df_list:
            raise FileNotFoundError(f"No task files found for dates: {dates}")

        return pd.concat(df_list, ignore_index=True)

    @staticmethod
    def _filter_hours(df: pd.DataFrame, start_hour: int, end_hour: int) -> pd.DataFrame:
        filtered = df.copy()
        filtered['first_time'] = pd.to_datetime(filtered['first_time'])
        filtered['hour'] = filtered['first_time'].dt.hour
        filtered = filtered[(filtered['hour'] >= start_hour) & (filtered['hour'] < end_hour)].reset_index(drop=True)
        return filtered

    def _build_region_cell_index(self) -> None:
        lon_centers = 0.5 * (self.dataset.lon_bins[:-1] + self.dataset.lon_bins[1:])
        lat_centers = 0.5 * (self.dataset.lat_bins[:-1] + self.dataset.lat_bins[1:])
        tree = KDTree(self.coords)

        region_cell_index = {rid: [] for rid in self.centers.keys()}
        for y_idx, lat in enumerate(lat_centers):
            for x_idx, lon in enumerate(lon_centers):
                _, nearest_idx = tree.query([[lon, lat]])
                node = self.nodes[nearest_idx[0]]
                region_id = self.partition.get(node)
                if region_id in region_cell_index:
                    region_cell_index[region_id].append((y_idx, x_idx))

        self.region_cell_index = region_cell_index

    def _aggregate_grid_to_regions(self, grid: np.ndarray) -> Dict[int, float]:
        region_totals = {}
        for rid, cell_list in self.region_cell_index.items():
            total = 0.0
            for y_idx, x_idx in cell_list:
                total += float(grid[y_idx, x_idx])
            region_totals[rid] = total
        return region_totals

    def _compute_region_slot_means(self, region_targets: np.ndarray, slot_ids: np.ndarray) -> None:
        self.region_slot_means = {rid: {} for rid in self.centers.keys()}
        for rid_idx, rid in enumerate(sorted(self.centers.keys())):
            for slot_id in range(self.num_time_slots):
                mask = slot_ids == slot_id
                if np.any(mask):
                    self.region_slot_means[rid][slot_id] = float(np.mean(region_targets[mask, rid_idx]))
                else:
                    self.region_slot_means[rid][slot_id] = float(np.mean(region_targets[:, rid_idx])) if len(region_targets) > 0 else 0.0

    def reset_online_state(self) -> None:
        self.online_region_bias = {rid: 0.0 for rid in self.centers.keys()}
        self.online_region_scale = {rid: 1.0 for rid in self.centers.keys()}
        self.online_slot_bias = {rid: {} for rid in self.centers.keys()}
        self.online_weekday_slot_bias = {rid: {} for rid in self.centers.keys()}
        self.online_region_abs_error = {rid: 0.0 for rid in self.centers.keys()}
        self.online_region_sq_error = {rid: 0.0 for rid in self.centers.keys()}
        self.online_region_under_rate = {rid: 0.0 for rid in self.centers.keys()}
        self.online_slot_abs_error = {rid: {} for rid in self.centers.keys()}
        self.online_slot_sq_error = {rid: {} for rid in self.centers.keys()}
        self.online_slot_under_rate = {rid: {} for rid in self.centers.keys()}

    def _slot_id_for_timestamp(self, slot_timestamp: pd.Timestamp) -> int:
        slot_id = ((slot_timestamp.hour * 60 + slot_timestamp.minute) - self.history_start_hour * 60) // self.time_interval
        return int(np.clip(slot_id, 0, self.num_time_slots - 1))

    def _fit_region_linear_models(
            self,
            pred_region_scaled: np.ndarray,
            actual_region: np.ndarray,
            periodic_region: np.ndarray,
            weekly_region: np.ndarray,
            slot_ids: np.ndarray
    ) -> None:
        sorted_region_ids = sorted(self.centers.keys())
        for rid_idx, rid in enumerate(sorted_region_ids):
            slot_feature = np.array(
                [self.region_slot_means[rid].get(int(slot_id), 0.0) for slot_id in slot_ids],
                dtype=np.float32
            )
            features = np.column_stack([
                pred_region_scaled[:, rid_idx],
                slot_feature,
                periodic_region[:, rid_idx],
                weekly_region[:, rid_idx],
                np.ones(len(slot_ids), dtype=np.float32)
            ])
            target = actual_region[:, rid_idx].astype(np.float32)
            if len(target) == 0:
                continue

            coef, *_ = np.linalg.lstsq(features, target, rcond=None)
            coef = coef.astype(np.float32)
            coef[:3] = np.clip(coef[:3], 0.0, 3.0)
            coef[3] = float(np.clip(coef[3], -100.0, 100.0))
            self.region_linear_models[rid] = coef

    def _fit_region_uncertainty_models(
            self,
            pred_region_scaled: np.ndarray,
            actual_region: np.ndarray,
            periodic_region: np.ndarray,
            weekly_region: np.ndarray,
            slot_ids: np.ndarray
    ) -> None:
        sorted_region_ids = sorted(self.centers.keys())
        self.region_uncertainty_stats = {
            rid: {'sigma': 1.0, 'q_resid': 1.0, 'under_rate': 0.0}
            for rid in sorted_region_ids
        }
        self.region_slot_uncertainty = {rid: {} for rid in sorted_region_ids}

        for rid_idx, rid in enumerate(sorted_region_ids):
            slot_feature = np.array(
                [self.region_slot_means[rid].get(int(slot_id), 0.0) for slot_id in slot_ids],
                dtype=np.float32
            )
            coef = self.region_linear_models.get(rid)
            if coef is None:
                calibrated = pred_region_scaled[:, rid_idx]
            else:
                calibrated = (
                    coef[0] * pred_region_scaled[:, rid_idx]
                    + coef[1] * slot_feature
                    + coef[2] * periodic_region[:, rid_idx]
                    + coef[3] * weekly_region[:, rid_idx]
                    + coef[4]
                )
            calibrated = np.clip(calibrated.astype(np.float32), 0.0, None)
            residual = actual_region[:, rid_idx].astype(np.float32) - calibrated
            positive_residual = np.clip(residual, 0.0, None)
            sigma = float(np.std(residual)) if len(residual) > 1 else 0.0
            sigma = max(sigma, self.min_sigma_ratio * max(float(np.mean(actual_region[:, rid_idx])), 1.0))
            q_resid = float(np.quantile(positive_residual, self.uncertainty_quantile)) if len(positive_residual) > 0 else sigma
            under_rate = float(np.mean((residual > 0).astype(np.float32))) if len(residual) > 0 else 0.0
            self.region_uncertainty_stats[rid] = {
                'sigma': sigma,
                'q_resid': max(q_resid, sigma),
                'under_rate': under_rate,
            }

            unique_slots = sorted(set(int(slot_id) for slot_id in slot_ids))
            for slot_id in unique_slots:
                mask = slot_ids == slot_id
                if not np.any(mask):
                    continue
                slot_residual = residual[mask]
                slot_positive = positive_residual[mask]
                slot_sigma = float(np.std(slot_residual)) if len(slot_residual) > 1 else sigma
                slot_sigma = max(slot_sigma, self.min_sigma_ratio * max(float(np.mean(actual_region[mask, rid_idx])), 1.0))
                slot_q_resid = float(np.quantile(slot_positive, self.uncertainty_quantile)) if len(slot_positive) > 0 else slot_sigma
                slot_under_rate = float(np.mean((slot_residual > 0).astype(np.float32))) if len(slot_residual) > 0 else under_rate
                self.region_slot_uncertainty[rid][int(slot_id)] = {
                    'sigma': slot_sigma,
                    'q_resid': max(slot_q_resid, slot_sigma),
                    'under_rate': slot_under_rate,
                }

    def _build_periodic_features(self, demand_tensor, all_slots):
        slot_to_idx = {pd.Timestamp(ts): idx for idx, ts in enumerate(all_slots)}
        periodic_inputs = []
        weekly_inputs = []
        slot_ids = []
        weekday_ids = []

        for i in range(len(demand_tensor) - self.seq_len - self.pre_len + 1):
            target_ts = pd.Timestamp(all_slots[i + self.seq_len])
            prev_day_ts = target_ts - pd.Timedelta(days=1)
            prev_week_ts = target_ts - pd.Timedelta(days=7)
            if prev_day_ts in slot_to_idx:
                periodic_frame = demand_tensor[slot_to_idx[prev_day_ts]]
            else:
                periodic_frame = np.zeros_like(demand_tensor[0])
            if prev_week_ts in slot_to_idx:
                weekly_frame = demand_tensor[slot_to_idx[prev_week_ts]]
            else:
                weekly_frame = np.zeros_like(demand_tensor[0])

            slot_id = ((target_ts.hour * 60 + target_ts.minute) - self.history_start_hour * 60) // self.time_interval
            slot_id = int(np.clip(slot_id, 0, self.num_time_slots - 1))

            periodic_inputs.append(periodic_frame[None, ...])
            weekly_inputs.append(weekly_frame[None, ...])
            slot_ids.append(slot_id)
            weekday_ids.append(int(target_ts.dayofweek))

        return (
            np.array(periodic_inputs),
            np.array(weekly_inputs),
            np.array(slot_ids, dtype=np.int64),
            np.array(weekday_ids, dtype=np.int64)
        )

    def fit(
            self,
            train_dates: List[str],
            val_dates: List[str],
            target_date: str,
            history_start_hour: int,
            end_hour: int
    ) -> None:
        self.history_start_hour = history_start_hour
        self.end_hour = end_hour
        self.num_time_slots = max(1, ((end_hour - history_start_hour) * 60) // self.time_interval)

        train_df = self._filter_hours(self._load_dates(train_dates), history_start_hour, end_hour)
        val_df = self._filter_hours(self._load_dates(val_dates), history_start_hour, end_hour)
        target_df = self._filter_hours(self._load_dates([target_date]), history_start_hour, end_hour)

        demand_tensor_train, slots_train = self.dataset.load_and_gridify_from_dataframe(
            train_df, start_hour=history_start_hour, end_hour=end_hour
        )
        demand_tensor_val, slots_val = self.dataset.load_and_gridify_from_dataframe(
            val_df, start_hour=history_start_hour, end_hour=end_hour
        )
        self.target_tensor, self.target_slots = self.dataset.load_and_gridify_from_dataframe(
            target_df, start_hour=history_start_hour, end_hour=end_hour
        )
        self.target_slot_to_idx = {pd.Timestamp(ts): idx for idx, ts in enumerate(self.target_slots)}

        X_train_raw, Y_train_raw = self.dataset.create_seq_data_single_tensor(
            demand_tensor_train, seq_len=self.seq_len, pre_len=self.pre_len
        )
        X_val_raw, Y_val_raw = self.dataset.create_seq_data_single_tensor(
            demand_tensor_val, seq_len=self.seq_len, pre_len=self.pre_len
        )
        X_train_periodic_raw, X_train_weekly_raw, train_slot_ids, train_weekday_ids = self._build_periodic_features(
            demand_tensor_train, slots_train
        )
        X_val_periodic_raw, X_val_weekly_raw, val_slot_ids, val_weekday_ids = self._build_periodic_features(
            demand_tensor_val, slots_val
        )

        if len(X_train_raw) == 0 or len(X_val_raw) == 0:
            raise ValueError("Not enough samples to train MCTGNet dispatch predictor.")

        self.grid_size = (X_train_raw.shape[2], X_train_raw.shape[3])
        self._build_region_cell_index()
        self._build_region_masks()

        sorted_region_ids = self.sorted_region_ids
        train_region_targets = np.array(
            [
                [self._aggregate_grid_to_regions(grid)[rid] for rid in sorted_region_ids]
                for grid in Y_train_raw[:, 0]
            ],
            dtype=np.float32
        )
        val_region_targets = np.array(
            [
                [self._aggregate_grid_to_regions(grid)[rid] for rid in sorted_region_ids]
                for grid in Y_val_raw[:, 0]
            ],
            dtype=np.float32
        )
        val_region_periodic = np.array(
            [
                [self._aggregate_grid_to_regions(grid[0])[rid] for rid in sorted_region_ids]
                for grid in X_val_periodic_raw
            ],
            dtype=np.float32
        )
        val_region_weekly = np.array(
            [
                [self._aggregate_grid_to_regions(grid[0])[rid] for rid in sorted_region_ids]
                for grid in X_val_weekly_raw
            ],
            dtype=np.float32
        )
        self._compute_region_slot_means(train_region_targets, train_slot_ids)

        X_train = self._transform(X_train_raw)
        Y_train = self._transform(Y_train_raw)
        X_val = self._transform(X_val_raw)
        Y_val = self._transform(Y_val_raw)
        X_train_periodic = self._transform(X_train_periodic_raw)
        X_val_periodic = self._transform(X_val_periodic_raw)
        X_train_weekly = self._transform(X_train_weekly_raw)
        X_val_weekly = self._transform(X_val_weekly_raw)

        self.model = self._build_model()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=max(10, self.patience // 4)
        )

        X_tr_t = torch.FloatTensor(X_train).to(self.device)
        Y_tr_t = torch.FloatTensor(Y_train[:, 0]).to(self.device)
        Y_tr_raw_t = torch.FloatTensor(Y_train_raw[:, 0]).to(self.device)
        Y_tr_region_raw_t = torch.FloatTensor(train_region_targets).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        Y_val_t = torch.FloatTensor(Y_val[:, 0]).to(self.device)
        Y_val_raw_t = torch.FloatTensor(Y_val_raw[:, 0]).to(self.device)
        Y_val_region_raw_t = torch.FloatTensor(val_region_targets).to(self.device)
        X_tr_periodic_t = torch.FloatTensor(X_train_periodic).to(self.device)
        X_val_periodic_t = torch.FloatTensor(X_val_periodic).to(self.device)
        X_tr_weekly_t = torch.FloatTensor(X_train_weekly).to(self.device)
        X_val_weekly_t = torch.FloatTensor(X_val_weekly).to(self.device)
        train_slot_ids_t = torch.LongTensor(train_slot_ids).to(self.device)
        val_slot_ids_t = torch.LongTensor(val_slot_ids).to(self.device)
        train_weekday_ids_t = torch.LongTensor(train_weekday_ids).to(self.device)
        val_weekday_ids_t = torch.LongTensor(val_weekday_ids).to(self.device)

        best_val_loss = float('inf')
        best_state = self._clone_state(self.model)
        patience_counter = 0
        best_epoch = 0

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            pred_train = self.model(
                X_tr_t,
                periodic_x=X_tr_periodic_t,
                weekly_x=X_tr_weekly_t,
                slot_ids=train_slot_ids_t,
                weekday_ids=train_weekday_ids_t
            )
            train_grid_loss = self._weighted_mse(pred_train, Y_tr_t, Y_tr_raw_t)
            train_center_loss = self._center_weighted_mse(pred_train, Y_tr_region_raw_t)
            train_loss = train_grid_loss + self.center_loss_weight * train_center_loss
            train_loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                pred_val = self.model(
                    X_val_t,
                    periodic_x=X_val_periodic_t,
                    weekly_x=X_val_weekly_t,
                    slot_ids=val_slot_ids_t,
                    weekday_ids=val_weekday_ids_t
                )
                val_grid_loss = self._weighted_mse(pred_val, Y_val_t, Y_val_raw_t)
                val_center_loss = self._center_weighted_mse(pred_val, Y_val_region_raw_t)
                val_loss = (val_grid_loss + self.center_loss_weight * val_center_loss).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self._clone_state(self.model)
                patience_counter = 0
                best_epoch = epoch
            else:
                patience_counter += 1

            if epoch % self.log_interval == 0 or epoch == 1:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"   [MCTGNet Dispatch] Epoch [{epoch:04d}/{self.max_epochs}], "
                    f"Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss:.4f}, "
                    f"Train Center: {train_center_loss.item():.2f}, Val Center: {val_center_loss.item():.2f}, "
                    f"LR: {current_lr:.6f}"
                )

            if patience_counter >= self.patience:
                print(
                    f"   [MCTGNet Dispatch] Early stop at epoch {epoch}, "
                    f"best epoch = {best_epoch}, best val loss = {best_val_loss:.4f}"
                )
                break

        self.model.load_state_dict(best_state)
        self.model.to(self.device)
        self.model.eval()
        print(f"   [MCTGNet Dispatch] Best Epoch: {best_epoch}, Best Val Loss: {best_val_loss:.4f}")

        if self.refit_on_all_pretarget and best_epoch > 0:
            print(f"   [MCTGNet Dispatch] Refit on train+val for {best_epoch} epochs...")
            X_refit_raw = np.concatenate([X_train_raw, X_val_raw], axis=0)
            Y_refit_raw = np.concatenate([Y_train_raw, Y_val_raw], axis=0)
            X_refit_periodic_raw = np.concatenate([X_train_periodic_raw, X_val_periodic_raw], axis=0)
            X_refit_weekly_raw = np.concatenate([X_train_weekly_raw, X_val_weekly_raw], axis=0)
            refit_slot_ids = np.concatenate([train_slot_ids, val_slot_ids], axis=0)
            refit_weekday_ids = np.concatenate([train_weekday_ids, val_weekday_ids], axis=0)
            refit_region_targets = np.concatenate([train_region_targets, val_region_targets], axis=0)

            self.model = self._build_model()
            self._train_fixed_epochs(
                X_t=torch.FloatTensor(self._transform(X_refit_raw)).to(self.device),
                Y_t=torch.FloatTensor(self._transform(Y_refit_raw)[:, 0]).to(self.device),
                Y_raw_t=torch.FloatTensor(Y_refit_raw[:, 0]).to(self.device),
                Y_region_raw_t=torch.FloatTensor(refit_region_targets).to(self.device),
                X_periodic_t=torch.FloatTensor(self._transform(X_refit_periodic_raw)).to(self.device),
                X_weekly_t=torch.FloatTensor(self._transform(X_refit_weekly_raw)).to(self.device),
                slot_ids_t=torch.LongTensor(refit_slot_ids).to(self.device),
                weekday_ids_t=torch.LongTensor(refit_weekday_ids).to(self.device),
                epochs=best_epoch,
            )

        pred_region_totals = {rid: 0.0 for rid in self.centers.keys()}
        actual_region_totals = {rid: 0.0 for rid in self.centers.keys()}

        with torch.no_grad():
            pred_val_np = self.model(
                X_val_t,
                periodic_x=X_val_periodic_t,
                weekly_x=X_val_weekly_t,
                slot_ids=val_slot_ids_t,
                weekday_ids=val_weekday_ids_t
            ).detach().cpu().numpy()

        pred_val_np = self._inverse(pred_val_np)
        pred_val_np = np.clip(pred_val_np, 0.0, None)
        actual_val_np = np.clip(Y_val_raw[:, 0], 0.0, None)

        total_pred = 0.0
        total_actual = 0.0
        for pred_grid, actual_grid in zip(pred_val_np, actual_val_np):
            pred_regions = self._aggregate_grid_to_regions(pred_grid)
            actual_regions = self._aggregate_grid_to_regions(actual_grid)
            for rid in self.centers.keys():
                pred_region_totals[rid] += pred_regions[rid]
                actual_region_totals[rid] += actual_regions[rid]
                total_pred += pred_regions[rid]
                total_actual += actual_regions[rid]

        global_scale = total_actual / max(total_pred, 1e-6)
        global_scale = float(np.clip(global_scale, 0.75, 2.5))
        for rid in self.centers.keys():
            rid_pred = pred_region_totals[rid]
            if rid_pred <= 1e-6:
                region_scale = global_scale
            else:
                region_scale = actual_region_totals[rid] / rid_pred
            blended_scale = 0.5 * region_scale + 0.5 * global_scale
            self.region_scale_factors[rid] = float(np.clip(blended_scale, 0.75, 2.5))

        pred_region_scaled = np.array(
            [
                [
                    self._aggregate_grid_to_regions(pred_grid)[rid] * self.region_scale_factors.get(rid, 1.0)
                    for rid in sorted_region_ids
                ]
                for pred_grid in pred_val_np
            ],
            dtype=np.float32
        )
        self._fit_region_linear_models(
            pred_region_scaled=pred_region_scaled,
            actual_region=val_region_targets,
            periodic_region=val_region_periodic,
            weekly_region=val_region_weekly,
            slot_ids=val_slot_ids
        )
        self._fit_region_uncertainty_models(
            pred_region_scaled=pred_region_scaled,
            actual_region=val_region_targets,
            periodic_region=val_region_periodic,
            weekly_region=val_region_weekly,
            slot_ids=val_slot_ids
        )

    def _predict_region_values(
            self,
            slot_timestamp: pd.Timestamp
    ) -> Optional[Tuple[Dict[int, float], Dict[int, float], Dict[int, float], int, int]]:
        if self.model is None or self.target_tensor is None or self.target_slots is None:
            raise RuntimeError("MCTGNet dispatch predictor must be fitted before prediction.")

        target_slot = pd.Timestamp(slot_timestamp)
        target_idx = self.target_slot_to_idx.get(target_slot)
        if target_idx is None or target_idx < self.seq_len:
            return None

        window = self.target_tensor[target_idx - self.seq_len:target_idx]
        prev_day_idx = self.target_slot_to_idx.get(target_slot - pd.Timedelta(days=1))
        prev_week_idx = self.target_slot_to_idx.get(target_slot - pd.Timedelta(days=7))
        if prev_day_idx is None:
            periodic_frame = np.zeros_like(self.target_tensor[0])
        else:
            periodic_frame = self.target_tensor[prev_day_idx]
        if prev_week_idx is None:
            weekly_frame = np.zeros_like(self.target_tensor[0])
        else:
            weekly_frame = self.target_tensor[prev_week_idx]

        slot_id = self._slot_id_for_timestamp(target_slot)
        weekday_id = int(target_slot.dayofweek)

        X_t = torch.FloatTensor(self._transform(window)).unsqueeze(0).to(self.device)
        X_periodic_t = torch.FloatTensor(self._transform(periodic_frame[None, ...])).unsqueeze(0).to(self.device)
        X_weekly_t = torch.FloatTensor(self._transform(weekly_frame[None, ...])).unsqueeze(0).to(self.device)
        slot_ids_t = torch.LongTensor([slot_id]).to(self.device)
        weekday_ids_t = torch.LongTensor([weekday_id]).to(self.device)

        with torch.no_grad():
            pred_grid = self.model(
                X_t,
                periodic_x=X_periodic_t,
                weekly_x=X_weekly_t,
                slot_ids=slot_ids_t,
                weekday_ids=weekday_ids_t
            ).detach().cpu().numpy()[0]

        pred_grid = self._inverse(pred_grid)
        pred_grid = np.clip(pred_grid, 0.0, None)

        raw_region_demand = self._aggregate_grid_to_regions(pred_grid)
        periodic_region_demand = self._aggregate_grid_to_regions(periodic_frame)
        weekly_region_demand = self._aggregate_grid_to_regions(weekly_frame)
        calibrated_region_demand = {}
        for rid, total in raw_region_demand.items():
            scale = self.region_scale_factors.get(rid, 1.0)
            scaled_total = total * scale
            slot_mean = self.region_slot_means.get(rid, {}).get(slot_id, scaled_total)
            periodic_total = periodic_region_demand.get(rid, 0.0)
            weekly_total = weekly_region_demand.get(rid, 0.0)
            coef = self.region_linear_models.get(rid)
            if coef is None:
                calibrated_total = scaled_total
            else:
                calibrated_total = (
                    coef[0] * scaled_total
                    + coef[1] * slot_mean
                    + coef[2] * periodic_total
                    + coef[3] * weekly_total
                    + coef[4]
                )
            if self.use_online_adaptation:
                slot_bias = self.online_slot_bias.get(rid, {}).get(slot_id, 0.0)
                weekday_slot_bias = self.online_weekday_slot_bias.get(rid, {}).get((weekday_id, slot_id), 0.0)
                online_scale = self.online_region_scale.get(rid, 1.0)
                online_bias = self.online_region_bias.get(rid, 0.0)
                calibrated_total = calibrated_total * online_scale + online_bias + slot_bias + weekday_slot_bias
            calibrated_region_demand[rid] = float(max(0.0, calibrated_total))
        return calibrated_region_demand, raw_region_demand, periodic_region_demand, slot_id, weekday_id

    def predict_region_demand(self, slot_timestamp: pd.Timestamp) -> Optional[Dict[int, int]]:
        try:
            prediction = self._predict_region_values(slot_timestamp)
        except RuntimeError as exc:
            if self.device.type == "cuda" and self._is_cuda_runtime_error(exc):
                self._fallback_to_cpu(str(exc))
                prediction = self._predict_region_values(slot_timestamp)
            else:
                raise
        if prediction is None:
            return None

        calibrated_region_demand, _, _, _, _ = prediction
        return {
            rid: int(round(total))
            for rid, total in calibrated_region_demand.items()
        }

    def predict_region_distribution(self, slot_timestamp: pd.Timestamp) -> Optional[Dict[int, Dict[str, float]]]:
        try:
            prediction = self._predict_region_values(slot_timestamp)
        except RuntimeError as exc:
            if self.device.type == "cuda" and self._is_cuda_runtime_error(exc):
                self._fallback_to_cpu(str(exc))
                prediction = self._predict_region_values(slot_timestamp)
            else:
                raise
        if prediction is None:
            return None

        calibrated_region_demand, _, _, slot_id, _ = prediction
        distribution = {}
        for rid, mu in calibrated_region_demand.items():
            global_stats = self.region_uncertainty_stats.get(rid, {})
            slot_stats = self.region_slot_uncertainty.get(rid, {}).get(slot_id, global_stats)

            sigma = (
                self.uncertainty_slot_blend * float(slot_stats.get('sigma', global_stats.get('sigma', 1.0)))
                + (1.0 - self.uncertainty_slot_blend) * float(global_stats.get('sigma', 1.0))
            )
            q_resid = (
                self.uncertainty_slot_blend * float(slot_stats.get('q_resid', global_stats.get('q_resid', sigma)))
                + (1.0 - self.uncertainty_slot_blend) * float(global_stats.get('q_resid', sigma))
            )
            under_rate = (
                self.uncertainty_slot_blend * float(slot_stats.get('under_rate', global_stats.get('under_rate', 0.0)))
                + (1.0 - self.uncertainty_slot_blend) * float(global_stats.get('under_rate', 0.0))
            )

            online_sigma = float(np.sqrt(max(self.online_region_sq_error.get(rid, 0.0), 0.0)))
            online_slot_sigma = float(np.sqrt(max(self.online_slot_sq_error.get(rid, {}).get(slot_id, 0.0), 0.0)))
            sigma = max(
                sigma,
                0.5 * online_sigma + 0.5 * online_slot_sigma,
                self.min_sigma_ratio * max(mu, 1.0)
            )
            q_resid = max(
                q_resid,
                max(0.0, self.online_region_bias.get(rid, 0.0)) + 0.5 * self.online_region_abs_error.get(rid, 0.0),
                sigma
            )
            burst_prob = float(np.clip(
                0.5 * under_rate
                + 0.3 * self.online_region_under_rate.get(rid, 0.0)
                + 0.2 * self.online_slot_under_rate.get(rid, {}).get(slot_id, 0.0),
                0.0,
                1.0
            ))
            distribution[rid] = {
                'mu': float(mu),
                'sigma': float(sigma),
                'q90': float(mu + max(q_resid, sigma)),
                'burst_prob': burst_prob,
            }
        return distribution

    def update_online(
            self,
            slot_timestamp: pd.Timestamp,
            actual_region_demand: Dict[int, int],
            predicted_region_demand: Optional[Dict[int, int]] = None
    ) -> None:
        if not self.use_online_adaptation:
            return

        ts = pd.Timestamp(slot_timestamp)
        slot_id = self._slot_id_for_timestamp(ts)
        weekday_id = int(ts.dayofweek)

        for rid in self.sorted_region_ids:
            actual = float(actual_region_demand.get(rid, 0.0))
            pred = float(predicted_region_demand.get(rid, actual)) if predicted_region_demand is not None else actual
            error = actual - pred
            abs_error = abs(error)
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

            self.online_region_abs_error[rid] = (
                (1.0 - self.online_uncertainty_alpha) * self.online_region_abs_error.get(rid, 0.0)
                + self.online_uncertainty_alpha * abs_error
            )
            self.online_region_sq_error[rid] = (
                (1.0 - self.online_uncertainty_alpha) * self.online_region_sq_error.get(rid, 0.0)
                + self.online_uncertainty_alpha * (error ** 2)
            )
            self.online_region_under_rate[rid] = (
                (1.0 - self.online_uncertainty_alpha) * self.online_region_under_rate.get(rid, 0.0)
                + self.online_uncertainty_alpha * (1.0 if error > 0 else 0.0)
            )

            prev_slot_abs_error = self.online_slot_abs_error.setdefault(rid, {}).get(slot_id, 0.0)
            self.online_slot_abs_error[rid][slot_id] = (
                (1.0 - self.online_uncertainty_alpha) * prev_slot_abs_error
                + self.online_uncertainty_alpha * abs_error
            )
            prev_slot_sq_error = self.online_slot_sq_error.setdefault(rid, {}).get(slot_id, 0.0)
            self.online_slot_sq_error[rid][slot_id] = (
                (1.0 - self.online_uncertainty_alpha) * prev_slot_sq_error
                + self.online_uncertainty_alpha * (error ** 2)
            )
            prev_slot_under_rate = self.online_slot_under_rate.setdefault(rid, {}).get(slot_id, 0.0)
            self.online_slot_under_rate[rid][slot_id] = (
                (1.0 - self.online_uncertainty_alpha) * prev_slot_under_rate
                + self.online_uncertainty_alpha * (1.0 if error > 0 else 0.0)
            )
