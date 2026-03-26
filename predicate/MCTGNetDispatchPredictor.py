import os
from typing import Any, Dict, List, Optional

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

    @staticmethod
    def _clone_state(model):
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

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

    def _build_periodic_features(self, demand_tensor, all_slots):
        slot_to_idx = {pd.Timestamp(ts): idx for idx, ts in enumerate(all_slots)}
        periodic_inputs = []
        slot_ids = []

        for i in range(len(demand_tensor) - self.seq_len - self.pre_len + 1):
            target_ts = pd.Timestamp(all_slots[i + self.seq_len])
            prev_day_ts = target_ts - pd.Timedelta(days=1)
            if prev_day_ts in slot_to_idx:
                periodic_frame = demand_tensor[slot_to_idx[prev_day_ts]]
            else:
                periodic_frame = np.zeros_like(demand_tensor[0])

            slot_id = ((target_ts.hour * 60 + target_ts.minute) - self.history_start_hour * 60) // self.time_interval
            slot_id = int(np.clip(slot_id, 0, self.num_time_slots - 1))

            periodic_inputs.append(periodic_frame[None, ...])
            slot_ids.append(slot_id)

        return np.array(periodic_inputs), np.array(slot_ids, dtype=np.int64)

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
        X_train_periodic_raw, train_slot_ids = self._build_periodic_features(demand_tensor_train, slots_train)
        X_val_periodic_raw, val_slot_ids = self._build_periodic_features(demand_tensor_val, slots_val)

        if len(X_train_raw) == 0 or len(X_val_raw) == 0:
            raise ValueError("Not enough samples to train MCTGNet dispatch predictor.")

        self.grid_size = (X_train_raw.shape[2], X_train_raw.shape[3])
        self._build_region_cell_index()

        X_train = self._transform(X_train_raw)
        Y_train = self._transform(Y_train_raw)
        X_val = self._transform(X_val_raw)
        Y_val = self._transform(Y_val_raw)
        X_train_periodic = self._transform(X_train_periodic_raw)
        X_val_periodic = self._transform(X_val_periodic_raw)

        self.model = MCTGNet(
            seq_len=self.seq_len,
            grid_size=self.grid_size,
            hidden_dim=64,
            num_centers=len(self.centers),
            num_time_slots=self.num_time_slots
        ).to(self.device)

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
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        Y_val_t = torch.FloatTensor(Y_val[:, 0]).to(self.device)
        Y_val_raw_t = torch.FloatTensor(Y_val_raw[:, 0]).to(self.device)
        X_tr_periodic_t = torch.FloatTensor(X_train_periodic).to(self.device)
        X_val_periodic_t = torch.FloatTensor(X_val_periodic).to(self.device)
        train_slot_ids_t = torch.LongTensor(train_slot_ids).to(self.device)
        val_slot_ids_t = torch.LongTensor(val_slot_ids).to(self.device)

        best_val_loss = float('inf')
        best_state = self._clone_state(self.model)
        patience_counter = 0
        best_epoch = 0

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            pred_train = self.model(X_tr_t, X_tr_periodic_t, train_slot_ids_t)
            train_loss = self._weighted_mse(pred_train, Y_tr_t, Y_tr_raw_t)
            train_loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                pred_val = self.model(X_val_t, X_val_periodic_t, val_slot_ids_t)
                val_loss = self._weighted_mse(pred_val, Y_val_t, Y_val_raw_t).item()

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
                    f"Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}"
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

        pred_region_totals = {rid: 0.0 for rid in self.centers.keys()}
        actual_region_totals = {rid: 0.0 for rid in self.centers.keys()}

        with torch.no_grad():
            pred_val_np = self.model(X_val_t, X_val_periodic_t, val_slot_ids_t).detach().cpu().numpy()

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

    def predict_region_demand(self, slot_timestamp: pd.Timestamp) -> Optional[Dict[int, int]]:
        if self.model is None or self.target_tensor is None or self.target_slots is None:
            raise RuntimeError("MCTGNet dispatch predictor must be fitted before prediction.")

        target_slot = pd.Timestamp(slot_timestamp)
        target_idx = self.target_slot_to_idx.get(target_slot)
        if target_idx is None or target_idx < self.seq_len:
            return None

        window = self.target_tensor[target_idx - self.seq_len:target_idx]
        prev_day_idx = self.target_slot_to_idx.get(target_slot - pd.Timedelta(days=1))
        if prev_day_idx is None:
            periodic_frame = np.zeros_like(self.target_tensor[0])
        else:
            periodic_frame = self.target_tensor[prev_day_idx]

        slot_id = ((target_slot.hour * 60 + target_slot.minute) - self.history_start_hour * 60) // self.time_interval
        slot_id = int(np.clip(slot_id, 0, self.num_time_slots - 1))

        X_t = torch.FloatTensor(self._transform(window)).unsqueeze(0).to(self.device)
        X_periodic_t = torch.FloatTensor(self._transform(periodic_frame[None, ...])).unsqueeze(0).to(self.device)
        slot_ids_t = torch.LongTensor([slot_id]).to(self.device)

        with torch.no_grad():
            pred_grid = self.model(X_t, X_periodic_t, slot_ids_t).detach().cpu().numpy()[0]

        pred_grid = self._inverse(pred_grid)
        pred_grid = np.clip(pred_grid, 0.0, None)

        raw_region_demand = self._aggregate_grid_to_regions(pred_grid)
        region_demand = {}
        for rid, total in raw_region_demand.items():
            scale = self.region_scale_factors.get(rid, 1.0)
            region_demand[rid] = int(round(total * scale))
        return region_demand
