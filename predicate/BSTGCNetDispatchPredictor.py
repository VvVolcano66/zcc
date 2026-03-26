import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial import KDTree

from predicate.data_pipeline import SpatioTemporalDataset
from predicate.model.BSTGCNet import BSTGCNet


class BSTGCNetDispatchPredictor:
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
            lr: float = 0.001,
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
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.dataset = SpatioTemporalDataset(data_dir=data_dir, time_interval=time_interval)
        self.model = None
        self.grid_size = None
        self.num_nodes = None
        self.G_s = None
        self.G_n = None
        self.G_d = None
        self.target_tensor = None
        self.target_slots = None
        self.region_cell_index = None

    @staticmethod
    def _clone_state(model):
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def _load_dates(self, dates: List[str]) -> pd.DataFrame:
        df_list = []
        for date_str in dates:
            file_path = os.path.join(self.data_dir, f'tasks_{date_str}.csv')
            if not os.path.exists(file_path):
                continue
            df = pd.read_csv(file_path)
            df_list.append(df)

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

    def _build_graphs(self, X_train: np.ndarray) -> None:
        self.grid_size = (X_train.shape[2], X_train.shape[3])
        self.num_nodes = self.grid_size[0] * self.grid_size[1]

        historical_demand = X_train.reshape(-1, self.num_nodes).T
        hist_tensor = torch.FloatTensor(historical_demand)
        norm = torch.norm(hist_tensor, p=2, dim=1, keepdim=True)
        norm_tensor = hist_tensor / (norm + 1e-8)
        self.G_s = torch.mm(norm_tensor, norm_tensor.t()).to(self.device)

        adj = torch.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                idx = i * self.grid_size[1] + j
                adj[idx, idx] = 1.0
                if i > 0:
                    adj[idx, (i - 1) * self.grid_size[1] + j] = 1.0
                if i < self.grid_size[0] - 1:
                    adj[idx, (i + 1) * self.grid_size[1] + j] = 1.0
                if j > 0:
                    adj[idx, i * self.grid_size[1] + (j - 1)] = 1.0
                if j < self.grid_size[1] - 1:
                    adj[idx, i * self.grid_size[1] + (j + 1)] = 1.0
        self.G_n = adj.to(self.device)

        G_d = torch.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                idx1 = i * self.grid_size[1] + j
                for x in range(self.grid_size[0]):
                    for y in range(self.grid_size[1]):
                        idx2 = x * self.grid_size[1] + y
                        dist = np.sqrt((i - x) ** 2 + (j - y) ** 2)
                        G_d[idx1, idx2] = 1.0 / (dist + 1.0)
        self.G_d = G_d.to(self.device)

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

    def fit(
            self,
            train_dates: List[str],
            val_dates: List[str],
            target_date: str,
            history_start_hour: int,
            end_hour: int
    ) -> None:
        train_df = self._filter_hours(self._load_dates(train_dates), history_start_hour, end_hour)
        val_df = self._filter_hours(self._load_dates(val_dates), history_start_hour, end_hour)
        target_df = self._filter_hours(self._load_dates([target_date]), history_start_hour, end_hour)

        demand_tensor_train, _ = self.dataset.load_and_gridify_from_dataframe(
            train_df, start_hour=history_start_hour, end_hour=end_hour
        )
        demand_tensor_val, _ = self.dataset.load_and_gridify_from_dataframe(
            val_df, start_hour=history_start_hour, end_hour=end_hour
        )
        self.target_tensor, self.target_slots = self.dataset.load_and_gridify_from_dataframe(
            target_df, start_hour=history_start_hour, end_hour=end_hour
        )

        X_train, Y_train = self.dataset.create_seq_data_single_tensor(
            demand_tensor_train, seq_len=self.seq_len, pre_len=self.pre_len
        )
        X_val, Y_val = self.dataset.create_seq_data_single_tensor(
            demand_tensor_val, seq_len=self.seq_len, pre_len=self.pre_len
        )

        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError("Not enough samples to train BSTGCNet dispatch predictor.")

        self._build_graphs(X_train)
        self._build_region_cell_index()

        self.model = BSTGCNet(
            num_nodes=self.num_nodes,
            in_features=1,
            hidden_dim=64,
            seq_len=self.seq_len,
            pre_len=self.pre_len
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        X_tr_t = torch.FloatTensor(X_train).to(self.device)
        Y_tr_t = torch.FloatTensor(Y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        Y_val_t = torch.FloatTensor(Y_val).to(self.device)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = self._clone_state(self.model)

        for epoch in range(self.max_epochs):
            self.model.train()
            optimizer.zero_grad()

            X_tr_bst = X_tr_t.view(X_tr_t.shape[0], X_tr_t.shape[1], self.num_nodes, 1)
            outputs_bst = self.model(X_tr_bst, self.G_s, self.G_n, self.G_d)
            target_bst = Y_tr_t.view(Y_tr_t.shape[0], self.pre_len, self.num_nodes)
            loss_bst = criterion(outputs_bst, target_bst)
            loss_bst.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                X_val_bst = X_val_t.view(X_val_t.shape[0], X_val_t.shape[1], self.num_nodes, 1)
                preds_val_bst = self.model(X_val_bst, self.G_s, self.G_n, self.G_d)
                target_val_bst = Y_val_t.view(Y_val_t.shape[0], self.pre_len, self.num_nodes)
                val_loss = criterion(preds_val_bst, target_val_bst).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self._clone_state(self.model)
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        self.model.load_state_dict(best_state)
        self.model.to(self.device)
        self.model.eval()

    def predict_region_demand(self, slot_timestamp: pd.Timestamp) -> Optional[Dict[int, int]]:
        if self.model is None or self.target_tensor is None or self.target_slots is None:
            raise RuntimeError("BSTGCNet dispatch predictor must be fitted before prediction.")

        target_slot = pd.Timestamp(slot_timestamp)
        target_indices = np.where(self.target_slots == target_slot)[0]
        if len(target_indices) == 0:
            return None

        target_idx = int(target_indices[0])
        if target_idx < self.seq_len:
            return None

        window = self.target_tensor[target_idx - self.seq_len:target_idx]
        X_t = torch.FloatTensor(window).unsqueeze(0).to(self.device)
        X_t = X_t.view(1, self.seq_len, self.num_nodes, 1)

        with torch.no_grad():
            preds = self.model(X_t, self.G_s, self.G_n, self.G_d)

        pred_grid = preds.view(self.pre_len, self.grid_size[0], self.grid_size[1])[0].detach().cpu().numpy()
        pred_grid = np.clip(pred_grid, 0.0, None)

        region_demand = {}
        for rid, cell_list in self.region_cell_index.items():
            total = 0.0
            for y_idx, x_idx in cell_list:
                total += float(pred_grid[y_idx, x_idx])
            region_demand[rid] = int(round(total))
        return region_demand
