import math

import pandas as pd
import numpy as np
import os

import config


class SpatioTemporalDataset:
    def __init__(self, data_dir=r'D:\biyelunwen\data\task', time_interval=30):
        self.data_dir = data_dir
        self.time_interval = time_interval  # 分钟

        # 1. 自动读取 config 中的网格划分数量 (10x10)
        self.grid_size = (config.NUM_ZONES, config.NUM_ZONES)

        # 2. 读取 config 中的中心点和下载半径
        center_lat, center_lon = config.CHENGDU_CENTER
        dist_m = config.DOWNLOAD_DIST

        # 3. 将米转换为经纬度偏移量 (近似计算)
        # 地球上纬度 1 度大约是 111320 米
        lat_delta = dist_m / 111320.0

        # 经度 1 度的实际距离受纬度影响，需要乘以 cos(纬度)
        lon_delta = dist_m / (111320.0 * math.cos(math.radians(center_lat)))

        # 4. 根据中心点和偏移量，动态生成边界
        self.lat_bins = np.linspace(center_lat - lat_delta, center_lat + lat_delta, self.grid_size[0] + 1)
        self.lon_bins = np.linspace(center_lon - lon_delta, center_lon + lon_delta, self.grid_size[1] + 1)

        print(
            f"动态网格生成完成: 经度范围 [{self.lon_bins[0]:.4f}, {self.lon_bins[-1]:.4f}], 纬度范围 [{self.lat_bins[0]:.4f}, {self.lat_bins[-1]:.4f}]")

    def load_and_gridify(self):
        """将所有 CSV 加载并映射到时空网格 (Time, H, W)"""
        files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.csv')])
        df_list = []
        for f in files:
            df = pd.read_csv(os.path.join(self.data_dir, f))
            df_list.append(df)

        all_data = pd.concat(df_list, ignore_index=True)
        all_data['first_time'] = pd.to_datetime(all_data['first_time'])

        # 按时间片聚合 (例如每30分钟)
        all_data['time_slot'] = all_data['first_time'].dt.floor(f'{self.time_interval}T')
        all_data['x_idx'] = np.digitize(all_data['first_lon'], self.lon_bins) - 1
        all_data['y_idx'] = np.digitize(all_data['first_lat'], self.lat_bins) - 1

        # 过滤掉边界外的数据
        valid = all_data[(all_data['x_idx'] >= 0) & (all_data['x_idx'] < self.grid_size[1]) &
                         (all_data['y_idx'] >= 0) & (all_data['y_idx'] < self.grid_size[0])]

        # 生成完整的时间序列
        min_time = valid['time_slot'].min()
        max_time = valid['time_slot'].max()
        all_slots = pd.date_range(min_time, max_time, freq=f'{self.time_interval}T')

        tensor_shape = (len(all_slots), self.grid_size[0], self.grid_size[1])
        demand_tensor = np.zeros(tensor_shape)

        # 填充张量
        grouped = valid.groupby(['time_slot', 'y_idx', 'x_idx']).size().reset_index(name='count')
        time_to_idx = {t: i for i, t in enumerate(all_slots)}

        for _, row in grouped.iterrows():
            t_idx = time_to_idx[row['time_slot']]
            demand_tensor[t_idx, int(row['y_idx']), int(row['x_idx'])] = row['count']

        return demand_tensor, all_slots

    def create_seq_data(self, demand_tensor, seq_len=6, pre_len=1, train_ratio=0.8):
        """构造滑动窗口数据集: 用前 seq_len 个时间步预测后 pre_len 个时间步"""
        X, Y = [], []
        for i in range(len(demand_tensor) - seq_len - pre_len + 1):
            X.append(demand_tensor[i: i + seq_len])
            Y.append(demand_tensor[i + seq_len: i + seq_len + pre_len])

        X = np.array(X)  # (Samples, seq_len, H, W)
        Y = np.array(Y)  # (Samples, pre_len, H, W)

        train_size = int(len(X) * train_ratio)
        return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]