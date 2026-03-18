import torch
import torch.nn as nn


class SpatioTemporalNet(nn.Module):
    def __init__(self, seq_len, grid_size=(10, 10)):
        super(SpatioTemporalNet, self).__init__()
        # 空间特征提取 (把 seq_len 当作通道数)
        self.conv1 = nn.Conv2d(in_channels=seq_len, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        # x shape: (Batch, seq_len, H, W)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)  # (Batch, 1, H, W)
        return out