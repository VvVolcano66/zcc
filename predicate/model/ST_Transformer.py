import torch
import torch.nn as nn


class ST_Transformer(nn.Module):
    def __init__(self, seq_len, grid_size=(10, 10), d_model=64, nhead=4, num_layers=2):
        super(ST_Transformer, self).__init__()
        self.seq_len = seq_len
        self.grid_size = grid_size

        # 1. 空间特征提取：将每个时间步的二维网格转化为一维特征向量
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 降采样，例如 10x10 -> 5x5, 5x5 -> 2x2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 💡 核心修复：根据传入的 grid_size 动态计算池化后的维度
        # MaxPool2d(kernel_size=2) 默认 stride 也为 2，尺寸折半并向下取整
        h_out = grid_size[0] // 2
        w_out = grid_size[1] // 2
        conv_out_size = h_out * w_out * 32

        # 将空间特征映射到 Transformer 的隐层维度
        self.feature_proj = nn.Linear(conv_out_size, d_model)

        # 2. 时间特征提取：Transformer Encoder 捕捉时序关系
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 预测输出层：将 Transformer 的输出映射回预测的网格数量
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, grid_size[0] * grid_size[1])
        )

    def forward(self, x):
        # x shape: (Batch, seq_len, H, W) -> 需要增加一个 channel 维度变为 (Batch, seq_len, 1, H, W)
        x = x.unsqueeze(2)
        batch_size = x.shape[0]

        # 处理每个时间步的空间特征
        spatial_features = []
        for t in range(self.seq_len):
            xt = x[:, t, :, :, :]  # 取出第 t 个时间步
            conv_out = self.spatial_conv(xt)  # (Batch, 32, H_out, W_out)
            flat_out = conv_out.view(batch_size, -1)  # 展平
            proj_out = self.feature_proj(flat_out)  # (Batch, d_model)
            spatial_features.append(proj_out.unsqueeze(1))

        # 拼接时间序列 (Batch, seq_len, d_model)
        time_seq = torch.cat(spatial_features, dim=1)

        # 输入 Transformer 进行全局时间建模
        trans_out = self.transformer(time_seq)  # (Batch, seq_len, d_model)

        # 展平所有时间步的特征进行解码
        trans_out_flat = trans_out.reshape(batch_size, -1)

        # 预测下一时刻的网格
        pred = self.output_proj(trans_out_flat)  # (Batch, H * W)
        pred = pred.view(batch_size, self.grid_size[0], self.grid_size[1])  # 恢复成 (Batch, H, W)

        return pred