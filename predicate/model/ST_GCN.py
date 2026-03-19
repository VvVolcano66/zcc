# predicate/model/ST_GCN.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    基础的图卷积层 (GCN Layer)
    公式: H^{(l+1)} = \sigma( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} )
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, text, adj):
        # text: (Batch, N, in_features)
        # adj: (N, N) 归一化后的拉普拉斯矩阵或邻接矩阵
        support = torch.matmul(text, self.weight)  # (Batch, N, out_features)

        # 稀疏矩阵乘法，融合邻居特征
        output = torch.matmul(adj, support)  # (Batch, N, out_features)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class ST_GCN(nn.Module):
    def __init__(self, num_nodes, seq_len, in_channels=1, hidden_dim=64):
        super(ST_GCN, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim  # <--- 新增这一行！

        # 1. 空间层: 图卷积提取空间相关性
        self.gcn1 = GraphConvolution(in_features=in_channels, out_features=32)
        self.gcn2 = GraphConvolution(in_features=32, out_features=hidden_dim)

        # 2. 时间层: 利用 1D-CNN 提取时间序列维度上的依赖
        # 这里把节点当成独立的 Channel 进行一维卷积
        self.tcn = nn.Conv1d(in_channels=num_nodes * hidden_dim,
                             out_channels=num_nodes * hidden_dim,
                             kernel_size=3, padding=1, groups=num_nodes)  # 深度可分离卷积形式

        # 3. 预测层: 压缩时间维度，输出下一时刻的节点任务量
        self.fc = nn.Linear(hidden_dim * seq_len, 1)

    def forward(self, x, adj):
        """
        x: 输入特征, shape (Batch, seq_len, num_nodes)
        adj: 预先计算好的路网邻接矩阵, shape (num_nodes, num_nodes)
        """
        batch_size = x.shape[0]

        # 将输入维度扩展以适应 GCN (Batch, seq_len, num_nodes, in_channels)
        x = x.unsqueeze(-1)

        # --- 步骤 1: 对每个时间步执行图卷积 (Spatial Convolution) ---
        gcn_outputs = []
        for t in range(self.seq_len):
            x_t = x[:, t, :, :]  # (Batch, N, 1)
            h1 = F.relu(self.gcn1(x_t, adj))  # (Batch, N, 32)
            h2 = F.relu(self.gcn2(h1, adj))  # (Batch, N, hidden_dim)
            gcn_outputs.append(h2)

        # 堆叠起来: (Batch, seq_len, num_nodes, hidden_dim)
        st_features = torch.stack(gcn_outputs, dim=1)

        # --- 步骤 2: 时间卷积 (Temporal Convolution) ---
        # 调整形状适应 Conv1d: 要求输入 (Batch, Channels, Length)
        # 我们把 num_nodes * hidden_dim 作为通道，seq_len 作为长度
        tcn_input = st_features.permute(0, 2, 3, 1).contiguous()
        tcn_input = tcn_input.view(batch_size, self.num_nodes * self.hidden_dim, self.seq_len)

        tcn_output = F.relu(self.tcn(tcn_input))  # (Batch, N * hidden_dim, seq_len)

        # --- 步骤 3: 全连接预测下一时刻 ---
        # 恢复形状 (Batch, num_nodes, hidden_dim * seq_len)
        out_features = tcn_output.view(batch_size, self.num_nodes, self.hidden_dim, self.seq_len)
        out_features = out_features.reshape(batch_size, self.num_nodes, -1)

        # 预测节点数值: (Batch, num_nodes, 1) -> (Batch, num_nodes)
        prediction = self.fc(out_features).squeeze(-1)

        return prediction