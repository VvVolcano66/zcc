import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    极速+省内存版 GAT Layer：
    避免了 N*N 的暴力特征拼接 (彻底解决 RuntimeError: not enough memory 报错)
    """

    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(alpha)

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # 优化核心：将原本大小为 [2*out_features, 1] 的 a 拆分为 a1 和 a2
        self.a1 = nn.Parameter(torch.empty(size=(out_features, 1)))
        self.a2 = nn.Parameter(torch.empty(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

    def forward(self, h, adj):
        # h: (Batch, N, in_features)
        Wh = torch.matmul(h, self.W)  # (Batch, N, out_features)

        # 核心加速与省内存魔法：利用 (Wh_i || Wh_j) * a = Wh_i * a1 + Wh_j * a2 的数学性质
        e1 = torch.matmul(Wh, self.a1)  # (Batch, N, 1)
        e2 = torch.matmul(Wh, self.a2)  # (Batch, N, 1)

        # 利用 PyTorch 广播机制计算 e_ij (Batch, N, N)
        e = self.leakyrelu(e1 + e2.transpose(-1, -2))

        # 掩码操作
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)  # (Batch, N, out_features)
        return F.elu(h_prime)


class DGANet(nn.Module):
    """
    双图注意力网络 (Dual Graph Attention Network)，处理地理空间相关性
    """

    def __init__(self, in_features, out_features):
        super(DGANet, self).__init__()
        self.gat_n = GraphAttentionLayer(in_features, out_features)
        self.gat_d = GraphAttentionLayer(in_features, out_features)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, h, adj_n, adj_d, G_nd):
        f_n_S = self.gat_n(h, adj_n)  # (B, N, out_features)
        f_d_S = self.gat_d(h, adj_d)  # (B, N, out_features)

        G_nd_flat = G_nd.view(2, -1)  # (2, N*N)
        M_c = F.softmax(torch.matmul(G_nd_flat, G_nd_flat.t()), dim=-1)  # (2, 2)

        f_nd_C_flat = self.scale * torch.matmul(M_c, G_nd_flat) + G_nd_flat
        f_nd_C = f_nd_C_flat.view(2, G_nd.size(1), G_nd.size(2))

        f_nd_S = f_n_S + f_d_S
        return f_nd_S


class BSTGCNet(nn.Module):
    """
    双边时空图卷积网络 (Bilateral Spatial-Temporal Graph Convolutional Network)
    """

    def __init__(self, num_nodes, in_features, hidden_dim, seq_len, pre_len):
        super(BSTGCNet, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.hidden_dim = hidden_dim

        self.semantic_gat = GraphAttentionLayer(in_features, hidden_dim)
        self.geographic_dga = DGANet(in_features, hidden_dim)
        self.spatial_fuse = nn.Linear(hidden_dim * 2, hidden_dim)

        self.gru = nn.GRU(input_size=hidden_dim + in_features,
                          hidden_size=hidden_dim,
                          batch_first=True)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, pre_len)

    def forward(self, X, G_s, G_n, G_d):
        Batch = X.size(0)
        G_nd = torch.stack([G_n, G_d], dim=0)  # (2, N, N)

        # ==== 核心加速：将 Batch 和 Seq_Len 维度合并，实现空间特征并行计算 ====
        # 原始 X 形状: (Batch, Seq_Len, N, In_Features) -> (Batch * Seq_Len, N, In_Features)
        x_flat = X.contiguous().view(Batch * self.seq_len, self.num_nodes, -1)

        f_s_S = self.semantic_gat(x_flat, G_s)
        f_nd_S = self.geographic_dga(x_flat, G_n, G_d, G_nd)

        spatial_feat = torch.cat([f_s_S, f_nd_S], dim=-1)
        spatial_feat = F.relu(self.spatial_fuse(spatial_feat))

        gru_in_flat = torch.cat([spatial_feat, x_flat], dim=-1)

        # ==== 恢复时间序列形状供 GRU 处理 ====
        gru_in = gru_in_flat.view(Batch, self.seq_len, self.num_nodes, -1)
        gru_in = gru_in.transpose(1, 2).contiguous()
        gru_in = gru_in.view(Batch * self.num_nodes, self.seq_len, -1)

        out, h_n = self.gru(gru_in)
        f_t_ST = out[:, -1, :]

        pred = F.relu(self.fc1(f_t_ST))
        pred = self.fc2(pred)

        pred = pred.view(Batch, self.num_nodes, self.pre_len)
        return pred.transpose(1, 2)