import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterTokenExtractor(nn.Module):
    """
    Use learned center queries to summarize grid demand into center-level tokens.
    """

    def __init__(self, hidden_dim, num_centers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_centers = num_centers
        self.center_queries = nn.Parameter(torch.randn(num_centers, hidden_dim))
        nn.init.xavier_uniform_(self.center_queries)

    def forward(self, spatial_tokens):
        # spatial_tokens: (B, N, C)
        batch_size = spatial_tokens.size(0)
        queries = self.center_queries.unsqueeze(0).expand(batch_size, -1, -1)  # (B, K, C)
        attn_scores = torch.matmul(queries, spatial_tokens.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, K, N)
        center_tokens = torch.matmul(attn_weights, spatial_tokens)  # (B, K, C)
        return center_tokens


class MCTGNet(nn.Module):
    """
    Multi-Center Trend-Gated Network

    Innovations tailored to the user's dispatch problem:
    1. Center-aware tokens summarize grid demand into dispatch-center level latent states.
    2. Grid-to-center cross attention decodes center collaboration back to each cell.
    3. Residual delta prediction and recent-trend gating preserve local bursts.
    """

    def __init__(self, seq_len, grid_size=(5, 5), hidden_dim=64, num_centers=5, num_time_slots=8):
        super().__init__()
        self.seq_len = seq_len
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        self.num_centers = num_centers
        self.num_time_slots = num_time_slots

        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.center_extractor = CenterTokenExtractor(hidden_dim=hidden_dim, num_centers=num_centers)

        self.global_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.center_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.global_proj = nn.Linear(hidden_dim, hidden_dim)

        self.grid_query = nn.Linear(hidden_dim, hidden_dim)
        self.center_key = nn.Linear(hidden_dim, hidden_dim)
        self.center_value = nn.Linear(hidden_dim, hidden_dim)
        self.time_embedding = nn.Embedding(num_time_slots, hidden_dim)

        self.periodic_encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.trend_encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1)
        )

        self.token_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.delta_head = nn.Linear(hidden_dim, 1)
        self.periodic_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, periodic_x=None, slot_ids=None):
        # x: (B, T, H, W)
        batch_size, seq_len, height, width = x.shape

        spatial_features = []
        global_features = []
        center_feature_seq = []

        for t in range(seq_len):
            frame = x[:, t].unsqueeze(1)  # (B, 1, H, W)
            feat = self.spatial_encoder(frame)  # (B, C, H, W)
            spatial_features.append(feat)

            pooled = F.adaptive_avg_pool2d(feat, 1).view(batch_size, self.hidden_dim)
            global_features.append(pooled)

            spatial_tokens = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
            center_tokens = self.center_extractor(spatial_tokens)  # (B, K, C)
            center_feature_seq.append(center_tokens)

        global_seq = torch.stack(global_features, dim=1)  # (B, T, C)
        _, global_hidden = self.global_gru(global_seq)
        global_context = self.global_proj(global_hidden[-1]).view(batch_size, self.hidden_dim, 1, 1)

        center_seq = torch.stack(center_feature_seq, dim=1)  # (B, T, K, C)
        center_seq = center_seq.permute(0, 2, 1, 3).contiguous()  # (B, K, T, C)
        center_seq = center_seq.view(batch_size * self.num_centers, seq_len, self.hidden_dim)
        _, center_hidden = self.center_gru(center_seq)
        center_hidden = center_hidden[-1].view(batch_size, self.num_centers, self.hidden_dim)

        last_feat = spatial_features[-1]
        last_frame = x[:, -1]
        grid_tokens = last_feat.flatten(2).transpose(1, 2)  # (B, N, C)

        if periodic_x is None:
            periodic_frame = torch.zeros_like(last_frame)
            periodic_tokens = torch.zeros_like(grid_tokens)
            periodic_global = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        else:
            if periodic_x.dim() == 4:
                periodic_frame = periodic_x[:, 0]
            elif periodic_x.dim() == 3:
                periodic_frame = periodic_x
            else:
                raise ValueError("periodic_x must have shape (B, 1, H, W) or (B, H, W)")
            periodic_feat = self.periodic_encoder(periodic_frame.unsqueeze(1))
            periodic_tokens = periodic_feat.flatten(2).transpose(1, 2)
            periodic_global = F.adaptive_avg_pool2d(periodic_feat, 1).view(batch_size, self.hidden_dim)

        if seq_len >= 2:
            trend_input = (x[:, -1] - x[:, -2]).unsqueeze(1)
        else:
            trend_input = x[:, -1].unsqueeze(1)
        trend_feat = self.trend_encoder(trend_input).flatten(2).transpose(1, 2)  # (B, N, C)

        queries = self.grid_query(grid_tokens)  # (B, N, C)
        keys = self.center_key(center_hidden)   # (B, K, C)
        values = self.center_value(center_hidden)  # (B, K, C)
        attn_scores = torch.matmul(queries, keys.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        center_message = torch.matmul(attn_weights, values)  # (B, N, C)

        global_message = global_context.view(batch_size, 1, self.hidden_dim).expand_as(grid_tokens)
        if slot_ids is None:
            time_message = torch.zeros_like(grid_tokens)
        else:
            time_embed = self.time_embedding(slot_ids).view(batch_size, 1, self.hidden_dim)
            time_message = time_embed.expand_as(grid_tokens)
        trend_gate = torch.sigmoid(trend_feat)

        fused_tokens = torch.cat([grid_tokens, center_message, global_message, periodic_tokens, time_message], dim=-1)
        fused_tokens = self.token_fusion(fused_tokens)
        fused_tokens = fused_tokens * (1.0 + trend_gate)

        delta = self.delta_head(fused_tokens).squeeze(-1).view(batch_size, height, width)
        global_flat = global_context.view(batch_size, self.hidden_dim)
        periodic_alpha = self.periodic_gate(torch.cat([global_flat, periodic_global], dim=-1)).view(batch_size, 1, 1)
        base = last_frame + periodic_alpha * (periodic_frame - last_frame)
        pred = base + delta
        return pred
