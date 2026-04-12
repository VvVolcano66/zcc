import torch
import torch.nn as nn


class CenterTaskPatternNet(nn.Module):
    def __init__(
        self,
        num_centers: int,
        seq_len: int,
        aux_dim: int,
        hidden_dim: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.15,
        slot_vocab_size: int = 96,
        weekday_vocab_size: int = 7,
        time_embed_dim: int = 16,
    ):
        super().__init__()
        self.num_centers = num_centers
        self.seq_len = seq_len
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim

        lstm_dropout = dropout if lstm_layers > 1 else 0.0
        self.short_term_lstm = nn.LSTM(
            input_size=num_centers,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True,
        )
        self.short_term_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        self.slot_embedding = nn.Embedding(slot_vocab_size, time_embed_dim)
        self.weekday_embedding = nn.Embedding(weekday_vocab_size, time_embed_dim)
        self.time_context_proj = nn.Sequential(
            nn.Linear(time_embed_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
        )

        self.center_aux_proj = nn.Sequential(
            nn.Linear(aux_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.center_embeddings = nn.Embedding(num_centers, hidden_dim)

        self.global_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        self.center_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        self.base_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        short_term_x: torch.Tensor,
        center_aux_x: torch.Tensor,
        base_components: torch.Tensor,
        slot_ids: torch.Tensor,
        weekday_ids: torch.Tensor,
        is_weekend: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = short_term_x.size(0)

        short_outputs, (short_hidden, _) = self.short_term_lstm(short_term_x)
        attn_outputs, _ = self.short_term_attn(short_outputs, short_outputs, short_outputs)

        short_last = short_hidden[-1]
        short_last_reverse = short_hidden[-2]
        short_temporal = torch.cat([short_last, short_last_reverse], dim=-1)
        short_attended = attn_outputs.mean(dim=1)

        slot_embed = self.slot_embedding(slot_ids)
        weekday_embed = self.weekday_embedding(weekday_ids)
        time_context = self.time_context_proj(
            torch.cat([slot_embed, weekday_embed, is_weekend.unsqueeze(-1)], dim=-1)
        )

        global_context = self.global_fusion(
            torch.cat([short_temporal, short_attended, time_context], dim=-1)
        )

        center_aux_context = self.center_aux_proj(center_aux_x)
        center_ids = torch.arange(self.num_centers, device=short_term_x.device)
        center_embeddings = self.center_embeddings(center_ids).unsqueeze(0).expand(batch_size, -1, -1)
        global_context_expanded = global_context.unsqueeze(1).expand(-1, self.num_centers, -1)

        center_context = self.center_fusion(
            torch.cat([global_context_expanded, center_embeddings, center_aux_context], dim=-1)
        )

        base_weights = torch.softmax(self.base_gate(center_context), dim=-1)
        base_prediction = (base_weights * base_components).sum(dim=-1)
        delta_prediction = self.delta_head(center_context).squeeze(-1)
        return base_prediction + delta_prediction
