from __future__ import annotations

import torch
from torch import nn


class CrossModalAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        attention_dropout: float | None = None,
        ffn_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_heads = self._valid_head_count(hidden_dim, num_heads)
        attn_dropout = min(float(dropout), 0.1) if attention_dropout is None else float(attention_dropout)
        attn_dropout = min(max(attn_dropout, 0.0), 0.5)
        dropout = min(max(float(dropout), 0.0), 0.8)
        ffn_dim = max(hidden_dim, int(hidden_dim * max(float(ffn_ratio), 1.0)))

        self.modality_embed = nn.Parameter(torch.zeros(1, 2, hidden_dim))
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.attn_scale = nn.Parameter(torch.zeros(1))
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_scale = nn.Parameter(torch.zeros(1))
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.output_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Hardswish(inplace=True),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )
        self.last_attention_weights: torch.Tensor | None = None
        self._init_gate()

    @staticmethod
    def _valid_head_count(hidden_dim: int, num_heads: int) -> int:
        heads = max(1, min(int(num_heads), hidden_dim))
        while hidden_dim % heads != 0 and heads > 1:
            heads -= 1
        return heads

    def _init_gate(self) -> None:
        gate_linear = self.output_gate[-2]
        if isinstance(gate_linear, nn.Linear):
            nn.init.zeros_(gate_linear.weight)
            nn.init.constant_(gate_linear.bias, -2.0)

    def forward(self, rgb_feat: torch.Tensor, hsi_feat: torch.Tensor) -> torch.Tensor:
        original_tokens = torch.stack([rgb_feat, hsi_feat], dim=1)
        tokens = original_tokens + self.modality_embed

        attn_tokens = self.attn_norm(tokens)
        context, attn_weights = self.attn(
            attn_tokens,
            attn_tokens,
            attn_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_attention_weights = attn_weights.detach()
        tokens = tokens + torch.tanh(self.attn_scale) * self.attn_dropout(context)
        tokens = tokens + torch.tanh(self.ffn_scale) * self.ffn(self.ffn_norm(tokens))

        gate_input = torch.cat([tokens[:, 0], tokens[:, 1]], dim=1)
        gate = self.output_gate(gate_input).unsqueeze(-1)
        stable_tokens = gate * self.out_norm(tokens) + (1.0 - gate) * original_tokens
        return torch.cat([stable_tokens[:, 0], stable_tokens[:, 1]], dim=1)
