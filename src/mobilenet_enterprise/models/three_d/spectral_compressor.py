from __future__ import annotations

import torch
from torch import nn


class SpectralQueryBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        ffn_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        channels = int(channels)
        num_heads = max(1, min(int(num_heads), channels))
        while channels % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        ffn_dim = max(channels, int(channels * max(float(ffn_ratio), 1.0)))
        attn_dropout = min(max(float(dropout), 0.0), 0.5)

        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.attn_scale = nn.Parameter(torch.tensor(0.25))
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(attn_dropout),
            nn.Linear(ffn_dim, channels),
            nn.Dropout(attn_dropout),
        )
        self.ffn_scale = nn.Parameter(torch.zeros(1))
        self._init_stable()

    def _init_stable(self) -> None:
        out_proj = self.attn.out_proj
        nn.init.xavier_uniform_(out_proj.weight, gain=0.5)
        nn.init.zeros_(out_proj.bias)
        last = self.ffn[-2]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, queries: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        queries = queries.float()
        tokens = tokens.float()
        q = self.norm_q(queries)
        kv = self.norm_kv(tokens)
        context, _ = self.attn(q, kv, kv, need_weights=False)
        queries = queries + torch.tanh(self.attn_scale) * self.attn_dropout(context)
        queries = queries + torch.tanh(self.ffn_scale) * self.ffn(self.ffn_norm(queries))
        return queries


class SpectralQueryCompressor(nn.Module):
    """
    Perceiver-style spectral compression.

    Input:  (N, C, B, H, W)
    Output: (N, C * num_queries, H, W)
    """

    def __init__(
        self,
        channels: int,
        num_queries: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_queries = max(1, int(num_queries))
        self.queries = nn.Parameter(torch.zeros(1, self.num_queries, self.channels))
        self.layers = nn.ModuleList(
            [
                SpectralQueryBlock(
                    channels=self.channels,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.out_norm = nn.BatchNorm2d(self.channels * self.num_queries)
        self.out_act = nn.Hardswish(inplace=True)
        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, _bands, h, w = x.shape
        tokens = x.permute(0, 3, 4, 2, 1).reshape(n * h * w, _bands, c)
        queries = self.queries.expand(n * h * w, -1, -1).to(dtype=tokens.dtype)
        for layer in self.layers:
            queries = layer(queries, tokens)
        out = queries.reshape(n, h, w, self.num_queries, c).permute(0, 4, 3, 1, 2)
        return self.out_act(self.out_norm(out.reshape(n, c * self.num_queries, h, w)))


class AdaptiveSpectralPool(nn.Module):
    """Original avg/max adaptive spectral pooling used by MobileNet3DV3."""

    def __init__(self, spectral_bins: int) -> None:
        super().__init__()
        self.spectral_bins = max(1, int(spectral_bins))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled_avg = torch.nn.functional.adaptive_avg_pool3d(
            x, (self.spectral_bins, x.shape[-2], x.shape[-1])
        )
        pooled_max = torch.nn.functional.adaptive_max_pool3d(
            x, (self.spectral_bins, x.shape[-2], x.shape[-1])
        )
        x = 0.5 * (pooled_avg + pooled_max)
        n, c, b, h, w = x.shape
        return x.view(n, c * b, h, w)


class SpectralGatedAdaptivePool(nn.Module):
    """
    Adaptive spectral pooling with a learnable residual band gate.

    This is intentionally conservative for ablation: the original adaptive
    avg/max pooling path is preserved, while the gate learns per-band emphasis.
    """

    def __init__(self, channels: int, spectral_bins: int, gate_kernel: int = 3) -> None:
        super().__init__()
        self.pool = AdaptiveSpectralPool(spectral_bins=spectral_bins)
        channels = int(channels)
        padding = max(0, int(gate_kernel) // 2)
        self.gate = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=int(gate_kernel), padding=padding, groups=channels),
            nn.BatchNorm1d(channels),
            nn.Hardswish(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.gate_scale = nn.Parameter(torch.zeros(1))
        self._init_gate()

    def _init_gate(self) -> None:
        last = self.gate[-1]
        if isinstance(last, nn.Conv1d):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        descriptor = x.mean(dim=(-2, -1))
        gate = torch.sigmoid(self.gate(descriptor)).unsqueeze(-1).unsqueeze(-1)
        scale = torch.tanh(self.gate_scale)
        gated = x * (1.0 + scale * (gate - 0.5))
        return self.pool(gated)


class ResidualSpectralQueryCompressor(nn.Module):
    """
    Residual version of query attention spectral compression.

    The known-good adaptive pooling output is the main path. Query attention is
    introduced as a learnable residual branch, initialized close to zero.
    """

    def __init__(
        self,
        channels: int,
        spectral_bins: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pool = AdaptiveSpectralPool(spectral_bins=spectral_bins)
        self.query = SpectralQueryCompressor(
            channels=channels,
            num_queries=spectral_bins,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.query_scale = nn.Parameter(torch.full((1,), -4.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x)
        query = self.query(x).to(dtype=pooled.dtype)
        return pooled + torch.sigmoid(self.query_scale) * query


def build_spectral_compressor(
    compressor_type: str,
    channels: int,
    spectral_bins: int,
    num_heads: int = 4,
    num_layers: int = 1,
    dropout: float = 0.1,
) -> nn.Module:
    compressor_type = str(compressor_type).lower()
    if compressor_type in {"query_attn", "spectral_query", "perceiver"}:
        return SpectralQueryCompressor(
            channels=channels,
            num_queries=spectral_bins,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
    if compressor_type in {"residual_query", "residual_query_attn", "query_residual"}:
        return ResidualSpectralQueryCompressor(
            channels=channels,
            spectral_bins=spectral_bins,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
    if compressor_type in {"spectral_gate", "gated_pool", "gate"}:
        return SpectralGatedAdaptivePool(channels=channels, spectral_bins=spectral_bins)
    if compressor_type in {"adaptive_pool", "adaptive", "pool"}:
        return AdaptiveSpectralPool(spectral_bins=spectral_bins)
    raise ValueError(f"Unsupported spectral_compressor: {compressor_type}")
