from __future__ import annotations

import torch
from torch import nn


def make_divisible(value: float, divisor: int = 8, min_value: int | None = None) -> int:
    min_value = divisor if min_value is None else min_value
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


class ConvBnAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activation: nn.Module | None = None,
    ) -> None:
        padding = (kernel_size - 1) // 2
        if activation is None:
            activation = nn.ReLU(inplace=True)
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            activation,
        )


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, se_ratio: float = 0.25) -> None:
        super().__init__()
        hidden = max(1, int(channels * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.pool(x))
        return x * scale


class SimAM(nn.Module):
    """
    Parameter-free attention (Yang et al., ICML 2021).

    Replaces SE in MBConv/Bneck without adding learnable parameters:
        e = (x-μ)^2 / (4*(σ^2 + λ)) + 0.5
        y = sigmoid(e) ⊙ x
    """

    def __init__(self, e_lambda: float = 1e-4) -> None:
        super().__init__()
        self.e_lambda = float(e_lambda)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Energy over spatial dims; broadcast per-location attention.
        n = x.shape[-2] * x.shape[-1] - 1
        if n <= 0:
            return x
        mu = x.mean(dim=(-2, -1), keepdim=True)
        x_minus_mu_square = (x - mu).pow(2)
        var = x_minus_mu_square.sum(dim=(-2, -1), keepdim=True) / float(n)
        energy = x_minus_mu_square / (4.0 * (var + self.e_lambda)) + 0.5
        return x * self.activation(energy)


def build_block_attention(
    attn_type: str,
    channels: int,
    *,
    se_ratio: float = 0.25,
    e_lambda: float = 1e-4,
) -> nn.Module:
    """Factory for in-block attention used by MobileNet Bneck/MBConv."""
    kind = str(attn_type).lower().strip()
    if kind in {"", "se", "squeeze_excitation", "squeeze-excitation"}:
        return SqueezeExcitation(channels, se_ratio=se_ratio)
    if kind in {"simam", "sim_am", "sim-am"}:
        return SimAM(e_lambda=e_lambda)
    if kind in {"none", "identity", "off"}:
        return nn.Identity()
    raise ValueError(f"Unsupported block attention type: {attn_type}")
