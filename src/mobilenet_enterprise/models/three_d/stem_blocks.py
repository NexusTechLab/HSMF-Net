from __future__ import annotations

import torch
from torch import nn

from mobilenet_enterprise.models.blocks import make_divisible


def _act(name: str = "relu") -> nn.Module:
    if name == "hswish":
        return nn.Hardswish(inplace=True)
    return nn.ReLU(inplace=True)


def _new_activation(activation: nn.Module | None) -> nn.Module:
    if activation is None:
        return nn.ReLU(inplace=True)
    return activation.__class__()


class Conv3dBnAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int] = (3, 3, 3),
        stride: tuple[int, int, int] = (1, 1, 1),
        activation: nn.Module | None = None,
    ) -> None:
        if activation is None:
            activation = nn.ReLU(inplace=True)
        padding = tuple((k - 1) // 2 for k in kernel_size)
        super().__init__(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            activation,
        )


class Conv2Plus1d(nn.Module):
    """Pseudo-3D block: spatial (1, k, k) followed by spectral (k, 1, 1)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_spatial: int = 3,
        kernel_spectral: int = 3,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if activation is None:
            activation = nn.ReLU(inplace=True)
        self.use_residual = in_channels == out_channels
        pad_s = kernel_spatial // 2
        pad_b = kernel_spectral // 2
        self.spatial = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(1, kernel_spatial, kernel_spatial),
                padding=(0, pad_s, pad_s),
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            _new_activation(activation),
        )
        self.spectral = nn.Sequential(
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=(kernel_spectral, 1, 1),
                padding=(pad_b, 0, 0),
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
        )
        self.out_act = _new_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.spectral(self.spatial(x))
        if self.use_residual:
            out = out + x
        return self.out_act(out)


class MBConv3D(nn.Module):
    """Lightweight 3D inverted bottleneck: expand -> depthwise 3D -> project."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        expand: float = 2.0,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if activation is None:
            activation = nn.ReLU(inplace=True)
        hidden = make_divisible(in_channels * expand)
        self.use_residual = in_channels == out_channels
        padding = kernel_size // 2
        layers: list[nn.Module] = []
        if hidden != in_channels:
            layers.extend(
                [
                    nn.Conv3d(in_channels, hidden, kernel_size=1, bias=False),
                    nn.BatchNorm3d(hidden),
                    _new_activation(activation),
                ]
            )
        layers.extend(
            [
                nn.Conv3d(
                    hidden,
                    hidden,
                    kernel_size=(kernel_size, kernel_size, kernel_size),
                    padding=(padding, padding, padding),
                    groups=hidden,
                    bias=False,
                ),
                nn.BatchNorm3d(hidden),
                _new_activation(activation),
                nn.Conv3d(hidden, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
            ]
        )
        self.block = nn.Sequential(*layers)
        self.out_act = _new_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = x + out
        return self.out_act(out)


def build_stem3d_encoder(
    stem_type: str,
    stem_channels: list[int],
    input_channels: int = 1,
    tiny_spatial_enabled: bool = False,
    max_spatial_pools: int | None = None,
) -> nn.Sequential:
    stem_type = str(stem_type).lower()
    stages: list[nn.Module] = []
    in_ch = int(input_channels)
    pool_count = 0
    for idx, out_ch in enumerate(stem_channels):
        out_ch = int(out_ch)
        if stem_type == "pseudo_3d":
            stages.append(Conv2Plus1d(in_ch, out_ch))
        elif stem_type == "mbconv3d":
            stages.append(MBConv3D(in_ch, out_ch))
        elif stem_type == "standard":
            stages.append(Conv3dBnAct(in_ch, out_ch, kernel_size=(3, 3, 3)))
        else:
            raise ValueError(f"Unsupported stem3d_type: {stem_type}")
        should_pool = not tiny_spatial_enabled or idx > 0
        if max_spatial_pools is not None:
            should_pool = should_pool and pool_count < int(max_spatial_pools)
        if should_pool:
            stages.append(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
            pool_count += 1
        in_ch = out_ch
    return nn.Sequential(*stages)
