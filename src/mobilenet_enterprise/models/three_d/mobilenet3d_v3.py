from __future__ import annotations

import ast
from typing import Iterable

import torch
from torch import nn

from mobilenet_enterprise.models.base import MobileNetBase
from mobilenet_enterprise.models.blocks import ConvBnAct, build_block_attention, make_divisible
from mobilenet_enterprise.models.three_d.spectral_compressor import build_spectral_compressor
from mobilenet_enterprise.models.three_d.stem_blocks import build_stem3d_encoder


def _act(name: str) -> nn.Module:
    if name == "hswish":
        return nn.Hardswish(inplace=True)
    return nn.ReLU(inplace=True)


def _parse_channels(value: str | Iterable[int] | None, default: list[int]) -> list[int]:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return default
        if isinstance(parsed, list):
            nums = [int(v) for v in parsed if int(v) > 0]
            return nums if nums else default
        return default
    nums = [int(v) for v in value if int(v) > 0]
    return nums if nums else default


class MBConvV3_2D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int,
        expand: float,
        use_se: bool,
        act: str,
        attn_type: str = "se",
    ) -> None:
        super().__init__()
        hidden = make_divisible(in_ch * expand)
        self.use_residual = stride == 1 and in_ch == out_ch
        layers: list[nn.Module] = []
        if hidden != in_ch:
            layers.append(ConvBnAct(in_ch, hidden, kernel_size=1, activation=_act(act)))
        layers.append(
            ConvBnAct(
                hidden,
                hidden,
                kernel_size=kernel,
                stride=stride,
                groups=hidden,
                activation=_act(act),
            )
        )
        if use_se:
            # Default "se" keeps original Squeeze-and-Excitation; "simam" is parameter-free.
            layers.append(build_block_attention(attn_type, hidden))
        layers.extend([nn.Conv2d(hidden, out_ch, kernel_size=1, bias=False), nn.BatchNorm2d(out_ch)])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            return x + out
        return out


class MobileNet3DV3(MobileNetBase):
    """
    HSI model: 3D shallow encoder -> spectral compression -> MobileNetV3-style 2D deep encoder.

    Input tensor shape: (N, 1, B, H, W)

    stem3d_type options:
      - standard: full Conv3d(3,3,3) blocks (default, backward compatible)
      - pseudo_3d: (2+1)D factorized conv blocks
      - mbconv3d: depthwise-separable MBConv3D blocks

    spectral_compressor options:
      - adaptive_pool: avg/max adaptive spectral pooling (default)
      - query_attn: learnable query cross-attention spectral compression
    """

    def __init__(
        self,
        num_classes: int = 5,
        width_mult: float = 1.0,
        input_channels: int = 1,
        dropout: float = 0.2,
        variant: str = "small",
        hsi_bands: int = 454,
        stem3d_channels: str | list[int] = "[16,32]",
        stem3d_type: str = "standard",
        stem3d_max_spatial_pools: int | None = None,
        spectral_bins: int = 4,
        spectral_compressor: str = "adaptive_pool",
        spectral_attn_heads: int = 4,
        spectral_attn_layers: int = 1,
        spectral_attn_dropout: float = 0.1,
        v3_variant: str = "small",
        tiny_spatial_enabled: bool = False,
        classifier_hidden: int | None = None,
        block_attn_type: str = "se",
    ) -> None:
        super().__init__()
        del variant  # variant kept for registry compatibility; real 2D variant uses v3_variant
        self.hsi_bands = int(hsi_bands)
        self.spectral_bins = max(1, int(spectral_bins))
        self.stem3d_type = str(stem3d_type).lower()
        self.spectral_compressor_type = str(spectral_compressor).lower()
        self.tiny_spatial_enabled = bool(tiny_spatial_enabled)
        self.block_attn_type = str(block_attn_type).lower().strip() or "se"
        stem_channels = _parse_channels(stem3d_channels, default=[16, 32])
        self.encoder3d = build_stem3d_encoder(
            stem_type=self.stem3d_type,
            stem_channels=stem_channels,
            input_channels=int(input_channels),
            tiny_spatial_enabled=self.tiny_spatial_enabled,
            max_spatial_pools=stem3d_max_spatial_pools,
        )
        in3d = int(stem_channels[-1])
        self.spectral_filter = nn.Sequential(
            nn.Conv3d(
                in3d,
                in3d,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                padding=(1, 0, 0),
                groups=in3d,
                bias=False,
            ),
            nn.BatchNorm3d(in3d),
            nn.Hardswish(inplace=True),
        )
        self.spectral_compressor = build_spectral_compressor(
            compressor_type=self.spectral_compressor_type,
            channels=in3d,
            spectral_bins=self.spectral_bins,
            num_heads=spectral_attn_heads,
            num_layers=spectral_attn_layers,
            dropout=spectral_attn_dropout,
        )

        if v3_variant in {"large", "v3_google_large", "google_large"}:
            setting = [
                (24, 3, 2, 4.0, False, "relu"),
                (24, 3, 1, 3.0, False, "relu"),
                (40, 5, 2, 3.0, True, "relu"),
                (40, 5, 1, 3.0, True, "relu"),
                (80, 3, 2, 6.0, False, "hswish"),
                (80, 3, 1, 2.5, False, "hswish"),
                (112, 3, 1, 6.0, True, "hswish"),
                (160, 5, 2, 6.0, True, "hswish"),
            ]
            base_head = 960
            cls_hidden = 1280
        else:
            setting = [
                (24, 3, 2, 4.0, False, "relu"),
                (24, 3, 1, 3.0, False, "relu"),
                (40, 5, 2, 3.0, True, "hswish"),
                (40, 5, 1, 3.0, True, "hswish"),
                (48, 5, 1, 3.0, True, "hswish"),
                (96, 5, 2, 6.0, True, "hswish"),
                (96, 5, 1, 6.0, True, "hswish"),
            ]
            base_head = 576
            cls_hidden = 1024
        if self.tiny_spatial_enabled:
            downsample_slowdown_budget = 2
            tuned_setting: list[tuple[int, int, int, float, bool, str]] = []
            for c, k, s, e, se, act in setting:
                stride = s
                if stride == 2 and downsample_slowdown_budget > 0:
                    stride = 1
                    downsample_slowdown_budget -= 1
                tuned_setting.append((c, k, stride, e, se, act))
            setting = tuned_setting

        in2d = make_divisible(in3d * self.spectral_bins * width_mult)
        features2d: list[nn.Module] = []
        for c, k, s, e, se, act in setting:
            out_ch = make_divisible(c * width_mult)
            features2d.append(
                MBConvV3_2D(
                    in2d,
                    out_ch,
                    kernel=k,
                    stride=s,
                    expand=e,
                    use_se=se,
                    act=act,
                    attn_type=self.block_attn_type,
                )
            )
            in2d = out_ch
        head_ch = make_divisible(base_head * width_mult)
        if classifier_hidden is not None:
            cls_hidden = make_divisible(int(classifier_hidden))
        else:
            cls_hidden = make_divisible(cls_hidden * width_mult)
        features2d.append(ConvBnAct(in2d, head_ch, kernel_size=1, activation=nn.Hardswish(inplace=True)))
        self.features2d = nn.Sequential(*features2d)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(head_ch, cls_hidden),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden, num_classes),
        )

    def _reshape_3d_to_2d(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spectral_filter(x)
        return self.spectral_compressor(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder3d(x)
        x = self._reshape_3d_to_2d(x)
        return self.features2d(x)

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).flatten(1)
        return self.classifier(x)
