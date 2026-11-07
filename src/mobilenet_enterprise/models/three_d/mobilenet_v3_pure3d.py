from __future__ import annotations

from functools import partial
from typing import Callable

import torch
from torch import nn

from mobilenet_enterprise.models.base import MobileNetBase
from mobilenet_enterprise.models.blocks import make_divisible
from mobilenet_enterprise.models.two_d.mobilenet_v3 import (
    InvertedResidualConfig,
    _mobilenet_v3_conf_local,
)


def _to_stride3d(stride: int) -> tuple[int, int, int]:
    """Map MobileNetV3 2D stride to HSI-friendly 3D stride (spectral, H, W)."""
    if stride == 2:
        return (1, 2, 2)
    return (1, 1, 1)


class Conv3dNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
        activation_layer: type[nn.Module] | None = nn.ReLU,
    ) -> None:
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm3d, eps=0.001, momentum=0.01)
        stride_3d = _to_stride3d(stride)
        padding = (kernel_size // 2, kernel_size // 2, kernel_size // 2)
        layers: list[nn.Module] = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, kernel_size, kernel_size),
                stride=stride_3d,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_channels),
        ]
        if activation_layer is not None:
            layers.append(activation_layer(inplace=True))
        super().__init__(*layers)


class SElayer3D(nn.Module):
    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        scale_activation: type[nn.Module] = nn.Hardsigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv3d(squeeze_channels, input_channels, 1)
        self.activation = nn.ReLU(inplace=True)
        self.scale_activation = scale_activation(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return x * scale


class InvertedResidual3D(nn.Module):
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = partial(SElayer3D, scale_activation=nn.Hardsigmoid),
    ) -> None:
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")
        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        layers: list[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                Conv3dNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            Conv3dNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                groups=cnf.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )
        if cnf.use_se:
            squeeze_channels = make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels))
        layers.append(
            Conv3dNormActivation(
                cnf.expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result = result + x
        return result


class MobileNetV3Pure3D(MobileNetBase):
    """
    Pure 3D MobileNetV3 mirror of the RGB MobileNetV3-Small backbone.

    Uses the same inverted-residual block table as MobileNetV3-Small, with
    Conv3d / depthwise Conv3d / SE3D / AdaptiveAvgPool3d replacements.
    Input shape: (N, 1, B, H, W)
    """

    def __init__(
        self,
        num_classes: int = 4,
        width_mult: float = 1.0,
        input_channels: int = 1,
        dropout: float = 0.2,
        variant: str = "v3_small",
        hsi_bands: int = 24,
        classifier_hidden: int | None = None,
    ) -> None:
        super().__init__()
        del hsi_bands
        variant_key = str(variant).lower()
        if variant_key in {"small", "v3_small"}:
            arch = "mobilenet_v3_small"
        elif variant_key in {"large", "v3_large"}:
            arch = "mobilenet_v3_large"
        else:
            raise ValueError(f"Unsupported Pure3D MobileNetV3 variant: {variant}")

        inverted_residual_setting, last_channel = _mobilenet_v3_conf_local(
            arch=arch,
            width_mult=width_mult,
        )
        norm_layer = partial(nn.BatchNorm3d, eps=0.001, momentum=0.01)
        firstconv_output_channels = inverted_residual_setting[0].input_channels

        layers: list[nn.Module] = [
            Conv3dNormActivation(
                int(input_channels),
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        ]
        for cnf in inverted_residual_setting:
            layers.append(InvertedResidual3D(cnf, norm_layer))

        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            Conv3dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
