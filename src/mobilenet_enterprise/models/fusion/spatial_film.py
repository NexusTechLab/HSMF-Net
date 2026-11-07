from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class AdaptiveCrossModalSpatialAlign(nn.Module):
    """
    Adaptive Cross-modal Spatial Alignment (ACSA).

    Learn a channel projection, then adaptively pool to the partner modality's
    spatial grid (HSI-native sizes in this project: typically 12x12 / 6x6).

        X --1x1 Conv-BN-Act--> AdaptiveAvgPool2d(H*, W*) --> aligned map
    """

    def __init__(self, in_channels: int, out_channels: int, align_mode: str = "acsa") -> None:
        super().__init__()
        self.align_mode = str(align_mode).lower().strip() or "acsa"
        if self.align_mode not in {"acsa", "bilinear"}:
            raise ValueError(f"Unsupported align_mode: {align_mode}")
        self.proj = nn.Sequential(
            nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(out_channels)),
            nn.Hardswish(inplace=True),
        )

    def forward(self, feat: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
        x = self.proj(feat)
        if x.shape[-2:] == size_hw:
            return x
        if self.align_mode == "bilinear":
            return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
        return F.adaptive_avg_pool2d(x, output_size=size_hw)


class SpatialResidualBilateralFiLM(nn.Module):
    """
    Residual / affine / unidirectional feature modulation on aligned maps (B, C, H, W).

    Default (paper HBFM):
        F'_rgb = F_rgb + γ_hsi ⊙ F_rgb
        F'_hsi = F_hsi + γ_rgb ⊙ F_hsi
    with near-zero initialization so γ≈0 at start.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int | None = None,
        dropout: float = 0.0,
        direction: str = "bilateral",
        residual: bool = True,
        affine: bool = False,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        channels = int(channels)
        hidden = int(hidden_channels) if hidden_channels is not None else channels
        drop = min(max(float(dropout), 0.0), 0.5)
        self.direction = str(direction).lower().strip() or "bilateral"
        if self.direction not in {"bilateral", "uni_hsi2rgb", "uni_rgb2hsi", "none"}:
            raise ValueError(f"Unsupported modulate direction: {direction}")
        self.residual = bool(residual)
        self.affine = bool(affine)
        self.zero_init = bool(zero_init)
        out_ch = channels * 2 if self.affine else channels

        self.hsi_to_rgb = None
        self.rgb_to_hsi = None
        if self.direction in {"bilateral", "uni_hsi2rgb"}:
            self.hsi_to_rgb = self._generator(channels, hidden, drop, out_ch)
        if self.direction in {"bilateral", "uni_rgb2hsi"}:
            self.rgb_to_hsi = self._generator(channels, hidden, drop, out_ch)
        if self.zero_init:
            self._init_identity()

    @staticmethod
    def _generator(channels: int, hidden: int, dropout: float, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.Hardswish(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=True),
        )

    def _init_identity(self) -> None:
        for gen in (self.hsi_to_rgb, self.rgb_to_hsi):
            if gen is None:
                continue
            last = gen[-1]
            if isinstance(last, nn.Conv2d):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def _apply_mod(self, feat: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        if self.affine:
            gamma, beta = params.chunk(2, dim=1)
            if self.residual:
                return feat + gamma * feat + beta
            return gamma * feat + beta
        gamma = params
        if self.residual:
            return feat + gamma * feat
        return gamma * feat

    def forward(self, rgb_feat: torch.Tensor, hsi_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if rgb_feat.shape[-2:] != hsi_feat.shape[-2:]:
            raise ValueError(
                "SpatialResidualBilateralFiLM expects aligned maps, "
                f"got {tuple(rgb_feat.shape)} vs {tuple(hsi_feat.shape)}"
            )
        if self.direction == "none":
            return rgb_feat, hsi_feat
        rgb_out = rgb_feat
        hsi_out = hsi_feat
        if self.hsi_to_rgb is not None:
            rgb_out = self._apply_mod(rgb_feat, self.hsi_to_rgb(hsi_feat))
        if self.rgb_to_hsi is not None:
            hsi_out = self._apply_mod(hsi_feat, self.rgb_to_hsi(rgb_feat))
        return rgb_out, hsi_out


# Backward-compatible alias used by earlier mid-film drafts.
class SpatialBilateralFiLM(SpatialResidualBilateralFiLM):
    """Alias of residual bidirectional FiLM (preferred formulation)."""


class _ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Hardswish(inplace=True),
        )


def align_to_size(feat: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    """Legacy bilinear resize helper (prefer AdaptiveCrossModalSpatialAlign)."""
    if feat.shape[-2:] == size_hw:
        return feat
    return F.interpolate(feat, size=size_hw, mode="bilinear", align_corners=False)
