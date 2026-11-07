from __future__ import annotations

import torch
from torch import nn

from mobilenet_enterprise.models.base import MobileNetBase
from mobilenet_enterprise.models.fusion.spatial_film import (
    AdaptiveCrossModalSpatialAlign,
    SpatialResidualBilateralFiLM,
    _ConvBNAct,
)
from mobilenet_enterprise.models.three_d.mobilenet3d_v3 import MobileNet3DV3
from mobilenet_enterprise.models.two_d.mobilenet_v3 import MobileNetV3


class RGBHSIMidFilmNet(MobileNetBase):
    """
    Hierarchical Bidirectional Feature Modulation (HBFM).

    Hybrid Spectral-Spatial Encoder
      -> Adaptive Cross-modal Spatial Alignment (ACSA)
      -> Residual Bidirectional FiLM (mid + late)
      -> Multi-scale Aggregation

    Ablation knobs (model.fusion.*):
      align_mode: acsa | bilinear
      modulate_direction: bilateral | uni_hsi2rgb | uni_rgb2hsi | none
      residual_mod / affine_mod / zero_init_modulators
      fusion_stages: both | mid | late
      aggregate_stages: match_fusion | both
    """

    def __init__(
        self,
        num_classes: int = 4,
        width_mult: float = 1.0,
        input_channels: int = 3,
        dropout: float = 0.35,
        variant: str = "v3_small",
        fusion_type: str = "hbfm",
        hidden_dim: int = 256,
        attention_heads: int = 4,
        attention_dropout: float | None = None,
        hsi_bands: int = 24,
        hsi_stem3d_channels: str | list[int] = "[16, 32]",
        hsi_stem3d_type: str = "standard",
        hsi_stem3d_max_spatial_pools: int | None = None,
        hsi_spectral_bins: int = 8,
        hsi_spectral_compressor: str = "adaptive_pool",
        hsi_spectral_attn_heads: int = 4,
        hsi_spectral_attn_layers: int = 1,
        hsi_spectral_attn_dropout: float = 0.1,
        hsi_variant: str = "small",
        hsi_tiny_spatial_enabled: bool = True,
        hsi_classifier_hidden: int | None = 192,
        hsi_backbone: str = "hybrid_3d2d",
        mid_channels: int = 40,
        late_channels: int = 96,
        num_fusion_stages: int = 2,
        block_attn_type: str = "se",
        align_mode: str = "acsa",
        modulate_direction: str = "bilateral",
        residual_mod: bool = True,
        affine_mod: bool = False,
        zero_init_modulators: bool = True,
        fusion_stages: str | None = None,
        aggregate_stages: str = "match_fusion",
    ) -> None:
        super().__init__()
        del fusion_type, attention_heads, attention_dropout, hidden_dim
        if str(hsi_backbone).lower() not in {"hybrid", "hybrid_3d2d", "3d2d", "3d+2d"}:
            raise ValueError("RGBHSIMidFilmNet requires hsi_backbone=hybrid_3d2d")

        # fusion_stages overrides legacy num_fusion_stages when provided.
        if fusion_stages is None:
            stages = "both" if int(num_fusion_stages) > 1 else "late"
        else:
            stages = str(fusion_stages).lower().strip() or "both"
        if stages not in {"both", "mid", "late"}:
            raise ValueError(f"Unsupported fusion_stages: {fusion_stages}")
        self.fusion_stages = stages
        self.use_mid_fusion = stages in {"both", "mid"}
        self.use_late_fusion = stages in {"both", "late"}
        aggregate = str(aggregate_stages).lower().strip() or "match_fusion"
        if aggregate not in {"match_fusion", "both"}:
            raise ValueError(f"Unsupported aggregate_stages: {aggregate_stages}")
        self.aggregate_stages = aggregate
        self.aggregate_mid = aggregate == "both" or stages in {"both", "mid"}
        self.aggregate_late = aggregate == "both" or stages in {"both", "late"}

        self.mid_channels = int(mid_channels)
        self.late_channels = int(late_channels)
        self.block_attn_type = str(block_attn_type).lower().strip() or "se"
        self.align_mode = str(align_mode).lower().strip() or "acsa"
        self.modulate_direction = str(modulate_direction).lower().strip() or "bilateral"
        self.residual_mod = bool(residual_mod)
        self.affine_mod = bool(affine_mod)
        self.zero_init_modulators = bool(zero_init_modulators)

        self.rgb_branch = MobileNetV3(
            num_classes=num_classes,
            width_mult=width_mult,
            input_channels=input_channels,
            dropout=dropout,
            variant=variant,
            block_attn_type=self.block_attn_type,
        )
        self.hsi_branch = MobileNet3DV3(
            num_classes=num_classes,
            width_mult=width_mult,
            input_channels=1,
            dropout=dropout,
            variant=hsi_variant,
            hsi_bands=hsi_bands,
            stem3d_channels=hsi_stem3d_channels,
            stem3d_type=hsi_stem3d_type,
            stem3d_max_spatial_pools=hsi_stem3d_max_spatial_pools,
            spectral_bins=hsi_spectral_bins,
            spectral_compressor=hsi_spectral_compressor,
            spectral_attn_heads=hsi_spectral_attn_heads,
            spectral_attn_layers=hsi_spectral_attn_layers,
            spectral_attn_dropout=hsi_spectral_attn_dropout,
            v3_variant=hsi_variant,
            tiny_spatial_enabled=hsi_tiny_spatial_enabled,
            classifier_hidden=hsi_classifier_hidden,
            block_attn_type=self.block_attn_type,
        )

        # MobileNetV3-Small: idx0..6 -> 14x14x40; full -> 7x7x576
        self.rgb_mid_end = 7
        # Hybrid tiny_spatial: features2d[0..3] -> 12x12x40; rest -> 6x6x576
        self.hsi_mid_end = 4

        rgb_mid_ch = 40
        hsi_mid_ch = 40
        rgb_late_ch = int(self.rgb_branch.classifier[0].in_features)
        hsi_late_ch = int(self.hsi_branch.classifier[0].in_features)

        film_kwargs = {
            "dropout": dropout,
            "direction": self.modulate_direction,
            "residual": self.residual_mod,
            "affine": self.affine_mod,
            "zero_init": self.zero_init_modulators,
        }

        self.rgb_mid_align = AdaptiveCrossModalSpatialAlign(
            rgb_mid_ch, self.mid_channels, align_mode=self.align_mode
        )
        self.hsi_mid_proj = _ConvBNAct(hsi_mid_ch, self.mid_channels)
        self.mid_film = SpatialResidualBilateralFiLM(self.mid_channels, **film_kwargs)
        self.hsi_mid_back = _ConvBNAct(self.mid_channels, hsi_mid_ch)

        self.rgb_late_align = AdaptiveCrossModalSpatialAlign(
            rgb_late_ch, self.late_channels, align_mode=self.align_mode
        )
        self.hsi_late_proj = _ConvBNAct(hsi_late_ch, self.late_channels)
        self.late_film = SpatialResidualBilateralFiLM(self.late_channels, **film_kwargs)

        self.pool = nn.AdaptiveAvgPool2d(1)
        head_in = 0
        if self.aggregate_late:
            head_in += self.late_channels * 2
        if self.aggregate_mid:
            head_in += self.mid_channels * 2
        cls_hidden = int(hsi_classifier_hidden) if hsi_classifier_hidden is not None else 192
        self.classifier = nn.Sequential(
            nn.Linear(head_in, cls_hidden),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden, num_classes),
        )
        self.fusion_type = "hbfm"
        # Keep legacy attribute for older summary code.
        self.num_fusion_stages = 2 if self.fusion_stages == "both" else 1

    def _rgb_mid_late(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = rgb
        for layer in self.rgb_branch.features[: self.rgb_mid_end]:
            x = layer(x)
        rgb_mid = x
        for layer in self.rgb_branch.features[self.rgb_mid_end :]:
            x = layer(x)
        return rgb_mid, x

    def _hsi_early(self, hsi: torch.Tensor) -> torch.Tensor:
        x = self.hsi_branch.encoder3d(hsi)
        return self.hsi_branch._reshape_3d_to_2d(x)

    def forward_features(self, inputs) -> torch.Tensor:
        if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            rgb, hsi = inputs
        elif isinstance(inputs, dict):
            rgb, hsi = inputs["rgb"], inputs["hsi"]
        else:
            raise TypeError("RGBHSIMidFilmNet expects (rgb, hsi) inputs.")

        rgb_mid, rgb_late = self._rgb_mid_late(rgb)
        hsi = self._hsi_early(hsi)
        for layer in self.hsi_branch.features2d[: self.hsi_mid_end]:
            hsi = layer(hsi)
        hsi_mid = hsi

        mid_hw = hsi_mid.shape[-2:]
        rgb_mid_a = self.rgb_mid_align(rgb_mid, mid_hw)
        hsi_mid_a = self.hsi_mid_proj(hsi_mid)
        if self.use_mid_fusion:
            rgb_mid_f, hsi_mid_f = self.mid_film(rgb_mid_a, hsi_mid_a)
        else:
            rgb_mid_f, hsi_mid_f = rgb_mid_a, hsi_mid_a

        hsi = self.hsi_mid_back(hsi_mid_f)
        for layer in self.hsi_branch.features2d[self.hsi_mid_end :]:
            hsi = layer(hsi)
        hsi_late = hsi

        late_hw = hsi_late.shape[-2:]
        rgb_late_a = self.rgb_late_align(rgb_late, late_hw)
        hsi_late_a = self.hsi_late_proj(hsi_late)
        if self.use_late_fusion:
            rgb_late_f, hsi_late_f = self.late_film(rgb_late_a, hsi_late_a)
        else:
            rgb_late_f, hsi_late_f = rgb_late_a, hsi_late_a

        parts: list[torch.Tensor] = []
        if self.aggregate_late:
            parts.extend(
                [
                    self.pool(rgb_late_f).flatten(1),
                    self.pool(hsi_late_f).flatten(1),
                ]
            )
        if self.aggregate_mid:
            parts.extend(
                [
                    self.pool(rgb_mid_f).flatten(1),
                    self.pool(hsi_mid_f).flatten(1),
                ]
            )
        return torch.cat(parts, dim=1)

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def forward(self, rgb, hsi: torch.Tensor | None = None) -> torch.Tensor:
        inputs = rgb if hsi is None else (rgb, hsi)
        return self.forward_head(self.forward_features(inputs))
