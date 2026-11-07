from __future__ import annotations

import torch
from torch import nn

from mobilenet_enterprise.models.base import MobileNetBase
from mobilenet_enterprise.models.fusion.attention import CrossModalAttention
from mobilenet_enterprise.models.three_d.mobilenet3d_v3 import MobileNet3DV3
from mobilenet_enterprise.models.three_d.mobilenet_v3_pure3d import MobileNetV3Pure3D
from mobilenet_enterprise.models.two_d.mobilenet_v3 import MobileNetV3


class RGBHSIFusionNet(MobileNetBase):
    def __init__(
        self,
        num_classes: int = 4,
        width_mult: float = 1.0,
        input_channels: int = 3,
        dropout: float = 0.35,
        variant: str = "v3_small",
        fusion_type: str = "concat",
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
    ) -> None:
        super().__init__()
        self.fusion_type = str(fusion_type).lower()
        self.hsi_backbone = str(hsi_backbone).lower()
        self.rgb_branch = MobileNetV3(
            num_classes=num_classes,
            width_mult=width_mult,
            input_channels=input_channels,
            dropout=dropout,
            variant=variant,
        )
        if self.hsi_backbone in {"pure3d", "3d"}:
            self.hsi_branch = MobileNetV3Pure3D(
                num_classes=num_classes,
                width_mult=width_mult,
                input_channels=1,
                dropout=dropout,
                variant=hsi_variant,
                hsi_bands=hsi_bands,
            )
        elif self.hsi_backbone in {"hybrid", "hybrid_3d2d", "3d2d", "3d+2d"}:
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
            )
        else:
            raise ValueError(f"Unsupported hsi_backbone: {hsi_backbone}")
        rgb_dim = int(self.rgb_branch.classifier[0].in_features)
        hsi_dim = int(self.hsi_branch.classifier[0].in_features)
        hidden_dim = int(hidden_dim)
        self.rgb_pool = nn.AdaptiveAvgPool2d(1)
        self.hsi_pool = nn.AdaptiveAvgPool3d(1) if self.hsi_backbone in {"pure3d", "3d"} else nn.AdaptiveAvgPool2d(1)
        self.rgb_proj = nn.Sequential(nn.Linear(rgb_dim, hidden_dim), nn.Hardswish(inplace=True))
        self.hsi_proj = nn.Sequential(nn.Linear(hsi_dim, hidden_dim), nn.Hardswish(inplace=True))

        if self.fusion_type == "cross_attention":
            self.attention = CrossModalAttention(
                hidden_dim=hidden_dim,
                num_heads=attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
            )
            fused_dim = hidden_dim * 2
        elif self.fusion_type == "gated":
            self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
            fused_dim = hidden_dim
        elif self.fusion_type == "concat":
            fused_dim = hidden_dim * 2
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _pool_project(self, feature_map: torch.Tensor, pool: nn.Module, proj: nn.Module) -> torch.Tensor:
        return proj(pool(feature_map).flatten(1))

    def forward_features(self, inputs) -> torch.Tensor:
        if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            rgb, hsi = inputs
        elif isinstance(inputs, dict):
            rgb, hsi = inputs["rgb"], inputs["hsi"]
        else:
            raise TypeError("RGBHSIFusionNet expects (rgb, hsi) inputs.")
        # Shared trunk path for all fusion heads: branch → GAP → modality projection.
        rgb_feat = self._pool_project(self.rgb_branch.forward_features(rgb), self.rgb_pool, self.rgb_proj)
        hsi_feat = self._pool_project(self.hsi_branch.forward_features(hsi), self.hsi_pool, self.hsi_proj)
        if self.fusion_type == "cross_attention":
            return self.attention(rgb_feat, hsi_feat)
        if self.fusion_type == "gated":
            gate = self.gate(torch.cat([rgb_feat, hsi_feat], dim=1))
            return gate * rgb_feat + (1.0 - gate) * hsi_feat
        return torch.cat([rgb_feat, hsi_feat], dim=1)

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def forward(self, rgb, hsi: torch.Tensor | None = None) -> torch.Tensor:
        inputs = rgb if hsi is None else (rgb, hsi)
        return self.forward_head(self.forward_features(inputs))


class RGBHSIAttentionNet(RGBHSIFusionNet):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["fusion_type"] = "cross_attention"
        super().__init__(*args, **kwargs)
