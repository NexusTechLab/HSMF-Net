from __future__ import annotations

from typing import Any

from mobilenet_enterprise.models import (
    HybridSN,
    MobileNet3DV3,
    MobileNetV3,
    Pure3DHSINet,
    RGBHSIAttentionNet,
    RGBHSIFusionNet,
    RGBHSIMidFilmNet,
)

MODEL_REGISTRY = {
    "v3": MobileNetV3,
    "v3_3d": MobileNet3DV3,
    "hsi_pure3d": Pure3DHSINet,
    "hsi_hybridsn": HybridSN,
    "rgb_hsi_fusion": RGBHSIFusionNet,
    "rgb_hsi_attention": RGBHSIAttentionNet,
    "rgb_hsi_mid_film": RGBHSIMidFilmNet,
}

_FUSION_MODEL_NAMES = {
    "rgb_hsi_fusion",
    "rgb_hsi_attention",
    "rgb_hsi_mid_film",
}
_MID_FILM_MODEL_NAMES = {"rgb_hsi_mid_film"}


def build_model(cfg: dict[str, Any]):
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "v3").lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model name: {model_name}")
    model_cls = MODEL_REGISTRY[model_name]
    kwargs = {
        "num_classes": model_cfg.get("num_classes", 100),
        "width_mult": model_cfg.get("width_mult", 1.0),
        "input_channels": model_cfg.get("input_channels", 3),
        "dropout": model_cfg.get("dropout", 0.2),
    }
    if model_name in {"v3", "v3_3d", "hsi_pure3d"}:
        kwargs["variant"] = model_cfg.get("variant", "v3_small")
    if model_name == "v3_3d":
        kwargs["input_channels"] = model_cfg.get("input_channels", 1)
        kwargs.update(
            {
                "hsi_bands": model_cfg.get("hsi_bands", cfg.get("data", {}).get("hsi_bands", 454)),
                "stem3d_channels": model_cfg.get("stem3d_channels", "[16, 32]"),
                "classifier_hidden": model_cfg.get("classifier_hidden", None),
                "stem3d_type": model_cfg.get("stem3d_type", "standard"),
                "stem3d_max_spatial_pools": model_cfg.get("stem3d_max_spatial_pools", None),
                "spectral_bins": model_cfg.get("spectral_bins", 4),
                "spectral_compressor": model_cfg.get("spectral_compressor", "adaptive_pool"),
                "spectral_attn_heads": model_cfg.get("spectral_attn_heads", 4),
                "spectral_attn_layers": model_cfg.get("spectral_attn_layers", 1),
                "spectral_attn_dropout": model_cfg.get("spectral_attn_dropout", 0.1),
                "v3_variant": model_cfg.get("v3_variant", "small"),
                "tiny_spatial_enabled": model_cfg.get("tiny_spatial_enabled", False),
            }
        )
    if model_name == "hsi_pure3d":
        kwargs["input_channels"] = model_cfg.get("input_channels", 1)
        kwargs["hsi_bands"] = model_cfg.get("hsi_bands", cfg.get("data", {}).get("hsi_bands", 454))
    if model_name == "hsi_hybridsn":
        kwargs["input_channels"] = model_cfg.get("input_channels", 1)
        kwargs["hsi_bands"] = model_cfg.get("hsi_bands", cfg.get("data", {}).get("hsi_bands", 24))
        kwargs["image_size"] = model_cfg.get(
            "image_size", cfg.get("data", {}).get("image_size", 24)
        )
    if model_name in _FUSION_MODEL_NAMES:
        fusion_cfg = model_cfg.get("fusion", {})
        kwargs["variant"] = fusion_cfg.get("rgb_variant", model_cfg.get("variant", "v3_small"))
        default_fusion = {
            "rgb_hsi_attention": "cross_attention",
            "rgb_hsi_mid_film": "mid_bilateral_film",
        }.get(model_name, "concat")
        kwargs.update(
            {
                "fusion_type": fusion_cfg.get("type", default_fusion),
                "hidden_dim": fusion_cfg.get("hidden_dim", 256),
                "attention_heads": fusion_cfg.get("attention_heads", 4),
                "attention_dropout": fusion_cfg.get("attention_dropout", None),
                "hsi_bands": fusion_cfg.get("hsi_bands", cfg.get("data", {}).get("hsi_bands", 24)),
                "hsi_stem3d_channels": fusion_cfg.get("hsi_stem3d_channels", "[16, 32]"),
                "hsi_stem3d_type": fusion_cfg.get("hsi_stem3d_type", "standard"),
                "hsi_stem3d_max_spatial_pools": fusion_cfg.get(
                    "hsi_stem3d_max_spatial_pools", None
                ),
                "hsi_spectral_bins": fusion_cfg.get("hsi_spectral_bins", 8),
                "hsi_spectral_compressor": fusion_cfg.get(
                    "hsi_spectral_compressor", "adaptive_pool"
                ),
                "hsi_spectral_attn_heads": fusion_cfg.get("hsi_spectral_attn_heads", 4),
                "hsi_spectral_attn_layers": fusion_cfg.get("hsi_spectral_attn_layers", 1),
                "hsi_spectral_attn_dropout": fusion_cfg.get("hsi_spectral_attn_dropout", 0.1),
                "hsi_variant": fusion_cfg.get("hsi_variant", "small"),
                "hsi_tiny_spatial_enabled": fusion_cfg.get("hsi_tiny_spatial_enabled", True),
                "hsi_classifier_hidden": fusion_cfg.get("hsi_classifier_hidden", 192),
                "hsi_backbone": fusion_cfg.get("hsi_backbone", "hybrid_3d2d"),
            }
        )
        if model_name in _MID_FILM_MODEL_NAMES:
            kwargs.update(
                {
                    "mid_channels": fusion_cfg.get("mid_channels", 40),
                    "late_channels": fusion_cfg.get("late_channels", 96),
                    "num_fusion_stages": fusion_cfg.get("num_fusion_stages", 2),
                    "block_attn_type": fusion_cfg.get("block_attn_type", "se"),
                    "align_mode": fusion_cfg.get("align_mode", "acsa"),
                    "modulate_direction": fusion_cfg.get("modulate_direction", "bilateral"),
                    "residual_mod": fusion_cfg.get("residual_mod", True),
                    "affine_mod": fusion_cfg.get("affine_mod", False),
                    "zero_init_modulators": fusion_cfg.get("zero_init_modulators", True),
                    "fusion_stages": fusion_cfg.get("fusion_stages", None),
                    "aggregate_stages": fusion_cfg.get("aggregate_stages", "match_fusion"),
                }
            )
    return model_cls(**kwargs)
