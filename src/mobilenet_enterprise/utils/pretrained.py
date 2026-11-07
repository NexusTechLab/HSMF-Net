from __future__ import annotations

from typing import Any

from torch import nn
from torchvision.models import mobilenet_v3_small


def _empty_result() -> dict[str, Any]:
    return {
        "loaded": False,
        "matched_layers": 0,
        "matched_params": 0,
        "matched_ratio": 0.0,
    }


def _load_matching_keys(model: nn.Module, src_state: dict[str, Any]) -> dict[str, Any]:
    target_state = model.state_dict()
    matched: dict[str, Any] = {}
    matched_target_keys: set[str] = set()
    matched_params = 0
    total_params = sum(int(v.numel()) for v in target_state.values())
    for key, value in src_state.items():
        candidate_keys = (
            key,
            f"backbone.{key}",
            f"rgb_branch.{key}",
            f"hsi_branch.{key}",
            f"hsi_branch.features2d.{key.removeprefix('features.')}",
        )
        for candidate in candidate_keys:
            if candidate in matched_target_keys:
                continue
            if candidate in target_state and target_state[candidate].shape == value.shape:
                matched[candidate] = value
                matched_target_keys.add(candidate)
                matched_params += int(value.numel())
                break
    if not matched:
        return _empty_result()
    model.load_state_dict(matched, strict=False)
    return {
        "loaded": True,
        "matched_layers": len(matched),
        "matched_params": matched_params,
        "matched_ratio": (matched_params / max(1, total_params)) * 100.0,
    }


def load_imagenet_pretrained_weights(model: nn.Module, cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Load torchvision ImageNet pretrained weights into current model.
    Returns load summary dict:
    {
      "loaded": bool,
      "matched_layers": int,
      "matched_params": int
    }
    """
    model_cfg = cfg.get("model", {})
    pretrained_cfg = cfg.get("pretrained", {})
    enabled = bool(pretrained_cfg.get("enabled", False))
    if not enabled:
        return _empty_result()

    source = str(pretrained_cfg.get("source", "torchvision")).lower()
    if source != "torchvision":
        return _empty_result()

    model_name = str(model_cfg.get("name", "")).lower()

    try:
        if model_name in {
            "v3",
            "v3_3d",
            "hsi_pure3d",
            "rgb_hsi_fusion",
            "rgb_hsi_attention",
            "rgb_hsi_mid_film",
        }:
            tv_model = mobilenet_v3_small(weights="DEFAULT")
            return _load_matching_keys(model, tv_model.state_dict())
        return _empty_result()
    except (RuntimeError, ValueError, TypeError, OSError):
        return _empty_result()

