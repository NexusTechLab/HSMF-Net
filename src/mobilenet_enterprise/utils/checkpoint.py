from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

from torch import nn


def clean_state_dict(state_dict: dict[str, object]) -> OrderedDict[str, object]:
    cleaned: OrderedDict[str, object] = OrderedDict()
    for key, value in state_dict.items():
        if key.endswith("total_ops") or key.endswith("total_params"):
            continue
        cleaned[key] = value
    return cleaned


def load_state_dict_with_report(
    model: nn.Module,
    state_dict: dict[str, Any],
    strict: bool = False,
) -> dict[str, Any]:
    target_state = model.state_dict()
    matched_keys = 0
    matched_params = 0
    total_params = sum(int(v.numel()) for v in target_state.values())
    for key, value in state_dict.items():
        if key in target_state and hasattr(value, "shape") and target_state[key].shape == value.shape:
            matched_keys += 1
            matched_params += int(value.numel())
    model.load_state_dict(state_dict, strict=strict)
    return {
        "matched_keys": matched_keys,
        "matched_params": matched_params,
        "matched_ratio": (matched_params / max(1, total_params)) * 100.0,
        "total_params": total_params,
    }


def resolve_checkpoint_path(cfg: dict[str, Any], ckpt_arg: str | None = None) -> Path:
    if ckpt_arg:
        path = Path(ckpt_arg)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    eval_cfg = cfg.get("eval", {})
    explicit_ckpt = eval_cfg.get("checkpoint_path", None)
    if explicit_ckpt:
        path = Path(str(explicit_ckpt))
        if not path.exists():
            raise FileNotFoundError(f"Configured checkpoint_path not found: {path}")
        return path

    preference = str(eval_cfg.get("checkpoint_preference", "best")).lower()
    exp_name = str(cfg.get("experiment_name", "default_exp"))
    output_dir = Path(str(cfg.get("output_dir", "artifacts")))
    ckpt_dir = output_dir / exp_name / "checkpoints"
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"
    ordered_candidates = (
        [best_path, last_path] if preference == "best" else [last_path, best_path]
    )
    for candidate in ordered_candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No checkpoint found under {ckpt_dir}. "
        "Pass --ckpt or configure eval.checkpoint_path."
    )
