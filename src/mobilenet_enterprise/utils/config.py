from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        cfg: dict[str, Any] = yaml.safe_load(handle) or {}
    inherit = cfg.pop("inherit", None)
    if inherit:
        inherit_list = inherit if isinstance(inherit, list) else [inherit]
        merged_parent: dict[str, Any] = {}
        for inherit_item in inherit_list:
            parent_path = Path(inherit_item)
            if not parent_path.is_absolute():
                parent_path = (config_path.parent / parent_path).resolve()
            parent_cfg = load_config(parent_path)
            merged_parent = deep_merge(merged_parent, parent_cfg)
        cfg = deep_merge(merged_parent, cfg)
    return cfg
