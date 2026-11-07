from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mobilenet_enterprise.data.wheat_seed import build_wheat_seed_datamodule
from mobilenet_enterprise.data.wheat_seed_fusion import build_wheat_seed_fusion_datamodule
from mobilenet_enterprise.data.wheat_seed_hsi import build_wheat_seed_hsi_datamodule

DATA_REGISTRY: dict[str, Callable[[dict[str, Any]], Any]] = {
    "wheat_seed": build_wheat_seed_datamodule,
    "wheat_seed_fusion": build_wheat_seed_fusion_datamodule,
    "wheat_seed_hsi": build_wheat_seed_hsi_datamodule,
}


def build_datamodule(cfg: dict[str, Any]):
    data_cfg = cfg.get("data", {})
    data_name = str(data_cfg.get("name", "wheat_seed")).lower()
    if data_name not in DATA_REGISTRY:
        raise ValueError(f"Unsupported data name: {data_name}")
    return DATA_REGISTRY[data_name](cfg)
