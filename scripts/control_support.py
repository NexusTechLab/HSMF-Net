"""Fusion-control controls. Historical model/data implementations stay unchanged."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mobilenet_enterprise.data.wheat_seed import _split_samples_per_class
from mobilenet_enterprise.data.wheat_seed_fusion import WheatSeedFusionDataModule
from mobilenet_enterprise.models.fusion.rgb_hsi_fusion import RGBHSIFusionNet
from mobilenet_enterprise.registry import build_model

CLASSES = ["CX186", "XK129", "ZM22", "ZM379"]
COUNTS = [900, 1000, 1000, 900]
VARIANTS = ["concat24", "gated24", "cross_attention24", "no_modulation24", "full24"]
ALL_VARIANTS = VARIANTS + ["joint_attention24"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def collect_manifest(rgb_root, hsi_root):
    """Author-selected legacy input: same-stem image exports in the HSI directory."""
    rgb_root, hsi_root = Path(rgb_root).resolve(), Path(hsi_root).resolve()
    pairs, unused_rgb = [], {}
    for label, cultivar in enumerate(CLASSES):
        rgb_dir, hsi_dir = rgb_root / cultivar / "hsi", hsi_root / cultivar / "hsi"
        if not rgb_dir.is_dir() or not hsi_dir.is_dir():
            raise FileNotFoundError(f"Expected directories: {rgb_dir} and {hsi_dir}")
        rgb_by_stem = {}
        for path in sorted(rgb_dir.rglob("*")):
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                key = path.stem.casefold()
                if key in rgb_by_stem:
                    raise ValueError(f"Ambiguous RGB stem: {path}")
                rgb_by_stem[key] = path
        hdrs = sorted(p for p in hsi_dir.rglob("*") if p.suffix.lower() == ".hdr")
        seen = set()
        for hdr in hdrs:
            key = hdr.stem.casefold()
            if key in seen or key not in rgb_by_stem:
                raise ValueError(f"Duplicate HSI stem or missing HSI-directory image: {hdr}")
            seen.add(key)
            rgb, raw = rgb_by_stem[key], hdr.with_suffix("")
            if not raw.is_file():
                raise FileNotFoundError(raw)
            pairs.append(((rgb, hdr), label))
        unused_rgb[cultivar] = len(rgb_by_stem.keys() - seen)
        if len(hdrs) != COUNTS[label]:
            raise ValueError(f"{cultivar}: expected {COUNTS[label]} paired kernels, got {len(hdrs)}")
    splits = _split_samples_per_class(pairs, 4, val_ratio=0.15, test_ratio=0.15, seed=42)
    records = []
    for split, samples in zip(["train", "val", "test"], splits):
        for (rgb, hdr), label in samples:
            files = [rgb, hdr, hdr.with_suffix("")]
            records.append({
                "sample_id": f"{CLASSES[label]}/{hdr.stem}", "label": label,
                "cultivar": CLASSES[label], "split": split,
                "rgb_path": str(rgb), "hdr_path": str(hdr),
                "file_signatures": [{"path": str(p), "size": p.stat().st_size,
                                     "mtime_ns": p.stat().st_mtime_ns} for p in files],
            })
    assert [sum(r["split"] == s for r in records) for s in ["train", "val", "test"]] == [2660, 570, 570]
    assert len({r["sample_id"] for r in records}) == 3800
    return {"schema": 1, "classes": CLASSES, "split_seed": 42,
            "rgb_policy": "<rgb_root>/<cultivar>/hsi/; same-stem HSI-directory image exports (legacy input)",
            "pairing": "matching cultivar and unique filename stem; physical pairing requires author confirmation",
            "fingerprint_type": "ordered paths, sizes and modification times; not full dataset content hashes",
            "unused_rgb_without_hsi": unused_rgb, "records": records}


class ManifestFusionDataModule(WheatSeedFusionDataModule):
    def _build_splits(self):
        if not hasattr(self, "_control_splits"):
            manifest = json.loads(Path(self.cfg["data"]["pair_manifest"]).read_text("utf-8"))
            parts = []
            for split in ["train", "val", "test"]:
                parts.append([((Path(r["rgb_path"]), Path(r["hdr_path"])), r["label"])
                              for r in manifest["records"] if r["split"] == split])
            self._control_splits = (manifest["classes"], *parts)
        return self._control_splits

    def _hsi_helper(self):
        # All three loaders share one helper fitted on the manifest's train subset.
        if not hasattr(self, "_control_helper"):
            self._control_helper = super()._hsi_helper()
        return self._control_helper


class SpatialCrossAttention(nn.Module):
    """Reciprocal attention: Q from one spatial sequence, K/V from the other."""
    def __init__(self, dim=256, heads=4, dropout=0.1):
        super().__init__()
        self.rgb_norm, self.hsi_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.rgb_queries = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.hsi_queries = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.scales = nn.Parameter(torch.zeros(2))

    def forward(self, rgb, hsi):
        r, h = self.rgb_norm(rgb), self.hsi_norm(hsi)
        r_context = self.rgb_queries(r, h, h, need_weights=False)[0]
        h_context = self.hsi_queries(h, r, r, need_weights=False)[0]
        r_out = rgb + torch.tanh(self.scales[0]) * self.drop(r_context)
        h_out = hsi + torch.tanh(self.scales[1]) * self.drop(h_context)
        return torch.cat([r_out.mean(1), h_out.mean(1)], dim=1)


class SpatialCrossAttentionNet(RGBHSIFusionNet):
    def __init__(self, **kwargs):
        kwargs["fusion_type"] = "concat"
        super().__init__(**kwargs)
        self.attention = SpatialCrossAttention(kwargs.get("hidden_dim", 256),
                                               kwargs.get("attention_heads", 4), 0.1)
        self.fusion_type = "spatial_cross_attention"

    def forward_features(self, inputs):
        rgb, hsi = (inputs["rgb"], inputs["hsi"]) if isinstance(inputs, dict) else inputs
        r = self.rgb_branch.forward_features(rgb).flatten(2).transpose(1, 2)
        h = self.hsi_branch.forward_features(hsi).flatten(2).transpose(1, 2)
        return self.attention(self.rgb_proj(r), self.hsi_proj(h))


def make_model(cfg):
    if cfg["reviewer6"]["variant"] == "cross_attention24":
        f, m = cfg["model"]["fusion"], cfg["model"]
        return SpatialCrossAttentionNet(
            num_classes=4, width_mult=m["width_mult"], dropout=m["dropout"],
            input_channels=3, variant=f["rgb_variant"], hidden_dim=256,
            hsi_bands=24, hsi_stem3d_channels=f["hsi_stem3d_channels"],
            hsi_spectral_bins=f["hsi_spectral_bins"],
            hsi_tiny_spatial_enabled=f["hsi_tiny_spatial_enabled"],
            hsi_classifier_hidden=f["hsi_classifier_hidden"], hsi_backbone=f["hsi_backbone"],
        )
    if cfg["reviewer6"]["variant"] == "no_modulation24":
        # Preserve initial weights of every retained module, including the head.
        # Merely omitting generators would otherwise change subsequent RNG draws.
        rng = torch.random.get_rng_state()
        full_cfg = copy.deepcopy(cfg)
        full_cfg["model"]["fusion"]["modulate_direction"] = "bilateral"
        reference = build_model(full_cfg)
        torch.random.set_rng_state(rng)
        control = build_model(cfg)
        retained = control.state_dict()
        reference_state = reference.state_dict()
        control.load_state_dict({k: reference_state[k] for k in retained}, strict=True)
        return control
    return build_model(cfg)


def variant_config(base, variant, seed, output, manifest_path, common_cache, band_path):
    cfg = copy.deepcopy(base)
    cfg.update(seed=seed, output_dir=str(output), experiment_name=f"control_{variant}_seed{seed}")
    cfg["reviewer6"] = {"variant": variant, "status": "prospective; no result yet"}
    d = cfg["data"]
    d.update(pair_manifest=str(manifest_path), split_seed=42, hsi_bands=24,
             rgb_source="hsi_directory_export")
    d["band_selection"]["cache_path"] = str(band_path)
    d["normalization"]["stats_cache_path"] = str(common_cache / "train_band_stats.npz")
    d["preprocessing_cache"]["root"] = str(common_cache / "tensor_cache")
    f = cfg["model"]["fusion"]
    f["hsi_bands"] = 24
    if variant in {"full24", "no_modulation24"}:
        f.update(align_mode="acsa", fusion_stages="both", aggregate_stages="both",
                 modulate_direction="none" if variant == "no_modulation24" else "bilateral")
    else:
        cfg["model"]["name"] = "rgb_hsi_fusion"
        # Strip HBFM-only options; shared encoder and training settings stay fixed.
        for key in ["num_fusion_stages", "mid_channels", "late_channels"]:
            f.pop(key, None)
        f.update(type={"concat24": "concat", "gated24": "gated",
                       "joint_attention24": "cross_attention", "cross_attention24": "spatial_cross_attention"}[variant],
                 hidden_dim=256, attention_heads=4, attention_dropout=0.1)
    return cfg


def validate_bands(path):
    bands = np.load(path)
    if bands.ndim != 1 or len(bands) != 24 or len(set(bands.tolist())) != 24:
        raise ValueError("SPA file must contain exactly 24 distinct integer indices")
    if not np.issubdtype(bands.dtype, np.integer) or np.any(np.diff(bands) <= 0):
        raise ValueError("SPA indices must be increasing integers")
    if bands.min() < 10 or bands.max() >= 439:
        raise ValueError("SPA indices must belong to the stated retained 454-band range [10, 438]")
    return bands
