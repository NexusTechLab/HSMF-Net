from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from PIL import ImageOps

from mobilenet_enterprise.data.wheat_seed import _split_samples_per_class
from mobilenet_enterprise.data.wheat_seed_hsi import (
    WheatSeedHSIDataModule,
    _cache_key_for_hsi_tensor,
    _crop_roi,
    _foreground_mask_hw,
    _normalize_cube_foreground,
    _read_hsi_cube,
    _resize_cube_spatial,
    _resolve_hsi_tensor_cache_path,
    _select_and_pad_bands,
)


PairSample = tuple[tuple[Path, Path], int]


def _sample_key(path: Path, mode: str) -> str:
    if mode == "relative":
        return str(path.with_suffix(""))
    return path.stem


def _collect_paired_samples(
    rgb_root: Path,
    hsi_root: Path,
    exclude_classes: list[str],
    pair_key: str,
) -> tuple[list[str], list[PairSample]]:
    excluded = {name.strip() for name in exclude_classes if name.strip()}
    classes = sorted([d.name for d in hsi_root.iterdir() if d.is_dir() and d.name not in excluded])
    if not classes:
        raise FileNotFoundError(f"No class directories found under {hsi_root}.")

    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    image_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    samples: list[PairSample] = []
    for class_name in classes:
        rgb_class_dir = rgb_root / class_name
        hsi_class_dir = hsi_root / class_name
        rgb_search = rgb_class_dir / "hsi" if (rgb_class_dir / "hsi").exists() else rgb_class_dir
        hsi_search = hsi_class_dir / "hsi" if (hsi_class_dir / "hsi").exists() else hsi_class_dir

        rgb_by_key: dict[str, Path] = {}
        if rgb_search.exists():
            for rgb_path in sorted(rgb_search.rglob("*")):
                if rgb_path.suffix.lower() in image_exts:
                    rgb_by_key.setdefault(_sample_key(rgb_path, pair_key), rgb_path)

        for hdr_path in sorted(hsi_search.rglob("*.HDR")):
            key = _sample_key(hdr_path, pair_key)
            rgb_path = rgb_by_key.get(key)
            if rgb_path is not None:
                samples.append(((rgb_path, hdr_path), class_to_idx[class_name]))

    if not samples:
        raise FileNotFoundError(
            "No paired RGB/HSI samples found. Expected matching stems like "
            "<root>/<class>/hsi/sample.png and <root>/<class>/hsi/sample.HDR."
        )
    return classes, samples


class WheatSeedFusionDataset(Dataset):
    def __init__(
        self,
        samples: list[PairSample],
        classes: list[str],
        rgb_transform,
        image_size: int,
        hsi_image_size: int,
        hsi_bands: int,
        roi_crop_enabled: bool,
        roi_threshold_percentile: float,
        roi_margin: int,
        selected_bands: np.ndarray | None,
        train: bool,
        sync_spatial_flip_prob: float,
        hsi_augment_cfg: dict[str, Any],
        band_mean: np.ndarray,
        band_std: np.ndarray,
        tensor_cache_enabled: bool = False,
        tensor_cache_root: str = "data/processed/hsi_tensor_cache",
    ) -> None:
        self.samples = samples
        self.classes = classes
        self.rgb_transform = rgb_transform
        self.image_size = int(image_size)
        self.hsi_image_size = int(hsi_image_size)
        self.hsi_bands = int(hsi_bands)
        self.roi_crop_enabled = bool(roi_crop_enabled)
        self.roi_threshold_percentile = float(roi_threshold_percentile)
        self.roi_margin = int(roi_margin)
        self.selected_bands = None if selected_bands is None else np.asarray(selected_bands, dtype=np.int32)
        self.train = bool(train)
        self.sync_spatial_flip_prob = float(sync_spatial_flip_prob)
        self.hsi_augment_cfg = hsi_augment_cfg
        self.band_mean = np.asarray(band_mean, dtype=np.float32).reshape(-1)
        self.band_std = np.maximum(np.asarray(band_std, dtype=np.float32).reshape(-1), 1e-6)
        self.tensor_cache_enabled = bool(tensor_cache_enabled)
        self.tensor_cache_root = str(tensor_cache_root)

    def __len__(self) -> int:
        return len(self.samples)

    def _augment_hsi_spectral(self, cube: np.ndarray) -> np.ndarray:
        if not self.train or not bool(self.hsi_augment_cfg.get("enabled", False)):
            return cube
        out = cube.copy().astype(np.float32, copy=False)
        roll_max = int(self.hsi_augment_cfg.get("spatial_roll_max", 0))
        if roll_max > 0:
            shift_h = random.randint(-roll_max, roll_max)
            shift_w = random.randint(-roll_max, roll_max)
            if shift_h != 0 or shift_w != 0:
                out = np.roll(out, shift=(shift_h, shift_w), axis=(0, 1))
        mask = _foreground_mask_hw(out)
        if mask.any():
            fg = out[mask]
            scale_min = float(self.hsi_augment_cfg.get("spectral_scale_min", 1.0))
            scale_max = float(self.hsi_augment_cfg.get("spectral_scale_max", 1.0))
            if scale_max > 0 and scale_max != scale_min:
                fg = fg * random.uniform(scale_min, scale_max)
            noise_std = float(self.hsi_augment_cfg.get("spectral_noise_std", 0.0))
            if noise_std > 0:
                fg = fg + np.random.normal(0.0, noise_std, size=fg.shape).astype(np.float32)
            fg = np.clip(fg, 0.0, 1.0)
            out = np.zeros_like(out, dtype=np.float32)
            out[mask] = fg
        return out.astype(np.float32, copy=False)

    def _apply_band_dropout_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if not self.train or not bool(self.hsi_augment_cfg.get("enabled", False)):
            return x
        dropout_prob = float(self.hsi_augment_cfg.get("band_dropout_prob", 0.0))
        dropout_fraction = float(self.hsi_augment_cfg.get("band_dropout_max_fraction", 0.0))
        if dropout_prob <= 0 or dropout_fraction <= 0 or random.random() >= dropout_prob:
            return x
        n_bands = x.shape[1]
        max_drop = max(1, int(n_bands * dropout_fraction))
        drop_idx = torch.as_tensor(
            np.random.choice(n_bands, size=random.randint(1, max_drop), replace=False),
            dtype=torch.long,
            device=x.device,
        )
        out = x.clone()
        out[:, drop_idx, :, :] = 0.0
        return out

    def _to_hsi_tensor(
        self,
        cube: np.ndarray,
        *,
        apply_augmentation: bool = True,
        bands_already_selected: bool = False,
    ) -> torch.Tensor:
        if not bands_already_selected:
            cube = _select_and_pad_bands(cube, self.selected_bands, self.hsi_bands)
        elif cube.shape[2] != self.hsi_bands:
            cube = _select_and_pad_bands(cube, None, self.hsi_bands)
        if apply_augmentation:
            cube = self._augment_hsi_spectral(cube)
        cube = _resize_cube_spatial(cube, self.hsi_image_size, self.hsi_image_size)
        cube = _normalize_cube_foreground(cube, self.band_mean, self.band_std)
        x = torch.from_numpy(np.ascontiguousarray(cube)).permute(2, 0, 1).unsqueeze(0)
        if apply_augmentation:
            x = self._apply_band_dropout_tensor(x)
        return x

    def _cache_path(self, hdr_path: Path) -> Path:
        cache_name = _cache_key_for_hsi_tensor(
            hdr_path,
            image_size=self.hsi_image_size,
            hsi_bands=self.hsi_bands,
            selected_bands=self.selected_bands,
            roi_crop_enabled=self.roi_crop_enabled,
            roi_threshold_percentile=self.roi_threshold_percentile,
            roi_margin=self.roi_margin,
            mean=self.band_mean,
            std=self.band_std,
        )
        return _resolve_hsi_tensor_cache_path(self.tensor_cache_root, hdr_path, cache_name)

    def _load_or_build_cached_hsi_cube(self, hdr_path: Path) -> np.ndarray:
        cache_path = self._cache_path(hdr_path)
        if cache_path.exists():
            data = np.load(cache_path)
            if "cube" in data:
                return data["cube"].astype(np.float32, copy=False)
        cube = _read_hsi_cube(hdr_path)
        if self.roi_crop_enabled:
            cropped = _crop_roi(
                cube,
                threshold_percentile=self.roi_threshold_percentile,
                margin=self.roi_margin,
            )
            if cropped.size > 0:
                cube = cropped
        cube = _select_and_pad_bands(cube, self.selected_bands, self.hsi_bands)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, cube=cube.astype(np.float32, copy=False))
        return cube.astype(np.float32, copy=False)

    def __getitem__(self, index: int):
        (rgb_path, hdr_path), target = self.samples[index]
        rgb = default_loader(str(rgb_path))
        if self.tensor_cache_enabled:
            cube = self._load_or_build_cached_hsi_cube(hdr_path)
            bands_already_selected = True
        else:
            cube = _read_hsi_cube(hdr_path)
            if self.roi_crop_enabled:
                cropped = _crop_roi(
                    cube,
                    threshold_percentile=self.roi_threshold_percentile,
                    margin=self.roi_margin,
                )
                if cropped.size > 0:
                    cube = cropped
            bands_already_selected = False

        if self.train and self.sync_spatial_flip_prob > 0:
            if random.random() < self.sync_spatial_flip_prob:
                rgb = ImageOps.mirror(rgb)
                cube = np.flip(cube, axis=1)
            if random.random() < self.sync_spatial_flip_prob:
                rgb = ImageOps.flip(rgb)
                cube = np.flip(cube, axis=0)

        rgb_tensor = self.rgb_transform(rgb)
        hsi_tensor = self._to_hsi_tensor(
            cube,
            apply_augmentation=True,
            bands_already_selected=bands_already_selected,
        )
        return (rgb_tensor, hsi_tensor), target


@dataclass
class WheatSeedFusionDataModule:
    cfg: dict[str, Any]

    def _data_cfg(self) -> dict[str, Any]:
        return self.cfg.get("data", {})

    def _build_splits(self):
        data_cfg = self._data_cfg()
        rgb_root = Path(data_cfg.get("rgb_root", data_cfg.get("root", "data/wheat-seed")))
        hsi_root = Path(data_cfg.get("hsi_root", data_cfg.get("root", "data/wheat-seed")))
        classes, samples = _collect_paired_samples(
            rgb_root=rgb_root,
            hsi_root=hsi_root,
            exclude_classes=[str(name) for name in data_cfg.get("exclude_classes", [])],
            pair_key=str(data_cfg.get("pair_key", "stem")),
        )
        train_samples, val_samples, test_samples = _split_samples_per_class(
            samples=samples,
            num_classes=len(classes),
            val_ratio=float(data_cfg.get("val_ratio", 0.15)),
            test_ratio=float(data_cfg.get("test_ratio", 0.15)),
            seed=int(data_cfg.get("split_seed", 42)),
        )
        return classes, train_samples, val_samples, test_samples

    def _rgb_transform(self, train: bool):
        data_cfg = self._data_cfg()
        rgb_aug_cfg = data_cfg.get("rgb_augmentation", {})
        image_size = int(data_cfg.get("image_size", 224))
        ops: list[Any] = [transforms.Resize((image_size, image_size))]
        if train and bool(rgb_aug_cfg.get("randaugment", data_cfg.get("randaugment", False))):
            ops.append(
                transforms.RandAugment(
                    num_ops=int(rgb_aug_cfg.get("randaugment_num_ops", 2)),
                    magnitude=int(rgb_aug_cfg.get("randaugment_magnitude", 9)),
                )
            )
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        erase_p = float(rgb_aug_cfg.get("random_erasing_p", 0.0)) if train else 0.0
        if erase_p > 0:
            ops.append(transforms.RandomErasing(p=erase_p))
        return transforms.Compose(ops)

    def _hsi_helper(self) -> WheatSeedHSIDataModule:
        helper_cfg = dict(self.cfg)
        helper_cfg["data"] = dict(self._data_cfg())
        helper_cfg["data"]["root"] = helper_cfg["data"].get("hsi_root", helper_cfg["data"].get("root", "data/wheat-seed"))
        helper_cfg["data"]["image_size"] = helper_cfg["data"].get(
            "hsi_image_size",
            helper_cfg["data"].get("image_size", 24),
        )
        from mobilenet_enterprise.data.wheat_seed_hsi import build_wheat_seed_hsi_datamodule

        return build_wheat_seed_hsi_datamodule(helper_cfg)

    def _dataset(self, samples: list[PairSample], classes: list[str], train: bool) -> WheatSeedFusionDataset:
        data_cfg = self._data_cfg()
        hsi_helper = self._hsi_helper()
        hsi_train_samples = [(hdr_path, target) for (_, hdr_path), target in samples]
        selected_bands = hsi_helper._compute_selected_bands(hsi_train_samples)
        band_mean, band_std = hsi_helper._compute_band_stats(hsi_train_samples, selected_bands)
        return WheatSeedFusionDataset(
            samples=samples,
            classes=classes,
            rgb_transform=self._rgb_transform(train=train),
            image_size=int(data_cfg.get("image_size", 224)),
            hsi_image_size=int(data_cfg.get("hsi_image_size", data_cfg.get("image_size", 24))),
            hsi_bands=int(data_cfg.get("hsi_bands", self.cfg.get("model", {}).get("hsi_bands", 24))),
            roi_crop_enabled=bool(data_cfg.get("roi_crop_enabled", True)),
            roi_threshold_percentile=float(data_cfg.get("roi_threshold_percentile", 80.0)),
            roi_margin=int(data_cfg.get("roi_margin", 1)),
            selected_bands=selected_bands,
            train=train,
            sync_spatial_flip_prob=float(data_cfg.get("sync_spatial_flip_prob", 0.0)),
            hsi_augment_cfg=data_cfg.get("hsi_augmentation", {}),
            band_mean=band_mean,
            band_std=band_std,
            tensor_cache_enabled=bool(data_cfg.get("preprocessing_cache", {}).get("enabled", False)),
            tensor_cache_root=str(
                data_cfg.get("preprocessing_cache", {}).get("root", "data/processed/hsi_tensor_cache")
            ),
        )

    def _loader_kwargs(self, train: bool) -> dict[str, Any]:
        data_cfg = self._data_cfg()
        kwargs: dict[str, Any] = {
            "batch_size": int(data_cfg.get("batch_size", 8)),
            "shuffle": train,
            "num_workers": int(data_cfg.get("num_workers", 2)),
            "pin_memory": bool(data_cfg.get("pin_memory", True)),
        }
        if kwargs["num_workers"] > 0:
            kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
            kwargs["prefetch_factor"] = max(2, int(data_cfg.get("prefetch_factor", 2)))
        return kwargs

    def train_loader(self) -> DataLoader:
        classes, train_samples, _, _ = self._build_splits()
        return DataLoader(self._dataset(train_samples, classes, train=True), **self._loader_kwargs(train=True))

    def val_loader(self) -> DataLoader:
        classes, train_samples, val_samples, _ = self._build_splits()
        hsi_helper = self._hsi_helper()
        hsi_train_samples = [(hdr_path, target) for (_, hdr_path), target in train_samples]
        selected_bands = hsi_helper._compute_selected_bands(hsi_train_samples)
        band_mean, band_std = hsi_helper._compute_band_stats(hsi_train_samples, selected_bands)
        data_cfg = self._data_cfg()
        dataset = WheatSeedFusionDataset(
            samples=val_samples,
            classes=classes,
            rgb_transform=self._rgb_transform(train=False),
            image_size=int(data_cfg.get("image_size", 224)),
            hsi_image_size=int(data_cfg.get("hsi_image_size", data_cfg.get("image_size", 24))),
            hsi_bands=int(data_cfg.get("hsi_bands", self.cfg.get("model", {}).get("hsi_bands", 24))),
            roi_crop_enabled=bool(data_cfg.get("roi_crop_enabled", False)),
            roi_threshold_percentile=float(data_cfg.get("roi_threshold_percentile", 80.0)),
            roi_margin=int(data_cfg.get("roi_margin", 1)),
            selected_bands=selected_bands,
            train=False,
            sync_spatial_flip_prob=0.0,
            hsi_augment_cfg={},
            band_mean=band_mean,
            band_std=band_std,
            tensor_cache_enabled=bool(data_cfg.get("preprocessing_cache", {}).get("enabled", False)),
            tensor_cache_root=str(
                data_cfg.get("preprocessing_cache", {}).get("root", "data/processed/hsi_tensor_cache")
            ),
        )
        return DataLoader(dataset, **self._loader_kwargs(train=False))

    def test_loader(self) -> DataLoader:
        classes, train_samples, _, test_samples = self._build_splits()
        hsi_helper = self._hsi_helper()
        hsi_train_samples = [(hdr_path, target) for (_, hdr_path), target in train_samples]
        selected_bands = hsi_helper._compute_selected_bands(hsi_train_samples)
        band_mean, band_std = hsi_helper._compute_band_stats(hsi_train_samples, selected_bands)
        data_cfg = self._data_cfg()
        dataset = WheatSeedFusionDataset(
            samples=test_samples,
            classes=classes,
            rgb_transform=self._rgb_transform(train=False),
            image_size=int(data_cfg.get("image_size", 224)),
            hsi_image_size=int(data_cfg.get("hsi_image_size", data_cfg.get("image_size", 24))),
            hsi_bands=int(data_cfg.get("hsi_bands", self.cfg.get("model", {}).get("hsi_bands", 24))),
            roi_crop_enabled=bool(data_cfg.get("roi_crop_enabled", False)),
            roi_threshold_percentile=float(data_cfg.get("roi_threshold_percentile", 80.0)),
            roi_margin=int(data_cfg.get("roi_margin", 1)),
            selected_bands=selected_bands,
            train=False,
            sync_spatial_flip_prob=0.0,
            hsi_augment_cfg={},
            band_mean=band_mean,
            band_std=band_std,
            tensor_cache_enabled=bool(data_cfg.get("preprocessing_cache", {}).get("enabled", False)),
            tensor_cache_root=str(
                data_cfg.get("preprocessing_cache", {}).get("root", "data/processed/hsi_tensor_cache")
            ),
        )
        return DataLoader(dataset, **self._loader_kwargs(train=False))

    def classes(self) -> list[str]:
        classes, _, _, _ = self._build_splits()
        return classes


def build_wheat_seed_fusion_datamodule(cfg: dict[str, Any]) -> WheatSeedFusionDataModule:
    return WheatSeedFusionDataModule(cfg=cfg)
