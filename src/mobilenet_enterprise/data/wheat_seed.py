from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets.folder import default_loader


def _collect_class_samples(
    root: Path,
    exclude_classes: set[str] | None = None,
) -> tuple[list[str], list[tuple[Path, int]]]:
    excluded = exclude_classes or set()
    classes = sorted([d.name for d in root.iterdir() if d.is_dir() and d.name not in excluded])
    if not classes:
        raise FileNotFoundError(
            f"No class directories found under {root} after excluding {sorted(excluded)}."
        )
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    samples: list[tuple[Path, int]] = []
    for class_name in classes:
        class_dir = root / class_name
        hsi_dir = class_dir / "hsi"
        search_dir = hsi_dir if hsi_dir.exists() else class_dir
        for file_path in sorted(search_dir.rglob("*")):
            if file_path.suffix.lower() in exts:
                samples.append((file_path, class_to_idx[class_name]))
    if not samples:
        raise FileNotFoundError(
            f"No image samples found under {root}. "
            "Expected files like <root>/<class_name>/hsi/*.png."
        )
    return classes, samples


def _split_counts(n: int, val_ratio: float, test_ratio: float = 0.0) -> tuple[int, int, int]:
    """Return (n_train, n_val, n_test) for a single class."""
    val_ratio = max(0.0, float(val_ratio))
    test_ratio = max(0.0, float(test_ratio))
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(f"val_ratio + test_ratio must be < 1, got {val_ratio} + {test_ratio}")

    n_test = int(n * test_ratio) if test_ratio > 0 else 0
    n_val = int(n * val_ratio) if val_ratio > 0 else 0

    if n >= 3:
        if test_ratio > 0:
            n_test = max(1, n_test)
        if val_ratio > 0:
            n_val = max(1, n_val)
        while n_test + n_val >= n:
            if n_test >= n_val and n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
            else:
                break
    elif n == 2:
        # Prefer keeping one train sample; assign the other to val if requested.
        n_test = 1 if test_ratio > 0 and val_ratio <= 0 else 0
        n_val = 1 if val_ratio > 0 and n_test == 0 else (1 if val_ratio > 0 and test_ratio > 0 else 0)
        if n_test + n_val > 1:
            n_test, n_val = 0, 1
    else:
        n_test, n_val = 0, 0

    n_train = n - n_val - n_test
    return n_train, n_val, n_test


def _split_samples_per_class(
    samples: list[tuple[Path, int]],
    num_classes: int,
    val_ratio: float,
    seed: int,
    test_ratio: float = 0.0,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Stratified split per class into train / val / test.

    After shuffle: [test | val | train]. SPA/stats must use train only.
    """
    by_class: list[list[tuple[Path, int]]] = [[] for _ in range(num_classes)]
    for item in samples:
        by_class[item[1]].append(item)
    rng = random.Random(seed)
    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []
    test_samples: list[tuple[Path, int]] = []
    for class_samples in by_class:
        rng.shuffle(class_samples)
        n = len(class_samples)
        _, n_val, n_test = _split_counts(n, val_ratio=val_ratio, test_ratio=test_ratio)
        test_part = class_samples[:n_test]
        val_part = class_samples[n_test : n_test + n_val]
        train_part = class_samples[n_test + n_val :]
        test_samples.extend(test_part)
        val_samples.extend(val_part)
        train_samples.extend(train_part)
    return train_samples, val_samples, test_samples


class WheatSeedImageDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], classes: list[str], transform=None) -> None:
        self.samples = samples
        self.classes = classes
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = default_loader(str(path))
        if self.transform is not None:
            image = self.transform(image)
        return image, target


@dataclass
class WheatSeedDataModule:
    root: str
    image_size: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    randaugment: bool
    randaugment_num_ops: int
    randaugment_magnitude: int
    random_erasing_p: float
    val_ratio: float
    test_ratio: float
    split_seed: int
    exclude_classes: list[str]
    class_balanced_sampler: bool
    class_balance_power: float
    sampler_target_classes: list[str]
    sampler_target_multiplier: float
    sampler_replacement: bool

    def _train_transform(self):
        transform_list: list[Any] = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(self.image_size, padding=4),
        ]
        if self.randaugment:
            transform_list.append(
                transforms.RandAugment(
                    num_ops=self.randaugment_num_ops,
                    magnitude=self.randaugment_magnitude,
                )
            )
        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        if self.random_erasing_p > 0.0:
            transform_list.append(transforms.RandomErasing(p=self.random_erasing_p))
        return transforms.Compose(transform_list)

    def _eval_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def _build_splits(self):
        root = Path(self.root)
        classes, samples = _collect_class_samples(root, set(self.exclude_classes))
        train_samples, val_samples, test_samples = _split_samples_per_class(
            samples=samples,
            num_classes=len(classes),
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.split_seed,
        )
        return classes, train_samples, val_samples, test_samples

    def train_loader(self) -> DataLoader:
        classes, train_samples, _, _ = self._build_splits()
        dataset = WheatSeedImageDataset(
            samples=train_samples,
            classes=classes,
            transform=self._train_transform(),
        )
        sampler = None
        if self.class_balanced_sampler:
            counts = [0 for _ in classes]
            for _, target in train_samples:
                counts[target] += 1
            weights = []
            focus_idx = {classes.index(name) for name in self.sampler_target_classes if name in classes}
            for _, target in train_samples:
                base = 1.0 / max(1.0, float(counts[target]) ** self.class_balance_power)
                if target in focus_idx:
                    base *= max(1.0, self.sampler_target_multiplier)
                weights.append(base)
            sampler = WeightedRandomSampler(
                weights=torch.tensor(weights, dtype=torch.double),
                num_samples=len(weights),
                replacement=self.sampler_replacement,
            )
        loader_kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": sampler is None,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if sampler is not None:
            loader_kwargs["sampler"] = sampler
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = self.persistent_workers
            loader_kwargs["prefetch_factor"] = max(2, self.prefetch_factor)
        return DataLoader(dataset, **loader_kwargs)

    def val_loader(self) -> DataLoader:
        classes, _, val_samples, _ = self._build_splits()
        dataset = WheatSeedImageDataset(
            samples=val_samples,
            classes=classes,
            transform=self._eval_transform(),
        )
        loader_kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = self.persistent_workers
            loader_kwargs["prefetch_factor"] = max(2, self.prefetch_factor)
        return DataLoader(dataset, **loader_kwargs)

    def test_loader(self) -> DataLoader:
        classes, _, _, test_samples = self._build_splits()
        dataset = WheatSeedImageDataset(
            samples=test_samples,
            classes=classes,
            transform=self._eval_transform(),
        )
        loader_kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = self.persistent_workers
            loader_kwargs["prefetch_factor"] = max(2, self.prefetch_factor)
        return DataLoader(dataset, **loader_kwargs)

    def classes(self) -> list[str]:
        classes, _, _, _ = self._build_splits()
        return classes


def build_wheat_seed_datamodule(cfg: dict[str, Any]) -> WheatSeedDataModule:
    data_cfg = cfg.get("data", {})
    sampler_cfg = data_cfg.get("sampler", {})
    return WheatSeedDataModule(
        root=data_cfg.get("root", "data/wheat-seed"),
        image_size=int(data_cfg.get("image_size", 128)),
        batch_size=int(data_cfg.get("batch_size", 64)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=bool(data_cfg.get("persistent_workers", True)),
        prefetch_factor=int(data_cfg.get("prefetch_factor", 4)),
        randaugment=bool(data_cfg.get("randaugment", True)),
        randaugment_num_ops=int(data_cfg.get("randaugment_num_ops", 2)),
        randaugment_magnitude=int(data_cfg.get("randaugment_magnitude", 9)),
        random_erasing_p=float(data_cfg.get("random_erasing_p", 0.25)),
        val_ratio=float(data_cfg.get("val_ratio", 0.15)),
        test_ratio=float(data_cfg.get("test_ratio", 0.15)),
        split_seed=int(data_cfg.get("split_seed", 42)),
        exclude_classes=[str(name) for name in data_cfg.get("exclude_classes", [])],
        class_balanced_sampler=bool(sampler_cfg.get("class_balanced", False)),
        class_balance_power=float(sampler_cfg.get("class_balance_power", 1.0)),
        sampler_target_classes=[str(name) for name in sampler_cfg.get("target_classes", [])],
        sampler_target_multiplier=float(sampler_cfg.get("target_multiplier", 1.0)),
        sampler_replacement=bool(sampler_cfg.get("replacement", True)),
    )
