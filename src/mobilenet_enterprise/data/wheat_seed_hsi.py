from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import re
import hashlib
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def _parse_hdr(hdr_path: Path) -> dict[str, Any]:
    text = hdr_path.read_text(encoding="utf-8", errors="ignore")
    keys = ["samples", "lines", "bands", "data type", "interleave", "byte order"]
    result: dict[str, Any] = {}
    for key in keys:
        match = re.search(rf"{re.escape(key)}\s*=\s*([^\r\n]+)", text, flags=re.IGNORECASE)
        if match:
            result[key.lower()] = match.group(1).strip().strip("{}")
    return result


def _parse_hdr_wavelengths(hdr_path: Path) -> np.ndarray | None:
    text = hdr_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"wavelength\s*=\s*\{([^}]+)\}", text, flags=re.IGNORECASE)
    if not match:
        return None
    values = [float(item.strip()) for item in match.group(1).split(",") if item.strip()]
    if not values:
        return None
    return np.asarray(values, dtype=np.float64)


def load_wavelength_table(root: Path) -> np.ndarray | None:
    for hdr_path in sorted(root.rglob("*.HDR")):
        wavelengths = _parse_hdr_wavelengths(hdr_path)
        if wavelengths is not None and wavelengths.size > 0:
            return wavelengths
    for hdr_path in sorted(root.rglob("*.hdr")):
        wavelengths = _parse_hdr_wavelengths(hdr_path)
        if wavelengths is not None and wavelengths.size > 0:
            return wavelengths
    return None


def _dtype_from_envi(data_type: int) -> np.dtype:
    mapping = {
        1: np.uint8,
        2: np.int16,
        3: np.int32,
        4: np.float32,
        5: np.float64,
        12: np.uint16,
        13: np.uint32,
        14: np.int64,
        15: np.uint64,
    }
    return mapping.get(data_type, np.float32)


def _read_hsi_cube(hdr_path: Path) -> np.ndarray:
    meta = _parse_hdr(hdr_path)
    samples = int(meta.get("samples", 0))
    lines = int(meta.get("lines", 0))
    bands = int(meta.get("bands", 0))
    data_type = int(meta.get("data type", 4))
    interleave = str(meta.get("interleave", "bil")).lower()
    raw_path = hdr_path.with_suffix("")
    if not raw_path.exists():
        raise FileNotFoundError(f"HSI raw file missing for HDR: {hdr_path}")
    dtype = _dtype_from_envi(data_type)
    arr = np.fromfile(raw_path, dtype=dtype)
    expected = samples * lines * bands
    if expected <= 0 or arr.size != expected:
        raise ValueError(f"Invalid HSI shape or size mismatch: {hdr_path}, expected={expected}, got={arr.size}")
    if interleave == "bil":
        cube = arr.reshape(lines, bands, samples).transpose(0, 2, 1)  # (H,W,B)
    elif interleave == "bsq":
        cube = arr.reshape(bands, lines, samples).transpose(1, 2, 0)
    elif interleave == "bip":
        cube = arr.reshape(lines, samples, bands)
    else:
        raise ValueError(f"Unsupported interleave={interleave} in {hdr_path}")
    return cube.astype(np.float32)


def _collect_hdr_samples(
    root: Path, exclude_classes: list[str] | None = None
) -> tuple[list[str], list[tuple[Path, int]]]:
    excluded = {name.strip() for name in (exclude_classes or []) if name.strip()}
    classes = sorted([d.name for d in root.iterdir() if d.is_dir() and d.name not in excluded])
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    samples: list[tuple[Path, int]] = []
    for class_name in classes:
        class_dir = root / class_name
        hsi_dir = class_dir / "hsi"
        search_dir = hsi_dir if hsi_dir.exists() else class_dir
        for hdr_path in sorted(search_dir.rglob("*.HDR")):
            samples.append((hdr_path, class_to_idx[class_name]))
    if not samples:
        raise FileNotFoundError(
            f"No HDR samples found under {root}. Expected files like <root>/<class_name>/hsi/*.HDR."
        )
    return classes, samples


from mobilenet_enterprise.data.wheat_seed import _split_samples_per_class


def _split_per_class(
    samples: list[tuple[Path, int]],
    num_classes: int,
    val_ratio: float,
    seed: int,
    test_ratio: float = 0.0,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    return _split_samples_per_class(
        samples=samples,
        num_classes=num_classes,
        val_ratio=val_ratio,
        seed=seed,
        test_ratio=test_ratio,
    )


def _largest_component_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=np.uint8)
    best_area = 0
    best_bbox: tuple[int, int, int, int] | None = None
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = 1
            area = 0
            min_y = max_y = y
            min_x = max_x = x
            while stack:
                cy, cx = stack.pop()
                area += 1
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = 1
                        stack.append((ny, nx))
            if area > best_area:
                best_area = area
                # return as half-open interval
                best_bbox = (min_y, max_y + 1, min_x, max_x + 1)
    return best_bbox


def _crop_roi(cube: np.ndarray, threshold_percentile: float, margin: int) -> np.ndarray:
    # cube shape: (H, W, B)
    if cube.ndim != 3 or cube.shape[0] <= 0 or cube.shape[1] <= 0 or cube.shape[2] <= 0:
        return cube
    energy_map = cube.mean(axis=2)
    pct = float(np.clip(threshold_percentile, 50.0, 99.5))
    threshold = float(np.percentile(energy_map, pct))
    mask = energy_map >= threshold
    bbox = _largest_component_bbox(mask)
    if bbox is None:
        return cube
    min_h, max_h, min_w, max_w = bbox
    m = max(0, int(margin))
    min_h = max(0, min_h - m)
    min_w = max(0, min_w - m)
    max_h = min(cube.shape[0], max_h + m)
    max_w = min(cube.shape[1], max_w + m)
    if max_h <= min_h or max_w <= min_w:
        return cube
    return cube[min_h:max_h, min_w:max_w, :]


def _foreground_mask_hw(cube: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Binary mask for already-corrected reflectance cubes (background == 0)."""
    if cube.ndim != 3:
        raise ValueError(f"Expected HWC cube, got shape={cube.shape}")
    return np.sum(cube, axis=2) > float(eps)


def _sample_mean_spectrum(cube: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    h, w, b = cube.shape
    flat = cube.reshape(h * w, b)
    valid_mask = flat.sum(axis=1) > float(eps)
    if valid_mask.any():
        return flat[valid_mask].mean(axis=0).astype(np.float32)
    return flat.mean(axis=0).astype(np.float32)


def _normalize_cube_foreground(
    cube: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Per-band z-score on foreground only; background stays exactly 0."""
    out = np.zeros_like(cube, dtype=np.float32)
    mask = _foreground_mask_hw(cube, eps=eps)
    if not mask.any():
        return out
    mean_b = np.asarray(mean, dtype=np.float32).reshape(-1)
    std_b = np.maximum(np.asarray(std, dtype=np.float32).reshape(-1), 1e-6)
    if mean_b.size == 1:
        mean_b = np.full(cube.shape[2], float(mean_b[0]), dtype=np.float32)
        std_b = np.full(cube.shape[2], float(std_b[0]), dtype=np.float32)
    fg = cube[mask].astype(np.float32, copy=False)
    out[mask] = (fg - mean_b.reshape(1, -1)) / std_b.reshape(1, -1)
    return out


def _resize_cube_spatial(
    cube: np.ndarray,
    size_h: int,
    size_w: int,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Bilinear resize HWC reflectance cube to fixed HxW (no hard ROI crop).

    Near-zero pixels are hard-reset after interpolation so background stays 0
    for subsequent foreground-only normalization.
    """
    if cube.ndim != 3:
        raise ValueError(f"Expected HWC cube, got shape={cube.shape}")
    size_h = max(1, int(size_h))
    size_w = max(1, int(size_w))
    h, w, _ = cube.shape
    if h == size_h and w == size_w:
        out = cube.astype(np.float32, copy=True)
    else:
        x = torch.from_numpy(np.ascontiguousarray(cube, dtype=np.float32))
        x = x.permute(2, 0, 1).unsqueeze(0)
        x = F.interpolate(x, size=(size_h, size_w), mode="bilinear", align_corners=False)
        out = x.squeeze(0).permute(1, 2, 0).contiguous().numpy().astype(np.float32, copy=False)
    mask = _foreground_mask_hw(out, eps=eps)
    cleaned = np.zeros_like(out, dtype=np.float32)
    if mask.any():
        cleaned[mask] = np.clip(out[mask], 0.0, None)
    return cleaned


def _pad_or_center_crop_spatial(cube: np.ndarray, size_h: int, size_w: int) -> np.ndarray:
    """Pad with zeros or center-crop to fixed HxW. Kept for optional/debug use."""
    if cube.ndim != 3:
        raise ValueError(f"Expected HWC cube, got shape={cube.shape}")
    h, w, b = cube.shape
    size_h = max(1, int(size_h))
    size_w = max(1, int(size_w))
    if h > size_h:
        y0 = (h - size_h) // 2
        cube = cube[y0 : y0 + size_h, :, :]
        h = size_h
    if w > size_w:
        x0 = (w - size_w) // 2
        cube = cube[:, x0 : x0 + size_w, :]
        w = size_w
    if h == size_h and w == size_w:
        return cube.astype(np.float32, copy=False)
    out = np.zeros((size_h, size_w, b), dtype=np.float32)
    y0 = (size_h - h) // 2
    x0 = (size_w - w) // 2
    out[y0 : y0 + h, x0 : x0 + w, :] = cube.astype(np.float32, copy=False)
    return out


def _detect_anomalous_bands(
    spectrum: np.ndarray, slope_threshold: float = 10.0, value_threshold: float = 3.0
) -> np.ndarray:
    gradients = np.abs(np.diff(spectrum))
    gradient_median = float(np.median(gradients))
    gradient_std = float(np.std(gradients))
    adaptive_threshold = gradient_median + slope_threshold * gradient_std

    anomalous_bands: set[int] = set()
    for i, grad in enumerate(gradients):
        if grad > adaptive_threshold:
            anomalous_bands.add(i)
            anomalous_bands.add(i + 1)

    median_value = float(np.median(spectrum))
    if median_value > 0:
        for i, value in enumerate(spectrum):
            if abs(float(value) - median_value) > value_threshold * median_value:
                anomalous_bands.add(i)
    return np.asarray(sorted(anomalous_bands), dtype=np.int32)


def _smooth_boundary_noise(spectrum: np.ndarray, head_bands: int = 10, tail_bands: int = 15) -> np.ndarray:
    smoothed = spectrum.copy()
    n = smoothed.size

    if head_bands > 0 and n > head_bands + 20:
        x_fit = np.arange(head_bands, head_bands + 20)
        y_fit = smoothed[head_bands : head_bands + 20]
        coeffs = np.polyfit(x_fit, y_fit, 2)
        x_extrap = np.arange(head_bands)
        smoothed[:head_bands] = np.clip(np.polyval(coeffs, x_extrap), 0, smoothed.max() * 1.2)

    if tail_bands > 0 and n > tail_bands + 60:
        fit_start = n - 60
        fit_end = n - 30
        x_fit = np.arange(fit_start, fit_end)
        y_fit = smoothed[fit_start:fit_end]
        coeffs = np.polyfit(x_fit, y_fit, 2)
        x_extrap = np.arange(n - tail_bands, n)
        smoothed[-tail_bands:] = np.clip(np.polyval(coeffs, x_extrap), 0, smoothed.max() * 1.2)

    return smoothed.astype(np.float32)


def _preprocess_spectrum_for_spa(spectrum: np.ndarray) -> np.ndarray:
    anomalous = _detect_anomalous_bands(spectrum, slope_threshold=5.0, value_threshold=3.0)
    if anomalous.size == 0:
        return _smooth_boundary_noise(spectrum, head_bands=10, tail_bands=15)

    n_bands = spectrum.size
    normal = np.setdiff1d(np.arange(n_bands, dtype=np.int32), anomalous)
    if normal.size < 2:
        return _smooth_boundary_noise(spectrum, head_bands=10, tail_bands=15)

    out = spectrum.copy()
    out[anomalous] = np.interp(anomalous, normal, out[normal])
    return _smooth_boundary_noise(out, head_bands=10, tail_bands=15)


def _select_and_pad_bands(
    cube: np.ndarray, selected_bands: np.ndarray | None, hsi_bands: int
) -> np.ndarray:
    bands = int(cube.shape[2]) if cube.ndim == 3 else 0
    if selected_bands is not None and bands > 0:
        valid_idx = selected_bands[(selected_bands >= 0) & (selected_bands < bands)]
        if valid_idx.size > 0:
            cube = cube[:, :, valid_idx]
            bands = int(cube.shape[2])

    if bands >= hsi_bands:
        return cube[:, :, :hsi_bands].astype(np.float32, copy=False)

    pad = np.zeros(
        (cube.shape[0], cube.shape[1], hsi_bands - max(0, bands)),
        dtype=np.float32,
    )
    return np.concatenate([cube, pad], axis=2).astype(np.float32, copy=False) if bands > 0 else pad


def _cache_key_for_hsi_tensor(
    hdr_path: Path,
    *,
    image_size: int,
    hsi_bands: int,
    selected_bands: np.ndarray | None,
    roi_crop_enabled: bool,
    roi_threshold_percentile: float,
    roi_margin: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> str:
    # v2: cache stores selected-band reflectance cubes (no norm / no spatial resize).
    selected_text = "none" if selected_bands is None else ",".join(str(int(v)) for v in selected_bands.tolist())
    payload = "|".join(
        [
            "v2_reflectance_cube",
            str(hdr_path.resolve()),
            str(int(image_size)),
            str(int(hsi_bands)),
            selected_text,
            str(bool(roi_crop_enabled)),
            f"{float(roi_threshold_percentile):.4f}",
            str(int(roi_margin)),
            str(tuple(np.asarray(mean).reshape(-1).shape)),
            f"{float(np.asarray(mean).reshape(-1).mean()):.8f}",
            f"{float(np.asarray(std).reshape(-1).mean()):.8f}",
        ]
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{hdr_path.stem}_{digest}.npz"


def _resolve_hsi_tensor_cache_path(cache_root: str | Path, hdr_path: Path, cache_name: str) -> Path:
    root = Path(cache_root)
    if not root.is_absolute():
        root = Path.cwd() / root
    class_name = hdr_path.parents[1].name if hdr_path.parent.name.lower() == "hsi" and len(hdr_path.parents) > 1 else hdr_path.parent.name
    return root / class_name / cache_name


def _spa_select_bands(
    spectra: np.ndarray, num_bands: int, exclude_head: int = 0, exclude_tail: int = 0
) -> np.ndarray:
    if spectra.ndim != 2:
        raise ValueError(f"SPA expects 2D spectra, got shape={spectra.shape}")
    n_samples, n_bands = spectra.shape
    if n_bands <= 0:
        raise ValueError("No spectral bands available for SPA.")
    start = max(0, int(exclude_head))
    end = max(start + 1, n_bands - max(0, int(exclude_tail)))
    candidate_idx = np.arange(start, end, dtype=np.int32)
    if candidate_idx.size <= 0:
        candidate_idx = np.arange(n_bands, dtype=np.int32)
    k = max(1, min(int(num_bands), int(candidate_idx.size)))

    # 先标准化，避免尺度差异干扰 SPA 投影过程
    x_full = spectra.astype(np.float32)
    x = x_full[:, candidate_idx]
    x = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    x = x / std

    selected: list[int] = []
    first = int(np.argmax(np.var(x, axis=0)))
    selected.append(first)

    all_idx = np.arange(x.shape[1], dtype=np.int32)
    while len(selected) < k:
        remain = [idx for idx in all_idx if idx not in selected]
        if not remain:
            break

        s_mat = x[:, selected]
        best_idx = remain[0]
        best_score = -1.0
        for candidate in remain:
            target = x[:, candidate]
            if s_mat.ndim == 1:
                s_use = s_mat.reshape(n_samples, 1)
            else:
                s_use = s_mat
            coef, *_ = np.linalg.lstsq(s_use, target, rcond=None)
            proj = s_use @ coef
            residual = target - proj
            score = float(np.linalg.norm(residual))
            if score > best_score:
                best_score = score
                best_idx = int(candidate)
        selected.append(best_idx)

    selected_local = np.asarray(selected, dtype=np.int32)
    return candidate_idx[selected_local]


def _spa_reconstruction_rmse(x: np.ndarray, selected_local: list[int]) -> float:
    if not selected_local:
        return float(np.sqrt(np.mean(np.square(x))))
    s_mat = x[:, selected_local]
    coef, *_ = np.linalg.lstsq(s_mat, x, rcond=None)
    residual = x - (s_mat @ coef)
    return float(np.sqrt(np.mean(np.square(residual))))


def _spa_selection_curve(
    spectra: np.ndarray,
    *,
    exclude_head: int = 0,
    exclude_tail: int = 0,
    max_bands: int | None = None,
) -> list[dict[str, float | int]]:
    if spectra.ndim != 2:
        raise ValueError(f"SPA expects 2D spectra, got shape={spectra.shape}")
    n_samples, n_bands = spectra.shape
    if n_bands <= 0:
        raise ValueError("No spectral bands available for SPA.")

    start = max(0, int(exclude_head))
    end = max(start + 1, n_bands - max(0, int(exclude_tail)))
    candidate_idx = np.arange(start, end, dtype=np.int32)
    if candidate_idx.size <= 0:
        candidate_idx = np.arange(n_bands, dtype=np.int32)
    k_max = int(candidate_idx.size if max_bands is None else min(int(max_bands), int(candidate_idx.size)))

    x_full = spectra.astype(np.float32)
    x = x_full[:, candidate_idx]
    x = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    x = x / std

    selected: list[int] = []
    rows: list[dict[str, float | int]] = []
    first = int(np.argmax(np.var(x, axis=0)))
    selected.append(first)
    rows.append(
        {
            "num_bands": 1,
            "band_index": int(candidate_idx[first]),
            "candidate_rank": 1,
            "projection_residual_norm": float(np.linalg.norm(x[:, first])),
            "reconstruction_rmse": _spa_reconstruction_rmse(x, selected),
        }
    )

    all_idx = np.arange(x.shape[1], dtype=np.int32)
    while len(selected) < k_max:
        remain = [idx for idx in all_idx if idx not in selected]
        if not remain:
            break

        s_mat = x[:, selected]
        best_idx = remain[0]
        best_score = -1.0
        for candidate in remain:
            target = x[:, candidate]
            s_use = s_mat.reshape(n_samples, 1) if s_mat.ndim == 1 else s_mat
            coef, *_ = np.linalg.lstsq(s_use, target, rcond=None)
            residual = target - (s_use @ coef)
            score = float(np.linalg.norm(residual))
            if score > best_score:
                best_score = score
                best_idx = int(candidate)
        selected.append(best_idx)
        rows.append(
            {
                "num_bands": len(selected),
                "band_index": int(candidate_idx[best_idx]),
                "candidate_rank": len(selected),
                "projection_residual_norm": best_score,
                "reconstruction_rmse": _spa_reconstruction_rmse(x, selected),
            }
        )
    return rows


class WheatSeedHSIDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        classes: list[str],
        image_size: int,
        hsi_bands: int,
        roi_crop_enabled: bool = False,
        roi_threshold_percentile: float = 80.0,
        roi_margin: int = 1,
        selected_bands: np.ndarray | None = None,
        train: bool = False,
        augment_enabled: bool = False,
        spatial_flip_prob: float = 0.0,
        spatial_roll_max: int = 0,
        spectral_noise_std: float = 0.0,
        spectral_scale_min: float = 1.0,
        spectral_scale_max: float = 1.0,
        band_dropout_prob: float = 0.0,
        band_dropout_max_fraction: float = 0.0,
        mean: float | np.ndarray = 0.0,
        std: float | np.ndarray = 1.0,
        tensor_cache_enabled: bool = False,
        tensor_cache_root: str = "data/processed/hsi_tensor_cache",
    ) -> None:
        self.samples = samples
        self.classes = classes
        self.image_size = image_size
        self.hsi_bands = hsi_bands
        self.roi_crop_enabled = bool(roi_crop_enabled)
        self.roi_threshold_percentile = float(roi_threshold_percentile)
        self.roi_margin = int(roi_margin)
        self.selected_bands = None if selected_bands is None else np.asarray(selected_bands, dtype=np.int32)
        self.train = bool(train)
        self.augment_enabled = bool(augment_enabled)
        self.spatial_flip_prob = float(spatial_flip_prob)
        self.spatial_roll_max = int(spatial_roll_max)
        self.spectral_noise_std = float(spectral_noise_std)
        self.spectral_scale_min = float(spectral_scale_min)
        self.spectral_scale_max = float(spectral_scale_max)
        self.band_dropout_prob = float(band_dropout_prob)
        self.band_dropout_max_fraction = float(band_dropout_max_fraction)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        self.std = np.maximum(np.asarray(std, dtype=np.float32).reshape(-1), 1e-6)
        self.tensor_cache_enabled = bool(tensor_cache_enabled)
        self.tensor_cache_root = str(tensor_cache_root)

    def __len__(self) -> int:
        return len(self.samples)

    def _augment_cube(self, cube: np.ndarray) -> np.ndarray:
        if not self.train or not self.augment_enabled:
            return cube

        out = cube.copy().astype(np.float32, copy=False)
        if self.spatial_flip_prob > 0 and random.random() < self.spatial_flip_prob:
            out = np.flip(out, axis=1)
        if self.spatial_flip_prob > 0 and random.random() < self.spatial_flip_prob:
            out = np.flip(out, axis=0)
        if self.spatial_roll_max > 0:
            shift_h = random.randint(-self.spatial_roll_max, self.spatial_roll_max)
            shift_w = random.randint(-self.spatial_roll_max, self.spatial_roll_max)
            if shift_h != 0 or shift_w != 0:
                out = np.roll(out, shift=(shift_h, shift_w), axis=(0, 1))

        mask = _foreground_mask_hw(out)
        if mask.any():
            fg = out[mask]
            if self.spectral_scale_max > 0 and self.spectral_scale_max != self.spectral_scale_min:
                scale = random.uniform(self.spectral_scale_min, self.spectral_scale_max)
                fg = fg * scale
            if self.spectral_noise_std > 0:
                noise = np.random.normal(0.0, self.spectral_noise_std, size=fg.shape).astype(np.float32)
                fg = fg + noise
            fg = np.clip(fg, 0.0, 1.0)
            out = np.zeros_like(out, dtype=np.float32)
            out[mask] = fg
        return out.astype(np.float32, copy=False)

    def _apply_band_dropout_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if not self.train or not self.augment_enabled:
            return x
        if self.band_dropout_prob <= 0 or self.band_dropout_max_fraction <= 0:
            return x
        if random.random() >= self.band_dropout_prob:
            return x
        n_bands = x.shape[1]
        max_drop = max(1, int(n_bands * self.band_dropout_max_fraction))
        drop_count = random.randint(1, max_drop)
        drop_idx = torch.as_tensor(
            np.random.choice(n_bands, size=drop_count, replace=False),
            dtype=torch.long,
            device=x.device,
        )
        out = x.clone()
        out[:, drop_idx, :, :] = 0.0
        return out

    def _to_tensor(
        self,
        cube: np.ndarray,
        *,
        apply_augmentation: bool = True,
        bands_already_selected: bool = False,
    ) -> torch.Tensor:
        # Reflectance cube (H,W,B) -> tensor (1,B,H,W)
        # Order: select bands -> reflectance aug -> resize -> fg-only norm (bg=0) -> band dropout
        if not bands_already_selected:
            cube = _select_and_pad_bands(cube, self.selected_bands, self.hsi_bands)
        elif cube.shape[2] != self.hsi_bands:
            cube = _select_and_pad_bands(cube, None, self.hsi_bands)
        if apply_augmentation:
            cube = self._augment_cube(cube)
        cube = _resize_cube_spatial(cube, self.image_size, self.image_size)
        cube = _normalize_cube_foreground(cube, self.mean, self.std)
        x = torch.from_numpy(np.ascontiguousarray(cube)).permute(2, 0, 1).unsqueeze(0)
        if apply_augmentation:
            x = self._apply_band_dropout_tensor(x)
        return x

    def _cache_path(self, hdr_path: Path) -> Path:
        cache_name = _cache_key_for_hsi_tensor(
            hdr_path,
            image_size=self.image_size,
            hsi_bands=self.hsi_bands,
            selected_bands=self.selected_bands,
            roi_crop_enabled=self.roi_crop_enabled,
            roi_threshold_percentile=self.roi_threshold_percentile,
            roi_margin=self.roi_margin,
            mean=self.mean,
            std=self.std,
        )
        return _resolve_hsi_tensor_cache_path(self.tensor_cache_root, hdr_path, cache_name)

    def _load_or_build_cached_cube(self, hdr_path: Path) -> np.ndarray:
        """Cache selected-band reflectance cubes only (no aug / norm / spatial)."""
        cache_path = self._cache_path(hdr_path)
        if cache_path.exists():
            data = np.load(cache_path)
            if "cube" in data:
                return data["cube"].astype(np.float32, copy=False)
            # Legacy normalized tensor caches are ignored; rebuild reflectance cubes.
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
        hdr_path, target = self.samples[index]
        if self.tensor_cache_enabled:
            cube = self._load_or_build_cached_cube(hdr_path)
            tensor = self._to_tensor(cube, apply_augmentation=True, bands_already_selected=True)
            return tensor, target
        cube = _read_hsi_cube(hdr_path)
        if self.roi_crop_enabled:
            cropped = _crop_roi(
                cube,
                threshold_percentile=self.roi_threshold_percentile,
                margin=self.roi_margin,
            )
            if cropped.size > 0:
                cube = cropped
        tensor = self._to_tensor(cube)
        return tensor, target


@dataclass
class WheatSeedHSIDataModule:
    root: str
    image_size: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    val_ratio: float
    test_ratio: float
    split_seed: int
    hsi_bands: int
    exclude_classes: list[str]
    roi_crop_enabled: bool
    roi_threshold_percentile: float
    roi_margin: int
    band_selection_method: str
    band_selection_num_bands: int
    band_selection_cache_path: str
    band_selection_exclude_head: int
    band_selection_exclude_tail: int
    band_selection_use_preprocessed_spectra: bool
    normalize_per_band: bool
    stats_cache_path: str
    augment_enabled: bool
    spatial_flip_prob: float
    spatial_roll_max: int
    spectral_noise_std: float
    spectral_scale_min: float
    spectral_scale_max: float
    band_dropout_prob: float
    band_dropout_max_fraction: float
    tensor_cache_enabled: bool
    tensor_cache_root: str

    _cached_selected_bands: np.ndarray | None = None
    _cached_band_stats: tuple[np.ndarray, np.ndarray] | None = None

    def _build_splits(self):
        root = Path(self.root)
        classes, samples = _collect_hdr_samples(root, exclude_classes=self.exclude_classes)
        train_samples, val_samples, test_samples = _split_per_class(
            samples=samples,
            num_classes=len(classes),
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.split_seed,
        )
        return classes, train_samples, val_samples, test_samples

    def _compute_selected_bands(
        self, train_samples: list[tuple[Path, int]]
    ) -> np.ndarray | None:
        method = str(self.band_selection_method or "").strip().lower()
        if method != "spa":
            return None

        if self._cached_selected_bands is not None:
            print(
                "[SPA] Use in-memory selected bands "
                f"(k={len(self._cached_selected_bands)}): {self._cached_selected_bands.tolist()}"
            )
            return self._cached_selected_bands

        cache_path = Path(self.band_selection_cache_path)
        if not cache_path.is_absolute():
            cache_path = Path.cwd() / cache_path
        if cache_path.exists():
            cached = np.load(cache_path).astype(np.int32)
            cached = np.sort(cached)
            self._cached_selected_bands = cached
            print(f"[SPA] Loaded selected bands from cache: {cache_path}")
            print(f"[SPA] Selected bands (k={len(cached)}): {cached.tolist()}")
            return cached

        print(
            "[SPA] Cache not found, start selecting bands with SPA "
            f"(k={self.band_selection_num_bands})..."
        )
        print(
            "[SPA] Candidate range excludes boundary bands: "
            f"head={self.band_selection_exclude_head}, tail={self.band_selection_exclude_tail}"
        )
        print(f"[SPA] Use denoised spectra for SPA: {self.band_selection_use_preprocessed_spectra}")
        spectra: list[np.ndarray] = []
        for hdr_path, _ in train_samples:
            cube = _read_hsi_cube(hdr_path)
            if self.roi_crop_enabled:
                cropped = _crop_roi(
                    cube,
                    threshold_percentile=self.roi_threshold_percentile,
                    margin=self.roi_margin,
                )
                if cropped.size > 0:
                    cube = cropped
            spec = _sample_mean_spectrum(cube)
            if self.band_selection_use_preprocessed_spectra:
                spec = _preprocess_spectrum_for_spa(spec)
            spectra.append(spec)

        if not spectra:
            return None

        spectra_arr = np.stack(spectra, axis=0)
        selected = _spa_select_bands(
            spectra_arr,
            num_bands=self.band_selection_num_bands,
            exclude_head=self.band_selection_exclude_head,
            exclude_tail=self.band_selection_exclude_tail,
        )
        selected = np.sort(selected.astype(np.int32))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, selected)
        self._cached_selected_bands = selected
        print(f"[SPA] Computed and saved selected bands to: {cache_path}")
        print(f"[SPA] Selected bands (k={len(selected)}): {selected.tolist()}")
        return selected

    def _prepare_cube_for_stats(
        self, hdr_path: Path, selected_bands: np.ndarray | None
    ) -> np.ndarray:
        cube = _read_hsi_cube(hdr_path)
        if self.roi_crop_enabled:
            cropped = _crop_roi(
                cube,
                threshold_percentile=self.roi_threshold_percentile,
                margin=self.roi_margin,
            )
            if cropped.size > 0:
                cube = cropped
        return _select_and_pad_bands(cube, selected_bands, self.hsi_bands)

    def _compute_band_stats(
        self, train_samples: list[tuple[Path, int]], selected_bands: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.normalize_per_band:
            return np.asarray([0.0], dtype=np.float32), np.asarray([1.0], dtype=np.float32)

        if self._cached_band_stats is not None:
            return self._cached_band_stats

        cache_path = Path(self.stats_cache_path)
        if not cache_path.is_absolute():
            cache_path = Path.cwd() / cache_path
        if cache_path.exists():
            data = np.load(cache_path)
            mean = data["mean"].astype(np.float32)
            std = np.maximum(data["std"].astype(np.float32), 1e-6)
            if mean.shape[0] == self.hsi_bands and std.shape[0] == self.hsi_bands:
                self._cached_band_stats = (mean, std)
                print(f"[HSI] Loaded per-band normalization stats from cache: {cache_path}")
                print(f"[HSI] Band mean range: {mean.min():.4f} - {mean.max():.4f}")
                print(f"[HSI] Band std range: {std.min():.4f} - {std.max():.4f}")
                return mean, std

        sums = np.zeros(self.hsi_bands, dtype=np.float64)
        sq_sums = np.zeros(self.hsi_bands, dtype=np.float64)
        count = 0
        for hdr_path, _ in train_samples:
            cube = self._prepare_cube_for_stats(hdr_path, selected_bands)
            flat = cube.reshape(-1, self.hsi_bands)
            valid_mask = flat.sum(axis=1) > 1e-6
            valid = flat[valid_mask] if valid_mask.any() else flat
            sums += valid.sum(axis=0)
            sq_sums += np.square(valid.astype(np.float64)).sum(axis=0)
            count += int(valid.shape[0])

        if count <= 0:
            mean = np.zeros(self.hsi_bands, dtype=np.float32)
            std = np.ones(self.hsi_bands, dtype=np.float32)
        else:
            mean64 = sums / count
            var64 = np.maximum(sq_sums / count - mean64**2, 1e-12)
            mean = mean64.astype(np.float32)
            std = np.sqrt(var64).astype(np.float32)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, mean=mean, std=std)
        self._cached_band_stats = (mean, std)
        print(f"[HSI] Computed and saved per-band normalization stats to: {cache_path}")
        print(f"[HSI] Band mean range: {mean.min():.4f} - {mean.max():.4f}")
        print(f"[HSI] Band std range: {std.min():.4f} - {std.max():.4f}")
        return mean, std

    def _loader_kwargs(self, train: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": train,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            kwargs["prefetch_factor"] = max(2, self.prefetch_factor)
        return kwargs

    def train_loader(self) -> DataLoader:
        classes, train_samples, _, _ = self._build_splits()
        selected_bands = self._compute_selected_bands(train_samples)
        band_mean, band_std = self._compute_band_stats(train_samples, selected_bands)
        dataset = WheatSeedHSIDataset(
            samples=train_samples,
            classes=classes,
            image_size=self.image_size,
            hsi_bands=self.hsi_bands,
            roi_crop_enabled=self.roi_crop_enabled,
            roi_threshold_percentile=self.roi_threshold_percentile,
            roi_margin=self.roi_margin,
            selected_bands=selected_bands,
            train=True,
            augment_enabled=self.augment_enabled,
            spatial_flip_prob=self.spatial_flip_prob,
            spatial_roll_max=self.spatial_roll_max,
            spectral_noise_std=self.spectral_noise_std,
            spectral_scale_min=self.spectral_scale_min,
            spectral_scale_max=self.spectral_scale_max,
            band_dropout_prob=self.band_dropout_prob,
            band_dropout_max_fraction=self.band_dropout_max_fraction,
            mean=band_mean,
            std=band_std,
            tensor_cache_enabled=self.tensor_cache_enabled,
            tensor_cache_root=self.tensor_cache_root,
        )
        return DataLoader(dataset, **self._loader_kwargs(train=True))

    def val_loader(self) -> DataLoader:
        classes, train_samples, val_samples, _ = self._build_splits()
        selected_bands = self._compute_selected_bands(train_samples)
        band_mean, band_std = self._compute_band_stats(train_samples, selected_bands)
        dataset = WheatSeedHSIDataset(
            samples=val_samples,
            classes=classes,
            image_size=self.image_size,
            hsi_bands=self.hsi_bands,
            roi_crop_enabled=self.roi_crop_enabled,
            roi_threshold_percentile=self.roi_threshold_percentile,
            roi_margin=self.roi_margin,
            selected_bands=selected_bands,
            train=False,
            augment_enabled=False,
            mean=band_mean,
            std=band_std,
            tensor_cache_enabled=self.tensor_cache_enabled,
            tensor_cache_root=self.tensor_cache_root,
        )
        return DataLoader(dataset, **self._loader_kwargs(train=False))

    def test_loader(self) -> DataLoader:
        classes, train_samples, _, test_samples = self._build_splits()
        selected_bands = self._compute_selected_bands(train_samples)
        band_mean, band_std = self._compute_band_stats(train_samples, selected_bands)
        dataset = WheatSeedHSIDataset(
            samples=test_samples,
            classes=classes,
            image_size=self.image_size,
            hsi_bands=self.hsi_bands,
            roi_crop_enabled=self.roi_crop_enabled,
            roi_threshold_percentile=self.roi_threshold_percentile,
            roi_margin=self.roi_margin,
            selected_bands=selected_bands,
            train=False,
            augment_enabled=False,
            mean=band_mean,
            std=band_std,
            tensor_cache_enabled=self.tensor_cache_enabled,
            tensor_cache_root=self.tensor_cache_root,
        )
        return DataLoader(dataset, **self._loader_kwargs(train=False))

    def classes(self) -> list[str]:
        classes, _, _, _ = self._build_splits()
        return classes


def build_wheat_seed_hsi_datamodule(cfg: dict[str, Any]) -> WheatSeedHSIDataModule:
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    band_sel_cfg = data_cfg.get("band_selection", {})
    augment_cfg = data_cfg.get("hsi_augmentation", {})
    norm_cfg = data_cfg.get("normalization", {})
    cache_cfg = data_cfg.get("preprocessing_cache", {})
    return WheatSeedHSIDataModule(
        root=data_cfg.get("root", "data/wheat-seed"),
        image_size=int(data_cfg.get("image_size", 64)),
        batch_size=int(data_cfg.get("batch_size", 8)),
        num_workers=int(data_cfg.get("num_workers", 2)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=bool(data_cfg.get("persistent_workers", True)),
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
        val_ratio=float(data_cfg.get("val_ratio", 0.15)),
        test_ratio=float(data_cfg.get("test_ratio", 0.15)),
        split_seed=int(data_cfg.get("split_seed", 42)),
        hsi_bands=int(data_cfg.get("hsi_bands", model_cfg.get("hsi_bands", 454))),
        exclude_classes=list(data_cfg.get("exclude_classes", [])),
        roi_crop_enabled=bool(data_cfg.get("roi_crop_enabled", False)),
        roi_threshold_percentile=float(data_cfg.get("roi_threshold_percentile", 80.0)),
        roi_margin=int(data_cfg.get("roi_margin", 1)),
        band_selection_method=str(band_sel_cfg.get("method", "none")),
        band_selection_num_bands=int(band_sel_cfg.get("num_bands", data_cfg.get("hsi_bands", 454))),
        band_selection_cache_path=str(
            band_sel_cfg.get("cache_path", "data/processed/selected_bands.npy")
        ),
        band_selection_exclude_head=int(band_sel_cfg.get("exclude_head", 0)),
        band_selection_exclude_tail=int(band_sel_cfg.get("exclude_tail", 0)),
        band_selection_use_preprocessed_spectra=bool(
            band_sel_cfg.get("use_preprocessed_spectra", False)
        ),
        normalize_per_band=bool(norm_cfg.get("per_band", False)),
        stats_cache_path=str(
            norm_cfg.get("stats_cache_path", "data/processed/hsi_band_stats_spa_24.npz")
        ),
        augment_enabled=bool(augment_cfg.get("enabled", False)),
        spatial_flip_prob=float(augment_cfg.get("spatial_flip_prob", 0.0)),
        spatial_roll_max=int(augment_cfg.get("spatial_roll_max", 0)),
        spectral_noise_std=float(augment_cfg.get("spectral_noise_std", 0.0)),
        spectral_scale_min=float(augment_cfg.get("spectral_scale_min", 1.0)),
        spectral_scale_max=float(augment_cfg.get("spectral_scale_max", 1.0)),
        band_dropout_prob=float(augment_cfg.get("band_dropout_prob", 0.0)),
        band_dropout_max_fraction=float(augment_cfg.get("band_dropout_max_fraction", 0.0)),
        tensor_cache_enabled=bool(cache_cfg.get("enabled", False)),
        tensor_cache_root=str(cache_cfg.get("root", "data/processed/hsi_tensor_cache")),
    )
