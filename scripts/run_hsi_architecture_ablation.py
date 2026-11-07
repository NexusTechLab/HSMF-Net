from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilenet_enterprise.registry import build_model
from mobilenet_enterprise.utils import collect_network_structure, export_network_structure, load_config
from mobilenet_enterprise.utils.network_structure import estimate_model_gflops

from scripts.export_comparison_extra_data import export_comparison_extra_data

# SCI S1-S8 ablation suite (kept paper experiments).
SCI_S1_RGB = "configs/experiments/sci_s1_rgb_baseline.yaml"
SCI_S2_HSI_PURE3D = "configs/experiments/sci_s2_hsi_pure3d_spa50.yaml"
SCI_S3_HSI_HYBRID = "configs/experiments/sci_s3_hsi_hybrid_spa50.yaml"
SCI_S4_FUSION_CONCAT = "configs/experiments/sci_s4_fusion_hybrid_concat_spa50.yaml"
SCI_S5_GATED = "configs/experiments/sci_s5_fusion_hybrid_gated_spa50.yaml"
SCI_S6_CROSS_ATTN = "configs/experiments/sci_s6_fusion_hybrid_cross_attn_spa50.yaml"
SCI_S7_MIDFILM = "configs/experiments/sci_s7_fusion_hybrid_midfilm_spa50.yaml"
SCI_CROSS_ATTN_SPA24 = "configs/experiments/sci_s6_fusion_hybrid_cross_attn_spa24.yaml"
SCI_S8_MIDFILM = "configs/experiments/sci_s8_fusion_hybrid_midfilm_spa24.yaml"

HBFM_A0_CONCAT = "configs/experiments/hbfm_ablation/hbfm_a0_concat_spa24.yaml"
HBFM_A1_FULL = "configs/experiments/hbfm_ablation/hbfm_a1_full.yaml"
HBFM_A2_LATE = "configs/experiments/hbfm_ablation/hbfm_a2_late_only.yaml"
HBFM_A3_MID = "configs/experiments/hbfm_ablation/hbfm_a3_mid_only.yaml"
HBFM_A4_NO_ACSA = "configs/experiments/hbfm_ablation/hbfm_a4_no_acsa.yaml"
HBFM_A5_UNI = "configs/experiments/hbfm_ablation/hbfm_a5_uni_hsi2rgb.yaml"
HBFM_A6_NO_RES = "configs/experiments/hbfm_ablation/hbfm_a6_no_residual.yaml"
HBFM_A7_AFFINE = "configs/experiments/hbfm_ablation/hbfm_a7_affine_film.yaml"
HBFM_A8_NO_ZERO = "configs/experiments/hbfm_ablation/hbfm_a8_no_zero_init.yaml"

DEFAULT_CONFIGS = [
    SCI_S1_RGB,
    SCI_S2_HSI_PURE3D,
    SCI_S3_HSI_HYBRID,
    SCI_S4_FUSION_CONCAT,
    SCI_S5_GATED,
    SCI_S6_CROSS_ATTN,
    SCI_S7_MIDFILM,
    SCI_S8_MIDFILM,
]

MIDFILM_CONFIGS = [SCI_S7_MIDFILM, SCI_S8_MIDFILM]

HBFM_INTERNAL_CONFIGS = [
    HBFM_A0_CONCAT,
    HBFM_A1_FULL,
    HBFM_A2_LATE,
    HBFM_A3_MID,
    HBFM_A4_NO_ACSA,
    HBFM_A5_UNI,
    HBFM_A6_NO_RES,
    HBFM_A7_AFFINE,
    HBFM_A8_NO_ZERO,
]

S6_BOOST_V1 = "configs/experiments/s6_boost/s6_boost_v1_long_cosine.yaml"
S6_BOOST_V2 = "configs/experiments/s6_boost/s6_boost_v2_mild_reg.yaml"
S6_BOOST_V3 = "configs/experiments/s6_boost/s6_boost_v3_balanced.yaml"
S6_BOOST_CONFIGS = [S6_BOOST_V1, S6_BOOST_V2, S6_BOOST_V3]

STAGE_CHOICES = {
    "s1": [SCI_S1_RGB],
    "s2": [SCI_S2_HSI_PURE3D],
    "s3": [SCI_S3_HSI_HYBRID],
    "s4": [SCI_S4_FUSION_CONCAT],
    "s5": [SCI_S5_GATED],
    "s6": [SCI_S6_CROSS_ATTN],
    "s7": [SCI_S7_MIDFILM],
    "s8": [SCI_S8_MIDFILM],
    "cross_attn_spa24": [SCI_CROSS_ATTN_SPA24],
    "midfilm": MIDFILM_CONFIGS,
    "hbfm": HBFM_INTERNAL_CONFIGS,
    "s6_boost": S6_BOOST_CONFIGS,
    "hsi": [SCI_S2_HSI_PURE3D, SCI_S3_HSI_HYBRID],
    "fusion": [
        SCI_S4_FUSION_CONCAT,
        SCI_S5_GATED,
        SCI_S6_CROSS_ATTN,
        SCI_S7_MIDFILM,
        SCI_S8_MIDFILM,
    ],
    "sci": DEFAULT_CONFIGS,
    "all": DEFAULT_CONFIGS,
}

_FUSION_MODEL_NAMES = {"rgb_hsi_fusion", "rgb_hsi_attention", "rgb_hsi_mid_film"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SCI-style MobileNet ablation experiments "
            "(RGB/HSI baselines, HSI single-factor variants, fusion transfer)."
        )
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Override experiment YAML files. Defaults to the SCI ablation suite.",
    )
    parser.add_argument(
        "--stage",
        choices=sorted(STAGE_CHOICES.keys()),
        default="all",
        help="Run a predefined stage subset instead of --configs.",
    )
    parser.add_argument("--train", action="store_true", help="Run training for each experiment.")
    parser.add_argument("--eval", action="store_true", help="Run eval.py on best checkpoints.")
    parser.add_argument("--auto-resume", action="store_true", help="Pass --auto-resume to train.py.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Training seeds for repeated SCI runs. When more than one seed is provided, "
            "each experiment is written to <experiment>_seed<seed> and aggregate mean/std tables are produced."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Override train.epochs for every selected config. The cosine scheduler period "
            "and early-stopping minimum epoch are aligned to the same value."
        ),
    )
    parser.add_argument(
        "--comparison-id",
        type=str,
        default=None,
        help="Custom output folder name under artifacts/.",
    )
    parser.add_argument("--tsne-max-samples", type=int, default=2000)
    parser.add_argument(
        "--complexity-device",
        type=str,
        default="cpu",
        help="Device for Params/FLOPs/latency profiling. Use cpu for comparable paper tables.",
    )
    parser.add_argument("--latency-warmup", type=int, default=5)
    parser.add_argument("--latency-iters", type=int, default=30)
    parser.add_argument(
        "--skip-complexity",
        action="store_true",
        help="Skip Params/FLOPs/latency profiling.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without training or evaluation.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help=(
            "Rebuild comparison CSV/MD from existing experiment dirs under --comparison-id. "
            "Does not train or eval. If --configs/--seeds omitted, they are read from run_manifest.json."
        ),
    )
    return parser.parse_args(argv)


def _resolve_config(path: str) -> Path:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return cfg_path


def _run_command(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _write_comparison_config(
    cfg: dict[str, Any],
    config_path: Path,
    comparison_root: Path,
    *,
    seed: int | None = None,
    epochs: int | None = None,
    suffix_experiment_name: bool = False,
) -> Path:
    run_cfg = copy.deepcopy(cfg)
    if seed is not None:
        run_cfg["seed"] = int(seed)
    if epochs is not None:
        if int(epochs) < 1:
            raise ValueError("epochs must be at least 1")
        run_cfg.setdefault("train", {})["epochs"] = int(epochs)
        run_cfg["train"]["early_stopping_min_epoch"] = int(epochs)
        run_cfg.setdefault("scheduler", {})["t_max"] = int(epochs)
    base_exp_name = str(run_cfg.get("experiment_name", config_path.stem))
    if suffix_experiment_name and seed is not None:
        run_cfg["experiment_name"] = f"{base_exp_name}_seed{seed}"
    run_cfg["output_dir"] = str(comparison_root)
    tmp_config_dir = comparison_root / "configs"
    tmp_config_dir.mkdir(parents=True, exist_ok=True)
    tmp_config_name = config_path.name
    if suffix_experiment_name and seed is not None:
        tmp_config_name = f"{config_path.stem}_seed{seed}{config_path.suffix}"
    tmp_config_path = tmp_config_dir / tmp_config_name
    with tmp_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(run_cfg, handle, sort_keys=False)
    return tmp_config_path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_prf(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    rows: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row.get("class_name", "")
            if name in {"macro_avg", "weighted_avg"}:
                rows[f"{name}_precision"] = float(row.get("precision", 0.0))
                rows[f"{name}_recall"] = float(row.get("recall", 0.0))
                rows[f"{name}_f1"] = float(row.get("f1", 0.0))
                if row.get("support", "") != "":
                    rows[f"{name}_support"] = float(row["support"])
    return rows


def _read_test_acc(path: Path) -> float | None:
    """Hold-out accuracy from eval_predictions.csv (eval.py defaults to --split test)."""
    if not path.exists():
        return None
    total = 0
    correct = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total += 1
            if row.get("correct", "") != "":
                correct += int(float(row["correct"]))
                continue
            true_v = row.get("true_label", row.get("true_class", ""))
            pred_v = row.get("pred_label", row.get("pred_class", ""))
            if true_v != "" and true_v == pred_v:
                correct += 1
    if total == 0:
        return None
    return correct / total


def _hsi_architecture_fields(cfg: dict[str, Any]) -> dict[str, str]:
    model_cfg = cfg.get("model", {})
    fusion_cfg = model_cfg.get("fusion", {})
    if model_cfg.get("name") == "v3":
        return {
            "stem3d_type": "",
            "spectral_compressor": "",
            "stem3d_channels": "",
            "stem3d_max_spatial_pools": "",
            "spectral_bins": "",
        }
    if model_cfg.get("name") in _FUSION_MODEL_NAMES:
        return {
            "stem3d_type": str(fusion_cfg.get("hsi_stem3d_type", "standard")),
            "spectral_compressor": str(fusion_cfg.get("hsi_spectral_compressor", "adaptive_pool")),
            "stem3d_channels": str(fusion_cfg.get("hsi_stem3d_channels", "[16, 32]")),
            "stem3d_max_spatial_pools": str(fusion_cfg.get("hsi_stem3d_max_spatial_pools", "")),
            "spectral_bins": str(fusion_cfg.get("hsi_spectral_bins", 8)),
        }
    return {
        "stem3d_type": str(model_cfg.get("stem3d_type", "standard")),
        "spectral_compressor": str(model_cfg.get("spectral_compressor", "adaptive_pool")),
        "stem3d_channels": str(model_cfg.get("stem3d_channels", "[16, 32]")),
        "stem3d_max_spatial_pools": str(model_cfg.get("stem3d_max_spatial_pools", "")),
        "spectral_bins": str(model_cfg.get("spectral_bins", 8)),
    }


def _stage_label(config_path: Path) -> str:
    name = config_path.stem
    mapping = {
        "sci_s1_rgb_baseline": "S1_rgb",
        "sci_s2_hsi_pure3d_spa50": "S2_hsi_pure3d",
        "sci_s3_hsi_hybrid_spa50": "S3_hsi_hybrid",
        "sci_s4_fusion_hybrid_concat_spa50": "S4_fusion_concat",
        "sci_s5_fusion_hybrid_gated_spa50": "S5_gated_fusion",
        "sci_s6_fusion_hybrid_cross_attn_spa50": "S6_cross_attn",
        "sci_s7_fusion_hybrid_midfilm_spa50": "S7_hsmf_spa50",
        "sci_s6_fusion_hybrid_cross_attn_spa24": "S6_extra_cross_attn_spa24",
        "sci_s8_fusion_hybrid_midfilm_spa24": "S8_hsmf_spa24",
        "hbfm_a0_concat_spa24": "HBFM_A0_concat",
        "hbfm_a1_full": "HBFM_A1_full",
        "hbfm_a2_late_only": "HBFM_A2_late",
        "hbfm_a3_mid_only": "HBFM_A3_mid",
        "hbfm_a4_no_acsa": "HBFM_A4_no_acsa",
        "hbfm_a5_uni_hsi2rgb": "HBFM_A5_uni",
        "hbfm_a6_no_residual": "HBFM_A6_no_res",
        "hbfm_a7_affine_film": "HBFM_A7_affine",
        "hbfm_a8_no_zero_init": "HBFM_A8_no_zero",
        "s6_boost_v1_long_cosine": "S6B_V1_long",
        "s6_boost_v2_mild_reg": "S6B_V2_mild",
        "s6_boost_v3_balanced": "S6B_V3_bal",
        "encoder_pure3d_spa24": "baseline_hsi",
        "encoder_hybridsn_spa24": "pilot_encoder_hybridsn",
        "encoder_hybrid3d2d_spa24": "pilot_encoder_hybrid3d2d",
        "fusion_aligned_concat_spa24": "baseline_fusion",
        "fusion_cross_attention_spa24": "pilot_fusion_cross_attention",
        "fusion_hbfm_spa24": "pilot_fusion_hbfm",
        "hierarchy_mid_only_spa24": "pilot_hierarchy_mid_only",
        "hierarchy_late_only_spa24": "pilot_hierarchy_late_only",
    }
    return mapping.get(name, name)


def _experiment_row(
    config_path: Path,
    cfg: dict[str, Any],
    exp_dir: Path,
    *,
    seed: int | None = None,
    base_experiment_name: str | None = None,
    complexity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = _load_json(exp_dir / "metrics.json")
    prf = _read_prf(exp_dir / "figures" / "per_class_prf.csv")
    test_acc = _read_test_acc(exp_dir / "figures" / "eval_predictions.csv")
    test_macro_f1 = prf.get("macro_avg_f1", "")
    test_weighted_f1 = prf.get("weighted_avg_f1", "")
    model_cfg = cfg.get("model", {})
    arch = _hsi_architecture_fields(cfg)
    model_name = str(model_cfg.get("name", ""))
    modality = "fusion" if model_name in _FUSION_MODEL_NAMES else ("rgb" if model_name == "v3" else "hsi")
    row = {
        "stage": _stage_label(config_path),
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "experiment_name": base_experiment_name or cfg.get("experiment_name", exp_dir.name),
        "run_experiment_name": cfg.get("experiment_name", exp_dir.name),
        "seed": cfg.get("seed", seed if seed is not None else ""),
        "model_name": model_name,
        "modality": modality,
        "stem3d_type": arch["stem3d_type"],
        "spectral_compressor": arch["spectral_compressor"],
        "stem3d_channels": arch["stem3d_channels"],
        "stem3d_max_spatial_pools": arch["stem3d_max_spatial_pools"],
        "spectral_bins": arch["spectral_bins"],
        # Primary paper metrics: hold-out TEST (from eval.py figures).
        "test_acc": "" if test_acc is None else test_acc,
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1,
        "delta_vs_baseline": "",
        "delta_acc": "",
        "delta_macro_f1": "",
        # Backward-compatible aliases of TEST metrics.
        "macro_f1": test_macro_f1,
        "weighted_f1": test_weighted_f1,
        # Validation diagnostics from training (model selection only).
        "best_val_acc": metrics.get("best_val_acc", ""),
        "val_acc": metrics.get("val_acc", ""),
        "epoch": metrics.get("epoch", ""),
        "result_dir": str(exp_dir),
        "config_snapshot": str(exp_dir / "config_snapshot.yaml"),
        "metrics_path": str(exp_dir / "metrics.json"),
        "per_class_prf": str(exp_dir / "figures" / "per_class_prf.csv"),
        "eval_predictions": str(exp_dir / "figures" / "eval_predictions.csv"),
    }
    ckpt_path = exp_dir / "checkpoints" / "best.pt"
    if ckpt_path.exists():
        row["checkpoint_file_size_mb"] = ckpt_path.stat().st_size / (1024**2)
    if complexity:
        row.update(complexity)
    return row


def _as_float(value: Any) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _add_baseline_deltas(rows: list[dict[str, Any]]) -> None:
    """Deltas use hold-out TEST metrics when available."""
    baselines: dict[str, float] = {}
    for row in rows:
        value = _as_float(row.get("test_acc"))
        if value is None:
            value = _as_float(row.get("best_val_acc"))
        if value is None:
            continue
        if row.get("stage") == "baseline_hsi":
            baselines["hsi"] = value
        elif row.get("stage") == "baseline_fusion":
            baselines["fusion"] = value
        elif row.get("stage") == "baseline_rgb":
            baselines["rgb"] = value
    for row in rows:
        value = _as_float(row.get("test_acc"))
        if value is None:
            value = _as_float(row.get("best_val_acc"))
        baseline = baselines.get(str(row.get("modality", "")))
        if value is None or baseline is None:
            row["delta_vs_baseline"] = ""
            row["delta_acc"] = ""
        else:
            row["delta_vs_baseline"] = value - baseline
            row["delta_acc"] = value - baseline

    macro_baselines: dict[str, float] = {}
    for row in rows:
        value = _as_float(row.get("test_macro_f1"))
        if value is None:
            value = _as_float(row.get("macro_f1"))
        if value is None:
            continue
        if row.get("stage") == "baseline_hsi":
            macro_baselines["hsi"] = value
        elif row.get("stage") == "baseline_fusion":
            macro_baselines["fusion"] = value
        elif row.get("stage") == "baseline_rgb":
            macro_baselines["rgb"] = value
    for row in rows:
        value = _as_float(row.get("test_macro_f1"))
        if value is None:
            value = _as_float(row.get("macro_f1"))
        baseline = macro_baselines.get(str(row.get("modality", "")))
        row["delta_macro_f1"] = "" if value is None or baseline is None else value - baseline


def _selected_configs(args: argparse.Namespace) -> list[str]:
    if args.configs:
        return list(args.configs)
    return list(STAGE_CHOICES[args.stage])


def _mean(values: list[float]) -> float | str:
    return statistics.mean(values) if values else ""


def _std(values: list[float]) -> float | str:
    if not values:
        return ""
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values)


def _numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _as_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _aggregate_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("stage", "")), str(row.get("experiment_name", ""))), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for _, group in grouped.items():
        first = group[0]
        seeds = [str(row.get("seed", "")) for row in group if row.get("seed", "") != ""]
        best_values = _numeric_values(group, "best_val_acc")
        val_values = _numeric_values(group, "val_acc")
        test_acc_values = _numeric_values(group, "test_acc")
        test_macro_values = _numeric_values(group, "test_macro_f1")
        test_weighted_values = _numeric_values(group, "test_weighted_f1")
        macro_values = _numeric_values(group, "macro_f1") or test_macro_values
        weighted_values = _numeric_values(group, "weighted_f1") or test_weighted_values
        row = {
            "stage": first.get("stage", ""),
            "config": first.get("config", ""),
            "experiment_name": first.get("experiment_name", ""),
            "model_name": first.get("model_name", ""),
            "modality": first.get("modality", ""),
            "stem3d_type": first.get("stem3d_type", ""),
            "spectral_compressor": first.get("spectral_compressor", ""),
            "stem3d_channels": first.get("stem3d_channels", ""),
            "stem3d_max_spatial_pools": first.get("stem3d_max_spatial_pools", ""),
            "spectral_bins": first.get("spectral_bins", ""),
            "seed_count": len(group),
            "seeds": ",".join(seeds),
            "test_acc": _mean(test_acc_values),
            "test_acc_mean": _mean(test_acc_values),
            "test_acc_std": _std(test_acc_values),
            "test_macro_f1": _mean(test_macro_values),
            "test_macro_f1_mean": _mean(test_macro_values),
            "test_macro_f1_std": _std(test_macro_values),
            "test_weighted_f1": _mean(test_weighted_values),
            "test_weighted_f1_mean": _mean(test_weighted_values),
            "test_weighted_f1_std": _std(test_weighted_values),
            "delta_vs_baseline": "",
            "delta_acc": "",
            "delta_macro_f1": "",
            "macro_f1": _mean(macro_values),
            "macro_f1_mean": _mean(macro_values),
            "macro_f1_std": _std(macro_values),
            "weighted_f1": _mean(weighted_values),
            "weighted_f1_mean": _mean(weighted_values),
            "weighted_f1_std": _std(weighted_values),
            "best_val_acc": _mean(best_values),
            "best_val_acc_mean": _mean(best_values),
            "best_val_acc_std": _std(best_values),
            "val_acc": _mean(val_values),
            "val_acc_mean": _mean(val_values),
            "val_acc_std": _std(val_values),
            "epoch": _mean(_numeric_values(group, "epoch")),
            "result_dir": "",
            "config_snapshot": "",
            "metrics_path": "",
            "per_class_prf": "",
            "eval_predictions": "",
        }
        for key in [
            "total_params",
            "trainable_params",
            "params_m",
            "gflops",
            "latency_ms_mean",
            "latency_ms_std",
            "fps",
            "peak_memory_mb",
            "model_weight_size_mb",
            "complexity_device",
            "input_shapes",
            "network_structure_txt",
            "network_structure_json",
            "network_structure_csv",
        ]:
            row[key] = first.get(key, "")
        aggregate_rows.append(row)

    _add_baseline_deltas(aggregate_rows)
    return aggregate_rows


def _format_mean_std(mean_value: Any, std_value: Any) -> str:
    mean_float = _as_float(mean_value)
    std_float = _as_float(std_value)
    if mean_float is None:
        return ""
    if std_float is None:
        return f"{mean_float:.4f}"
    return f"{mean_float:.4f} ± {std_float:.4f}"


def _dummy_inputs(cfg: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    model_name = str(model_cfg.get("name", "")).lower()
    image_size = int(data_cfg.get("image_size", 224))
    hsi_bands = int(data_cfg.get("hsi_bands", model_cfg.get("hsi_bands", model_cfg.get("fusion", {}).get("hsi_bands", 1))))
    if model_name in {"v3_3d", "hsi_pure3d", "hsi_hybridsn"}:
        return (torch.randn(1, 1, hsi_bands, image_size, image_size, device=device),)
    if model_name in _FUSION_MODEL_NAMES:
        hsi_size = int(data_cfg.get("hsi_image_size", image_size))
        return (
            torch.randn(1, 3, image_size, image_size, device=device),
            torch.randn(1, 1, hsi_bands, hsi_size, hsi_size, device=device),
        )
    return (torch.randn(1, 3, image_size, image_size, device=device),)


def _model_weight_size_mb(model: torch.nn.Module) -> float:
    total_bytes = 0
    for tensor in model.state_dict().values():
        if torch.is_tensor(tensor):
            total_bytes += int(tensor.numel()) * int(tensor.element_size())
    return total_bytes / (1024**2)


def _measure_peak_memory_mb(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    device: torch.device,
) -> float | str:
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            _ = model(*inputs)
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) / (1024**2)

    import tracemalloc

    tracemalloc.start()
    try:
        with torch.inference_mode():
            _ = model(*inputs)
        return tracemalloc.get_traced_memory()[1] / (1024**2)
    finally:
        tracemalloc.stop()


def _fps_from_latency_ms(latency_ms: float | str) -> float | str:
    latency = _as_float(latency_ms)
    if latency is None or latency <= 0:
        return ""
    return 1000.0 / latency


def _measure_latency_ms(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> tuple[float | str, float | str]:
    if iters <= 0:
        return "", ""
    times: list[float] = []
    model.eval()
    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            _ = model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for _ in range(iters):
            start = time.perf_counter()
            _ = model(*inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - start) * 1000.0)
    return _mean(times), _std(times)


def _complexity_row(
    cfg: dict[str, Any],
    *,
    device_text: str,
    latency_warmup: int,
    latency_iters: int,
    structure_output_dir: Path | None = None,
) -> dict[str, Any]:
    try:
        device = torch.device(device_text)
        model = build_model(cfg).to(device)
        spec = collect_network_structure(model, cfg)
        inputs = _dummy_inputs(cfg, device)
        latency_mean, latency_std = _measure_latency_ms(
            model,
            inputs,
            warmup=latency_warmup,
            iters=latency_iters,
            device=device,
        )
        peak_memory_mb = _measure_peak_memory_mb(model, inputs, device)
        model_weight_size_mb = _model_weight_size_mb(model)
        fps = _fps_from_latency_ms(latency_mean)
        total_params = int(spec["summary"]["total_params"])
        trainable_params = int(spec["summary"]["trainable_params"])
        structure_paths: dict[str, Path] = {}
        if structure_output_dir is not None:
            structure_paths = export_network_structure(
                model,
                cfg,
                structure_output_dir,
                formats=("txt", "json", "csv"),
                device=device,
            )
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "params_m": total_params / 1e6,
            "gflops": estimate_model_gflops(model, cfg, device),
            "latency_ms_mean": latency_mean,
            "latency_ms_std": latency_std,
            "fps": fps,
            "peak_memory_mb": peak_memory_mb,
            "model_weight_size_mb": model_weight_size_mb,
            "complexity_device": str(device),
            "input_shapes": "; ".join(spec.get("input_shapes", [])),
            "network_structure_txt": str(structure_paths.get("txt", "")),
            "network_structure_json": str(structure_paths.get("json", "")),
            "network_structure_csv": str(structure_paths.get("csv", "")),
        }
    except Exception as exc:
        return {
            "total_params": "",
            "trainable_params": "",
            "params_m": "",
            "gflops": "N/A",
            "latency_ms_mean": "",
            "latency_ms_std": "",
            "fps": "",
            "peak_memory_mb": "",
            "model_weight_size_mb": "",
            "complexity_device": device_text,
            "input_shapes": "",
            "complexity_error": str(exc),
        }


CSV_FIELD_ORDER = [
    "stage",
    "config",
    "experiment_name",
    "run_experiment_name",
    "seed",
    "seed_count",
    "seeds",
    "model_name",
    "modality",
    "stem3d_type",
    "spectral_compressor",
    "stem3d_channels",
    "stem3d_max_spatial_pools",
    "spectral_bins",
    "test_acc",
    "test_acc_mean",
    "test_acc_std",
    "test_macro_f1",
    "test_macro_f1_mean",
    "test_macro_f1_std",
    "test_weighted_f1",
    "test_weighted_f1_mean",
    "test_weighted_f1_std",
    "delta_vs_baseline",
    "delta_acc",
    "delta_macro_f1",
    "macro_f1",
    "macro_f1_mean",
    "macro_f1_std",
    "weighted_f1",
    "weighted_f1_mean",
    "weighted_f1_std",
    "best_val_acc",
    "best_val_acc_mean",
    "best_val_acc_std",
    "val_acc",
    "val_acc_mean",
    "val_acc_std",
    "epoch",
    "total_params",
    "trainable_params",
    "params_m",
    "gflops",
    "latency_ms_mean",
    "latency_ms_std",
    "fps",
    "peak_memory_mb",
    "model_weight_size_mb",
    "checkpoint_file_size_mb",
    "complexity_device",
    "input_shapes",
    "network_structure_txt",
    "network_structure_json",
    "network_structure_csv",
    "result_dir",
    "config_snapshot",
    "metrics_path",
    "per_class_prf",
    "eval_predictions",
]


def _csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    extra = sorted({key for row in rows for key in row.keys()} - set(CSV_FIELD_ORDER))
    return [key for key in CSV_FIELD_ORDER if any(key in row for row in rows)] + extra


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _csv_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_raw_csv_from_files(
    rows: list[dict[str, Any]],
    *,
    source_relative_path: str,
    output_path: Path,
) -> None:
    merged_rows: list[dict[str, Any]] = []
    source_parts = source_relative_path.split("/")
    for row in rows:
        source = Path(str(row.get("result_dir", "")))
        for part in source_parts:
            source = source / part
        if not source.exists():
            continue
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for source_row in reader:
                merged_rows.append(
                    {
                        "stage": row.get("stage", ""),
                        "config": row.get("config", ""),
                        "experiment_name": row.get("experiment_name", ""),
                        "run_experiment_name": row.get("run_experiment_name", ""),
                        "seed": row.get("seed", ""),
                        **source_row,
                    }
                )
    if merged_rows:
        _write_csv(merged_rows, output_path)


def _write_model_complexity_csv(rows: list[dict[str, Any]], path: Path) -> None:
    seen: set[tuple[str, str]] = set()
    complexity_rows: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("stage", "")), str(row.get("experiment_name", "")))
        if key in seen:
            continue
        seen.add(key)
        complexity_rows.append(
            {
                "stage": row.get("stage", ""),
                "config": row.get("config", ""),
                "experiment_name": row.get("experiment_name", ""),
                "model_name": row.get("model_name", ""),
                "modality": row.get("modality", ""),
                "stem3d_type": row.get("stem3d_type", ""),
                "spectral_compressor": row.get("spectral_compressor", ""),
                "stem3d_channels": row.get("stem3d_channels", ""),
                "spectral_bins": row.get("spectral_bins", ""),
                "total_params": row.get("total_params", ""),
                "trainable_params": row.get("trainable_params", ""),
                "params_m": row.get("params_m", ""),
                "gflops": row.get("gflops", ""),
                "latency_ms_mean": row.get("latency_ms_mean", ""),
                "latency_ms_std": row.get("latency_ms_std", ""),
                "fps": row.get("fps", ""),
                "peak_memory_mb": row.get("peak_memory_mb", ""),
                "model_weight_size_mb": row.get("model_weight_size_mb", ""),
                "complexity_device": row.get("complexity_device", ""),
                "input_shapes": row.get("input_shapes", ""),
                "network_structure_txt": row.get("network_structure_txt", ""),
                "network_structure_json": row.get("network_structure_json", ""),
                "network_structure_csv": row.get("network_structure_csv", ""),
            }
        )
    if complexity_rows:
        _write_csv(complexity_rows, path)


def _load_manifest_defaults(comparison_root: Path) -> tuple[list[str] | None, list[int] | None]:
    manifest_path = comparison_root / "run_manifest.json"
    if not manifest_path.exists():
        return None, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configs = manifest.get("configs") or None
    seeds = manifest.get("seeds") or None
    if configs:
        configs = [str(c).replace("\\", "/") for c in configs]
    if seeds:
        seeds = [int(s) for s in seeds]
    return configs, seeds


def _reuse_complexity_from_raw(comparison_dir: Path) -> dict[str, dict[str, Any]]:
    path = comparison_dir / "raw_per_run_metrics.csv"
    if not path.exists():
        path = comparison_dir / "raw_model_complexity.csv"
    if not path.exists():
        return {}
    by_config: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            config = str(row.get("config", ""))
            if not config or config in by_config:
                continue
            complexity = {
                key: row.get(key, "")
                for key in [
                    "total_params",
                    "trainable_params",
                    "params_m",
                    "gflops",
                    "latency_ms_mean",
                    "latency_ms_std",
                    "fps",
                    "peak_memory_mb",
                    "model_weight_size_mb",
                    "complexity_device",
                    "input_shapes",
                    "network_structure_txt",
                    "network_structure_json",
                    "network_structure_csv",
                ]
                if row.get(key, "") != ""
            }
            if complexity:
                by_config[config] = complexity
                by_config[config.replace("\\", "/")] = complexity
                by_config[config.replace("/", "\\")] = complexity
    return by_config


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    completed = [
        row
        for row in rows
        if _as_float(row.get("test_acc")) is not None or _as_float(row.get("best_val_acc")) is not None
    ]

    def _rank_key(row: dict[str, Any]) -> float:
        value = _as_float(row.get("test_acc"))
        if value is None:
            value = _as_float(row.get("best_val_acc"))
        return float(value or 0.0)

    best_hsi = max((row for row in completed if row.get("modality") == "hsi"), key=_rank_key, default=None)
    best_fusion = max((row for row in completed if row.get("modality") == "fusion"), key=_rank_key, default=None)
    lines = [
        "# SCI Ablation Comparison",
        "",
        "Primary metrics are **hold-out TEST** Acc / Macro-F1 (from `eval.py --split test`).",
        "`best_val_acc` / `val_acc` are training diagnostics used only for checkpoint selection.",
        "",
        "Stages:",
        "- `S1_rgb`: RGB MobileNetV3 baseline",
        "- `S2_hsi_pure3d`: HSI Pure3D SPA50 baseline",
        "- `S3_hsi_hybrid`: HSI hybrid 3D-2D SPA50 encoder",
        "- `S4_fusion_concat`: direct RGB-HSI concat fusion SPA50",
        "- `S5_gated_fusion`: representative gated RGB-HSI fusion SPA50 baseline",
        "- `S6_cross_attn`: representative cross-attention RGB-HSI fusion SPA50 baseline",
        "- `S7_hsmf_spa50`: HSMF-Net with SPA50",
        "- `S8_hsmf_spa24`: final lightweight HSMF-Net with SPA24",
        "",
    ]
    if best_hsi is not None:
        test_txt = _format_mean_std(
            best_hsi.get("test_acc_mean", best_hsi.get("test_acc", "")),
            best_hsi.get("test_acc_std", ""),
        )
        lines.append(f"- Best HSI: `{best_hsi['experiment_name']}` test_acc={test_txt or f'{_rank_key(best_hsi):.4f}'}")
    if best_fusion is not None:
        test_txt = _format_mean_std(
            best_fusion.get("test_acc_mean", best_fusion.get("test_acc", "")),
            best_fusion.get("test_acc_std", ""),
        )
        lines.append(
            f"- Best Fusion: `{best_fusion['experiment_name']}` test_acc={test_txt or f'{_rank_key(best_fusion):.4f}'}"
        )
    lines.extend(
        [
            "",
            "| Stage | Modality | Experiment | Seeds | Stem | Spectral Compressor | Test Acc | Test Macro-F1 | Delta Acc | Delta Macro-F1 | Best Val Acc | Params(M) | GFLOPs | Latency(ms) | FPS | Peak Mem(MB) | Weight(MB) |",
            "|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        delta_acc = row.get("delta_acc", row.get("delta_vs_baseline", ""))
        delta_acc_text = "" if delta_acc == "" else f"{float(delta_acc):+.4f}"
        delta_macro = row.get("delta_macro_f1", "")
        delta_macro_text = "" if delta_macro == "" else f"{float(delta_macro):+.4f}"
        test_acc_text = _format_mean_std(
            row.get("test_acc_mean", row.get("test_acc", "")),
            row.get("test_acc_std", ""),
        )
        test_macro_text = _format_mean_std(
            row.get("test_macro_f1_mean", row.get("test_macro_f1", row.get("macro_f1", ""))),
            row.get("test_macro_f1_std", row.get("macro_f1_std", "")),
        )
        best_text = _format_mean_std(
            row.get("best_val_acc_mean", row.get("best_val_acc", "")),
            row.get("best_val_acc_std", ""),
        )
        latency_text = _format_mean_std(row.get("latency_ms_mean", ""), row.get("latency_ms_std", ""))
        params = row.get("params_m", "")
        params_text = "" if params == "" else f"{float(params):.3f}"
        fps = row.get("fps", "")
        fps_text = "" if fps == "" else f"{float(fps):.1f}"
        peak_mem = row.get("peak_memory_mb", "")
        peak_mem_text = "" if peak_mem == "" else f"{float(peak_mem):.1f}"
        weight_mb = row.get("model_weight_size_mb", "")
        weight_text = "" if weight_mb == "" else f"{float(weight_mb):.2f}"
        lines.append(
            "| {stage} | {modality} | {experiment_name} | {seed_count} | {stem3d_type} | "
            "{spectral_compressor} | "
            f"{test_acc_text} | {test_macro_text} | {delta_acc_text} | {delta_macro_text} | {best_text} | "
            f"{params_text} | {row.get('gflops', '')} | {latency_text} | {fps_text} | {peak_mem_text} | {weight_text} |".format(
                **{**row, "seed_count": row.get("seed_count", "")}
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.summarize_only:
        if not args.comparison_id:
            raise SystemExit("--summarize-only requires --comparison-id pointing at an existing artifacts/ folder.")
        args.train = False
        args.eval = False

    comparison_id = args.comparison_id or datetime.now().strftime("hsi_architecture_ablation_%Y-%m-%d_%H-%M")
    comparison_root = PROJECT_ROOT / "artifacts" / comparison_id
    comparison_dir = comparison_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize_only and (not args.configs or not args.seeds):
        manifest_configs, manifest_seeds = _load_manifest_defaults(comparison_root)
        if not args.configs:
            args.configs = manifest_configs
        if not args.seeds:
            args.seeds = manifest_seeds
        if not args.configs:
            raise SystemExit(
                f"No configs found. Pass --configs or ensure {comparison_root / 'run_manifest.json'} exists."
            )

    configs = _selected_configs(args)
    seeds = args.seeds if args.seeds else [None]
    suffix_seed_runs = len([s for s in seeds if s is not None]) > 1
    if args.summarize_only and args.seeds and len(args.seeds) >= 1:
        suffix_seed_runs = True

    raw_rows: list[dict[str, Any]] = []
    complexity_by_config: dict[str, dict[str, Any]] = {}
    reused_complexity = _reuse_complexity_from_raw(comparison_dir) if args.summarize_only else {}
    skip_complexity = bool(args.skip_complexity) or bool(args.summarize_only)
    manifest: dict[str, Any] = {
        "comparison_id": comparison_id,
        "ablation_type": "hsi_architecture",
        "stage": args.stage,
        "train": bool(args.train),
        "eval": bool(args.eval),
        "summarize_only": bool(args.summarize_only),
        "dry_run": bool(args.dry_run),
        "seeds": [seed for seed in seeds if seed is not None],
        "epochs_override": args.epochs,
        "seed_policy": "Top-level training seed changes across repeats; data.split_seed is kept from each config.",
        "primary_metrics": ["test_acc", "test_macro_f1"],
        "complexity": {
            "enabled": not skip_complexity,
            "device": args.complexity_device,
            "latency_warmup": args.latency_warmup,
            "latency_iters": args.latency_iters,
            "reused_from_existing": bool(args.summarize_only),
        },
        "configs": [],
        "raw_results": [],
        "aggregate_results": [],
    }

    print("HSI architecture ablation suite")
    print(f"  comparison_id: {comparison_id}")
    print(f"  stage: {args.stage}")
    print(f"  summarize_only: {bool(args.summarize_only)}")
    print(f"  experiments: {len(configs)}")
    print(f"  seeds: {', '.join(str(seed) for seed in seeds if seed is not None) or 'config default'}")
    print(f"  epochs: {args.epochs if args.epochs is not None else 'config default'}")
    for config_arg in configs:
        print(f"    - {config_arg}")
    print("")

    for config_arg in configs:
        config_path = _resolve_config(config_arg)
        cfg = load_config(config_path)
        base_exp_name = str(cfg.get("experiment_name", config_path.stem))
        rel_config = str(config_path.relative_to(PROJECT_ROOT))
        manifest["configs"].append(rel_config)

        complexity: dict[str, Any] = {}
        if not skip_complexity and not args.dry_run:
            complexity = _complexity_row(
                cfg,
                device_text=args.complexity_device,
                latency_warmup=args.latency_warmup,
                latency_iters=args.latency_iters,
                structure_output_dir=comparison_dir / "model_structures",
            )
            complexity_by_config[rel_config] = complexity
        elif args.summarize_only:
            complexity = (
                reused_complexity.get(rel_config)
                or reused_complexity.get(rel_config.replace("\\", "/"))
                or reused_complexity.get(rel_config.replace("/", "\\"))
                or {}
            )
            if complexity:
                complexity_by_config[rel_config] = complexity

        for seed in seeds:
            run_config_path = _write_comparison_config(
                cfg,
                config_path,
                comparison_root,
                seed=seed,
                epochs=args.epochs,
                suffix_experiment_name=suffix_seed_runs,
            )
            run_cfg = load_config(run_config_path)
            run_exp_name = str(run_cfg.get("experiment_name", base_exp_name))
            exp_dir = comparison_root / run_exp_name

            train_cmd = [sys.executable, "scripts/train.py", "--config", str(run_config_path)]
            if args.auto_resume:
                train_cmd.append("--auto-resume")
            eval_cmd = [
                sys.executable,
                "scripts/eval.py",
                "--config",
                str(run_config_path),
                "--ckpt",
                str(exp_dir / "checkpoints" / "best.pt"),
                "--tsne-max-samples",
                str(args.tsne_max_samples),
            ]

            if args.train:
                if args.dry_run:
                    print("+ " + " ".join(train_cmd))
                else:
                    _run_command(train_cmd)
            elif args.dry_run and not args.summarize_only:
                print(f"[DRY-RUN] skip train for {run_exp_name}")

            ckpt_path = exp_dir / "checkpoints" / "best.pt"
            if args.eval:
                if args.dry_run:
                    print("+ " + " ".join(eval_cmd))
                elif ckpt_path.exists():
                    _run_command(eval_cmd)
                else:
                    print(f"[WARN] Skip eval because checkpoint is missing: {ckpt_path}")

            if args.summarize_only and not exp_dir.exists():
                print(f"[WARN] Missing experiment dir, skip: {exp_dir}")
                continue

            row = _experiment_row(
                config_path,
                run_cfg,
                exp_dir,
                seed=seed,
                base_experiment_name=base_exp_name,
                complexity=complexity,
            )
            raw_rows.append(row)
            manifest["raw_results"].append(row)

    _add_baseline_deltas(raw_rows)
    aggregate_rows = _aggregate_seed_rows(raw_rows)
    manifest["raw_results"] = raw_rows
    manifest["aggregate_results"] = aggregate_rows
    manifest["complexity_by_config"] = complexity_by_config

    _write_csv(aggregate_rows, comparison_dir / "sci_ablation_comparison.csv")
    _write_markdown(aggregate_rows, comparison_dir / "sci_ablation_comparison.md")
    _write_csv(aggregate_rows, comparison_dir / "hsi_architecture_comparison.csv")
    _write_markdown(aggregate_rows, comparison_dir / "hsi_architecture_comparison.md")
    _write_csv(raw_rows, comparison_dir / "raw_per_run_metrics.csv")
    _write_model_complexity_csv(aggregate_rows, comparison_dir / "raw_model_complexity.csv")
    _write_raw_csv_from_files(
        raw_rows,
        source_relative_path="history_epoch_table.csv",
        output_path=comparison_dir / "raw_epoch_history.csv",
    )
    _write_raw_csv_from_files(
        raw_rows,
        source_relative_path="figures/per_class_prf.csv",
        output_path=comparison_dir / "raw_per_class_prf.csv",
    )
    _write_raw_csv_from_files(
        raw_rows,
        source_relative_path="figures/top_confusion_pairs.csv",
        output_path=comparison_dir / "raw_confusion_pairs.csv",
    )
    _write_raw_csv_from_files(
        raw_rows,
        source_relative_path="figures/confusion_matrix_raw.csv",
        output_path=comparison_dir / "raw_confusion_matrix.csv",
    )
    _write_raw_csv_from_files(
        raw_rows,
        source_relative_path="figures/eval_predictions.csv",
        output_path=comparison_dir / "raw_eval_predictions.csv",
    )
    (comparison_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not args.dry_run:
        try:
            export_comparison_extra_data(comparison_root)
        except Exception as exc:
            print(f"[WARN] Failed to export comparison extra data: {exc}")
    shutil.copy2(Path(__file__), comparison_root / "run_hsi_architecture_ablation.py")
    print(f"Comparison written to: {comparison_root}")


if __name__ == "__main__":
    main()
