from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

DEFAULT_EXPERIMENT_CONFIGS = [
    "configs/experiments/sci_s1_rgb_baseline.yaml",
    "configs/experiments/sci_s2_hsi_pure3d_spa50.yaml",
    "configs/experiments/sci_s3_hsi_hybrid_spa50.yaml",
    "configs/experiments/sci_s4_fusion_hybrid_concat_spa50.yaml",
    "configs/experiments/sci_s5_fusion_hybrid_gated_spa50.yaml",
    "configs/experiments/sci_s6_fusion_hybrid_cross_attn_spa50.yaml",
    "configs/experiments/sci_s7_fusion_hybrid_midfilm_spa50.yaml",
    "configs/experiments/sci_s8_fusion_hybrid_midfilm_spa24.yaml",
]


def _module_param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters(recurse=False))


def _shorten_text(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    return "..." + text[-(width - 3) :]


def _module_arg_text(module: nn.Module) -> str:
    if isinstance(module, (nn.Conv2d, nn.Conv3d)):
        return str(
            [
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.groups,
            ]
        )
    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
        return str([module.num_features])
    if isinstance(module, nn.Linear):
        return str([module.in_features, module.out_features, module.bias is not None])
    if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)):
        return str([module.output_size])
    if isinstance(module, (nn.MaxPool2d, nn.MaxPool3d)):
        return str([module.kernel_size, module.stride])
    if isinstance(module, nn.Dropout):
        return str([module.p])
    if isinstance(module, nn.MultiheadAttention):
        return str([module.embed_dim, module.num_heads, module.dropout])
    if isinstance(module, (nn.Sequential, nn.ModuleList)):
        return str([len(module)])
    return "[]"


def _module_full_name(module: nn.Module) -> str:
    return f"{module.__class__.__module__}.{module.__class__.__name__}"


def architecture_rows(module: nn.Module, prefix: str = "") -> list[tuple[str, nn.Module]]:
    rows: list[tuple[str, nn.Module]] = []
    for name, child in module.named_children():
        path = f"{prefix}.{name}" if prefix else name
        if any(child.named_children()):
            rows.extend(architecture_rows(child, path))
            continue
        rows.append((path, child))
    if not rows:
        rows = [(name, module) for name, module in module.named_modules() if name]
    return rows


def _input_shapes(cfg: dict[str, Any]) -> list[str]:
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    model_name = str(model_cfg.get("name", "")).lower()
    image_size = int(data_cfg.get("image_size", 224))
    hsi_bands = int(data_cfg.get("hsi_bands", model_cfg.get("hsi_bands", model_cfg.get("fusion", {}).get("hsi_bands", 1))))
    if model_name in {"v3_3d", "hsi_pure3d", "hsi_hybridsn"}:
        return [f"(B, 1, {hsi_bands}, {image_size}, {image_size})"]
    if model_name in {
        "rgb_hsi_fusion",
        "rgb_hsi_attention",
        "rgb_hsi_mid_film",
    }:
        hsi_size = int(data_cfg.get("hsi_image_size", image_size))
        return [
            f"(B, 3, {image_size}, {image_size})",
            f"(B, 1, {hsi_bands}, {hsi_size}, {hsi_size})",
        ]
    return [f"(B, 3, {image_size}, {image_size})"]


def collect_network_structure(model: nn.Module, cfg: dict[str, Any]) -> dict[str, Any]:
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    exp_name = cfg.get("experiment_name", "default_exp")
    model_name = str(model_cfg.get("name", model.__class__.__name__))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    layers = []
    for idx, (path, module) in enumerate(architecture_rows(model)):
        layers.append(
            {
                "idx": idx,
                "path": path,
                "module": _module_full_name(module),
                "params": _module_param_count(module),
                "arguments": _module_arg_text(module),
            }
        )
    return {
        "experiment_name": exp_name,
        "model_name": model_name,
        "model_class": model.__class__.__name__,
        "variant": model_cfg.get("variant", model_cfg.get("fusion", {}).get("type", "")),
        "num_classes": model_cfg.get("num_classes", 4),
        "input_shapes": _input_shapes(cfg),
        "image_size": data_cfg.get("image_size", 224),
        "summary": {
            "layer_count": sum(1 for _ in model.modules()),
            "leaf_layers": len(layers),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": total_params - trainable_params,
        },
        "layers": layers,
    }


def format_network_structure_table(model: nn.Module, cfg: dict[str, Any]) -> str:
    spec = collect_network_structure(model, cfg)
    model_name = spec["model_name"]
    exp_name = spec["experiment_name"]
    lines = [
        "",
        f"{model_name} architecture: {exp_name}",
        f"{'idx':>4} {'from/name':<34}{'n':>4}{'params':>12}  {'module':<62}{'arguments'}",
    ]
    for layer in spec["layers"]:
        lines.append(
            f"{layer['idx']:>4} {_shorten_text(layer['path'], 34):<34}{1:>4}{layer['params']:>12}  "
            f"{_shorten_text(layer['module'], 62):<62}{layer['arguments']}"
        )
    summary = spec["summary"]
    lines.append(
        f"{model_name} summary: {summary['layer_count']} layers, "
        f"{summary['total_params']:,} parameters, {summary['trainable_params']:,} gradients"
    )
    return "\n".join(lines)


def print_network_structure(model: nn.Module, cfg: dict[str, Any]) -> None:
    print(format_network_structure_table(model, cfg))


def estimate_model_gflops(model: nn.Module, cfg: dict[str, Any], device: torch.device) -> str:
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    model_name = str(model_cfg.get("name", "")).lower()
    image_size = int(data_cfg.get("image_size", 224))
    try:
        from thop import profile

        profile_model = copy.deepcopy(model).to(device)
        profile_model.eval()
        hsi_bands = int(data_cfg.get("hsi_bands", model_cfg.get("hsi_bands", 1)))
        if model_name in {"v3_3d", "hsi_pure3d", "hsi_hybridsn"}:
            dummy_inputs = (torch.randn(1, 1, hsi_bands, image_size, image_size, device=device),)
        elif model_name in {
            "rgb_hsi_fusion",
            "rgb_hsi_attention",
            "rgb_hsi_mid_film",
        }:
            hsi_size = int(data_cfg.get("hsi_image_size", image_size))
            dummy_inputs = (
                torch.randn(1, 3, image_size, image_size, device=device),
                torch.randn(1, 1, hsi_bands, hsi_size, hsi_size, device=device),
            )
        else:
            dummy_inputs = (torch.randn(1, 3, image_size, image_size, device=device),)
        flops, _ = profile(profile_model, inputs=dummy_inputs, verbose=False)
        # 4 decimals: fusion concat vs cross-attn differ only ~0.0007G and round equal at 3dp.
        return f"{flops / 1e9:.4f}"
    except Exception:
        return "N/A"


def format_model_summary(model: nn.Module, cfg: dict[str, Any], device: torch.device) -> str:
    spec = collect_network_structure(model, cfg)
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    model_name = model_cfg.get("name", "unknown")
    variant = model_cfg.get("variant", model_cfg.get("fusion", {}).get("type", "-"))
    num_classes = model_cfg.get("num_classes", 100)
    image_size = data_cfg.get("image_size", 32)
    total_params = spec["summary"]["total_params"]
    trainable_params = spec["summary"]["trainable_params"]
    frozen_params = spec["summary"]["frozen_params"]
    layers = spec["summary"]["layer_count"]
    params_m = total_params / 1e6
    gflops_text = estimate_model_gflops(model, cfg, device)
    input_shapes = ", ".join(spec["input_shapes"])
    lines = [
        "",
        "Model Summary:",
        (
            f"  {'Model':<16}{'Variant':<12}{'Device':<10}{'Image':<10}{'Classes':<10}"
            f"{'Layers':<10}{'#Params':<12}{'Trainable':<12}{'FLOPs':<10}"
        ),
        (
            f"  {f'MobileNet-{str(model_name).upper()}':<16}{str(variant):<12}{str(device):<10}"
            f"{f'{image_size}x{image_size}':<10}{num_classes:<10}{layers:<10}"
            f"{f'{params_m:.3f}M':<12}{f'{trainable_params / 1e6:.3f}M':<12}{f'{gflops_text}G':<10}"
        ),
        (
            f"  {'(raw)':<58}params={total_params:,}, trainable={trainable_params:,}, "
            f"frozen={frozen_params:,}"
        ),
        f"  inputs: {input_shapes}",
        "",
    ]
    return "\n".join(lines)


def print_model_summary(model: nn.Module, cfg: dict[str, Any], device: torch.device) -> None:
    print(format_model_summary(model, cfg, device))


def export_network_structure(
    model: nn.Module,
    cfg: dict[str, Any],
    output_dir: Path,
    *,
    formats: tuple[str, ...] = ("txt", "json", "csv"),
    device: torch.device | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = collect_network_structure(model, cfg)
    exp_name = str(spec["experiment_name"])
    if device is not None:
        spec["gflops"] = estimate_model_gflops(model, cfg, device)
    paths: dict[str, Path] = {}

    if "txt" in formats:
        txt_path = output_dir / f"{exp_name}_network_structure.txt"
        txt_path.write_text(
            format_network_structure_table(model, cfg)
            + ("\n" + format_model_summary(model, cfg, device) if device is not None else ""),
            encoding="utf-8",
        )
        paths["txt"] = txt_path

    if "json" in formats:
        json_path = output_dir / f"{exp_name}_network_structure.json"
        json_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
        paths["json"] = json_path

    if "csv" in formats:
        csv_path = output_dir / f"{exp_name}_network_structure.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["idx", "path", "module", "params", "arguments"],
            )
            writer.writeheader()
            writer.writerows(spec["layers"])
        paths["csv"] = csv_path

    return paths


def _resolve_config(path: str, project_root: Path) -> Path:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    return cfg_path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def dump_models_from_configs(
    config_paths: list[str],
    *,
    output_dir: Path | None = None,
    formats: tuple[str, ...] = ("txt", "json", "csv"),
    device_name: str = "cpu",
) -> list[dict[str, Any]]:
    from mobilenet_enterprise.registry import build_model
    from mobilenet_enterprise.utils.config import load_config
    from mobilenet_enterprise.utils.runtime import resolve_device

    project_root = _project_root()
    device = resolve_device(device_name)
    results: list[dict[str, Any]] = []

    for config_arg in config_paths:
        config_path = _resolve_config(config_arg, project_root)
        cfg = load_config(config_path)
        model = build_model(cfg).to(device)
        print_network_structure(model, cfg)
        print_model_summary(model, cfg, device)

        export_paths: dict[str, Path] = {}
        if output_dir is not None:
            export_paths = export_network_structure(
                model,
                cfg,
                output_dir,
                formats=formats,
                device=device,
            )
            print(f"Exported: {', '.join(str(path) for path in export_paths.values())}")

        results.append(
            {
                "config": str(config_path),
                "structure": collect_network_structure(model, cfg),
                "export_paths": {key: str(path) for key, path in export_paths.items()},
            }
        )
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print detailed network structure for ablation models (YOLO-style table)."
    )
    parser.add_argument(
        "--config",
        nargs="*",
        default=None,
        help="Experiment YAML path(s). Defaults to --all ablation configs when omitted.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use all four ablation experiment configs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to export txt/json/csv structure files.",
    )
    parser.add_argument(
        "--formats",
        nargs="*",
        default=("txt", "json", "csv"),
        choices=("txt", "json", "csv"),
        help="Export formats when --output-dir is set.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for GFLOPs estimation (default: cpu).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.config:
        config_paths = args.config
    elif args.all or not args.config:
        config_paths = DEFAULT_EXPERIMENT_CONFIGS
    else:
        config_paths = DEFAULT_EXPERIMENT_CONFIGS

    output_dir = Path(args.output_dir) if args.output_dir else None
    dump_models_from_configs(
        config_paths,
        output_dir=output_dir,
        formats=tuple(args.formats),
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
