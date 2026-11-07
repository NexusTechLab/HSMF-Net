from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from torch import nn


@dataclass
class DiagramNode:
    id: str
    label: str
    subtitle: str
    group: str
    params: int
    kind: str
    x: float = 0.0
    y: float = 0.0


@dataclass
class DiagramSpec:
    title: str
    nodes: list[DiagramNode]
    edges: list[tuple[str, str]]


KIND_COLORS = {
    "input": "#F7F7F7",
    "stem": "#DDEAF7",
    "block": "#EAF4EA",
    "spectral": "#EFE6F7",
    "pool": "#F5E8D7",
    "fusion": "#F7E1E1",
    "head": "#E8E8E8",
}


def _params(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return sum(int(p.numel()) for p in module.parameters())


def _format_params(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def _model_title(cfg: dict[str, Any], model: nn.Module) -> str:
    model_cfg = cfg.get("model", {})
    exp_name = cfg.get("experiment_name", "network")
    model_name = str(model_cfg.get("name", model.__class__.__name__))
    return f"{exp_name} ({model_name})"


def _node(
    node_id: str,
    label: str,
    subtitle: str,
    group: str,
    kind: str,
    module: nn.Module | None = None,
) -> DiagramNode:
    return DiagramNode(
        id=node_id,
        label=label,
        subtitle=subtitle,
        group=group,
        kind=kind,
        params=_params(module),
    )


def _rgb_nodes(model: nn.Module) -> tuple[list[DiagramNode], list[tuple[str, str]]]:
    features = getattr(model, "features", None)
    classifier = getattr(model, "classifier", None)
    nodes = [_node("input", "RGB image", "3 x H x W", "RGB", "input")]
    edges: list[tuple[str, str]] = []
    if features is None:
        nodes.append(_node("backbone", "MobileNetV3", "feature extractor", "RGB", "block", model))
        edges.append(("input", "backbone"))
    else:
        nodes.append(_node("stem", "Stem", "3x3 conv, stride 2", "RGB", "stem", features[0]))
        edges.append(("input", "stem"))
        previous = "stem"
        block_count = max(0, len(features) - 2)
        for idx in range(1, len(features) - 1):
            node_id = f"ir_{idx}"
            nodes.append(
                _node(
                    node_id,
                    f"IR Block {idx}",
                    "MobileNetV3 inverted residual",
                    "RGB",
                    "block",
                    features[idx],
                )
            )
            edges.append((previous, node_id))
            previous = node_id
        nodes.append(
            _node(
                "head_conv",
                "Head conv",
                f"after {block_count} inverted residual blocks",
                "RGB",
                "stem",
                features[-1],
            )
        )
        edges.append((previous, "head_conv"))
        previous = "head_conv"
        nodes.append(_node("gap", "Global average pool", "1 x 1 descriptor", "RGB", "pool"))
        edges.append((previous, "gap"))
        previous = "gap"
        nodes.append(_node("classifier", "Classifier", "linear head", "RGB", "head", classifier))
        edges.append((previous, "classifier"))
    return nodes, edges


def _hsi_nodes(model: nn.Module) -> tuple[list[DiagramNode], list[tuple[str, str]]]:
    encoder3d = getattr(model, "encoder3d", None)
    spectral_filter = getattr(model, "spectral_filter", None)
    spectral_compressor = getattr(model, "spectral_compressor", None)
    features2d = getattr(model, "features2d", None)
    classifier = getattr(model, "classifier", None)
    bands = getattr(model, "hsi_bands", "B")
    spectral_bins = getattr(model, "spectral_bins", "-")
    stem3d_type = getattr(model, "stem3d_type", "standard")
    compressor_type = getattr(model, "spectral_compressor_type", "adaptive_pool")
    stem_label = {
        "standard": "Conv3D encoder",
        "pseudo_3d": "(2+1)D pseudo-3D encoder",
        "mbconv3d": "MBConv3D depthwise encoder",
    }.get(str(stem3d_type), "Conv3D encoder")
    bin_label = {
        "adaptive_pool": f"adaptive avg/max to {spectral_bins} bins",
        "query_attn": f"learnable query attn to {spectral_bins} bins",
    }.get(str(compressor_type), f"spectral compression to {spectral_bins} bins")
    nodes = [
        _node("input", "HSI cube", f"1 x {bands} x H x W", "HSI", "input"),
        _node("stem3d", "3D spectral-spatial stem", stem_label, "HSI", "stem", encoder3d),
        _node("spectral_filter", "Spectral filter", "depthwise Conv3D", "HSI", "spectral", spectral_filter),
        _node("spectral_bins", "Spectral binning", bin_label, "HSI", "spectral", spectral_compressor),
    ]
    edges = [("input", "stem3d"), ("stem3d", "spectral_filter"), ("spectral_filter", "spectral_bins")]
    previous = "spectral_bins"
    if features2d is not None:
        for idx, block in enumerate(features2d):
            node_id = f"mbconv_{idx + 1}"
            label = "2D MBConv" if idx < len(features2d) - 1 else "Head conv"
            subtitle = "MobileNetV3-style 2D block" if idx < len(features2d) - 1 else "1x1 projection"
            nodes.append(_node(node_id, f"{label} {idx + 1}", subtitle, "HSI", "block", block))
            edges.append((previous, node_id))
            previous = node_id
    nodes.append(_node("gap", "Global average pool", "1 x 1 descriptor", "HSI", "pool"))
    edges.append((previous, "gap"))
    nodes.append(_node("classifier", "Classifier", "linear head", "HSI", "head", classifier))
    edges.append(("gap", "classifier"))
    return nodes, edges


def _fusion_nodes(model: nn.Module) -> tuple[list[DiagramNode], list[tuple[str, str]]]:
    fusion_type = str(getattr(model, "fusion_type", "concat")).replace("_", " ").title()
    nodes = [
        _node("rgb_input", "RGB image", "3 x H x W", "RGB branch", "input"),
        _node("rgb_backbone", "RGB MobileNetV3", "2D feature extractor", "RGB branch", "block", model.rgb_branch),
        _node("rgb_proj", "RGB projection", "linear embedding", "RGB branch", "head", model.rgb_proj),
        _node("hsi_input", "HSI cube", "1 x B x H x W", "HSI branch", "input"),
        _node("hsi_backbone", "HSI MobileNetV3-3D", "3D to 2D spectral encoder", "HSI branch", "block", model.hsi_branch),
        _node("hsi_proj", "HSI projection", "linear embedding", "HSI branch", "head", model.hsi_proj),
        _node(
            "fusion",
            fusion_type,
            "cross-modal fusion",
            "Fusion",
            "fusion",
            getattr(model, "late_film", None)
            or getattr(model, "film", None)
            or getattr(model, "attention", None),
        ),
        _node("classifier", "Classifier", "final prediction", "Classifier", "head", model.classifier),
    ]
    edges = [
        ("rgb_input", "rgb_backbone"),
        ("rgb_backbone", "rgb_proj"),
        ("hsi_input", "hsi_backbone"),
        ("hsi_backbone", "hsi_proj"),
        ("rgb_proj", "fusion"),
        ("hsi_proj", "fusion"),
        ("fusion", "classifier"),
    ]
    return nodes, edges


def build_diagram_spec(model: nn.Module, cfg: dict[str, Any]) -> DiagramSpec:
    if hasattr(model, "rgb_branch") and hasattr(model, "hsi_branch"):
        nodes, edges = _fusion_nodes(model)
    elif hasattr(model, "encoder3d") and hasattr(model, "features2d"):
        nodes, edges = _hsi_nodes(model)
    else:
        nodes, edges = _rgb_nodes(model)
    return DiagramSpec(title=_model_title(cfg, model), nodes=nodes, edges=edges)


def _layout_sequential(spec: DiagramSpec) -> None:
    max_per_row = 7
    x_gap = 2.7
    y_gap = 1.8
    for idx, node in enumerate(spec.nodes):
        row = idx // max_per_row
        col = idx % max_per_row
        if row % 2 == 1:
            col = max_per_row - 1 - col
        node.x = col * x_gap
        node.y = -row * y_gap


def _layout_fusion(spec: DiagramSpec) -> None:
    positions = {
        "rgb_input": (0.0, 1.25),
        "rgb_backbone": (2.8, 1.25),
        "rgb_proj": (5.6, 1.25),
        "hsi_input": (0.0, -1.25),
        "hsi_backbone": (2.8, -1.25),
        "hsi_proj": (5.6, -1.25),
        "fusion": (8.4, 0.0),
        "classifier": (11.2, 0.0),
    }
    for node in spec.nodes:
        node.x, node.y = positions[node.id]


def layout_diagram(spec: DiagramSpec) -> DiagramSpec:
    if any(node.id == "fusion" for node in spec.nodes):
        _layout_fusion(spec)
    else:
        _layout_sequential(spec)
    return spec


def _node_by_id(spec: DiagramSpec) -> dict[str, DiagramNode]:
    return {node.id: node for node in spec.nodes}


def render_publication_figure(spec: DiagramSpec, output_base: str | Path, formats: tuple[str, ...]) -> None:
    layout_diagram(spec)
    xs = [node.x for node in spec.nodes]
    ys = [node.y for node in spec.nodes]
    width = max(9.0, max(xs) - min(xs) + 3.5)
    height = max(3.8, max(ys) - min(ys) + 2.8)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.axis("off")
    node_lookup = _node_by_id(spec)
    box_w = 2.25
    box_h = 0.82

    for start_id, end_id in spec.edges:
        start = node_lookup[start_id]
        end = node_lookup[end_id]
        arrow = FancyArrowPatch(
            (start.x + box_w / 2, start.y),
            (end.x - box_w / 2, end.y),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.05,
            color="#3A3A3A",
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(arrow)

    for node in spec.nodes:
        color = KIND_COLORS.get(node.kind, "#EAEAEA")
        rect = FancyBboxPatch(
            (node.x - box_w / 2, node.y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.025,rounding_size=0.08",
            linewidth=1.1,
            edgecolor="#222222",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(node.x, node.y + 0.15, node.label, ha="center", va="center", weight="bold", fontsize=8.5)
        ax.text(node.x, node.y - 0.06, node.subtitle, ha="center", va="center", fontsize=6.7, color="#3B3B3B")
        ax.text(
            node.x,
            node.y - 0.27,
            f"{_format_params(node.params)} params",
            ha="center",
            va="center",
            fontsize=6.4,
            color="#555555",
        )

    ax.text(min(xs) - 1.1, max(ys) + 0.9, spec.title, ha="left", va="center", fontsize=11, weight="bold")
    ax.set_xlim(min(xs) - 1.5, max(xs) + 1.5)
    ax.set_ylim(min(ys) - 1.2, max(ys) + 1.25)
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def export_dot(spec: DiagramSpec, path: str | Path) -> None:
    layout_diagram(spec)
    lines = [
        "digraph network {",
        '  graph [rankdir=LR, bgcolor="white", splines=ortho];',
        '  node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10, color="#222222"];',
        '  edge [color="#3A3A3A", arrowsize=0.7];',
    ]
    for node in spec.nodes:
        label = f"{node.label}\\n{node.subtitle}\\n{_format_params(node.params)} params"
        color = KIND_COLORS.get(node.kind, "#EAEAEA")
        lines.append(f'  {node.id} [label="{label}", fillcolor="{color}"];')
    for start, end in spec.edges:
        lines.append(f"  {start} -> {end};")
    lines.append("}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_json(spec: DiagramSpec, path: str | Path) -> None:
    layout_diagram(spec)
    payload = {
        "title": spec.title,
        "nodes": [asdict(node) for node in spec.nodes],
        "edges": [{"source": start, "target": end} for start, end in spec.edges],
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def export_network_diagram(
    model: nn.Module,
    cfg: dict[str, Any],
    output_dir: str | Path,
    stem: str | None = None,
    formats: tuple[str, ...] = ("svg", "pdf", "png"),
) -> dict[str, str]:
    spec = build_diagram_spec(model, cfg)
    exp_name = str(cfg.get("experiment_name", "network"))
    output_stem = stem or f"{exp_name}_network"
    output_base = Path(output_dir) / output_stem
    render_publication_figure(spec, output_base, formats=formats)
    dot_path = output_base.with_name(f"{output_stem}_visio_source.dot")
    json_path = output_base.with_name(f"{output_stem}_visio_source.json")
    export_dot(spec, dot_path)
    export_json(spec, json_path)
    svg_path = output_base.with_suffix(".svg")
    visio_svg_path = output_base.with_name(f"{output_stem}_visio.svg")
    if svg_path.exists():
        visio_svg_path.write_text(svg_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {
        "svg": str(svg_path),
        "pdf": str(output_base.with_suffix(".pdf")),
        "png": str(output_base.with_suffix(".png")),
        "visio_svg": str(visio_svg_path),
        "visio_dot": str(dot_path),
        "visio_json": str(json_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export publication-ready network architecture diagrams.")
    parser.add_argument("--config", required=True, help="Experiment YAML config.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to artifacts/<exp>/figures.")
    parser.add_argument("--stem", default=None, help="Output file stem.")
    parser.add_argument("--formats", nargs="*", default=["svg", "pdf", "png"], help="Figure formats.")
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from mobilenet_enterprise.registry import build_model
    from mobilenet_enterprise.utils.config import load_config

    args = parse_args()
    cfg = load_config(args.config)
    model = build_model(cfg)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(cfg.get("output_dir", "artifacts")) / str(cfg.get("experiment_name", "network")) / "figures"
    paths = export_network_diagram(
        model=model,
        cfg=cfg,
        output_dir=output_dir,
        stem=args.stem,
        formats=tuple(args.formats),
    )
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
