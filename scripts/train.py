from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilenet_enterprise.data import build_datamodule
from mobilenet_enterprise.engine import Trainer
from mobilenet_enterprise.registry import build_model
from mobilenet_enterprise.utils import (
    ensure_dirs,
    load_config,
    load_imagenet_pretrained_weights,
    print_model_summary,
    print_network_structure,
    resolve_device,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNet locally.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for resume training.")
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Auto resume from artifacts/<experiment_name>/checkpoints/last.pt if exists.",
    )
    parser.add_argument(
        "--finetune-from",
        type=str,
        default=None,
        help="Load model/EMA weights only (reset optimizer/epoch) for clean fine-tuning.",
    )
    return parser.parse_args()


def print_dataset_summary(cfg: dict, train_loader, val_loader, test_loader=None) -> None:
    data_cfg = cfg.get("data", {})
    data_name = data_cfg.get("name", "unknown")
    data_root = data_cfg.get("root", "")
    train_samples = len(getattr(train_loader, "dataset", []))
    val_samples = len(getattr(val_loader, "dataset", []))
    test_samples = len(getattr(test_loader, "dataset", [])) if test_loader is not None else 0
    classes = getattr(train_loader.dataset, "classes", None)
    num_classes = len(classes) if classes is not None else "N/A"
    print("Dataset Summary:")
    print(
        f"  {'Name':<16}{'Root':<45}{'Classes':<10}{'Train':<12}{'Val':<12}{'Test':<12}{'Batch':<10}"
    )
    print(
        f"  {str(data_name):<16}{str(data_root):<45}{str(num_classes):<10}"
        f"{train_samples:<12}{val_samples:<12}{test_samples:<12}{data_cfg.get('batch_size', 'N/A'):<10}"
    )
    if classes is not None:
        preview = ", ".join(classes[:10])
        if len(classes) > 10:
            preview += ", ..."
        print(f"  classes: {preview}")
    print(
        f"  split ratios: val_ratio={data_cfg.get('val_ratio', 'N/A')}, "
        f"test_ratio={data_cfg.get('test_ratio', 'N/A')}, split_seed={data_cfg.get('split_seed', 'N/A')}"
    )
    print("")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42), deterministic=cfg.get("train", {}).get("deterministic", False))
    device = resolve_device(cfg.get("device", "auto"))
    output = ensure_dirs(cfg.get("output_dir", "artifacts"))
    exp_name = cfg.get("experiment_name", "default_exp")
    exp_dir = output["root"] / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    ensure_dirs(exp_dir)
    with (exp_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    model = build_model(cfg).to(device)
    resume_path: Path | None = None
    finetune_path: Path | None = None
    if args.finetune_from:
        finetune_path = Path(args.finetune_from)
    elif args.resume:
        resume_path = Path(args.resume)
    elif args.auto_resume:
        candidate = exp_dir / "checkpoints" / "last.pt"
        if candidate.exists():
            resume_path = candidate

    pretrained_cfg = cfg.get("pretrained", {})
    pretrained_enabled = bool(pretrained_cfg.get("enabled", False))
    pretrained_source = str(pretrained_cfg.get("source", "unknown"))
    use_pretrained = pretrained_enabled and resume_path is None and finetune_path is None
    preload = (
        load_imagenet_pretrained_weights(model, cfg)
        if use_pretrained
        else {"loaded": False, "matched_layers": 0, "matched_params": 0, "matched_ratio": 0.0}
    )
    if not pretrained_enabled:
        print("Pretrained: disabled by config (pretrained.enabled=false).")
    elif finetune_path is not None:
        print(f"Pretrained: skipped due to fine-tune weights ({finetune_path}).")
    elif resume_path is not None:
        print(f"Pretrained: skipped due to resume checkpoint ({resume_path}).")
    else:
        print(
            "Pretrained ImageNet weights loaded: "
            f"{preload['loaded']} "
            f"(source={pretrained_source}, "
            f"matched_layers={preload['matched_layers']}, "
            f"matched_params={preload['matched_params'] / 1e6:.3f}M, "
            f"matched_ratio={preload['matched_ratio']:.2f}%)"
        )
    print_network_structure(model, cfg)
    print_model_summary(model, cfg, device)
    data = build_datamodule(cfg)
    train_loader = data.train_loader()
    val_loader = data.val_loader()
    test_loader = data.test_loader() if hasattr(data, "test_loader") else None
    print_dataset_summary(cfg, train_loader, val_loader, test_loader)
    class_names = data.classes() if hasattr(data, "classes") else None
    trainer = Trainer(
        model=model,
        device=device,
        cfg=cfg,
        output_dir=Path(exp_dir),
        class_names=class_names,
    )
    if finetune_path is not None:
        if not finetune_path.exists():
            raise FileNotFoundError(f"Fine-tune checkpoint not found: {finetune_path}")
        trainer.load_weights_only(finetune_path)
        print(f"Fine-tune weights loaded from {finetune_path} (optimizer/epoch reset)")
    elif resume_path is not None and resume_path.exists():
        resumed_epoch = trainer.load_checkpoint(resume_path)
        print(f"Resumed from {resume_path} (start_epoch={resumed_epoch})")
    metrics = trainer.fit(train_loader, val_loader)
    print(metrics)


if __name__ == "__main__":
    main()
