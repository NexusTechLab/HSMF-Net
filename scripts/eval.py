from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilenet_enterprise.data import build_datamodule
from mobilenet_enterprise.engine import Evaluator
from mobilenet_enterprise.registry import build_model
from mobilenet_enterprise.utils import ensure_dirs, load_config, resolve_device
from mobilenet_enterprise.utils.checkpoint import (
    clean_state_dict,
    load_state_dict_with_report,
    resolve_checkpoint_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MobileNet checkpoint.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory to save evaluation figures."
    )
    parser.add_argument(
        "--tsne-max-samples",
        type=int,
        default=2000,
        help="Maximum number of samples used to draw t-SNE feature space.",
    )
    parser.add_argument(
        "--min-match-ratio",
        type=float,
        default=50.0,
        help="Minimum checkpoint-model parameter match ratio (percent).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Evaluation split. Default: test (hold-out). Use val only for debugging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "auto"))
    model = build_model(cfg).to(device)
    ckpt_path = resolve_checkpoint_path(cfg, args.ckpt)
    print(f"Using checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    report = load_state_dict_with_report(model, clean_state_dict(state["model"]), strict=False)
    print(
        "Checkpoint load summary: "
        f"matched_keys={report['matched_keys']}, "
        f"matched_params={report['matched_params']:,}, "
        f"matched_ratio={report['matched_ratio']:.2f}%"
    )
    if report["matched_ratio"] < float(args.min_match_ratio):
        raise RuntimeError(
            "Checkpoint/model mismatch too high: "
            f"matched_ratio={report['matched_ratio']:.2f}% < min_match_ratio={args.min_match_ratio:.2f}%. "
            "This usually means model architecture or variant changed after checkpoint was created."
        )
    data = build_datamodule(cfg)
    if args.split == "test":
        if not hasattr(data, "test_loader"):
            raise RuntimeError("Datamodule has no test_loader(); cannot evaluate on test split.")
        eval_loader = data.test_loader()
    else:
        eval_loader = data.val_loader()
    print(f"Evaluating on split={args.split}, samples={len(getattr(eval_loader, 'dataset', []))}")
    evaluator = Evaluator(model=model, device=device)
    exp_name = cfg.get("experiment_name", "default_exp")
    if args.output_dir:
        eval_dir = Path(args.output_dir)
    else:
        base = ensure_dirs(cfg.get("output_dir", "artifacts"))["root"]
        eval_dir = Path(base) / exp_name / "figures"
    result = evaluator.evaluate(
        eval_loader,
        class_names=data.classes(),
        output_dir=eval_dir,
        tsne_max_samples=args.tsne_max_samples,
    )
    print(result)


if __name__ == "__main__":
    main()
