"""Matched SPA-24 controls; legacy HSI-directory images, fixed manifest, validation-only selection.

Default: prepare/check only. Training starts only with --run.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

import numpy as np
import torch
import yaml

from control_support import (
    ALL_VARIANTS, CLASSES, VARIANTS, ManifestFusionDataModule, collect_manifest,
    digest, file_hash, make_model, validate_bands, variant_config, write_json,
)
from mobilenet_enterprise.engine import Trainer
from mobilenet_enterprise.utils import load_imagenet_pretrained_weights, resolve_device, set_seed


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=PROJECT / "data/wheat-seed")
    p.add_argument("--rgb-root", type=Path, help="Root containing <cultivar>/hsi/ image exports")
    p.add_argument("--hsi-root", type=Path, help="Root containing <cultivar>/hsi/")
    p.add_argument("--base-config", type=Path, default=PROJECT / "configs/control_base.yaml")
    p.add_argument("--spa-indices", type=Path, default=PROJECT / "spectral_selection/spa/archived/selected_bands_spa_24_e10_t15_smooth.npy")
    p.add_argument("--output", type=Path, default=PROJECT / "artifacts/fusion_controls24")
    p.add_argument("--variants", nargs="+", choices=ALL_VARIANTS, default=VARIANTS)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 47)))
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--run", action="store_true", help="Prepare, then train and evaluate selected runs")
    p.add_argument("--resume", action="store_true", help="Resume unfinished runs from last.pt")
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    return p


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def source_hashes():
    paths = list((PROJECT / "src").rglob("*.py"))
    paths += [PROJECT / "scripts" / n for n in ["control_support.py", "run_fusion_controls.py"]]
    return {p.relative_to(PROJECT).as_posix(): file_hash(p) for p in sorted(paths)}


def prepare(args):
    output = args.output.resolve()
    manifest = collect_manifest(args.rgb_root or args.data_root, args.hsi_root or args.data_root)
    manifest_hash = digest(manifest)
    bands = validate_bands(args.spa_indices)
    base = yaml.safe_load(args.base_config.read_text("utf-8"))
    base["device"] = args.device
    base["data"]["num_workers"] = args.num_workers
    base["data"].update(rgb_root=str((args.rgb_root or args.data_root).resolve()),
                         hsi_root=str((args.hsi_root or args.data_root).resolve()))
    common_cache = output / "shared" / digest({"manifest": manifest_hash, "bands": bands.tolist(),
                                               "roi": base["data"].get("roi_crop_enabled", False)})[:20]
    plan = {"manifest_sha256": manifest_hash, "base_config_sha256": file_hash(args.base_config),
            "spa24_sha256": file_hash(args.spa_indices), "spa24_indices": bands.tolist(),
            "source_hashes": source_hashes(), "device": args.device, "num_workers": args.num_workers,
            "rgb_policy": manifest["rgb_policy"],
            "normalization": "recomputed from the 2660 training HDR files in a fresh suite-specific cache",
            "band_selection": "frozen archived SPA-24 indices; historical selection fit membership remains unverified",
            "comparison": "prospective internally matched suite; do not combine with historical S8 as if unchanged",
            "checkpoint_rule": "best validation accuracy, first maximum; EMA/main stream stored in checkpoint",
            "augmentation": "historical S8 augmentation retained; shared flips plus RGB RandAugment and HSI roll/noise",
            "split_identity": "fixed current filename-paired manifest; identity with historical prediction rows unverified"}
    plan_path = output / "suite_protocol.json"
    if plan_path.exists() and json.loads(plan_path.read_text("utf-8")) != plan:
        raise RuntimeError("Suite inputs/code/settings changed. Use a new --output directory; existing runs are preserved.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(plan_path, plan)
    manifest_path, band_path = output / "pair_manifest.json", common_cache / "spa24_indices.npy"
    write_json(manifest_path, manifest)
    common_cache.mkdir(parents=True, exist_ok=True)
    if band_path.exists() and file_hash(band_path) != plan["spa24_sha256"]:
        raise RuntimeError("Frozen SPA cache changed")
    # Byte-for-byte copy preserves the archived input hash.
    band_path.write_bytes(args.spa_indices.read_bytes())
    configs = []
    config_dir = output / "configs"
    config_dir.mkdir(exist_ok=True)
    for variant, seed in itertools.product(args.variants, args.seeds):
        cfg = variant_config(base, variant, seed, output, manifest_path, common_cache, band_path)
        cfg["reviewer6"].update(protocol_sha256=digest(plan), manifest_sha256=manifest_hash,
                               source_sha256=digest(plan["source_hashes"]))
        path = config_dir / f"{variant}_seed{seed}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        configs.append(path)
    print(f"Checked 3800 pairs: train=2660, val=570, test=570; {len(configs)} runs prepared.")
    print(f"RGB branch input: {manifest['rgb_policy']}")
    print(f"Manifest and configs: {output}")
    return configs


def checkpoint_stream(state):
    stream = "ema_model" if state.get("ema_eval_active", False) else "model"
    if stream not in state:
        raise RuntimeError(f"Selected checkpoint stream {stream} is absent")
    return stream


def metrics_from_rows(rows):
    cm = np.zeros((4, 4), dtype=np.int64)
    for row in rows:
        cm[int(row["true_label"]), int(row["pred_label"])] += 1
    supports = cm.sum(1).tolist()
    if supports != [135, 150, 150, 135] or len({r["sample_id"] for r in rows}) != 570:
        raise RuntimeError(f"Unexpected test support or duplicate identity: {supports}")
    tp = cm.diagonal()
    den = cm.sum(0) + cm.sum(1)
    return {"n_test": len(rows), "correct": int(tp.sum()),
            "accuracy_pct": 100 * float(tp.sum()) / len(rows),
            "macro_f1_pct": float(np.mean(np.divide(200 * tp, den, out=np.zeros(4), where=den != 0))),
            "confusion_counts": cm.tolist()}


def worker(config_path, resume=False):
    cfg = yaml.safe_load(config_path.read_text("utf-8"))
    out = Path(cfg["output_dir"]) / cfg["experiment_name"]
    out.mkdir(parents=True, exist_ok=True)
    signature = digest(cfg)
    stamp = out / "run_signature.json"
    if stamp.exists() and json.loads(stamp.read_text("utf-8"))["sha256"] != signature:
        raise RuntimeError("Run configuration changed; choose a new output directory")
    write_json(stamp, {"sha256": signature})
    if (out / "test_metrics.json").exists():
        print(f"Completed run retained: {out.name}")
        return
    existing_best, last = out / "checkpoints/best.pt", out / "checkpoints/last.pt"
    if existing_best.exists() and not resume:
        raise RuntimeError("Partial run exists; use --resume or a new --output directory")
    set_seed(cfg["seed"], cfg["train"].get("deterministic", False))
    device = resolve_device(cfg["device"])
    model = make_model(cfg).to(device)
    preload = None
    if not (resume and last.exists()):
        preload = load_imagenet_pretrained_weights(model, cfg)
        if cfg["pretrained"].get("enabled") and not preload["loaded"]:
            raise RuntimeError("Requested pretraining did not load. Check torchvision weight cache/network, then rerun.")
    environment = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                   "numpy": np.__version__, "cuda": torch.version.cuda,
                   "device": str(device), "torch_threads": torch.get_num_threads(),
                   "cudnn": torch.backends.cudnn.version(), "pretraining": preload,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "resume_from": str(last) if resume and last.exists() else None,
                   "resume_note": "optimizer/epoch/EMA restored; legacy checkpoints do not save RNG states"}
    if not (out / "environment.json").exists():
        write_json(out / "environment.json", environment)
    elif resume:
        write_json(out / "resume_environment.json", environment)
    dm = ManifestFusionDataModule(cfg)
    train, val = dm.train_loader(), dm.val_loader()
    helper = dm._hsi_helper()
    stats_path = Path(cfg["data"]["normalization"]["stats_cache_path"])
    stats_provenance = {"manifest_sha256": cfg["reviewer6"]["manifest_sha256"],
                        "fit_split": "train", "fit_n": 2660,
                        "spa_indices": helper._cached_selected_bands.tolist(),
                        "stats_sha256": file_hash(stats_path)}
    stats_stamp = stats_path.with_suffix(".provenance.json")
    if stats_stamp.exists() and json.loads(stats_stamp.read_text("utf-8")) != stats_provenance:
        raise RuntimeError("Normalization cache provenance mismatch")
    write_json(stats_stamp, stats_provenance)
    (out / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    # Reset run RNG after model/data preparation, before training.
    set_seed(cfg["seed"], cfg["train"].get("deterministic", False))
    trainer = Trainer(model=model, device=device, cfg=cfg, output_dir=out, class_names=CLASSES)
    if resume and last.exists():
        trainer.load_checkpoint(last)
    if trainer.start_epoch <= cfg["train"]["epochs"]:
        trainer.fit(train, val)
    if trainer.writer is not None:
        trainer.writer.close()
    # Test only the validation-selected best epoch and its recorded weight stream.
    state = torch.load(existing_best, map_location=device, weights_only=False)
    stream = checkpoint_stream(state)
    model.load_state_dict(state[stream], strict=True)
    model.eval()
    rows = []
    test = dm.test_loader()
    manifest = json.loads(Path(cfg["data"]["pair_manifest"]).read_text("utf-8"))
    identities = [r for r in manifest["records"] if r["split"] == "test"]
    with torch.inference_mode():
        for (rgb, hsi), targets in test:
            logits = model(rgb.to(device), hsi.to(device))
            probs = logits.softmax(1).cpu().numpy()
            predictions = logits.argmax(1).cpu().tolist()
            for target, pred, probability in zip(targets.tolist(), predictions, probs):
                identity = identities[len(rows)]
                if target != identity["label"]:
                    raise RuntimeError("Test sample order differs from frozen manifest")
                row = {"sample_id": identity["sample_id"], "true_label": target, "pred_label": pred,
                       "rgb_path": identity["rgb_path"], "hdr_path": identity["hdr_path"]}
                row.update({f"prob_{i}": float(p) for i, p in enumerate(probability)})
                rows.append(row)
    csv_write(out / "test_predictions.csv", rows)
    result = metrics_from_rows(rows)
    result.update(variant=cfg["reviewer6"]["variant"], seed=cfg["seed"],
                  checkpoint_epoch=state["epoch"], checkpoint_stream=stream,
                  checkpoint_sha256=file_hash(existing_best), config_sha256=signature,
                  protocol_sha256=cfg["reviewer6"]["protocol_sha256"],
                  manifest_sha256=cfg["reviewer6"]["manifest_sha256"],
                  predictions_sha256=file_hash(out / "test_predictions.csv"),
                  parameters=environment["parameters"])
    write_json(out / "test_metrics.json", result)
    print(json.dumps(result, ensure_ascii=False))


def mean_sd_ci(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    half = 2.7764451051977987 * sd / math.sqrt(5) if len(values) == 5 else None
    return {"mean": mean, "sd": sd, "ci95_lower": mean - half if half is not None else None,
            "ci95_upper": mean + half if half is not None else None}


def signed_rank(diffs):
    diffs = [d for d in diffs if d != 0]
    ranks = [1 + sum(abs(y) < abs(x) for y in diffs)
             + (sum(abs(y) == abs(x) for y in diffs) - 1) / 2 for x in diffs]
    observed = sum(r for r, d in zip(ranks, diffs) if d > 0)
    null = [sum(r for r, s in zip(ranks, signs) if s)
            for signs in itertools.product([0, 1], repeat=len(diffs))]
    return sum(v >= observed - 1e-12 for v in null) / len(null)


def summarize(output):
    output = Path(output)
    run_rows, groups = [], {}
    for path in sorted(output.glob("control_*/test_metrics.json")):
        r = json.loads(path.read_text("utf-8"))
        predictions = path.parent / "test_predictions.csv"
        if file_hash(predictions) != r["predictions_sha256"]:
            raise RuntimeError(f"Prediction file changed: {predictions}")
        with predictions.open(encoding="utf-8") as f:
            prediction_rows = list(csv.DictReader(f))
        recalculated = metrics_from_rows(prediction_rows)
        if any(recalculated[k] != r[k] for k in recalculated):
            raise RuntimeError(f"Stored metrics differ from predictions: {path}")
        r["test_identity_sha256"] = digest([(x["sample_id"], x["true_label"]) for x in prediction_rows])
        groups.setdefault(r["variant"], {})[r["seed"]] = r
        run_rows.append({k: v for k, v in r.items() if k != "confusion_counts"})
    if not run_rows:
        print("No completed test results yet.")
        return
    if len({(r["protocol_sha256"], r["test_identity_sha256"]) for r in run_rows}) != 1:
        raise RuntimeError("Runs do not share one protocol and ordered test set")
    csv_write(output / "per_seed_metrics.csv", run_rows)
    summaries, contrasts = [], []
    for variant, runs in groups.items():
        row = {"variant": variant, "n_runs": len(runs), "complete_five_seeds": sorted(runs) == list(range(42, 47))}
        for metric in ["accuracy_pct", "macro_f1_pct"]:
            row.update({metric + "_" + k: v for k, v in mean_sd_ci([r[metric] for r in runs.values()]).items()})
        summaries.append(row)
        if variant == "full24" or sorted(runs) != list(range(42, 47)):
            continue
        full = groups.get("full24", {})
        if sorted(full) != list(range(42, 47)):
            continue
        # Exact integer count differences avoid floating-point tie artifacts.
        counts = [full[s]["correct"] - runs[s]["correct"] for s in range(42, 47)]
        differences = [100 * d / 570 for d in counts]
        contrasts.append({"contrast": f"full24 vs {variant}", "n_pairs": 5,
                          **mean_sd_ci(differences), "one_sided_signed_rank_p": signed_rank(counts)})
    family_complete = all(sorted(groups.get(v, {})) == list(range(42, 47)) for v in VARIANTS)
    # Optional historical joint-attention control joins the family only once complete.
    optional_partial = "joint_attention24" in groups and sorted(groups["joint_attention24"]) != list(range(42, 47))
    previous = 0.0
    for i, r in enumerate(sorted(contrasts, key=lambda x: x["one_sided_signed_rank_p"])):
        previous = max(previous, min(1.0, (len(contrasts) - i) * r["one_sided_signed_rank_p"]))
        r["holm_adjusted_p"] = previous if family_complete and not optional_partial else None
        r["family_complete"] = family_complete and not optional_partial
    csv_write(output / "model_summary.csv", summaries)
    csv_write(output / "paired_full24_contrasts.csv", contrasts)
    print(f"Summarized {len(run_rows)} completed runs. Full default family complete: {family_complete}")


def main():
    args = parser().parse_args()
    os.chdir(PROJECT)
    if len(args.seeds) != len(set(args.seeds)) or len(args.variants) != len(set(args.variants)):
        raise ValueError("Duplicate seeds or variants are not allowed")
    if args.worker:
        worker(args.worker, args.resume)
        return
    if args.summarize_only:
        summarize(args.output)
        return
    configs = prepare(args)
    if not args.run:
        print("Check only; add --run to start training. No training or test inference performed.")
        return
    for config in configs:
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(config)]
        if args.resume:
            command.append("--resume")
        print(f"Starting {config.stem}", flush=True)
        subprocess.run(command, check=True, cwd=PROJECT)
        summarize(args.output)


if __name__ == "__main__":
    main()
