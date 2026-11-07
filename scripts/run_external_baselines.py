"""External-model comparison suite. Default checks/prepares; --run trains all frozen configurations."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ.get("HSMF_EXTERNAL_PROJECT", str(CODE_ROOT))).resolve()
sys.path.insert(0, str(PROJECT / ".external_runtime"))
sys.path.insert(0, str(CODE_ROOT / "src"))
sys.path.insert(0, str(CODE_ROOT / "scripts"))

import numpy as np
import torch
import yaml

from mobilenet_enterprise.engine import Trainer
from mobilenet_enterprise.models.external import METHODS, build_external_model
from mobilenet_enterprise.utils import resolve_device, set_seed
from control_support import CLASSES, ManifestFusionDataModule, digest, file_hash, validate_bands, write_json

DEFAULT_REFERENCE = PROJECT / "artifacts/spectral_controls24/configs/refitted_spa24_seed42.yaml"
DEFAULT_OUTPUT = PROJECT / "artifacts/external_spa24"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-config", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42], help="Training seeds (default: only seed42)")
    p.add_argument("--run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true", help="One train batch and validation batch per model; never test")
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--wait-for-prior", action="store_true", help="Wait for pre-existing experiment coordinators")
    p.add_argument("--latency-iters", type=int, default=200)
    return p


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve_file(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT / path


def model_sources():
    paths = [p for p in (PROJECT / "src").rglob("*.py") if "upstream" not in p.parts]
    paths += [PROJECT / "scripts" / p for p in ("run_external_baselines.py", "control_support.py")]
    return {p.relative_to(PROJECT).as_posix(): file_hash(p) for p in sorted(paths)}


def prepare(args):
    output = args.output.resolve()
    if output == PROJECT / "artifacts" or "reviewer6" in output.name:
        raise ValueError("Use a separate Reviewer 4 output directory")
    reference = yaml.safe_load(args.reference_config.read_text("utf-8"))
    d = reference["data"]
    manifest_path = resolve_file(d["pair_manifest"])
    manifest = json.loads(manifest_path.read_text("utf-8"))
    records = manifest["records"]
    if manifest["classes"] != CLASSES or [sum(r["split"] == s for r in records) for s in ("train", "val", "test")] != [2660, 570, 570]:
        raise RuntimeError("Unexpected class mapping or split sizes")
    if len({r["sample_id"] for r in records}) != 3800:
        raise RuntimeError("Duplicate sample identity")
    for r in records:
        for item in r["file_signatures"]:
            st = Path(item["path"]).stat()
            if (st.st_size, st.st_mtime_ns) != (item["size"], item["mtime_ns"]):
                raise RuntimeError("Reference input changed: " + item["path"])
    band_path = resolve_file(d["band_selection"]["cache_path"])
    bands = validate_bands(band_path)
    stats_path = resolve_file(d["normalization"]["stats_cache_path"])
    stats = np.load(stats_path)
    if stats["mean"].shape != (24,) or stats["std"].shape != (24,) or not np.isfinite(stats["mean"]).all() or not (stats["std"] > 0).all():
        raise RuntimeError("Invalid frozen normalization statistics")
    if (d["image_size"], d["hsi_image_size"], d["hsi_bands"], d["split_seed"]) != (224, 24, 24, 42):
        raise RuntimeError("Expected the previous RGB224 / HSI24x24x24 / split42 protocol")
    sources = model_sources()
    catalog = {}
    for method, metadata in METHODS.items():
        entry = copy.deepcopy(metadata)
        if entry["source"]:
            path = PROJECT / "src/mobilenet_enterprise/models/external/upstream" / entry["source"] / "provenance.json"
            entry["upstream"] = json.loads(path.read_text("utf-8"))
            for item in entry["upstream"]["files"]:
                if file_hash(path.parent / item["path"]) != item["sha256"]:
                    raise RuntimeError("Author archive changed: " + str(path))
        catalog[method] = entry
    protocol = {"schema": 1, "reference_config": str(args.reference_config.resolve()),
        "reference_sha256": file_hash(args.reference_config), "manifest_sha256": file_hash(manifest_path),
        "ordered_test_identity_sha256": digest([(r["sample_id"], r["label"]) for r in records if r["split"] == "test"]),
        "spa24_sha256": file_hash(band_path), "spa_indices": bands.tolist(),
        "normalization_sha256": file_hash(stats_path), "source_hashes": sources,
        "models": catalog, "seeds": args.seeds, "methods": args.methods,
        "training": "Reference training/optimizer/scheduler/augmentation settings unchanged; all models start from scratch including matched HSMF comparator.",
        "selection": "First maximum validation accuracy; recorded EMA/main stream; test only after training completes.",
        "split": "Frozen paired 2660/570/570; split seed42 is independent of training seeds.",
        "preprocessing": "Byte-identical prior SPA24 indices and normalization statistics, identical loader implementation; suite-owned cache copy.",
        "rgb_policy": manifest["rgb_policy"], "pairing_limitation": manifest["pairing"],
        "latency_iters": args.latency_iters,
        "excluded_candidates": {"pyformer": "Official multi-axis attention over WS*WS*k positions and full-width 3D kernels is costly at matched 24x24x24; replaced with author PyTorch 3D-ConvSST (2024).", "ssft": "Official author implementation not verified; not represented by a custom architecture."}}
    stamp = output / "suite_protocol.json"
    if stamp.exists() and json.loads(stamp.read_text("utf-8")) != protocol:
        raise RuntimeError("Frozen suite differs; use a NEW --output directory. No old run overwritten.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(stamp, protocol)
    shared = output / "shared"
    shared.mkdir(exist_ok=True)
    for src, dest in [(manifest_path, output / "pair_manifest.json"), (band_path, shared / "spa24_indices.npy"), (stats_path, shared / "train_band_stats.npz")]:
        if dest.exists() and file_hash(dest) != file_hash(src):
            raise RuntimeError("Frozen suite input changed: " + str(dest))
        if not dest.exists():
            shutil.copy2(src, dest)
    # Copies are independent files; no new process writes into previous experiment directories.
    cache = resolve_file(d["preprocessing_cache"]["root"])
    cache_dest = shared / "tensor_cache"
    if not (shared / "cache_copy.json").exists():
        cache_dest.mkdir(exist_ok=True)
        copied = 0
        if cache.exists():
            for source in cache.rglob("*.npz"):
                dest = cache_dest / source.relative_to(cache)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copy2(source, dest)
                copied += 1
        write_json(shared / "cache_copy.json", {"source": str(cache), "copied": copied, "mode": "independent copies"})
    snapshot = output / "source_snapshot"
    for relative, expected in sources.items():
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and file_hash(target) != expected:
            raise RuntimeError("Source snapshot changed")
        if not target.exists():
            shutil.copy2(PROJECT / relative, target)
    write_json(output / "model_catalog.json", catalog)
    (output / "configs").mkdir(exist_ok=True)
    configs = []
    for method in args.methods:
        for seed in args.seeds:
            cfg = copy.deepcopy(reference)
            cfg.pop("reviewer6", None)
            cfg.update(seed=seed, output_dir=str(output), experiment_name=f"r4_{method}_seed{seed}")
            cfg["pretrained"] = {"enabled": False, "source": "none (matched scratch comparison)"}
            cfg["data"]["pair_manifest"] = str(output / "pair_manifest.json")
            cfg["data"]["band_selection"]["cache_path"] = str(shared / "spa24_indices.npy")
            cfg["data"]["normalization"]["stats_cache_path"] = str(shared / "train_band_stats.npz")
            cfg["data"]["preprocessing_cache"]["root"] = str(cache_dest)
            cfg["logging"]["show_progress_bar"] = False
            cfg["external"] = {"method": method, "metadata": catalog[method], "protocol_sha256": digest(protocol),
                "reference_sha256": protocol["reference_sha256"], "project": str(PROJECT), "latency_iters": args.latency_iters}
            path = output / "configs" / f"{method}_seed{seed}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
            configs.append(path)
    print(f"Prepared {len(configs)} isolated runs, reference={args.reference_config}, SPA={bands.tolist()}", flush=True)
    write_json(output / "status.json", {"status": "prepared", "runs_planned": len(configs)})
    return configs


def metrics_from_rows(rows):
    cm = np.zeros((4, 4), dtype=np.int64)
    for row in rows:
        cm[int(row["true_label"]), int(row["pred_label"])] += 1
    if len(rows) != 570 or cm.sum(1).tolist() != [135, 150, 150, 135] or len({r["sample_id"] for r in rows}) != 570:
        raise RuntimeError("Unexpected test support/identity")
    tp = cm.diagonal()
    denominator = cm.sum(0) + cm.sum(1)
    return {"n_test": len(rows), "correct": int(tp.sum()), "accuracy_pct": 100 * float(tp.sum()) / len(rows),
            "macro_f1_pct": float(np.divide(200 * tp, denominator, out=np.zeros(4), where=denominator != 0).mean()),
            "confusion_counts": cm.tolist()}


def profile_model(model, cfg, device, latency=True):
    from fvcore.nn import FlopCountAnalysis
    model.eval()
    inputs = (torch.zeros(1, 3, 224, 224, device=device), torch.zeros(1, 1, 24, 24, 24, device=device))
    count = FlopCountAnalysis(model, inputs).unsupported_ops_warnings(False).uncalled_modules_warnings(False).tracer_warnings("none")
    # fvcore counts one multiply-add as one; explicitly count convolution/linear/einsum/matmul MACs only.
    by_operator = count.by_operator()
    major = {k: v for k, v in by_operator.items() if k in {"conv", "linear", "einsum", "matmul", "bmm"}}
    result = {"parameters": sum(p.numel() for p in model.parameters()), "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
        "gflops": 2 * sum(major.values()) / 1e9, "macs_by_operator": major,
        "flops_convention": "2 FLOPs per MAC; convolution, linear, einsum, matmul only; excludes norm, activations, resize and pointwise ops",
        "unsupported_ops": dict(count.unsupported_ops()), "latency_ms_mean": None, "latency_ms_sd": None,
        "latency_ms_p50": None, "latency_ms_p95": None, "latency_scope": "FP32 eval/inference_mode, batch1; includes model input adapters, excludes file I/O and CPU preprocessing/transfers"}
    if latency:
        samples = []
        with torch.inference_mode():
            for _ in range(50):
                model(*inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            for _ in range(cfg["external"]["latency_iters"]):
                if device.type == "cuda":
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    model(*inputs)
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end))
                else:
                    start = time.perf_counter()
                    model(*inputs)
                    samples.append((time.perf_counter() - start) * 1000)
        result.update(latency_ms_mean=statistics.mean(samples), latency_ms_sd=statistics.stdev(samples),
                      latency_ms_p50=float(np.percentile(samples, 50)), latency_ms_p95=float(np.percentile(samples, 95)))
    return result


def worker(path, resume):
    cfg = yaml.safe_load(path.read_text("utf-8"))
    output = Path(cfg["output_dir"])
    protocol = json.loads((output / "suite_protocol.json").read_text("utf-8"))
    if digest(protocol) != cfg["external"]["protocol_sha256"]:
        raise RuntimeError("Protocol hash differs from frozen configuration")
    for relative, expected in protocol["source_hashes"].items():
        if file_hash(CODE_ROOT / relative) != expected:
            raise RuntimeError("Frozen source changed: " + relative)
    for key, source in (("manifest_sha256", cfg["data"]["pair_manifest"]),
                        ("spa24_sha256", cfg["data"]["band_selection"]["cache_path"]),
                        ("normalization_sha256", cfg["data"]["normalization"]["stats_cache_path"])):
        if file_hash(source) != protocol[key]:
            raise RuntimeError("Frozen input changed: " + str(source))
    out = output / cfg["experiment_name"]
    out.mkdir(exist_ok=True)
    signature = digest(cfg)
    stamp = out / "run_signature.json"
    if stamp.exists() and json.loads(stamp.read_text("utf-8"))["sha256"] != signature:
        raise RuntimeError("Run configuration changed")
    write_json(stamp, {"sha256": signature})
    if (out / "test_metrics.json").exists():
        return
    last = out / "checkpoints/last.pt"
    if (out / "checkpoints/best.pt").exists() and not resume:
        raise RuntimeError("Partial run exists; add --resume")
    torch.set_num_threads(4)
    set_seed(cfg["seed"], cfg["train"].get("deterministic", False))
    device = resolve_device(cfg["device"])
    model = build_external_model(cfg["external"]["method"], cfg).to(device)
    write_json(out / "environment.json", {"python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "threads": torch.get_num_threads(), "pretrained": False,
        "resume_limit": "Legacy Trainer restores model/optimizer/scheduler/EMA/epoch but not RNG/worker states; resumed trajectories are not bitwise-equivalent."})
    (out / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    dm = ManifestFusionDataModule(cfg)
    train, val = dm.train_loader(), dm.val_loader()
    set_seed(cfg["seed"], cfg["train"].get("deterministic", False))
    trainer = Trainer(model=model, device=device, cfg=cfg, output_dir=out, class_names=CLASSES)
    if resume and last.exists():
        trainer.load_checkpoint(last)
    if trainer.start_epoch <= cfg["train"]["epochs"]:
        trainer.fit(train, val)
    if trainer.writer is not None:
        trainer.writer.close()
    best = out / "checkpoints/best.pt"
    state = torch.load(best, map_location=device, weights_only=False)
    stream = "ema_model" if state.get("ema_eval_active", False) else "model"
    model.load_state_dict(state[stream], strict=True)
    model.eval()
    identities = [r for r in json.loads(Path(cfg["data"]["pair_manifest"]).read_text("utf-8"))["records"] if r["split"] == "test"]
    rows = []
    with torch.inference_mode():
        for (rgb, hsi), targets in dm.test_loader():
            logits = model(rgb.to(device), hsi.to(device))
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite test logits")
            probabilities = logits.softmax(1).cpu().tolist()
            for target, pred, probability in zip(targets.tolist(), logits.argmax(1).cpu().tolist(), probabilities):
                identity = identities[len(rows)]
                if identity["label"] != target:
                    raise RuntimeError("Test ordering mismatch")
                rows.append({"sample_id": identity["sample_id"], "true_label": target, "pred_label": pred,
                             **{f"prob_{i}": v for i, v in enumerate(probability)}})
    csv_write(out / "test_predictions.csv", rows)
    complexity = profile_model(model, cfg, device)
    write_json(out / "efficiency.json", complexity)
    metrics = metrics_from_rows(rows)
    metrics.update(method=cfg["external"]["method"], seed=cfg["seed"], protocol_sha256=cfg["external"]["protocol_sha256"],
        config_sha256=signature, predictions_sha256=file_hash(out / "test_predictions.csv"),
        checkpoint_sha256=file_hash(best), checkpoint_epoch=state["epoch"], checkpoint_stream=stream,
        parameters=complexity["parameters"], params_m=complexity["params_m"], gflops=complexity["gflops"],
        latency_ms_mean=complexity["latency_ms_mean"], latency_ms_sd=complexity["latency_ms_sd"])
    write_json(out / "test_metrics.json", metrics)
    print(json.dumps(metrics), flush=True)


def smoke(configs, output):
    # No validation metric selection and no test access in implementation checks.
    torch.set_num_threads(2)
    cfg = yaml.safe_load(configs[0].read_text("utf-8"))
    cfg["data"]["num_workers"] = 0
    dm = ManifestFusionDataModule(cfg)
    train_inputs, targets = next(iter(dm.train_loader()))
    val_inputs, val_targets = next(iter(dm.val_loader()))
    rows = []
    for path in configs:
        c = yaml.safe_load(path.read_text("utf-8"))
        if c["seed"] != cfg["seed"]:
            continue
        set_seed(c["seed"])
        model = build_external_model(c["external"]["method"], c)
        model.train()
        logits = model(*(x[:2] for x in train_inputs))
        loss = torch.nn.functional.cross_entropy(logits, targets[:2])
        loss.backward()
        if logits.shape != (2, 4) or not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Invalid model gradient: " + str(path))
        optimizer = torch.optim.AdamW(model.parameters(), lr=c["optimizer"]["lr"])
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            prediction = model(*(x[:2] for x in val_inputs))
        if prediction.shape != (2, 4) or not torch.isfinite(prediction).all():
            raise RuntimeError("Invalid validation output")
        row = {"method": c["external"]["method"], "train_backward_and_step": True, "validation_forward": True,
               "complexity": profile_model(model, c, torch.device("cpu"), latency=False)}
        rows.append(row)
        print("Smoke passed:", row["method"], row["complexity"]["params_m"], row["complexity"]["gflops"], flush=True)
    write_json(output / "smoke_checks.json", {"status": "passed", "test_used": False,
        "train_shapes": [list(x.shape) for x in train_inputs], "rows": rows})


def summarize(output):
    rows, groups = [], {}
    protocol = json.loads((output / "suite_protocol.json").read_text("utf-8"))
    for path in sorted(output.glob("r4_*/test_metrics.json")):
        result = json.loads(path.read_text("utf-8"))
        predictions = path.parent / "test_predictions.csv"
        if result["protocol_sha256"] != digest(protocol) or file_hash(predictions) != result["predictions_sha256"]:
            raise RuntimeError("Result provenance mismatch")
        with predictions.open(encoding="utf-8-sig") as handle:
            predictions_rows = list(csv.DictReader(handle))
        if digest([(r["sample_id"], int(r["true_label"])) for r in predictions_rows]) != protocol["ordered_test_identity_sha256"]:
            raise RuntimeError("Test identity differs")
        recalculated = metrics_from_rows(predictions_rows)
        if any(recalculated[k] != result[k] for k in recalculated):
            raise RuntimeError("Metrics differ from predictions")
        rows.append({k: v for k, v in result.items() if k != "confusion_counts"})
        groups.setdefault(result["method"], []).append(result)
    csv_write(output / "per_seed_metrics.csv", rows)
    summaries = []
    for method in protocol["methods"]:
        runs = groups.get(method, [])
        row = {"method": method, "label": METHODS[method]["label"], "year": METHODS[method]["year"],
            "modalities": METHODS[method]["modalities"], "pretrained": False,
            "n_runs": len(runs), "expected_runs": len(protocol["seeds"]),
            "status": "complete" if sorted(r["seed"] for r in runs) == sorted(protocol["seeds"]) else "pending"}
        for metric in ("accuracy_pct", "macro_f1_pct", "params_m", "gflops", "latency_ms_mean"):
            values = [r[metric] for r in runs]
            row[metric + "_mean"] = statistics.mean(values) if values else None
            row[metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
        summaries.append(row)
    csv_write(output / "model_summary.csv", summaries)
    lines = ["# Reviewer 4 external comparisons", "", "Status: " + ("complete" if all(r["status"] == "complete" for r in summaries) else "incomplete; pending values are not results"), "",
        "All models use the frozen prior SPA24 preprocessing and same sample manifest. All start from scratch, including HSMF-Net. Historical pretrained/internal ablations are separate.", "",
        "GFLOPs: convolution/linear/einsum/matmul only, 2 FLOPs per MAC. Latency: FP32 batch1 model forward including input adapters, excluding CPU preprocessing and transfers.", "",
        "| Method | Input | Runs | Accuracy (%) | Macro-F1 (%) | Params (M) | GFLOPs | Latency (ms) |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        def fmt(key):
            value, sd = r[key + "_mean"], r[key + "_sd"]
            return "pending" if value is None else f"{value:.4f}" + (f" ± {sd:.4f}" if sd is not None else "")
        lines.append(f"| {r['label']} | {r['modalities']} | {r['n_runs']}/{r['expected_runs']} | {fmt('accuracy_pct')} | {fmt('macro_f1_pct')} | {fmt('params_m')} | {fmt('gflops')} | {fmt('latency_ms_mean')} |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(rows)


def prior_experiments():
    if os.name != "nt":
        return []
    command = "Get-CimInstance Win32_Process -Filter \"name = 'python.exe'\" | Where-Object {$_.CommandLine -match 'run_reviewer6_|run_sci_|run_hsi_architecture_ablation.py|run_hbfm_internal'} | Select-Object -ExpandProperty ProcessId | ConvertTo-Json -Compress"
    result = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, check=True)
    value = json.loads(result.stdout) if result.stdout.strip() else []
    return value if isinstance(value, list) else [value]


def run_with_live_log(command, log, environment):
    """Stream each worker line to the project terminal and its persistent UTF-8 log."""
    environment = dict(environment, PYTHONIOENCODING="utf-8")
    with log.open("a", encoding="utf-8", buffering=1) as handle:
        heading = f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} | {' '.join(command)} ===\n"
        handle.write(heading)
        print(f"Training log: {log}", flush=True)
        process = subprocess.Popen(command, cwd=PROJECT, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", bufsize=1)
        try:
            for line in process.stdout:
                handle.write(line)
                print(line, end="", flush=True)
            returncode = process.wait()
            handle.write(f"=== Worker exit code: {returncode} ===\n")
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
        except KeyboardInterrupt:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            handle.write("=== Interrupted; resume from last saved checkpoint ===\n")
            raise
        finally:
            process.stdout.close()


def main():
    args = parser().parse_args()
    os.chdir(PROJECT)
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods) or args.latency_iters < 2:
        raise ValueError("Duplicate methods/seeds or invalid latency iterations")
    if args.worker:
        worker(args.worker, args.resume)
        return
    if args.summarize_only:
        summarize(args.output)
        return
    if args.run and (args.output / "runner.lock").exists():
        raise RuntimeError("This suite already has a coordinator (runner.lock); inspect status.json instead of launching twice")
    configs = prepare(args)
    if args.smoke:
        smoke(configs, args.output)
    summarize(args.output)
    if not args.run:
        return
    snapshot = args.output.resolve() / "source_snapshot"
    environment = dict(os.environ, HSMF_EXTERNAL_PROJECT=str(PROJECT), PYTHONUNBUFFERED="1")
    status_path = args.output / "status.json"
    lock = args.output / "runner.lock"
    with lock.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "python": sys.executable}, handle)
    try:
        if args.wait_for_prior:
            while pids := prior_experiments():
                write_json(status_path, {"status": "waiting_for_prior_experiments", "pids": pids,
                    "runs_planned": len(configs), "checked_at": time.strftime("%Y-%m-%d %H:%M:%S")})
                print("Waiting for existing experiments:", pids, flush=True)
                time.sleep(30)
        for config in configs:
            write_json(status_path, {"status": "training", "current": config.stem, "runs_completed": summarize(args.output), "runs_planned": len(configs)})
            command = [sys.executable, "-u", str(snapshot / "scripts/run_external_baselines.py"), "--worker", str(config)]
            if args.resume:
                command.append("--resume")
            log = args.output / (config.stem + ".log")
            print("Starting", config.stem, flush=True)
            run_with_live_log(command, log, environment)
            summarize(args.output)
    except KeyboardInterrupt:
        write_json(status_path, {"status": "interrupted_ready_to_resume",
            "current": config.stem if "config" in locals() else None,
            "runs_completed": summarize(args.output), "runs_planned": len(configs),
            "note": "Interrupted in terminal; resume from last saved checkpoint."})
        print("\nInterrupted. Training records retained; add --resume to continue.", flush=True)
        return
    except Exception as exc:
        write_json(status_path, {"status": "failed", "current": config.stem if "config" in locals() else None,
                                "error": str(exc), "log": str(log) if "log" in locals() else None})
        raise
    finally:
        lock.unlink(missing_ok=True)
    write_json(status_path, {"status": "completed", "runs_completed": summarize(args.output), "runs_planned": len(configs)})


if __name__ == "__main__":
    main()
