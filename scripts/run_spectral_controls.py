"""Isolated two-arm SPA-refit/Uniform-24 study; existing controls stay unchanged.

Default prepares ten configurations and checks the protocol. --run trains them
sequentially; --resume retains completed runs and resumes legacy last.pt files.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(PROJECT / 'scripts'))
import numpy as np
import yaml
from control_support import digest, file_hash, variant_config, validate_bands, write_json
from run_fusion_controls import source_hashes, mean_sd_ci, metrics_from_rows

ROOT = PROJECT / 'artifacts/spectral_controls24'
ARMS = ['refitted_spa24', 'uniform24']
SEEDS = list(range(42, 47))


def csv_write(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize_training_config(cfg):
    """Remove only provenance/output/cache paths to compare material settings."""
    cfg = copy.deepcopy(cfg)
    cfg.pop('experiment_name', None)
    cfg.pop('output_dir', None)
    cfg.pop('reviewer6', None)
    cfg['data']['band_selection'].pop('cache_path', None)
    cfg['data']['normalization'].pop('stats_cache_path', None)
    cfg['data']['preprocessing_cache'].pop('root', None)
    return cfg


def verify_metadata(manifest):
    for row in manifest['records']:
        for item in row['file_signatures']:
            st = Path(item['path']).stat()
            if (st.st_size, st.st_mtime_ns) != (item['size'], item['mtime_ns']):
                raise RuntimeError('Frozen input changed: ' + item['path'])


def prepare(output):
    refit = output / 'refit_v2'
    report = json.loads((refit / 'refit_report.json').read_text('utf-8'))
    if report['status'] != 'completed_training_only_refit' or not report['independent_24_matches_50_prefix']:
        raise RuntimeError('Controlled training-only SPA refit is incomplete')
    if report['reuse_decision'] != 'fresh_spa24_and_uniform24_five_seeds_required':
        raise RuntimeError('This ten-run branch requires a documented non-reuse decision')
    original_manifest = PROJECT / 'artifacts/reviewecontrol_matched24_hsi_rgb/pair_manifest.json'
    manifest = json.loads(original_manifest.read_text('utf-8'))
    # The author selected HSI-directory image exports in the current control
    # task. SPA itself uses no RGB images: certify the same TRAIN HSI members
    # and contents before reusing the completed spectral fit.
    ledger = json.loads((refit/'training_content_manifest.json').read_text('utf-8'))
    train = [r for r in manifest['records'] if r['split']=='train']
    if [(r['sample_id'],r['label'],r['hdr_path']) for r in train] != [(r['sample_id'],r['label'],r['hdr']['path']) for r in ledger['records']]:
        raise RuntimeError('Current training HSI membership differs from the refit')
    for r in ledger['records']:
        for field in ['hdr','hsi_raw']:
            if file_hash(r[field]['path']) != r[field]['sha256']:
                raise RuntimeError('Training HSI contents differ from the refit')
    equivalence = {'n_training_hsi':2660,'ordered_ids_labels_hdr_paths_identical':True,
                   'hdr_and_raw_content_hashes_rechecked':True,
                   'refit_source_manifest_sha256':report['input_fingerprint']['manifest_sha256'],
                   'active_image_policy':manifest['rgb_policy'],
                   'reason':'Only image paths changed; SPA fitting used the same authenticated HSI training records and no image branch data.'}
    write_json(output/'selection_training_equivalence.json',equivalence)
    if Counter(r['split'] for r in manifest['records']) != {'train': 2660, 'val': 570, 'test': 570}:
        raise RuntimeError('Unexpected subset sizes')
    verify_metadata(manifest)
    manifest_path = output / 'pair_manifest.json'
    if manifest_path.exists() and manifest_path.read_bytes() != original_manifest.read_bytes():
        raise RuntimeError('Frozen study manifest changed')
    manifest_path.write_bytes(original_manifest.read_bytes())
    bands = {arm: validate_bands(refit / ('spa24_indices.npy' if arm == 'refitted_spa24' else 'uniform24_indices.npy')) for arm in ARMS}
    source = source_hashes()
    old_protocol = json.loads((PROJECT / 'artifacts/reviewecontrol_matched24_hsi_rgb/suite_protocol.json').read_text('utf-8'))
    differences = {k: {'previous': old_protocol['source_hashes'].get(k), 'current': source.get(k)}
                   for k in set(source) | set(old_protocol['source_hashes'])
                   if source.get(k) != old_protocol['source_hashes'].get(k)}
    if any(k.startswith('src/') for k in differences):
        raise RuntimeError('Model/training implementation differs from the existing source snapshot')
    # Existing control entry/support scripts have changed since their original
    # protocol stamp. Do not reuse their checkpoints. Freeze current code for
    # both new arms so edits in another task cannot change a running study.
    snapshot = output / 'source_snapshot'
    for relative, expected_hash in source.items():
        target = snapshot / relative
        if target.exists() and file_hash(target) != expected_hash:
            raise RuntimeError('Frozen study source differs from current preparation sources')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes((PROJECT / relative).read_bytes())
    base_path = PROJECT / 'configs/control_base.yaml'
    base = yaml.safe_load(base_path.read_text('utf-8'))
    base['device'] = 'cuda'
    if base['data'].get('roi_crop_enabled', False):
        raise RuntimeError('Refit assumes the archived no-ROI-crop protocol')
    if base['train']['epochs'] != 100:
        raise RuntimeError('Do not reduce the matched training budget')
    # Keep the same worker count and all substantive training settings.
    if base['data']['num_workers'] != 2:
        raise RuntimeError('Unexpected worker count in base protocol')
    base['data'].update(rgb_root=str(PROJECT / 'data/wheat-seed'), hsi_root=str(PROJECT / 'data/wheat-seed'))
    plan = {'schema': 1, 'arms': ARMS, 'seeds': SEEDS, 'n_runs': 10,
            'manifest_sha256': digest(manifest), 'manifest_file_sha256': file_hash(manifest_path),
            'training_content_manifest_sha256': file_hash(refit / 'training_content_manifest.json'),
            'selection_training_equivalence':equivalence,
            'image_input_policy':manifest['rgb_policy'],
            'refit_report_sha256': file_hash(refit / 'refit_report.json'),
            'base_config_sha256': file_hash(base_path), 'source_hashes': source,
            'previous_control_script_differences': differences,
            'worker_source_policy': 'Run copied source_snapshot/scripts/run_fusion_controls.py; do not modify or reuse old control checkpoints.',
            'runner_sha256': file_hash(Path(__file__)), 'python': sys.version,
            'numpy': np.__version__, 'band_indices': {k: v.tolist() for k,v in bands.items()},
            'normalization': 'Same training-only estimator; independent arm-specific statistics/caches.',
            'primary_comparison': 'refitted_spa24 minus uniform24 accuracy; five seed-paired differences and unadjusted two-sided 95% t interval.',
            'secondary_metric': 'Macro-F1 with the same paired descriptive interval; no multiplicity-adjusted superiority claim.',
            'scope': 'Current frozen filename-paired dataset; historical sample mapping and source independence remain unverified.',
            'checkpoint': 'First maximum validation accuracy; selected main/EMA stream recorded by existing worker.',
            'resume_note': 'Existing trainer checkpoints do not retain RNG state. Interrupted continuation is recorded; not bitwise equivalent to uninterrupted training.'}
    plan_path = output / 'study_protocol.json'
    if plan_path.exists() and json.loads(plan_path.read_text('utf-8')) != plan:
        raise RuntimeError('Study inputs/code/settings changed; do not mix protocols')
    write_json(plan_path, plan)
    configs = []
    canonical = None
    for seed in SEEDS:
        for arm in ARMS:
            # Both arms are the exact same complete HSMF-Net variant.
            shared = output / 'shared' / arm
            shared.mkdir(parents=True, exist_ok=True)
            band_path = shared / 'indices.npy'
            np.save(band_path, bands[arm])
            cfg = variant_config(base, 'full24', seed, output, manifest_path, shared, band_path)
            cfg['experiment_name'] = f'control_{arm}_seed{seed}'
            cfg['reviewer6'].update(selection_arm=arm, protocol_sha256=digest(plan),
                                    manifest_sha256=digest(manifest), source_sha256=digest(source))
            material = normalize_training_config(cfg)
            material.pop('seed')
            if canonical is None:
                canonical = material
            if material != canonical:
                raise RuntimeError('Material model/training settings differ between arms')
            path = output / 'configs' / f'{arm}_seed{seed}.yaml'
            path.parent.mkdir(exist_ok=True)
            path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
            configs.append((arm, seed, path))
    print('Prepared 10 runs; identical architecture/training; separate 24-band input sets and caches.', flush=True)
    return configs, plan


def summarize(output):
    plan = json.loads((output / 'study_protocol.json').read_text('utf-8'))
    manifest = json.loads((output / 'pair_manifest.json').read_text('utf-8'))
    expected = [(r['sample_id'], int(r['label'])) for r in manifest['records'] if r['split'] == 'test']
    groups = {arm: {} for arm in ARMS}
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            run = output / f'control_{arm}_seed{seed}'
            path = run / 'test_metrics.json'
            if not path.exists():
                continue
            metrics = json.loads(path.read_text('utf-8'))
            if metrics['protocol_sha256'] != digest(plan):
                raise RuntimeError('Completed run uses a different study protocol')
            predictions = run / 'test_predictions.csv'
            if file_hash(predictions) != metrics['predictions_sha256']:
                raise RuntimeError('Prediction hash mismatch')
            with predictions.open(encoding='utf-8') as f:
                pred = list(csv.DictReader(f))
            if [(r['sample_id'], int(r['true_label'])) for r in pred] != expected:
                raise RuntimeError('Test IDs/labels differ from the frozen test set')
            recalculated = metrics_from_rows(pred)
            for key, value in recalculated.items():
                if value != metrics[key]:
                    raise RuntimeError('Reported metric differs from prediction-derived result')
            groups[arm][seed] = metrics
            rows.append({'arm': arm, 'seed': seed, 'correct': metrics['correct'],
                         'accuracy_pct': metrics['accuracy_pct'], 'macro_f1_pct': metrics['macro_f1_pct'],
                         'checkpoint_epoch': metrics['checkpoint_epoch'],
                         'checkpoint_stream': metrics['checkpoint_stream'],
                         'predictions_sha256': metrics['predictions_sha256']})
    csv_write(output / 'per_seed_metrics.csv', rows)
    summaries = []
    for arm, runs in groups.items():
        if not runs:
            continue
        row = {'arm': arm, 'n_runs': len(runs), 'complete_five_seeds': sorted(runs) == SEEDS}
        for metric in ['accuracy_pct', 'macro_f1_pct']:
            row.update({metric+'_'+k: v for k,v in mean_sd_ci([runs[s][metric] for s in sorted(runs)]).items()})
        summaries.append(row)
    csv_write(output / 'model_summary.csv', summaries)
    complete = all(sorted(groups[arm]) == SEEDS for arm in ARMS)
    if complete:
        paired = []
        for metric in ['accuracy_pct', 'macro_f1_pct']:
            diffs = [groups[ARMS[0]][s][metric] - groups[ARMS[1]][s][metric] for s in SEEDS]
            if metric == 'accuracy_pct':
                diffs = [100*(groups[ARMS[0]][s]['correct']-groups[ARMS[1]][s]['correct'])/570 for s in SEEDS]
            paired.append({'contrast': 'refitted_spa24 minus uniform24', 'metric': metric, 'n_seed_pairs': 5,
                           **mean_sd_ci(diffs), 'interval': 'unadjusted two-sided t, df=4; percentage points'})
        csv_write(output / 'paired_spa_uniform_contrasts.csv', paired)
    write_json(output / 'summary_status.json', {'n_completed_runs': len(rows), 'n_planned_runs': 10,
               'complete': complete, 'scope': plan['scope']})
    return len(rows), complete


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT)
    p.add_argument('--run', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--summarize-only', action='store_true')
    args = p.parse_args()
    output = args.output.resolve()
    os.chdir(PROJECT)
    if args.summarize_only:
        print(summarize(output))
        return
    configs, plan = prepare(output)
    if not args.run:
        write_json(output / 'queue_status.json', {'state': 'prepared', 'n_planned_runs': 10, 'completed': summarize(output)[0]})
        return
    queue = output / 'queue_status.json'
    try:
        # Match original interpreter used by the current controlled study.
        if '3.12.' not in sys.version or np.__version__ != '2.3.4':
            raise RuntimeError('Run with the existing yolov3-9.6.0 Python 3.12 / NumPy 2.3.4 environment')
        for arm, seed, config in configs:
            done = output / f'control_{arm}_seed{seed}/test_metrics.json'
            if done.exists():
                summarize(output)
                continue
            verify_metadata(json.loads((output/'pair_manifest.json').read_text('utf-8')))
            write_json(queue, {'state': 'training', 'active_arm': arm, 'active_seed': seed,
                              'completed': summarize(output)[0], 'n_planned_runs': 10,
                              'started_at_unix': time.time(), 'pid': os.getpid()})
            log = output / 'logs' / f'{arm}_seed{seed}.log'
            log.parent.mkdir(exist_ok=True)
            for relative, expected_hash in plan['source_hashes'].items():
                if file_hash(output/'source_snapshot'/relative) != expected_hash:
                    raise RuntimeError('Frozen worker source changed')
            command = [sys.executable, '-u', str(output/'source_snapshot/scripts/run_fusion_controls.py'), '--worker', str(config)]
            if args.resume:
                command.append('--resume')
            print(f'Starting {arm} seed {seed}; log: {log}', flush=True)
            with log.open('a', encoding='utf-8') as stream:
                subprocess.run(command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            summarize(output)
        count, complete = summarize(output)
        if not complete:
            raise RuntimeError('Queue finished without ten complete verified runs')
        write_json(queue, {'state': 'complete', 'completed': count, 'n_planned_runs': 10, 'finished_at_unix': time.time()})
    except Exception as error:
        write_json(queue, {'state': 'failed', 'error': str(error), 'completed': summarize(output)[0],
                          'n_planned_runs': 10, 'failed_at_unix': time.time()})
        raise


if __name__ == '__main__':
    main()
