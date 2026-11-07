# HSMF-Net

Implementation and experimental data records for paired RGB–HSI wheat cultivar classification.

## Contents

| Directory | Contents |
| --- | --- |
| src/ | HSMF-Net and external-model implementations |
| scripts/ | Training, evaluation, fusion controls, spectral controls and external comparisons |
| configs/ | 40 original S1–S8 configurations, two portable reference examples, 35 control configurations and external-model settings |
| datasets/ | 200 camera-RGB/HSI pairs per cultivar (800 total), sample metadata, frozen current partitions and data-layout instructions |
| preprocessing/ | RGB/HSI processing tools and three current-control normalization caches |
| spectral_selection/ | Recorded SPA lists, current training-only fit, selection trace, numerical contract and wavelengths |
| checkpoints/reference/ | S7/S8 seed42 ordinary and EMA reference tensors |
| results/ | Saved predictions, scalar metrics, class-wise/confusion statistics, two-sided contrasts and spectral analysis |
| docs/ | Architecture, baseline and modulation specifications |
| reproducibility/ | Metric checks, example-data reader, weight-loading check and environment metadata |

## Installation

```text
python -m pip install -e .
```

For external-model profiling, install `python -m pip install -e ".[external]"`. The HSI preprocessing application has separate requirements in preprocessing/tools/.

## Verification

Run from the repository root:

```text
python reproducibility/verify_records.py
python reproducibility/verify_controlled_records.py
python reproducibility/verify_example_inputs.py
python reproducibility/recompute_statistics.py
python spectral_selection/replay_frozen_spa.py --output spa_replay_result.json
python reproducibility/inspect_reference_weights.py --stream ema_model
python reproducibility/inspect_reference_weights.py --stream model
```

The numerical commands recalculate saved predictions or current training-matrix selections. The example reader checks file hashes and finite HSI parsing. Reference loading uses synthetic tensors. These checks do not reproduce historical image-based accuracy. The current two-sided inference outputs are in results/statistics/two_sided/.

## Image-based training

```text
python scripts/train.py --config configs/portable/S8_seed42.yaml
```

Complete image-based execution requires the paired layout and frozen membership described in [datasets/DATA_LAYOUT.md](datasets/DATA_LAYOUT.md), with the matching selected bands and normalization cache. Portable examples adjust paths only; adapt them to the actual dataset location before running. The original evaluation entry loads ordinary model weights. The reference files contain S7/S8 seed42 tensors, not all 40 checkpoints or optimizer states.

## Data and result scope

Full data archives are distributed separately from the code directory. This repository contains 800 camera-RGB/HSI pairs (200 per cultivar), not a complete dataset NPY export. The three normalization caches describe current controls, not authenticated historical S1–S8 cache provenance. Current control PNG-to-camera correspondence and the historical prediction weight/input attribution remain unresolved; see reproducibility/evidence/current_control_image_mapping_status.json.

Original S8 results and band lists remain unchanged. Current refitted SPA, archived SPA and Uniform configurations retain separate identities. Five-seed external summaries are in results/external_comparison/; earlier per-seed reference records are in results/external_seed42/. Uniform operation counts are supplied separately from legacy THOP estimates. Historical CPU and external CUDA latency values do not define a common speed ranking.

Source code and numerical records are prepared for publication; an actual full-data access link has not yet been established. GitHub's ordinary repository limit is 100 MiB per file; large data archives are therefore kept outside this directory ([GitHub documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)).
