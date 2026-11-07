# Paired data and cache layout

The 800 paired examples (200 per cultivar) in examples/ contain camera RGB PNGs, ENVI headers and same-stem binary files. Their manifest records archive members, class labels, sample IDs and file hashes. Complete dataset archives are retained separately and are not part of this code directory; archive membership is not the study inclusion list.

## Complete-data contract

Keep camera images and hyperspectral exports in separate roots:

```text
rgb_camera/CX186/CX186_0001.png
hsi/CX186/hsi/CX186_0001.HDR
hsi/CX186/hsi/CX186_0001
```

Use the same pattern for XK129, ZM22 and ZM379. Configure data.rgb_root to rgb_camera and data.hsi_root to hsi. Pair by cultivar and filename stem. Preserve the ENVI header/binary relation; do not rename the binary independently. Camera inputs must be the intended camera images, not spectral display panels sharing their stem. Retain the frozen manifest and its train/validation/test membership rather than deriving a new split from extra archive files.

The supplied historical portable configs adjust paths only. Current control scripts can load explicit pair-manifest entries; the existing current-control image correspondence still requires reconciliation with the camera dataset. This document does not assert that replacing paths alone reproduces any historical accuracy.

## Statistics and caches

preprocessing/normalization/index.json maps the three supplied current-control mean/std caches to their ordered selected bands and 35 run identities. These are not the missing historical S1–S8 normalization caches. The loader’s native per-kernel NPZ cache stores a cube array. Any separately distributed NPY dataset must declare shape, dtype, axis order, sample ordering, class labels and split membership; no complete dataset NPY is asserted here.

## Checks

After installing project dependencies, run python reproducibility/verify_example_inputs.py from the repository root to inspect the packaged examples. This verifies hashes, RGB dimensions and finite 454-band HSI parsing, without model inference or accuracy estimation. Full-dataset installation/execution and actual access publication remain pending.
