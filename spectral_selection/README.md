# Spectral selection

- spa/archived/: original SPA-24 and SPA-50 index lists.
- spa/training_refit/controlled_refit_2026-11-05/: current 2,660-record training matrix, member/content ledger, refitted indices, standardization diagnostics and the 50-step trace.
- wavelengths/: recorded calibration and selected-band mapping.
- uniform/: physical-wavelength Uniform-24 definition.
- SPA_NUMERICAL_CONTRACT.md and replay_frozen_spa.py: explicit current numerical path and guarded replay.

The current fit reproduced archived SPA-50 but not archived SPA-24. Do not replace recorded lists while retaining their old trained results. The selector implementation is in src/mobilenet_enterprise/data/wheat_seed_hsi.py; fusion/spectral study entries are in scripts/. The supplied mean-spectrum matrix is not the full paired image dataset.
