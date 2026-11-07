# Spectral band parameters

- `spa/archived/selected_bands_spa_24_e10_t15_smooth.npy`: fixed 24-band input indices.
- `spa/archived/selected_bands_spa_50_e10_t15_smooth.npy`: fixed 50-band input indices.
- `wavelengths/wavelength_calibration.csv`: the 454-band index-to-wavelength calibration.

All indices are zero-based. The two lists are separate input definitions; the 24-band list is not obtained by truncating the 50-band list. SPA and normalization code are provided in `src/mobilenet_enterprise/data/wheat_seed_hsi.py`.
