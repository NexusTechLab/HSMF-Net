# Spectral preprocessing

Savitzky-Golay smoothing uses an 11-band window and a third-order polynomial with derivative order zero and nearest-value boundary handling. Foreground averaging produces one spectrum per kernel for band selection.

Band indices are zero-based. For 454 bands, excluding the first 10 and last 15 leaves candidate indices 10 through 438. Selected input bands are ordered by increasing index. Fixed SPA-24 and SPA-50 lists and the wavelength calibration are in `spectral_selection/`.

The SPA implementation in `src/mobilenet_enterprise/data/wheat_seed_hsi.py` standardizes candidate columns and selects successive projection residuals. Its initial maximum-variance choice after standardization is sensitive to floating-point rounding; use the supplied fixed indices when the same input bands are required. Estimate new selections from training data only.
