# Spectral preprocessing and wavelength indexing

Foreground-pixel SG smoothing uses window length 11, polynomial order 3, derivative order 0 and nearest-value boundary handling. SPA then operates on one foreground-averaged spectrum per designated training kernel. Per-kernel anomaly interpolation/boundary treatment is distinct from SG smoothing and from network-input normalization.

SPA columns are centered and divided by population SD; divisors below 1e-8 are set to one. The float32 selector starts at the first exact computed maximum of standardized variance, then maximizes least-squares residual norms in ascending candidate order. Near-unit variances make the initialization numerically sensitive. See spectral_selection/SPA_NUMERICAL_CONTRACT.md and the supplied numerical audit.

Indices are zero-based. With 454 bands, excluding 0–9 and 439–453 leaves candidates 10–438 inclusive. The calibrated selected-band mapping is in spectral_selection/wavelengths/. Input order is ascending wavelength order, not progressive SPA selection order. Archived SPA-24/50 and the current training-only refit are distinct records; a current fit does not recover missing historical fit inputs.
