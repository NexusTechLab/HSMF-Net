# SPA numerical contract and frozen-refit replay

This records the implemented current refit. It does not certify historical fitting membership or make standardized-variance initialization robust. Reported trained band sets are unchanged.

## Four distinct identities

| Identity | Meaning |
| --- | --- |
| Historical SPA-24 | Archived fixed index list; original fit input/environment not authenticated |
| Historical SPA-50 | Archived fixed index list; matching a current refit does not authenticate its original membership |
| Training-refitted SPA-24 | Current frozen 2,660-record training-only refit, with hashes and selection trace |
| Uniform-24 | Fixed physical-wavelength targets mapped to nearest calibrated bands; not a SPA output |

## Numerical contract

Original saved processed spectra: float32, 2,660×454. Candidate indices 10–438. The current recorded standardized matrix is F-contiguous float32, as verified by spa_initialization_numerical_audit.json. Recorded refit: Python 3.12.11, NumPy 2.3.4, requested one linear-algebra thread. The original BLAS/LAPACK vendor/version is not recoverable from those records. The new replay records its own NumPy build information and requested thread environment, without claiming it is the historical library or independently measured worker count.

Initialization after column standardization is numerically ill-conditioned as a selection rule: nonconstant variances are approximately one. The observed initial indices 12/F-float32, 189/C-float32 and 24/float64 demonstrate sensitivity. They are not an explanation for historical caches, and only first-step sensitivity was tested in that audit.

## Complete selector pseudocode

```text
Input: frozen training-only foreground-mean spectra after recorded preprocessing;
       band budget K; ascending candidate indices J = [10,...,438].
X = float32(spectra[:, J]); use recorded F-contiguous arithmetic path.
X = X - mean(X, axis=0, dtype=float32)
sd = std(X, axis=0, ddof=0, dtype=float32)
sd[sd < 1e-8] = 1
Z = F_contiguous_float32(X / sd)
j1 = first argmax(var(Z, axis=0, ddof=0, dtype=float32))
selected = [j1]
while len(selected) < K:
    best_score = -1
    for candidate j in ascending candidates not in selected:
        B = Z[:, selected]
        coefficient = numpy.linalg.lstsq(B, Z[:, j], rcond=None)[0]
        residual = Z[:, j] - B @ coefficient
        score = Euclidean_norm(residual)
        if score > best_score: update best_score and selected_candidate
    append selected_candidate; record index and winning residual norm
Return progressive J[selected]; sort a copy only for network input storage.
```

Preprocessing details and the frozen input/content manifest accompany training_refit/controlled_refit_2026-11-05/. spa50_selection_trace.csv supplies each progressive step, physical wavelength and winning residual norm. Step 1 records its column norm; later steps record the winning least-squares residual. Independent 24-step/prefix agreement was recorded in refit_report.json.

## Guarded replay

From repository root:

```text
python spectral_selection/replay_frozen_spa.py --output reproducibility/evidence/spa_guarded_replay_check.json
```

The program checks the frozen matrix hash, explicitly preserves float32/F-layout, requests one thread before NumPy import and compares all 50 indices with the immutable recorded trajectory. It aborts on any mismatch instead of replacing lists or presenting different bands as the trained configuration. A successful run in the additional current environment is retained in spa_guarded_replay_2026-11-06.json. This is finite replay evidence, not a guarantee of cross-environment determinism. A new robust initialization policy would define a new selector and require fitting plus new model evaluation; it has not been substituted here.
