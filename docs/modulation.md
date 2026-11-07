# Hierarchical bidirectional modulation

At intermediate and late levels, channel projections and feature-grid pooling produce compatible RGB/HSI representations. Each partner-generated coefficient map uses Conv1×1 → Hardswish → Dropout2d(0.35) → Conv1×1. The output layer is zero-initialized. The update is F_tilde = A + gamma * A, without an additive beta term or output bound.

Aligned intermediate/late widths are 40/96. Four pooled vectors are concatenated in the order late RGB, late HSI, intermediate RGB, intermediate HSI; the classifier is 272→192→4. Only modulated intermediate HSI continues through its deeper encoder, while RGB continues from the original intermediate backbone feature.

The no-modulation control retains ACSA, all four vectors and the same classifier. Registered totals differ because modulation adds generators. The saved controls do not establish an independent accuracy gain from modulation. Endpoint diagnostics are in results/statistics/matched24/hbfm_code_audit_2026-11-06.json; paired results are in results/statistics/two_sided/ and matched24/. Current camera-input correspondence is described in the repository data-scope statement.

Source: src/mobilenet_enterprise/models/fusion/rgb_hsi_mid_film.py, spatial_film.py and scripts/control_support.py.
