# Updated external summaries supplied by the author

{
  "date": "2026-11-06",
  "source": "User-provided updated table; user explicitly confirmed five training seeds 42–46",
  "training_seeds": [
    42,
    43,
    44,
    45,
    46
  ],
  "classification_values": "Reported mean and SD supplied by author",
  "latency_values": "Mean and reported SD supplied by author; within-run vs between-run aggregation not separately confirmed",
  "local_run_ledger_at_update": "Existing external five-seed suite still records seed42 imports and additional seeds in progress; not used to independently recompute these new SD values",
  "old_single_seed_records_preserved": true,
  "new_summaries_independently_recomputed": false
}

Table 4(E). External-method comparison.

| Model | Modality | Accuracy (%) | Macro-F1 (%) | Params (M) | GFLOPs* | Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| CLOLN [52] | HSI | 88.25 ± 2.21 | 88.35 ± 2.25 | 0.0032 | 0.0026 | 0.70 ± 1.73 |
| MixerSENet [53] | HSI | 94.39 ± 2.25 | 94.44 ± 2.17 | 0.0520 | 0.0560 | 6.08 ± 1.27 |
| 3D-ConvSST [54] | HSI | 93.51 ± 1.96 | 93.63 ± 2.11 | 0.2294 | 0.4850 | 8.49 ± 1.59 |
| ResNet-R&H [55] | RGB + HSI mean spectrum | 85.26 ± 0.89 | 85.31 ± 0.96 | 11.1715 | 0.6255 | 4.27 ± 0.56 |
| MFT [56] | RGB + HSI | 90.35 ± 1.12 | 90.43 ± 1.27 | 0.2263 | 0.0405 | 7.39 ± 0.79 |
| **HSMF-Net (S8)** | RGB + HSI | 95.58 ± 1.70 | 95.63 ± 1.67 | 2.659 | 0.3294 | 22.73 ± 0.86 |

Accuracy and Macro-F1 are mean ± SD across seeds 42–46. S8 values are unchanged. The external suite counts selected operations at two FLOPs per multiply–accumulate; S8 retains the THOP convention in Section 5.1. Operator coverage and timing protocols differ, so cross-protocol GFLOPs and latency values are not directly ranked. ResNet-R&H uses RGB plus a foreground-mean HSI spectrum; external methods are input-adapted implementations.

The earlier scratch-trained seed42 HSMF-Net reference is retained in ../external_seed42/ as supporting evidence. It is not substituted for S8. Additional per-seed prediction records and the latency-SD aggregation remain unverified in this package.
