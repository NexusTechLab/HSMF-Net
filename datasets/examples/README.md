# Paired RGB / HSI examples

Each cultivar (CX186, XK129, ZM22 and ZM379) contains 200 paired samples: 800 pairs in total. Every pair retains its original filename stem:

```text
CX186/CX186_0025.png   # cropped camera RGB
CX186/CX186_0025.HDR   # ENVI header
CX186/CX186_0025       # corresponding HSI binary
```

The existing 100 test pairs per cultivar are retained. Each cultivar additionally includes the first 50 ascending training IDs and 50 validation IDs from the frozen retained-sample manifest. Original split labels are preserved in manifest.json; the full experimental partition is unchanged. Camera PNGs are copied unchanged from the cropped RGB archive, and HSI files from the recorded same-stem paths. manifest.json records labels, original split, source entries, dimensions and file hashes. No resizing, resampling or normalization is applied during copying.

This is an example subset, not the complete dataset or a new performance evaluation. Filename pairing does not independently authenticate historical physical identities or the image provenance of previous control runs. Run python reproducibility/verify_example_inputs.py from the repository root to check all 800 pairs.
