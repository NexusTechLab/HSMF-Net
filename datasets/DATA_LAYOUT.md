# Paired sample format

`examples/` contains four cultivar folders: CX186, XK129, ZM22 and ZM379. Each folder contains 200 sample groups. Every group shares a filename stem and contains:

- `.png`: an RGB image captured with an independent camera.
- `.HDR`: an ENVI hyperspectral header.
- An extensionless file: the hyperspectral binary paired with the header.

Keep each header and its binary together. HSI cubes have 454 spectral bands; spatial dimensions are read from each header. RGB dimensions are stored in `examples/manifest.json`.

Labels are CX186 = 0, XK129 = 1, ZM22 = 2 and ZM379 = 3. The manifest supplies relative paths, file sizes and SHA-256 checksums. Original split labels are retained: each cultivar contributes 50 training, 50 validation and 100 test examples. These 800 pairs are a subset of the dataset.

Readers and transforms are implemented in `src/mobilenet_enterprise/data/`. The fusion model uses RGB tensors shaped `[N, 3, 224, 224]` and HSI tensors shaped `[N, 1, 24, 24, 24]` after selecting 24 spectral bands and resizing spatially to 24 by 24.
