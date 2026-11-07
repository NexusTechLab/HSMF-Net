# RGB/HSI input preparation

RGB foreground suppression uses intensity and saturation thresholds, morphological processing and boundary blending. Cropped RGB images are resized to 224 by 224 and normalized with ImageNet channel means and standard deviations. The processing utilities are in `../tools/rgb/`.

The HSI application reads ENVI cubes, segments individual kernels and exports each kernel with its header. The data reader selects spectral bands and resizes each cube spatially to 24 by 24. Keep foreground masks consistent with the normalization operation.

When estimating per-band normalization statistics, fit means and standard deviations on training foreground pixels and reuse them for validation and test data. RGB and HSI transforms and paired spatial augmentation are implemented in `src/mobilenet_enterprise/data/`.
