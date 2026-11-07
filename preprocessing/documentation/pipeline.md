# Paired-image preprocessing

The acquisition account uses independent-camera RGB and HSI measurements obtained sequentially with fixed kernel positions/orientations in a common holder. Centroid row ordering and fixed-holder position checks establish the reported sample pairing; they are distinct from retrospective prediction-row identity verification.

RGB foreground suppression combines normalized HSV value, grayscale and saturation (0.5/0.3/0.2), Otsu-derived thresholds, morphology and boundary alpha blending. Camera crops are resized bilinearly to 224×224 and normalized using ImageNet means/SDs. RGB source scripts are in preprocessing/tools/rgb/.

HSI uses Otsu segmentation on the default middle spectral band, component filtering and a mask broadcast across bands, with background set to zero. White/dark-reference calibration yields relative reflectance. Pixel-level SG uses an 11-band window, polynomial order 3, derivative 0 and nearest boundaries before kernel-mean extraction. HSI Toolkit sources are in preprocessing/tools/hsi_toolkit/.

Selected HSI bands are spatially resized to 24×24 without corner alignment. Per-band means/population SDs are fitted on training foreground pixels; current caches and their ordered-band/run mapping are in preprocessing/normalization/index.json. These current caches do not certify historical normalization provenance.

Paired flips share directions, while RGB RandAugment/erasing and HSI roll/scaling/noise/band dropout are modality-specific. They do not imply pixel-wise registration. Exact settings are retained in the run configurations. Current frozen control PNGs require reconciliation with the camera dataset; file-pairing alone does not resolve that provenance question.
