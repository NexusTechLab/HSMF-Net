# HSMF-Net

Model definitions and RGB/HSI data-processing tools for wheat cultivar identification.

## Contents

| Directory | Contents |
| --- | --- |
| `src/mobilenet_enterprise/models/` | HSMF-Net components and comparison-model definitions |
| `src/mobilenet_enterprise/data/` | RGB/HSI readers, pairing, transforms, band selection and normalization |
| `configs/` | S7 and S8 model architecture settings |
| `checkpoints/reference/` | S7 and S8 model weights |
| `datasets/examples/` | 200 paired RGB/HSI samples per cultivar; 800 pairs in total |
| `preprocessing/` | RGB foreground/cropping tools and the HSI processing application |
| `spectral_selection/` | Fixed SPA band-index lists and wavelength calibration |

## Installation

```text
python -m pip install -e .
```

The HSI application has separate dependencies in `preprocessing/tools/hsi_toolkit/requirements.txt`.

## Model definition

```python
from pathlib import Path
import yaml
from mobilenet_enterprise import build_model

config = yaml.safe_load(Path("configs/S8_model.yaml").read_text(encoding="utf-8"))
model = build_model(config)
```

Model construction does not load pretrained weights automatically. Weight formats are described in `checkpoints/reference/README.md`. Data formats and sample labels are described in `datasets/DATA_LAYOUT.md`.
