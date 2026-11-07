# Model weights

S7 and S8 reference weights contain two state dictionaries: `model` (ordinary weights) and `ema_model` (exponential moving-average weights). The index lists filenames, stream names and SHA-256 checksums.

Construct the corresponding model using `configs/S7_model.yaml` or `configs/S8_model.yaml`, then load the chosen state dictionary:

```python
import torch

weights = torch.load("checkpoints/reference/S8_seed42.pt", map_location="cpu", weights_only=True)
model.load_state_dict(weights["ema_model"], strict=True)
```

The files contain model tensors only. Set the model to evaluation mode before inference. Filenames identify the supplied references; they do not establish which weight stream produced a previously reported metric.
