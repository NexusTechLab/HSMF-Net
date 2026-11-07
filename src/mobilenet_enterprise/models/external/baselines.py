"""Adapters preserve author model cores; every input/framework change is catalogued."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ._cloln_author import BaseNet
from ._convsst_author import VisionTransformer
from ._mft_author import MFT
from ._resnet_rh_author import BasicBlock, ResNet_image

METHODS = {
    "cloln": {"year": 2024, "modalities": "HSI", "source": "cloln", "label": "CLOLN (input-adapted)",
              "doi": "10.1109/TGRS.2024.3350055",
              "adaptation": "Author PyTorch classes; 24 input bands, four classes, same 24x24 object crop; original dropout 0.4 retained."},
    "mixersenet": {"year": 2025, "modalities": "HSI", "source": "mixersenet", "label": "MixerSENet (PyTorch port)",
                   "doi": "10.1109/LGRS.2025.3616338",
                   "adaptation": "Keras architecture port; depth=5, filters=64, SE ratio=8; Keras BN epsilon/momentum/biased running variance and exact GELU retained; softmax removed for logits loss; common training replaces original Adam recipe."},
    "convsst": {"year": 2024, "modalities": "HSI", "source": "convsst", "label": "3D-ConvSST (input-adapted)",
                "doi": "10.48550/arXiv.2404.13252",
                "adaptation": "Author PyTorch classes; native 11x11 positional grid/reshape/pooling expanded to 24x24; 24 bands and four classes; embedding=64, depth=2, heads=8 as author training cell."},
    "resnet_rh": {"year": 2024, "modalities": "RGB+HSI mean spectrum", "source": "resnet_rh", "label": "ResNet-R&H (SPA24-adapted)",
                  "doi": "10.1038/s41598-024-51668-6",
                  "adaptation": "Author four-channel 3x3-stem ResNet18; mean of standardized nonzero HSI foreground; SPA24 vector reshaped 4x6 and bilinearly expanded to 24x24, RGB resized to 24x24; adaptive final pooling; removes author debug print/exit; correct channel ordering; common SPA/normalization replaces original MSC. Substantial input adaptation, not an exact reproduction."},
    "mft": {"year": 2023, "modalities": "RGB+HSI", "source": "mft", "label": "MFT (RGB-HSI adapted)",
            "doi": "10.1109/TGRS.2023.3286826",
            "adaptation": "Author MFT classes, FM=16, 24 bands, four classes; auxiliary input changed to three RGB channels and RGB resized 224->24; patchsize=24 instead of 11. Legacy attention baseline, outside 2024-2026 window."},
    "hsmf": {"year": None, "modalities": "RGB+HSI", "source": None, "label": "HSMF-Net (matched protocol, scratch)",
             "doi": None, "adaptation": "Current author architecture; rerun with same preprocessing and training budget, from scratch like external models; historical pretrained results stay separate."},
}


class KerasBatchNorm2d(nn.BatchNorm2d):
    """Keras defaults: epsilon .001, moving momentum .99, biased batch variance."""
    def __init__(self, width):
        super().__init__(width, eps=1e-3, momentum=0.01)

    def forward(self, x):
        if not self.training:
            return super().forward(x)
        mean = x.mean((0, 2, 3))
        variance = x.var((0, 2, 3), unbiased=False)
        with torch.no_grad():
            self.running_mean.lerp_(mean.detach(), 0.01)
            self.running_var.lerp_(variance.detach(), 0.01)
            self.num_batches_tracked.add_(1)
        normalized = (x - mean[None, :, None, None]) * torch.rsqrt(variance[None, :, None, None] + self.eps)
        return normalized * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class MixerBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.depthwise = nn.ModuleList(nn.Conv2d(width, width, k, padding=k // 2, groups=width) for k in (3, 5, 7))
        self.pointwise = nn.Conv2d(width, width, 1)
        self.activation = nn.GELU()
        self.bn = KerasBatchNorm2d(width)

    def forward(self, x):
        mixed = x + self.depthwise[0](x) + self.depthwise[1](x) + self.depthwise[2](x)
        return self.bn(self.activation(self.pointwise(mixed))) + x


class MixerSENet(nn.Module):
    def __init__(self, bands=24, classes=4, width=64, depth=5):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(bands, width, 1), nn.GELU(), KerasBatchNorm2d(width))
        self.blocks = nn.Sequential(*(MixerBlock(width) for _ in range(depth)))
        self.se = nn.Sequential(nn.Linear(width, width // 8), nn.ReLU(), nn.Linear(width // 8, width), nn.Sigmoid())
        self.head = nn.Linear(width, classes)
        # Keras Conv/Dense default Glorot uniform, zero biases.
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if isinstance(module, nn.Conv2d) and module.groups == width:
                    # Keras depthwise kernel layout is H,W,C,1, not O,I,H,W.
                    k = module.kernel_size[0]
                    bound = (6 / (k * k * (width + 1))) ** 0.5
                    nn.init.uniform_(module.weight, -bound, bound)
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        x = x * self.se(x.mean((2, 3)))[:, :, None, None]
        return self.head(x.mean((2, 3)))


class ExternalAdapter(nn.Module):
    def __init__(self, method, core):
        super().__init__()
        self.method = method
        self.core = core

    def forward(self, rgb, hsi):
        cube = hsi.squeeze(1)
        if self.method in {"cloln", "mixersenet", "convsst"}:
            return self.core(cube)
        rgb_small = F.interpolate(rgb, size=cube.shape[-2:], mode="bilinear", align_corners=False)
        if self.method == "mft":
            return self.core(cube, rgb_small)
        if self.method == "resnet_rh":
            mask = cube.abs().sum(1, keepdim=True).ne(0).to(cube.dtype)
            mean = (cube * mask).sum((2, 3)) / mask.sum((2, 3)).clamp_min(1)
            plane = F.interpolate(mean.reshape(-1, 1, 4, 6), size=cube.shape[-2:], mode="bilinear", align_corners=False)
            return self.core(torch.cat((rgb_small, plane), 1))
        raise ValueError(self.method)


def build_external_model(method, cfg):
    if method == "hsmf":
        from mobilenet_enterprise.registry import build_model
        return build_model(cfg)
    if method == "cloln":
        core = BaseNet(24, 4)
    elif method == "mixersenet":
        core = MixerSENet()
    elif method == "convsst":
        core = VisionTransformer(in_chans=24, n_classes=4, embed_dim=64, depth=2, n_heads=8)
        core.image_size = 24
        core.pos_embed = nn.Parameter(torch.zeros(1, 24 * 24, 64))
        core.pool = nn.AvgPool1d(24 * 24)
    elif method == "mft":
        core = MFT(16, 24, 3, 4, False)
    elif method == "resnet_rh":
        core = ResNet_image(BasicBlock, [2, 2, 2, 2], num_classes=4)
    else:
        raise ValueError(method)
    return ExternalAdapter(method, core)
