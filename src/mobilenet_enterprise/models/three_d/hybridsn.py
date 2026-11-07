from __future__ import annotations

import torch
from torch import nn

from mobilenet_enterprise.models.base import MobileNetBase


class HybridSN(MobileNetBase):
    """Classic HybridSN 3D-to-2D hyperspectral classifier.

    The three valid 3D convolutions jointly reduce the spectral and spatial
    dimensions. Their output is folded into 2D channels before the final 2D
    convolution, following the architecture proposed for HybridSN.
    """

    def __init__(
        self,
        num_classes: int = 4,
        width_mult: float = 1.0,
        input_channels: int = 1,
        dropout: float = 0.4,
        hsi_bands: int = 24,
        image_size: int = 24,
    ) -> None:
        super().__init__()
        if width_mult != 1.0:
            raise ValueError("HybridSN uses its canonical channel widths; width_mult must be 1.0")
        if input_channels != 1:
            raise ValueError("HybridSN expects a single HSI cube input channel")

        spectral_after_3d = int(hsi_bands) - 6 - 4 - 2
        spatial_after_3d = int(image_size) - 2 - 2 - 2
        spatial_after_2d = spatial_after_3d - 2
        if spectral_after_3d <= 0 or spatial_after_2d <= 0:
            raise ValueError(
                "HybridSN input is too small for valid convolutions: "
                f"hsi_bands={hsi_bands}, image_size={image_size}"
            )

        self.hsi_bands = int(hsi_bands)
        self.image_size = int(image_size)
        self.encoder3d = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3)),
            nn.ReLU(inplace=True),
        )
        self.features2d = nn.Sequential(
            nn.Conv2d(32 * spectral_after_3d, 64, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        flattened_dim = 64 * spatial_after_2d * spatial_after_2d
        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        expected = (self.hsi_bands, self.image_size, self.image_size)
        if x.ndim != 5 or tuple(x.shape[2:]) != expected:
            raise ValueError(
                "HybridSN expects input shaped "
                f"(N, 1, {self.hsi_bands}, {self.image_size}, {self.image_size}), "
                f"got {tuple(x.shape)}"
            )
        x = self.encoder3d(x)
        batch, channels, spectral, height, width = x.shape
        x = x.reshape(batch, channels * spectral, height, width)
        x = self.features2d(x)
        return x.flatten(1)

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


__all__ = ["HybridSN"]
