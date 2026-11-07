"""Backward-compatible alias for the pure 3D MobileNetV3 HSI encoder."""

from mobilenet_enterprise.models.three_d.mobilenet_v3_pure3d import MobileNetV3Pure3D

Pure3DHSINet = MobileNetV3Pure3D

__all__ = ["Pure3DHSINet", "MobileNetV3Pure3D"]
