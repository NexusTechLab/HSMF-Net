# ==========================================================
# 文件名: preprocess.py
# 模块功能: 高光谱数据预处理（Hyperspectral Preprocessing Pipeline）
# ----------------------------------------------------------
# 本模块提供科研级高光谱数据的标准预处理流程，包括：
#   - 空间与光谱维度裁剪（ROI/Band）
#   - 光谱归一化、标准化、平滑与导数变换
#   - 散射校正（SNV）
#
# 设计特点：
#   1. 可组合的流水线式结构（Pipeline）
#   2. 每一步都封装为独立的 Step 类，便于扩展与可视化控制
#   3. 支持光谱分析、分类、聚类、PCA 等任务的前处理
#
# ==========================================================

from __future__ import annotations
from typing import Callable, List
import numpy as np

# ==========================================================
# ⬛ Base Step 基类
# ----------------------------------------------------------
# 所有预处理步骤都应继承自 Step，并实现 apply() 方法。
# ==========================================================
class Step:
    """Base preprocessing step"""
    name: str = "Base"

    def apply(self, cube: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ==========================================================
# ⬛ ROI 区域裁剪
# ----------------------------------------------------------
# 用于去除无关背景区域，减少计算量。
# 在光谱提取实验中，ROI 通常由种子区域掩膜或手动框选获得。
# ==========================================================
class ROICrop(Step):
    name = "ROICrop"
    def __init__(self, y0: int, y1: int, x0: int, x1: int):
        self.y0, self.y1, self.x0, self.x1 = y0, y1, x0, x1

    def apply(self, cube: np.ndarray) -> np.ndarray:
        return cube[self.y0:self.y1, self.x0:self.x1, :]


# ==========================================================
# ⬛ 波段截取（BandCrop）
# ----------------------------------------------------------
# 剔除仪器边缘噪声波段或无效波长区。
# 常见做法：保留 950–1700 nm 的稳定范围。
# ==========================================================
class BandCrop(Step):
    name = "BandCrop"
    def __init__(self, start: int, end: int):
        self.start, self.end = start, end

    def apply(self, cube: np.ndarray) -> np.ndarray:
        return cube[:, :, self.start:self.end]


# ==========================================================
# ⬛ 归一化 Normalize01
# ----------------------------------------------------------
# 将光谱数据线性映射到 [0, 1]。
# ✅ 优点：
#   - 消除量纲差异
#   - 方便可视化与机器学习模型输入
# ⚠️ 注意：
#   - 不改变光谱形状，仅缩放幅度。
# ==========================================================
class Normalize01(Step):
    name = "Normalize01"
    def apply(self, cube: np.ndarray) -> np.ndarray:
        mx, mn = np.nanmax(cube), np.nanmin(cube)
        if mx == mn:
            return cube
        return (cube - mn) / (mx - mn)


# ==========================================================
# ⬛ 标准化 StandardScale (Z-score)
# ----------------------------------------------------------
# 将所有像素的光谱整体中心化到均值为0、方差为1。
# ✅ 用于：
#   - PCA、聚类、PLS 等算法前的数据标准化
# ⚠️ 对单像素光谱意义有限，更多用于整体矩阵标准化。
# ==========================================================
class StandardScale(Step):
    name = "StandardScale"
    def apply(self, cube: np.ndarray) -> np.ndarray:
        mean, std = np.nanmean(cube), np.nanstd(cube)
        if std == 0:
            return cube - mean
        return (cube - mean) / std


# ==========================================================
# ⬛ 光谱平滑（Spectral Smoothing）
# ----------------------------------------------------------
# 目的：
#   - 去除探测噪声与随机抖动，提高信噪比。
#   - 保留主要吸收特征，增强曲线连续性。
#
# 对高光谱数据的影响：
#   ✅ 优点：
#       1. 提高信噪比，使曲线更清晰；
#       2. 避免导数运算放大噪声；
#   ⚠️ 风险：
#       - 过度平滑会掩盖细节特征。
#
# 推荐窗口大小：
#   - 小窗口(3~5)：保留细节；
#   - 大窗口(>9)：平滑强但风险高。
# ==========================================================
class MovingAverage(Step):
    name = "MovingAverage"
    def __init__(self, window: int = 3):
        self.window = window

    def apply(self, cube: np.ndarray) -> np.ndarray:
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(cube, size=self.window, axis=2, mode='nearest')


# ==========================================================
# ⬛ Savitzky–Golay 平滑
# ----------------------------------------------------------
# 通过多项式拟合窗口内光谱点，既平滑又保留吸收峰形状。
# ✅ 优点：
#   - 曲线保真度高
#   - 可同时用于求导
# ⚠️ 注意：
#   - 窗口需为奇数，polyorder < window。
# ==========================================================
class SavitzkyGolay(Step):
    name = "SavitzkyGolay"
    def __init__(self, window: int = 5, polyorder: int = 2):
        self.window, self.polyorder = window, polyorder

    def apply(self, cube: np.ndarray) -> np.ndarray:
        from scipy.signal import savgol_filter
        return savgol_filter(cube, window_length=self.window,
                           polyorder=self.polyorder, axis=2)


# ==========================================================
# ⬛ 光谱导数变换（Derivative Transform）
# ----------------------------------------------------------
# 一阶导数：
#   - 去除基线漂移；
#   - 强调光谱变化速率；
# 二阶导数：
#   - 突出细微吸收特征；
#   - 常用于检测“吸收谷”位置。
# ⚠️ 注意：
#   - 对噪声敏感，建议先平滑。
# ==========================================================
class FirstDerivative(Step):
    name = "FirstDerivative"
    def apply(self, cube: np.ndarray) -> np.ndarray:
        return np.diff(cube, n=1, axis=2)

class SecondDerivative(Step):
    name = "SecondDerivative"
    def apply(self, cube: np.ndarray) -> np.ndarray:
        return np.diff(cube, n=2, axis=2)


# ==========================================================
# ⬛ SNV 标准正态变换 (Standard Normal Variate)
# ----------------------------------------------------------
# 对每个像素的光谱单独标准化（消除散射效应）。
# ✅ 优点：
#   - 消除样品表面不均匀或粒径差异引起的反射强度变化；
#   - 常与 MSC 或导数配合使用。
# ⚠️ 不建议用于整图全局操作，应逐像素独立计算。
# ==========================================================
class SNVTransform(Step):
    name = "SNV"
    def apply(self, cube: np.ndarray) -> np.ndarray:
        h, w, bands = cube.shape
        cube_2d = cube.reshape(-1, bands)
        means = np.nanmean(cube_2d, axis=1, keepdims=True)
        stds = np.nanstd(cube_2d, axis=1, keepdims=True)
        stds[stds == 0] = 1.0
        snv_2d = (cube_2d - means) / stds
        return snv_2d.reshape(h, w, bands)


# ==========================================================
# ⬛ Pipeline 预处理流水线
# ----------------------------------------------------------
# 可顺序执行多个预处理步骤，适合科研分析与批处理。
# 支持进度回调与可视化集成。
# ==========================================================
class Pipeline:
    def __init__(self, steps: List[Step]):
        self.steps = steps

    def run(self, cube: np.ndarray, progress: Callable[[str, int], None] | None = None) -> np.ndarray:
        result = cube.copy()
        total = len(self.steps)

        for i, step in enumerate(self.steps):
            if progress:
                progress(step.name, int(100 * i / total))
            result = step.apply(result)

        if progress:
            progress("完成", 100)
        return result

    def add_step(self, step: Step):
        """动态添加步骤"""
        self.steps.append(step)

    def clear(self):
        """清空所有步骤"""
        self.steps = []
