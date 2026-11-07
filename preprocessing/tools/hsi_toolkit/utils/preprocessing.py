"""光谱显示与导出共用的预处理规则。"""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter


def selected_preprocess_steps(viewer):
    steps = []
    for group in (
        (('chk_smooth_ma', 'MA平滑'), ('chk_smooth_sg', 'SG平滑')),
        (('chk_deriv_1st', '一阶导数'), ('chk_deriv_2nd', '二阶导数')),
        (('chk_normalize', '归一化'), ('chk_standardize', 'Z-score')),
    ):
        for name, method in group:
            checkbox = getattr(viewer, name, None)
            if checkbox is not None and checkbox.isChecked():
                steps.append(method)
                break
    return steps


def preprocess_spectrum(spectrum, x_data, steps):
    processed = np.array(spectrum, dtype=float, copy=True)
    for method in steps:
        if method == 'MA平滑':
            processed = uniform_filter1d(processed, size=5, mode='nearest')
        elif method == 'SG平滑':
            # 与光谱图使用相同的窗口、三阶多项式和边界模式。
            window = min(11, len(processed) if len(processed) % 2 else len(processed) - 1)
            if window >= 5:
                processed = savgol_filter(processed, window, 3, mode='nearest')
        elif method in ('一阶导数', '二阶导数'):
            if len(processed) < 2:
                raise ValueError("求导至少需要两个波段")
            processed = np.gradient(processed)
            if method == '二阶导数':
                processed = np.gradient(processed)
        elif method == '归一化':
            low, high = processed.min(), processed.max()
            if high > low:
                processed = (processed - low) / (high - low)
        elif method == 'Z-score':
            mean, std = processed.mean(), processed.std()
            if std > 0:
                processed = (processed - mean) / std
        else:
            raise ValueError(f"未知预处理方法: {method}")
    return processed, np.array(x_data, copy=True)
