"""批量导出的命名与坐标轴设置。"""
import re
import numpy as np


def validate_filename_prefix(prefix):
    prefix = prefix.strip()
    if re.search(r'[<>:"/\\|?*\x00-\x1f]', prefix) or prefix.endswith('.'):
        raise ValueError("品种名称不能包含路径分隔符或 Windows 文件名禁用字符")
    return prefix


def grain_filename(number, digits=4, prefix=''):
    number_text = f"{number:0{digits}d}"
    return f"{prefix}_{number_text}" if prefix else number_text


def read_axis_limits(viewer):
    limits = {}
    for axis in ('x', 'y'):
        pair = []
        for bound in ('min', 'max'):
            edit = getattr(viewer, f'edit_{axis}{bound}', None)
            text = edit.text().strip() if edit is not None else ''
            value = float(text) if text else None
            if value is not None and not np.isfinite(value):
                raise ValueError("坐标轴范围必须为有限数值")
            pair.append(value)
        if all(v is not None for v in pair) and pair[0] >= pair[1]:
            raise ValueError(f"{axis}Min 必须小于 {axis}Max")
        limits[axis] = tuple(pair)
    return limits


def axis_band_range(cube, wavelengths, x_limits):
    x = (np.asarray(wavelengths, dtype=float)
         if wavelengths is not None and len(wavelengths) else np.arange(cube.shape[2]))
    low, high = x_limits
    selected = np.ones(len(x), dtype=bool)
    if low is not None:
        selected &= x >= low
    if high is not None:
        selected &= x <= high
    indices = np.flatnonzero(selected)
    if not len(indices):
        raise ValueError("当前 X 轴范围内没有可导出的波段，请调整坐标轴范围")
    return int(indices[0]), int(indices[-1]) + 1
