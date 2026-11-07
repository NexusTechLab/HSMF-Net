"""
数据验证 - 确保数据完整性和正确性
"""
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from config import Config
from .error_handler import DataError
from .logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_hsi_cube(cube: np.ndarray, wavelengths: Optional[list] = None) -> bool:
        """
        验证高光谱数据立方体
        
        Args:
            cube: 高光谱数据立方体 (H, W, B)
            wavelengths: 波长列表
        
        Returns:
            bool: 验证是否通过
        
        Raises:
            DataError: 数据不符合要求时抛出
        """
        # 检查维度
        if cube.ndim != 3:
            raise DataError(
                f"数据维度错误：期望3D立方体，实际{cube.ndim}维",
                details=f"shape={cube.shape}"
            )
        
        height, width, bands = cube.shape
        
        # 检查尺寸
        if height == 0 or width == 0 or bands == 0:
            raise DataError(
                f"数据尺寸无效: ({height}, {width}, {bands})",
                details="所有维度必须大于0"
            )
        
        # 检查波段数
        if bands > Config.VALIDATION['max_bands']:
            logger.warning(f"波段数({bands})超过建议值({Config.VALIDATION['max_bands']})")
        
        # 检查数据大小
        size_mb = cube.nbytes / (1024 * 1024)
        if size_mb > Config.VALIDATION['max_cube_size_mb']:
            logger.warning(f"数据大小({size_mb:.1f}MB)超过建议值({Config.VALIDATION['max_cube_size_mb']}MB)")
        
        # 检查数据类型
        if not np.issubdtype(cube.dtype, np.floating) and not np.issubdtype(cube.dtype, np.integer):
            raise DataError(
                f"数据类型错误: {cube.dtype}",
                details="期望浮点型或整型数据"
            )
        
        # 检查是否有NaN或Inf
        if np.any(np.isnan(cube)):
            nan_count = np.sum(np.isnan(cube))
            logger.warning(f"数据包含{nan_count}个NaN值")
        
        if np.any(np.isinf(cube)):
            inf_count = np.sum(np.isinf(cube))
            logger.warning(f"数据包含{inf_count}个Inf值")
        
        # 检查波长信息
        if wavelengths is not None:
            if len(wavelengths) != bands:
                raise DataError(
                    f"波长数量({len(wavelengths)})与波段数({bands})不匹配",
                    details=f"wavelengths length={len(wavelengths)}, bands={bands}"
                )
            
            # 检查波长是否单调递增
            wavelengths_array = np.array(wavelengths, dtype=np.float64)
            if not np.all(wavelengths_array[1:] > wavelengths_array[:-1]):
                logger.warning("波长序列不是严格单调递增")
        
        # 检查反射率范围
        min_val, max_val = float(np.nanmin(cube)), float(np.nanmax(cube))
        ref_min, ref_max = Config.VALIDATION['reflectance_range']
        
        if max_val > ref_max:
            logger.info(f"检测到DN值数据 (max={max_val:.2f})，建议转换为反射率")
        
        logger.debug(f"数据验证通过: shape={cube.shape}, range=[{min_val:.4f}, {max_val:.4f}]")
        return True
    
    @staticmethod
    def validate_file_path(path: Path, must_exist: bool = True, extensions: Optional[list] = None) -> bool:
        """
        验证文件路径
        
        Args:
            path: 文件路径
            must_exist: 文件是否必须存在
            extensions: 允许的文件扩展名列表
        
        Returns:
            bool: 验证是否通过
        """
        path = Path(path)
        
        if must_exist and not path.exists():
            raise DataError(f"文件不存在: {path}")
        
        if extensions and path.suffix.lower() not in [ext.lower() for ext in extensions]:
            raise DataError(
                f"文件类型错误: {path.suffix}",
                details=f"允许的类型: {', '.join(extensions)}"
            )
        
        return True
    
    @staticmethod
    def validate_segmentation_params(params: dict, method: str = 'otsu') -> dict:
        """
        验证分割参数
        
        Args:
            params: 分割参数字典
            method: 分割方法
        
        Returns:
            dict: 验证并补全的参数字典
        """
        defaults = Config.get_segmentation_defaults(method)
        validated = defaults.copy()
        validated.update(params)
        
        # 验证参数范围
        if validated.get('threshold', 0) < 0 or validated.get('threshold', 0) > 1:
            logger.warning(f"阈值({validated['threshold']})超出[0,1]范围，已调整")
            validated['threshold'] = max(0, min(1, validated['threshold']))
        
        if validated.get('min_area', 0) < 1:
            validated['min_area'] = 1
            logger.warning("最小面积必须≥1，已调整")
        
        if validated.get('max_area', 0) < validated.get('min_area', 0):
            validated['max_area'] = validated['min_area'] * 10
            logger.warning("最大面积小于最小面积，已自动调整")
        
        return validated
    
    @staticmethod
    def validate_wavelength_range(wavelengths: list, start: float, end: float) -> Tuple[int, int]:
        """
        验证并转换波长范围到波段索引
        
        Args:
            wavelengths: 波长列表
            start: 起始波长
            end: 结束波长
        
        Returns:
            tuple: (band_start, band_end) 波段索引范围
        """
        if not wavelengths:
            raise DataError("波长信息为空")
        
        wavelengths = np.array(wavelengths, dtype=np.float64)
        
        if start >= end:
            raise DataError(f"波长范围错误: start({start}) >= end({end})")
        
        # 转换为波段索引
        band_start = int(np.searchsorted(wavelengths, start, side='left'))
        band_end = int(np.searchsorted(wavelengths, end, side='right'))
        
        # 确保在有效范围内
        band_start = max(0, min(band_start, len(wavelengths) - 1))
        band_end = max(band_start + 1, min(band_end, len(wavelengths)))
        
        logger.debug(f"波长范围 [{start:.1f}, {end:.1f}] nm -> 波段 [{band_start}, {band_end})")
        
        return band_start, band_end

