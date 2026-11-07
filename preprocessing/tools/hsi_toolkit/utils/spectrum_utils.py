"""
光谱工具 - 统一的光谱数据提取和处理
"""
import numpy as np
from typing import Optional, Dict, List
from .logger import get_logger
from .error_handler import DataError

logger = get_logger(__name__)


class SpectrumExtractor:
    """光谱提取器 - 提供统一的光谱数据提取方法"""
    
    @staticmethod
    def extract_roi_spectrum(cube: np.ndarray, 
                            mask: np.ndarray, 
                            wavelengths: Optional[list] = None,
                            method: str = 'mean') -> Dict:
        """
        从ROI提取光谱
        
        Args:
            cube: 高光谱数据立方体 (H, W, B)
            mask: ROI掩膜 (H, W)
            wavelengths: 波长列表
            method: 提取方法 ('mean', 'median', 'std')
        
        Returns:
            dict: {
                'spectrum': np.ndarray,
                'wavelengths': list or None,
                'valid': bool,
                'n_pixels': int,
                'method': str
            }
        """
        if cube.shape[:2] != mask.shape:
            raise DataError(
                f"立方体和掩膜尺寸不匹配: cube={cube.shape[:2]}, mask={mask.shape}"
            )
        
        # 扩展掩膜到3D
        mask_3d = np.repeat(mask[:, :, None], cube.shape[2], axis=2)
        
        # 应用掩膜
        masked_cube = cube.copy()
        masked_cube[~mask_3d] = np.nan
        
        # 计算有效像素数
        n_pixels = int(np.sum(mask))
        
        # 根据方法提取光谱
        if method == 'mean':
            spectrum = np.nanmean(masked_cube, axis=(0, 1))
        elif method == 'median':
            spectrum = np.nanmedian(masked_cube, axis=(0, 1))
        elif method == 'std':
            spectrum = np.nanstd(masked_cube, axis=(0, 1))
        else:
            raise ValueError(f"不支持的提取方法: {method}")
        
        # 检查是否有效
        valid = not np.all(np.isnan(spectrum)) and n_pixels > 0
        
        if not valid:
            logger.warning(f"ROI光谱提取失败: n_pixels={n_pixels}, all_nan={np.all(np.isnan(spectrum))}")
        
        return {
            'spectrum': spectrum,
            'wavelengths': wavelengths,
            'valid': valid,
            'n_pixels': n_pixels,
            'method': method
        }
    
    @staticmethod
    def extract_pixel_spectrum(cube: np.ndarray, 
                               row: int, 
                               col: int,
                               wavelengths: Optional[list] = None) -> Dict:
        """
        提取单个像素的光谱
        
        Args:
            cube: 高光谱数据立方体
            row: 行坐标
            col: 列坐标
            wavelengths: 波长列表
        
        Returns:
            dict: 光谱数据字典
        """
        if row < 0 or row >= cube.shape[0] or col < 0 or col >= cube.shape[1]:
            raise DataError(f"像素坐标超出范围: ({row}, {col}), shape={cube.shape[:2]}")
        
        spectrum = cube[row, col, :]
        
        return {
            'spectrum': spectrum,
            'wavelengths': wavelengths,
            'position': (row, col),
            'valid': not np.all(np.isnan(spectrum))
        }
    
    @staticmethod
    def extract_all_grains_spectra(cube: np.ndarray,
                                    labels: np.ndarray,
                                    regions: List,
                                    wavelengths: Optional[list] = None) -> Dict[str, np.ndarray]:
        """
        批量提取所有籽粒的平均光谱
        
        Args:
            cube: 高光谱数据立方体
            labels: 标签图
            regions: 区域列表
            wavelengths: 波长列表
        
        Returns:
            dict: {grain_name: spectrum_array}
        """
        all_spectra = {}
        
        for idx, region in enumerate(regions, start=1):
            bbox = region.bbox
            if len(bbox) == 6:
                minr, minc, maxr, maxc = bbox[0], bbox[1], bbox[3], bbox[4]
            else:
                minr, minc, maxr, maxc = bbox
            
            # 提取子立方体
            sub_cube = cube[minr:maxr, minc:maxc, :].copy()
            
            if sub_cube.size == 0:
                continue
            
            # 提取掩膜
            sub_mask = labels[minr:maxr, minc:maxc] == idx
            if sub_mask.ndim == 3:
                sub_mask = np.squeeze(sub_mask, axis=-1)
            
            # 提取光谱
            result = SpectrumExtractor.extract_roi_spectrum(
                sub_cube, sub_mask, wavelengths
            )
            
            if result['valid']:
                grain_name = f"籽粒-{idx:04d}"
                all_spectra[grain_name] = result['spectrum']
        
        logger.info(f"成功提取 {len(all_spectra)}/{len(regions)} 个籽粒的光谱")
        
        return all_spectra
    
    @staticmethod
    def normalize_spectrum(spectrum: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        归一化光谱
        
        Args:
            spectrum: 光谱数组
            method: 归一化方法 ('minmax', 'zscore', 'snv')
        
        Returns:
            np.ndarray: 归一化后的光谱
        """
        spectrum = spectrum.copy()
        
        if method == 'minmax':
            # Min-Max归一化到[0,1]
            min_val = np.nanmin(spectrum)
            max_val = np.nanmax(spectrum)
            if max_val > min_val:
                spectrum = (spectrum - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            # Z-score标准化
            mean_val = np.nanmean(spectrum)
            std_val = np.nanstd(spectrum)
            if std_val > 0:
                spectrum = (spectrum - mean_val) / std_val
        
        elif method == 'snv':
            # 标准正态变换 (SNV)
            mean_val = np.nanmean(spectrum)
            std_val = np.nanstd(spectrum)
            if std_val > 0:
                spectrum = (spectrum - mean_val) / std_val
        
        else:
            raise ValueError(f"不支持的归一化方法: {method}")
        
        return spectrum
    
    @staticmethod
    def smooth_spectrum(spectrum: np.ndarray, 
                       method: str = 'SG',
                       window_size: int = 5,
                       polyorder: int = 2) -> np.ndarray:
        """
        平滑光谱
        
        Args:
            spectrum: 光谱数组
            method: 平滑方法 ('MA', 'SG')
            window_size: 窗口大小
            polyorder: 多项式阶数 (仅用于SG)
        
        Returns:
            np.ndarray: 平滑后的光谱
        """
        from scipy.signal import savgol_filter
        
        if method == 'MA':
            # 移动平均
            kernel = np.ones(window_size) / window_size
            return np.convolve(spectrum, kernel, mode='same')
        
        elif method == 'SG':
            # Savitzky-Golay滤波
            window_length = min(window_size, len(spectrum))
            if window_length % 2 == 0:
                window_length -= 1
            window_length = max(5, window_length)
            
            return savgol_filter(spectrum, window_length, polyorder)
        
        else:
            raise ValueError(f"不支持的平滑方法: {method}")

