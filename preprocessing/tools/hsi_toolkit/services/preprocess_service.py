"""
预处理服务 - 数据预处理功能
"""
import numpy as np
from typing import Tuple, Optional
from scipy.signal import savgol_filter

from config import Config
from utils import get_logger, handle_errors, DataValidator
from modules.preprocess import Normalize01, MovingAverage

logger = get_logger(__name__)


class PreprocessService:
    """数据预处理服务"""
    
    @staticmethod
    @handle_errors(title="波段截取", show_dialog=False)
    def crop_bands(cube: np.ndarray, 
                   wavelengths: Optional[list],
                   start_band: int, 
                   end_band: int) -> Tuple[np.ndarray, Optional[list]]:
        """
        截取波段
        
        Args:
            cube: 高光谱数据立方体 (H, W, B)
            wavelengths: 波长列表
            start_band: 起始波段索引
            end_band: 结束波段索引（不包含）
        
        Returns:
            tuple: (cropped_cube, cropped_wavelengths)
        """
        logger.info(f"波段截取: {start_band} - {end_band}")
        
        # 验证参数
        total_bands = cube.shape[2]
        if start_band < 0 or end_band > total_bands or start_band >= end_band:
            raise ValueError(
                f"波段范围无效: start={start_band}, end={end_band}, total={total_bands}"
            )
        
        # 截取
        cropped_cube = cube[:, :, start_band:end_band].copy()
        
        # 截取波长
        cropped_wavelengths = None
        if wavelengths:
            cropped_wavelengths = wavelengths[start_band:end_band]
        
        logger.info(f"波段截取完成: {cube.shape[2]} -> {cropped_cube.shape[2]} 个波段")
        return cropped_cube, cropped_wavelengths
    
    @staticmethod
    @handle_errors(title="光谱平滑", show_dialog=False)
    def smooth_spectrum(cube: np.ndarray, method: str = "MA", **kwargs) -> np.ndarray:
        """
        光谱平滑
        
        Args:
            cube: 高光谱数据立方体
            method: 平滑方法 ('MA' 或 'SG')
            **kwargs: 方法参数
        
        Returns:
            np.ndarray: 平滑后的数据立方体
        """
        logger.info(f"光谱平滑: method={method}")
        
        if method == "MA":
            # 移动平均
            window = kwargs.get('window', Config.PREPROCESS['smooth']['MA']['window_size'])
            step = MovingAverage(window=window)
            smoothed_cube = step.apply(cube)
            logger.info(f"移动平均平滑完成: window={window}")
        
        elif method == "SG":
            # Savitzky-Golay滤波
            window_length = kwargs.get('window_length', 
                                      Config.PREPROCESS['smooth']['SG']['window_length'])
            polyorder = kwargs.get('polyorder', 
                                  Config.PREPROCESS['smooth']['SG']['polyorder'])
            
            # 确保窗口长度为奇数
            if window_length % 2 == 0:
                window_length += 1
            
            smoothed_cube = savgol_filter(cube, window_length, polyorder, axis=2)
            logger.info(f"SG平滑完成: window={window_length}, polyorder={polyorder}")
        
        else:
            raise ValueError(f"不支持的平滑方法: {method}")
        
        return smoothed_cube
    
    @staticmethod
    @handle_errors(title="光谱求导", show_dialog=False)
    def derivative_spectrum(cube: np.ndarray, 
                           wavelengths: Optional[list],
                           order: int = 1) -> Tuple[np.ndarray, Optional[list]]:
        """
        光谱求导
        
        Args:
            cube: 高光谱数据立方体
            wavelengths: 波长列表
            order: 导数阶数 (1 或 2)
        
        Returns:
            tuple: (derivative_cube, updated_wavelengths)
        """
        logger.info(f"光谱求导: order={order}")
        
        if order not in [1, 2]:
            raise ValueError(f"导数阶数必须是1或2，当前: {order}")
        
        # 计算导数
        derivative_cube = np.diff(cube, n=order, axis=2)
        
        # 更新波长
        updated_wavelengths = None
        if wavelengths:
            updated_wavelengths = wavelengths[:-order]
        
        logger.info(f"{order}阶导数完成: {cube.shape[2]} -> {derivative_cube.shape[2]} 个波段")
        return derivative_cube, updated_wavelengths
    
    @staticmethod
    @handle_errors(title="数据归一化", show_dialog=False)
    def normalize_data(cube: np.ndarray, method: str = "minmax") -> np.ndarray:
        """
        数据归一化
        
        Args:
            cube: 高光谱数据立方体
            method: 归一化方法 ('minmax' 归一化到[0,1])
        
        Returns:
            np.ndarray: 归一化后的数据立方体
        """
        logger.info(f"数据归一化: method={method}")
        
        if method == "minmax":
            # Min-Max归一化到[0,1]
            step = Normalize01()
            normalized_cube = step.apply(cube)
            logger.info(f"Min-Max归一化完成: range=[{normalized_cube.min():.4f}, {normalized_cube.max():.4f}]")
        
        else:
            raise ValueError(f"不支持的归一化方法: {method}")
        
        return normalized_cube
    
    @staticmethod
    @handle_errors(title="数据标准化", show_dialog=False)
    def standardize_data(cube: np.ndarray) -> np.ndarray:
        """
        数据标准化 (Z-score标准化)
        
        Args:
            cube: 高光谱数据立方体
        
        Returns:
            np.ndarray: 标准化后的数据立方体
        """
        logger.info("数据标准化 (Z-score)")
        
        mean = np.nanmean(cube)
        std = np.nanstd(cube)
        
        if std == 0:
            logger.warning("数据标准差为0，无法标准化")
            return cube
        
        standardized_cube = (cube - mean) / std
        
        logger.info(f"Z-score标准化完成: mean={mean:.4f}, std={std:.4f}")
        return standardized_cube
    
    @staticmethod
    @handle_errors(title="应用预处理管道", show_dialog=False)
    def apply_pipeline(cube: np.ndarray,
                       wavelengths: Optional[list],
                       steps: list) -> Tuple[np.ndarray, Optional[list]]:
        """
        应用预处理管道
        
        Args:
            cube: 高光谱数据立方体
            wavelengths: 波长列表
            steps: 预处理步骤列表，例如:
                [
                    ('crop', {'start': 0, 'end': 100}),
                    ('smooth', {'method': 'SG', 'window': 5}),
                    ('normalize', {})
                ]
        
        Returns:
            tuple: (processed_cube, updated_wavelengths)
        """
        logger.info(f"应用预处理管道: {len(steps)}个步骤")
        
        processed_cube = cube
        processed_wavelengths = wavelengths
        
        for step_name, params in steps:
            logger.debug(f"执行步骤: {step_name}")
            
            if step_name == 'crop':
                processed_cube, processed_wavelengths = PreprocessService.crop_bands(
                    processed_cube, processed_wavelengths,
                    params['start'], params['end']
                )
            
            elif step_name == 'smooth':
                method = params.pop('method', 'MA')
                processed_cube = PreprocessService.smooth_spectrum(
                    processed_cube, method, **params
                )
            
            elif step_name == 'derivative':
                order = params.get('order', 1)
                processed_cube, processed_wavelengths = PreprocessService.derivative_spectrum(
                    processed_cube, processed_wavelengths, order
                )
            
            elif step_name == 'normalize':
                method = params.get('method', 'minmax')
                processed_cube = PreprocessService.normalize_data(
                    processed_cube, method
                )
            
            elif step_name == 'standardize':
                processed_cube = PreprocessService.standardize_data(processed_cube)
            
            else:
                logger.warning(f"未知的预处理步骤: {step_name}")
        
        logger.info("预处理管道执行完成")
        return processed_cube, processed_wavelengths

