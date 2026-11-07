"""
文件工具 - 统一的文件操作
"""
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from .logger import get_logger
from .error_handler import FileError

logger = get_logger(__name__)


class FileHelper:
    """文件操作辅助类"""
    
    @staticmethod
    def save_spectrum_csv(spectra_dict: Dict[str, np.ndarray],
                          wavelengths: Optional[List],
                          output_path: Path,
                          include_stats: bool = False):
        """
        保存光谱数据到CSV
        
        Args:
            spectra_dict: {name: spectrum_array}
            wavelengths: 波长列表
            output_path: 输出文件路径
            include_stats: 是否包含统计信息
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                
                # 写入表头
                grain_names = list(spectra_dict.keys())
                if wavelengths:
                    header = ['Wavelength(nm)'] + grain_names
                else:
                    header = ['Band'] + grain_names
                
                if include_stats:
                    header.extend(['Mean', 'Std', 'Min', 'Max'])
                
                writer.writerow(header)
                
                # 写入数据
                all_spectra = np.array([spectra_dict[name] for name in grain_names])
                num_bands = all_spectra.shape[1]
                
                for i in range(num_bands):
                    if wavelengths and i < len(wavelengths):
                        # 确保波长是浮点数（可能是字符串类型）
                        row = [f"{float(wavelengths[i]):.2f}"]
                    else:
                        row = [i]
                    
                    # 添加各个光谱的值
                    for name in grain_names:
                        row.append(f"{spectra_dict[name][i]:.6f}")
                    
                    # 添加统计信息
                    if include_stats:
                        band_values = all_spectra[:, i]
                        row.extend([
                            f"{np.mean(band_values):.6f}",
                            f"{np.std(band_values):.6f}",
                            f"{np.min(band_values):.6f}",
                            f"{np.max(band_values):.6f}"
                        ])
                    
                    writer.writerow(row)
            
            logger.info(f"CSV文件保存成功: {output_path}")
            
        except Exception as e:
            raise FileError(f"CSV保存失败: {str(e)}") from e
    
    @staticmethod
    def save_spectrum_mat(spectra_dict: Dict[str, np.ndarray],
                          wavelengths: Optional[List],
                          output_path: Path):
        """
        保存光谱数据到MAT文件 (MATLAB格式)
        
        Args:
            spectra_dict: {name: spectrum_array}
            wavelengths: 波长列表
            output_path: 输出文件路径
        """
        try:
            import scipy.io as sio
        except ImportError:
            raise FileError("需要安装scipy库才能保存MAT文件: pip install scipy")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        try:
            # 准备数据
            mat_data = {}
            
            # 光谱数据
            all_spectra = np.array([spectra_dict[name] for name in spectra_dict.keys()])
            mat_data['spectra'] = all_spectra.T  # 转置为 (bands, n_grains)
            
            # 名称列表
            mat_data['grain_names'] = list(spectra_dict.keys())
            
            # 波长
            if wavelengths:
                mat_data['wavelengths'] = np.array(wavelengths)
            
            # 保存
            sio.savemat(output_path, mat_data)
            logger.info(f"MAT文件保存成功: {output_path}")
            
        except Exception as e:
            raise FileError(f"MAT保存失败: {str(e)}") from e
    
    @staticmethod
    def save_spectrum_npy(spectra_dict: Dict[str, np.ndarray],
                          wavelengths: Optional[List],
                          output_path: Path):
        """
        保存光谱数据到NPY文件 (NumPy格式)
        
        Args:
            spectra_dict: {name: spectrum_array}
            wavelengths: 波长列表
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        try:
            # 准备数据
            npy_data = {
                'spectra': spectra_dict,
                'wavelengths': wavelengths,
                'grain_names': list(spectra_dict.keys())
            }
            
            # 保存
            np.save(output_path, npy_data, allow_pickle=True)
            logger.info(f"NPY文件保存成功: {output_path}")
            
        except Exception as e:
            raise FileError(f"NPY保存失败: {str(e)}") from e
    
    @staticmethod
    def create_envi_hdr(shape: tuple,
                        wavelengths: Optional[List],
                        data_type: int = 4,
                        interleave: str = 'bil',
                        description: str = "") -> str:
        """
        创建ENVI HDR文件内容
        
        Args:
            shape: 数据形状 (lines, samples, bands)
            wavelengths: 波长列表
            data_type: ENVI数据类型代码
            interleave: 交织方式
            description: 描述信息
        
        Returns:
            str: HDR文件内容
        """
        lines, samples, bands = shape
        
        hdr_content = f"""ENVI
description = {{{description}}}
samples = {samples}
lines = {lines}
bands = {bands}
header offset = 0
file type = ENVI Standard
data type = {data_type}
interleave = {interleave}
byte order = 0
"""
        
        if wavelengths:
            # 确保波长是浮点数（可能是字符串类型）
            wl_str = ', '.join([f"{float(w):.2f}" for w in wavelengths])
            hdr_content += f"wavelength = {{{wl_str}}}\n"
        
        return hdr_content
    
    @staticmethod
    def ensure_output_dir(base_dir: Path, subdir: str = "") -> Path:
        """
        确保输出目录存在
        
        Args:
            base_dir: 基础目录
            subdir: 子目录名
        
        Returns:
            Path: 输出目录路径
        """
        output_dir = Path(base_dir) / subdir if subdir else Path(base_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        logger.debug(f"输出目录已确保存在: {output_dir}")
        return output_dir
    
    @staticmethod
    def get_unique_filename(base_path: Path, suffix: str = "") -> Path:
        """
        获取唯一的文件名（避免覆盖）
        
        Args:
            base_path: 基础路径
            suffix: 文件后缀
        
        Returns:
            Path: 唯一的文件路径
        """
        base_path = Path(base_path)
        
        if not base_path.exists():
            return base_path
        
        # 添加数字后缀
        stem = base_path.stem
        ext = base_path.suffix
        parent = base_path.parent
        
        counter = 1
        while True:
            new_name = f"{stem}_{counter}{ext}"
            new_path = parent / new_name
            if not new_path.exists():
                return new_path
            counter += 1

