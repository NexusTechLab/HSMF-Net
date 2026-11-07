"""
导出服务 - 封装所有数据导出功能
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PySide6.QtWidgets import QMessageBox

from config import Config
from utils.export_settings import grain_filename, validate_filename_prefix
from utils.preprocessing import preprocess_spectrum, selected_preprocess_steps
from utils import (
    get_logger,
    handle_errors,
    DataValidator,
    SpectrumExtractor,
    FileHelper
)

logger = get_logger(__name__)


class ExportService:
    """数据导出服务"""
    
    @staticmethod
    @handle_errors(title="导出ROI光谱")
    def export_roi_spectrum(viewer, format_type: str, output_path: Path) -> bool:
        """
        导出ROI光谱数据
        
        Args:
            viewer: 主视图对象
            format_type: 导出格式 ('csv', 'mat', 'npy')
            output_path: 输出路径
        
        Returns:
            bool: 是否成功
        """
        logger.info(f"开始导出ROI光谱: format={format_type}, path={output_path}")
        
        # 检查数据
        if viewer.cube is None:
            raise ValueError("没有加载数据")
        
        if not viewer.roi_list:
            raise ValueError("没有ROI数据")
        
        # 提取所有ROI的光谱
        roi_spectra = {}
        roi_names = []
        
        for roi_info in viewer.roi_list:
            roi_name = roi_info.get('name', 'Unknown')
            roi_obj = roi_info.get('roi')
            
            if roi_obj is None:
                logger.warning(f"跳过无效ROI: {roi_name}")
                continue
            
            try:
                # 获取掩膜
                mask = roi_obj.getArrayRegion(
                    np.ones(viewer.cube.shape[:2]), 
                    viewer.image_view
                ).astype(bool)
                
                # 提取光谱
                result = SpectrumExtractor.extract_roi_spectrum(
                    viewer.cube, mask, viewer.wavelengths
                )
                
                if result['valid']:
                    roi_spectra[roi_name] = result['spectrum']
                    roi_names.append(roi_name)
                    logger.debug(f"成功提取ROI光谱: {roi_name}, n_pixels={result['n_pixels']}")
                
            except Exception as e:
                logger.error(f"提取ROI {roi_name} 失败: {e}")
                continue
        
        if not roi_spectra:
            raise ValueError("无法提取任何有效的ROI光谱数据")
        
        # 根据格式保存
        if format_type == 'csv':
            FileHelper.save_spectrum_csv(roi_spectra, viewer.wavelengths, output_path)
        elif format_type == 'mat':
            FileHelper.save_spectrum_mat(roi_spectra, viewer.wavelengths, output_path)
        elif format_type == 'npy':
            FileHelper.save_spectrum_npy(roi_spectra, viewer.wavelengths, output_path)
        else:
            raise ValueError(f"不支持的格式: {format_type}")
        
        logger.info(f"ROI光谱导出成功: {len(roi_spectra)}个ROI")
        return True
    
    @staticmethod
    @handle_errors(title="导出分割掩膜")
    def export_segmentation_mask(labels: np.ndarray, output_path: Path) -> bool:
        """
        导出分割掩膜
        
        Args:
            labels: 标签图
            output_path: 输出路径
        
        Returns:
            bool: 是否成功
        """
        logger.info(f"导出分割掩膜: {output_path}")
        
        if labels is None:
            raise ValueError("没有分割结果")
        
        import matplotlib.pyplot as plt
        
        plt.imsave(output_path, labels, cmap='nipy_spectral')
        
        logger.info(f"掩膜导出成功: {output_path}")
        return True
    
    @staticmethod
    @handle_errors(title="批量导出ENVI")
    def export_seeds_envi(viewer,
                          output_dir: Path,
                          band_range: Optional[Tuple[int, int]] = None,
                          save_images: bool = True,
                          save_csv: bool = True,
                          start_number: int = 1,
                          number_digits: int = 4,
                          filename_prefix: str = '',
                          apply_preprocess: bool = True,
                          axis_limits: Optional[Dict] = None) -> Dict:
        """
        批量导出籽粒ENVI文件
        
        Args:
            viewer: 主视图对象
            output_dir: 输出目录
            band_range: 波段范围 [start, end)，None表示全部
            save_images: 是否保存PNG图像
            save_csv: 是否保存光谱CSV
            start_number: 文件输出起始编号，不改变分割标签
            number_digits: 输出编号的最小位数，不足补零
            filename_prefix: 小麦品种名称，如 ZM22
            apply_preprocess: 是否追加光谱工具栏勾选的预处理
            axis_limits: PNG 光谱图的 x/y 坐标轴范围
        
        Returns:
            dict: 导出统计信息
        """
        logger.info(f"开始批量导出ENVI: output_dir={output_dir}, band_range={band_range}")
        
        # 验证数据
        if viewer.cube is None:
            raise ValueError("请先打开数据文件")
        
        if viewer.labels is None or not viewer.regions:
            raise ValueError("请先执行分割操作")
        
        DataValidator.validate_hsi_cube(viewer.cube, viewer.wavelengths)
        filename_prefix = validate_filename_prefix(filename_prefix)

        if not isinstance(start_number, (int, np.integer)) or start_number < 0:
            raise ValueError("输出起始编号必须为非负整数")
        if not isinstance(number_digits, (int, np.integer)) or not 1 <= number_digits <= 10:
            raise ValueError("编号位数必须为 1–10 之间的整数")
        if band_range is None:
            band_start, band_end = 0, viewer.cube.shape[2]
        else:
            band_start, band_end = band_range
        if (not all(isinstance(b, (int, np.integer)) for b in (band_start, band_end))
                or not 0 <= band_start < band_end <= viewer.cube.shape[2]):
            raise ValueError("导出波段范围无效，请检查起始和结束波段")
        
        preprocess_steps = selected_preprocess_steps(viewer) if apply_preprocess else []
        logger.info(f"导出预处理: {preprocess_steps or '不追加工具栏预处理'}")

        # 创建输出目录
        output_path = FileHelper.ensure_output_dir(output_dir)
        
        # 提取波长信息
        export_wavelengths = None
        if viewer.wavelengths is not None and len(viewer.wavelengths):
            export_wavelengths = list(viewer.wavelengths[band_start:band_end])
        
        # 批量处理籽粒
        all_spectra = {}
        success_count = 0
        
        total = len(viewer.regions)
        logger.info(f"准备导出 {total} 个籽粒")
        
        for offset, region in enumerate(viewer.regions):
            idx = start_number + offset
            try:
                # 获取边界框
                bbox = region.bbox
                if len(bbox) == 6:
                    minr, minc, maxr, maxc = bbox[0], bbox[1], bbox[3], bbox[4]
                else:
                    minr, minc, maxr, maxc = bbox
                
                # 先对完整光谱预处理，再截取波段，保证范围边缘与光谱图一致。
                sub_cube_full = np.array(viewer.cube[minr:maxr, minc:maxc, :], dtype=float)
                sub_cube_original = sub_cube_full[:, :, band_start:band_end].copy()
                
                if sub_cube_original.size == 0:
                    logger.warning(f"籽粒 {idx} 数据为空，跳过")
                    continue
                
                # 应用掩膜
                # 输出编号和分割标签相互独立，避免自定义编号后提取空掩膜。
                label_id = region.label
                sub_mask = viewer.labels[minr:maxr, minc:maxc] == label_id
                if sub_mask.ndim == 3:
                    sub_mask = np.squeeze(sub_mask, axis=-1)
                
                # ========== 应用预处理（如果启用）==========
                sub_cube_processed = sub_cube_full.copy() if preprocess_steps else sub_cube_original.copy()
                
                if preprocess_steps:
                    # 应用预处理到整个立方体的每个像素
                    h, w, bands = sub_cube_processed.shape
                    for i in range(h):
                        for j in range(w):
                            if sub_mask[i, j]:  # 只处理掩膜内的像素
                                spectrum = sub_cube_processed[i, j, :]
                                x_data = (viewer.wavelengths if viewer.wavelengths is not None
                                          and len(viewer.wavelengths) else np.arange(bands))
                                
                                # 应用与界面相同的预处理逻辑
                                processed_spectrum, _ = preprocess_spectrum(
                                    spectrum, x_data, preprocess_steps
                                )
                                sub_cube_processed[i, j, :] = processed_spectrum
                    sub_cube_processed = sub_cube_processed[:, :, band_start:band_end]
                
                # 提取光谱（使用预处理后的数据）
                result = SpectrumExtractor.extract_roi_spectrum(
                    sub_cube_processed, sub_mask, export_wavelengths
                )
                
                if not result['valid']:
                    logger.warning(f"籽粒 {idx} 光谱无效，跳过")
                    continue
                
                file_name = grain_filename(idx, number_digits, filename_prefix)
                grain_name = file_name if filename_prefix else f"籽粒-{file_name}"
                
                # 保存ENVI文件（使用预处理后的数据）
                ExportService._save_envi_grain(
                    sub_cube_processed, sub_mask, idx,
                    output_path, export_wavelengths, number_digits=number_digits,
                    filename_prefix=filename_prefix
                )
                
                # 保存图像（使用原始数据用于可视化，但光谱使用预处理后的）
                if save_images:
                    img_path = output_path / f"{file_name}.png"
                    ExportService._save_grain_image(
                        sub_cube_original, sub_mask, idx,
                        img_path, export_wavelengths,
                        processed_spectrum=result['spectrum'],
                        number_digits=number_digits, filename_prefix=filename_prefix,
                        axis_limits=axis_limits, band_offset=band_start,
                        preprocess_steps=preprocess_steps
                    )
                
                all_spectra[grain_name] = result['spectrum']
                success_count += 1
                
            except Exception as e:
                logger.error(f"导出籽粒 {idx} 失败: {e}")
                continue
        
        # 保存汇总CSV
        if save_csv and all_spectra:
            csv_path = output_path / "all_grains_spectra.csv"
            FileHelper.save_spectrum_csv(
                all_spectra, export_wavelengths, csv_path, include_stats=True
            )
        
        # 返回统计信息
        stats = {
            'total': total,
            'success': success_count,
            'failed': total - success_count,
            'output_dir': str(output_path),
            'band_range': (band_start, band_end),
            'start_number': start_number,
            'number_digits': number_digits,
            'filename_prefix': filename_prefix,
            'axis_limits': axis_limits,
            'preprocess_steps': preprocess_steps  # 添加预处理步骤信息
        }
        
        logger.info(f"批量导出完成: 成功{success_count}/{total}")
        return stats
    
    @staticmethod
    def _apply_preprocessing(spectrum: np.ndarray, x_data, viewer) -> tuple:
        """与界面共用预处理实现。"""
        return preprocess_spectrum(spectrum, x_data, selected_preprocess_steps(viewer))

    @staticmethod
    def _save_envi_grain(sub_cube: np.ndarray,
                         sub_mask: np.ndarray,
                         idx: int,
                         output_path: Path,
                         wavelengths: Optional[List],
                         number_digits: int = 4,
                         filename_prefix: str = '') -> None:
        """保存单个籽粒的ENVI文件"""
        # 广播掩膜
        sub_mask_3d = np.repeat(sub_mask[:, :, None], sub_cube.shape[2], axis=2)
        
        # 创建掩膜数据（背景=0）
        sub_cube_masked = sub_cube.copy()
        sub_cube_masked[~sub_mask_3d] = 0
        
        # 转换为BIL格式
        lines, samples, bands = sub_cube_masked.shape
        cube_bil = np.transpose(sub_cube_masked, (0, 2, 1)).astype(np.float32)
        
        # 保存RAW文件
        file_name = grain_filename(idx, number_digits, filename_prefix)
        raw_path = output_path / file_name
        with open(raw_path, 'wb') as f:
            cube_bil.tofile(f)
        
        # 生成HDR文件
        hdr_content = FileHelper.create_envi_hdr(
            (lines, samples, bands),
            wavelengths,
            description=f"小麦籽粒 {file_name}"
        )
        
        hdr_path = output_path / f"{file_name}.HDR"
        with open(hdr_path, 'w', encoding='utf-8') as f:
            f.write(hdr_content)
        
        logger.debug(f"ENVI文件保存成功: {file_name}")
    
    @staticmethod
    def _save_grain_image(sub_cube: np.ndarray,
                         sub_mask: np.ndarray,
                         idx: int,
                         img_path: Path,
                         wavelengths: Optional[List],
                         processed_spectrum: Optional[np.ndarray] = None,
                         number_digits: int = 4,
                         filename_prefix: str = '',
                         axis_limits: Optional[Dict] = None,
                         band_offset: int = 0,
                         preprocess_steps: Optional[List] = None) -> None:
        """保存单个籽粒的图像（左：图像，右：光谱）
        
        Args:
            sub_cube: 子立方体数据
            sub_mask: 掩膜
            idx: 籽粒索引
            img_path: 输出路径
            wavelengths: 波长信息
            processed_spectrum: 预处理后的光谱（如果有）
        """
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
        matplotlib.rcParams['axes.unicode_minus'] = False
        
        # 检查掩膜
        if sub_mask.sum() == 0:
            logger.warning(f"籽粒 {idx} 掩膜为空，跳过图像保存")
            return
        
        # 选择显示波段（优先选择波段60，如果不存在则选择中间波段）
        band_idx = 60 - band_offset
        if not 0 <= band_idx < sub_cube.shape[2]:
            band_idx = sub_cube.shape[2] // 2
        single_band = sub_cube[:, :, band_idx]
        
        # 使用预处理后的光谱或计算原始平均光谱
        if processed_spectrum is not None:
            mean_spectrum = processed_spectrum
        else:
            sub_mask_3d = np.repeat(sub_mask[:, :, None], sub_cube.shape[2], axis=2)
            masked_cube = sub_cube.copy()
            masked_cube[~sub_mask_3d] = np.nan
            mean_spectrum = np.nanmean(masked_cube, axis=(0, 1))
        
        # 创建图形
        fig = plt.figure(figsize=(12, 5), dpi=100)
        
        # 左：图像
        ax_img = plt.subplot(1, 2, 1)
        background = np.zeros_like(single_band)
        ax_img.imshow(background, cmap='gray', vmin=0, vmax=1)
        
        masked_band = np.ma.masked_where(~sub_mask, single_band)
        valid_pixels = single_band[sub_mask]
        vmin = valid_pixels.min()
        vmax = np.percentile(valid_pixels, 98)
        
        ax_img.imshow(masked_band, cmap='gray', vmin=vmin, vmax=vmax)
        ax_img.axis('off')
        # 添加波段信息到标题
        file_name = grain_filename(idx, number_digits, filename_prefix)
        ax_img.set_title(f'{file_name} - 波段 {band_idx + band_offset}', fontsize=14, fontweight='bold')
        
        # 右：光谱
        ax_spec = plt.subplot(1, 2, 2)
        
        if wavelengths and len(wavelengths) == len(mean_spectrum):
            # 转换波长为浮点数数组
            try:
                x_data = [float(w) for w in wavelengths]
            except (ValueError, TypeError):
                x_data = np.arange(len(mean_spectrum)) + band_offset
            x_label = '波长 (nm)'
        else:
            x_data = np.arange(len(mean_spectrum)) + band_offset
            x_label = '波段索引'
        
        # 绘制光谱曲线
        ax_spec.plot(x_data, mean_spectrum, 'b-', linewidth=2)
        ax_spec.set_xlabel(x_label, fontsize=12)
        y_label = '光谱值' if preprocess_steps else '反射率'
        ax_spec.set_ylabel(y_label, fontsize=12)
        ax_spec.set_title('平均光谱', fontsize=14, fontweight='bold')
        ax_spec.grid(True, alpha=0.3, linestyle='--')
        
        # 确保X轴刻度标签可见
        ax_spec.tick_params(axis='both', which='major', labelsize=10)
        
        # 如果有波长信息，设置合理的X轴刻度
        if wavelengths and len(wavelengths) == len(mean_spectrum):
            x_min, x_max = min(x_data), max(x_data)
            # 设置5-7个刻度点
            x_ticks = np.linspace(x_min, x_max, 6)
            ax_spec.set_xticks(x_ticks)
            ax_spec.set_xticklabels([f'{int(x)}' for x in x_ticks])

        if axis_limits:
            x_low, x_high = axis_limits.get('x', (None, None))
            y_low, y_high = axis_limits.get('y', (None, None))
            if x_low is not None or x_high is not None:
                ax_spec.set_xlim(left=x_low, right=x_high)
                # 采用指定坐标范围时，让刻度随坐标范围生成。
                from matplotlib.ticker import AutoLocator, ScalarFormatter
                ax_spec.xaxis.set_major_locator(AutoLocator())
                ax_spec.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            if y_low is not None or y_high is not None:
                ax_spec.set_ylim(bottom=y_low, top=y_high)
        
        plt.tight_layout()
        plt.savefig(img_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"籽粒图像保存成功: {img_path}")

