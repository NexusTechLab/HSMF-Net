"""
科研服务 - 封装所有论文级别的导出功能
"""
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib

from config import Config
from utils import (
    get_logger,
    handle_errors,
    SpectrumExtractor,
    FileHelper
)

logger = get_logger(__name__)


class ResearchService:
    """科研导出服务 - 生成论文级别的图表"""
    
    @staticmethod
    def setup_publication_style():
        """设置论文级别的绘图样式"""
        pub_config = Config.get_export_config('publication')
        matplotlib.rcParams['font.family'] = pub_config['font_family']
        matplotlib.rcParams['font.size'] = pub_config['font_size']
        matplotlib.rcParams['axes.linewidth'] = pub_config['axis_linewidth']
        matplotlib.rcParams['xtick.major.width'] = pub_config['axis_linewidth']
        matplotlib.rcParams['ytick.major.width'] = pub_config['axis_linewidth']
        matplotlib.rcParams['xtick.labelsize'] = pub_config['tick_labelsize']
        matplotlib.rcParams['ytick.labelsize'] = pub_config['tick_labelsize']
    
    @staticmethod
    @handle_errors(title="导出平均光谱图")
    def export_publication_spectrum(viewer, output_dir: Path) -> Path:
        """
        导出论文级别的平均光谱图
        
        Args:
            viewer: 主视图对象
            output_dir: 输出目录
        
        Returns:
            Path: 输出文件路径
        """
        logger.info("开始导出论文级光谱图")
        
        # 验证数据
        if viewer.cube is None or viewer.labels is None or not viewer.regions:
            raise ValueError("请先加载数据并完成分割")
        
        # 提取所有籽粒光谱
        all_spectra_dict = SpectrumExtractor.extract_all_grains_spectra(
            viewer.cube, viewer.labels, viewer.regions, viewer.wavelengths
        )
        
        if not all_spectra_dict:
            raise ValueError("无法提取有效的籽粒光谱数据")
        
        all_spectra = np.array(list(all_spectra_dict.values()))
        wavelengths = np.array(viewer.wavelengths, dtype=np.float64) if viewer.wavelengths else np.arange(all_spectra.shape[1])
        
        # 设置样式
        ResearchService.setup_publication_style()
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        
        # 绘制所有光谱
        colors = plt.cm.jet(np.linspace(0, 1, len(all_spectra)))
        for i, spectrum in enumerate(all_spectra):
            ax.plot(wavelengths, spectrum, linewidth=0.8, alpha=0.7, color=colors[i])
        
        # 设置坐标轴
        ax.set_xlabel('Wavelength (nm)', fontsize=14)
        ax.set_ylabel('Reflectance', fontsize=14)
        ax.set_xlim(wavelengths[0], wavelengths[-1])
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        
        # 设置X轴刻度
        x_ticks = np.arange(np.ceil(wavelengths[0] / 200) * 200, wavelengths[-1], 200)
        ax.set_xticks(x_ticks)
        
        # 美化
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 保存
        output_path = FileHelper.ensure_output_dir(output_dir, "research_export")
        img_path = output_path / "average_spectrum_publication.png"
        
        img_config = Config.get_export_config('image')
        plt.savefig(img_path, **img_config)
        plt.close()
        
        logger.info(f"论文级光谱图已保存: {img_path}, 共{len(all_spectra)}条光谱")
        return img_path
    
    @staticmethod
    @handle_errors(title="导出导数光谱图")
    def export_derivative_spectrum(viewer, output_dir: Path) -> Path:
        """
        导出导数光谱图（一阶和二阶）
        
        Args:
            viewer: 主视图对象
            output_dir: 输出目录
        
        Returns:
            Path: 输出文件路径
        """
        from scipy.signal import savgol_filter
        
        logger.info("开始导出导数光谱图")
        
        # 提取光谱
        all_spectra_dict = SpectrumExtractor.extract_all_grains_spectra(
            viewer.cube, viewer.labels, viewer.regions, viewer.wavelengths
        )
        
        if not all_spectra_dict:
            raise ValueError("无法提取有效的籽粒光谱数据")
        
        all_spectra = np.array(list(all_spectra_dict.values()))
        wavelengths = np.array(viewer.wavelengths, dtype=np.float64) if viewer.wavelengths else np.arange(all_spectra.shape[1])
        
        # 计算导数
        window_length = min(11, len(wavelengths) if len(wavelengths) % 2 == 1 else len(wavelengths) - 1)
        window_length = max(5, window_length)
        
        first_derivatives = []
        second_derivatives = []
        
        for spectrum in all_spectra:
            first_deriv = savgol_filter(spectrum, window_length, polyorder=2, deriv=1)
            second_deriv = savgol_filter(spectrum, window_length, polyorder=2, deriv=2)
            first_derivatives.append(first_deriv)
            second_derivatives.append(second_deriv)
        
        first_derivatives = np.array(first_derivatives)
        second_derivatives = np.array(second_derivatives)
        
        # 设置样式
        ResearchService.setup_publication_style()
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), dpi=150)
        colors = plt.cm.jet(np.linspace(0, 1, len(all_spectra)))
        
        # 一阶导数
        for i, deriv in enumerate(first_derivatives):
            ax1.plot(wavelengths, deriv, linewidth=0.8, alpha=0.7, color=colors[i])
        ax1.set_xlabel('Wavelength (nm)', fontsize=14)
        ax1.set_ylabel('First Derivative', fontsize=14)
        ax1.set_xlim(wavelengths[0], wavelengths[-1])
        ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_title('(A) First Derivative Spectra', fontsize=14, loc='left', fontweight='bold')
        
        # 二阶导数
        for i, deriv in enumerate(second_derivatives):
            ax2.plot(wavelengths, deriv, linewidth=0.8, alpha=0.7, color=colors[i])
        ax2.set_xlabel('Wavelength (nm)', fontsize=14)
        ax2.set_ylabel('Second Derivative', fontsize=14)
        ax2.set_xlim(wavelengths[0], wavelengths[-1])
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_title('(B) Second Derivative Spectra', fontsize=14, loc='left', fontweight='bold')
        
        plt.tight_layout()
        
        # 保存
        output_path = FileHelper.ensure_output_dir(output_dir, "research_export")
        img_path = output_path / "derivative_spectra.png"
        
        img_config = Config.get_export_config('image')
        plt.savefig(img_path, **img_config)
        plt.close()
        
        logger.info(f"导数光谱图已保存: {img_path}")
        return img_path
    
    @staticmethod
    @handle_errors(title="导出分组对比图")
    def export_group_comparison(viewer, output_dir: Path, num_groups: int = 4) -> Path:
        """
        导出分组对比图
        
        Args:
            viewer: 主视图对象
            output_dir: 输出目录
            num_groups: 分组数量
        
        Returns:
            Path: 输出文件路径
        """
        logger.info(f"开始导出分组对比图: num_groups={num_groups}")
        
        # 提取光谱
        all_spectra_dict = SpectrumExtractor.extract_all_grains_spectra(
            viewer.cube, viewer.labels, viewer.regions, viewer.wavelengths
        )
        
        if not all_spectra_dict:
            raise ValueError("无法提取有效的籽粒光谱数据")
        
        all_spectra = np.array(list(all_spectra_dict.values()))
        wavelengths = np.array(viewer.wavelengths, dtype=np.float64) if viewer.wavelengths else np.arange(all_spectra.shape[1])
        
        # 根据平均反射率分组
        mean_reflectance = np.mean(all_spectra, axis=1)
        sorted_indices = np.argsort(mean_reflectance)
        group_size = len(all_spectra) // num_groups
        
        groups = []
        group_labels = []
        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = (i + 1) * group_size if i < num_groups - 1 else len(all_spectra)
            group_indices = sorted_indices[start_idx:end_idx]
            group_spectra = all_spectra[group_indices]
            groups.append(group_spectra)
            group_labels.append(f"Group {i+1}")
        
        # 设置样式
        ResearchService.setup_publication_style()
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        group_colors = plt.cm.tab10(np.linspace(0, 1, num_groups))
        
        for i, (group_spectra, label) in enumerate(zip(groups, group_labels)):
            mean_spectrum = np.mean(group_spectra, axis=0)
            std_spectrum = np.std(group_spectra, axis=0)
            
            ax.plot(wavelengths, mean_spectrum, linewidth=2,
                   color=group_colors[i], label=label, alpha=0.9)
            ax.fill_between(wavelengths,
                           mean_spectrum - std_spectrum,
                           mean_spectrum + std_spectrum,
                           color=group_colors[i], alpha=0.2)
        
        ax.set_xlabel('Wavelength (nm)', fontsize=14)
        ax.set_ylabel('Reflectance', fontsize=14)
        ax.set_xlim(wavelengths[0], wavelengths[-1])
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.legend(loc='best', frameon=False, fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # 保存
        output_path = FileHelper.ensure_output_dir(output_dir, "research_export")
        img_path = output_path / "group_comparison.png"
        
        img_config = Config.get_export_config('image')
        plt.savefig(img_path, **img_config)
        plt.close()
        
        logger.info(f"分组对比图已保存: {img_path}")
        return img_path
    
    @staticmethod
    @handle_errors(title="导出2D相关光谱图")
    def export_2d_correlation(viewer, output_dir: Path) -> Path:
        """
        导出2D相关光谱图（同步谱和异步谱）
        
        Args:
            viewer: 主视图对象
            output_dir: 输出目录
        
        Returns:
            Path: 输出文件路径
        """
        logger.info("开始导出2D相关光谱图")
        
        # 提取光谱
        all_spectra_dict = SpectrumExtractor.extract_all_grains_spectra(
            viewer.cube, viewer.labels, viewer.regions, viewer.wavelengths
        )
        
        if not all_spectra_dict:
            raise ValueError("无法提取有效的籽粒光谱数据")
        
        all_spectra = np.array(list(all_spectra_dict.values()))
        wavelengths = np.array(viewer.wavelengths, dtype=np.float64) if viewer.wavelengths else np.arange(all_spectra.shape[1])
        
        # 计算2D相关光谱
        mean_spectrum = np.mean(all_spectra, axis=0)
        centered_spectra = all_spectra - mean_spectrum
        
        # 同步谱
        sync_spectrum = np.dot(centered_spectra.T, centered_spectra) / (len(all_spectra) - 1)
        
        # 异步谱（Hilbert-Noda变换）
        n = len(wavelengths)
        noda_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    noda_matrix[i, j] = 1.0 / (np.pi * (j - i))
        
        async_spectrum = np.dot(centered_spectra.T, np.dot(centered_spectra, noda_matrix.T)) / (len(all_spectra) - 1)
        
        # 设置样式
        ResearchService.setup_publication_style()
        matplotlib.rcParams['font.size'] = 12
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
        
        # 同步谱
        im1 = ax1.contourf(wavelengths, wavelengths, sync_spectrum,
                          levels=20, cmap='RdYlBu_r')
        ax1.contour(wavelengths, wavelengths, sync_spectrum,
                   levels=20, colors='black', linewidths=0.5, alpha=0.3)
        ax1.set_xlabel('Wavelength (nm)', fontsize=14)
        ax1.set_ylabel('Wavelength (nm)', fontsize=14)
        ax1.set_title('(A) Synchronous Spectrum', fontsize=14, fontweight='bold')
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.ax.tick_params(labelsize=10)
        
        # 异步谱
        im2 = ax2.contourf(wavelengths, wavelengths, async_spectrum,
                          levels=20, cmap='RdYlBu_r')
        ax2.contour(wavelengths, wavelengths, async_spectrum,
                   levels=20, colors='black', linewidths=0.5, alpha=0.3)
        ax2.set_xlabel('Wavelength (nm)', fontsize=14)
        ax2.set_ylabel('Wavelength (nm)', fontsize=14)
        ax2.set_title('(B) Asynchronous Spectrum', fontsize=14, fontweight='bold')
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.ax.tick_params(labelsize=10)
        
        plt.tight_layout()
        
        # 保存
        output_path = FileHelper.ensure_output_dir(output_dir, "research_export")
        img_path = output_path / "2d_correlation_spectrum.png"
        
        img_config = Config.get_export_config('image')
        plt.savefig(img_path, **img_config)
        plt.close()
        
        logger.info(f"2D相关光谱图已保存: {img_path}")
        return img_path
    
    @staticmethod
    @handle_errors(title="导出光谱数据表")
    def export_spectrum_table(viewer, output_dir: Path) -> Path:
        """
        导出光谱数据CSV表
        
        Args:
            viewer: 主视图对象
            output_dir: 输出目录
        
        Returns:
            Path: 输出文件路径
        """
        logger.info("开始导出光谱数据表")
        
        # 提取光谱
        all_spectra_dict = SpectrumExtractor.extract_all_grains_spectra(
            viewer.cube, viewer.labels, viewer.regions, viewer.wavelengths
        )
        
        if not all_spectra_dict:
            raise ValueError("无法提取有效的籽粒光谱数据")
        
        # 保存
        output_path = FileHelper.ensure_output_dir(output_dir, "research_export")
        csv_path = output_path / "spectrum_data_table.csv"
        
        FileHelper.save_spectrum_csv(
            all_spectra_dict,
            viewer.wavelengths,
            csv_path,
            include_stats=True
        )
        
        logger.info(f"光谱数据表已保存: {csv_path}")
        return csv_path

