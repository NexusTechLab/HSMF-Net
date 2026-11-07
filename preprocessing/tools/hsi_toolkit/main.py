"""
HSI Toolkit - 高光谱图像分析工具 (重构版)
主窗口 - 使用服务层架构
"""
import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QStatusBar, QLabel, 
    QMessageBox, QFileDialog, QInputDialog, QDialog, 
    QVBoxLayout, QHBoxLayout, QSpinBox, QPushButton
)
from PySide6.QtGui import QAction, QIcon
from PySide6.QtCore import Qt
from pathlib import Path

# 导入配置
from config import Config

# 导入工具
from utils import setup_logger, get_logger

# 导入服务
from services import ExportService, ResearchService, AnalysisService, PreprocessService

# 导入UI组件
from ui.main_viewer import MainViewer
from ui.seed_export_dialog import SeedExportDialog
from utils.export_settings import axis_band_range, grain_filename, read_axis_limits
from utils.preprocessing import selected_preprocess_steps

# 设置资源目录
ASSET_DIR = Path(__file__).resolve().parent / "assets"

# 设置日志
logger = setup_logger()


def load_styles(app):
    """加载应用样式"""
    qss_path = ASSET_DIR / "styles.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding='utf-8'))


class MainWindow(QMainWindow):
    """主窗口 - 重构后的版本"""
    
    def __init__(self):
        super().__init__()
        logger.info("初始化主窗口")
        
        # 设置窗口
        self.setWindowTitle(Config.UI['window_title'])
        self.resize(*Config.UI['window_size'])
        
        # 主视图
        self.viewer = MainViewer(self)
        self.setCentralWidget(self.viewer)

        # 创建状态栏
        self._create_status_bar()
        
        # 创建菜单
        self._create_menus()
        
        # 信号连接
        self._connect_signals()
        
        logger.info("主窗口初始化完成")
    
    def _create_status_bar(self):
        """创建状态栏"""
        status = QStatusBar()
        self.setStatusBar(status)
        
        self.label_band = QLabel("🟢 波段: -")
        self.label_pos = QLabel("🧭 坐标: -")
        self.label_value = QLabel("💡 反射率: -")
        
        # 设置状态栏标签样式
        for lbl in [self.label_band, self.label_pos, self.label_value]:
            lbl.setStyleSheet("padding: 2px 8px; font-weight: 500;")
        
        status.addPermanentWidget(self.label_band)
        sep1 = QLabel(" ┃ ")
        sep1.setStyleSheet("color: #ccc;")
        status.addPermanentWidget(sep1)
        status.addPermanentWidget(self.label_pos)
        sep2 = QLabel(" ┃ ")
        sep2.setStyleSheet("color: #ccc;")
        status.addPermanentWidget(sep2)
        status.addPermanentWidget(self.label_value)

    def _connect_signals(self):
        """连接信号"""
        self.viewer.control_panel.band_changed.connect(
            lambda idx: self.label_band.setText(f"🟢 波段: {idx}"))
        self.viewer.pixel_info.connect(self._update_status)

    def _update_status(self, r, c, v):
        """更新状态栏"""
        self.label_pos.setText(f"🧭 坐标: ({r}, {c})")
        self.label_value.setText(f"💡 反射率: {v:.4f}")
    
    def _create_menus(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # ========== 🗂 文件菜单 ==========
        file_menu = menubar.addMenu("🗂 文件")
        
        # 打开数据
        open_action = QAction("📁 批量打开高光谱数据...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.viewer._batch_open_files)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        # 导出菜单
        export_menu = file_menu.addMenu("💾 导出数据")
        export_menu.addAction("CSV格式", lambda: self.export_roi_spectrum("csv"))
        export_menu.addAction("MAT格式 (MATLAB)", lambda: self.export_roi_spectrum("mat"))
        export_menu.addAction("NPY格式 (NumPy)", lambda: self.export_roi_spectrum("npy"))
        
        file_menu.addAction("🖼 导出分割掩膜", self.export_mask)
        file_menu.addAction("📊 导出完整报告", self.export_report)
        file_menu.addSeparator()
        file_menu.addAction("🌾 批量保存种子ENVI", self.export_seeds_envi)
        file_menu.addSeparator()
        file_menu.addAction("🔬 分析籽粒健康等级", self.analyze_grain_health)
        file_menu.addAction("🔍 数据质量诊断", self.diagnose_data_quality)
        file_menu.addSeparator()
        file_menu.addAction("🚪 退出", self.close)
        
        # ========== ⚙️ 预处理菜单 ==========
        preproc_menu = menubar.addMenu("⚙️ 预处理")
        preproc_menu.addAction("✂️ 波段截取", self.crop_bands)
        preproc_menu.addSeparator()
        
        smooth_menu = preproc_menu.addMenu("🌀 光谱平滑")
        smooth_menu.addAction("移动平均 (MA)", lambda: self.smooth_spectrum("MA"))
        smooth_menu.addAction("Savitzky-Golay (SG)", lambda: self.smooth_spectrum("SG"))
        
        deriv_menu = preproc_menu.addMenu("📈 求导")
        deriv_menu.addAction("一阶导数 [会产生负值]", lambda: self.derivative_spectrum(1))
        deriv_menu.addAction("二阶导数 [会产生负值]", lambda: self.derivative_spectrum(2))
        
        preproc_menu.addSeparator()
        preproc_menu.addAction("📊 归一化 (0-1)", self.normalize_data)
        preproc_menu.addAction("📐 标准化 (Z-score) [会产生负值]", self.standardize_data)
        
        # ========== 🌾 提取菜单 ==========
        segment_menu = menubar.addMenu("🌾 提取")
        segment_menu.addAction("🔲 Otsu阈值分割", lambda: self._quick_segment("otsu"))
        segment_menu.addAction("💧 Watershed分割", lambda: self._quick_segment("watershed"))
        
        # ========== 🔬 科研菜单 ==========
        research_menu = menubar.addMenu("🔬 科研")
        research_menu.addAction("📈 导出平均光谱图 (Publication)", self.export_publication_spectrum)
        research_menu.addAction("📊 导出光谱数据表 (CSV)", self.export_spectrum_table)
        research_menu.addSeparator()
        research_menu.addAction("📉 导出导数光谱图 (Derivative)", self.export_derivative_spectrum)
        research_menu.addAction("📊 导出分组对比图 (Group Compare)", self.export_group_comparison)
        research_menu.addAction("🔥 导出2D相关光谱图 (2D-COS)", self.export_2d_correlation)
        research_menu.addSeparator()
        research_menu.addAction("🖼 导出真彩+假彩图 (B + NIR-R-G)", self.export_true_false_color_images)
        
        # ========== ❓ 帮助菜单 ==========
        help_menu = menubar.addMenu("❓ 帮助")
        help_menu.addAction("ℹ️ 关于", self.show_about)

    # ==================== 文件操作 ====================
    
    def export_roi_spectrum(self, format_type: str):
        """导出ROI光谱"""
        logger.info(f"用户请求导出ROI光谱: format={format_type}")
        
        if self.viewer.cube is None:
            QMessageBox.warning(self, "提示", "请先打开数据文件")
            return
        
        if len(self.viewer.roi_list) == 0:
            QMessageBox.warning(self, "提示", "请先绘制 ROI 区域")
            return
        
        # 选择输出文件
        filters = {
            'csv': "CSV文件 (*.csv)",
            'mat': "MAT文件 (*.mat)",
            'npy': "NPY文件 (*.npy)"
        }
        
        output_path, _ = QFileDialog.getSaveFileName(
            self, "保存光谱数据", "", filters.get(format_type, "所有文件 (*.*)")
        )
        
        if not output_path:
            return
        
        try:
            ExportService.export_roi_spectrum(
                self.viewer, format_type, Path(output_path)
            )
            
            QMessageBox.information(
                self, "导出成功",
                f"ROI光谱已保存到:\n{output_path}\n\n"
                f"已导出 {len(self.viewer.roi_list)} 个ROI"
            )
            logger.info(f"ROI光谱导出成功: {output_path}")
        
        except Exception as e:
            logger.error(f"ROI导出失败: {e}")
            QMessageBox.critical(self, "错误", f"导出失败:\n{e}")
    
    def export_mask(self):
        """导出分割掩膜"""
        logger.info("用户请求导出分割掩膜")
        
        if self.viewer.labels is None:
            QMessageBox.warning(self, "提示", "请先执行分割操作")
            return
        
        output_path, _ = QFileDialog.getSaveFileName(
            self, "保存分割掩膜", "", "PNG图像 (*.png)"
        )
        
        if not output_path:
            return
        
        try:
            ExportService.export_segmentation_mask(
                self.viewer.labels, Path(output_path)
            )
            
            QMessageBox.information(
                self, "导出成功", f"分割掩膜已保存到:\n{output_path}"
            )
            logger.info(f"掩膜导出成功: {output_path}")
        
        except Exception as e:
            logger.error(f"掩膜导出失败: {e}")
            QMessageBox.critical(self, "错误", f"导出失败:\n{e}")
    
    def export_report(self):
        """导出完整报告"""
        if self.viewer.cube is None:
            QMessageBox.warning(self, "提示", "请先打开数据文件")
            return
        
        path, _ = QFileDialog.getSaveFileName(self, "导出报告", "", "HTML (*.html)")
        if not path:
            return
        
        # 生成HTML报告
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>HSI分析报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #0969da; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f0f4f8; }}
    </style>
</head>
<body>
    <h1>高光谱图像分析报告</h1>
    <h2>数据信息</h2>
    <table>
        <tr><th>维度</th><td>{self.viewer.cube.shape}</td></tr>
        <tr><th>波段数</th><td>{self.viewer.cube.shape[2]}</td></tr>
        <tr><th>值域</th><td>[{self.viewer.cube.min():.4f}, {self.viewer.cube.max():.4f}]</td></tr>
    </table>
    <h2>分割结果</h2>
    <p>检测到颗粒数: {len(self.viewer.regions) if self.viewer.regions else 0}</p>
</body>
</html>"""
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        QMessageBox.information(self, "成功", f"报告已导出至:\n{path}")
    
    def export_seeds_envi(self):
        """批量导出籽粒ENVI文件"""
        logger.info("用户请求批量导出ENVI")
        
        if self.viewer.cube is None:
            QMessageBox.warning(self, "提示", "请先打开数据文件")
            return
        
        if self.viewer.labels is None or not self.viewer.regions:
            QMessageBox.warning(self, "提示", "请先执行分割操作")
            return
        
        # 光谱图范围仅作为快捷选项，导出范围始终由设置窗口明确选择。
        plot_band_range = None
        if hasattr(self.viewer, 'plot') and self.viewer.plot is not None:
            try:
                view_range = self.viewer.plot.plotItem.vb.viewRange()
                x_min, x_max = view_range[0]
                
                if self.viewer.wavelengths is not None and len(self.viewer.wavelengths) > 0:
                    wavelengths = np.array(self.viewer.wavelengths, dtype=np.float64)
                else:
                    wavelengths = np.arange(self.viewer.cube.shape[2])
                visible = np.flatnonzero((wavelengths >= x_min) & (wavelengths <= x_max))
                if visible.size:
                    plot_band_range = (int(visible[0]), int(visible[-1]) + 1)
            except Exception as e:
                logger.warning(f"无法获取光谱范围: {e}")
        
        dialog = SeedExportDialog(
            self.viewer.cube.shape[2], len(self.viewer.regions),
            wavelengths=self.viewer.wavelengths,
            plot_band_range=plot_band_range, parent=self,
            filename_prefix=getattr(self, '_export_filename_prefix', ''),
        )
        if dialog.exec() != QDialog.Accepted:
            return
        options = dialog.export_options()
        self._export_filename_prefix = options['filename_prefix']
        steps = selected_preprocess_steps(self.viewer)
        axis_text = []
        for axis in ('x', 'y'):
            values = []
            for bound in ('min', 'max'):
                edit = getattr(self.viewer, f'edit_{axis}{bound}', None)
                values.append((edit.text().strip() if edit is not None else '') or '自动')
            axis_text.append(f"{axis.upper()} 轴：{values[0]} ～ {values[1]}")
        reply = QMessageBox.question(
            self, "是否按当前坐标轴与预处理设置导出？",
            "\n".join(axis_text) + "\n预处理：" + (' → '.join(steps) or '无')
            + "\n\n【是】按 X 轴范围选择导出波段，并追加上述预处理；"
              "PNG 光谱图使用上述 X/Y 坐标轴范围。"
              "\n【否】使用上一步选择的波段，不追加工具栏预处理，PNG 坐标轴自动缩放。"
              "\nY 轴范围只控制图片显示，不截断或修改光谱数值。",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes,
        )
        if reply == QMessageBox.Cancel:
            return
        options['apply_preprocess'] = reply == QMessageBox.Yes
        if reply == QMessageBox.Yes:
            try:
                options['axis_limits'] = read_axis_limits(self.viewer)
                options['band_range'] = axis_band_range(
                    self.viewer.cube, self.viewer.wavelengths, options['axis_limits']['x']
                )
            except ValueError as error:
                QMessageBox.warning(self, "导出范围无效", str(error))
                return
        output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if not output_dir:
            return
        
        try:
            # 调用服务层
            stats = ExportService.export_seeds_envi(
                self.viewer,
                Path(output_dir),
                **options
            )
            
            # 显示统计信息
            band_start, band_end = stats['band_range']
            if self.viewer.wavelengths is not None and len(self.viewer.wavelengths):
                wavelengths = np.array(self.viewer.wavelengths)
                # 确保波长是浮点数（可能是字符串类型）
                wl_start = float(wavelengths[band_start])
                wl_end = float(wavelengths[band_end-1])
                range_info = (
                    f"波段 {band_start} ({wl_start:.2f} nm) - "
                    f"波段 {band_end-1} ({wl_end:.2f} nm)\n"
                    f"共 {band_end - band_start} 个波段"
                )
            else:
                range_info = f"波段 {band_start} - 波段 {band_end-1}\n共 {band_end - band_start} 个波段"
            
            # 获取预处理信息
            preprocess_steps = stats.get('preprocess_steps', [])
            if preprocess_steps:
                preprocess_info = "、".join(preprocess_steps)
            else:
                preprocess_info = "不追加工具栏预处理"
            
            first_name = grain_filename(
                options['start_number'], options['number_digits'], options['filename_prefix']
            )
            file_info = f"  • ENVI数据: {first_name}, {first_name}.HDR, ...\n"
            if options['save_images']:
                file_info += f"  • 籽粒图像: {first_name}.png, ...\n"
            if options['save_csv']:
                file_info += "  • 光谱数据: all_grains_spectra.csv"

            QMessageBox.information(
                self, "导出成功",
                f"✅ 成功导出 {stats['success']}/{stats['total']} 个籽粒\n\n"
                f"📁 输出目录:\n{stats['output_dir']}\n\n"
                f"📊 导出范围:\n{range_info}\n\n"
                f"🔧 数据处理:\n{preprocess_info}\n\n"
                f"📁 导出文件:\n"
                f"{file_info}"
            )
            logger.info(f"ENVI批量导出完成: {stats}")
        
        except Exception as e:
            logger.error(f"ENVI导出失败: {e}")
            QMessageBox.critical(self, "错误", f"导出失败:\n{e}")
    
    # ==================== 分析功能 ====================
    
    def analyze_grain_health(self):
        """分析籽粒健康等级"""
        logger.info("用户请求分析籽粒健康等级")
        
        csv_path, _ = QFileDialog.getOpenFileName(
            self, "选择光谱CSV文件", "", "CSV Files (*.csv)"
        )
        if not csv_path:
            return
        
        try:
            output_path, stats = AnalysisService.analyze_grain_health(
                Path(csv_path)
            )
            
            # 显示结果摘要
            grade_counts = stats['grade_counts']
            summary = f"健康等级分析完成！\n\n"
            summary += f"总籽粒数: {stats['total']}\n\n"
            summary += "各等级统计:\n"
            for grade in ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3']:
                count = grade_counts.get(grade, 0)
                desc = {'Grade 0': '健康', 'Grade 1': '轻微霉变',
                       'Grade 2': '中度霉变', 'Grade 3': '重度霉变'}[grade]
                summary += f"  {grade} ({desc}): {count} 颗\n"
            
            summary += f"\n结果已保存至:\n{output_path}"
            
            QMessageBox.information(self, "分析完成", summary)
        
        except Exception as e:
            logger.error(f"健康等级分析失败: {e}")
            QMessageBox.critical(self, "错误", f"分析失败:\n{e}")
    
    def diagnose_data_quality(self):
        """数据质量诊断"""
        logger.info("用户请求数据质量诊断")
        
        csv_path, _ = QFileDialog.getOpenFileName(
            self, "选择光谱CSV文件", "", "CSV Files (*.csv)"
        )
        if not csv_path:
            return
        
        try:
            diagnosis_csv, report_txt, chart_png, stats = AnalysisService.diagnose_data_quality(
                Path(csv_path)
            )
            
            # 显示摘要
            summary = f"数据质量诊断完成！\n\n"
            summary += f"总籽粒数: {stats['total']}\n"
            summary += f"异常籽粒数: {stats['abnormal']}\n"
            summary += f"健康率: {stats['health_rate']:.1f}%\n\n"
            
            if stats['abnormal'] > 0:
                summary += "⚠️ 检测到异常数据！\n\n"
                summary += "可能原因:\n"
                summary += "1. 分割掩膜包含背景像素\n"
                summary += "2. 光照或校正问题\n"
                summary += "3. 数据预处理不当\n\n"
            else:
                summary += "✅ 所有数据符合健康标准！\n\n"
            
            summary += f"详细报告已保存至:\n"
            summary += f"- {Path(diagnosis_csv).name}\n"
            summary += f"- {Path(report_txt).name}\n"
            summary += f"- {Path(chart_png).name}"
            
            QMessageBox.information(self, "诊断完成", summary)
        
        except Exception as e:
            logger.error(f"数据质量诊断失败: {e}")
            QMessageBox.critical(self, "错误", f"诊断失败:\n{e}")
    
    # ==================== 预处理功能 ====================
    
    def crop_bands(self):
        """波段截取"""
        if self.viewer.cube is None:
            QMessageBox.warning(self, "提示", "请先打开数据文件")
            return
        
        # 创建对话框
        dlg = QDialog(self)
        dlg.setWindowTitle("波段截取")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel(f"当前波段数: {self.viewer.cube.shape[2]}"))
        
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("起始:"))
        spin_start = QSpinBox()
        spin_start.setMaximum(self.viewer.cube.shape[2]-1)
        h1.addWidget(spin_start)
        layout.addLayout(h1)
        
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("结束:"))
        spin_end = QSpinBox()
        spin_end.setMaximum(self.viewer.cube.shape[2])
        spin_end.setValue(self.viewer.cube.shape[2])
        h2.addWidget(spin_end)
        layout.addLayout(h2)
        
        btn = QPushButton("确定")
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)
        
        if dlg.exec():
            try:
                start = spin_start.value()
                end = spin_end.value()
                
                # 使用服务层
                self.viewer.cube, self.viewer.wavelengths = PreprocessService.crop_bands(
                    self.viewer.cube, self.viewer.wavelengths, start, end
                )
                
                # 更新UI
                bands = self.viewer.cube.shape[2]
                self.viewer.control_panel.set_band_range(bands, self.viewer.wavelengths)
                self.viewer.update_image()
                
                QMessageBox.information(
                    self, "完成", f"已截取波段 {start}-{end-1}\n当前波段数: {bands}"
                )
                logger.info(f"波段截取完成: {start}-{end-1}")
            
            except Exception as e:
                logger.error(f"波段截取失败: {e}")
                QMessageBox.critical(self, "错误", f"截取失败:\n{e}")
    
    def smooth_spectrum(self, method: str):
        """光谱平滑"""
        if self.viewer.cube is None:
            QMessageBox.warning(self, "提示", "请先打开数据文件")
            return
        
        try:
            # 使用服务层
            self.viewer.cube = PreprocessService.smooth_spectrum(
                self.viewer.cube, method
            )
            
            # 更新UI
            self.viewer.update_image()
            
            msg = f"{'移动平均' if method == 'MA' else 'Savitzky-Golay'}平滑已完成"
            QMessageBox.information(self, "完成", msg)
            logger.info(f"光谱平滑完成: method={method}")
        
        except Exception as e:
            logger.error(f"光谱平滑失败: {e}")
            QMessageBox.critical(self, "错误", f"平滑失败:\n{e}")
    
    def derivative_spectrum(self, order: int):
        """光谱求导"""
        if self.viewer.cube is None:
            QMessageBox.warning(self, "提示", "请先打开数据文件")
            return
        
        try:
            # 使用服务层
            self.viewer.cube, self.viewer.wavelengths = PreprocessService.derivative_spectrum(
                self.viewer.cube, self.viewer.wavelengths, order
            )
            
            # 更新UI
            bands = self.viewer.cube.shape[2]
            self.viewer.control_panel.set_band_range(bands, self.viewer.wavelengths)
            self.viewer.update_image()
            
            QMessageBox.information(
                self, "完成", f"{order}阶导数已完成\n当前波段数: {bands}"
            )
            logger.info(f"光谱求导完成: order={order}")
        
        except Exception as e:
            logger.error(f"光谱求导失败: {e}")
            QMessageBox.critical(self, "错误", f"求导失败:\n{e}")
    
    def normalize_data(self):
        """数据归一化"""
        if self.viewer.cube is None:
            QMessageBox.warning(self, "提示", "请先打开数据文件")
            return
        
        try:
            # 使用服务层
            self.viewer.cube = PreprocessService.normalize_data(self.viewer.cube)
            
            # 更新UI
            self.viewer.update_image()
            
            QMessageBox.information(self, "完成", "数据归一化已完成 (0-1)")
            logger.info("数据归一化完成")
        
        except Exception as e:
            logger.error(f"数据归一化失败: {e}")
            QMessageBox.critical(self, "错误", f"归一化失败:\n{e}")
    
    def standardize_data(self):
        """数据标准化"""
        if self.viewer.cube is None:
            QMessageBox.warning(self, "提示", "请先打开数据文件")
            return
        
        try:
            # 使用服务层
            self.viewer.cube = PreprocessService.standardize_data(self.viewer.cube)
            
            # 更新UI
            self.viewer.update_image()
            
            QMessageBox.information(self, "完成", "数据标准化已完成 (Z-score)")
            logger.info("数据标准化完成")
        
        except Exception as e:
            logger.error(f"数据标准化失败: {e}")
            QMessageBox.critical(self, "错误", f"标准化失败:\n{e}")
    
    # ==================== 分割功能 ====================
    
    def _quick_segment(self, method: str):
        """快速分割"""
        if self.viewer.cube is None:
            QMessageBox.warning(self, "提示", "请先打开数据文件")
            return
        
        self.viewer.control_panel.combo_method.setCurrentText(
            "Otsu阈值" if method == "otsu" else "Watershed")
        self.viewer._on_segment_requested()
    
    # ==================== 科研导出 ====================
    
    def export_publication_spectrum(self):
        """导出论文级光谱图"""
        logger.info("用户请求导出论文级光谱图")
        
        if self.viewer.cube is None or self.viewer.labels is None or not self.viewer.regions:
            QMessageBox.warning(self, "提示", "请先加载数据并完成分割")
            return
        
        output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if not output_dir:
            return
        
        try:
            img_path = ResearchService.export_publication_spectrum(
                self.viewer, Path(output_dir)
            )
            
            # 统计信息
            n_grains = len(self.viewer.regions)
            wl_range = (self.viewer.wavelengths[0], self.viewer.wavelengths[-1]) if self.viewer.wavelengths else (0, self.viewer.cube.shape[2])
            
            QMessageBox.information(
                self, "导出成功",
                f"✅ 论文级光谱图已保存:\n{img_path}\n\n"
                f"📊 已绘制 {n_grains} 条籽粒光谱曲线\n"
                f"📏 波长范围: {wl_range[0]:.1f} - {wl_range[1]:.1f} nm\n"
                f"🎨 分辨率: 300 DPI"
            )
        
        except Exception as e:
            logger.error(f"论文光谱图导出失败: {e}")
            QMessageBox.critical(self, "错误", f"导出失败:\n{e}")
    
    def export_spectrum_table(self):
        """导出光谱数据表"""
        logger.info("用户请求导出光谱数据表")
        
        if self.viewer.cube is None or self.viewer.labels is None or not self.viewer.regions:
            QMessageBox.warning(self, "提示", "请先加载数据并完成分割")
            return
        
        output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if not output_dir:
            return
        
        try:
            csv_path = ResearchService.export_spectrum_table(
                self.viewer, Path(output_dir)
            )
            
            n_grains = len(self.viewer.regions)
            n_bands = self.viewer.cube.shape[2]
            
            QMessageBox.information(
                self, "导出成功",
                f"✅ 光谱数据表已保存:\n{csv_path}\n\n"
                f"📊 包含 {n_grains} 个籽粒的统计数据\n"
                f"📏 波段数: {n_bands}\n"
                f"📈 字段: 波长、平均值、标准差、样本数"
            )
        
        except Exception as e:
            logger.error(f"光谱数据表导出失败: {e}")
            QMessageBox.critical(self, "错误", f"导出失败:\n{e}")
    
    def export_derivative_spectrum(self):
        """导出导数光谱图"""
        logger.info("用户请求导出导数光谱图")
        
        if self.viewer.cube is None or self.viewer.labels is None or not self.viewer.regions:
            QMessageBox.warning(self, "提示", "请先加载数据并完成分割")
            return
        
        output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if not output_dir:
            return
        
        try:
            img_path = ResearchService.export_derivative_spectrum(
                self.viewer, Path(output_dir)
            )
            
            n_grains = len(self.viewer.regions)
            wl_range = (self.viewer.wavelengths[0], self.viewer.wavelengths[-1]) if self.viewer.wavelengths else (0, self.viewer.cube.shape[2])
            
            QMessageBox.information(
                self, "导出成功",
                f"✅ 导数光谱图已保存:\n{img_path}\n\n"
                f"📊 已绘制 {n_grains} 条籽粒光谱的导数\n"
                f"📏 波长范围: {wl_range[0]:.1f} - {wl_range[1]:.1f} nm\n"
                f"🎨 分辨率: 300 DPI"
            )
        
        except Exception as e:
            logger.error(f"导数光谱图导出失败: {e}")
            QMessageBox.critical(self, "错误", f"导出失败:\n{e}")
    
    def export_group_comparison(self):
        """导出分组对比图"""
        logger.info("用户请求导出分组对比图")
        
        if self.viewer.cube is None or self.viewer.labels is None or not self.viewer.regions:
            QMessageBox.warning(self, "提示", "请先加载数据并完成分割")
            return
        
        # 询问分组数量
        num_groups, ok = QInputDialog.getInt(
            self, "设置分组数量", 
            "请输入分组数量（例如：4个健康等级）:",
            4, 2, 10, 1
        )
        
        if not ok:
            return
        
        output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if not output_dir:
            return
        
        try:
            img_path = ResearchService.export_group_comparison(
                self.viewer, Path(output_dir), num_groups=num_groups
            )
            
            n_grains = len(self.viewer.regions)
            wl_range = (self.viewer.wavelengths[0], self.viewer.wavelengths[-1]) if self.viewer.wavelengths else (0, self.viewer.cube.shape[2])
            
            QMessageBox.information(
                self, "导出成功",
                f"✅ 分组对比图已保存:\n{img_path}\n\n"
                f"📊 已将 {n_grains} 个籽粒分为 {num_groups} 组\n"
                f"📏 波长范围: {wl_range[0]:.1f} - {wl_range[1]:.1f} nm\n"
                f"🎨 分辨率: 300 DPI"
            )
        
        except Exception as e:
            logger.error(f"分组对比图导出失败: {e}")
            QMessageBox.critical(self, "错误", f"导出失败:\n{e}")
    
    def export_2d_correlation(self):
        """导出2D相关光谱图"""
        logger.info("用户请求导出2D相关光谱图")
        
        if self.viewer.cube is None or self.viewer.labels is None or not self.viewer.regions:
            QMessageBox.warning(self, "提示", "请先加载数据并完成分割")
            return
        
        output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if not output_dir:
            return
        
        try:
            img_path = ResearchService.export_2d_correlation(
                self.viewer, Path(output_dir)
            )
            
            n_grains = len(self.viewer.regions)
            wl_range = (self.viewer.wavelengths[0], self.viewer.wavelengths[-1]) if self.viewer.wavelengths else (0, self.viewer.cube.shape[2])
            
            QMessageBox.information(
                self, "导出成功",
                f"✅ 2D相关光谱图已保存:\n{img_path}\n\n"
                f"📊 基于 {n_grains} 个籽粒光谱计算\n"
                f"📏 波长范围: {wl_range[0]:.1f} - {wl_range[1]:.1f} nm\n"
                f"🔥 包含同步谱和异步谱\n"
                f"🎨 分辨率: 300 DPI"
            )
        
        except Exception as e:
            logger.error(f"2D相关光谱图导出失败: {e}")
            QMessageBox.critical(self, "错误", f"导出失败:\n{e}")

    def export_true_false_color_images(self):
        """导出真彩色(B)与假彩色(NIR-R-G)图像"""
        logger.info("用户请求导出真彩色和假彩色图像")

        if self.viewer.cube is None:
            QMessageBox.warning(self, "提示", "请先加载高光谱数据")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if not output_dir:
            return

        try:
            result = self.viewer.export_true_false_color_images(Path(output_dir))
            QMessageBox.information(
                self, "导出成功",
                f"✅ 已生成两套图像：\n\n"
                f"1) 真彩色(B, CMF积分近似)\n{result['true_color_path']}\n\n"
                f"2) 假彩色(NIR-R-G)\n{result['false_color_path']}"
            )
            logger.info(f"真彩/假彩图导出成功: {result}")
        except Exception as e:
            logger.error(f"真彩/假彩图导出失败: {e}")
            QMessageBox.critical(self, "错误", f"导出失败:\n{e}")
    
    # ==================== 帮助 ====================
    
    def show_about(self):
        """关于对话框"""
        QMessageBox.information(
            self, "关于",
            f"{Config.APP_NAME} v{Config.APP_VERSION}\n\n"
            f"高光谱图像分析工具\n\n"
            f"功能：数据加载、预处理、批量分割、批量导出、光谱分析\n\n"
            f"作者：想暴富的小许--13505211023"
        )


def main():
    """主函数"""
    # 确保必要的目录存在
    Config.ensure_dirs()
    
    # 记录启动
    logger.info("="*50)
    logger.info(f"启动 {Config.APP_NAME} v{Config.APP_VERSION}")
    logger.info("="*50)
    
    # 创建应用
    app = QApplication(sys.argv)
    
    # 加载样式
    load_styles(app)
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 运行
    logger.info("应用程序已启动")
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

