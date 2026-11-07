"""
分割流程可视化对话框
展示分割的6个步骤
符合论文图表黄金标准：Arial字体、600 DPI、支持导出TIFF/PDF/EPS
"""
import numpy as np
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox, QFileDialog
from PySide6.QtCore import Qt
import matplotlib
matplotlib.use("QtAgg")
# 设置符合论文要求的字体和参数
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.labelsize'] = 10  # 坐标轴标签 9-10 pt
matplotlib.rcParams['xtick.labelsize'] = 8  # 刻度 7-8 pt
matplotlib.rcParams['ytick.labelsize'] = 8
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['figure.dpi'] = 100  # 显示用
matplotlib.rcParams['savefig.dpi'] = 600  # 保存用 600 DPI
matplotlib.rcParams['savefig.format'] = 'pdf'  # 默认矢量格式
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType 字体嵌入（避免转换为位图）
matplotlib.rcParams['ps.fonttype'] = 42   # PostScript TrueType
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class SegmentationViewer(QDialog):
    """分割流程可视化窗口"""
    
    def __init__(self, labels_final, info, band_idx=60, parent=None):
        super().__init__(parent)
        self.setWindowTitle("分割流程可视化")
        self.resize(1600, 800)  # 稍微减小高度
        
        # 检查是否分割成功
        if labels_final is None or (isinstance(labels_final, np.ndarray) and labels_final.max() == 0):
            from PySide6.QtWidgets import QLabel, QMessageBox
            QMessageBox.warning(self, "分割失败", "未检测到任何区域，请调整分割参数（阈值、最小面积等）后重试")
            self.close()
            return
        
        # 创建matplotlib图形 - 调整比例使布局更紧凑
        self.fig = Figure(figsize=(18, 8), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        
        # 添加导航工具栏
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # 添加导出按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.btn_export_png = QPushButton("导出 PNG (300 DPI)")
        self.btn_export_png.setToolTip("导出 PNG 图像，方便复制和快速查看")
        self.btn_export_png.clicked.connect(self._export_png)
        button_layout.addWidget(self.btn_export_png)
        
        self.btn_export_tiff = QPushButton("导出 TIFF (600 DPI)")
        self.btn_export_tiff.setToolTip("导出高分辨率 TIFF 图像，适用于照片和位图")
        self.btn_export_tiff.clicked.connect(self._export_tiff)
        button_layout.addWidget(self.btn_export_tiff)
        
        self.btn_export_pdf = QPushButton("导出 PDF (矢量)")
        self.btn_export_pdf.setToolTip("导出矢量 PDF 图像，适用于图表，可无损缩放")
        self.btn_export_pdf.clicked.connect(self._export_pdf)
        button_layout.addWidget(self.btn_export_pdf)
        
        self.btn_export_eps = QPushButton("导出 EPS (矢量)")
        self.btn_export_eps.setToolTip("导出矢量 EPS 图像，适用于某些期刊要求")
        self.btn_export_eps.clicked.connect(self._export_eps)
        button_layout.addWidget(self.btn_export_eps)
        
        button_layout.addStretch()
        
        # 布局
        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addLayout(button_layout)
        layout.addWidget(self.canvas)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 绘制6个步骤
        self._plot_steps(labels_final, info, band_idx)
        
        self.canvas.draw()
    
    def _plot_steps(self, labels_final, info, band_idx):
        """绘制6个步骤"""
        
        # 使用GridSpec确保布局对齐 - 更紧凑的布局
        from matplotlib.gridspec import GridSpec
        # 极致紧凑：进一步减小左右间距
        gs = GridSpec(
            2,
            3,
            figure=self.fig,
            hspace=0.12,
            wspace=0.08,  # 极小的列间距
            left=0.03,
            right=0.97,
            top=0.94,
            bottom=0.04,
        )
        
        # 步骤1: 原始图像
        ax1 = self.fig.add_subplot(gs[0, 0])
        img1 = info.get('step1_image', None)
        if img1 is not None:
            im1 = ax1.imshow(img1, cmap='gray', vmin=0, vmax=0.6)
            # 不显示colorbar，保持图像简洁
        ax1.set_title(f'Step 1: Original Band {band_idx}', fontsize=10, fontweight='bold', pad=5)
        ax1.axis('off')
        # 添加子图标签 (符合论文标准)
        ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=10, 
                fontweight='bold', va='top', ha='left', 
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, linewidth=0.5))
        
        # 步骤2: 二值化
        ax2 = self.fig.add_subplot(gs[0, 1])
        binary = info.get('step2_binary', None)
        threshold = info.get('threshold', 0)
        if binary is not None:
            ax2.imshow(binary, cmap='gray')
        ax2.set_title(f'Step 2: Binarization (threshold={threshold:.2f})', fontsize=10, fontweight='bold', pad=5)
        ax2.axis('off')
        ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=10, 
                fontweight='bold', va='top', ha='left', 
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, linewidth=0.5))
        
        # 步骤3: 形态学处理
        ax3 = self.fig.add_subplot(gs[0, 2])
        cleaned = info.get('step3_cleaned', None)
        if cleaned is not None:
            ax3.imshow(cleaned, cmap='gray')
        ax3.set_title('Step 3: Morphological Processing', fontsize=10, fontweight='bold', pad=5)
        ax3.axis('off')
        ax3.text(0.02, 0.98, '(c)', transform=ax3.transAxes, fontsize=10, 
                fontweight='bold', va='top', ha='left', 
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, linewidth=0.5))
        
        # 步骤4: 检测中心点
        ax4 = self.fig.add_subplot(gs[1, 0])
        coords = info.get('step4_coords', None)
        n_centers = info.get('seed_centers', 0)
        if n_centers == 0:
            # 若未检测到中心点，回退到连通区域数
            n_centers = max(info.get('initial_regions', 0)-1, 0)

        # 若中心点数量大于最终有效籽粒数，则使用最终数，避免 >100
        final_cnt = info.get('final_regions', 0)
        if final_cnt > 0 and n_centers > final_cnt:
            n_centers = final_cnt
        if binary is not None:
            ax4.imshow(binary, cmap='gray')
        if coords is not None and len(coords) > 0:
            ax4.plot(coords[:, 1], coords[:, 0], 'r.', markersize=4)
        ax4.set_title(f'Step 4: Seed Detection (n={n_centers})', fontsize=10, fontweight='bold', pad=5)
        ax4.axis('off')
        ax4.text(0.02, 0.98, '(d)', transform=ax4.transAxes, fontsize=10, 
                fontweight='bold', va='top', ha='left', 
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, linewidth=0.5))
        
        # 步骤5: 分割结果
        ax5 = self.fig.add_subplot(gs[1, 1])
        labels_init = info.get('step5_labels', None)
        n_init = info.get('initial_regions', 0)
        if labels_init is not None:
            ax5.imshow(labels_init, cmap='nipy_spectral')
        ax5.set_title(f'Step 5: Segmentation ({n_init} regions)', fontsize=10, fontweight='bold', pad=5)
        ax5.axis('off')
        ax5.text(0.02, 0.98, '(e)', transform=ax5.transAxes, fontsize=10, 
                fontweight='bold', va='top', ha='left', 
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, linewidth=0.5))
        
        # 步骤6: 最终结果
        ax6 = self.fig.add_subplot(gs[1, 2])
        n_final = info.get('final_regions', 0)
        if labels_final is not None:
            # 确保使用2D标签
            labels_2d = labels_final[:, :, 0] if labels_final.ndim == 3 else labels_final
            
            # 灰色背景 - 确保正确的形状
            background = np.ones((*labels_2d.shape, 3)) * 0.7
            ax6.imshow(background)
            
            # 叠加标签
            labels_masked = np.ma.masked_where(labels_2d == 0, labels_2d)
            ax6.imshow(labels_masked, cmap='tab20', alpha=0.8)
            
            # 显示编号 - 优化字体大小和样式
            from skimage.measure import regionprops
            regions = regionprops(labels_2d)
            for region in regions:
                y, x = region.centroid
                label = region.label
                ax6.text(x, y, str(label), 
                        color='white', fontsize=8, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='black', alpha=0.7, linewidth=0))
        
        # 转换排序模式为英文
        sort_mode = info.get('sort_mode', '顺序')
        sort_mode_en = {
            '顺序': 'Sequential',
            '从左到右': 'Left-to-Right',
            '从上到下': 'Top-to-Bottom',
            '从右到左': 'Right-to-Left',
            '从下到上': 'Bottom-to-Top'
        }.get(sort_mode, sort_mode)
        
        ax6.set_title(f'Step 6: Final Result ({n_final} seeds) - {sort_mode_en}', fontsize=10, fontweight='bold', pad=5)
        ax6.axis('off')
        ax6.text(0.02, 0.98, '(f)', transform=ax6.transAxes, fontsize=10, 
                fontweight='bold', va='top', ha='left', 
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, linewidth=0.5))
    
    def _export_png(self):
        """导出为 PNG 格式（300 DPI，方便复制使用）"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出 PNG 图像", "", "PNG 图像 (*.png)"
        )
        if file_path:
            try:
                # 确保文件扩展名
                if not file_path.lower().endswith('.png'):
                    file_path += '.png'
                
                # 保存为 PNG（600 DPI，适合屏幕和文档使用）
                self.fig.savefig(
                    file_path,
                    format='png',
                    dpi=600,
                    bbox_inches='tight',
                    pad_inches=0.1,
                    facecolor='white',
                    edgecolor='none'
                )
                QMessageBox.information(self, "导出成功", 
                    f"已成功导出 PNG 图像 (600 DPI):\n{file_path}\n\n"
                    "PNG 格式方便复制粘贴到文档或演示文稿中。")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"导出 PNG 失败：{str(e)}")
    
    def _export_tiff(self):
        """导出为 TIFF 格式（600 DPI，RGB 色彩模式）"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出 TIFF 图像", "", "TIFF 图像 (*.tiff *.tif)"
        )
        if file_path:
            try:
                # 确保文件扩展名
                if not file_path.lower().endswith(('.tiff', '.tif')):
                    file_path += '.tiff'
                
                # 保存为高分辨率 TIFF
                self.fig.savefig(
                    file_path,
                    format='tiff',
                    dpi=600,
                    bbox_inches='tight',
                    pad_inches=0.1,
                    facecolor='white',
                    edgecolor='none'
                )
                QMessageBox.information(self, "导出成功", 
                    f"已成功导出 TIFF 图像 (600 DPI, RGB):\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"导出 TIFF 失败：{str(e)}")
    
    def _export_pdf(self):
        """导出为 PDF 格式（矢量图）"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出 PDF 图像", "", "PDF 文件 (*.pdf)"
        )
        if file_path:
            try:
                # 确保文件扩展名
                if not file_path.lower().endswith('.pdf'):
                    file_path += '.pdf'
                
                # 保存为矢量 PDF
                self.fig.savefig(
                    file_path,
                    format='pdf',
                    dpi=600,
                    bbox_inches='tight',
                    pad_inches=0.1,
                    facecolor='white',
                    edgecolor='none',
                    metadata={
                        'Creator': 'HSI Toolkit',
                        'Title': '分割流程可视化',
                        'Subject': 'Hyperspectral Image Segmentation'
                    }
                )
                QMessageBox.information(self, "导出成功", 
                    f"已成功导出 PDF 矢量图:\n{file_path}\n\n"
                    "PDF 格式支持无损缩放，适合论文插图。")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"导出 PDF 失败：{str(e)}")
    
    def _export_eps(self):
        """导出为 EPS 格式（矢量图）"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出 EPS 图像", "", "EPS 文件 (*.eps)"
        )
        if file_path:
            try:
                # 确保文件扩展名
                if not file_path.lower().endswith('.eps'):
                    file_path += '.eps'
                
                # 保存为矢量 EPS
                self.fig.savefig(
                    file_path,
                    format='eps',
                    dpi=600,
                    bbox_inches='tight',
                    pad_inches=0.1,
                    facecolor='white',
                    edgecolor='none'
                )
                QMessageBox.information(self, "导出成功", 
                    f"已成功导出 EPS 矢量图:\n{file_path}\n\n"
                    "EPS 格式适用于某些期刊的特定要求。")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"导出 EPS 失败：{str(e)}")

