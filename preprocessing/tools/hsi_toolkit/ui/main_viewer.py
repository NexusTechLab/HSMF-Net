"""
主视图组件 - 使用 pyqtgraph 实现
包含图像显示、ROI工具、光谱分析等功能
"""
import sys
import numpy as np
from pathlib import Path
import re
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter, 
                               QMessageBox, QInputDialog, QFileDialog, QLabel,
                               QLineEdit, QPushButton)
from PySide6.QtCore import Qt, Signal
import spectral.io.envi as envi
import pyqtgraph as pg

from ui.control_panel import ControlPanel
from ui.dataset_panel import DatasetPanel
from modules.segmentation import segment_seeds
from utils.preprocessing import preprocess_spectrum, selected_preprocess_steps


class MainViewer(QWidget):
    """主视图组件 - pyqtgraph 版本"""
    
    # 信号定义
    pixel_info = Signal(int, int, float)  # (x, y, value)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        
        # ========== 数据相关 ==========
        self.cube = None  # 高光谱数据立方体 (H, W, Bands)
        self.wavelengths = None  # 波长信息
        self.band_idx = 0  # 当前显示的波段
        self.file_path = None
        # 保存原始数据，用于对比预处理前后的光谱
        self.original_cube = None
        self.original_wavelengths = None
        
        # ========== ROI 相关 ==========
        self.roi_list = []  # ROI 信息列表
        self.current_drawing_roi = None  # 当前正在绘制的 ROI
        self.is_drawing_mode = False  # 是否处于绘制模式
        self.drawing_shape_type = None  # 正在绘制的形状类型
        self.drawing_start_pos = None  # 绘制起始位置
        self.is_mouse_pressed = False  # 鼠标是否按下
        
        # ========== 统一颜色管理 ==========
        # 高对比度、交错排列的颜色列表 - 确保相邻ROI颜色差异最大
        self.shared_colors = [
            # 第一轮：主要色系（高饱和度，最大对比）
            '#FF0000',  # 鲜红
            '#0000FF',  # 鲜蓝
            '#00FF00',  # 鲜绿
            '#FFFF00',  # 鲜黄
            '#FF00FF',  # 洋红
            '#00FFFF',  # 青色
            '#FF8000',  # 橙色
            '#8000FF',  # 紫色
            '#FF0080',  # 粉红
            '#00FF80',  # 青绿
            
            # 第二轮：深色系（与第一轮形成对比）
            '#800000',  # 深红
            '#000080',  # 深蓝
            '#008000',  # 深绿
            '#808000',  # 橄榄绿
            '#800080',  # 紫色
            '#008080',  # 深青
            '#DC143C',  # 猩红
            '#4169E1',  # 皇家蓝
            '#32CD32',  # 酸橙绿
            '#FFD700',  # 金色
            
            # 第三轮：亮色系
            '#FF6347',  # 番茄红
            '#1E90FF',  # 道奇蓝
            '#00FA9A',  # 春绿
            '#FFA500',  # 橙色
            '#FF1493',  # 深粉
            '#00CED1',  # 暗青
            '#FF4500',  # 橙红
            '#9370DB',  # 中紫
            '#00FF7F',  # 春绿
            '#FFB300',  # 琥珀色
            
            # 第四轮：补充色
            '#A52A2A',  # 棕色
            '#4682B4',  # 钢蓝
            '#228B22',  # 森林绿
            '#DAA520',  # 金棒色
            '#BA55D3',  # 中兰花紫
            '#20B2AA',  # 浅海绿
            '#CD853F',  # 秘鲁色
            '#6495ED',  # 矢车菊蓝
            '#2E8B57',  # 海洋绿
            '#FF69B4'   # 热粉
        ]
        self.color_usage = {}  # 记录颜色使用情况 {'color': 'roi'/'spectrum'/'both'}
        
        # ========== 光谱相关 ==========
        self.spectrum_curves = []  # 光谱曲线列表 [(curve, name, marker), ...]
        self.point_spectrum_curve = None  # 单点光谱曲线（单线模式用）
        self.spectrum_markers = []  # 像素点标记列表（pyqtgraph scatter plot items）
        
        # ========== 分割相关 ==========
        self.labels = None
        self.regions = None
        self.segmentation_info = None
        
        # ========== 光谱预处理预览相关 ==========
        self.last_spectrum_info = None  # 存储最后一次点击的光谱信息
        self.multi_line_spectrum_infos = []  # 存储多线模式下所有光谱信息
        
        # ========== 数据集管理 ==========
        self.datasets = {}  # {name: {cube, wavelengths, file_path}}
        self.current_dataset_name = None
        self.current_dci_rgb = None  # (r_idx, g_idx, b_idx)
        self._ref_rgb_warned = False
        self._true_color_warned = False
        
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """初始化UI"""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # ========== 左侧控制面板（包装在滚动区域中）==========
        from PySide6.QtWidgets import QScrollArea
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # 自动调整内部控件大小
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 禁用横向滚动条
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # 需要时显示纵向滚动条
        scroll_area.setMinimumWidth(220)  # 最小宽度
        scroll_area.setMaximumWidth(450)  # 最大宽度
        scroll_area.setFrameShape(QScrollArea.NoFrame)  # 无边框，更美观
        
        # 设置滚动条样式（可选，让滚动条更美观）
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: #f0f0f0;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #c0c0c0;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a0a0a0;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # 创建控制面板并放入滚动区域
        self.control_panel = ControlPanel()
        scroll_area.setWidget(self.control_panel)
        
        main_layout.addWidget(scroll_area)
        
        # ========== 中间图像显示（pyqtgraph ImageView）==========
        center_widget = QWidget()
        center_widget.setObjectName("ImageArea")
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建 ImageView
        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()  # 隐藏默认ROI按钮
        self.image_view.ui.menuBtn.hide()  # 隐藏菜单按钮
        self.image_view.ui.histogram.hide()  # 隐藏直方图（可选）
        
        # 隐藏颜色条
        if hasattr(self.image_view.ui, 'colorbar'):
            self.image_view.ui.colorbar.hide()
        
        # 获取关键组件
        self.view_box = self.image_view.getView()
        self.image_item = self.image_view.getImageItem()
        
        # 设置背景为白色
        self.view_box.setBackgroundColor('w')
        
        center_layout.addWidget(self.image_view, stretch=1)
        
        # ========== 右侧光谱分析 ==========
        right_widget = QWidget()
        right_widget.setObjectName("SpectrumArea")
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 8, 8, 8)
        
        # 创建光谱图
        self.plot = pg.PlotWidget(title="📈 光谱曲线")
        self.plot.setLabel("left", "反射率", **{'font-size': '10pt', 'color': '#333'})
        self.plot.setLabel("bottom", "波长 (nm)", **{'font-size': '10pt', 'color': '#333'})
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setBackground('w')
        
        # 设置轴颜色
        self.plot.getAxis('left').setPen(pg.mkPen(color='#666', width=1))
        self.plot.getAxis('bottom').setPen(pg.mkPen(color='#666', width=1))
        self.plot.getAxis('left').setTextPen(pg.mkPen(color='#333'))
        self.plot.getAxis('bottom').setTextPen(pg.mkPen(color='#333'))
        
        # 启用鼠标交互并添加十字线
        self.plot.scene().sigMouseMoved.connect(self._on_spectrum_mouse_move)
        
        # 创建十字线
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('gray', width=1, style=pg.QtCore.Qt.DashLine))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('gray', width=1, style=pg.QtCore.Qt.DashLine))
        self.plot.addItem(self.vLine, ignoreBounds=True)
        self.plot.addItem(self.hLine, ignoreBounds=True)
        
        # 创建文本标签显示坐标信息
        self.spectrum_coord_label = pg.TextItem(anchor=(0, 1), color='#333', fill=pg.mkBrush(255, 255, 255, 200))
        self.spectrum_coord_label.setPos(0, 0)
        self.plot.addItem(self.spectrum_coord_label)
        self.spectrum_coord_label.hide()  # 初始隐藏
        
        right_layout.addWidget(self.plot)
        
        # ========== 光谱控制工具栏（预处理 + 坐标轴，占据一行） ==========
        from PySide6.QtWidgets import QCheckBox
        control_bar = QWidget()
        bar_layout = QHBoxLayout(control_bar)
        bar_layout.setContentsMargins(4, 2, 4, 2)
        bar_layout.setSpacing(6)
        
        # 原始光谱开关
        bar_layout.addWidget(QLabel("📊"))
        self.chk_show_original = QCheckBox("原始")
        self.chk_show_original.setToolTip("显示原始光谱曲线")
        self.chk_show_original.setChecked(True)  # 默认显示
        bar_layout.addWidget(self.chk_show_original)
        
        # 分隔符
        sep0 = QLabel("|")
        sep0.setStyleSheet("color: #ccc;")
        bar_layout.addWidget(sep0)
        
        # 平滑
        bar_layout.addWidget(QLabel("🌀"))
        self.chk_smooth_ma = QCheckBox("MA")
        self.chk_smooth_ma.setToolTip("移动平均平滑")
        self.chk_smooth_sg = QCheckBox("SG")
        self.chk_smooth_sg.setToolTip("Savitzky-Golay平滑")
        bar_layout.addWidget(self.chk_smooth_ma)
        bar_layout.addWidget(self.chk_smooth_sg)
        
        # 分隔符
        sep1 = QLabel("|")
        sep1.setStyleSheet("color: #ccc;")
        bar_layout.addWidget(sep1)
        
        # 求导
        bar_layout.addWidget(QLabel("📈"))
        self.chk_deriv_1st = QCheckBox("1阶")
        self.chk_deriv_1st.setToolTip("一阶导数")
        self.chk_deriv_2nd = QCheckBox("2阶")
        self.chk_deriv_2nd.setToolTip("二阶导数")
        bar_layout.addWidget(self.chk_deriv_1st)
        bar_layout.addWidget(self.chk_deriv_2nd)
        
        # 分隔符
        sep2 = QLabel("|")
        sep2.setStyleSheet("color: #ccc;")
        bar_layout.addWidget(sep2)
        
        # 标准化
        bar_layout.addWidget(QLabel("📊"))
        self.chk_normalize = QCheckBox("0-1")
        self.chk_normalize.setToolTip("归一化 (0-1)")
        self.chk_standardize = QCheckBox("Z-score")
        self.chk_standardize.setToolTip("Z-score标准化")
        bar_layout.addWidget(self.chk_normalize)
        bar_layout.addWidget(self.chk_standardize)
        
        # ========== 弹性空间，分隔预处理（左）和坐标轴（右）==========
        bar_layout.addStretch()
        
        # ========== 坐标轴范围控制（右对齐）==========
        # 标题
        range_label = QLabel("坐标轴范围:")
        range_label.setStyleSheet("font-weight: bold; color: #555; font-size: 9pt;")
        bar_layout.addWidget(range_label)
        
        # xMin
        xmin_label = QLabel("xMin:")
        bar_layout.addWidget(xmin_label)
        
        self.edit_xmin = QLineEdit()
        self.edit_xmin.setPlaceholderText("自动")
        self.edit_xmin.setFixedWidth(60)
        self.edit_xmin.setToolTip("X轴最小值（波长或波段号）")
        self.edit_xmin.setText("950")  # ✅ 默认值
        bar_layout.addWidget(self.edit_xmin)
        
        # xMax
        xmax_label = QLabel("xMax:")
        bar_layout.addWidget(xmax_label)
        
        self.edit_xmax = QLineEdit()
        self.edit_xmax.setPlaceholderText("自动")
        self.edit_xmax.setFixedWidth(60)
        self.edit_xmax.setToolTip("X轴最大值（波长或波段号）")
        self.edit_xmax.setText("1700")  # ✅ 默认值
        bar_layout.addWidget(self.edit_xmax)
        
        # yMin
        ymin_label = QLabel("yMin:")
        bar_layout.addWidget(ymin_label)
        
        self.edit_ymin = QLineEdit()
        self.edit_ymin.setPlaceholderText("自动")
        self.edit_ymin.setFixedWidth(60)
        self.edit_ymin.setToolTip("Y轴最小值（反射率）")
        self.edit_ymin.setText("0.0")  # ✅ 默认值
        bar_layout.addWidget(self.edit_ymin)
        
        # yMax
        ymax_label = QLabel("yMax:")
        bar_layout.addWidget(ymax_label)
        
        self.edit_ymax = QLineEdit()
        self.edit_ymax.setPlaceholderText("自动")
        self.edit_ymax.setFixedWidth(60)
        self.edit_ymax.setToolTip("Y轴最大值（反射率）")
        self.edit_ymax.setText("1.0")  # ✅ 默认值
        bar_layout.addWidget(self.edit_ymax)
        
        # 刷新按钮
        self.btn_refresh_range = QPushButton("刷新")
        self.btn_refresh_range.setFixedWidth(60)
        self.btn_refresh_range.setToolTip("应用坐标轴范围设置")
        bar_layout.addWidget(self.btn_refresh_range)
        
        right_layout.addWidget(control_bar)
        
        # ========== 使用分割器 ==========
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(center_widget)
        self.splitter.addWidget(right_widget)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setSizes([600, 400])
        
        main_layout.addWidget(self.splitter, 5)
    
    def _connect_signals(self):
        """连接信号和槽"""
        # 控制面板信号
        self.control_panel.band_changed.connect(self._on_band_changed)
        self.control_panel.segment_requested.connect(self._on_segment_requested)
        self.control_panel.btn_show_pipeline.clicked.connect(self._show_pipeline)
        self.control_panel.btn_roi.toggled.connect(self._toggle_roi)
        self.control_panel.combo_colormode.currentTextChanged.connect(lambda: self.update_image())
        self.control_panel.slider_brightness.valueChanged.connect(self.update_image)
        
        # 显示模式切换信号
        self.control_panel.combo_display_mode.currentTextChanged.connect(self._on_display_mode_changed)
        
        # RGB波段选择信号
        self.control_panel.spin_r.valueChanged.connect(lambda: self._update_rgb_image())
        self.control_panel.spin_g.valueChanged.connect(lambda: self._update_rgb_image())
        self.control_panel.spin_b.valueChanged.connect(lambda: self._update_rgb_image())
        
        # RGB滑块同步波长标签
        self.control_panel.spin_r.valueChanged.connect(lambda: self.control_panel._update_rgb_wavelength_labels())
        self.control_panel.spin_g.valueChanged.connect(lambda: self.control_panel._update_rgb_wavelength_labels())
        self.control_panel.spin_b.valueChanged.connect(lambda: self.control_panel._update_rgb_wavelength_labels())
        
        # 图像变换按钮
        self.control_panel.btn_rotate_left.clicked.connect(self._rotate_left)
        self.control_panel.btn_rotate_right.clicked.connect(self._rotate_right)
        self.control_panel.btn_flip_ud.clicked.connect(self._flip_ud)
        self.control_panel.btn_flip_lr.clicked.connect(self._flip_lr)
        
        # 光谱绘制按钮
        self.control_panel.btn_clear_spectra.clicked.connect(self._clear_all_spectra)
        self.control_panel.btn_avg_spectra.clicked.connect(self._average_spectra)
        
        # 光谱模式切换信号
        self.control_panel.combo_spectrum_mode.currentTextChanged.connect(self._on_spectrum_mode_changed)
        
        # 波长范围刷新按钮（现在在光谱图下方）
        self.btn_refresh_range.clicked.connect(self._apply_axis_range)
        
        # 数据集管理信号
        self.control_panel.dataset_panel.btn_add.clicked.connect(self._batch_open_files)
        self.control_panel.dataset_panel.btn_remove.clicked.connect(self._batch_remove_datasets)
        self.control_panel.dataset_panel.dataset_changed.connect(self._switch_dataset)
        
        # ROI 管理信号
        self.control_panel.roi_delete_requested.connect(self._delete_roi)
        self.control_panel.roi_clear_requested.connect(self._clear_all_rois)
        self.control_panel.roi_export_requested.connect(self._export_all_rois)
        self.control_panel.roi_item_selected.connect(self._highlight_roi)
        
        # 图像鼠标事件
        self.image_view.scene.sigMouseClicked.connect(self._on_image_click)
        
        # 安装事件过滤器以捕获鼠标按下和移动事件
        self.view_box.scene().sigMouseMoved.connect(self._on_mouse_move)
        
        # 预处理复选框信号（实时更新光谱）
        for checkbox in [self.chk_show_original, self.chk_smooth_ma, self.chk_smooth_sg, 
                        self.chk_deriv_1st, self.chk_deriv_2nd, self.chk_normalize, 
                        self.chk_standardize]:
            checkbox.toggled.connect(self._redraw_spectrum_with_preprocess)
    
    # ========== 数据加载 ==========
    def _batch_open_files(self):
        """批量打开ENVI文件"""
        hdr_paths, _ = QFileDialog.getOpenFileNames(
            self, "批量选择 HDR 文件", str(Path.cwd()), "HDR (*.hdr *.HDR)"
        )
        if not hdr_paths:
            return
        
        success_count = 0
        failed_files = []
        
        for hdr_path in hdr_paths:
            try:
                # 读取ENVI文件
                img = envi.open(hdr_path)
                cube = img.load()
                
                # 获取波长信息
                wavelengths = img.metadata.get('wavelength', None)
                
                # 添加到数据集列表
                name = Path(hdr_path).stem
                
                # 检查是否是参考数据文件
                is_reference = '_ref' in name.lower() or 'reference' in name.lower()
                if is_reference:
                    print(f"[提示] 检测到参考数据文件: {name}")
                
                dataset_dict = {
                    'cube': cube,
                    'wavelengths': wavelengths,
                    'file_path': hdr_path,
                    'is_reference': is_reference,  # 标记是否为参考数据
                    'dci_rgb': self._load_dci_rgb_for_hdr(hdr_path)
                }
                self.datasets[name] = dataset_dict
                
                # 添加到UI列表（传递完整的数据集字典）
                self.control_panel.dataset_panel.add_dataset(name, dataset_dict)
                success_count += 1
                
                # 如果是第一个文件，自动加载（不显示提示）
                if success_count == 1:
                    self._load_dataset(name, show_message=False)
                    
            except Exception as e:
                failed_files.append(f"{Path(hdr_path).name}: {str(e)}")
        
        # 显示汇总结果（只显示一次）
        if success_count > 0 or failed_files:
            msg_parts = []
            if success_count > 0:
                msg_parts.append(f"✅ 成功加载: {success_count} 个数据集")
            if failed_files:
                msg_parts.append(f"❌ 加载失败: {len(failed_files)} 个")
                if len(failed_files) <= 3:
                    msg_parts.append("\n失败文件:")
                    msg_parts.extend([f"  • {f}" for f in failed_files])
                else:
                    msg_parts.append(f"\n失败文件（仅显示前3个）:")
                    msg_parts.extend([f"  • {f}" for f in failed_files[:3]])
            
            msg = "\n".join(msg_parts)
            if failed_files and success_count > 0:
                QMessageBox.warning(self, "批量加载完成", msg)
            elif success_count > 0:
                QMessageBox.information(self, "批量加载完成", msg)
            else:
                QMessageBox.critical(self, "加载失败", msg)
    
    def _load_dataset(self, name, show_message=False):
        """加载指定数据集
        
        Args:
            name: 数据集名称
            show_message: 是否显示加载成功提示（默认False，避免频繁打扰）
        """
        if name not in self.datasets:
            return
        
        dataset = self.datasets[name]
        self.cube = dataset['cube']
        self.wavelengths = dataset['wavelengths']
        self.file_path = dataset['file_path']
        self.current_dataset_name = name
        self.current_dci_rgb = dataset.get('dci_rgb')
        self._ref_rgb_warned = False
        self._true_color_warned = False
        
        # ⭐ 保存原始数据副本（用于预处理前后对比）
        self.original_cube = self.cube.copy()
        self.original_wavelengths = list(self.wavelengths) if self.wavelengths else None
        
        # 初始化波段索引
        bands = self.cube.shape[2]
        self.band_idx = bands // 2
        
        # 更新控制面板
        self.control_panel.set_band_range(bands, self.wavelengths)
        self.control_panel.btn_roi.setEnabled(True)
        self.control_panel.btn_segment.setEnabled(True)  # 启用分割按钮
        self._apply_dataset_rgb_defaults()
        self._print_dataset_summary(name)
        
        # 更新波段输入框显示
        self.control_panel.edit_band.setText(str(self.band_idx))
        
        # 清除旧的ROI和光谱
        self._clear_all_rois()
        self.plot.clear()
        
        # 根据当前显示模式更新图像
        if self.control_panel.combo_display_mode.currentText() in ("RGB合成", "真彩色(B)", "假彩色(NIR-R-G)"):
            self._update_rgb_image()
        else:
            self.update_image()
        
        # 仅在需要时显示提示
        if show_message:
            QMessageBox.information(self, "加载成功", 
                f"数据集: {name}\n尺寸: {self.cube.shape}")
    
    def _load_dataset_from_dict(self, dataset_dict):
        """从数据集字典加载数据"""
        self.cube = dataset_dict['cube']
        self.wavelengths = dataset_dict.get('wavelengths', None)  # 使用get方法避免KeyError
        self.file_path = dataset_dict.get('file_path', '')
        self.current_dci_rgb = dataset_dict.get('dci_rgb')
        self._ref_rgb_warned = False
        self._true_color_warned = False
        
        # 如果没有波长信息，创建默认波长
        if self.wavelengths is None:
            bands = self.cube.shape[2]
            self.wavelengths = np.arange(bands)  # 创建默认波长数组
        
        # ⭐ 保存原始数据副本（用于预处理前后对比）
        self.original_cube = self.cube.copy()
        self.original_wavelengths = list(self.wavelengths) if self.wavelengths else None
        
        # 初始化波段索引
        bands = self.cube.shape[2]
        self.band_idx = bands // 2
        
        # 更新控制面板
        self.control_panel.set_band_range(bands, self.wavelengths)
        self.control_panel.btn_roi.setEnabled(True)
        self.control_panel.btn_segment.setEnabled(True)
        self._apply_dataset_rgb_defaults()
        dataset_name = self.current_dataset_name or "未命名数据集"
        self._print_dataset_summary(dataset_name)
        
        # 更新波段输入框显示
        self.control_panel.edit_band.setText(str(self.band_idx))
        
        # 清除旧的ROI和光谱
        self._clear_all_rois()
        self.plot.clear()
        
        # 根据当前显示模式更新图像
        if self.control_panel.combo_display_mode.currentText() in ("RGB合成", "真彩色(B)", "假彩色(NIR-R-G)"):
            self._update_rgb_image()
        else:
            self.update_image()
    
    def _switch_dataset(self, idx):
        """切换数据集（静默切换，不显示提示）"""
        if idx < 0:
            return
        
        # 从数据集面板获取当前选中的数据集
        dataset = self.control_panel.dataset_panel.get_current_dataset()
        if dataset:
            # 获取数据集名称
            dataset_name = self.control_panel.dataset_panel.get_dataset_name_by_index(idx)
            if dataset_name:
                print(f"切换数据集: {dataset_name} (索引: {idx})")  # 调试信息
                self.current_dataset_name = dataset_name
                
                # 同步更新主视图的数据集字典
                self.datasets[dataset_name] = dataset
                
                # 加载数据集
                self._load_dataset_from_dict(dataset)
            else:
                print(f"无法获取数据集名称，索引: {idx}")  # 调试信息
        else:
            print(f"无法获取数据集，索引: {idx}")  # 调试信息
    
    def _batch_remove_datasets(self):
        """批量删除所有数据集"""
        if len(self.control_panel.dataset_panel.datasets) == 0:
            return
        
        count = len(self.control_panel.dataset_panel.datasets)
        reply = QMessageBox.question(
            self, "确认批量删除", 
            f"确定要删除所有 {count} 个数据集吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.control_panel.dataset_panel.remove_all()
            
            # 清空主视图的数据集字典
            self.datasets.clear()
            self.current_dataset_name = None
            
            # 清空视图
            self.cube = None
            self.wavelengths = None
            self.labels = None
            self.regions = None
            self.segmentation_info = None
            self.plot.clear()
            
            # 禁用相关按钮
            self.control_panel.btn_roi.setEnabled(False)
            self.control_panel.btn_segment.setEnabled(False)
            self.control_panel.btn_show_pipeline.setEnabled(False)
            
            # 清空图像显示
            self.image_view.clear()
            self.plot.clear()
    
    # ========== 图像更新 ==========
    def update_image(self):
        """更新图像显示"""
        if self.cube is None:
            return
        
        # 获取当前波段图像
        img = self.cube[:, :, self.band_idx].copy()
        
        # 应用亮度调整
        brightness = self.control_panel.slider_brightness.value()
        if brightness != 0:
            # 亮度调整：-100到100映射到-1到1
            adjustment = brightness / 100.0
            
            # 方法1：线性调整
            img_adjusted = img + adjustment * 0.5
            
            # 方法2：对比度调整（可选）
            if abs(adjustment) > 0.5:
                # 高亮度时增加对比度
                contrast_factor = 1 + abs(adjustment) * 0.5
                img_mean = img_adjusted.mean()
                img_adjusted = (img_adjusted - img_mean) * contrast_factor + img_mean
            
            # 确保值在有效范围内
            img = np.clip(img_adjusted, 0, 1)
        
        # 设置图像（默认进行上下镜像翻转，注意转置以匹配 pyqtgraph 坐标系）
        # 不进行上下翻转，保持原始图像方向
        self.image_view.setImage(img.T, autoRange=False, autoLevels=False, 
                                 autoHistogramRange=False)
        
        # 应用颜色映射
        colormode = self.control_panel.combo_colormode.currentText()
        if colormode != "灰度":
            import matplotlib.cm as cm
            colormap_dict = {
                "RGB伪彩色": 'viridis',
                "热力图": 'hot',
                "光谱色": 'nipy_spectral'
            }
            cmap_name = colormap_dict.get(colormode, 'gray')
            cmap = cm.get_cmap(cmap_name)
            lut = (cmap(np.arange(256)) * 255).astype(np.uint8)
            self.image_view.setColorMap(pg.ColorMap(pos=np.linspace(0, 1, 256), color=lut))
        else:
            # 灰度模式：使用灰度颜色映射
            lut = np.column_stack([np.arange(256)] * 3).astype(np.uint8)
            self.image_view.setColorMap(pg.ColorMap(pos=np.linspace(0, 1, 256), color=lut))
        
        # 设置显示范围
        vmin, vmax = np.nanpercentile(img, [1, 99])
        self.image_view.setLevels(vmin, vmax)
        
        # 更新控制面板
        self.control_panel.update_wavelength_label(self.band_idx)
    
    def _on_display_mode_changed(self, mode: str):
        """显示模式切换"""
        if mode == "灰度图":
            # 切换到灰度模式，使用单波段
            self.update_image()
        else:
            # RGB合成 / 真彩色 / 假彩色
            self._update_rgb_image()

    def export_true_false_color_images(self, output_dir: Path):
        """导出真彩色(B, CMF近似积分)与假彩色(NIR-R-G)图像"""
        if self.cube is None:
            raise ValueError("请先加载高光谱数据")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        true_color_img = self._compose_true_color_cmf()
        false_color_img = self._compose_false_color_nir_rg()

        import matplotlib.pyplot as plt

        dataset_name = self.current_dataset_name or "dataset"
        true_color_path = output_dir / f"{dataset_name}_true_color.png"
        false_color_path = output_dir / f"{dataset_name}_false_color_nir_rg.png"

        plt.imsave(true_color_path, true_color_img)
        plt.imsave(false_color_path, false_color_img)

        return {
            "true_color_path": str(true_color_path),
            "false_color_path": str(false_color_path)
        }

    def _compose_true_color_cmf(self):
        """使用CIE 1931 CMF近似模型积分，生成真彩色图像"""
        cube = self._get_cube_3d()
        if self.wavelengths is None:
            # 无波长信息时退化为三分位伪彩
            return self._compose_fallback_distinct_rgb(cube)

        wavelengths = np.array([float(w) for w in self.wavelengths], dtype=np.float64)
        mask = (wavelengths >= 380.0) & (wavelengths <= 780.0)
        if mask.sum() < 3:
            # 无可见光覆盖时，退化为三分位伪彩（避免三通道重合）
            return self._compose_fallback_distinct_rgb(cube)

        wl = wavelengths[mask]
        cube_vis = cube[:, :, mask].astype(np.float64)

        # 用波长步长近似积分权重
        dw = np.gradient(wl)
        x_bar, y_bar, z_bar = self._cie1931_cmf_approx(wl)

        X = np.tensordot(cube_vis, x_bar * dw, axes=([2], [0]))
        Y = np.tensordot(cube_vis, y_bar * dw, axes=([2], [0]))
        Z = np.tensordot(cube_vis, z_bar * dw, axes=([2], [0]))

        xyz = np.stack([X, Y, Z], axis=-1)

        # XYZ -> linear sRGB (D65)
        m_xyz_to_rgb = np.array([
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570]
        ], dtype=np.float64)
        rgb_linear = np.tensordot(xyz, m_xyz_to_rgb.T, axes=([2], [0]))
        rgb_linear = np.clip(rgb_linear, 0.0, None)

        # 使用高分位做白平衡/曝光归一
        scale = np.percentile(rgb_linear, 99.5)
        if scale <= 0:
            scale = np.max(rgb_linear) if np.max(rgb_linear) > 0 else 1.0
        rgb_linear = np.clip(rgb_linear / scale, 0.0, 1.0)

        # sRGB gamma
        threshold = 0.0031308
        rgb = np.where(
            rgb_linear <= threshold,
            12.92 * rgb_linear,
            1.055 * np.power(rgb_linear, 1.0 / 2.4) - 0.055
        )
        return np.clip(rgb, 0.0, 1.0)

    def _compose_false_color_nir_rg(self):
        """生成假彩色图像：R=NIR(760-900), G=Red(630-690), B=Green(520-580)"""
        cube = self._get_cube_3d()

        # 优先使用设备DCI文件给出的真实通道映射
        if self.current_dci_rgb is not None:
            r_idx, g_idx, b_idx = self.current_dci_rgb
            bands = cube.shape[2]
            if 0 <= r_idx < bands and 0 <= g_idx < bands and 0 <= b_idx < bands:
                r = cube[:, :, r_idx]
                g = cube[:, :, g_idx]
                b = cube[:, :, b_idx]
                return np.stack(
                    [
                        self._robust_normalize_channel(r),
                        self._robust_normalize_channel(g),
                        self._robust_normalize_channel(b),
                    ],
                    axis=-1,
                )

        if self.wavelengths is not None:
            wavelengths = np.array([float(w) for w in self.wavelengths], dtype=np.float64)
            idx_r = np.where((wavelengths >= 760.0) & (wavelengths <= 900.0))[0]
            idx_g = np.where((wavelengths >= 630.0) & (wavelengths <= 690.0))[0]
            idx_b = np.where((wavelengths >= 520.0) & (wavelengths <= 580.0))[0]

            # 任一关键波段缺失时，回退为三分位伪彩，避免产生无意义黑白纹理图
            if idx_r.size == 0 or idx_g.size == 0 or idx_b.size == 0:
                return self._compose_fallback_distinct_rgb(cube)

            r = np.squeeze(np.mean(cube[:, :, idx_r], axis=2, dtype=np.float64))
            g = np.squeeze(np.mean(cube[:, :, idx_g], axis=2, dtype=np.float64))
            b = np.squeeze(np.mean(cube[:, :, idx_b], axis=2, dtype=np.float64))
        else:
            # 无波长信息时，用波段位置近似
            return self._compose_fallback_distinct_rgb(cube)

        r_norm = self._robust_normalize_channel(r)
        g_norm = self._robust_normalize_channel(g)
        b_norm = self._robust_normalize_channel(b)
        return np.stack([r_norm, g_norm, b_norm], axis=-1)

    def _mean_band_in_wavelength_range(self, wavelengths, wl_min, wl_max):
        """取指定波长范围内所有波段均值；若为空则退化为最近波段"""
        cube = self._get_cube_3d()
        idx = np.where((wavelengths >= wl_min) & (wavelengths <= wl_max))[0]
        if idx.size == 0:
            target = (wl_min + wl_max) / 2.0
            nearest = int(np.argmin(np.abs(wavelengths - target)))
            return np.squeeze(cube[:, :, nearest]).astype(np.float64)
        return np.squeeze(np.mean(cube[:, :, idx], axis=2, dtype=np.float64))

    def _compose_fallback_distinct_rgb(self, cube):
        """回退三通道：从当前波段范围取相互分离的三个通道"""
        n = cube.shape[2]
        r_idx = min(n - 1, int(n * 0.80))
        g_idx = min(n - 1, int(n * 0.55))
        b_idx = min(n - 1, int(n * 0.30))
        r = cube[:, :, r_idx]
        g = cube[:, :, g_idx]
        b = cube[:, :, b_idx]
        return np.stack(
            [
                self._robust_normalize_channel(r),
                self._robust_normalize_channel(g),
                self._robust_normalize_channel(b),
            ],
            axis=-1,
        )

    def _is_reference_dataset(self):
        """判断当前数据集是否参考板/白板数据"""
        name = (self.current_dataset_name or "").lower()
        if "_ref" in name or "reference" in name:
            return True
        ds = self.datasets.get(self.current_dataset_name) if self.current_dataset_name else None
        if isinstance(ds, dict):
            return bool(ds.get("is_reference", False))
        return False

    def _has_visible_wavelengths(self):
        """判断当前数据是否覆盖可见光范围"""
        if not self.wavelengths:
            return False
        try:
            w = np.array([float(x) for x in self.wavelengths], dtype=np.float64)
        except Exception:
            return False
        return bool(np.any((w >= 380.0) & (w <= 780.0)))

    def _load_dci_rgb_for_hdr(self, hdr_path: str):
        """读取与HDR同名的DCI文件中的RGB通道索引"""
        dci_path = Path(hdr_path).with_suffix(".dci")
        if not dci_path.exists():
            return None
        try:
            text = dci_path.read_text(encoding="utf-8", errors="ignore")
            r_match = re.search(r"^\s*Red\s*=\s*(\d+)\s*$", text, flags=re.MULTILINE)
            g_match = re.search(r"^\s*Green\s*=\s*(\d+)\s*$", text, flags=re.MULTILINE)
            b_match = re.search(r"^\s*Blue\s*=\s*(\d+)\s*$", text, flags=re.MULTILINE)
            if not (r_match and g_match and b_match):
                return None
            return int(r_match.group(1)), int(g_match.group(1)), int(b_match.group(1))
        except Exception:
            return None

    def _apply_dataset_rgb_defaults(self):
        """按真实数据设置默认RGB通道（优先DCI）"""
        if self.cube is None:
            return
        cube = self._get_cube_3d()
        bands = cube.shape[2]

        if self.current_dci_rgb is not None:
            r_idx, g_idx, b_idx = self.current_dci_rgb
            if 0 <= r_idx < bands and 0 <= g_idx < bands and 0 <= b_idx < bands:
                self.control_panel.spin_r.setValue(r_idx)
                self.control_panel.spin_g.setValue(g_idx)
                self.control_panel.spin_b.setValue(b_idx)
                self.control_panel._update_rgb_wavelength_labels()
                return

    def _print_dataset_summary(self, dataset_name: str):
        """加载数据后在终端打印关键参数"""
        try:
            cube = self._get_cube_3d()
            bands = cube.shape[2]
            print("\n" + "=" * 60)
            print(f"[数据集] {dataset_name}")
            print(f"[尺寸] {cube.shape}  [dtype] {cube.dtype}")
            print(
                f"[数值] min={float(np.nanmin(cube)):.4f}, "
                f"max={float(np.nanmax(cube)):.4f}, "
                f"mean={float(np.nanmean(cube)):.4f}"
            )

            if self.wavelengths:
                try:
                    w = np.array([float(x) for x in self.wavelengths], dtype=np.float64)
                    print(f"[波长] {w.min():.2f} ~ {w.max():.2f} nm  (bands={w.size})")
                    visible = np.where((w >= 380.0) & (w <= 780.0))[0]
                    print(f"[可见光覆盖] {'是' if visible.size > 0 else '否'}")
                except Exception:
                    print("[波长] 解析失败")
            else:
                print("[波长] 无")

            if self.current_dci_rgb is not None:
                r_idx, g_idx, b_idx = self.current_dci_rgb
                msg = f"[DCI RGB] R={r_idx}, G={g_idx}, B={b_idx}"
                if self.wavelengths:
                    try:
                        w = np.array([float(x) for x in self.wavelengths], dtype=np.float64)
                        if 0 <= r_idx < len(w) and 0 <= g_idx < len(w) and 0 <= b_idx < len(w):
                            msg += (
                                f" -> ({w[r_idx]:.2f}, {w[g_idx]:.2f}, {w[b_idx]:.2f}) nm"
                            )
                    except Exception:
                        pass
                print(msg)
            else:
                print("[DCI RGB] 无")
            print("=" * 60 + "\n")
        except Exception as e:
            print(f"[警告] 数据摘要输出失败: {e}")

    def _get_cube_3d(self):
        """统一获取(H, W, Bands)三维数据，兼容单例维度"""
        cube = np.asarray(self.cube)
        cube = np.squeeze(cube)
        if cube.ndim != 3:
            raise ValueError(f"数据维度异常，期望3维(H,W,Bands)，实际为 {cube.shape}")
        return cube

    @staticmethod
    def _robust_normalize_channel(channel):
        """通道鲁棒拉伸到[0,1]"""
        c = np.nan_to_num(np.squeeze(channel).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if c.ndim != 2:
            raise ValueError(f"通道维度异常，期望2维，实际为 {c.shape}")
        p2, p98 = np.percentile(c, [2, 98])
        if p98 > p2:
            c = (c - p2) / (p98 - p2)
        else:
            c_min, c_max = np.min(c), np.max(c)
            if c_max > c_min:
                c = (c - c_min) / (c_max - c_min)
        return np.clip(c, 0.0, 1.0)

    @staticmethod
    def _cie1931_cmf_approx(wavelengths):
        """CIE 1931 2° CMF 的高斯近似"""
        wl = np.asarray(wavelengths, dtype=np.float64)
        x_bar = (
            1.056 * np.exp(-0.5 * ((wl - 599.8) / 37.9) ** 2) +
            0.362 * np.exp(-0.5 * ((wl - 442.0) / 16.0) ** 2) -
            0.065 * np.exp(-0.5 * ((wl - 501.1) / 20.4) ** 2)
        )
        y_bar = (
            0.821 * np.exp(-0.5 * ((wl - 568.8) / 46.9) ** 2) +
            0.286 * np.exp(-0.5 * ((wl - 530.9) / 16.3) ** 2)
        )
        z_bar = (
            1.217 * np.exp(-0.5 * ((wl - 437.0) / 11.8) ** 2) +
            0.681 * np.exp(-0.5 * ((wl - 459.0) / 26.0) ** 2)
        )
        return np.clip(x_bar, 0.0, None), np.clip(y_bar, 0.0, None), np.clip(z_bar, 0.0, None)
    
    def _update_rgb_image(self):
        """更新RGB合成图像"""
        if self.cube is None:
            return

        mode = self.control_panel.combo_display_mode.currentText()
        if mode not in ("RGB合成", "真彩色(B)", "假彩色(NIR-R-G)"):
            return

        # 参考板数据不适合真彩/假彩预览，自动回退灰度
        if mode in ("真彩色(B)", "假彩色(NIR-R-G)") and self._is_reference_dataset():
            if not self._ref_rgb_warned:
                QMessageBox.information(
                    self,
                    "提示",
                    "当前是参考板数据（_ref），不适合真彩/假彩预览。\n已自动切回灰度图显示。"
                )
                self._ref_rgb_warned = True
            self.control_panel.combo_display_mode.blockSignals(True)
            self.control_panel.combo_display_mode.setCurrentText("灰度图")
            self.control_panel.combo_display_mode.blockSignals(False)
            self.update_image()
            return

        # 当前数据无可见光时，真彩色自动切到假彩色（按真实数据显示）
        if mode == "真彩色(B)" and not self._has_visible_wavelengths():
            if not self._true_color_warned:
                QMessageBox.information(
                    self,
                    "提示",
                    "当前数据波长范围不含可见光（约380-780nm），无法生成真彩色。\n"
                    "已自动切换为假彩色(NIR-R-G/DCI)。"
                )
                self._true_color_warned = True
            self.control_panel.combo_display_mode.blockSignals(True)
            self.control_panel.combo_display_mode.setCurrentText("假彩色(NIR-R-G)")
            self.control_panel.combo_display_mode.blockSignals(False)
            mode = "假彩色(NIR-R-G)"
        
        try:
            if mode == "真彩色(B)":
                rgb_img = self._compose_true_color_cmf()
                r_band, g_band, b_band = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]
            elif mode == "假彩色(NIR-R-G)":
                rgb_img = self._compose_false_color_nir_rg()
                r_band, g_band, b_band = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]
            else:
                # 手动RGB合成
                cube = self._get_cube_3d()
                r_idx = self.control_panel.spin_r.value()
                g_idx = self.control_panel.spin_g.value()
                b_idx = self.control_panel.spin_b.value()

                # 提取波段，确保是2D数组
                r_band = cube[:, :, r_idx].copy()
                g_band = cube[:, :, g_idx].copy()
                b_band = cube[:, :, b_idx].copy()
            
            # 压缩多余的维度（如果存在）
            r_band = np.squeeze(r_band)
            g_band = np.squeeze(g_band)
            b_band = np.squeeze(b_band)
            
            # 确保数据是2D的
            if r_band.ndim != 2 or g_band.ndim != 2 or b_band.ndim != 2:
                print(f"[错误] 波段维度不正确: R={r_band.shape}, G={g_band.shape}, B={b_band.shape}")
                return
            
            # 检查数据范围和统计信息
            max_val = max(r_band.max(), g_band.max(), b_band.max())
            min_val = min(r_band.min(), g_band.min(), b_band.min())
            mean_val = (r_band.mean() + g_band.mean() + b_band.mean()) / 3
            
            print(f"[调试] RGB波段统计:")
            print(f"  R: [{r_band.min():.4f}, {r_band.max():.4f}], mean={r_band.mean():.4f}")
            print(f"  G: [{g_band.min():.4f}, {g_band.max():.4f}], mean={g_band.mean():.4f}")
            print(f"  B: [{b_band.min():.4f}, {b_band.max():.4f}], mean={b_band.mean():.4f}")
            
            # 检查是否可能是错误的文件或需要特殊处理
            if max_val < 0.1 and mean_val < 0.05:
                print(f"\n❌ [严重警告] 数据异常！")
                print(f"   均值={mean_val:.4f}, 最大值={max_val:.4f}")
                print(f"\n💡 [建议]:")
                print(f"   1. 如果要查看小麦籽粒，请打开 'newrawfile.hdr' (不带_ref后缀)")
                print(f"   2. 如果必须查看白板数据，当前文件可能已经过反射率校正")
                print(f"   3. 尝试打开 'newrawfile.dcor' 或 'newrawfile.raw' 查看原始数据")
                print(f"\n🔧 [当前处理]: 尝试读取 .raw 或 .dcor 文件替代...")
                
                # 提示用户
                msg = QMessageBox(self.parent() if hasattr(self, 'parent') else None)
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("数据异常")
                msg.setText("当前数据无法正常显示（数值过低）")
                msg.setInformativeText(
                    "建议操作：\n\n"
                    "1. 如果要查看小麦籽粒图像：\n"
                    "   → 打开 'newrawfile.hdr' (不带_ref)\n\n"
                    "2. 如果必须使用白板数据：\n"
                    "   → 尝试 .raw 或 .dcor 文件\n\n"
                    "3. 白板数据通常用于校正计算，\n"
                    "   不直接用于可视化"
                )
                msg.exec()
            
            # 数据归一化处理
            if max_val > 1.5:  # DN值或其他未归一化数据
                print(f"[处理] DN值数据，归一化到 [0, 1]")
                r_band = r_band / max_val
                g_band = g_band / max_val
                b_band = b_band / max_val
            
            # 移除异常值（NaN, Inf等）
            r_band = np.nan_to_num(r_band, nan=0.0, posinf=1.0, neginf=0.0)
            g_band = np.nan_to_num(g_band, nan=0.0, posinf=1.0, neginf=0.0)
            b_band = np.nan_to_num(b_band, nan=0.0, posinf=1.0, neginf=0.0)
            
            # 应用亮度调整
            brightness = self.control_panel.slider_brightness.value()
            if brightness != 0:
                adjustment = brightness / 100.0
                r_band = np.clip(r_band + adjustment * 0.5, 0, 1)
                g_band = np.clip(g_band + adjustment * 0.5, 0, 1)
                b_band = np.clip(b_band + adjustment * 0.5, 0, 1)
            
            # 对真彩/假彩输入仍做鲁棒拉伸，使显示风格与手动RGB一致
            r_norm = self._robust_normalize_channel(r_band)
            g_norm = self._robust_normalize_channel(g_band)
            b_norm = self._robust_normalize_channel(b_band)

            # 合成RGB图像 (H, W, 3)
            rgb_img = np.stack([r_norm, g_norm, b_norm], axis=-1)
            
            # 确保形状正确
            if rgb_img.ndim != 3 or rgb_img.shape[2] != 3:
                print(f"[错误] RGB图像维度不正确: {rgb_img.shape}")
                return
            
            # 应用自适应直方图均衡化（可选，提升对比度）
            rgb_mean = rgb_img.mean()
            rgb_std = rgb_img.std()
            print(f"[调试] RGB图像统计: mean={rgb_mean:.3f}, std={rgb_std:.3f}")
            
            # 检查三通道是否过于相似（接近灰度）
            channel_delta = max(
                float(np.mean(np.abs(rgb_img[:, :, 0] - rgb_img[:, :, 1]))),
                float(np.mean(np.abs(rgb_img[:, :, 1] - rgb_img[:, :, 2]))),
                float(np.mean(np.abs(rgb_img[:, :, 0] - rgb_img[:, :, 2]))),
            )

            # 检查图像对比度和亮度
            if rgb_mean < 0.1 and channel_delta >= 0.01:
                print(f"[信息] 图像均值较低 (mean={rgb_mean:.3f})")
                print("[处理] 应用强对比度增强以提升可视化效果")
                # 对于白板数据或暗图像，应用强对比度拉伸
                for i in range(3):
                    channel = rgb_img[:, :, i]
                    # 使用0.5-99.5百分位，更激进的拉伸
                    p_low, p_high = np.percentile(channel, [0.5, 99.5])
                    if p_high > p_low:
                        rgb_img[:, :, i] = np.clip((channel - p_low) / (p_high - p_low), 0, 1)
                    else:
                        # 如果没有变化，尝试直接拉伸到全范围
                        c_min, c_max = channel.min(), channel.max()
                        if c_max > c_min:
                            rgb_img[:, :, i] = (channel - c_min) / (c_max - c_min)
            elif rgb_std < 0.15 and channel_delta >= 0.01:  # 对比度较低时增强
                print("[信息] 应用对比度增强")
                # 简单的对比度拉伸
                for i in range(3):
                    channel = rgb_img[:, :, i]
                    p_low, p_high = np.percentile(channel, [5, 95])
                    if p_high > p_low:
                        rgb_img[:, :, i] = np.clip((channel - p_low) / (p_high - p_low), 0, 1)
            
            # 转换为uint8 (0-255)
            rgb_img_uint8 = (rgb_img * 255).astype(np.uint8)
            print(f"[调试] RGB uint8 范围: min={rgb_img_uint8.min()}, max={rgb_img_uint8.max()}")

            # 黑帧保护：如果结果几乎全黑，回退到三分位伪彩
            if int(rgb_img_uint8.max()) <= 1:
                print("[警告] RGB结果接近全黑，回退到三分位伪彩显示")
                cube = self._get_cube_3d()
                fallback = self._compose_fallback_distinct_rgb(cube)
                rgb_img_uint8 = (np.clip(fallback, 0.0, 1.0) * 255).astype(np.uint8)
            
            # pyqtgraph的ImageView对RGB图像的处理：
            # 直接传入 (H, W, 3) 格式，不需要转置
            # 设置RGB图像
            self.image_view.setImage(rgb_img_uint8, autoRange=False, 
                                    autoLevels=False, autoHistogramRange=False)
            # 强制RGB显示范围，避免沿用灰度levels导致整图压黑
            self.image_view.setLevels(0, 255)
            
        except Exception as e:
            print(f"[错误] RGB图像更新失败: {e}")
            import traceback
            traceback.print_exc()
    
    # ========== 事件处理 ==========
    def _on_band_changed(self, idx: int):
        """波段变化"""
        self.band_idx = idx
        self.update_image()
        
        # 更新 ROI 光谱
        self._update_all_roi_spectra()
    
    def _on_image_click(self, event):
        """处理鼠标点击事件"""
        if self.cube is None:
            return
        
        # 获取点击位置
        pos = event.scenePos()
        if not self.view_box.sceneBoundingRect().contains(pos):
            return
            
        mouse_point = self.view_box.mapSceneToView(pos)
        x_view = int(mouse_point.x())
        y_view = int(mouse_point.y())
        
        # 检查显示坐标有效性
        h, w = self.cube.shape[:2]
        if not (0 <= x_view < w and 0 <= y_view < h):
            return
        
        # 坐标转换：显示坐标 -> 数据坐标
        # 显示: img.T（只转置，不翻转）
        # 转置: (row, col) -> (col, row)，所以 x_view 对应 col, y_view 对应 row
        # 不再需要反向 flipud
        row_data = y_view
        col_data = x_view
        
        # 检查数据坐标有效性
        if not (0 <= row_data < h and 0 <= col_data < w):
            return
        
        # 处理 ROI 绘制模式
        if self.is_drawing_mode:
            if event.button() == Qt.LeftButton:
                # 左键按下：开始绘制（使用显示坐标）
                if not self.is_mouse_pressed:
                    self._start_roi_drawing(x_view, y_view)
                    self.is_mouse_pressed = True
                # 左键释放：完成绘制
                else:
                    self._finish_roi_drawing()
                    self.is_mouse_pressed = False
            return
        
        # 非绘制模式：显示单点光谱（使用数据坐标）
        if event.button() == Qt.LeftButton and event.double():
            return  # 忽略双击
            
        spectrum = self.cube[row_data, col_data, :].ravel()
        pixel_value = self.cube[row_data, col_data, self.band_idx]
        
        # 发射像素信息信号（使用数据坐标）
        self.pixel_info.emit(col_data, row_data, float(pixel_value))
        
        # 准备 X 轴数据
        if self.wavelengths and len(self.wavelengths) == len(spectrum):
            x_data = np.array([float(w) for w in self.wavelengths])
        else:
            x_data = np.arange(len(spectrum))
        
        # ⭐ 保存最后一次点击的光谱信息（用于预处理实时更新）
        self.last_spectrum_info = {
            'row': row_data,
            'col': col_data,
            'spectrum': spectrum.copy(),
            'x_data': x_data.copy()
        }
        
        # 获取当前光谱模式
        mode = self.control_panel.combo_spectrum_mode.currentText()
        
        if mode == "单线":
            # 单线模式：清空之前的所有曲线和标记
            self._clear_all_spectra()
            
            # 收集所有需要绘制的光谱（原始 + 预处理）
            spectra_to_plot = []
            
            # 1. 根据复选框决定是否绘制原始光谱
            if self.chk_show_original.isChecked():
                spectra_to_plot.append({
                    'x': x_data,
                    'y': spectrum,
                    'name': f"原始({row_data},{col_data})",
                    'pen': pg.mkPen(self.shared_colors[0], width=3)  # 实线、加粗
                })
            
            # 2. 根据勾选的预处理选项，添加预处理后的光谱（使用组合预处理）
            has_preprocess = any([
                self.chk_smooth_ma.isChecked(),
                self.chk_smooth_sg.isChecked(),
                self.chk_deriv_1st.isChecked(),
                self.chk_deriv_2nd.isChecked(),
                self.chk_normalize.isChecked(),
                self.chk_standardize.isChecked()
            ])
            
            if has_preprocess:
                # 应用所有预处理组合
                processed_spectrum, processed_x = self._get_processed_spectrum(spectrum, x_data)
                if processed_spectrum is not None:
                    # 生成预处理名称
                    preprocess_names = []
                    if self.chk_smooth_ma.isChecked():
                        preprocess_names.append("MA")
                    if self.chk_smooth_sg.isChecked():
                        preprocess_names.append("SG")
                    if self.chk_deriv_1st.isChecked():
                        preprocess_names.append("1阶")
                    if self.chk_deriv_2nd.isChecked():
                        preprocess_names.append("2阶")
                    if self.chk_normalize.isChecked():
                        preprocess_names.append("0-1")
                    if self.chk_standardize.isChecked():
                        preprocess_names.append("Z-score")
                    
                    method_name = "+".join(preprocess_names)
                    spectra_to_plot.append({
                        'x': processed_x,
                        'y': processed_spectrum,
                        'name': f"预处理[{method_name}]",
                        'pen': pg.mkPen(self.shared_colors[1], width=3)  # 实线、加粗
                    })
            
            # 绘制所有光谱
            for spec_info in spectra_to_plot:
                curve = self.plot.plot(
                    spec_info['x'], spec_info['y'],
                    pen=spec_info['pen'],
                    name=spec_info['name']
                )
                self.spectrum_curves.append((curve, spec_info['name'], None))
            
            # 添加图例（如果有多条曲线）
            if len(spectra_to_plot) > 1:
                self.plot.addLegend()
            
        else:  # 多线模式
            # 多线模式：添加新的曲线和像素点标记
            color = self._get_color_for_spectrum()
            
            # 保存光谱信息（用于预处理实时更新）
            spectrum_info = {
                'row': row_data,
                'col': col_data,
                'spectrum': spectrum.copy(),
                'x_data': x_data.copy(),
                'x_view': x_view,
                'y_view': y_view,
                'color': color
            }
            self.multi_line_spectrum_infos.append(spectrum_info)
            
            # 在图像上添加像素点标记
            marker = self._add_pixel_marker(x_view, y_view, color)
            
            # 判断需要绘制哪些曲线
            has_preprocess = any([
                self.chk_smooth_ma.isChecked(),
                self.chk_smooth_sg.isChecked(),
                self.chk_deriv_1st.isChecked(),
                self.chk_deriv_2nd.isChecked(),
                self.chk_normalize.isChecked(),
                self.chk_standardize.isChecked()
            ])
            
            # 如果勾选了"原始"，绘制原始光谱
            if self.chk_show_original.isChecked():
                curve_orig = self.plot.plot(
                    x_data, spectrum,
                    pen=pg.mkPen(color, width=3),  # 实线、加粗
                    name=f"原始-点({row_data},{col_data})"
                )
                self.spectrum_curves.append((curve_orig, f"原始-点({row_data},{col_data})", marker))
            
            # 如果勾选了预处理，绘制预处理后的光谱
            if has_preprocess:
                processed_spectrum, processed_x = self._get_processed_spectrum(spectrum, x_data)
                curve_proc = self.plot.plot(
                    processed_x, processed_spectrum,
                    pen=pg.mkPen(color, width=3),  # 实线、加粗
                    name=f"处理-点({row_data},{col_data})"
                )
                # 如果没有勾选原始，marker 附加到预处理曲线上；否则为 None（marker已附加到原始曲线）
                marker_for_proc = marker if not self.chk_show_original.isChecked() else None
                self.spectrum_curves.append((curve_proc, f"处理-点({row_data},{col_data})", marker_for_proc))
            
            # 如果既没勾选原始也没勾选预处理，至少绘制原始光谱
            if not self.chk_show_original.isChecked() and not has_preprocess:
                curve = self.plot.plot(
                    x_data, spectrum,
                    pen=pg.mkPen(color, width=3),  # 实线、加粗
                    name=f"点({row_data},{col_data})"
                )
                self.spectrum_curves.append((curve, f"点({row_data},{col_data})", marker))
        
        # 更新曲线计数和平均按钮状态
        self._update_spectrum_ui()
    
    def _on_mouse_move(self, pos):
        """处理鼠标移动事件（用于实时更新ROI大小）"""
        if not self.is_drawing_mode or not self.is_mouse_pressed or not self.current_drawing_roi:
            return
        
        if not self.view_box.sceneBoundingRect().contains(pos):
            return
            
        mouse_point = self.view_box.mapSceneToView(pos)
        x = int(mouse_point.x())
        y = int(mouse_point.y())
        
        # 检查坐标有效性
        h, w = self.cube.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        
        # 更新 ROI 大小
        self._update_roi_size(x, y)
    
    # ========== ROI 功能 ==========
    def _toggle_roi(self, checked: bool):
        """切换ROI绘制模式"""
        if checked:
            if self.cube is None:
                QMessageBox.warning(self, "提示", "请先加载数据")
                self.control_panel.btn_roi.setChecked(False)
                return
            
            # 进入绘制模式
            self.is_drawing_mode = True
            self.drawing_shape_type = self.control_panel.combo_roi_shape.currentText()
            self.drawing_start_pos = None
            self.is_mouse_pressed = False
            
            # 禁止图像平移和缩放，防止绘制时图像跟着动
            self.view_box.setMouseEnabled(x=False, y=False)
            
            # 修改鼠标光标提示用户处于绘制模式
            self.view_box.setCursor(Qt.CrossCursor)
        else:
            # 退出绘制模式
            if self.current_drawing_roi and not self.is_mouse_pressed:
                # 如果有未完成的ROI，删除它
                self.view_box.removeItem(self.current_drawing_roi)
                self.current_drawing_roi = None
            
            self.is_drawing_mode = False
            self.drawing_shape_type = None
            self.drawing_start_pos = None
            self.is_mouse_pressed = False
            
            # 恢复图像平移和缩放功能
            self.view_box.setMouseEnabled(x=True, y=True)
            
            self.view_box.setCursor(Qt.ArrowCursor)
    
    def _start_roi_drawing(self, x, y):
        """开始绘制ROI（鼠标按下时）"""
        if self.current_drawing_roi is not None:
            return
        
        # 记录起始位置
        self.drawing_start_pos = (x, y)
        
        # 获取颜色
        color = self._get_color_for_roi()
        
        # 根据形状类型创建 ROI（初始大小为1x1）
        if self.drawing_shape_type == "矩形":
            roi = pg.RectROI([x, y], [1, 1], 
                            pen=pg.mkPen(color, width=2),
                            removable=False)
        elif self.drawing_shape_type == "椭圆":
            roi = pg.EllipseROI([x, y], [1, 1], 
                               pen=pg.mkPen(color, width=2),
                               removable=False)
        elif self.drawing_shape_type == "多边形":
            # 多边形从一个点开始
            roi = pg.PolyLineROI([[x, y], [x+1, y], [x, y+1]], 
                                closed=True, 
                                pen=pg.mkPen(color, width=2),
                                removable=False)
        else:
            return
        
        # 添加到视图
        self.view_box.addItem(roi)
        self.current_drawing_roi = roi
    
    def _update_roi_size(self, x, y):
        """更新ROI大小（鼠标拖动时）"""
        if not self.current_drawing_roi or not self.drawing_start_pos:
            return
        
        x0, y0 = self.drawing_start_pos
        
        # 计算新的位置和大小
        new_x = min(x0, x)
        new_y = min(y0, y)
        new_w = abs(x - x0)
        new_h = abs(y - y0)
        
        # 确保最小尺寸
        new_w = max(new_w, 5)
        new_h = max(new_h, 5)
        
        # 更新ROI
        if self.drawing_shape_type in ["矩形", "椭圆"]:
            self.current_drawing_roi.setPos([new_x, new_y])
            self.current_drawing_roi.setSize([new_w, new_h])
        elif self.drawing_shape_type == "多边形":
            # 更新多边形为矩形的四个顶点
            points = [[new_x, new_y], [new_x + new_w, new_y], 
                     [new_x + new_w, new_y + new_h], [new_x, new_y + new_h]]
            self.current_drawing_roi.setPoints(points)
    
    def _finish_roi_drawing(self):
        """完成当前ROI的绘制"""
        if not self.current_drawing_roi:
            return
        
        roi = self.current_drawing_roi
        shape_type = self.drawing_shape_type
        
        # 弹出命名对话框
        default_name = f"ROI-{len(self.roi_list) + 1} ({shape_type})"
        name, ok = QInputDialog.getText(
            self, "命名ROI", "请输入ROI名称:", text=default_name
        )
        
        if not ok or not name.strip():
            # 用户取消，删除这个 ROI
            self.view_box.removeItem(roi)
            self.current_drawing_roi = None
            self.drawing_start_pos = None
            # 保持绘制模式，允许继续绘制
            return
        
        name = name.strip()
        
        # 获取 ROI 颜色（从 pen 中提取）
        pen = roi.pen
        color = pen.color().name()
        
        # 获取 ROI 区域数据并计算平均光谱
        try:
            # getArrayRegion 返回 ROI 覆盖的数据
            roi_data = roi.getArrayRegion(self.cube, self.image_item, axes=(1, 0))
            
            # 计算平均光谱（沿空间维度平均）
            mean_spectrum = np.nanmean(roi_data, axis=(0, 1)).ravel()  # 确保是1D数组
            
            # 准备 X 轴数据
            if self.wavelengths and len(self.wavelengths) == len(mean_spectrum):
                x_data = np.array([float(w) for w in self.wavelengths])
            else:
                x_data = np.arange(len(mean_spectrum))
            
            # 绘制光谱曲线
            curve = self.plot.plot(x_data, mean_spectrum, 
                                  pen=pg.mkPen(color, width=2), 
                                  name=name)
            
            # 保存 ROI 信息
            roi_info = {
                'name': name,
                'roi_object': roi,
                'curve': curve,
                'shape_type': shape_type,
                'color': color
            }
            self.roi_list.append(roi_info)
            
            # 更新列表显示
            from PySide6.QtWidgets import QListWidgetItem
            from PySide6.QtGui import QColor
            item = QListWidgetItem(f"■ {name}")
            item.setForeground(QColor(color))
            self.control_panel.list_rois.addItem(item)
            
            # 启用按钮
            self.control_panel.btn_delete_roi.setEnabled(True)
            self.control_panel.btn_clear_roi.setEnabled(True)
            self.control_panel.btn_export_roi.setEnabled(True)
            
            # 重置状态，允许继续绘制下一个ROI
            self.current_drawing_roi = None
            self.drawing_start_pos = None
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"ROI处理失败:\n{str(e)}")
            self.view_box.removeItem(roi)
            self.current_drawing_roi = None
            self.drawing_start_pos = None
    
    def _update_all_roi_spectra(self):
        """更新所有ROI的光谱（波段切换后）"""
        for roi_info in self.roi_list:
            try:
                roi = roi_info['roi_object']
                curve = roi_info['curve']
                
                # 重新获取ROI数据
                roi_data = roi.getArrayRegion(self.cube, self.image_item, axes=(1, 0))
                mean_spectrum = np.nanmean(roi_data, axis=(0, 1)).ravel()  # 确保是1D数组
                
                # 更新曲线数据
                if self.wavelengths and len(self.wavelengths) == len(mean_spectrum):
                    x_data = np.array([float(w) for w in self.wavelengths])
                else:
                    x_data = np.arange(len(mean_spectrum))
                
                curve.setData(x_data, mean_spectrum)
            except:
                pass
    
    def _delete_roi(self, idx: int):
        """删除指定ROI"""
        if 0 <= idx < len(self.roi_list):
            roi_info = self.roi_list[idx]
            
            # 释放颜色
            if 'color' in roi_info:
                self._release_color(roi_info['color'], 'roi')
            
            # 从视图中删除ROI对象
            self.view_box.removeItem(roi_info['roi_object'])
            
            # 删除光谱曲线
            self.plot.removeItem(roi_info['curve'])
            
            # 从列表中删除
            self.roi_list.pop(idx)
            self.control_panel.list_rois.takeItem(idx)
            
            # 如果没有ROI了，禁用按钮
            if len(self.roi_list) == 0:
                self.control_panel.btn_delete_roi.setEnabled(False)
                self.control_panel.btn_clear_roi.setEnabled(False)
                self.control_panel.btn_export_roi.setEnabled(False)
    
    def _clear_all_rois(self):
        """清空所有ROI"""
        while len(self.roi_list) > 0:
            self._delete_roi(0)
    
    def _highlight_roi(self, idx: int):
        """高亮选中的ROI"""
        if idx < 0 or idx >= len(self.roi_list):
            return
        
        # 重置所有ROI的线宽
        for i, roi_info in enumerate(self.roi_list):
            roi = roi_info['roi_object']
            color = roi_info['color']
            if i == idx:
                roi.setPen(pg.mkPen(color, width=4))  # 加粗
            else:
                roi.setPen(pg.mkPen(color, width=2))  # 正常
    
    def _export_all_rois(self):
        """导出所有ROI光谱"""
        if len(self.roi_list) == 0:
            QMessageBox.warning(self, "提示", "没有可导出的ROI")
            return
        
        # 选择导出格式
        file_filter = "CSV文件 (*.csv);;MAT文件 (*.mat);;NumPy文件 (*.npy)"
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "导出所有ROI光谱", "", file_filter
        )
        
        if not path:
            return
        
        try:
            if "CSV" in selected_filter:
                self._export_rois_csv(path)
            elif "MAT" in selected_filter:
                self._export_rois_mat(path)
            elif "NumPy" in selected_filter:
                self._export_rois_npy(path)
            
            QMessageBox.information(self, "成功", 
                f"已导出 {len(self.roi_list)} 个ROI光谱到:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败:\n{str(e)}")
    
    def _export_rois_csv(self, path: str):
        """导出为CSV格式"""
        import csv
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入表头
            header = ['Wavelength/Band']
            for roi_info in self.roi_list:
                header.append(roi_info['name'])
            writer.writerow(header)
            
            # 写入数据
            n_bands = self.cube.shape[2]
            for i in range(n_bands):
                row = []
                
                # 第一列：波长或波段号
                if self.wavelengths and i < len(self.wavelengths):
                    row.append(self.wavelengths[i])
                else:
                    row.append(i)
                
                # 后续列：各ROI的平均光谱
                for roi_info in self.roi_list:
                    roi = roi_info['roi_object']
                    roi_data = roi.getArrayRegion(self.cube[:,:,i], self.image_item, axes=(1, 0))
                    mean_val = np.nanmean(roi_data)
                    row.append(mean_val)
                
                writer.writerow(row)
    
    def _export_rois_mat(self, path: str):
        """导出为MAT格式"""
        import scipy.io
        
        data_dict = {}
        
        # 添加波长信息
        if self.wavelengths:
            data_dict['wavelengths'] = np.array([float(w) for w in self.wavelengths])
        
        # 添加每个ROI的光谱
        for roi_info in self.roi_list:
            roi = roi_info['roi_object']
            roi_data = roi.getArrayRegion(self.cube, self.image_item, axes=(1, 0))
            mean_spec = np.nanmean(roi_data, axis=(0, 1)).ravel()  # 确保是1D数组
            
            # MAT文件中的变量名不能有特殊字符
            safe_name = roi_info['name'].replace('-', '_').replace(' ', '_')
            data_dict[safe_name] = mean_spec
        
        scipy.io.savemat(path, data_dict)
    
    def _export_rois_npy(self, path: str):
        """导出为NPY格式"""
        data_dict = {}
        
        # 添加波长信息
        if self.wavelengths:
            data_dict['wavelengths'] = np.array([float(w) for w in self.wavelengths])
        
        # 添加每个ROI的光谱
        for roi_info in self.roi_list:
            roi = roi_info['roi_object']
            roi_data = roi.getArrayRegion(self.cube, self.image_item, axes=(1, 0))
            mean_spec = np.nanmean(roi_data, axis=(0, 1)).ravel()  # 确保是1D数组
            data_dict[roi_info['name']] = mean_spec
        
        np.save(path, data_dict, allow_pickle=True)
    
    # ========== 统一颜色管理功能 ==========
    def _get_next_color(self, usage_type='spectrum'):
        """获取下一个可用颜色
        
        Args:
            usage_type: 'roi' 或 'spectrum'
            
        Returns:
            color: 十六进制颜色字符串
        """
        # 查找未使用的颜色
        for color in self.shared_colors:
            if color not in self.color_usage:
                self.color_usage[color] = usage_type
                return color
        
        # 如果所有颜色都被使用，查找只被另一种类型使用的颜色
        for color in self.shared_colors:
            if self.color_usage[color] != usage_type:
                self.color_usage[color] = 'both'
                return color
        
        # 如果所有颜色都被两种类型使用，循环使用
        color_idx = len(self.color_usage) % len(self.shared_colors)
        return self.shared_colors[color_idx]
    
    def _release_color(self, color, usage_type):
        """释放颜色使用
        
        Args:
            color: 要释放的颜色
            usage_type: 'roi' 或 'spectrum'
        """
        if color in self.color_usage:
            if self.color_usage[color] == 'both':
                # 如果被两种类型使用，只移除当前类型
                self.color_usage[color] = 'roi' if usage_type == 'spectrum' else 'spectrum'
            elif self.color_usage[color] == usage_type:
                # 如果只被当前类型使用，完全移除
                del self.color_usage[color]
    
    def _get_color_for_roi(self):
        """为ROI获取颜色"""
        return self._get_next_color('roi')
    
    def _get_color_for_spectrum(self):
        """为光谱获取颜色"""
        return self._get_next_color('spectrum')
    
    # ========== 光谱管理功能 ==========
    def _add_pixel_marker(self, x, y, color):
        """在图像上添加像素点标记
        
        Args:
            x: 像素X坐标（整数）
            y: 像素Y坐标（整数）
            color: 标记颜色（十六进制字符串）
            
        Returns:
            marker: QGraphicsRectItem
        """
        # 创建一个矩形来填充整个像素
        from PySide6.QtWidgets import QGraphicsRectItem
        from PySide6.QtGui import QColor, QPen, QBrush
        
        marker = QGraphicsRectItem(x, y, 1, 1)  # (x, y, width, height)
        
        # 设置填充颜色（半透明）
        color_obj = QColor(color)
        color_obj.setAlpha(128)  # 半透明
        marker.setBrush(QBrush(color_obj))
        
        # 设置边框（稍微深一点的颜色）
        pen_color = QColor(color)
        pen_color.setAlpha(200)
        marker.setPen(QPen(pen_color, 0.1))  # 细边框
        
        # 添加到图像视图
        self.view_box.addItem(marker)
        
        return marker
    
    def _clear_all_spectra(self):
        """清空所有光谱曲线和像素点标记"""
        # 清除所有曲线和标记
        for item in self.spectrum_curves:
            if len(item) >= 3:
                curve, name, marker = item[0], item[1], item[2] if len(item) > 2 else None
            else:
                curve, name = item[0], item[1]
                marker = None
                
            # 清除曲线
            self.plot.removeItem(curve)
            
            # 清除标记点
            if marker is not None:
                self.view_box.removeItem(marker)
        
        # 释放所有光谱使用的颜色
        for color in list(self.color_usage.keys()):
            if self.color_usage[color] in ['spectrum', 'both']:
                self._release_color(color, 'spectrum')
        
        self.spectrum_curves.clear()
        self.spectrum_markers.clear()
        
        # 清除单点光谱
        if self.point_spectrum_curve:
            self.plot.removeItem(self.point_spectrum_curve)
            self.point_spectrum_curve = None
        
        # 清空多线模式保存的光谱信息
        self.multi_line_spectrum_infos.clear()
        
        # 更新UI
        self._update_spectrum_ui()
    
    def _average_spectra(self):
        """计算并显示所有光谱的平均值"""
        if len(self.spectrum_curves) < 2:
            return
        
        # 收集所有光谱数据（排除已有的平均曲线）
        all_spectra = []
        for item in self.spectrum_curves:
            curve = item[0]
            name = item[1]
            if name != "平均光谱":  # 排除已有的平均曲线
                x_data, y_data = curve.getData()
                all_spectra.append(y_data)
        
        if not all_spectra:
            return
        
        # 计算平均值
        avg_spectrum = np.nanmean(all_spectra, axis=0)
        
        # 获取 x 轴数据
        x_data, _ = self.spectrum_curves[0][0].getData()
        
        # 绘制平均曲线（加粗显示，黑色）
        avg_curve = self.plot.plot(
            x_data, avg_spectrum,
            pen=pg.mkPen('#000000', width=3, style=Qt.SolidLine),
            name="平均光谱"
        )
        
        # 平均曲线不添加标记点
        self.spectrum_curves.append((avg_curve, "平均光谱", None))
        
        # 更新UI
        self._update_spectrum_ui()
    
    def _update_spectrum_ui(self):
        """更新光谱相关的UI状态"""
        count = len(self.spectrum_curves)
        
        # 更新曲线计数（如果控制面板有这个标签）
        if hasattr(self.control_panel, 'label_spectrum_count'):
            self.control_panel.label_spectrum_count.setText(f"曲线数: {count}")
        
        # 更新平均按钮状态
        mode = self.control_panel.combo_spectrum_mode.currentText()
        if mode == "多线" and count >= 2:
            self.control_panel.btn_avg_spectra.setEnabled(True)
        else:
            self.control_panel.btn_avg_spectra.setEnabled(False)
    
    def _redraw_spectrum_with_preprocess(self):
        """
        当预处理选项变化时，重新绘制光谱
        支持所有模式（单线、多线、ROI等）
        """
        # 检查条件
        if self.last_spectrum_info is None:
            return
        
        mode = self.control_panel.combo_spectrum_mode.currentText()
        
        # 单线模式：重新绘制最后一个像素的光谱
        if mode == "单线":
            self._redraw_single_line_spectrum()
        # 多线模式：重新绘制所有已保存的光谱
        elif mode == "多线":
            self._redraw_multi_line_spectra()
    
    def _get_processed_spectrum(self, spectrum, x_data):
        """显示和导出使用同一组预处理规则。"""
        return preprocess_spectrum(spectrum, x_data, selected_preprocess_steps(self))

    def _redraw_single_line_spectrum(self):
        """重新绘制单线模式的光谱"""
        if self.last_spectrum_info is None:
            return
        
        # 获取上次保存的光谱信息
        row_data = self.last_spectrum_info['row']
        col_data = self.last_spectrum_info['col']
        spectrum = self.last_spectrum_info['spectrum']
        x_data = self.last_spectrum_info['x_data']
        
        # 清空之前的所有曲线
        self._clear_all_spectra()
        
        # 收集所有需要绘制的光谱（原始 + 预处理）
        spectra_to_plot = []
        
        # 1. 根据复选框决定是否绘制原始光谱
        if self.chk_show_original.isChecked():
            spectra_to_plot.append({
                'x': x_data,
                'y': spectrum,
                'name': f"原始({row_data},{col_data})",
                'pen': pg.mkPen(self.shared_colors[0], width=3)  # 实线、加粗
            })
        
        # 2. 根据勾选的预处理选项，添加预处理后的光谱（使用组合预处理）
        has_preprocess = any([
            self.chk_smooth_ma.isChecked(),
            self.chk_smooth_sg.isChecked(),
            self.chk_deriv_1st.isChecked(),
            self.chk_deriv_2nd.isChecked(),
            self.chk_normalize.isChecked(),
            self.chk_standardize.isChecked()
        ])
        
        if has_preprocess:
            # 应用所有预处理组合
            processed_spectrum, processed_x = self._get_processed_spectrum(spectrum, x_data)
            if processed_spectrum is not None:
                # 生成预处理名称
                preprocess_names = []
                if self.chk_smooth_ma.isChecked():
                    preprocess_names.append("MA")
                if self.chk_smooth_sg.isChecked():
                    preprocess_names.append("SG")
                if self.chk_deriv_1st.isChecked():
                    preprocess_names.append("1阶")
                if self.chk_deriv_2nd.isChecked():
                    preprocess_names.append("2阶")
                if self.chk_normalize.isChecked():
                    preprocess_names.append("0-1")
                if self.chk_standardize.isChecked():
                    preprocess_names.append("Z-score")
                
                method_name = "+".join(preprocess_names)
                spectra_to_plot.append({
                    'x': processed_x,
                    'y': processed_spectrum,
                    'name': f"预处理[{method_name}]",
                    'pen': pg.mkPen(self.shared_colors[1], width=3)  # 实线、加粗
                })
        
        # 绘制所有光谱
        for spec_info in spectra_to_plot:
            curve = self.plot.plot(
                spec_info['x'], spec_info['y'],
                pen=spec_info['pen'],
                name=spec_info['name']
            )
            self.spectrum_curves.append((curve, spec_info['name'], None))
        
        # 添加图例（如果有多条曲线）
        if len(spectra_to_plot) > 1:
            self.plot.addLegend()
        
        # 更新UI
        self._update_spectrum_ui()
    
    def _redraw_multi_line_spectra(self):
        """重新绘制多线模式下的所有光谱（与单线模式逻辑一致）"""
        if not self.multi_line_spectrum_infos:
            return
        
        # 清空之前的所有曲线和标记点
        for curve, _, marker in self.spectrum_curves:
            self.plot.removeItem(curve)
            # 清除标记点
            if marker is not None:
                self.view_box.removeItem(marker)
        self.spectrum_curves.clear()
        
        # 判断需要绘制哪些曲线
        show_original = self.chk_show_original.isChecked()
        has_preprocess = any([
            self.chk_smooth_ma.isChecked(),
            self.chk_smooth_sg.isChecked(),
            self.chk_deriv_1st.isChecked(),
            self.chk_deriv_2nd.isChecked(),
            self.chk_normalize.isChecked(),
            self.chk_standardize.isChecked()
        ])
        
        # 重新绘制所有保存的光谱
        for info in self.multi_line_spectrum_infos:
            row_data = info['row']
            col_data = info['col']
            spectrum = info['spectrum']
            x_data = info['x_data']
            x_view = info['x_view']
            y_view = info['y_view']
            color = info['color']
            
            # 重新添加像素点标记
            marker = self._add_pixel_marker(x_view, y_view, color)
            
            # 如果勾选了"原始"，绘制原始光谱
            if show_original:
                curve_orig = self.plot.plot(
                    x_data, spectrum,
                    pen=pg.mkPen(color, width=3),  # 实线、加粗
                    name=f"原始-点({row_data},{col_data})"
                )
                self.spectrum_curves.append((curve_orig, f"原始-点({row_data},{col_data})", marker))
            
            # 如果勾选了预处理，绘制预处理后的光谱
            if has_preprocess:
                processed_spectrum, processed_x = self._get_processed_spectrum(spectrum, x_data)
                curve_proc = self.plot.plot(
                    processed_x, processed_spectrum,
                    pen=pg.mkPen(color, width=3),  # 实线、加粗
                    name=f"处理-点({row_data},{col_data})"
                )
                # 如果没有勾选原始，marker 附加到预处理曲线上；否则为 None（marker已附加到原始曲线）
                marker_for_proc = marker if not show_original else None
                self.spectrum_curves.append((curve_proc, f"处理-点({row_data},{col_data})", marker_for_proc))
            
            # 如果既没勾选原始也没勾选预处理，至少绘制原始光谱
            if not show_original and not has_preprocess:
                curve = self.plot.plot(
                    x_data, spectrum,
                    pen=pg.mkPen(color, width=3),  # 实线、加粗
                    name=f"点({row_data},{col_data})"
                )
                self.spectrum_curves.append((curve, f"点({row_data},{col_data})", marker))
        
        # 添加图例（如果有多条曲线）
        if len(self.spectrum_curves) > 1:
            self.plot.addLegend()
        
        # 更新UI
        self._update_spectrum_ui()
    
    def _on_spectrum_mode_changed(self, mode):
        """处理光谱模式切换"""
        # 当切换到多线模式时，清空之前单线模式的光谱曲线
        if mode == "多线":
            self._clear_all_spectra()
            # 清空单线模式保存的光谱信息
            self.last_spectrum_info = None
        
        # 更新UI状态
        self._update_spectrum_ui()
    
    def _apply_preprocess(self, spectrum, x_data, method_name):
        """对光谱应用单项预处理，不修改原始数据。"""
        return preprocess_spectrum(spectrum, x_data, [method_name])

    def _apply_axis_range(self):
        """应用坐标轴范围设置"""
        try:
            # 获取输入的范围值（现在从光谱图下方的控件获取）
            xmin_text = self.edit_xmin.text().strip()
            xmax_text = self.edit_xmax.text().strip()
            ymin_text = self.edit_ymin.text().strip()
            ymax_text = self.edit_ymax.text().strip()
            
            # 解析X轴范围
            if xmin_text or xmax_text:
                if xmin_text and xmax_text:
                    xmin = float(xmin_text)
                    xmax = float(xmax_text)
                    if xmin >= xmax:
                        QMessageBox.warning(self, "范围错误", "xMin 必须小于 xMax")
                        return
                    self.plot.setXRange(xmin, xmax, padding=0)
                elif xmin_text:
                    xmin = float(xmin_text)
                    self.plot.setXRange(xmin, None, padding=0)
                else:
                    xmax = float(xmax_text)
                    self.plot.setXRange(None, xmax, padding=0)
            else:
                # 如果都为空，恢复自动范围
                self.plot.enableAutoRange(axis='x')
            
            # 解析Y轴范围
            if ymin_text or ymax_text:
                if ymin_text and ymax_text:
                    ymin = float(ymin_text)
                    ymax = float(ymax_text)
                    if ymin >= ymax:
                        QMessageBox.warning(self, "范围错误", "yMin 必须小于 yMax")
                        return
                    self.plot.setYRange(ymin, ymax, padding=0)
                elif ymin_text:
                    ymin = float(ymin_text)
                    self.plot.setYRange(ymin, None, padding=0)
                else:
                    ymax = float(ymax_text)
                    self.plot.setYRange(None, ymax, padding=0)
            else:
                # 如果都为空，恢复自动范围
                self.plot.enableAutoRange(axis='y')
                
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的数值")
    
    def _on_spectrum_mouse_move(self, pos):
        """处理光谱图上的鼠标移动事件"""
        # 检查鼠标是否在光谱图的绘图区域内
        if self.plot.plotItem.vb.sceneBoundingRect().contains(pos):
            # 将场景坐标转换为数据坐标
            mouse_point = self.plot.plotItem.vb.mapSceneToView(pos)
            x = mouse_point.x()
            y = mouse_point.y()
            
            # 更新十字线位置
            self.vLine.setPos(x)
            self.hLine.setPos(y)
            
            # 显示坐标信息
            if self.wavelengths and len(self.wavelengths) > 0:
                # 如果有波长信息，显示波长
                text = f"波长: {x:.2f} nm\n反射率: {y:.4f}"
            else:
                # 否则显示波段号
                text = f"波段: {int(x)}\n反射率: {y:.4f}"
            
            self.spectrum_coord_label.setText(text)
            self.spectrum_coord_label.setPos(x, y)
            self.spectrum_coord_label.show()
        else:
            # 鼠标离开图表区域，隐藏十字线和标签
            self.spectrum_coord_label.hide()
    
    # ========== 分割功能 ==========
    def _on_segment_requested(self):
        """执行分割（基于 extract_grains.py 的完整流程）"""
        if self.cube is None:
            QMessageBox.warning(self, "提示", "请先打开数据")
            return
        
        from modules.segmentation import segment_seeds
        
        # 获取分割方法
        method_map = {"Otsu阈值": "otsu", "Watershed": "watershed"}
        method = method_map[self.control_panel.combo_method.currentText()]
        
        # 获取分割参数（从输入框，支持留空使用默认值）
        try:
            min_area = int(self.control_panel.edit_min_area.text()) if self.control_panel.edit_min_area.text() else 150
        except ValueError:
            min_area = 150
        
        try:
            max_area = int(self.control_panel.edit_max_area.text()) if self.control_panel.edit_max_area.text() else 1000
        except ValueError:
            max_area = 1000
        
        # 获取阈值（支持手动设置）
        threshold = None
        threshold_text = self.control_panel.edit_threshold.text().strip()
        if threshold_text:
            try:
                threshold = float(threshold_text)
                if not (0 <= threshold <= 1):
                    QMessageBox.warning(self, "参数错误", "阈值必须在 0-1 之间")
                    return
            except ValueError:
                QMessageBox.warning(self, "参数错误", "阈值必须是数字")
                return
        
        # 验证参数
        if min_area >= max_area:
            QMessageBox.warning(self, "参数错误", "最小面积必须小于最大面积")
            return
        
        # 提取当前波段图像
        img = self.cube[:, :, self.band_idx]
        
        # 获取排序模式
        order_mode = self.control_panel.combo_order_mode.currentText()
        sort_mode = "顺序" if order_mode == "顺序" else "乱序"
        
        # 使用网格优化算法（固定）

        # 执行完整分割流程（包含ROI、形态学、排序）
        try:
            self.labels, self.regions, info = segment_seeds(
                img,
                method=method,
                min_area=min_area,
                max_area=max_area,
                threshold=threshold,  # 支持手动设置
                roi=None,  # 可通过ROI工具设置
                min_distance=15,  # 分水岭中心点间距
                closing_r=2,  # 形态学闭运算半径（参考extract_grains.py）
                opening_r=1,   # 形态学开运算半径（参考extract_grains.py）
                sort_mode=sort_mode  # 排序模式
            )
            
            # 检查是否检测到有效区域
            if not self.regions or len(self.regions) == 0:
                QMessageBox.warning(self, "分割结果", 
                    "未检测到有效的种子区域。\n\n"
                    "可能原因：\n"
                    "• 图像质量不佳或对比度低\n"
                    "• 分割参数设置不当\n"
                    "• 图像中没有明显的种子\n\n"
                    "建议：\n"
                    "• 调整最小/最大面积参数\n"
                    "• 尝试不同的分割方法\n"
                    "• 检查图像是否清晰")
                return
                
        except Exception as e:
            QMessageBox.critical(self, "分割错误", f"分割过程中发生错误：\n{str(e)}")
            return
        
        # 更新显示
        self.update_image()
        
        # 构建简洁的统计信息
        msg = self._build_segmentation_message(method, order_mode, min_area, max_area, threshold, info)
        
        # 保存分割信息并启用可视化按钮
        self.segmentation_info = info
        self.control_panel.btn_show_pipeline.setEnabled(True)
        
        QMessageBox.information(self, "分割完成", msg)
    
    def _build_segmentation_message(self, method, order_mode, min_area, max_area, threshold, info):
        """构建分割结果消息（高内聚：消息构建逻辑独立）"""
        msg = f"✅ 分割完成\n\n"
        msg += f"📊 参数设置:\n"
        msg += f"  • 方法: {method}\n"
        msg += f"  • 排序模式: {order_mode}\n"
        msg += f"  • 面积范围: {min_area} - {max_area} px\n"
        msg += f"  • 阈值: {info['threshold']:.4f} {'(手动)' if threshold else '(自动)'}\n\n"
        msg += f"🎯 检测结果:\n"
        msg += f"  • 检测到颗粒: {info['final_regions']} 个\n"
        if method == "watershed":
            msg += f"  • 种子中心: {info['seed_centers']} 个\n"
        msg += f"  • 初始区域: {info['initial_regions']}\n"
        msg += f"  • 过滤后: {info['final_regions']}\n\n"
        
        # 智能提示
        if info['final_regions'] < 30:
            msg += "⚠️ 提示：检测数量较少，可尝试：\n"
            msg += "  • 降低阈值（如 0.05-0.08）\n"
            msg += "  • 减小最小面积（如 100）\n"
            if order_mode == "顺序":
                msg += "  • 切换到乱序模式可能检测到更多颗粒\n"
        elif info['final_regions'] > 50:
            msg += "⚠️ 提示：检测数量较多，可尝试：\n"
            msg += "  • 提高阈值\n"
            msg += "  • 增大最小面积\n"
            if order_mode == "乱序":
                msg += "  • 切换到顺序模式可获得更好的空间排列\n"
        
        return msg
    
    def _show_pipeline(self):
        """显示分割流程可视化窗口"""
        if self.labels is None or self.segmentation_info is None:
            QMessageBox.warning(self, "提示", "请先执行分割")
            return
        
        from ui.segmentation_viewer import SegmentationViewer
        viewer = SegmentationViewer(self.labels, self.segmentation_info, self.band_idx, self)
        viewer.exec()
    
    # ========== 图像变换功能 ==========
    def _rotate_left(self):
        """逆时针旋转90度"""
        if self.cube is None:
            return
        
        # 旋转数据立方体 (H, W, C) -> (W, H, C)，然后翻转
        self.cube = np.rot90(self.cube, k=1, axes=(0, 1))
        
        # 重置分割结果和ROI
        self._reset_analysis_results()
        
        self.update_image()
    
    def _rotate_right(self):
        """顺时针旋转90度"""
        if self.cube is None:
            return
        
        # 旋转数据立方体
        self.cube = np.rot90(self.cube, k=-1, axes=(0, 1))
        
        # 重置分割结果和ROI
        self._reset_analysis_results()
        
        self.update_image()
    
    def _flip_ud(self):
        """上下镜像（垂直翻转）"""
        if self.cube is None:
            return
        
        self.cube = np.flipud(self.cube)
        self._reset_analysis_results()
        self.update_image()
    
    def _flip_lr(self):
        """左右镜像（水平翻转）"""
        if self.cube is None:
            return
        
        self.cube = np.fliplr(self.cube)
        self._reset_analysis_results()
        self.update_image()
    
    def _reset_analysis_results(self):
        """重置所有分析结果（高内聚：统一管理结果重置）"""
        # 重置分割结果
        self.labels = None
        self.regions = None
        self.segmentation_info = None
        self.control_panel.btn_show_pipeline.setEnabled(False)
        
        # 清除所有ROI
        self._clear_all_rois()

