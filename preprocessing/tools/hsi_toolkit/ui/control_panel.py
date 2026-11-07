"""
左侧控制面板
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QSlider, QSpinBox, 
    QHBoxLayout, QPushButton, QComboBox, QCheckBox, QLineEdit, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIntValidator
from ui.dataset_panel import DatasetPanel

class ControlPanel(QWidget):
    """左侧参数控制面板"""
    
    band_changed = Signal(int)
    method_changed = Signal(str)
    segment_requested = Signal()
    
    # ROI 管理信号
    roi_delete_requested = Signal(int)  # 删除指定索引的ROI
    roi_clear_requested = Signal()      # 清空所有ROI
    roi_export_requested = Signal()     # 导出所有ROI
    roi_item_selected = Signal(int)     # 选中ROI项
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ControlPanel")
        
        # 响应式布局设置
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)  # 改为Minimum，允许滚动
        self.setMinimumWidth(200)  # 最小宽度保证基本可用性
        self.setMaximumWidth(500)  # 最大宽度避免过宽
        
        layout = QVBoxLayout(self)
        layout.setSpacing(8)  # 增加间距
        layout.setContentsMargins(8, 8, 8, 8)  # 增加边距
        
        # 设置布局的拉伸因子，让组件能够自适应
        layout.setStretchFactor(layout, 1)
        
        # ========== 数据集管理 ==========
        self.dataset_panel = DatasetPanel()
        self.dataset_panel.setMaximumHeight(220)
        layout.addWidget(self.dataset_panel)
        
        # ========== 波段控制 ==========
        group_band = QGroupBox("🎚 波段控制")
        group_band_layout = QVBoxLayout()
        
        # 显示模式选择（移到最前面）
        h_displaymode = QHBoxLayout()
        h_displaymode.addWidget(QLabel("显示模式:"))
        self.combo_display_mode = QComboBox()
        self.combo_display_mode.addItems(["灰度图", "RGB合成", "真彩色(B)", "假彩色(NIR-R-G)"])
        self.combo_display_mode.setToolTip(
            "灰度图：单波段显示\n"
            "RGB合成：手动选择三个波段合成彩色图像\n"
            "真彩色(B)：CMF积分近似的人眼真彩预览\n"
            "假彩色(NIR-R-G)：农业常用植被增强预览"
        )
        self.combo_display_mode.currentTextChanged.connect(self._on_display_mode_changed)
        h_displaymode.addWidget(self.combo_display_mode)
        group_band_layout.addLayout(h_displaymode)
        
        # ========== 灰度图控件（默认显示）==========
        self.gray_controls = QWidget()
        gray_layout = QVBoxLayout(self.gray_controls)
        gray_layout.setContentsMargins(0, 0, 0, 0)
        gray_layout.setSpacing(4)
        
        # 波段输入框
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("波段:"))
        self.edit_band = QLineEdit()
        self.edit_band.setEnabled(False)
        self.edit_band.setPlaceholderText("输入波段号")
        self.edit_band.returnPressed.connect(self._on_band_input)
        h1.addWidget(self.edit_band)
        gray_layout.addLayout(h1)
        
        # 滑块
        self.slider_band = QSlider(Qt.Horizontal)
        self.slider_band.setEnabled(False)
        self.slider_band.valueChanged.connect(self._on_slider_changed)
        gray_layout.addWidget(self.slider_band)
        
        # 波长显示
        self.label_wavelength = QLabel("波长: - nm")
        gray_layout.addWidget(self.label_wavelength)
        
        # 颜色映射选择
        h_colormode = QHBoxLayout()
        h_colormode.addWidget(QLabel("颜色映射:"))
        self.combo_colormode = QComboBox()
        self.combo_colormode.addItems(["灰度", "RGB伪彩色", "热力图", "光谱色"])
        self.combo_colormode.setToolTip("选择单波段图像的颜色映射")
        h_colormode.addWidget(self.combo_colormode)
        gray_layout.addLayout(h_colormode)
        
        group_band_layout.addWidget(self.gray_controls)
        
        # ========== RGB合成控件（初始隐藏）==========
        self.rgb_controls = QWidget()
        rgb_layout = QVBoxLayout(self.rgb_controls)
        rgb_layout.setContentsMargins(0, 0, 0, 0)
        rgb_layout.setSpacing(4)
        
        # R波段
        h_r = QHBoxLayout()
        h_r.addWidget(QLabel("R:"))
        self.spin_r = QSpinBox()
        self.spin_r.setEnabled(False)
        self.spin_r.setToolTip("红色通道波段")
        h_r.addWidget(self.spin_r)
        self.slider_r = QSlider(Qt.Horizontal)
        self.slider_r.setEnabled(False)
        self.slider_r.valueChanged.connect(lambda v: self.spin_r.setValue(v))
        self.spin_r.valueChanged.connect(lambda v: self.slider_r.setValue(v))
        h_r.addWidget(self.slider_r, stretch=1)
        self.label_wl_r = QLabel("-")
        self.label_wl_r.setMinimumWidth(60)
        h_r.addWidget(self.label_wl_r)
        rgb_layout.addLayout(h_r)
        
        # G波段
        h_g = QHBoxLayout()
        h_g.addWidget(QLabel("G:"))
        self.spin_g = QSpinBox()
        self.spin_g.setEnabled(False)
        self.spin_g.setToolTip("绿色通道波段")
        h_g.addWidget(self.spin_g)
        self.slider_g = QSlider(Qt.Horizontal)
        self.slider_g.setEnabled(False)
        self.slider_g.valueChanged.connect(lambda v: self.spin_g.setValue(v))
        self.spin_g.valueChanged.connect(lambda v: self.slider_g.setValue(v))
        h_g.addWidget(self.slider_g, stretch=1)
        self.label_wl_g = QLabel("-")
        self.label_wl_g.setMinimumWidth(60)
        h_g.addWidget(self.label_wl_g)
        rgb_layout.addLayout(h_g)
        
        # B波段
        h_b = QHBoxLayout()
        h_b.addWidget(QLabel("B:"))
        self.spin_b = QSpinBox()
        self.spin_b.setEnabled(False)
        self.spin_b.setToolTip("蓝色通道波段")
        h_b.addWidget(self.spin_b)
        self.slider_b = QSlider(Qt.Horizontal)
        self.slider_b.setEnabled(False)
        self.slider_b.valueChanged.connect(lambda v: self.spin_b.setValue(v))
        self.spin_b.valueChanged.connect(lambda v: self.slider_b.setValue(v))
        h_b.addWidget(self.slider_b, stretch=1)
        self.label_wl_b = QLabel("-")
        self.label_wl_b.setMinimumWidth(60)
        h_b.addWidget(self.label_wl_b)
        rgb_layout.addLayout(h_b)
        
        # 快速预设按钮
        h_preset = QHBoxLayout()
        self.btn_rgb_preset = QPushButton("📐 预设")
        self.btn_rgb_preset.setEnabled(False)
        self.btn_rgb_preset.setToolTip("快速选择常用RGB波段组合")
        self.btn_rgb_preset.clicked.connect(self._show_rgb_presets)
        h_preset.addWidget(self.btn_rgb_preset)
        h_preset.addStretch()
        rgb_layout.addLayout(h_preset)
        
        self.rgb_controls.setVisible(False)  # 初始隐藏
        group_band_layout.addWidget(self.rgb_controls)
        
        group_band.setLayout(group_band_layout)
        layout.addWidget(group_band)
        
        # ========== 分割参数 ==========
        group_seg = QGroupBox("✂️ 分割参数")
        group_seg.setCheckable(True)
        group_seg.setChecked(True)  # 默认展开
        seg_layout = QVBoxLayout()
        
        # 使用响应式布局创建方法选择
        self.combo_method = QComboBox()
        self.combo_method.addItems(["Otsu阈值", "Watershed"])
        self.combo_method.currentTextChanged.connect(self.method_changed)
        seg_layout.addLayout(self._create_responsive_layout("方法:", self.combo_method))

        # 使用响应式布局创建排序模式选择
        self.combo_order_mode = QComboBox()
        self.combo_order_mode.addItems(["顺序", "乱序"])
        self.combo_order_mode.setToolTip("顺序：按空间位置排序（从上到下、从左到右）\n乱序：保持原始分割顺序")
        seg_layout.addLayout(self._create_responsive_layout("排序模式:", self.combo_order_mode))
        
        # 网格优化算法（固定使用，无需选择）
        grid_label = QLabel("排序算法: 网格优化")
        grid_label.setStyleSheet("color: #666; font-style: italic;")
        seg_layout.addWidget(grid_label)
        
        # 面积参数（最小和最大放在一行）
        area_layout = QHBoxLayout()
        area_layout.setContentsMargins(0, 2, 0, 2)
        
        # 最小面积
        min_area_label = QLabel("最小面积:")
        min_area_label.setMinimumWidth(60)
        min_area_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        area_layout.addWidget(min_area_label)
        
        self.edit_min_area = QLineEdit("100")
        self.edit_min_area.setValidator(QIntValidator(10, 1000))
        self.edit_min_area.setPlaceholderText("10-1000")
        self.edit_min_area.setToolTip("最小颗粒面积（像素）")
        self.edit_min_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        area_layout.addWidget(self.edit_min_area)
        
        # 最大面积
        max_area_label = QLabel("最大面积:")
        max_area_label.setMinimumWidth(60)
        max_area_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        area_layout.addWidget(max_area_label)
        
        self.edit_max_area = QLineEdit("1000")
        self.edit_max_area.setValidator(QIntValidator(100, 5000))
        self.edit_max_area.setPlaceholderText("100-5000")
        self.edit_max_area.setToolTip("最大颗粒面积（像素）")
        self.edit_max_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        area_layout.addWidget(self.edit_max_area)
        
        seg_layout.addLayout(area_layout)
        
        # 使用响应式布局创建阈值输入
        self.edit_threshold = QLineEdit("")
        self.edit_threshold.setPlaceholderText("自动")
        self.edit_threshold.setToolTip("反射率阈值（0-1）。留空=自动计算")
        from PySide6.QtGui import QDoubleValidator
        self.edit_threshold.setValidator(QDoubleValidator(0.0, 1.0, 4))
        seg_layout.addLayout(self._create_responsive_layout("阈值:", self.edit_threshold))
        
        btn_row = QHBoxLayout()
        
        self.btn_preset = QPushButton("⚙️")
        self.btn_preset.setMaximumWidth(35)
        self.btn_preset.setToolTip("参数预设")
        self.btn_preset.setEnabled(False)
        self.btn_preset.clicked.connect(self._show_presets)
        btn_row.addWidget(self.btn_preset)
        
        self.btn_segment = QPushButton("▶ 执行分割")
        self.btn_segment.setEnabled(False)
        self.btn_segment.clicked.connect(self.segment_requested)
        btn_row.addWidget(self.btn_segment)
        
        seg_layout.addLayout(btn_row)
        
        self.btn_show_pipeline = QPushButton("📊 查看分割流程")
        self.btn_show_pipeline.setEnabled(False)
        self.btn_show_pipeline.setToolTip("可视化展示分割的6个步骤")
        seg_layout.addWidget(self.btn_show_pipeline)
        
        group_seg.setLayout(seg_layout)
        layout.addWidget(group_seg)
        
        # ========== 显示选项 ==========
        group_display = QGroupBox("🎨 显示选项")
        display_layout = QVBoxLayout()
        
        # 亮度调整
        adjust_label = QLabel("亮度调整:")
        adjust_label.setStyleSheet("font-weight: bold; color: #555; font-size: 10pt;")
        display_layout.addWidget(adjust_label)
        
        brightness_layout = QHBoxLayout()
        brightness_layout.addWidget(QLabel("暗"))
        
        self.slider_brightness = QSlider(Qt.Horizontal)
        self.slider_brightness.setRange(-100, 100)  # -100到100
        self.slider_brightness.setValue(0)  # 默认0（无调整）
        self.slider_brightness.setToolTip("调整图像显示亮度（不改变原始数据）")
        brightness_layout.addWidget(self.slider_brightness)
        
        brightness_layout.addWidget(QLabel("亮"))
        display_layout.addLayout(brightness_layout)
        
        self.label_brightness = QLabel("当前: 0")
        self.label_brightness.setAlignment(Qt.AlignCenter)
        self.label_brightness.setStyleSheet("color: #666; font-size: 9pt;")
        display_layout.addWidget(self.label_brightness)
        
        # 连接滑块信号
        self.slider_brightness.valueChanged.connect(self._update_brightness_label)
        
        display_layout.addSpacing(6)
        
        # 图像变换
        transform_label = QLabel("图像变换:")
        transform_label.setStyleSheet("font-weight: bold; color: #555; font-size: 10pt;")
        display_layout.addWidget(transform_label)
        
        # 旋转和镜像按钮
        btn_layout1 = QHBoxLayout()
        self.btn_rotate_left = QPushButton("↺ 左旋")
        self.btn_rotate_left.setToolTip("逆时针旋转90度")
        self.btn_rotate_right = QPushButton("↻ 右旋")
        self.btn_rotate_right.setToolTip("顺时针旋转90度")
        btn_layout1.addWidget(self.btn_rotate_left)
        btn_layout1.addWidget(self.btn_rotate_right)
        display_layout.addLayout(btn_layout1)
        
        btn_layout2 = QHBoxLayout()
        self.btn_flip_ud = QPushButton("⇅ 上下镜像")
        self.btn_flip_ud.setToolTip("垂直翻转")
        self.btn_flip_lr = QPushButton("⇄ 左右镜像")
        self.btn_flip_lr.setToolTip("水平翻转")
        btn_layout2.addWidget(self.btn_flip_ud)
        btn_layout2.addWidget(self.btn_flip_lr)
        display_layout.addLayout(btn_layout2)
        
        group_display.setLayout(display_layout)
        layout.addWidget(group_display)
        
        # ========== ROI工具 ==========
        group_roi = QGroupBox("✂️ ROI工具")
        roi_layout = QVBoxLayout()
        
        # ROI 形状选择
        shape_row = QHBoxLayout()
        lbl_shape = QLabel("形状:")
        lbl_shape.setMinimumWidth(50)
        shape_row.addWidget(lbl_shape)
        self.combo_roi_shape = QComboBox()
        self.combo_roi_shape.addItems(["矩形", "椭圆", "多边形"])
        self.combo_roi_shape.setToolTip(
            "矩形：拖拽绘制矩形区域\n"
            "椭圆：拖拽绘制椭圆形区域\n"
            "多边形：点击多个点绘制不规则区域，双击完成"
        )
        shape_row.addWidget(self.combo_roi_shape)
        roi_layout.addLayout(shape_row)
        
        # ROI 操作按钮
        btn_row = QHBoxLayout()
        self.btn_roi = QPushButton("➕ 绘制")
        self.btn_roi.setCheckable(True)
        self.btn_roi.setToolTip("点击后在图像上拖拽绘制ROI")
        self.btn_roi.setEnabled(False)
        btn_row.addWidget(self.btn_roi)
        
        self.btn_delete_roi = QPushButton("🗑")
        self.btn_delete_roi.setToolTip("删除选中的ROI")
        self.btn_delete_roi.setMaximumWidth(35)
        self.btn_delete_roi.setEnabled(False)
        btn_row.addWidget(self.btn_delete_roi)
        
        self.btn_clear_roi = QPushButton("🧹")
        self.btn_clear_roi.setToolTip("清空所有ROI")
        self.btn_clear_roi.setMaximumWidth(35)
        self.btn_clear_roi.setEnabled(False)
        btn_row.addWidget(self.btn_clear_roi)
        
        self.btn_export_roi = QPushButton("💾")
        self.btn_export_roi.setToolTip("批量导出所有ROI光谱")
        self.btn_export_roi.setMaximumWidth(35)
        self.btn_export_roi.setEnabled(False)
        btn_row.addWidget(self.btn_export_roi)
        
        roi_layout.addLayout(btn_row)
        
        # ROI 列表
        from PySide6.QtWidgets import QListWidget
        self.list_rois = QListWidget()
        self.list_rois.setMaximumHeight(120)
        self.list_rois.setToolTip("双击可重命名ROI")
        self.list_rois.itemDoubleClicked.connect(self._on_roi_rename)
        self.list_rois.currentRowChanged.connect(lambda idx: self.roi_item_selected.emit(idx) if idx >= 0 else None)
        roi_layout.addWidget(self.list_rois)
        
        # 连接按钮信号
        self.btn_delete_roi.clicked.connect(self._on_delete_roi)
        self.btn_clear_roi.clicked.connect(self.roi_clear_requested.emit)
        self.btn_export_roi.clicked.connect(self.roi_export_requested.emit)
        
        group_roi.setLayout(roi_layout)
        layout.addWidget(group_roi)
        
        # ========== 光谱绘制模式 ==========
        group_spectrum = QGroupBox("📈 光谱绘制")
        spectrum_layout = QVBoxLayout()
        
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("模式:"))
        self.combo_spectrum_mode = QComboBox()
        self.combo_spectrum_mode.addItems(["单线", "多线"])
        self.combo_spectrum_mode.setToolTip("单线：每次点击替换光谱\n多线：连续点击添加多条光谱")
        mode_layout.addWidget(self.combo_spectrum_mode)
        spectrum_layout.addLayout(mode_layout)
        
        btn_spectrum_layout = QHBoxLayout()
        self.btn_clear_spectra = QPushButton("🗑 清空")
        self.btn_clear_spectra.setToolTip("清空所有光谱曲线")
        btn_spectrum_layout.addWidget(self.btn_clear_spectra)

        self.btn_avg_spectra = QPushButton("📊 平均")
        self.btn_avg_spectra.setToolTip("计算所有曲线的平均值")
        self.btn_avg_spectra.setEnabled(False)
        btn_spectrum_layout.addWidget(self.btn_avg_spectra)

        self.label_spectrum_count = QLabel("曲线数: 0")
        self.label_spectrum_count.setStyleSheet("color: #666; font-size: 9pt;")
        btn_spectrum_layout.addWidget(self.label_spectrum_count)
        spectrum_layout.addLayout(btn_spectrum_layout)
        
        group_spectrum.setLayout(spectrum_layout)
        layout.addWidget(group_spectrum)
        
        # # ========== 模型识别 ==========
        # group_model = QGroupBox("🤖 模型识别")
        # group_model.setCheckable(True)
        # group_model.setChecked(False)
        # model_layout = QVBoxLayout()
        #
        # self.btn_load_model = QPushButton("📂 加载模型")
        # self.btn_load_model.setEnabled(True)
        # self.btn_load_model.setToolTip("加载训练好的分类模型 (.pkl/.h5/.pt)")
        # model_layout.addWidget(self.btn_load_model)
        #
        # self.label_model_status = QLabel("未加载")
        # self.label_model_status.setStyleSheet("color: #999; font-size: 8pt;")
        # self.label_model_status.setToolTip("模型状态")
        # model_layout.addWidget(self.label_model_status)
        #
        # self.btn_predict = QPushButton("▶ 开始识别")
        # self.btn_predict.setEnabled(False)
        # self.btn_predict.setToolTip("识别小麦霉变等级")
        # model_layout.addWidget(self.btn_predict)
        #
        # group_model.setLayout(model_layout)
        # layout.addWidget(group_model)
        
        layout.addStretch()
    
    def _on_band_input(self):
        """波段输入框回车确认"""
        try:
            value = int(self.edit_band.text())
            if 0 <= value <= self.slider_band.maximum():
                self.slider_band.setValue(value)
                self.band_changed.emit(value)
        except ValueError:
            # 恢复为当前滑块值
            self.edit_band.setText(str(self.slider_band.value()))
    
    def _on_slider_changed(self, value: int):
        """滑块变化时更新输入框"""
        self.edit_band.setText(str(value))
        self.band_changed.emit(value)
    
    def set_band_range(self, max_band: int, wavelengths=None):
        """设置波段范围"""
        # 灰度模式控件
        self.edit_band.setEnabled(True)
        self.edit_band.setValidator(QIntValidator(0, max_band - 1))
        self.slider_band.setEnabled(True)
        self.slider_band.setMaximum(max_band - 1)
        self.slider_band.setMinimum(0)
        
        # RGB模式控件
        for spin, slider in [(self.spin_r, self.slider_r), 
                              (self.spin_g, self.slider_g), 
                              (self.spin_b, self.slider_b)]:
            spin.setEnabled(True)
            spin.setMaximum(max_band - 1)
            spin.setMinimum(0)
            slider.setEnabled(True)
            slider.setMaximum(max_band - 1)
            slider.setMinimum(0)
        
        self.btn_rgb_preset.setEnabled(True)
        
        # 设置默认RGB波段
        if max_band >= 3:
            if wavelengths and len(wavelengths) >= 3:
                # 基于波长智能选择（尝试找到接近 650nm, 550nm, 450nm 的波段）
                try:
                    wavelengths_float = [float(w) for w in wavelengths]
                    # 查找最接近目标波长的波段
                    r_target, g_target, b_target = 650, 550, 450
                    r_idx = min(range(len(wavelengths_float)), key=lambda i: abs(wavelengths_float[i] - r_target))
                    g_idx = min(range(len(wavelengths_float)), key=lambda i: abs(wavelengths_float[i] - g_target))
                    b_idx = min(range(len(wavelengths_float)), key=lambda i: abs(wavelengths_float[i] - b_target))
                    
                    # 如果找到的波段过于接近，使用等间距分布
                    if abs(r_idx - g_idx) < max_band // 10 or abs(g_idx - b_idx) < max_band // 10:
                        # 使用图2的分布模式：约1/3, 1/2, 2/3位置
                        r_idx = min(max_band - 1, max_band * 2 // 3)
                        g_idx = max_band // 2
                        b_idx = max_band // 3
                    
                    self.spin_r.setValue(r_idx)
                    self.spin_g.setValue(g_idx)
                    self.spin_b.setValue(b_idx)
                except (ValueError, TypeError):
                    # 波长格式错误，使用默认分布
                    self.spin_r.setValue(min(max_band - 1, max_band * 2 // 3))
                    self.spin_g.setValue(max_band // 2)
                    self.spin_b.setValue(max_band // 3)
            else:
                # 无波长信息，使用等间距分布（参考图2：约1/3, 1/2, 2/3位置）
                self.spin_r.setValue(min(max_band - 1, max_band * 2 // 3))
                self.spin_g.setValue(max_band // 2)
                self.spin_b.setValue(max_band // 3)
        
        self.btn_segment.setEnabled(True)
        self.btn_preset.setEnabled(True)  # 启用参数预设

        # 启用分割参数控件
        self.combo_method.setEnabled(True)
        self.combo_order_mode.setEnabled(True)
        self.edit_min_area.setEnabled(True)
        self.edit_max_area.setEnabled(True)
        self.edit_threshold.setEnabled(True)

        # 展开分割参数面板
        if hasattr(self, 'group_seg'):
            self.group_seg.setChecked(True)

        self._wavelengths = wavelengths
        self._update_rgb_wavelength_labels()
    
    def update_wavelength_label(self, band_idx: int):
        """更新波长显示"""
        if hasattr(self, '_wavelengths') and self._wavelengths:
            wl = self._wavelengths[band_idx]
            self.label_wavelength.setText(f"波长: {wl} nm")
        else:
            self.label_wavelength.setText(f"波段索引: {band_idx}")
    
    def _show_presets(self):
        """显示参数预设"""
        from PySide6.QtWidgets import QMenu
        from PySide6.QtGui import QAction
        
        menu = QMenu(self)
        
        # 预设1: 小麦籽粒
        act1 = QAction("🌾 小麦", self)
        act1.triggered.connect(lambda: self._apply_preset(100, 1000, 0.12))
        menu.addAction(act1)
        
        # 预设2: 玉米籽粒
        act2 = QAction("🌽 玉米", self)
        act2.triggered.connect(lambda: self._apply_preset(400, 2000, 0.12))
        menu.addAction(act2)
        
        # 预设3: 水稻籽粒
        act3 = QAction("🌾 水稻", self)
        act3.triggered.connect(lambda: self._apply_preset(80, 800, 0.06))
        menu.addAction(act3)
        
        menu.addSeparator()
        
        # 重置
        act_reset = QAction("🔄 重置为默认", self)
        act_reset.triggered.connect(lambda: self._apply_preset(150, 1000, None))
        menu.addAction(act_reset)
        
        # 显示菜单
        menu.exec(self.btn_preset.mapToGlobal(self.btn_preset.rect().bottomLeft()))
    
    def _apply_preset(self, min_area: int, max_area: int, threshold: float = None):
        """应用参数预设"""
        self.edit_min_area.setText(str(min_area))
        self.edit_max_area.setText(str(max_area))
        if threshold is not None:
            self.edit_threshold.setText(f"{threshold:.2f}")
        else:
            self.edit_threshold.setText("")
    
    def _update_brightness_label(self, value: int):
        """更新亮度标签"""
        if value > 0:
            self.label_brightness.setText(f"当前: +{value}")
        else:
            self.label_brightness.setText(f"当前: {value}")
    
    def _on_delete_roi(self):
        """删除选中的ROI"""
        idx = self.list_rois.currentRow()
        if idx >= 0:
            self.roi_delete_requested.emit(idx)
    
    def _on_roi_rename(self, item):
        """重命名ROI"""
        from PySide6.QtWidgets import QInputDialog
        old_name = item.text().split(' ', 1)[-1]  # 去掉色块前缀
        new_name, ok = QInputDialog.getText(self, "重命名ROI", "新名称:", text=old_name)
        if ok and new_name.strip():
            # 通过信号通知主视图更新
            idx = self.list_rois.row(item)
            item.setText(f"■ {new_name.strip()}")  # 保留色块，更新名称
    
    def _create_responsive_layout(self, label_text, widget, stretch_widget=True):
        """创建响应式布局的标签-控件组合
        
        Args:
            label_text: 标签文本
            widget: 控件（QComboBox, QLineEdit等）
            stretch_widget: 是否让控件拉伸填充剩余空间
        """
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 2, 0, 2)  # 减少垂直边距
        
        # 创建标签
        label = QLabel(label_text)
        label.setMinimumWidth(60)  # 设置最小宽度
        label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        layout.addWidget(label)
        
        # 添加控件
        if stretch_widget:
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        else:
            widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        layout.addWidget(widget)
        
        return layout
    
    def _on_display_mode_changed(self, mode: str):
        """显示模式切换"""
        if mode == "灰度图":
            self.gray_controls.setVisible(True)
            self.rgb_controls.setVisible(False)
        elif mode == "RGB合成":
            self.gray_controls.setVisible(False)
            self.rgb_controls.setVisible(True)
        else:
            # 真彩色/假彩色为自动映射，不需要手动RGB滑块
            self.gray_controls.setVisible(False)
            self.rgb_controls.setVisible(False)
    
    def _update_rgb_wavelength_labels(self):
        """更新RGB波段的波长标签"""
        if not hasattr(self, '_wavelengths') or not self._wavelengths:
            self.label_wl_r.setText("-")
            self.label_wl_g.setText("-")
            self.label_wl_b.setText("-")
            return
        
        try:
            r_idx = self.spin_r.value()
            g_idx = self.spin_g.value()
            b_idx = self.spin_b.value()
            
            if r_idx < len(self._wavelengths):
                self.label_wl_r.setText(f"{float(self._wavelengths[r_idx]):.1f}nm")
            if g_idx < len(self._wavelengths):
                self.label_wl_g.setText(f"{float(self._wavelengths[g_idx]):.1f}nm")
            if b_idx < len(self._wavelengths):
                self.label_wl_b.setText(f"{float(self._wavelengths[b_idx]):.1f}nm")
        except (IndexError, ValueError):
            pass
    
    def _show_rgb_presets(self):
        """显示RGB波段预设菜单"""
        from PySide6.QtWidgets import QMenu
        from PySide6.QtGui import QAction
        
        menu = QMenu(self)
        
        # 预设1: 真彩色 (根据波长)
        if hasattr(self, '_wavelengths') and self._wavelengths:
            act1 = QAction("🌈 真彩色 (650/550/450nm)", self)
            act1.triggered.connect(lambda: self._apply_rgb_preset_by_wavelength(650, 550, 450))
            menu.addAction(act1)
            
            act2 = QAction("🌿 假彩色 (800/650/550nm)", self)
            act2.triggered.connect(lambda: self._apply_rgb_preset_by_wavelength(800, 650, 550))
            menu.addAction(act2)
            
            act3 = QAction("🔬 近红外 (900/800/700nm)", self)
            act3.triggered.connect(lambda: self._apply_rgb_preset_by_wavelength(900, 800, 700))
            menu.addAction(act3)
        else:
            # 没有波长信息时，使用波段索引
            max_band = self.slider_r.maximum()
            act1 = QAction("🌈 均匀分布", self)
            act1.triggered.connect(lambda: self._apply_rgb_preset_by_index(
                max_band * 2 // 3, max_band // 2, max_band // 3
            ))
            menu.addAction(act1)
        
        menu.exec(self.btn_rgb_preset.mapToGlobal(self.btn_rgb_preset.rect().bottomLeft()))
    
    def _apply_rgb_preset_by_wavelength(self, r_wl: float, g_wl: float, b_wl: float):
        """根据目标波长应用RGB预设"""
        if not hasattr(self, '_wavelengths') or not self._wavelengths:
            return
        
        # 查找最接近的波段
        wavelengths = [float(w) for w in self._wavelengths]
        r_idx = min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i] - r_wl))
        g_idx = min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i] - g_wl))
        b_idx = min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i] - b_wl))
        
        self.spin_r.setValue(r_idx)
        self.spin_g.setValue(g_idx)
        self.spin_b.setValue(b_idx)
        self._update_rgb_wavelength_labels()
    
    def _apply_rgb_preset_by_index(self, r_idx: int, g_idx: int, b_idx: int):
        """根据波段索引应用RGB预设"""
        self.spin_r.setValue(r_idx)
        self.spin_g.setValue(g_idx)
        self.spin_b.setValue(b_idx)
        self._update_rgb_wavelength_labels()

