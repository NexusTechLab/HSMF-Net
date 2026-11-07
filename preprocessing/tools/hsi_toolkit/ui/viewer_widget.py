from pathlib import Path
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QSpinBox, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, Signal
from spectral import io as envi_io

import matplotlib
matplotlib.use("QtAgg")
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.patches
from matplotlib.widgets import RectangleSelector

import pyqtgraph as pg


class HSIViewerWidget(QWidget):
    """v0.1 组件：文件打开 + 波段显示 + 像素光谱查看"""

    band_changed = Signal(int)
    pixel_info = Signal(int, int, float)  # row, col, value
    roi_selected = Signal(tuple)  # (row0,row1,col0,col1)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cube = None
        self.wavelengths = None
        self.band_idx = 0

        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # 左侧图像 + 控件
        left_layout = QVBoxLayout()
        left_layout.setSpacing(4)
        main_layout.addLayout(left_layout, 3)

        # Matplotlib figure
        self.fig = Figure(figsize=(5, 4))
        self.fig.patch.set_facecolor('white')
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        
        # Navigation toolbar (simplified)
        self.navbar = NavigationToolbar(self.canvas, self)
        # 隐藏不常用按钮，只保留：主页、缩放、平移、保存
        unwanted = ['Customize', 'Subplots']
        for action in self.navbar.actions():
            if action.text() in unwanted:
                self.navbar.removeAction(action)
        self.navbar.setIconSize(self.navbar.iconSize() * 0.8)
        left_layout.addWidget(self.navbar)
        left_layout.addWidget(self.canvas, stretch=1)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)
        ctrl.setContentsMargins(4, 4, 4, 4)
        left_layout.addLayout(ctrl)

        self.btn_open = QPushButton("打开…")
        self.btn_open.clicked.connect(self.open_file)
        ctrl.addWidget(self.btn_open)

        ctrl.addWidget(QLabel("波段:"))
        self.spin_band = QSpinBox()
        self.spin_band.setEnabled(False)
        self.spin_band.valueChanged.connect(self.spin_changed)
        ctrl.addWidget(self.spin_band)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.slider_changed)
        ctrl.addWidget(self.slider, stretch=1)

        self.btn_roi = QPushButton("ROI")
        self.btn_roi.setEnabled(False)
        self.btn_roi.setCheckable(True)
        self.btn_roi.toggled.connect(self.toggle_roi)
        ctrl.addWidget(self.btn_roi)
        # selector placeholder
        self._roi_selector = None
        self.roi_bounds = None
        
        # segmentation results
        self.labels = None
        self.regions = None
        self._show_labels = False

        # 右侧光谱曲线
        self.plot = pg.PlotWidget(title="光谱曲线")
        self.plot.setLabel("left", "反射率")
        self.plot.setLabel("bottom", "波段索引")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        # 初始提示文字
        text_item = pg.TextItem("请打开数据文件并点击图像像素查看光谱", color=(150, 150, 150))
        text_item.setPos(0.5, 0.5)
        self.plot.addItem(text_item)
        main_layout.addWidget(self.plot, 2)

    # ---------------- handlers ----------------
    def open_file(self):
        hdr_path, _ = QFileDialog.getOpenFileName(self, "选择 HDR", str(Path.cwd()), "HDR (*.hdr *.HDR)")
        if not hdr_path:
            return
        hdr_path = Path(hdr_path)
        raw_path = hdr_path.with_suffix(".raw")
        if not raw_path.exists():
            raw_path = hdr_path.with_suffix("")
        if not raw_path.exists():
            QMessageBox.critical(self, "错误", f"未找到数据文件:\n{raw_path}")
            return
        spec = envi_io.envi.open(str(hdr_path), str(raw_path))
        cube = spec.load().astype(np.float32)
        
        # 自动检测DN值并转换为反射率
        cube_max = float(np.nanmax(cube))
        cube_min = float(np.nanmin(cube))
        if cube_max > 1.5:  # 阈值1.5避免误判（反射率理论上应≤1.0）
            print(f"[自动转换] 检测到DN值数据: 范围[{cube_min:.2f}, {cube_max:.2f}]")
            # 归一化到0-1范围
            if cube_min >= 0:
                cube = cube / cube_max
            else:
                cube = (cube - cube_min) / (cube_max - cube_min)
            print(f"[转换完成] 已归一化为反射率: 范围[{np.nanmin(cube):.4f}, {np.nanmax(cube):.4f}]")
        
        self.cube = cube
        
        # 提取波长信息并转换为float列表
        wavelengths_raw = spec.metadata.get("wavelength")
        if wavelengths_raw:
            try:
                self.wavelengths = [float(w) for w in wavelengths_raw]
            except (ValueError, TypeError):
                self.wavelengths = None
        else:
            self.wavelengths = None

        bands = self.cube.shape[2]
        self.spin_band.setEnabled(True)
        self.spin_band.setMaximum(bands - 1)
        self.spin_band.setValue(bands // 2)

        self.slider.setEnabled(True)
        self.slider.setMaximum(bands - 1)
        self.slider.setValue(bands // 2)

        self.band_idx = bands // 2
        self.update_image()
        self.plot.clear()
        self.btn_roi.setEnabled(True)

    def toggle_roi(self, checked: bool):
        if checked:
            self._roi_selector = RectangleSelector(
                self.ax, self._roi_onselect,
                drawtype='box', useblit=True, button=[1], interactive=True,
                rectprops=dict(facecolor='none', edgecolor='red', linewidth=1.5))
        else:
            if self._roi_selector:
                self._roi_selector.set_active(False)
                self._roi_selector = None

    def _roi_onselect(self, eclick, erelease):
        r0, c0 = int(eclick.ydata), int(eclick.xdata)
        r1, c1 = int(erelease.ydata), int(erelease.xdata)
        r0, r1 = sorted((r0, r1))
        c0, c1 = sorted((c0, c1))
        self.roi_bounds = (r0, r1, c0, c1)
        self.roi_selected.emit(self.roi_bounds)
        # 显示 ROI 平均光谱
        self._show_roi_spectrum()

    def _show_roi_spectrum(self):
        """显示 ROI 区域的平均光谱"""
        if not self.roi_bounds or self.cube is None:
            return
        r0, r1, c0, c1 = self.roi_bounds
        roi_data = self.cube[r0:r1, c0:c1, :]
        mean_spec = roi_data.mean(axis=(0, 1)).astype(float)
        mean_spec[mean_spec <= 0] = np.nan
        
        self.plot.clear()
        if self.wavelengths and len(self.wavelengths) == len(mean_spec):
            x = np.array(list(map(float, self.wavelengths)))
            self.plot.setLabel("bottom", "波长 (nm)")
        else:
            x = np.arange(len(mean_spec))
            self.plot.setLabel("bottom", "波段索引")
        
        self.plot.plot(x, mean_spec, pen=pg.mkPen('g', width=2))
        ymin, ymax = np.nanmin(mean_spec), np.nanmax(mean_spec)
        if np.isnan(ymin) or np.isnan(ymax):
            return
        if ymin == ymax:
            ymin, ymax = ymin - 0.01, ymax + 0.01
        else:
            margin = (ymax - ymin) * 0.05
            ymin, ymax = ymin - margin, ymax + margin
        self.plot.setYRange(ymin, ymax)
        self.plot.setTitle(f"ROI 平均光谱 [{r1-r0}×{c1-c0} 像素]")

    # update_image should clear previous rect if any
    def update_image(self):
        if self.cube is None:
            return
        img = self.cube[:, :, self.band_idx]
        self.ax.clear()
        vmin, vmax = np.nanpercentile(img, [0, 99.5])
        self.ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        self.ax.set_title(f"波段 {self.band_idx}", color='black')
        self.ax.axis('off')
        
        # 叠加分割标签
        if self.labels is not None and self._show_labels:
            labels_masked = np.ma.masked_where(self.labels == 0, self.labels)
            self.ax.imshow(labels_masked, cmap='nipy_spectral', alpha=0.5)
            # 绘制编号
            if self.regions:
                for r in self.regions:
                    cy, cx = r.centroid
                    self.ax.text(cx, cy, str(r.label), color='white', fontsize=8,
                                ha='center', va='center', fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        # 绘制ROI框
        if self.roi_bounds:
            r0,r1,c0,c1 = self.roi_bounds
            self.ax.add_patch(matplotlib.patches.Rectangle((c0,r0), c1-c0, r1-r0,
                            linewidth=1.5, edgecolor='red', facecolor='none'))
        
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.draw()
        self.band_changed.emit(self.band_idx)

    def slider_changed(self, val):
        if val != self.band_idx:
            self.band_idx = val
            self.spin_band.blockSignals(True)
            self.spin_band.setValue(val)
            self.spin_band.blockSignals(False)
            self.update_image()

    def spin_changed(self, val):
        if val != self.band_idx:
            self.band_idx = val
            self.slider.blockSignals(True)
            self.slider.setValue(val)
            self.slider.blockSignals(False)
            self.update_image()

    def on_click(self, evt):
        if self.cube is None or evt.xdata is None or evt.ydata is None:
            return
        r, c = int(round(evt.ydata)), int(round(evt.xdata))
        if r < 0 or c < 0 or r >= self.cube.shape[0] or c >= self.cube.shape[1]:
            return
        spec = self.cube[r, c, :].astype(float)
        spec[spec <= 0] = np.nan
        if np.all(np.isnan(spec)):
            return
        self.plot.clear()
        if self.wavelengths and len(self.wavelengths) == len(spec):
            x = np.array(list(map(float, self.wavelengths)))
            self.plot.setLabel("bottom", "波长 (nm)")
        else:
            x = np.arange(len(spec))
            self.plot.setLabel("bottom", "波段索引")
        self.plot.plot(x, spec, pen=pg.mkPen('y', width=2))
        ymin, ymax = np.nanmin(spec), np.nanmax(spec)
        if ymin == ymax:
            ymin, ymax = ymin - 0.01, ymax + 0.01
        else:
            margin = (ymax - ymin) * 0.05
            ymin, ymax = ymin - margin, ymax + margin
        self.plot.setYRange(ymin, ymax)
        self.plot.setTitle(f"光谱 (row={r}, col={c})")
        self.pixel_info.emit(r, c, float(self.cube[r, c, self.band_idx]))
    
    def apply_band_crop(self, start: int, end: int):
        """波段截取"""
        if self.cube is not None:
            self.cube = self.cube[:, :, start:end+1]
            if self.wavelengths:
                self.wavelengths = self.wavelengths[start:end+1]
            # 更新控件
            bands = self.cube.shape[2]
            self.spin_band.setMaximum(bands - 1)
            self.slider.setMaximum(bands - 1)
            self.band_idx = min(self.band_idx, bands - 1)
            self.update_image()
    
    def set_segmentation(self, labels, regions):
        """设置分割结果"""
        self.labels = labels
        self.regions = regions
        self.update_image()
    
    def toggle_labels(self, checked: bool):
        """切换标签显示"""
        self._show_labels = checked
        self.update_image()
