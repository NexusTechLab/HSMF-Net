"""单颗粒 ENVI 批量导出设置。"""
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QVBoxLayout,
)
from utils.export_settings import grain_filename, validate_filename_prefix


class SeedExportDialog(QDialog):
    def __init__(self, band_count, grain_count, wavelengths=None,
                 plot_band_range=None, parent=None, filename_prefix=''):
        super().__init__(parent)
        self.setWindowTitle("批量导出单颗粒高光谱数据")
        self.setMinimumWidth(460)
        self.wavelengths = wavelengths
        self.grain_count = grain_count
        self.plot_band_range = plot_band_range
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"共 {grain_count} 个颗粒，当前数据共 {band_count} 个波段"))

        numbering = QGroupBox("输出编号")
        form = QFormLayout(numbering)
        self.edit_filename_prefix = QLineEdit(filename_prefix)
        self.edit_filename_prefix.setPlaceholderText("例如 ZM22，留空则只使用编号")
        form.addRow("小麦品种名称：", self.edit_filename_prefix)
        self.spin_start_number = QSpinBox()
        self.spin_start_number.setRange(0, 1_000_000_000)
        self.spin_start_number.setValue(1)
        form.addRow("起始编号：", self.spin_start_number)
        self.spin_number_digits = QSpinBox()
        self.spin_number_digits.setRange(1, 10)
        self.spin_number_digits.setValue(4)
        form.addRow("编号位数（不足补零）：", self.spin_number_digits)
        self.label_number_preview = QLabel()
        form.addRow("文件名预览：", self.label_number_preview)
        layout.addWidget(numbering)

        bands = QGroupBox("波段范围")
        band_layout = QVBoxLayout(bands)
        self.chk_all_bands = QCheckBox("导出全部波段")
        self.chk_all_bands.setChecked(True)
        band_layout.addWidget(self.chk_all_bands)
        row = QHBoxLayout()
        self.spin_band_start = QSpinBox()
        self.spin_band_end = QSpinBox()
        for spin in (self.spin_band_start, self.spin_band_end):
            spin.setRange(0, band_count - 1)
        self.spin_band_end.setValue(band_count - 1)
        row.addWidget(QLabel("起始波段："))
        row.addWidget(self.spin_band_start)
        row.addWidget(QLabel("结束波段："))
        row.addWidget(self.spin_band_end)
        band_layout.addLayout(row)
        band_layout.addWidget(QLabel("波段索引从 0 开始，起始和结束波段均包含在导出范围内。"))
        self.label_band_preview = QLabel()
        band_layout.addWidget(self.label_band_preview)
        if plot_band_range is not None:
            button = QPushButton("使用当前光谱图显示范围")
            button.clicked.connect(self._use_plot_range)
            band_layout.addWidget(button)
        layout.addWidget(bands)

        self.chk_save_images = QCheckBox("同时保存颗粒图像和光谱图（PNG）")
        self.chk_save_images.setChecked(True)
        layout.addWidget(self.chk_save_images)
        self.chk_save_csv = QCheckBox("同时保存平均光谱汇总（CSV）")
        self.chk_save_csv.setChecked(True)
        layout.addWidget(self.chk_save_csv)
        hint = QLabel("下一步将询问是否采用光谱图坐标轴范围及勾选的预处理方法。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("下一步")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.spin_start_number.valueChanged.connect(self._update_number_preview)
        self.edit_filename_prefix.textChanged.connect(self._update_number_preview)
        self.spin_number_digits.valueChanged.connect(self._update_number_preview)
        self.chk_all_bands.toggled.connect(self._toggle_all_bands)
        self.spin_band_start.valueChanged.connect(self._start_band_changed)
        self.spin_band_end.valueChanged.connect(self._end_band_changed)
        self._update_number_preview()
        self._toggle_all_bands(True)

    def _update_number_preview(self):
        start = self.spin_start_number.value()
        digits = self.spin_number_digits.value()
        end = start + self.grain_count - 1
        prefix = self.edit_filename_prefix.text().strip()
        first = grain_filename(start, digits, prefix)
        last = grain_filename(end, digits, prefix)
        self.label_number_preview.setText(f"{first}.png … {last}.png")
        self.label_number_preview.setWordWrap(True)

    def accept(self):
        try:
            validate_filename_prefix(self.edit_filename_prefix.text())
        except ValueError as error:
            QMessageBox.warning(self, "品种名称无效", str(error))
            return
        super().accept()

    def _toggle_all_bands(self, checked):
        self.spin_band_start.setEnabled(not checked)
        self.spin_band_end.setEnabled(not checked)
        self._update_band_preview()

    def _start_band_changed(self, value):
        if value > self.spin_band_end.value():
            self.spin_band_end.setValue(value)
        self._update_band_preview()

    def _end_band_changed(self, value):
        if value < self.spin_band_start.value():
            self.spin_band_start.setValue(value)
        self._update_band_preview()

    def _use_plot_range(self):
        start, end = self.plot_band_range
        self.chk_all_bands.setChecked(False)
        self.spin_band_end.setValue(end - 1)
        self.spin_band_start.setValue(start)

    def _update_band_preview(self):
        start, end = self.band_range()
        text = f"将导出波段 {start}–{end - 1}，共 {end - start} 个波段"
        if self.wavelengths is not None and len(self.wavelengths):
            text += f"（{float(self.wavelengths[start]):.2f}–{float(self.wavelengths[end - 1]):.2f} nm）"
        self.label_band_preview.setText(text)

    def band_range(self):
        """转换为 Python 切片使用的左闭右开区间。"""
        if self.chk_all_bands.isChecked():
            return 0, self.spin_band_end.maximum() + 1
        return self.spin_band_start.value(), self.spin_band_end.value() + 1

    def export_options(self):
        return {
            'band_range': self.band_range(),
            'start_number': self.spin_start_number.value(),
            'number_digits': self.spin_number_digits.value(),
            'save_images': self.chk_save_images.isChecked(),
            'save_csv': self.chk_save_csv.isChecked(),
            'filename_prefix': validate_filename_prefix(self.edit_filename_prefix.text()),
        }
