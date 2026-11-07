"""
数据集管理面板
支持多个样本加载、切换
"""
from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, 
    QListWidget, QListWidgetItem, QLabel
)
from PySide6.QtCore import Signal
from pathlib import Path

class DatasetPanel(QGroupBox):
    """数据集列表管理"""
    
    dataset_changed = Signal(int)  # 切换数据集
    batch_add_requested = Signal()  # 批量添加
    batch_remove_requested = Signal()  # 批量删除
    
    def __init__(self, parent=None):
        super().__init__("📁 数据集列表", parent)
        
        layout = QVBoxLayout(self)
        
        # 列表控件
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self._on_current_row_changed)
        layout.addWidget(self.list_widget)
        
        # 按钮栏
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)
        
        self.btn_add = QPushButton("➕")
        self.btn_add.setMaximumWidth(35)
        self.btn_add.setToolTip("批量添加数据集")
        
        self.btn_remove = QPushButton("➖")
        self.btn_remove.setMaximumWidth(35)
        self.btn_remove.setToolTip("批量删除所有数据集")
        self.btn_remove.setEnabled(False)
        
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        self.label_count = QLabel("总数: 0")
        layout.addWidget(self.label_count)
        
        # 数据存储
        self.datasets = []
    
    def add_dataset(self, name: str, data_dict: dict):
        """添加数据集"""
        self.datasets.append(data_dict)
        
        # 单行显示文件名和尺寸信息
        cube = data_dict.get('cube')
        if cube is not None:
            h, w, bands = cube.shape
            display_name = f"{name} ({h}×{w}×{bands})"
        else:
            display_name = name
        
        item = QListWidgetItem(display_name)
        self.list_widget.addItem(item)
        
        # 自动选中新添加的项
        self.list_widget.setCurrentRow(len(self.datasets) - 1)
        
        self.label_count.setText(f"总数: {len(self.datasets)}")
        self.btn_remove.setEnabled(True)
    
    def remove_current(self):
        """移除当前选中"""
        row = self.list_widget.currentRow()
        if row >= 0:
            self.datasets.pop(row)
            self.list_widget.takeItem(row)
            self.label_count.setText(f"总数: {len(self.datasets)}")
            if len(self.datasets) == 0:
                self.btn_remove.setEnabled(False)
    
    def remove_all(self):
        """批量删除所有数据集"""
        self.datasets.clear()
        self.list_widget.clear()
        self.label_count.setText("总数: 0")
        self.btn_remove.setEnabled(False)
    
    def get_current_dataset(self):
        """获取当前数据集"""
        row = self.list_widget.currentRow()
        if row >= 0 and row < len(self.datasets):
            return self.datasets[row]
        return None
    
    def get_dataset_name_by_index(self, idx):
        """根据索引获取数据集名称"""
        if 0 <= idx < self.list_widget.count():
            item_text = self.list_widget.item(idx).text()
            # 提取名称（去掉尺寸信息）
            return item_text.split(' (')[0]
        return None
    
    def _on_current_row_changed(self, row):
        """处理当前行变化事件"""
        print(f"数据集面板: 当前行变化到 {row}")  # 调试信息
        self.dataset_changed.emit(row)

