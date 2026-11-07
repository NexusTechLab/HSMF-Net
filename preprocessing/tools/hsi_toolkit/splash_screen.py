"""
启动界面 - 显示加载进度和版本信息
"""
import sys
import time
from PySide6.QtWidgets import (QApplication, QSplashScreen, QLabel, 
                               QVBoxLayout, QWidget, QProgressBar)
from PySide6.QtGui import QPixmap, QPainter, QFont, QColor
from PySide6.QtCore import Qt, QTimer, Signal, QThread

class SplashScreen(QSplashScreen):
    """启动画面"""
    
    def __init__(self):
        # 创建启动画面图片
        pixmap = QPixmap(400, 300)
        pixmap.fill(QColor(240, 248, 255))  # 浅蓝色背景
        
        # 绘制启动画面
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制标题
        painter.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
        painter.setPen(QColor(25, 118, 210))  # 蓝色
        painter.drawText(50, 80, "ENVI高光谱数据分析工具")
        
        # 绘制版本
        painter.setFont(QFont("Microsoft YaHei", 12))
        painter.setPen(QColor(100, 100, 100))
        painter.drawText(50, 110, "Version 2.0")
        
        # 绘制功能列表
        features = [
            "• 高光谱数据加载与显示",
            "• ROI区域绘制与光谱提取", 
            "• 智能种子分割与排序",
            "• 网格优化排序算法",
            "• 多格式数据导出"
        ]
        
        painter.setFont(QFont("Microsoft YaHei", 10))
        painter.setPen(QColor(60, 60, 60))
        for i, feature in enumerate(features):
            painter.drawText(50, 140 + i * 20, feature)
        
        # 绘制加载提示
        painter.setFont(QFont("Microsoft YaHei", 9))
        painter.setPen(QColor(120, 120, 120))
        painter.drawText(50, 260, "正在加载程序组件...")
        
        painter.end()
        
        super().__init__(pixmap)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.SplashScreen)
        
        # 设置进度
        self.progress = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(50)  # 50ms更新一次
    
    def update_progress(self):
        """更新进度"""
        self.progress += 2
        if self.progress >= 100:
            self.timer.stop()
            self.close()
        else:
            self.showMessage(f"加载中... {self.progress}%", 
                           Qt.AlignBottom | Qt.AlignCenter, 
                           QColor(25, 118, 210))

class LoadingThread(QThread):
    """加载线程"""
    progress_updated = Signal(int)
    finished = Signal()
    
    def run(self):
        """执行加载任务"""
        # 模拟加载过程
        steps = [
            ("初始化应用程序", 10),
            ("加载UI组件", 20),
            ("初始化图像处理模块", 30),
            ("加载分割算法", 40),
            ("初始化ROI工具", 50),
            ("加载光谱分析模块", 60),
            ("初始化导出功能", 70),
            ("加载样式文件", 80),
            ("完成初始化", 100)
        ]
        
        for step_name, progress in steps:
            time.sleep(0.3)  # 模拟加载时间
            self.progress_updated.emit(progress)
        
        self.finished.emit()

def show_splash_screen():
    """显示启动画面"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # 创建启动画面
    splash = SplashScreen()
    splash.show()
    
    # 创建加载线程
    loading_thread = LoadingThread()
    loading_thread.progress_updated.connect(splash.update_progress)
    loading_thread.finished.connect(splash.close)
    loading_thread.start()
    
    # 处理事件
    app.processEvents()
    
    return splash, loading_thread

if __name__ == "__main__":
    # 测试启动画面
    app = QApplication(sys.argv)
    splash, thread = show_splash_screen()
    
    # 等待加载完成
    thread.wait()
    
    # 显示主窗口（这里只是测试）
    from PySide6.QtWidgets import QMainWindow, QLabel
    window = QMainWindow()
    window.setWindowTitle("主程序")
    window.setCentralWidget(QLabel("主程序已启动"))
    window.show()
    
    sys.exit(app.exec())
