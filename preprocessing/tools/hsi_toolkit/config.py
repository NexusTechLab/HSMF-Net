"""
配置文件 - 集中管理所有配置参数
"""
from pathlib import Path

class Config:
    """应用程序配置"""
    
    # ==================== 应用信息 ====================
    APP_NAME = "HSI Toolkit"
    APP_VERSION = "0.1.0"
    APP_AUTHOR = "HSI Lab"
    
    # ==================== 路径配置 ====================
    BASE_DIR = Path(__file__).parent
    ASSETS_DIR = BASE_DIR / "assets"
    LOG_DIR = BASE_DIR / "logs"
    OUTPUT_DIR = BASE_DIR / "output"
    
    # ==================== 分割参数默认值 ====================
    SEGMENTATION = {
        'otsu': {
            'threshold': 0.98,
            'min_area': 100,
            'max_area': 10000,
            'erosion_size': 3,
            'dilation_size': 3
        },
        'watershed': {
            'threshold': 0.5,
            'min_area': 100,
            'max_area': 10000,
            'min_distance': 20,
            'erosion_size': 3,
            'dilation_size': 3
        }
    }
    
    # ==================== 预处理参数 ====================
    PREPROCESS = {
        'smooth': {
            'MA': {'window_size': 5},
            'SG': {'window_length': 11, 'polyorder': 2}
        },
        'derivative': {
            'window_length': 11,
            'polyorder': 2
        }
    }
    
    # ==================== 导出设置 ====================
    EXPORT = {
        'image': {
            'dpi': 300,
            'format': 'png',
            'bbox_inches': 'tight',
            'facecolor': 'white'
        },
        'publication': {
            'font_family': 'Times New Roman',
            'font_size': 14,
            'axis_linewidth': 1.0,
            'tick_labelsize': 12
        },
        'envi': {
            'data_type': 4,  # float32
            'interleave': 'bil',
            'byte_order': 0
        }
    }
    
    # ==================== UI配置 ====================
    UI = {
        'window_title': 'HSI Toolkit v0.1 - 高光谱图像分析工具',
        'window_size': (1400, 900),
        'status_bar_timeout': 3000  # ms
    }
    
    # ==================== ROI颜色配置 ====================
    ROI_COLORS = [
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
    ]
    
    # ==================== 光谱颜色配置 ====================
    SPECTRUM_COLORS = ROI_COLORS  # 与ROI颜色一致
    
    # ==================== 日志配置 ====================
    LOGGING = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'standard',
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': str(LOG_DIR / 'hsi_toolkit.log'),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'encoding': 'utf-8'
            }
        },
        'root': {
            'level': 'DEBUG',
            'handlers': ['console', 'file']
        }
    }
    
    # ==================== 数据验证配置 ====================
    VALIDATION = {
        'max_cube_size_mb': 2000,  # 最大数据立方体大小 (MB)
        'max_bands': 1000,  # 最大波段数
        'max_regions': 10000,  # 最大区域数
        'reflectance_range': (0, 1.5),  # 反射率合理范围
    }
    
    @classmethod
    def ensure_dirs(cls):
        """确保必要的目录存在"""
        cls.LOG_DIR.mkdir(exist_ok=True, parents=True)
        cls.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    @classmethod
    def get_segmentation_defaults(cls, method='otsu'):
        """获取分割方法的默认参数"""
        return cls.SEGMENTATION.get(method, cls.SEGMENTATION['otsu']).copy()
    
    @classmethod
    def get_export_config(cls, export_type='image'):
        """获取导出配置"""
        return cls.EXPORT.get(export_type, {}).copy()

