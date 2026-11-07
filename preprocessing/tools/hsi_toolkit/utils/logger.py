"""
日志系统 - 统一的日志管理
"""
import logging
import logging.handlers
from pathlib import Path
from config import Config

_loggers = {}

def setup_logger(name='HSI_Toolkit'):
    """
    设置日志系统
    
    Args:
        name: 日志记录器名称
    
    Returns:
        logger: 配置好的日志记录器
    """
    # 确保日志目录存在
    Config.ensure_dirs()
    
    # 如果已经设置过，直接返回
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # 如果已有处理器，不重复添加
    if logger.handlers:
        _loggers[name] = logger
        return logger
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # 创建文件处理器（带日志轮转）
    log_file = Config.LOG_DIR / 'hsi_toolkit.log'
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # 缓存
    _loggers[name] = logger
    
    logger.info(f"日志系统初始化完成 - {name}")
    return logger


def get_logger(name=None):
    """
    获取日志记录器
    
    Args:
        name: 模块名称，默认使用调用者的模块名
    
    Returns:
        logger: 日志记录器
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'HSI_Toolkit')
    
    if name not in _loggers:
        return setup_logger(name)
    
    return _loggers[name]


# 创建默认日志记录器
default_logger = setup_logger()

