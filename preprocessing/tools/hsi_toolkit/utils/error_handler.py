"""
错误处理 - 统一的异常处理机制
"""
import functools
import traceback
from PySide6.QtWidgets import QMessageBox
from .logger import get_logger

logger = get_logger(__name__)


class AppError(Exception):
    """应用程序自定义异常基类"""
    def __init__(self, message, details=None):
        super().__init__(message)
        self.message = message
        self.details = details or ""


class DataError(AppError):
    """数据相关错误"""
    pass


class FileError(AppError):
    """文件操作错误"""
    pass


class ProcessError(AppError):
    """处理流程错误"""
    pass


def handle_errors(title="错误", show_dialog=True):
    """
    统一错误处理装饰器
    
    Args:
        title: 错误对话框标题
        show_dialog: 是否显示错误对话框
    
    Usage:
        @handle_errors(title="数据加载错误")
        def load_data(self, path):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            
            except FileNotFoundError as e:
                error_msg = f"找不到文件: {str(e)}"
                logger.error(error_msg)
                if show_dialog:
                    QMessageBox.critical(None, f"{title} - 文件错误", error_msg)
                raise FileError(error_msg) from e
            
            except PermissionError as e:
                error_msg = f"没有文件访问权限: {str(e)}"
                logger.error(error_msg)
                if show_dialog:
                    QMessageBox.critical(None, f"{title} - 权限错误", error_msg)
                raise FileError(error_msg) from e
            
            except ValueError as e:
                error_msg = f"数据格式错误: {str(e)}"
                logger.error(error_msg)
                if show_dialog:
                    QMessageBox.warning(None, f"{title} - 数据错误", error_msg)
                raise DataError(error_msg) from e
            
            except IndexError as e:
                error_msg = f"索引超出范围: {str(e)}"
                logger.error(error_msg)
                if show_dialog:
                    QMessageBox.warning(None, f"{title} - 索引错误", error_msg)
                raise DataError(error_msg) from e
            
            except MemoryError as e:
                error_msg = "内存不足，请关闭其他程序或减小数据量"
                logger.error(f"内存错误: {str(e)}")
                if show_dialog:
                    QMessageBox.critical(None, f"{title} - 内存错误", error_msg)
                raise
            
            except AppError as e:
                # 应用程序自定义错误
                logger.error(f"应用错误: {e.message}")
                if e.details:
                    logger.debug(f"错误详情: {e.details}")
                if show_dialog:
                    QMessageBox.warning(None, title, e.message)
                raise
            
            except Exception as e:
                # 未知错误
                error_msg = f"发生未知错误: {str(e)}"
                error_trace = traceback.format_exc()
                logger.exception(error_msg)
                logger.debug(f"错误堆栈:\n{error_trace}")
                
                if show_dialog:
                    QMessageBox.critical(
                        None,
                        f"{title} - 未知错误",
                        f"{error_msg}\n\n详细信息已记录到日志文件"
                    )
                raise
        
        return wrapper
    return decorator


def handle_errors_silent(func):
    """
    静默错误处理装饰器（不显示对话框，只记录日志）
    
    Usage:
        @handle_errors_silent
        def some_function(self):
            ...
    """
    return handle_errors(show_dialog=False)(func)


class ErrorContext:
    """
    错误上下文管理器
    
    Usage:
        with ErrorContext("数据处理"):
            # 处理代码
            pass
    """
    def __init__(self, operation, show_dialog=True):
        self.operation = operation
        self.show_dialog = show_dialog
        self.logger = get_logger(__name__)
    
    def __enter__(self):
        self.logger.debug(f"开始操作: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.logger.debug(f"操作成功完成: {self.operation}")
            return True
        
        # 有异常发生
        error_msg = f"{self.operation}失败: {str(exc_val)}"
        self.logger.error(error_msg)
        
        if self.show_dialog:
            if isinstance(exc_val, AppError):
                QMessageBox.warning(None, f"{self.operation}错误", exc_val.message)
            else:
                QMessageBox.critical(None, f"{self.operation}错误", error_msg)
        
        return False  # 不抑制异常

