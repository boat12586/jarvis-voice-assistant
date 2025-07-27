"""
Enhanced Logging System for Jarvis Voice Assistant
Provides comprehensive error tracking and debugging capabilities
"""

import logging
import traceback
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import inspect
from functools import wraps


class JarvisLogger:
    """Enhanced logger with structured error tracking"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        
        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
        
        # Error tracking
        self.error_count = 0
        self.warning_count = 0
        self.errors_log = []
    
    def _setup_handlers(self):
        """Setup logging handlers"""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        
        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"{self.name.replace('.', '_')}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
        )
        file_handler.setFormatter(file_format)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def _get_caller_info(self) -> Dict[str, Any]:
        """Get information about the calling function"""
        frame = inspect.currentframe()
        try:
            # Go up 3 frames: _get_caller_info -> log_method -> actual_caller
            caller_frame = frame.f_back.f_back.f_back
            if caller_frame:
                return {
                    "file": Path(caller_frame.f_code.co_filename).name,
                    "function": caller_frame.f_code.co_name,
                    "line": caller_frame.f_lineno
                }
        finally:
            del frame
        return {"file": "unknown", "function": "unknown", "line": 0}
    
    def debug(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log debug message with context"""
        caller = self._get_caller_info()
        self.logger.debug(f"[{caller['function']}:{caller['line']}] {message}")
        
        if extra_data:
            self.logger.debug(f"Extra data: {json.dumps(extra_data, indent=2)}")
    
    def info(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log info message"""
        self.logger.info(message)
        if extra_data:
            self.logger.debug(f"Extra data: {json.dumps(extra_data, indent=2)}")
    
    def warning(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log warning with tracking"""
        self.warning_count += 1
        caller = self._get_caller_info()
        
        warning_info = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "caller": caller,
            "extra_data": extra_data
        }
        
        self.logger.warning(f"[WARNING #{self.warning_count}] {message}")
        if extra_data:
            self.logger.warning(f"Context: {json.dumps(extra_data, indent=2)}")
    
    def error(self, message: str, exception: Optional[Exception] = None, 
              extra_data: Optional[Dict[str, Any]] = None):
        """Log error with full tracking"""
        self.error_count += 1
        caller = self._get_caller_info()
        
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_id": f"ERR_{self.error_count:04d}",
            "message": message,
            "caller": caller,
            "exception": None,
            "traceback": None,
            "extra_data": extra_data
        }
        
        if exception:
            error_info["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception)
            }
            error_info["traceback"] = traceback.format_exc()
        
        self.errors_log.append(error_info)
        
        # Log to file
        self.logger.error(f"[ERROR #{self.error_count}] {message}")
        if exception:
            self.logger.error(f"Exception: {type(exception).__name__}: {exception}")
            self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        if extra_data:
            self.logger.error(f"Context: {json.dumps(extra_data, indent=2)}")
    
    def critical(self, message: str, exception: Optional[Exception] = None,
                 extra_data: Optional[Dict[str, Any]] = None):
        """Log critical error"""
        self.error(f"CRITICAL: {message}", exception, extra_data)
        self.logger.critical(message)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return {
            "logger_name": self.name,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "recent_errors": self.errors_log[-10:] if self.errors_log else []
        }
    
    def export_errors(self, filepath: str):
        """Export all errors to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "logger": self.name,
                "exported_at": datetime.now().isoformat(),
                "total_errors": self.error_count,
                "total_warnings": self.warning_count,
                "errors": self.errors_log
            }, f, indent=2, ensure_ascii=False)


def with_error_handling(logger: Optional[JarvisLogger] = None, 
                       return_on_error: Any = None,
                       reraise: bool = False):
    """Decorator for automatic error handling and logging"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = logger or JarvisLogger(f"decorator.{func.__name__}")
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                func_logger.error(
                    f"Function {func.__name__} failed", 
                    exception=e,
                    extra_data={
                        "args": str(args)[:200],  # Limit size
                        "kwargs": str(kwargs)[:200]
                    }
                )
                
                if reraise:
                    raise
                return return_on_error
        
        return wrapper
    return decorator


class ComponentLogger(JarvisLogger):
    """Specialized logger for component-level tracking"""
    
    def __init__(self, component_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(f"jarvis.{component_name}", config)
        self.component_name = component_name
        self.operations_count = 0
        self.performance_metrics = []
    
    def operation_start(self, operation: str) -> str:
        """Start tracking an operation"""
        operation_id = f"{self.component_name}_{operation}_{self.operations_count}"
        self.operations_count += 1
        
        self.debug(f"Starting operation: {operation}", {"operation_id": operation_id})
        return operation_id
    
    def operation_end(self, operation_id: str, success: bool = True, 
                     metrics: Optional[Dict[str, Any]] = None):
        """End tracking an operation"""
        if success:
            self.info(f"Operation completed: {operation_id}")
        else:
            self.warning(f"Operation failed: {operation_id}")
        
        if metrics:
            self.performance_metrics.append({
                "operation_id": operation_id,
                "timestamp": datetime.now().isoformat(),
                "success": success,
                **metrics
            })
    
    def get_component_stats(self) -> Dict[str, Any]:
        """Get component-specific statistics"""
        base_stats = self.get_stats()
        base_stats.update({
            "component": self.component_name,
            "operations_count": self.operations_count,
            "recent_metrics": self.performance_metrics[-10:]
        })
        return base_stats


# Global logger factory
_loggers: Dict[str, JarvisLogger] = {}

def get_logger(name: str, config: Optional[Dict[str, Any]] = None) -> JarvisLogger:
    """Get or create a logger instance"""
    if name not in _loggers:
        _loggers[name] = JarvisLogger(name, config)
    return _loggers[name]

def get_component_logger(component: str, config: Optional[Dict[str, Any]] = None) -> ComponentLogger:
    """Get or create a component logger"""
    logger_name = f"component.{component}"
    if logger_name not in _loggers:
        _loggers[logger_name] = ComponentLogger(component, config)
    return _loggers[logger_name]

def export_all_errors(directory: str):
    """Export errors from all loggers"""
    Path(directory).mkdir(exist_ok=True)
    
    for name, logger in _loggers.items():
        if logger.error_count > 0:
            filepath = Path(directory) / f"{name.replace('.', '_')}_errors.json"
            logger.export_errors(str(filepath))