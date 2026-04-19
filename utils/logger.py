# utils/logger.py
"""
统一日志配置模块

功能：
- 提供统一的日志配置接口
- 支持控制台和文件输出
- 支持日志轮转（按大小）
- 支持多进程安全写入
- 支持通过环境变量配置日志级别

使用方式：
    from utils.logger import setup_logger
    logger = setup_logger(__name__)
    logger.info("日志消息")

完整数据流：
    调用 setup_logger("rag.agent")
        │
        ├─→ getLogger("rag.agent")   # 从全局注册表取
        │
        ├─→ 检查 handlers → 已有则返回
        │
        ├─→ 设置 level + formatter
        │
        ├─→ 添加 ConsoleHandler      # 始终添加
        │
        ├─→ 创建日志目录
        │       └─→ 失败 → warning + 仅返回控制台Logger
        │
        └─→ 添加 FileHandler         # 按环境选择类型
                └─→ 失败 → warning + 仅保留控制台Logger


日志调用：logger.info("消息")
        │
        ├─→ ConsoleHandler → 打印到终端
        └─→ FileHandler   → 写入 logs/app.log
                                └─→ 超过10MB → 轮转
"""

import os
import logging
import logging.config
from typing import Optional

try:
    from concurrent_log_handler import ConcurrentRotatingFileHandler

    _HAS_CONCURRENT_HANDLER = True  # 标记：支持多进程安全写入
except ImportError:
    from logging.handlers import RotatingFileHandler

    _HAS_CONCURRENT_HANDLER = False  # 降级：使用标准库的 RotatingFileHandler


LOG_DIR = "logs"
LOG_FILE = "app.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5
# 日志格式示例输出：
# 2024-01-15 14:30:00 - utils.logger - INFO - 这是一条日志
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def _get_log_level() -> int:
    """
    从环境变量获取日志级别。

    Returns:
        int: 日志级别常量
    """
    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    return _LOG_LEVEL_MAP.get(level_str, logging.INFO)


def setup_logger(
    name: str,
    level: Optional[int] = None,
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    创建并配置一个 Logger 实例。

    Args:
        name: Logger 名称，通常使用 __name__
        level: 日志级别，默认从环境变量 LOG_LEVEL 读取
        log_dir: 日志目录，默认为 'logs'
        log_file: 日志文件名，默认为 'app.log'

    Returns:
        logging.Logger: 配置好的 Logger 实例

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("这是一条日志消息")
    """
    logger = logging.getLogger(name)  # 从全局注册表获取Logger实例

    if logger.handlers:
        return logger

    if level is None:
        level = _get_log_level()

    logger.setLevel(level)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    console_handler = logging.StreamHandler()  # 输出到 stderr（默认）
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_dir = log_dir or LOG_DIR
    log_file = log_file or LOG_FILE

    if not os.path.exists(log_dir):
        try:
            os.makedirs(
                log_dir, exist_ok=True
            )  # exist_ok=True：并发时多进程同时创建不会报错
        except (OSError, PermissionError) as e:
            logger.warning(f"无法创建日志目录 {log_dir}: {e}")
            return logger  # ← 降级处理：仅保留控制台输出处理器，不崩溃

    log_path = os.path.join(log_dir, log_file)

    try:
        if _HAS_CONCURRENT_HANDLER:
            # 多进程安全
            file_handler = ConcurrentRotatingFileHandler(
                log_path,
                maxBytes=LOG_MAX_BYTES,
                backupCount=LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
        else:
            # 标准库的 RotatingFileHandler单线程安全
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=LOG_MAX_BYTES,
                backupCount=LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (OSError, PermissionError) as e:
        logger.warning(f"无法创建日志文件处理器: {e}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取已配置的 Logger 实例。

    Args:
        name: Logger 名称

    Returns:
        logging.Logger: Logger 实例
    """
    return logging.getLogger(name)
