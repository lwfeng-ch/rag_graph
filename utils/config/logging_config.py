# utils/config/logging_config.py
"""
日志配置模块

功能：
- 管理日志相关配置
- 提供日志初始化方法
"""

import logging
import os


class LoggingConfig:
    """
    日志配置类。

    Attributes:
        LOG_FILE: 日志文件路径
        LOG_LEVEL: 日志级别
        MAX_BYTES: 日志文件最大字节数
        BACKUP_COUNT: 日志备份数量
    """

    LOG_FILE: str = "output/app.log"
    LOG_LEVEL: int = logging.INFO
    MAX_BYTES: int = 5 * 1024 * 1024
    BACKUP_COUNT: int = 3

    @classmethod
    def setup_logging(cls) -> None:
        """
        初始化日志配置（委托给 utils.logger 统一管理）。
        """
        try:
            from utils.logger import setup_logger

            setup_logger(
                name="root",
                log_file=cls.LOG_FILE,
                level=cls.LOG_LEVEL,
                max_bytes=cls.MAX_BYTES,
                backup_count=cls.BACKUP_COUNT,
            )
        except ImportError:
            logging.basicConfig(
                level=cls.LOG_LEVEL,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                force=True,
            )
