# utils/config/middleware_config.py
"""
中间件配置模块

功能：
- 管理中间件相关配置
- 支持环境变量覆盖
"""
import os


class MiddlewareConfig:
    """
    中间件配置类。

    Attributes:
        MW_MAX_MODEL_CALLS: 最大模型调用次数
        MW_PII_MODE: PII 处理模式
        MW_SUMMARIZATION_THRESHOLD: 摘要阈值
        MW_SUMMARIZATION_KEEP_RECENT: 保留最近消息数
        MW_TOOL_MAX_RETRIES: 工具最大重试次数
        MW_TOOL_BACKOFF_FACTOR: 工具重试退避因子
    """

    MW_MAX_MODEL_CALLS: int = int(os.getenv("MW_MAX_MODEL_CALLS", "10"))
    MW_PII_MODE: str = os.getenv("MW_PII_MODE", "warn")
    MW_SUMMARIZATION_THRESHOLD: int = int(os.getenv("MW_SUMMARIZATION_THRESHOLD", "20"))
    MW_SUMMARIZATION_KEEP_RECENT: int = int(os.getenv("MW_SUMMARIZATION_KEEP_RECENT", "5"))
    MW_TOOL_MAX_RETRIES: int = int(os.getenv("MW_TOOL_MAX_RETRIES", "2"))
    MW_TOOL_BACKOFF_FACTOR: float = float(os.getenv("MW_TOOL_BACKOFF_FACTOR", "0.5"))
