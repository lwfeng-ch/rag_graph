# utils/config/base_config.py
"""
基础配置模块

功能：
- 提供 LangSmith 追踪配置
- 提供 Markdown 切分配置
- 提供文件路径配置
- 组合所有子配置（兼容层）
"""
import os
import logging
from typing import Dict, List, Tuple, Any

from dotenv import load_dotenv

from utils.config.llm_config import LLMConfig
from utils.config.vectorstore_config import VectorStoreConfig
from utils.config.middleware_config import MiddlewareConfig
from utils.config.service_config import ServiceConfig
from utils.config.logging_config import LoggingConfig

load_dotenv()

os.environ['NO_PROXY'] = 'localhost,127.0.0.1'


class LangSmithConfig:
    """
    LangSmith 追踪配置类。

    Attributes:
        LANGCHAIN_TRACING_V2: 是否启用 LangSmith 追踪
        LANGCHAIN_API_KEY: LangSmith API Key
        LANGCHAIN_PROJECT: LangSmith 项目名称
        LANGCHAIN_ENDPOINT: LangSmith 端点 URL
    """

    LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "ragAgent-Prod")
    LANGCHAIN_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    @classmethod
    def validate(cls) -> Dict[str, Any]:
        """
        验证 LangSmith 配置。

        Returns:
            Dict[str, Any]: 验证结果
        """
        issues = []

        if cls.LANGCHAIN_TRACING_V2 and not cls.LANGCHAIN_API_KEY:
            issues.append("LangSmith 追踪已启用但 API Key 未设置")

        return {
            "enabled": cls.LANGCHAIN_TRACING_V2,
            "valid": len(issues) == 0,
            "issues": issues,
        }


class MarkdownConfig:
    """
    Markdown 切分配置类。

    Attributes:
        MARKDOWN_HEADERS: Markdown 标题层级配置
        CHUNK_SIZE: 切分块大小
        CHUNK_OVERLAP: 切分重叠大小
        SEPARATORS: 分隔符列表
        MAX_CHUNK_LENGTH: 最大块长度
    """

    MARKDOWN_HEADERS: List[Tuple[str, str]] = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ]
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 125
    SEPARATORS: List[str] = ["\n\n", "\n", "。", "；", ".", ";", " ", ""]
    MAX_CHUNK_LENGTH: int = 1000


class FilePathConfig:
    """
    文件路径配置类。

    Attributes:
        INPUT_DIR: 输入目录
        OUTPUT_DIR: 输出目录
        SUPPORTED_EXTENSIONS: 支持的文件扩展名
    """

    INPUT_DIR: str = "input"
    OUTPUT_DIR: str = "output"
    SUPPORTED_EXTENSIONS: set = {".pdf", ".docx", ".pptx", ".html", ".htm", ".xlsx", ".doc", ".ppt"}


class Config(
    LangSmithConfig,
    LLMConfig,
    VectorStoreConfig,
    MiddlewareConfig,
    ServiceConfig,
    LoggingConfig,
    MarkdownConfig,
    FilePathConfig
):
    """
    统一配置类（兼容层）。

    组合所有子配置，保持向后兼容。
    现有代码无需修改导入语句。

    Example:
        >>> from utils.config import Config
        >>> print(Config.LLM_TYPE)
        qwen
        >>> print(Config.get_api_base())
        https://dashscope.aliyuncs.com/compatible-mode/v1
    """

    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """
        验证所有配置是否有效。

        Returns:
            Dict[str, Any]: 配置状态报告
        """
        issues = []

        llm_result = LLMConfig.validate()
        issues.extend(llm_result.get("issues", []))

        langsmith_result = LangSmithConfig.validate()
        issues.extend(langsmith_result.get("issues", []))

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "llm_type": cls.LLM_TYPE,
            "mineru_url": cls.MINERU_API_URL,
            "qdrant_url": cls.QDRANT_URL,
            "langsmith_enabled": cls.LANGCHAIN_TRACING_V2,
        }
