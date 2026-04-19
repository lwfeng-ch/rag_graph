# utils/config/llm_config.py
"""
LLM 供应商配置模块

功能：
- 管理 OpenAI/Qwen/Ollama/OneAPI 配置
- 提供统一的 API 获取方法
- 支持环境变量覆盖
"""
import os
from typing import Dict, Optional


class LLMConfig:
    """
    LLM 供应商配置类。

    Attributes:
        LLM_TYPE: 当前使用的 LLM 供应商类型
        EMBEDDING_BATCH_SIZE: Embedding 批处理大小
        OPENAI_API_BASE: OpenAI API 基础 URL
        OPENAI_API_KEY: OpenAI API Key
        OPENAI_EMBEDDING_MODEL: OpenAI Embedding 模型名称
        QWEN_API_BASE: 通义千问 API 基础 URL
        QWEN_API_KEY: 通义千问 API Key
        QWEN_EMBEDDING_MODEL: 通义千问 Embedding 模型名称
        OLLAMA_API_BASE: Ollama API 基础 URL
        OLLAMA_API_KEY: Ollama API Key（固定值）
        OLLAMA_EMBEDDING_MODEL: Ollama Embedding 模型名称
        ONEAPI_API_BASE: OneAPI 基础 URL
        ONEAPI_API_KEY: OneAPI Key
        ONEAPI_EMBEDDING_MODEL: OneAPI Embedding 模型名称
    """

    LLM_TYPE: str = os.getenv("LLM_TYPE", "qwen")
    EMBEDDING_BATCH_SIZE: int = 25

    OPENAI_API_BASE: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    QWEN_API_BASE: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
    QWEN_EMBEDDING_MODEL: str = "text-embedding-v1"

    OLLAMA_API_BASE: str = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1")
    OLLAMA_API_KEY: str = "ollama"
    OLLAMA_EMBEDDING_MODEL: str = "bge-m3:latest"

    ONEAPI_API_BASE: str = os.getenv("ONEAPI_API_BASE", "http://localhost:3000/v1")
    ONEAPI_API_KEY: str = os.getenv("ONEAPI_API_KEY", "")
    ONEAPI_EMBEDDING_MODEL: str = "text-embedding-v1"

    @classmethod
    def get_api_base(cls) -> str:
        """
        根据 LLM_TYPE 返回对应的 API Base URL。

        Returns:
            str: API Base URL
        """
        api_bases: Dict[str, str] = {
            "openai": cls.OPENAI_API_BASE,
            "qwen": cls.QWEN_API_BASE,
            "ollama": cls.OLLAMA_API_BASE,
            "oneapi": cls.ONEAPI_API_BASE,
        }
        return api_bases.get(cls.LLM_TYPE, cls.OPENAI_API_BASE)

    @classmethod
    def get_api_key(cls) -> str:
        """
        根据 LLM_TYPE 返回对应的 API Key。

        Returns:
            str: API Key
        """
        api_keys: Dict[str, str] = {
            "openai": cls.OPENAI_API_KEY,
            "qwen": cls.QWEN_API_KEY,
            "ollama": cls.OLLAMA_API_KEY,
            "oneapi": cls.ONEAPI_API_KEY,
        }
        return api_keys.get(cls.LLM_TYPE, "")

    @classmethod
    def get_embedding_model(cls) -> str:
        """
        根据 LLM_TYPE 返回对应的 Embedding 模型名称。

        Returns:
            str: Embedding 模型名称
        """
        models: Dict[str, str] = {
            "openai": cls.OPENAI_EMBEDDING_MODEL,
            "qwen": cls.QWEN_EMBEDDING_MODEL,
            "ollama": cls.OLLAMA_EMBEDDING_MODEL,
            "oneapi": cls.ONEAPI_EMBEDDING_MODEL,
        }
        return models.get(cls.LLM_TYPE, cls.OPENAI_EMBEDDING_MODEL)

    @classmethod
    def validate(cls) -> Dict[str, any]:
        """
        验证 LLM 配置是否有效。

        Returns:
            Dict[str, any]: 验证结果
        """
        issues = []

        if cls.LLM_TYPE not in ["openai", "qwen", "ollama", "oneapi"]:
            issues.append(f"未知的 LLM_TYPE: {cls.LLM_TYPE}")

        if not cls.get_api_key():
            issues.append(f"{cls.LLM_TYPE} API Key 未设置")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "llm_type": cls.LLM_TYPE,
        }
