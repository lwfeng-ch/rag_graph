# utils/config/vectorstore_config.py
"""
向量数据库配置模块

功能：
- 管理 Qdrant 向量数据库配置
- 管理 Chroma 数据库配置（兼容旧代码）
- 支持环境变量覆盖
"""

import os
from typing import Optional


class VectorStoreConfig:
    """
    向量数据库配置类。

    Attributes:
        QDRANT_URL: Qdrant 服务 URL
        QDRANT_API_KEY: Qdrant API Key
        QDRANT_COLLECTION_NAME: Qdrant 集合名称
        QDRANT_LOCAL_PATH: Qdrant 本地存储路径
        CHROMADB_DIRECTORY: Chroma 数据库目录
        CHROMADB_COLLECTION_NAME: Chroma 集合名称
    """

    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY", None)
    QDRANT_COLLECTION_NAME: str = os.getenv(
        "QDRANT_COLLECTION_NAME", "knowledge_base_v2"
    )
    QDRANT_LOCAL_PATH: str = os.getenv("QDRANT_LOCAL_PATH", "qdrantDB")

    CHROMADB_DIRECTORY: str = "chromaDB"
    CHROMADB_COLLECTION_NAME: str = "demo001"

    @classmethod
    def is_memory_mode(cls) -> bool:
        """
        检查是否为内存模式。

        Returns:
            bool: 是否为内存模式
        """
        return cls.QDRANT_URL == ":memory:"

    @classmethod
    def is_remote_mode(cls) -> bool:
        """
        检查是否为远程模式。

        Returns:
            bool: 是否为远程模式
        """
        return cls.QDRANT_URL and cls.QDRANT_URL != ":memory:"
