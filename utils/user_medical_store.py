# utils/user_medical_store.py
"""
用户医疗数据存储模块

功能：
- 管理用户上传的医疗文档元数据
- 提供文档列表查询、删除、统计功能
- 基于 Qdrant 向量数据库实现

使用方式：
    store = get_user_medical_store()
    documents = store.list_documents(user_id="user123", limit=10, offset=0)
    stats = store.get_stats(user_id="user123")
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UserMedicalStoreConfig:
    """
    用户医疗数据存储配置。

    Attributes:
        collection_name: Qdrant 集合名称（独立于系统知识库）
        content_preview_length: 内容预览长度
    """

    collection_name: str = "user_medical_documents"
    content_preview_length: int = 200


class UserMedicalStore:
    """
    用户医疗数据存储类。

    提供用户医疗文档的元数据管理功能，包括：
    - 文档列表查询
    - 文档删除
    - 统计信息查询
    """

    def __init__(self, config: Optional[UserMedicalStoreConfig] = None):
        """
        初始化用户医疗数据存储。

        Args:
            config: 存储配置，为 None 时使用默认配置
        """
        self.config = config or UserMedicalStoreConfig()
        self._qdrant_client = None

    def _get_qdrant_client(self):
        """
        获取 Qdrant 客户端实例（延迟加载）。

        Returns:
            QdrantClient: Qdrant 客户端实例
        """
        if self._qdrant_client is None:
            try:
                from qdrant_client import QdrantClient
                from utils.config import Config

                if Config.QDRANT_URL == ":memory:":
                    self._qdrant_client = QdrantClient(location=":memory:")
                elif Config.QDRANT_URL:
                    self._qdrant_client = QdrantClient(
                        url=Config.QDRANT_URL,
                        api_key=getattr(Config, "QDRANT_API_KEY", None),
                    )
                else:
                    self._qdrant_client = QdrantClient(path=Config.QDRANT_LOCAL_PATH)

                logger.info("Qdrant 客户端初始化成功")
            except Exception as e:
                logger.error(f"Qdrant 客户端初始化失败: {e}")
                raise
        return self._qdrant_client

    def list_documents(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        获取用户的文档列表。

        Args:
            user_id: 用户 ID
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            List[Dict[str, Any]]: 文档列表，每个文档包含：
                - doc_id: 文档 ID
                - filename: 文件名
                - doc_type: 文档类型
                - upload_time: 上传时间
                - file_md5: 文件 MD5
                - content_preview: 内容预览

        Example:
            >>> documents = store.list_documents("user123", limit=10, offset=0)
            >>> for doc in documents:
            ...     print(doc["filename"])
        """
        try:
            client = self._get_qdrant_client()

            from qdrant_client.http import models as qdrant_models

            result = client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="user_id",
                            match=qdrant_models.MatchValue(value=user_id),
                        )
                    ]
                ),
                limit=limit + offset,
                with_payload=True,
                with_vectors=False,
            )

            points = result[0]

            unique_docs = {}
            for point in points:
                payload = point.payload or {}
                file_md5 = payload.get("file_md5")

                if file_md5 and file_md5 not in unique_docs:
                    unique_docs[file_md5] = {
                        "doc_id": str(point.id),
                        "filename": payload.get("filename", "unknown"),
                        "doc_type": payload.get("doc_type", "other"),
                        "upload_time": payload.get("upload_time", ""),
                        "file_md5": file_md5,
                        "content_preview": (payload.get("text", "") or "")[
                            : self.config.content_preview_length
                        ],
                    }

            documents = list(unique_docs.values())
            documents.sort(key=lambda x: x.get("upload_time", ""), reverse=True)

            return documents[offset : offset + limit]

        except Exception as e:
            logger.error(f"获取文档列表失败: {e}", exc_info=True)
            return []

    def delete_document(self, user_id: str, file_md5: str) -> Dict[str, Any]:
        """
        删除用户的指定文档。

        Args:
            user_id: 用户 ID
            file_md5: 文件 MD5 哈希值

        Returns:
            Dict[str, Any]: 删除结果，包含：
                - success: 是否成功
                - deleted_chunks: 删除的分块数量
                - error: 错误信息（失败时）

        Example:
            >>> result = store.delete_document("user123", "abc123...")
            >>> print(f"删除了 {result['deleted_chunks']} 个分块")
        """
        try:
            client = self._get_qdrant_client()

            from qdrant_client.http import models as qdrant_models

            search_result = client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="user_id",
                            match=qdrant_models.MatchValue(value=user_id),
                        ),
                        qdrant_models.FieldCondition(
                            key="file_md5",
                            match=qdrant_models.MatchValue(value=file_md5),
                        ),
                    ]
                ),
                limit=1000,
                with_payload=False,
                with_vectors=False,
            )

            points = search_result[0]
            point_ids = [point.id for point in points]

            if not point_ids:
                return {
                    "success": False,
                    "deleted_chunks": 0,
                    "error": "未找到匹配的文档",
                }

            client.delete(
                collection_name=self.config.collection_name,
                points_selector=qdrant_models.PointIdsList(
                    points=point_ids,
                ),
            )

            logger.info(
                f"删除文档成功: user_id={user_id}, file_md5={file_md5}, chunks={len(point_ids)}"
            )

            return {
                "success": True,
                "deleted_chunks": len(point_ids),
            }

        except Exception as e:
            logger.error(f"删除文档失败: {e}", exc_info=True)
            return {
                "success": False,
                "deleted_chunks": 0,
                "error": str(e),
            }

    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户的文档统计信息。

        Args:
            user_id: 用户 ID

        Returns:
            Dict[str, Any]: 统计信息，包含：
                - user_id: 用户 ID
                - total_documents: 文档总数
                - total_chunks: 分块总数
                - doc_types: 各类型文档数量

        Example:
            >>> stats = store.get_stats("user123")
            >>> print(f"文档总数: {stats['total_documents']}")
        """
        try:
            client = self._get_qdrant_client()

            from qdrant_client.http import models as qdrant_models

            result = client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="user_id",
                            match=qdrant_models.MatchValue(value=user_id),
                        )
                    ]
                ),
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )

            points = result[0]

            unique_files = set()
            doc_types = {}

            for point in points:
                payload = point.payload or {}
                file_md5 = payload.get("file_md5")
                doc_type = payload.get("doc_type", "other")

                if file_md5:
                    unique_files.add(file_md5)

                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

            return {
                "user_id": user_id,
                "total_documents": len(unique_files),
                "total_chunks": len(points),
                "doc_types": doc_types,
            }

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}", exc_info=True)
            return {
                "user_id": user_id,
                "total_documents": 0,
                "total_chunks": 0,
                "doc_types": {},
            }


_user_medical_store: Optional[UserMedicalStore] = None


def get_user_medical_store() -> UserMedicalStore:
    """
    获取用户医疗数据存储实例（单例模式）。

    Returns:
        UserMedicalStore: 用户医疗数据存储实例

    Example:
        >>> store = get_user_medical_store()
        >>> documents = store.list_documents("user123")
    """
    global _user_medical_store

    if _user_medical_store is None:
        _user_medical_store = UserMedicalStore()

    return _user_medical_store
