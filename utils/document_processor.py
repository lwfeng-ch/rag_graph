# utils/document_processor.py
"""
文档处理器模块

功能：
- 处理用户上传的医疗文档
- 调用 MinerU 解析文档
- 将文档内容向量化并存储到 Qdrant
- 支持多种文档类型（PDF、DOCX、TXT）

使用方式：
    processor = get_document_processor(embedding_model=llm_embedding)
    result = processor.process_and_store(
        user_id="user123",
        file_content=file_bytes,
        filename="report.pdf",
        doc_type="health_report"
    )
"""
import os
import hashlib
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DocumentProcessorConfig:
    """
    文档处理器配置。

    Attributes:
        supported_types: 支持的文件类型
        max_file_size: 最大文件大小（字节）
        chunk_size: 文本分块大小
        chunk_overlap: 分块重叠大小
    """
    supported_types: Dict[str, str] = None
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    chunk_size: int = 500
    chunk_overlap: int = 125

    def __post_init__(self):
        if self.supported_types is None:
            self.supported_types = {
                ".pdf": "application/pdf",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".doc": "application/msword",
                ".txt": "text/plain",
            }


class DocumentProcessor:
    """
    文档处理器类。

    负责处理用户上传的文档，包括：
    - 文件类型验证
    - 文档解析（调用 MinerU）
    - 文本分块
    - 向量化存储
    """

    def __init__(self, embedding_model, config: Optional[DocumentProcessorConfig] = None):
        """
        初始化文档处理器。

        Args:
            embedding_model: Embedding 模型实例
            config: 处理器配置，为 None 时使用默认配置
        """
        self.embedding_model = embedding_model
        self.config = config or DocumentProcessorConfig()
        self._mineru_client = None
        self._vector_store = None
        logger.info("DocumentProcessor 初始化完成")

    def _get_mineru_client(self):
        """
        获取 MinerU 客户端实例（延迟加载）。

        Returns:
            MinerUClient: MinerU 客户端实例
        """
        if self._mineru_client is None:
            try:
                from mineru_client import MinerUClient
                self._mineru_client = MinerUClient()
                logger.info("MinerU 客户端初始化成功")
            except Exception as e:
                logger.warning(f"MinerU 客户端初始化失败: {e}，将使用简单文本提取")
                self._mineru_client = None
        return self._mineru_client

    def _get_vector_store(self):
        """
        获取向量存储实例（延迟加载）。

        Returns:
            QdrantVectorStore: 向量存储实例
        """
        if self._vector_store is None:
            try:
                from langchain_qdrant import QdrantVectorStore
                from utils.config import Config
                
                self._vector_store = QdrantVectorStore.from_existing_collection(
                    embedding=self.embedding_model,
                    collection_name=Config.QDRANT_COLLECTION_NAME,
                    url=Config.QDRANT_URL,
                    vector_name="text-dense",
                    sparse_vector_name="text-sparse",
                )
                logger.info("向量存储初始化成功")
            except Exception as e:
                logger.error(f"向量存储初始化失败: {e}")
                raise
        return self._vector_store

    def _calculate_md5(self, content: bytes) -> str:
        """
        计算文件内容的 MD5 哈希值。

        Args:
            content: 文件内容字节

        Returns:
            str: MD5 哈希值
        """
        return hashlib.md5(content).hexdigest()

    def _validate_file(self, filename: str, content: bytes) -> tuple:
        """
        验证文件类型和大小。

        Args:
            filename: 文件名
            content: 文件内容

        Returns:
            tuple: (is_valid, error_message)

        Raises:
            None: 所有错误通过返回值传递
        """
        if not filename:
            return False, "文件名不能为空"

        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        if ext not in self.config.supported_types:
            return False, f"不支持的文件类型: {ext}，支持的类型: {list(self.config.supported_types.keys())}"

        if len(content) > self.config.max_file_size:
            max_mb = self.config.max_file_size / (1024 * 1024)
            return False, f"文件大小超过限制 {max_mb}MB"

        return True, None

    def _extract_text(self, filename: str, content: bytes) -> str:
        """
        从文件中提取文本内容。

        Args:
            filename: 文件名
            content: 文件内容

        Returns:
            str: 提取的文本内容
        """
        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        if ext == ".txt":
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError:
                return content.decode("gbk", errors="ignore")

        mineru_client = self._get_mineru_client()
        if mineru_client:
            try:
                result = mineru_client.convert_file(content, filename)
                if result.get("success"):
                    return result.get("content", "")
                logger.warning(f"MinerU 解析失败: {result.get('error')}")
            except Exception as e:
                logger.warning(f"MinerU 解析异常: {e}")

        logger.warning(f"无法解析文件 {filename}，返回空文本")
        return ""

    def _split_text(self, text: str) -> list:
        """
        将文本分割成块。

        Args:
            text: 待分割的文本

        Returns:
            list: 文本块列表
        """
        if not text:
            return []

        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=["\n\n", "\n", "。", "；", ".", ";", " ", ""],
            )
            chunks = splitter.split_text(text)
            return chunks
        except Exception as e:
            logger.warning(f"文本分割失败: {e}，返回原始文本")
            return [text] if text else []

    def process_and_store(
        self,
        user_id: str,
        file_content: bytes,
        filename: str,
        doc_type: str = "other",
    ) -> Dict[str, Any]:
        """
        处理并存储文档。

        Args:
            user_id: 用户 ID
            file_content: 文件内容字节
            filename: 文件名
            doc_type: 文档类型（health_report, medical_record, lab_report, prescription, other）

        Returns:
            Dict[str, Any]: 处理结果，包含：
                - success: 是否成功
                - file_md5: 文件 MD5 哈希
                - filename: 文件名
                - doc_type: 文档类型
                - chunks_count: 分块数量
                - upload_time: 上传时间
                - error: 错误信息（失败时）

        Example:
            >>> result = processor.process_and_store(
            ...     user_id="user123",
            ...     file_content=file_bytes,
            ...     filename="report.pdf",
            ...     doc_type="health_report"
            ... )
            >>> print(result["success"])
            True
        """
        try:
            is_valid, error_msg = self._validate_file(filename, file_content)
            if not is_valid:
                return {
                    "success": False,
                    "error": error_msg,
                }

            file_md5 = self._calculate_md5(file_content)

            text = self._extract_text(filename, file_content)
            if not text:
                return {
                    "success": False,
                    "error": "无法从文件中提取文本内容",
                }

            chunks = self._split_text(text)
            if not chunks:
                return {
                    "success": False,
                    "error": "文本分块失败",
                }

            vector_store = self._get_vector_store()
            
            metadatas = [
                {
                    "user_id": user_id,
                    "file_md5": file_md5,
                    "filename": filename,
                    "doc_type": doc_type,
                    "chunk_index": i,
                    "upload_time": datetime.now().isoformat(),
                }
                for i in range(len(chunks))
            ]

            vector_store.add_texts(texts=chunks, metadatas=metadatas)

            logger.info(f"文档处理成功: {filename}, MD5: {file_md5}, 分块数: {len(chunks)}")

            return {
                "success": True,
                "file_md5": file_md5,
                "filename": filename,
                "doc_type": doc_type,
                "chunks_count": len(chunks),
                "upload_time": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"文档处理失败: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }


_document_processor: Optional[DocumentProcessor] = None


def get_document_processor(embedding_model=None) -> DocumentProcessor:
    """
    获取文档处理器实例（单例模式）。

    Args:
        embedding_model: Embedding 模型实例，首次调用时必须提供

    Returns:
        DocumentProcessor: 文档处理器实例

    Raises:
        ValueError: 首次调用时未提供 embedding_model

    Example:
        >>> processor = get_document_processor(embedding_model=llm_embedding)
        >>> result = processor.process_and_store(...)
    """
    global _document_processor

    if _document_processor is None:
        if embedding_model is None:
            raise ValueError("首次调用 get_document_processor 必须提供 embedding_model 参数")
        _document_processor = DocumentProcessor(embedding_model=embedding_model)

    return _document_processor
