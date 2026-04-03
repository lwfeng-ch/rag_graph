# vectorSave.py
"""
向量存储引擎 v2 - 增强版
完整知识库构建链路: 任意格式文件 → MinerU高保真Markdown → 两阶段语义切分 → 带元数据的向量存储

相比 v1 (vectorSave.py) 的改进:
1. 支持 PDF/DOCX/PPTX/HTML 等多种文件格式（通过 MinerU GPU 解析）
2. 两阶段语义切分（标题树 + RecursiveCharacterTextSplitter）
3. 完整元数据保存（标题层级、文件名、来源等）
4. 标题前置拼接提升检索精度
5. 批量处理与缓存机制
"""
import os
import uuid
import logging
from typing import List, Dict, Any, Optional, Callable

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from utils.config import Config
from mineru_client import MinerUClient
from utils.markdown_splitter import MarkdownSplitter

logger = logging.getLogger(__name__)

os.environ['NO_PROXY'] = 'localhost,127.0.0.1'


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    根据 LLM_TYPE 配置调用对应的 Embedding 模型计算向量。

    Args:
        texts: 待计算向量的文本列表

    Returns:
        List[List[float]]: 向量列表，失败返回空列表
    """
    llm_type = Config.LLM_TYPE

    client_config = {
        "openai": {
            "base_url": Config.OPENAI_API_BASE,
            "api_key": Config.OPENAI_API_KEY,
            "model": Config.OPENAI_EMBEDDING_MODEL,
        },
        "qwen": {
            "base_url": Config.QWEN_API_BASE,
            "api_key": Config.QWEN_API_KEY,
            "model": Config.QWEN_EMBEDDING_MODEL,
        },
        "ollama": {
            "base_url": Config.OLLAMA_API_BASE,
            "api_key": Config.OLLAMA_API_KEY,
            "model": Config.OLLAMA_EMBEDDING_MODEL,
        },
        "oneapi": {
            "base_url": Config.ONEAPI_API_BASE,
            "api_key": Config.ONEAPI_API_KEY,
            "model": Config.ONEAPI_EMBEDDING_MODEL,
        },
    }

    config = client_config.get(llm_type)
    if not config:
        logger.error(f"不支持的 LLM_TYPE: {llm_type}")
        return []

    try:
        client = OpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"]
        )
        data = client.embeddings.create(
            input=texts,
            model=config["model"]
        ).data
        return [item.embedding for item in data]
    except Exception as e:
        logger.error(f"生成向量时出错: {e}")
        return []


def generate_vectors(data: List[str], max_batch_size: int = None) -> List[List[float]]:
    """
    对文本按批次进行向量计算，支持批量处理提升效率。

    Args:
        data: 文本列表
        max_batch_size: 每批最大数量，默认从配置读取

    Returns:
        List[List[float]]: 向量列表
    """
    max_batch_size = max_batch_size or Config.EMBEDDING_BATCH_SIZE
    results = []

    for i in range(0, len(data), max_batch_size):
        batch = data[i:i + max_batch_size]
        response = get_embeddings(batch)
        results.extend(response)

    return results


class VectorStoreV2:
    """
    增强版向量数据库连接器（Qdrant）
    
    相比 V1 (MyVectorDBConnector) 的改进:
    - 支持完整元数据保存（标题层级、文件名、来源等）
    - 标题前置拼接的 Embedding 策略
    - 更灵活的数据插入接口
    - 支持带 metadata 的 upsert 操作
    """

    def __init__(
        self,
        collection_name: str = None,
        embedding_fn: Callable = None,
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        qdrant_local_path: str = None
    ):
        """
        初始化向量存储引擎。

        Args:
            collection_name: 集合名称
            embedding_fn: 向量处理函数
            qdrant_url: Qdrant 服务地址
            qdrant_api_key: Qdrant API密钥
            qdrant_local_path: 本地持久化路径
        """
        self.collection_name = collection_name or Config.QDRANT_COLLECTION_NAME
        self.embedding_fn = embedding_fn or generate_vectors
        self._collection_initialized = False

        url = qdrant_url or Config.QDRANT_URL
        api_key = qdrant_api_key or Config.QDRANT_API_KEY
        local_path = qdrant_local_path or Config.QDRANT_LOCAL_PATH

        try:
            if url and url != ":memory:":
                self.client = QdrantClient(
                    url=url,
                    api_key=api_key,
                    prefer_grpc=False,
                )
                logger.info(f"连接到 Qdrant 服务器: {url}")
            else:
                self.client = QdrantClient(
                    path=local_path,
                    prefer_grpc=False,
                )
                logger.info(f"使用 Qdrant 本地持久化模式: {local_path}")
        except Exception as e:
            logger.error(f"连接 Qdrant 失败: {e}")
            raise

    def _ensure_collection(self, vector_size: int):
        """确保集合存在，如不存在则创建"""
        if self._collection_initialized:
            return

        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=vector_size,
                        distance=qdrant_models.Distance.COSINE
                    )
                )
                logger.info(f"创建 Qdrant 集合: {self.collection_name}, 维度: {vector_size}")
            else:
                logger.info(f"Qdrant 集合已存在: {self.collection_name}")

            self._collection_initialized = True
        except Exception as e:
            logger.error(f"确保集合存在时出错: {e}")
            raise

    def add_documents(self, documents: List[str]):
        """
        添加纯文本文档到集合（兼容原接口）。

        Args:
            documents: 文本文档列表
        """
        embeddings = self.embedding_fn(documents)
        if not embeddings:
            logger.error("向量计算结果为空，无法添加文档")
            return

        self._ensure_collection(len(embeddings[0]))

        points = []
        for doc, embedding in zip(documents, embeddings):
            point = qdrant_models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "document": doc,
                    "page_content": doc,
                    "source": "unknown",
                }
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"成功添加 {len(points)} 个文档到集合 '{self.collection_name}'")

    def upsert_with_metadata(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str] = None,
        use_context_prefix: bool = True
    ) -> List[str]:
        """
        【核心方法】带完整元数据的文档插入。

        Args:
            texts: 文本内容列表
            metadatas: 元数据列表（每项对应一个文本的元数据）
            ids: 自定义 ID 列表（可选，自动生成 UUID）
            use_context_prefix: 是否在 Embedding 时使用标题前置拼接

        Returns:
            List[str]: 插入的 ID 列表
        """
        if len(texts) != len(metadatas):
            raise ValueError(f"texts({len(texts)}) 与 metadatas({len(metadatas)}) 数量不匹配")

        embed_texts = texts
        if use_context_prefix:
            splitter = MarkdownSplitter()
            embed_texts = [
                splitter.build_context_string({"content": t, "metadata": m})
                for t, m in zip(texts, metadatas)
            ]

        embeddings = self.embedding_fn(embed_texts)
        if not embeddings:
            logger.error("向量计算结果为空")
            return []

        self._ensure_collection(len(embeddings[0]))

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]

        points = []
        for id_, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
            payload = {
                "document": text,
                "page_content": text,
                **metadata
            }
            point = qdrant_models.PointStruct(
                id=id_,
                vector=embedding,
                payload=payload
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"成功写入 {len(points)} 条数据（含元数据）到 '{self.collection_name}'")
        return ids

    def search(
        self,
        query: str,
        top_n: int = 5,
        query_filter: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        相似性搜索。

        Args:
            query: 查询文本
            top_n: 返回最相似的前 n 个结果
            query_filter: Qdrant 过滤条件（可选）

        Returns:
            dict: {"documents": [[...]], "distances": [[...]]}
        """
        try:
            query_embedding = self.embedding_fn([query])
            if not query_embedding:
                logger.error("查询向量计算失败")
                return {"documents": [[]], "distances": [[]]}

            search_kwargs = {
                "collection_name": self.collection_name,
                "query": query_embedding[0],
                "limit": top_n,
                "with_payload": True,
            }

            if query_filter:
                search_kwargs["query_filter"] = query_filter

            search_results = self.client.query_points(**search_kwargs)

            documents = []
            distances = []
            for point in search_results.points:
                doc_text = point.payload.get(
                    "document",
                    point.payload.get("page_content", "")
                )
                documents.append(doc_text)
                distances.append(point.score)

            return {
                "documents": [documents],
                "distances": [distances]
            }
        except Exception as e:
            logger.error(f"检索向量数据库时出错: {e}")
            return {"documents": [[]], "distances": [[]]}

    def clear_collection(self, clear: bool = False):
        """
        删除当前集合（用于重新灌库清空脏数据）。

        Args:
            clear: 是否执行清理
        """
        if not clear:
            return

        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name in collection_names:
                self.client.delete_collection(collection_name=self.collection_name)
                logger.info(f"已删除旧集合: {self.collection_name}，脏数据已清空")
                self._collection_initialized = False
            else:
                logger.info(f"集合 {self.collection_name} 不存在，无需清理")
        except Exception as e:
            logger.error(f"清理集合时出错: {e}")


class KnowledgeBaseBuilder:
    """
    知识库构建器 - 整合完整链路的顶层入口
    
    链路: 文件 → MinerU转换 → Markdown切分 → 向量化 → 存储
    """

    def __init__(
        self,
        collection_name: str = None,
        mineru_api_url: str = None,
        clear_existing: bool = False
    ):
        """
        初始化知识库构建器。

        Args:
            collection_name: Qdrant 集合名称
            mineru_api_url: MinerU 服务地址
            clear_existing: 是否清空已有数据后重建
        """
        self.mineru_client = MinerUClient(api_url=mineru_api_url)
        self.splitter = MarkdownSplitter()
        self.vector_store = VectorStoreV2(collection_name=collection_name)
        self.clear_existing = clear_existing

    def build_from_file(
        self,
        file_path: str,
        parse_method: str = None
    ) -> Dict[str, Any]:
        """
        从单个文件构建知识库。

        Args:
            file_path: 文件路径
            parse_method: MinerU 解析方法 (auto/ocr/txt)

        Returns:
            dict: 构建结果统计
        """
        logger.info(f"开始构建知识库，源文件: {file_path}")

        result = self.mineru_client.convert_file(file_path, parse_method=parse_method)
        if not result["success"] or not result["markdown"]:
            logger.error(f"MinerU 转换失败: {result.get('error')}")
            return {"success": False, "error": result.get("error"), "chunks_count": 0}

        markdown_text = result["markdown"]
        chunks = self.splitter.split_text(markdown_text)

        if self.clear_existing:
            self.vector_store.clear_collection(clear=True)

        texts = [chunk["content"] for chunk in chunks]
        metadatas = [
            {
                "filename": result.get("filename", ""),
                "source": file_path,
                **chunk.get("metadata", {})
            }
            for chunk in chunks
        ]

        self.vector_store.upsert_with_metadata(texts, metadatas)

        return {
            "success": True,
            "filename": result.get("filename"),
            "chunks_count": len(chunks),
            "markdown_length": len(markdown_text),
        }

    def build_from_directory(
        self,
        input_dir: str = None,
        output_dir: str = None,
        parse_method: str = None
    ) -> Dict[str, Any]:
        """
        从目录批量构建知识库。

        Args:
            input_dir: 输入文件目录
            output_dir: Markdown 缓存输出目录
            parse_method: MinerU 解析方法

        Returns:
            dict: 构建结果统计
        """
        input_dir = input_dir or Config.INPUT_DIR
        md_results = self.mineru_client.convert_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            parse_method=parse_method
        )

        all_chunks = []
        for filename, markdown_text in md_results.items():
            if not markdown_text:
                continue
            chunks = self.splitter.split_text(markdown_text)
            for chunk in chunks:
                chunk["source_filename"] = filename
                all_chunks.append(chunk)

        if self.clear_existing:
            self.vector_store.clear_collection(clear=True)

        texts = [chunk["content"] for chunk in all_chunks]
        metadatas = [
            {
                "filename": chunk.get("source_filename", ""),
                "source": chunk.get("source_filename", ""),
                **chunk.get("metadata", {})
            }
            for chunk in all_chunks
        ]

        self.vector_store.upsert_with_metadata(texts, metadatas)

        total_files = sum(1 for v in md_results.values() if v)
        return {
            "success": True,
            "files_processed": total_files,
            "total_files": len(md_results),
            "chunks_count": len(all_chunks),
        }

    def search(self, query: str, top_n: int = 5) -> Dict[str, Any]:
        """
        搜索知识库。

        Args:
            query: 查询文本
            top_n: 返回结果数量

        Returns:
            dict: 搜索结果
        """
        return self.vector_store.search(query, top_n=top_n)


def vectorStoreSave():
    """
    兼容原 vectorSave.py 接口的入口函数。
    使用新的知识库构建链路替代旧的 pdfminer 方案。
    """
    builder = KnowledgeBaseBuilder(
        collection_name=Config.QDRANT_COLLECTION_NAME,
        clear_existing=True
    )

    input_file = "input/健康档案.pdf"

    if os.path.exists(input_file):
        result = builder.build_from_file(input_file)
        logger.info(f"知识库构建完成: {result}")

        user_query = "张三九的基本信息是什么"
        search_results = builder.search(user_query, top_n=3)
        logger.info(f"检索结果: {search_results}")
    else:
        logger.warning(f"输入文件不存在: {input_file}")


if __name__ == "__main__":
    vectorStoreSave()
