"""
混合检索器模块

功能：
- 实现两阶段检索（粗排 + 精排）
- 支持 Qdrant 混合检索（BM25 + 向量）
- 集成 Reranker 精排模型

遵循 LangChain Tool 最佳实践：
- 清晰的模块职责
- 统一的接口设计
- 完善的错误处理
"""

import os
import logging
from typing import List, Union

from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.tools import BaseTool
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
    AsyncCallbackManagerForRetrieverRun,
)

from utils.config import Config
from utils.llms import get_reranker

logger = logging.getLogger(__name__)


class RerankRetriever(BaseRetriever):
    """
    粗排 + 精排 两阶段检索器。

    工作流程:
        query ──► base_retriever (混合检索, Top-k 粗排)
              ──► base_compressor (重排模型, Top-n 精排)
              ──► 最终文档列表
    """

    base_retriever: BaseRetriever
    """第一阶段：粗排检索器（如 Qdrant 混合检索）"""

    base_compressor: BaseDocumentCompressor
    """第二阶段：精排压缩器（如 DashScope Reranker）"""

    class Config:
        arbitrary_types_allowed = True  # BaseDocumentCompressor 含非标准 Pydantic 类型

    # ── 同步入口 ──
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """同步检索：粗排 → 精排"""

        # 第一阶段：粗排召回
        docs = self.base_retriever.invoke(query)
        logger.debug(f"[RerankRetriever] 粗排召回 {len(docs)} 篇文档")

        if not docs:
            logger.warning("[RerankRetriever] 粗排召回 0 篇文档，跳过重排")
            return []

        # 第二阶段：精排重排
        compressed = self.base_compressor.compress_documents(docs, query)
        result = list(compressed)
        logger.debug(f"[RerankRetriever] 精排后保留 {len(result)} 篇文档")
        return result

    # ── 异步入口 ──
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """异步检索：粗排 → 精排（适配 LangGraph 异步调度）"""

        docs = await self.base_retriever.ainvoke(query)
        logger.debug(f"[RerankRetriever] 异步粗排召回 {len(docs)} 篇文档")

        if not docs:
            logger.warning("[RerankRetriever] 异步粗排召回 0 篇文档，跳过重排")
            return []

        compressed = await self.base_compressor.acompress_documents(docs, query)
        result = list(compressed)
        logger.debug(f"[RerankRetriever] 异步精排后保留 {len(result)} 篇文档")
        return result


def create_hybrid_retriever(llm_embedding, llm_type: str = "qwen") -> BaseRetriever:
    """
    创建混合检索器（两阶段检索：粗排 + 精排）。

    Args:
        llm_embedding: 嵌入模型实例，用于初始化向量存储
        llm_type: LLM 供应商类型，传递给 get_reranker()（默认 "qwen"）

    Returns:
        BaseRetriever: 两阶段检索器实例

    Raises:
        RuntimeError: 如果 Qdrant 初始化失败
    """
    try:
        # 1. 挂载稀疏向量模型（生成本地 BM25 特征的基建）
        os.environ["FASTEMBED_CACHE_PATH"] = os.path.abspath("model/model/sparsemodel")
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        # 2. 组装 Qdrant 连接参数
        qdrant_kwargs = {
            "embedding": llm_embedding,
            "sparse_embedding": sparse_embeddings,
            "collection_name": Config.QDRANT_COLLECTION_NAME,
            "retrieval_mode": RetrievalMode.HYBRID,
            "vector_name": "text-dense",
            "sparse_vector_name": "text-sparse",
        }

        # 动态处理 Qdrant 的三种部署环境
        if Config.QDRANT_URL == ":memory:":
            logger.info("Initializing Qdrant in memory mode.")
            qdrant_kwargs["location"] = ":memory:"
        elif Config.QDRANT_URL:
            logger.info(f"Connecting to Qdrant server at {Config.QDRANT_URL}")
            qdrant_kwargs["url"] = Config.QDRANT_URL
            qdrant_kwargs["api_key"] = getattr(Config, "QDRANT_API_KEY", None)
        else:
            logger.info(f"Using local Qdrant path: {Config.QDRANT_LOCAL_PATH}")
            qdrant_kwargs["path"] = Config.QDRANT_LOCAL_PATH

        # 初始化向量存储
        vectorstore = QdrantVectorStore.from_existing_collection(**qdrant_kwargs)

    except Exception as e:
        logger.error(f"Qdrant 初始化失败: {e}")
        raise RuntimeError(
            f"检索库连接失败，请确认集合 {Config.QDRANT_COLLECTION_NAME} 是否存在。"
        ) from e

    # 3. 构建粗排引擎：混合检索召回 Top 5
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 4. 构建精排引擎：调用 llms.py 的工厂方法获取 DashScope 重排器，截取 Top 3
    compressor = get_reranker(llm_type=llm_type, top_n=3)

    # 5. 构建两阶段检索器：粗排 → 精排
    final_retriever = RerankRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor,
    )

    logger.info("混合检索器创建成功")
    return final_retriever


def create_retriever_tool_from_retriever(retriever: BaseRetriever) -> BaseTool:
    """
    从检索器创建检索工具。

    Args:
        retriever: 检索器实例

    Returns:
        BaseTool: 检索工具
    """
    # 封装成 Agent 检索工具
    # 注意：名称必须包含 "retrieve"，适配 ragAgent.py 的路由判断逻辑
    retriever_tool = create_retriever_tool(
        retriever,
        name="health_record_retriever",
        description=(
            "查询用户健康档案和病史信息。"
            "适用场景：需要获取用户的体检数据、病史记录、健康档案等信息时。"
            "输入：医学术语、关键词或短语。"
        ),
    )

    logger.info("检索工具创建成功")
    return retriever_tool
