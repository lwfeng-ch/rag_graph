# utils/tools_config.py
"""
工具配置模块

功能：
- 提供工具工厂函数
- 支持双路由架构（通用 RAG / 医疗 Agent）

注意：
- RerankRetriever 类已移至 retriever.py
- 本模块专注于工具创建和配置
"""

import os
import logging
from typing import List

from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.tools import tool, BaseTool
from langchain_core.retrievers import BaseRetriever

from utils.config import Config
from utils.llms import get_reranker
from utils.retriever import RerankRetriever

logger = logging.getLogger(__name__)


def get_tools(llm_embedding, llm_type: str = "qwen") -> List[BaseTool]:
    """
    创建并返回工具列表，兼容 LangGraph v1 的 ParallelToolNode 调用。

    Args:
        llm_embedding: 嵌入模型实例，用于初始化向量存储
        llm_type: LLM 供应商类型，传递给 get_reranker()（默认 "qwen"）

    Returns:
        List[BaseTool]: 工具列表
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

    # 5. 封装成 Agent 检索工具
    # 注意：名称必须包含 "retrieve"，适配 ragAgent.py 的路由判断逻辑
    retriever_tool = create_retriever_tool(
        final_retriever,
        name="health_record_retriever",
        description=(
            "这是核心健康档案查询工具。内部采用【BM25 关键字 + 向量】混合检索与深度重排 rerank 技术。"
            "当你需要回答有关用户的健康档案、病史、体检数据等信息时，必须使用此工具进行搜索。"
            "输入必须是明确的医学术语、关键词或短语。"
        ),
    )

    return [retriever_tool]


def get_rag_tools(llm_embedding, llm_type: str = "qwen") -> List[BaseTool]:
    """
    获取 RAG Agent 工具列表（仅检索工具）。

    Args:
        llm_embedding: 嵌入模型实例
        llm_type: LLM 供应商类型

    Returns:
        List[BaseTool]: RAG 工具列表
    """
    return get_tools(llm_embedding, llm_type)


def get_medical_agent_tools_with_user_docs(
    llm_embedding,
    llm_type: str = "qwen",
    include_user_docs: bool = True,
) -> List[BaseTool]:
    """
    获取医疗 Agent 工具列表（包含完整工具集）。

    Args:
        llm_embedding: 嵌入模型实例
        llm_type: LLM 供应商类型
        include_user_docs: 是否包含用户医疗文档检索工具

    Returns:
        List[BaseTool]: 医疗 Agent 工具列表

    Example:
        >>> tools = get_medical_agent_tools_with_user_docs(
        ...     llm_embedding=embedding_model,
        ...     llm_type="qwen",
        ...     include_user_docs=True
        ... )
    """
    base_tools = get_tools(llm_embedding, llm_type)

    try:
        from utils.medical_analysis import get_medical_tools

        medical_tools = get_medical_tools()
        base_tools.extend(medical_tools)
        logger.info(f"加载医疗分析工具成功，共 {len(medical_tools)} 个")
    except ImportError as e:
        logger.warning(f"医疗分析工具模块未找到: {e}，将仅使用基础工具")
    except (AttributeError, TypeError) as e:
        logger.warning(f"医疗分析工具加载配置错误: {e}")

    if include_user_docs:
        try:
            user_doc_retriever = _create_user_doc_retriever(llm_embedding)
            if user_doc_retriever:
                user_doc_tool = create_retriever_tool(
                    user_doc_retriever,
                    name="user_medical_document_retriever",
                    description=(
                        "这是用户个人医疗文档检索工具。"
                        "用于查询用户上传的体检报告、病历、检验报告等个人医疗文档。"
                        "输入应为具体的医疗指标名称或症状描述。"
                    ),
                )
                base_tools.append(user_doc_tool)
                logger.info("用户医疗文档检索工具加载成功")
        except (ImportError, AttributeError, ValueError) as e:
            logger.warning(f"用户医疗文档检索工具加载失败: {e}")

    logger.info(f"医疗 Agent 工具列表加载完成，共 {len(base_tools)} 个工具")
    return base_tools


def get_medical_agent_tools(llm_embedding, llm_type: str = "qwen") -> List[BaseTool]:
    """
    获取医疗 Agent 工具列表（简化接口，包含用户文档）。

    这是 get_medical_agent_tools_with_user_docs 的别名，
    用于向后兼容 ragAgent.py 等模块的导入。

    Args:
        llm_embedding: 嵌入模型实例
        llm_type: LLM 供应商类型

    Returns:
        List[BaseTool]: 医疗 Agent 工具列表

    Example:
        >>> tools = get_medical_agent_tools(embedding_model)
        >>> print(len(tools))
    """
    return get_medical_agent_tools_with_user_docs(
        llm_embedding=llm_embedding, llm_type=llm_type, include_user_docs=True
    )


def _create_user_doc_retriever(llm_embedding) -> BaseRetriever | None:
    """
    创建用户医疗文档检索器。

    Args:
        llm_embedding: 嵌入模型实例

    Returns:
        BaseRetriever | None: 用户文档检索器，失败时返回 None
    """
    try:
        from langchain_qdrant import QdrantVectorStore

        os.environ["FASTEMBED_CACHE_PATH"] = os.path.abspath("model/model/sparsemodel")
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        qdrant_kwargs = {
            "embedding": llm_embedding,
            "sparse_embedding": sparse_embeddings,
            "collection_name": Config.QDRANT_COLLECTION_NAME,
            "retrieval_mode": RetrievalMode.HYBRID,
            "vector_name": "text-dense",
            "sparse_vector_name": "text-sparse",
        }

        if Config.QDRANT_URL == ":memory:":
            qdrant_kwargs["location"] = ":memory:"
        elif Config.QDRANT_URL:
            qdrant_kwargs["url"] = Config.QDRANT_URL
            qdrant_kwargs["api_key"] = getattr(Config, "QDRANT_API_KEY", None)
        else:
            qdrant_kwargs["path"] = Config.QDRANT_LOCAL_PATH

        vectorstore = QdrantVectorStore.from_existing_collection(**qdrant_kwargs)

        base_retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 3,
                "filter": {
                    "must": [
                        {
                            "key": "doc_type",
                            "match": {
                                "any": ["health_report", "medical_record", "lab_report"]
                            },
                        }
                    ]
                },
            }
        )

        compressor = get_reranker(llm_type=Config.LLM_TYPE, top_n=2)

        return RerankRetriever(
            base_retriever=base_retriever,
            base_compressor=compressor,
        )

    except Exception as e:
        logger.error(f"创建用户文档检索器失败: {e}")
        return None
