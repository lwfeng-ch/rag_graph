from langchain_qdrant import QdrantVectorStore
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from utils.config import Config


def get_tools(llm_embedding):
    """
    创建并返回工具列表

    Args:
        llm_embedding: 嵌入模型实例，用于初始化向量存储

    Returns:
        list: 工具列表
    """

    if Config.QDRANT_URL and Config.QDRANT_URL != ":memory:":
        # 服务器模式：连接到 Qdrant 服务器（本地部署或云服务）
        vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=llm_embedding,
            collection_name=Config.QDRANT_COLLECTION_NAME,
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
        )
    else:
        # 本地持久化模式：使用本地文件系统存储向量数据
        vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=llm_embedding,
            collection_name=Config.QDRANT_COLLECTION_NAME,
            path=Config.QDRANT_LOCAL_PATH,
        )


    # 将向量存储转换为检索器
    retriever = vectorstore.as_retriever()
    # 创建检索工具
    retriever_tool = create_retriever_tool(
        retriever,
        name="retrieve",
        description="这是健康档案查询工具，搜索并返回有关用户的健康档案信息。"
    )


    # 自定义 multiply 工具
    @tool
    def multiply(a: float, b: float) -> float:
        """这是计算两个数的乘积的工具，返回最终的计算结果"""
        return a * b


    # 返回工具列表
    return [retriever_tool, multiply]