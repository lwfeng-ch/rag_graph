# vectorSave.py
# 功能说明：将PDF文件进行向量计算并持久化存储到向量数据库（Qdrant）
import os
import logging
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
import uuid
from utils import pdfSplitTest_Ch
from utils import pdfSplitTest_En

os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# GPT大模型 OpenAI相关配置
OPENAI_API_BASE = os.getenv("OPENAI_BASE_URL")
OPENAI_EMBEDDING_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# 国产大模型 OneAPI相关配置,通义千问为例
ONEAPI_API_BASE = "http://139.224.72.218:3000/v1"
ONEAPI_EMBEDDING_API_KEY = "sk-GseYmJ8pX1D4I32323506e8fDf514a51A3C4B714FfD45aD9"
ONEAPI_EMBEDDING_MODEL = "text-embedding-v1"

# 阿里通义千问大模型
QWen_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWen_EMBEDDING_API_KEY = os.getenv("DASHSCOPE_API_KEY")
QWen_EMBEDDING_MODEL = "text-embedding-v1"

# 本地开源大模型 vLLM 方式
# 本地开源大模型 xinference 方式
# 本地开源大模型 Ollama 方式,bge-m3为例
OLLAMA_API_BASE = "http://localhost:11434/v1"
OLLAMA_EMBEDDING_API_KEY = "ollama"
OLLAMA_EMBEDDING_MODEL = "bge-m3:latest"


# openai:调用gpt模型, qwen:调用阿里通义千问大模型, oneapi:调用oneapi方案支持的模型, ollama:调用本地开源大模型
llmType = "qwen"

# 设置测试文本类型 Chinese 或 English
TEXT_LANGUAGE = 'Chinese'
INPUT_PDF = "input/健康档案.pdf"
# TEXT_LANGUAGE = 'English'
# INPUT_PDF = "input/deepseek-v3-1-4.pdf"

# 指定文件中待处理的页码，全部页码则填None
PAGE_NUMBERS=None
# PAGE_NUMBERS=[2, 3]

# ===== Qdrant 向量数据库配置 (替换原 ChromaDB 配置) =====
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
# Qdrant API密钥，本地部署可为 None
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
# Qdrant 集合名称（对应原 CHROMADB_COLLECTION_NAME）
QDRANT_COLLECTION_NAME = "demo001"
# Qdrant 本地持久化路径（当使用本地模式时生效，对应原 CHROMADB_DIRECTORY）
QDRANT_LOCAL_PATH = "qdrantDB"


# get_embeddings方法计算向量
def get_embeddings(texts):
    global llmType
    global ONEAPI_API_BASE, ONEAPI_EMBEDDING_API_KEY, ONEAPI_EMBEDDING_MODEL
    global OPENAI_API_BASE, OPENAI_EMBEDDING_API_KEY, OPENAI_EMBEDDING_MODEL
    global QWen_API_BASE, QWen_EMBEDDING_API_KEY, QWen_EMBEDDING_MODEL
    global OLLAMA_API_BASE, OLLAMA_EMBEDDING_API_KEY, OLLAMA_EMBEDDING_MODEL
    if llmType == 'oneapi':
        try:
            client = OpenAI(
                base_url=ONEAPI_API_BASE,
                api_key=ONEAPI_EMBEDDING_API_KEY
            )
            data = client.embeddings.create(input=texts,model=ONEAPI_EMBEDDING_MODEL).data
            return [x.embedding for x in data]
        except Exception as e:
            logger.info(f"生成向量时出错: {e}")
            return []
    elif llmType == 'qwen':
        try:
            client = OpenAI(
                base_url=QWen_API_BASE,
                api_key=QWen_EMBEDDING_API_KEY
            )
            data = client.embeddings.create(input=texts,model=QWen_EMBEDDING_MODEL).data
            return [x.embedding for x in data]
        except Exception as e:
            logger.info(f"生成向量时出错: {e}")
            return []
    elif llmType == 'ollama':
        try:
            client = OpenAI(
                base_url=OLLAMA_API_BASE,
                api_key=OLLAMA_EMBEDDING_API_KEY
            )
            data = client.embeddings.create(input=texts,model=OLLAMA_EMBEDDING_MODEL).data
            return [x.embedding for x in data]
        except Exception as e:
            logger.info(f"生成向量时出错: {e}")
            return []
    else:
        try:
            client = OpenAI(
                base_url=OPENAI_API_BASE,
                api_key=OPENAI_EMBEDDING_API_KEY
            )
            data = client.embeddings.create(input=texts,model=OPENAI_EMBEDDING_MODEL).data
            return [x.embedding for x in data]
        except Exception as e:
            logger.info(f"生成向量时出错: {e}")
            return []


# 对文本按批次进行向量计算
def generate_vectors(data, max_batch_size=25):
    results = []
    for i in range(0, len(data), max_batch_size):
        batch = data[i:i + max_batch_size]
        # 调用向量生成get_embeddings方法  根据调用的API不同进行选择
        response = get_embeddings(batch)
        results.extend(response)
    return results


# 封装向量数据库Qdrant类，提供两种方法（替换原 MyVectorDBConnector 基于 ChromaDB 的实现）
class MyVectorDBConnector:

    def __init__(self, collection_name, embedding_fn):
        """
        初始化 Qdrant 向量数据库连接器
        
        Args:
            collection_name: 集合名称
            embedding_fn: 向量处理函数，接收文本列表返回向量列表
            
        LangChain v1 变更说明：
            - 原使用 chromadb.PersistentClient 替换为 qdrant_client.QdrantClient
            - 支持服务器模式（url）和本地持久化模式（path）
        """
        # 申明使用全局变量
        global QDRANT_URL, QDRANT_API_KEY, QDRANT_LOCAL_PATH
        
        # 实例化一个 Qdrant 客户端对象
        # 优先使用服务器模式，如未配置则使用本地持久化模式
        try:
            if QDRANT_URL and QDRANT_URL != ":memory:":
                # 服务器模式：连接到 Qdrant 服务器
                self.client = QdrantClient(
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY,
                    prefer_grpc=False,  # ⚠️ 强制使用 HTTP REST，避免 502
                )
                logger.info(f"连接到 Qdrant 服务器: {QDRANT_URL}")
            else:
                # 本地持久化模式：使用本地文件系统存储
                self.client = QdrantClient(
                    path=QDRANT_LOCAL_PATH,
                    prefer_grpc=False,  # 这里也加上，避免潜在冲突
                )
                logger.info(f"使用 Qdrant 本地持久化模式: {QDRANT_LOCAL_PATH}")
        except Exception as e:
            logger.error(f"连接 Qdrant 失败: {e}")
            raise
        
        # 保存集合名称
        self.collection_name = collection_name
        # embedding处理函数
        self.embedding_fn = embedding_fn
        
        # 初始化时检查集合是否存在，标记用于后续创建
        self._collection_initialized = False

    def _ensure_collection(self, vector_size):
        """确保集合存在，如不存在则创建
        
        Args:
            vector_size: 向量维度大小
        """
        if self._collection_initialized:
            return
            
        try:
            # 检查集合是否已存在
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                # 创建新集合，配置向量参数
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=vector_size,
                        distance=qdrant_models.Distance.COSINE
                    )
                )
                logger.info(f"创建 Qdrant 集合: {self.collection_name}, 向量维度: {vector_size}")
            else:
                logger.info(f"Qdrant 集合已存在: {self.collection_name}")
            
            self._collection_initialized = True
        except Exception as e:
            logger.error(f"确保集合存在时出错: {e}")
            raise

    # 添加文档到集合（替换原 ChromaDB 的 add_documents 方法）
    # 文档通常包括文本数据和其对应的向量表示，这些向量可以用于后续的搜索和相似度计算
    def add_documents(self, documents):
        """
        将文档添加到 Qdrant 集合
        
        Args:
            documents: 文本文档列表
        """
        # 调用函数计算出文档中文本数据对应的向量
        embeddings = self.embedding_fn(documents)
        
        if not embeddings:
            logger.error("向量计算结果为空，无法添加文档")
            return
        
        # 确保集合存在（根据第一个向量的维度创建）
        self._ensure_collection(len(embeddings[0]))
        
        # 构建 Qdrant 点数据列表
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point = qdrant_models.PointStruct(
                id=str(uuid.uuid4()),  # 文档的唯一标识符 自动生成uuid
                vector=embedding,       # 文档对应的向量数据
                payload={
                    "document": doc,    # 文档的文本数据
                    "page_content": doc  # 兼容 LangChain 的 page_content 字段
                }
            )
            points.append(point)
        
        # 批量写入 Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"成功向 Qdrant 集合 '{self.collection_name}' 添加 {len(points)} 个文档")
        
    def clear_collection(self, clear=False):
        """
        强制删除当前集合（用于重新灌库清空脏数据）
        """
        if clear:
            try:
                # 检查集合是否存在
                collections = self.client.get_collections().collections
                collection_names = [c.name for c in collections]
                
                if self.collection_name in collection_names:
                    # 删除集合
                    self.client.delete_collection(collection_name=self.collection_name)
                    logger.info(f"✅ 成功删除旧集合: {self.collection_name}，历史脏数据已清空。")
                    # 状态重置，确保下次调用 add_documents 时会重新创建集合
                    self._collection_initialized = False 
                else:
                    logger.info(f"集合 {self.collection_name} 不存在，无需清理。")
            except Exception as e:
                logger.error(f"清理集合时出错: {e}")   
        
    
    # 检索向量数据库，返回包含查询结果的对象或列表（替换原 ChromaDB 的 search 方法）
    # query：查询文本
    # top_n：返回与查询向量最相似的前 n 个向量
    def search(self, query, top_n):
        """
        检索向量数据库
        
        Args:
            query: 查询文本
            top_n: 返回最相似的前 n 个结果
            
        Returns:
            dict: 包含 'documents' 和 'distances' 的结果字典，格式兼容原 ChromaDB 输出
        """
        try:
            # 计算查询文本的向量
            query_embedding = self.embedding_fn([query])
            if not query_embedding:
                logger.error("查询向量计算失败")
                return {"documents": [[]], "distances": [[]]}
            
            # 在 Qdrant 中进行相似度检索
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding[0],
                limit=top_n,
                with_payload=True,
            )
            
            # 将 Qdrant 结果转换为兼容原 ChromaDB 的格式
            documents = []
            distances = []
            for point in search_results.points:
                doc_text = point.payload.get("document", point.payload.get("page_content", ""))
                documents.append(doc_text)
                distances.append(point.score)
            
            return {
                "documents": [documents],
                "distances": [distances]
            }
        except Exception as e:
            logger.error(f"检索向量数据库时出错: {e}")
            return {"documents": [[]], "distances": [[]]}



# 封装文本预处理及灌库方法, 提供外部调用
def vectorStoreSave():
    global TEXT_LANGUAGE, QDRANT_COLLECTION_NAME, INPUT_PDF, PAGE_NUMBERS

    # 测试中文文本
    if TEXT_LANGUAGE == 'Chinese':
        # 1、获取处理后的文本数据
        # 演示测试对指定的全部页进行处理，其返回值为划分为段落的文本列表
        paragraphs = pdfSplitTest_Ch.getParagraphs(
            filename=INPUT_PDF,
            page_numbers=PAGE_NUMBERS,
            min_line_length=1
        )
        # 2、将文本片段灌入向量数据库（Qdrant）
        # 实例化一个向量数据库对象
        # 其中，传参collection_name为集合名称, embedding_fn为向量处理函数
        vector_db = MyVectorDBConnector(QDRANT_COLLECTION_NAME, generate_vectors)
        vector_db.clear_collection(clear=True)
        # 向向量数据库中添加文档（文本数据、文本数据对应的向量数据）
        vector_db.add_documents(paragraphs)
        # 3、封装检索接口进行检索测试
        user_query = "张三九的基本信息是什么"
        # 将检索出的3个近似的结果
        search_results = vector_db.search(user_query, 3)
        logger.info(f"检索向量数据库的结果: {search_results}")

    # 测试英文文本
    elif TEXT_LANGUAGE == 'English':
        # 1、获取处理后的文本数据
        # 演示测试对指定的全部页进行处理，其返回值为划分为段落的文本列表
        paragraphs = pdfSplitTest_En.getParagraphs(
            filename=INPUT_PDF,
            page_numbers=PAGE_NUMBERS,
            min_line_length=1
        )
        # 2、将文本片段灌入向量数据库（Qdrant）
        # 实例化一个向量数据库对象
        # 其中，传参collection_name为集合名称, embedding_fn为向量处理函数
        vector_db = MyVectorDBConnector(QDRANT_COLLECTION_NAME, generate_vectors)
        # 向向量数据库中添加文档（文本数据、文本数据对应的向量数据）
        vector_db.add_documents(paragraphs)
        # 3、封装检索接口进行检索测试
        user_query = "deepseek V3有多少参数"
        # 将检索出的3个近似的结果
        search_results = vector_db.search(user_query, 3)
        logger.info(f"检索向量数据库的结果: {search_results}")


if __name__ == "__main__":
    # 测试文本预处理及灌库
    vectorStoreSave()
    
    

