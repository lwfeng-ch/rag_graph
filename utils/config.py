# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """统一的配置类，集中管理所有常量"""
    # prompt文件路径
    PROMPT_TEMPLATE_TXT_AGENT = "prompts/prompt_template_agent.txt"
    PROMPT_TEMPLATE_TXT_GRADE = "prompts/prompt_template_grade.txt"
    PROMPT_TEMPLATE_TXT_REWRITE = "prompts/prompt_template_rewrite.txt"
    PROMPT_TEMPLATE_TXT_GENERATE = "prompts/prompt_template_generate.txt"

    # Chroma 数据库配置
    CHROMADB_DIRECTORY = "chromaDB"
    CHROMADB_COLLECTION_NAME = "demo001"

    # ===== Qdrant 向量数据库配置 (替换原 ChromaDB 配置) =====
    # Qdrant 服务地址，本地部署默认 http://localhost:6333
    # 也可使用 ":memory:" 进行内存模式测试，或指定本地路径进行磁盘持久化
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    # Qdrant API密钥，本地部署可为 None，云服务需要配置
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
    # Qdrant 集合名称（对应原 CHROMADB_COLLECTION_NAME）
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "demo001")
    # Qdrant 本地持久化路径（当使用本地模式时生效，对应原 CHROMADB_DIRECTORY）
    QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "qdrantDB")
    
    # 日志持久化存储
    LOG_FILE = "output/app.log"
    MAX_BYTES=5*1024*1024
    BACKUP_COUNT=3

    # 数据库 URI，默认值
    DB_URI = os.getenv("DB_URI", "postgresql://lwfeng:123456@localhost:5433/postgres?sslmode=disable")

    # openai:调用gpt模型, qwen:调用阿里通义千问大模型, oneapi:调用oneapi方案支持的模型, ollama:调用本地开源大模型
    LLM_TYPE = "qwen"

    # API服务地址和端口
    HOST = "0.0.0.0"
    PORT = 8012