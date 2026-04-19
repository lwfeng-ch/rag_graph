# config.py
"""
统一配置文件 - MinerU + 知识库构建链路
集中管理所有配置项，包括：
- LangSmith 追踪配置
- MinerU 远程服务配置
- Embedding 模型配置
- Qdrant 向量数据库配置
- Markdown 切分配置
- 文件路径配置
- 日志配置
- 数据库配置
- API 服务配置
- Middleware 配置

使用方式：
- 所有非敏感配置项通过此类定义默认值，无需 .env 文件即可运行
- 敏感信息（API Key 等）可通过 .env 文件覆盖，优先级：.env > 代码默认值
"""

import os
import logging
from dotenv import load_dotenv

# 加载 .env 文件（如存在），用于覆盖敏感信息（API Key 等）
# .env 文件是可选的，不提供时所有配置使用下方代码中的默认值
load_dotenv()

os.environ["NO_PROXY"] = "localhost,127.0.0.1"


class Config:
    """
    统一的配置类，集中管理所有常量。

    所有配置项优先从环境变量（.env）读取，未设置时使用下方代码中的默认值。
    """

    # ============================================================
    # LangSmith 追踪配置（可选）
    # ============================================================
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "ragAgent-Prod")
    LANGCHAIN_ENDPOINT = os.getenv(
        "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
    )

    # ============================================================
    # MinerU 远程服务配置
    # ============================================================
    MINERU_API_URL = os.getenv(
        "MINERU_API_URL",
        "https://600d97fdc573450290c6285c14a1837e--8000.ap-shanghai2.cloudstudio.club",
    )
    MINERU_TIMEOUT = int(os.getenv("MINERU_TIMEOUT", "300"))
    MINERU_PARSE_METHOD = os.getenv("MINERU_PARSE_METHOD", "auto")

    # ============================================================
    # Markdown 切分配置
    # ============================================================
    MARKDOWN_HEADERS = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ]
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 125
    SEPARATORS = ["\n\n", "\n", "。", "；", ".", ";", " ", ""]
    MAX_CHUNK_LENGTH = 1000

    # ============================================================
    # Embedding 模型配置
    # ============================================================
    LLM_TYPE = os.getenv("LLM_TYPE", "qwen")
    EMBEDDING_BATCH_SIZE = 25

    # OpenAI 配置
    OPENAI_API_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

    # Qwen（阿里通义千问）配置
    QWEN_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
    QWEN_EMBEDDING_MODEL = "text-embedding-v1"

    # Ollama 本地配置
    OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1")
    OLLAMA_API_KEY = "ollama"
    OLLAMA_EMBEDDING_MODEL = "bge-m3:latest"

    # OneAPI 配置
    ONEAPI_API_BASE = os.getenv("ONEAPI_API_BASE", "http://localhost:3000/v1")
    ONEAPI_API_KEY = os.getenv("ONEAPI_API_KEY", "")
    ONEAPI_EMBEDDING_MODEL = "text-embedding-v1"

    # ============================================================
    # Qdrant 向量数据库配置
    # ============================================================
    QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "knowledge_base_v2")
    QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "qdrantDB")

    # ============================================================
    # Chroma 数据库配置（兼容旧代码）
    # ============================================================
    CHROMADB_DIRECTORY = "chromaDB"
    CHROMADB_COLLECTION_NAME = "demo001"

    # ============================================================
    # 文件路径配置
    # ============================================================
    INPUT_DIR = "input"
    OUTPUT_DIR = "output"
    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".pptx",
        ".html",
        ".htm",
        ".xlsx",
        ".doc",
        ".ppt",
    }

    # ============================================================
    # 日志配置
    # ============================================================
    LOG_FILE = "output/app.log"
    LOG_LEVEL = logging.INFO
    MAX_BYTES = 5 * 1024 * 1024
    BACKUP_COUNT = 3

    # ============================================================
    # 数据库配置
    # ============================================================
    DB_URI = os.getenv(
        "DB_URI", "postgresql://user:password@localhost:5432/database?sslmode=disable"
    )

    # ============================================================
    # API 服务配置
    # ============================================================
    HOST = "0.0.0.0"
    PORT = 8012

    # ============================================================
    # Gradio WebUI 配置
    # ============================================================
    GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")

    # ============================================================
    # Prompt 文件路径
    # ============================================================
    PROMPT_TEMPLATE_TXT_AGENT = "prompts/prompt_template_agent.txt"
    PROMPT_TEMPLATE_TXT_GRADE = "prompts/prompt_template_grade.txt"
    PROMPT_TEMPLATE_TXT_REWRITE = "prompts/prompt_template_rewrite.txt"
    PROMPT_TEMPLATE_TXT_GENERATE = "prompts/prompt_template_generate.txt"

    # ============================================================
    # Middleware 配置
    # ============================================================
    MW_MAX_MODEL_CALLS = int(os.getenv("MW_MAX_MODEL_CALLS", "10"))
    MW_PII_MODE = os.getenv("MW_PII_MODE", "warn")
    MW_SUMMARIZATION_THRESHOLD = int(os.getenv("MW_SUMMARIZATION_THRESHOLD", "20"))
    MW_SUMMARIZATION_KEEP_RECENT = int(os.getenv("MW_SUMMARIZATION_KEEP_RECENT", "5"))
    MW_TOOL_MAX_RETRIES = int(os.getenv("MW_TOOL_MAX_RETRIES", "2"))
    MW_TOOL_BACKOFF_FACTOR = float(os.getenv("MW_TOOL_BACKOFF_FACTOR", "0.5"))

    # ============================================================
    # 工具方法：根据 LLM_TYPE 获取对应配置
    # ============================================================
    @classmethod
    def get_api_base(cls) -> str:
        """
        根据 LLM_TYPE 返回对应的 API Base URL。

        Returns:
            str: API Base URL
        """
        api_bases = {
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
        api_keys = {
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
        models = {
            "openai": cls.OPENAI_EMBEDDING_MODEL,
            "qwen": cls.QWEN_EMBEDDING_MODEL,
            "ollama": cls.OLLAMA_EMBEDDING_MODEL,
            "oneapi": cls.ONEAPI_EMBEDDING_MODEL,
        }
        return models.get(cls.LLM_TYPE, cls.OPENAI_EMBEDDING_MODEL)

    @classmethod
    def validate_config(cls) -> dict:
        """
        验证必要配置是否已设置，返回配置状态报告。

        Returns:
            dict: 配置状态报告
        """
        issues = []

        if cls.LLM_TYPE not in ["openai", "qwen", "ollama", "oneapi"]:
            issues.append(f"未知的 LLM_TYPE: {cls.LLM_TYPE}")

        if not cls.get_api_key():
            issues.append(f"{cls.LLM_TYPE} API Key 未设置")

        if cls.LANGCHAIN_TRACING_V2 and not cls.LANGCHAIN_API_KEY:
            issues.append("LangSmith 追踪已启用但 API Key 未设置")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "llm_type": cls.LLM_TYPE,
            "mineru_url": cls.MINERU_API_URL,
            "qdrant_url": cls.QDRANT_URL,
            "langsmith_enabled": cls.LANGCHAIN_TRACING_V2,
        }


logging.basicConfig(
    level=Config.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
