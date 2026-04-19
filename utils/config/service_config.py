# utils/config/service_config.py
"""
服务配置模块

功能：
- 管理 API 服务配置
- 管理 Gradio WebUI 配置
- 管理 MinerU 远程服务配置
- 管理 Prompt 文件路径
"""

import os


class ServiceConfig:
    """
    服务配置类。

    Attributes:
        HOST: API 服务主机
        PORT: API 服务端口
        GRADIO_SERVER_PORT: Gradio 服务端口
        GRADIO_SERVER_NAME: Gradio 服务名称
        MINERU_API_URL: MinerU API URL
        MINERU_TIMEOUT: MinerU 超时时间
        MINERU_PARSE_METHOD: MinerU 解析方法
        PROMPT_TEMPLATE_TXT_AGENT: Agent Prompt 模板路径
        PROMPT_TEMPLATE_TXT_GRADE: Grade Prompt 模板路径
        PROMPT_TEMPLATE_TXT_REWRITE: Rewrite Prompt 模板路径
        PROMPT_TEMPLATE_TXT_GENERATE: Generate Prompt 模板路径
        PROMPT_TEMPLATE_TXT_INTENT_ROUTER: Intent Router Prompt 模板路径（医疗线路）
        PROMPT_TEMPLATE_TXT_MEDICAL_AGENT: Medical Agent Prompt 模板路径
        PROMPT_TEMPLATE_TXT_MEDICAL_AGENT_CB: Medical Agent 断路器 Prompt 模板路径
        PROMPT_TEMPLATE_TXT_MEDICAL_ANALYSIS: Medical Analysis Prompt 模板路径
        PROMPT_CACHE_MAX_SIZE: Prompt 缓存最大容量
        PROMPT_CACHE_TTL: Prompt 缓存有效期（秒）
        DB_URI: 数据库连接 URI
    """

    HOST: str = "0.0.0.0"
    PORT: int = 8012

    GRADIO_SERVER_PORT: int = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    GRADIO_SERVER_NAME: str = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")

    MINERU_API_URL: str = os.getenv(
        "MINERU_API_URL",
        "https://600d97fdc573450290c6285c14a1837e--8000.ap-shanghai2.cloudstudio.club",
    )
    MINERU_TIMEOUT: int = int(os.getenv("MINERU_TIMEOUT", "300"))
    MINERU_PARSE_METHOD: str = os.getenv("MINERU_PARSE_METHOD", "auto")

    PROMPT_TEMPLATE_TXT_AGENT: str = "prompts/prompt_template_agent.txt"
    PROMPT_TEMPLATE_TXT_GRADE: str = "prompts/prompt_template_grade.txt"
    PROMPT_TEMPLATE_TXT_REWRITE: str = "prompts/prompt_template_rewrite.txt"
    PROMPT_TEMPLATE_TXT_GENERATE: str = "prompts/prompt_template_generate.txt"

    PROMPT_TEMPLATE_TXT_INTENT_ROUTER: str = "prompts/prompt_template_intent_router.txt"
    PROMPT_TEMPLATE_TXT_MEDICAL_AGENT: str = "prompts/prompt_template_medical_agent.txt"
    PROMPT_TEMPLATE_TXT_MEDICAL_AGENT_CB: str = (
        "prompts/prompt_template_medical_agent_cb.txt"
    )
    PROMPT_TEMPLATE_TXT_MEDICAL_ANALYSIS: str = (
        "prompts/prompt_template_medical_analysis.txt"
    )

    PROMPT_CACHE_MAX_SIZE: int = int(os.getenv("PROMPT_CACHE_MAX_SIZE", "100"))
    PROMPT_CACHE_TTL: int = int(os.getenv("PROMPT_CACHE_TTL", "300"))

    DB_URI: str = os.getenv(
        "DB_URI", "postgresql://lwfeng:123456@localhost:5433/postgres?sslmode=disable"
    )

    # 并行工具执行配置
    PARALLEL_TOOL_MAX_WORKERS: int = int(os.getenv("PARALLEL_TOOL_MAX_WORKERS", "4"))
    PARALLEL_TOOL_TIMEOUT: int = int(os.getenv("PARALLEL_TOOL_TIMEOUT", "120"))
