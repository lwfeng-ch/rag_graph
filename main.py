# main.py
"""
FastAPI API 服务入口 — 适配 ragAgent_copy.py 双路由架构（General RAG + Medical Agent）。

适配变更：
- 使用 get_rag_tools / get_medical_agent_tools + ToolConfig(rag_tools=, medical_tools=)
- 消费 final_payload（安全警告、免责声明、分诊建议）
- 流式/非流式均兼容 general 和 medical 两条业务线路
"""

import os
import re
import json
from contextlib import asynccontextmanager
from typing import List, Tuple
from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    UploadFile,
    File,
    Query,
    Form,
    Header,
)
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
import uuid
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from ragAgent import (
    ToolConfig,
    create_graph,
    extract_graph_response,
    RagAgentError,
    ResponseExtractionError,
)
from utils.config import Config
from utils.logger import setup_logger
from utils.llms import get_llm
from utils.tools_config import get_rag_tools, get_medical_agent_tools_with_user_docs
from utils.auth import get_current_user_id, AuthConfig
from utils.user_medical_store import get_user_medical_store
from utils.document_processor import get_document_processor

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

logger = setup_logger(__name__)


#   消息模型
class Message(BaseModel):
    role: str
    content: str


#   聊天请求模型
class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False
    userId: Optional[str] = None
    conversationId: Optional[str] = None


#   聊天响应模型
class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


#   分诊数据模型
class TriageData(BaseModel):
    recommended_departments: List[str] = []
    urgency_level: str = "routine"
    triage_reason: str = ""
    triage_confidence: float = 0.8


#   结构化医疗数据模型
class StructuredMedicalData(BaseModel):
    triage: TriageData
    analysis: Optional[Dict[str, Any]] = None


#   医疗扩展模型
class MedicalExtension(BaseModel):
    risk_level: str = "low"
    risk_warning: str = ""
    disclaimer: str = ""
    structured_data: Optional[StructuredMedicalData] = None


#   聊天完成响应模型
class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseChoice]
    system_fingerprint: Optional[str] = None
    medical: Optional[MedicalExtension] = None


#   文本格式化模型
def format_response(response: str) -> str:
    """对输入的文本进行段落分隔、添加适当的换行符。"""
    paragraphs = re.split(r"\n{2,}", response)
    formatted_paragraphs = []
    for para in paragraphs:
        if "```" in para:
            parts = para.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    parts[i] = f"\n```\n{part.strip()}\n```\n"
            para = "".join(parts)
        else:
            para = para.replace(". ", ".\n")
        formatted_paragraphs.append(para.strip())
    return "\n\n".join(formatted_paragraphs)


# 全局变量
graph = None
tool_config: Optional[ToolConfig] = None
llm_embedding = None  # 全局 embedding model


# Lifespan — 启动/关闭管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时初始化图谱，关闭时清理资源。"""
    global graph, tool_config, llm_embedding
    try:
        llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

        rag_tools = get_rag_tools(llm_embedding)
        medical_tools = get_medical_agent_tools_with_user_docs(
            llm_embedding=llm_embedding,
            llm_type=Config.LLM_TYPE,
            include_user_docs=True,
        )
        tool_config = ToolConfig(rag_tools=rag_tools, medical_tools=medical_tools)

        graph = create_graph(llm_chat, llm_embedding, tool_config)
        logger.info("服务初始化完成，图谱已就绪")
        logger.info("医疗 Agent 已启用用户医疗文档检索功能")

    except Exception as e:
        logger.error(f"服务初始化失败: {e}", exc_info=True)
        raise

    yield
    logger.info("服务已关闭")


app = FastAPI(lifespan=lifespan)


# ============================================================
# CORS 中间件配置
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite 开发服务器
        "http://127.0.0.1:5173",
        "http://localhost:5174",  # Vite 自动递增端口
        "http://127.0.0.1:5174",
        "http://localhost:7860",  # Gradio WebUI
        "http://127.0.0.1:7860",
        "http://localhost:7861",
        "http://127.0.0.1:7861",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)


# ============================================================
# 健康检查
# ============================================================
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "智能分诊系统后端服务"}


# ============================================================
# final_payload 感知的事件处理器（核心修复 P0 #1）
# ============================================================
def _build_medical_extension(final_payload: dict) -> Optional[MedicalExtension]:
    """
    从 final_payload 构建 MedicalExtension 对象（HTTP API 层专用）。

    Args:
        final_payload: medical_safety_guard 节点生成的结构化载荷

    Returns:
        Optional[MedicalExtension]: 医疗扩展信息，非 medical 路由时返回 None
    """
    if not final_payload or final_payload.get("route") != "medical":
        return None

    medical_ext = MedicalExtension(
        risk_level=final_payload.get("risk_level", "low"),
        risk_warning=final_payload.get("risk_warning", ""),
        disclaimer=final_payload.get("disclaimer", ""),
        structured_data=None,
    )

    raw_structured = final_payload.get("structured_data", {})
    if raw_structured:
        try:
            triage_raw = raw_structured.get("triage", {})
            triage_obj = TriageData(
                recommended_departments=triage_raw.get("recommended_departments", []),
                urgency_level=triage_raw.get("urgency_level", "routine"),
                triage_reason=triage_raw.get("triage_reason", ""),
                triage_confidence=triage_raw.get("triage_confidence", 0.8),
            )
            medical_ext.structured_data = StructuredMedicalData(
                triage=triage_obj,
                analysis=raw_structured.get("analysis"),
            )
        except Exception as e:
            logger.warning(f"解析 final_payload.structured_data 失败: {e}")
            medical_ext.structured_data = None

    return medical_ext


def _extract_response_from_events(events) -> Tuple[str, Optional[MedicalExtension]]:
    """
    遍历 graph.stream 事件，提取响应文本和医疗扩展信息。

    内部委托给 ragAgent.extract_graph_response() 进行核心事件解析，
    本函数仅负责 HTTP API 层的 MedicalExtension 构建逻辑。

    Returns:
        (content_text, medical_extension_or_none)

    Raises:
        ResponseExtractionError: 响应提取失败时抛出
    """
    content_text, final_payload = extract_graph_response(events)
    medical_ext = _build_medical_extension(final_payload)
    return content_text, medical_ext


# ============================================================
# 非流式响应
# ============================================================
async def handle_non_stream_response(user_input: str, graph, config: dict):
    try:
        events = graph.stream(
            {"messages": [{"role": "user", "content": user_input}], "rewrite_count": 0},
            config,
        )

        content_text, medical_ext = _extract_response_from_events(events)

        formatted_text = format_response(content_text)
        logger.info(f"Final Response: {formatted_text[:300]}")

        choices = [
            ChatCompletionResponseChoice(
                index=0,
                message=Message(role="assistant", content=formatted_text),
                finish_reason="stop",
            )
        ]

        return ChatCompletionResponse(choices=choices, medical=medical_ext).model_dump()

    except ResponseExtractionError as ree:
        logger.error(f"非流式响应提取失败: {ree.to_dict()}")
        raise HTTPException(
            status_code=502,
            detail={"error": "响应提取失败", "code": ree.code, "message": ree.message},
        )
    except RagAgentError as rae:
        logger.error(f"RagAgent 异常 [{rae.code}]: {rae.message}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Agent 处理异常",
                "code": rae.code,
                "message": rae.message,
            },
        )


# ============================================================
# 流式响应
# ============================================================
async def handle_stream_response(user_input: str, graph, config: dict):
    async def generate_stream():
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"

        try:
            stream_data = graph.stream(
                {
                    "messages": [{"role": "user", "content": user_input}],
                    "rewrite_count": 0,
                },
                config,
                stream_mode=["messages", "values"],
            )
        except Exception as e:
            logger.error(f"流式请求启动失败: {e}", exc_info=True)
            error_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"[ERROR] 流式请求启动失败: {str(e)}"},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            return

        medical_nodes = {
            "medical_agent",
            "medical_analysis",
            "department_triage",
            "medical_safety_guard",
            "generate",
        }
        general_nodes = {"agent", "generate"}
        all_valid_nodes = medical_nodes | general_nodes
        final_payload_collected: Optional[Dict] = None
        error_count = 0
        MAX_STREAM_ERRORS = 10

        for event in stream_data:
            try:
                if isinstance(event, tuple) and len(event) == 2:
                    event_type, event_data = event

                    if event_type == "messages":
                        message_chunk, metadata = event_data
                        node_name = metadata.get("langgraph_node") if metadata else None
                        chunk = getattr(message_chunk, "content", "")

                        if node_name in all_valid_nodes and chunk:
                            logger.debug(f"Stream [{node_name}]: {chunk[:120]}")
                            data = {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": chunk},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

                    elif event_type == "values":
                        if (
                            isinstance(event_data, dict)
                            and "final_payload" in event_data
                        ):
                            final_payload_collected = event_data["final_payload"]
                            if final_payload_collected:
                                route = final_payload_collected.get("route", "unknown")
                                logger.info(
                                    f"流式模式检测到 final_payload，route={route}"
                                )

            except Exception as e:
                error_count += 1
                logger.error(
                    f"Stream chunk error ({error_count}/{MAX_STREAM_ERRORS}): {e}"
                )
                if error_count >= MAX_STREAM_ERRORS:
                    logger.error(f"流式错误超过阈值 {MAX_STREAM_ERRORS}，终止流")
                    break
                continue

        if (
            final_payload_collected
            and final_payload_collected.get("route") == "medical"
        ):
            medical_ext = _build_medical_extension(final_payload_collected)
            if medical_ext:
                medical_event = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"medical": medical_ext.model_dump()},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(medical_event, ensure_ascii=False)}\n\n"
                logger.info("流式模式已发送医疗扩展信息")

        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


# ============================================================
# 依赖注入
# ============================================================
async def get_dependencies() -> Tuple[Any, Any]:
    if not graph:
        raise HTTPException(status_code=503, detail="图谱未初始化，服务不可用")
    if not tool_config:
        raise HTTPException(status_code=503, detail="工具配置未初始化")
    if not hasattr(graph, "invoke"):
        raise HTTPException(status_code=500, detail="图谱实例异常（缺少 invoke 方法）")
    return graph, tool_config


# ============================================================
# 路由
# ============================================================
@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    dependencies: Tuple[Any, Any] = Depends(get_dependencies),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
):
    """
    聊天完成端点

    认证方式：
    1. API Key（Header: X-API-Key）- 服务间调用
    2. JWT Token（Header: Authorization）- 前端用户
    3. 开发模式（请求体 userId）- 仅开发环境

    安全约束：
    - user_id 必须从认证体系获取，不能从请求体直接读取
    - 防止用户伪造 user_id 查询其他用户数据
    """
    try:
        g, tc = dependencies
        if not request.messages or not request.messages[-1].content:
            raise HTTPException(
                status_code=400, detail="Messages cannot be empty or invalid"
            )

        user_input = request.messages[-1].content
        logger.info(f"User input: {user_input}")

        # 安全获取 user_id（从认证体系）
        user_id = get_current_user_id(
            x_api_key=x_api_key,
            authorization=authorization,
            request_user_id=request.userId,
        )

        conversation_id = request.conversationId or "default"

        config = {
            "configurable": {
                "thread_id": f"{user_id}@@{conversation_id}",
                "user_id": user_id,
                "conversation_id": conversation_id,
            }
        }

        if request.stream:
            return await handle_stream_response(user_input, g, config)
        return await handle_non_stream_response(user_input, g, config)

    except HTTPException:
        raise
    except ResponseExtractionError as ree:
        logger.error(f"聊天完成响应提取失败: {ree.to_dict()}")
        raise HTTPException(
            status_code=502,
            detail={"error": "响应提取失败", "code": ree.code, "message": ree.message},
        )
    except RagAgentError as rae:
        logger.error(
            f"聊天完成 RagAgent 异常 [{rae.code}]: {rae.message}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Agent 处理异常",
                "code": rae.code,
                "message": rae.message,
                "details": rae.details,
            },
        )
    except Exception as e:
        logger.error(f"Error handling chat completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 文档管理 API
# ============================================================


class DocumentUploadResponse(BaseModel):
    """文档上传响应"""

    success: bool
    file_md5: Optional[str] = None
    filename: Optional[str] = None
    doc_type: Optional[str] = None
    chunks_count: Optional[int] = None
    upload_time: Optional[str] = None
    error: Optional[str] = None


class DocumentInfo(BaseModel):
    """文档信息"""

    doc_id: str
    filename: str
    doc_type: str
    upload_time: str
    file_md5: str
    content_preview: str


class DocumentListResponse(BaseModel):
    """文档列表响应"""

    user_id: str
    total: int
    documents: List[DocumentInfo]


class DocumentDeleteResponse(BaseModel):
    """文档删除响应"""

    success: bool
    file_md5: str
    deleted_chunks: int
    error: Optional[str] = None


class DocumentStatsResponse(BaseModel):
    """文档统计响应"""

    user_id: str
    total_documents: int
    total_chunks: int
    doc_types: Dict[str, int]


@app.post("/v1/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str = Form("other"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
):
    """
    上传文档

    支持的文件类型：
    - PDF (.pdf)
    - Word (.docx)
    - 文本文件 (.txt)

    文档类型（doc_type）：
    - health_report: 体检报告
    - medical_record: 病历
    - lab_report: 检验报告
    - prescription: 处方
    - other: 其他

    认证方式：
    1. API Key（Header: X-API-Key）
    2. JWT Token（Header: Authorization）
    3. 开发模式（请求体 userId）
    """
    try:
        # 安全获取 user_id
        user_id = get_current_user_id(x_api_key=x_api_key, authorization=authorization)

        # 读取文件内容
        file_content = await file.read()
        filename = file.filename or "unknown"

        # 获取 embedding model
        if not llm_embedding:
            raise HTTPException(status_code=503, detail="服务未初始化")

        # 获取文档处理器
        processor = get_document_processor(embedding_model=llm_embedding)

        # 处理并存储文档
        result = processor.process_and_store(
            user_id=user_id,
            file_content=file_content,
            filename=filename,
            doc_type=doc_type,
        )

        return DocumentUploadResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档上传失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")


@app.get("/v1/documents", response_model=DocumentListResponse)
async def list_documents(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
):
    """
    获取文档列表

    认证方式：
    1. API Key（Header: X-API-Key）
    2. JWT Token（Header: Authorization）
    """
    try:
        # 安全获取 user_id
        user_id = get_current_user_id(x_api_key=x_api_key, authorization=authorization)

        # 获取文档列表
        store = get_user_medical_store()
        documents = store.list_documents(user_id, limit=limit, offset=offset)

        # 转换为响应格式
        doc_infos = [DocumentInfo(**doc) for doc in documents]

        return DocumentListResponse(
            user_id=user_id, total=len(doc_infos), documents=doc_infos
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档列表失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@app.delete("/v1/documents/{file_md5}", response_model=DocumentDeleteResponse)
async def delete_document(
    file_md5: str,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
):
    """
    删除文档

    认证方式：
    1. API Key（Header: X-API-Key）
    2. JWT Token（Header: Authorization）
    """
    try:
        # 安全获取 user_id
        user_id = get_current_user_id(x_api_key=x_api_key, authorization=authorization)

        # 删除文档
        store = get_user_medical_store()
        deleted_chunks = store.delete_file(user_id, file_md5)

        if deleted_chunks == -1:
            return DocumentDeleteResponse(
                success=False, file_md5=file_md5, deleted_chunks=0, error="删除失败"
            )

        return DocumentDeleteResponse(
            success=True, file_md5=file_md5, deleted_chunks=deleted_chunks
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")


@app.get("/v1/documents/stats", response_model=DocumentStatsResponse)
async def get_document_stats(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
):
    """
    获取文档统计信息

    认证方式：
    1. API Key（Header: X-API-Key）
    2. JWT Token（Header: Authorization）
    """
    try:
        # 安全获取 user_id
        user_id = get_current_user_id(x_api_key=x_api_key, authorization=authorization)

        # 获取统计信息
        store = get_user_medical_store()
        stats = store.get_collection_stats(user_id)

        return DocumentStatsResponse(
            user_id=user_id,
            total_documents=stats.get("total_documents", 0),
            total_chunks=stats.get("total_chunks", 0),
            doc_types=stats.get("doc_types", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档统计失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取文档统计失败: {str(e)}")


if __name__ == "__main__":
    logger.info(f"Starting server on {Config.HOST}:{Config.PORT}")
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
