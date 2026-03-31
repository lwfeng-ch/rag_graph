import os
import re
import json
from contextlib import asynccontextmanager
from typing import List, Tuple
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
import sys
import time
import uuid
from typing import Optional
from pydantic import BaseModel, Field
from ragAgent_v1 import (
    ToolConfig,
    create_graph_v1,
    save_graph_visualization,
    get_llm,
    get_tools,
    Config,
    Context,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.handlers = []
handler = ConcurrentRotatingFileHandler(
    Config.LOG_FILE,
    maxBytes=Config.MAX_BYTES,
    backupCount=Config.BACKUP_COUNT
)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(handler)


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False
    userId: Optional[str] = None
    conversationId: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseChoice]
    system_fingerprint: Optional[str] = None


def format_response(response):
    paragraphs = re.split(r'\n{2,}', response)
    formatted_paragraphs = []
    for para in paragraphs:
        if '```' in para:
            parts = para.split('```')
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    parts[i] = f"\n```\n{part.strip()}\n```\n"
            para = ''.join(parts)
        else:
            para = para.replace('. ', '.\n')
        formatted_paragraphs.append(para.strip())
    return '\n\n'.join(formatted_paragraphs)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph, tool_config
    try:
        llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)
        tools = get_tools(llm_embedding)
        tool_config = ToolConfig(tools)

        graph = create_graph_v1(llm_chat, llm_embedding, tool_config, use_middleware=True)
        save_graph_visualization(graph, filename="graph_v1.png")
        
        logger.info("LangChain v1 agent initialized successfully with middleware support")
    except ValueError as ve:
        logger.error(f"Value error in response processing: {ve}")
    except Exception as e:
        logger.error(f"Error processing response: {e}")
        
    yield
    logger.info("The v1 service has been shut down")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "智能分诊系统后端服务 v1", "version": "LangChain v1"}


async def handle_non_stream_response(user_input, graph, tool_config, config):
    content = None
    try:
        events = graph.stream({"messages": [{"role": "user", "content": user_input}], "rewrite_count": 0}, config)
        for event in events:
            for value in event.values():
                if "messages" not in value or not isinstance(value["messages"], list):
                    logger.warning("No valid messages in response")
                    continue

                last_message = value["messages"][-1]

                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    for tool_call in last_message.tool_calls:
                        if isinstance(tool_call, dict) and "name" in tool_call:
                            logger.info(f"Calling tool: {tool_call['name']}")
                    continue

                if hasattr(last_message, "content"):
                    content = last_message.content

                    if hasattr(last_message, "name") and last_message.name in tool_config.get_tool_names():
                        tool_name = last_message.name
                        logger.info(f"Tool Output [{tool_name}]: {content}")
                    else:
                        logger.info(f"Final Response is: {content}")
                        
                        if hasattr(last_message, 'content_blocks'):
                            logger.info(f"[v1] Response contains {len(last_message.content_blocks)} content blocks")
                else:
                    logger.info("Message has no content, skipping")
    except ValueError as ve:
        logger.error(f"Value error in response processing: {ve}")
    except Exception as e:
        logger.error(f"Error processing response: {e}")

    formatted_response = str(format_response(content)) if content else "No response generated"
    logger.info(f"Results for Formatting: {formatted_response}")

    try:
        response = ChatCompletionResponse(
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=formatted_response),
                    finish_reason="stop"
                )
            ]
        )
    except Exception as resp_error:
        logger.error(f"Error creating response object: {resp_error}")
        response = ChatCompletionResponse(
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content="Error generating response"),
                    finish_reason="error"
                )
            ]
        )

    logger.info(f"Send response content: \n{response}")
    return JSONResponse(content=response.model_dump())


async def handle_stream_response(user_input, graph, config):
    async def generate_stream():
        try:
            chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
            stream_data = graph.stream(
                {"messages": [{"role": "user", "content": user_input}], "rewrite_count": 0},
                config,
                stream_mode="messages"
            )
            for message_chunk, metadata in stream_data:
                try:
                    node_name = metadata.get("langgraph_node") if metadata else None
                    if node_name in ["generate", "agent"]:
                        chunk = getattr(message_chunk, 'content', '')
                        logger.info(f"Streaming chunk from {node_name}: {chunk}")
                        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"
                except Exception as chunk_error:
                    logger.error(f"Error processing stream chunk: {chunk_error}")
                    continue

            yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        except Exception as stream_error:
            logger.error(f"Stream generation error: {stream_error}")
            yield f"data: {json.dumps({'error': 'Stream processing failed'})}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


async def get_dependencies() -> Tuple[any, any]:
    if not graph or not tool_config:
        raise HTTPException(status_code=500, detail="Service not initialized")
    return graph, tool_config


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, dependencies: Tuple[any, any] = Depends(get_dependencies)):
    try:
        graph, tool_config = dependencies
        if not request.messages or not request.messages[-1].content:
            logger.error("Invalid request: Empty or invalid messages")
            raise HTTPException(status_code=400, detail="Messages cannot be empty or invalid")
        user_input = request.messages[-1].content
        logger.info(f"[v1] The user's user_input is: {user_input}")

        user_id = request.userId if request.userId else "unknown"
        conversation_id = request.conversationId if request.conversationId else "default"
        
        config = {
            "configurable": {
                "thread_id": f"{user_id}@@{conversation_id}",
                "user_id": user_id,
                "conversation_id": conversation_id
            }
        }

        if request.stream:
            return await handle_stream_response(user_input, graph, config)
        return await handle_non_stream_response(user_input, graph, tool_config, config)

    except Exception as e:
        logger.error(f"Error handling chat completion:\n\n {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info(f"Start v1 server on port {Config.PORT}")
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
