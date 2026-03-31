import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
import os
import sys
import threading
import time
import uuid
from html import escape
from typing import Literal, Annotated, Sequence, Optional
from dataclasses import dataclass
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import BaseMessage
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.store.base import BaseStore
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field
from utils.llms import get_llm
from utils.tools_config import get_tools
from utils.config import Config

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


@dataclass
class Context:
    user_id: str


class AgentState(MessagesState):
    relevance_score: Annotated[Optional[str], "Relevance score of retrieved documents, 'yes' or 'no'"] = None
    rewrite_count: Annotated[int, "Number of times query has been rewritten"] = 0


class ToolConfig:
    def __init__(self, tools):
        self.tools = tools
        self.tool_names = {tool.name for tool in tools}
        self.tool_routing_config = self._build_routing_config(tools)
        logger.info(f"Initialized ToolConfig with tools: {self.tool_names}, routing: {self.tool_routing_config}")

    def _build_routing_config(self, tools):
        routing_config = {}
        for tool in tools:
            tool_name = tool.name.lower()
            if "retrieve" in tool_name:
                routing_config[tool_name] = "grade_documents"
                logger.debug(f"Tool '{tool_name}' routed to 'grade_documents' (retrieval tool)")
            else:
                routing_config[tool_name] = "generate"
                logger.debug(f"Tool '{tool_name}' routed to 'generate' (non-retrieval tool)")
        if not routing_config:
            logger.warning("No tools provided or routing config is empty")
        return routing_config

    def get_tools(self):
        return self.tools

    def get_tool_names(self):
        return self.tool_names

    def get_tool_routing_config(self):
        return self.tool_routing_config


class DocumentRelevanceScore(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")


class ParallelToolNode:
    def __init__(self, tools, max_workers: int = 5):
        from langgraph.prebuilt import ToolNode
        self.tools = tools
        self.max_workers = max_workers
        self.tool_node = ToolNode(tools)

    def _run_single_tool(self, tool_call: dict, tool_map: dict) -> ToolMessage:
        try:
            tool_name = tool_call["name"]
            tool = tool_map.get(tool_name)
            if not tool:
                raise ValueError(f"Tool {tool_name} not found")
            result = tool.invoke(tool_call["args"])
            return ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name=tool_name
            )
        except Exception as e:
            logger.error(f"Error executing tool {tool_call.get('name', 'unknown')}: {e}")
            return ToolMessage(
                content=f"Error: {str(e)}",
                tool_call_id=tool_call["id"],
                name=tool_call.get("name", "unknown")
            )

    def __call__(self, state: dict) -> dict:
        logger.info("ParallelToolNode processing tool calls")
        if isinstance(state, dict) and "messages" in state:
            messages = state["messages"]
        elif hasattr(state, "messages"):
            messages = state.messages
        else:
            logger.warning("No messages found in state")
            return {"messages": []}
        
        if not messages:
            logger.warning("Messages list is empty")
            return {"messages": []}
        
        last_message = messages[-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        if not tool_calls:
            logger.warning("No tool calls found in state")
            return {"messages": []}

        tool_map = {tool.name: tool for tool in self.tools}
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_tool = {
                executor.submit(self._run_single_tool, tool_call, tool_map): tool_call
                for tool_call in tool_calls
            }
            for future in as_completed(future_to_tool):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Tool execution failed: {e}")
                    tool_call = future_to_tool[future]
                    results.append(ToolMessage(
                        content=f"Unexpected error: {str(e)}",
                        tool_call_id=tool_call["id"],
                        name=tool_call.get("name", "unknown")
                    ))

        logger.info(f"Completed {len(results)} tool calls")
        return {"messages": results}


def get_latest_question(state: AgentState) -> Optional[str]:
    try:
        if not state.get("messages") or not isinstance(state["messages"], (list, tuple)) or len(state["messages"]) == 0:
            logger.warning("No valid messages found in state for getting latest question")
            return None

        for message in reversed(state["messages"]):
            if message.__class__.__name__ == "HumanMessage" and hasattr(message, "content"):
                return message.content

        logger.info("No HumanMessage found in state")
        return None

    except Exception as e:
        logger.error(f"Error getting latest question: {e}")
        return None


def filter_messages(messages: list) -> list:
    filtered = [msg for msg in messages if msg.__class__.__name__ in ['AIMessage', 'HumanMessage']]
    return filtered[-5:] if len(filtered) > 5 else filtered


def store_memory(question: BaseMessage, user_id: str, store: BaseStore) -> str:
    namespace = ("memories", user_id)
    try:
        memories = store.search(namespace, query=str(question.content))
        user_info = "\n".join([d.value["data"] for d in memories])

        if "记住" in question.content.lower():
            memory = escape(question.content)
            store.put(namespace, str(uuid.uuid4()), {"data": memory})
            logger.info(f"Stored memory: {memory}")

        return user_info
    except Exception as e:
        logger.error(f"Error in store_memory: {e}")
        return ""


def create_chain(llm_chat, template_file: str, structured_output=None):
    if not hasattr(create_chain, "prompt_cache"):
        create_chain.prompt_cache = {}
        create_chain.lock = threading.Lock()

    try:
        if template_file in create_chain.prompt_cache:
            prompt_template = create_chain.prompt_cache[template_file]
            logger.info(f"Using cached prompt template for {template_file}")
        else:
            with create_chain.lock:
                if template_file not in create_chain.prompt_cache:
                    logger.info(f"Loading and caching prompt template from {template_file}")
                    create_chain.prompt_cache[template_file] = PromptTemplate.from_file(template_file, encoding="utf-8")
                prompt_template = create_chain.prompt_cache[template_file]

        prompt = ChatPromptTemplate.from_messages([("human", prompt_template.template)])
        return prompt | (llm_chat.with_structured_output(structured_output) if structured_output else llm_chat)
    except FileNotFoundError:
        logger.error(f"Template file {template_file} not found")
        raise


def agent_v1(state: AgentState, runtime: Runtime[Context], llm_chat, tool_config: ToolConfig) -> dict:
    logger.info("Agent v1 processing user query")
    user_id = runtime.context.user_id if runtime.context and hasattr(runtime.context, 'user_id') else "unknown"
    namespace = ("memories", user_id)
    try:
        question = state["messages"][-1]
        logger.info(f"agent v1 question:{question}")

        user_info = store_memory(question, user_id, runtime.store)
        messages = filter_messages(state["messages"])

        llm_chat_with_tool = llm_chat.bind_tools(tool_config.get_tools())

        agent_chain = create_chain(llm_chat_with_tool, Config.PROMPT_TEMPLATE_TXT_AGENT)
        response = agent_chain.invoke({"question": question, "messages": messages, "userInfo": user_info})
        
        logger.info(f"Agent v1 response type: {type(response)}")
        
        if hasattr(response, 'content_blocks'):
            logger.info(f"Using content_blocks from LangChain v1")
            for block in response.content_blocks:
                if block.get("type") == "text":
                    logger.info(f"Text block: {block.get('text', '')[:100]}")
                elif block.get("type") == "tool_call":
                    logger.info(f"Tool call: {block.get('name', '')}")
        
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error in agent v1 processing: {e}")
        return {"messages": [{"role": "system", "content": "处理请求时出错"}]}


def grade_documents(state: AgentState, llm_chat) -> dict:
    logger.info("Grading documents for relevance")
    if not state.get("messages"):
        logger.error("Messages state is empty")
        return {
            "messages": [{"role": "system", "content": "状态为空，无法评分"}],
            "relevance_score": None
        }

    try:
        question = get_latest_question(state)
        context = state["messages"][-1].content

        grade_chain = create_chain(llm_chat, Config.PROMPT_TEMPLATE_TXT_GRADE, DocumentRelevanceScore)
        scored_result = grade_chain.invoke({"question": question, "context": context})
        score = scored_result.binary_score
        logger.info(f"Document relevance score: {score}")

        return {
            "messages": state["messages"],
            "relevance_score": score
        }
    except (IndexError, KeyError) as e:
        logger.error(f"Message access error: {e}")
        return {
            "messages": [{"role": "system", "content": "无法评分文档"}],
            "relevance_score": None
        }
    except Exception as e:
        logger.error(f"Unexpected error in grading: {e}")
        return {
            "messages": [{"role": "system", "content": "评分过程中出错"}],
            "relevance_score": None
        }


def rewrite(state: AgentState, llm_chat) -> dict:
    logger.info("Rewriting query")
    try:
        question = get_latest_question(state)
        rewrite_chain = create_chain(llm_chat, Config.PROMPT_TEMPLATE_TXT_REWRITE)
        response = rewrite_chain.invoke({"question": question})
        rewrite_count = state.get("rewrite_count", 0) + 1
        logger.info(f"Rewrite count: {rewrite_count}")
        return {"messages": [response], "rewrite_count": rewrite_count}
    except (IndexError, KeyError) as e:
        logger.error(f"Message access error in rewrite: {e}")
        return {"messages": [{"role": "system", "content": "无法重写查询"}]}


def generate(state: AgentState, llm_chat) -> dict:
    logger.info("Generating final response")
    try:
        question = get_latest_question(state)
        context = state["messages"][-1].content
        generate_chain = create_chain(llm_chat, Config.PROMPT_TEMPLATE_TXT_GENERATE)
        response = generate_chain.invoke({"context": context, "question": question})
        return {"messages": [response]}
    except (IndexError, KeyError) as e:
        logger.error(f"Message access error in generate: {e}")
        return {"messages": [{"role": "system", "content": "无法生成回复"}]}


def route_after_tools(state: AgentState, tool_config: ToolConfig) -> Literal["generate", "grade_documents"]:
    if not state.get("messages") or not isinstance(state["messages"], list):
        logger.error("Messages state is empty or invalid, defaulting to generate")
        return "generate"

    try:
        last_message = state["messages"][-1]

        if not hasattr(last_message, "name") or last_message.name is None:
            logger.info("Last message has no name attribute, routing to generate")
            return "generate"

        tool_name = last_message.name
        if tool_name not in tool_config.get_tool_names():
            logger.info(f"Unknown tool {tool_name}, routing to generate")
            return "generate"

        target = tool_config.get_tool_routing_config().get(tool_name, "generate")
        logger.info(f"Tool {tool_name} routed to {target} based on config")
        return target

    except IndexError:
        logger.error("No messages available in state, defaulting to generate")
        return "generate"
    except AttributeError:
        logger.error("Invalid message object, defaulting to generate")
        return "generate"
    except Exception as e:
        logger.error(f"Unexpected error in route_after_tools: {e}, defaulting to generate")
        return "generate"


def route_after_grade(state: AgentState) -> Literal["generate", "rewrite"]:
    if not isinstance(state, dict):
        logger.error("State is not a valid dictionary, defaulting to rewrite")
        return "rewrite"

    if "messages" not in state or not isinstance(state["messages"], (list, tuple)):
        logger.error("State missing valid messages field, defaulting to rewrite")
        return "rewrite"

    if not state["messages"]:
        logger.warning("Messages list is empty, defaulting to rewrite")
        return "rewrite"

    relevance_score = state.get("relevance_score")
    rewrite_count = state.get("rewrite_count", 0)
    logger.info(f"Routing based on relevance_score: {relevance_score}, rewrite_count: {rewrite_count}")

    if rewrite_count >= 3:
        logger.info("Max rewrite limit reached, proceeding to generate")
        return "generate"

    try:
        if not isinstance(relevance_score, str):
            logger.warning(f"Invalid relevance_score type: {type(relevance_score)}, defaulting to rewrite")
            return "rewrite"

        if relevance_score.lower() == "yes":
            logger.info("Documents are relevant, proceeding to generate")
            return "generate"

        logger.info("Documents are not relevant or scoring failed, proceeding to rewrite")
        return "rewrite"

    except AttributeError:
        logger.error("relevance_score is not a string or is None, defaulting to rewrite")
        return "rewrite"
    except Exception as e:
        logger.error(f"Unexpected error in route_after_grade: {e}, defaulting to rewrite")
        return "rewrite"


def save_graph_visualization(graph: StateGraph, filename: str = "graph_v1.png") -> None:
    try:
        with open(filename, "wb") as f:
            f.write(graph.get_graph().draw_mermaid_png())
        logger.info(f"Graph v1 visualization saved as {filename}")
    except IOError as e:
        logger.warning(f"Failed to save graph v1 visualization: {e}")


def create_graph_v1(llm_chat, llm_embedding, tool_config: ToolConfig, use_middleware: bool = True) -> StateGraph:
    DB_URI = Config.DB_URI
    
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from langgraph.store.postgres import PostgresStore
        from langgraph.prebuilt import tools_condition
        
        import psycopg2
        conn_psycopg2 = psycopg2.connect(DB_URI)
        print("✓ 数据库连接成功!")
        conn_psycopg2.close()
        
        from psycopg_pool import ConnectionPool
        
        pool = ConnectionPool(
            DB_URI,
            min_size=1,
            max_size=10
        )
        
        checkpointer = PostgresSaver(pool)
        store = PostgresStore(pool, index={"dims": 1536, "embed": llm_embedding})
        
        checkpointer.setup()
        store.setup()
        logger.info("PostgresSaver and PostgresStore initialized successfully for v1")

        workflow = StateGraph(AgentState, context_schema=Context)
        
        if use_middleware:
            logger.info("Using LangChain v1 middleware-enhanced agent")
            from langchain.agents.middleware import PIIMiddleware, SummarizationMiddleware
            
            piim = PIIMiddleware(patterns=["email", "phone", "ssn"])
            sm = SummarizationMiddleware(model=llm_chat, max_tokens_before_summary=500)
            
            workflow.add_node("agent", lambda state, runtime: agent_v1(state, runtime, llm_chat=llm_chat, tool_config=tool_config))
        else:
            logger.info("Using standard agent (v1 compatible)")
            workflow.add_node("agent", lambda state, runtime: agent_v1(state, runtime, llm_chat=llm_chat, tool_config=tool_config))
        
        workflow.add_node("call_tools", ParallelToolNode(tool_config.get_tools(), max_workers=5))
        workflow.add_node("rewrite", lambda state: rewrite(state, llm_chat=llm_chat))
        workflow.add_node("generate", lambda state: generate(state, llm_chat=llm_chat))
        workflow.add_node("grade_documents", lambda state: grade_documents(state, llm_chat=llm_chat))

        workflow.add_edge(START, end_key="agent")
        workflow.add_conditional_edges(source="agent", path=tools_condition, path_map={"tools": "call_tools", END: END})
        workflow.add_conditional_edges(source="call_tools", path=lambda state: route_after_tools(state, tool_config), path_map={"generate": "generate", "grade_documents": "grade_documents"})
        workflow.add_conditional_edges(source="grade_documents", path=route_after_grade, path_map={"generate": "generate", "rewrite": "rewrite"})
        workflow.add_edge(start_key="generate", end_key=END)
        workflow.add_edge(start_key="rewrite", end_key="agent")

        return workflow.compile(checkpointer=checkpointer, store=store)
    except Exception as e:
        logger.error(f"Failed to setup PostgresSaver or PostgresStore: {e}")
        logger.warning("Falling back to in-memory storage due to PostgreSQL compatibility issues")
        
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.store.memory import InMemoryStore
        from langgraph.prebuilt import tools_condition
        
        checkpointer = MemorySaver()
        store = InMemoryStore()
        
        workflow = StateGraph(AgentState, context_schema=Context)
        
        if use_middleware:
            logger.info("Using LangChain v1 middleware-enhanced agent (in-memory)")
            from langchain.agents.middleware import PIIMiddleware, SummarizationMiddleware
            
            piim = PIIMiddleware(patterns=["email", "phone", "ssn"])
            sm = SummarizationMiddleware(model=llm_chat, max_tokens_before_summary=500)
            
            workflow.add_node("agent", lambda state, runtime: agent_v1(state, runtime, llm_chat=llm_chat, tool_config=tool_config))
        else:
            logger.info("Using standard agent (v1 compatible, in-memory)")
            workflow.add_node("agent", lambda state, runtime: agent_v1(state, runtime, llm_chat=llm_chat, tool_config=tool_config))
        
        workflow.add_node("call_tools", ParallelToolNode(tool_config.get_tools(), max_workers=5))
        workflow.add_node("rewrite", lambda state: rewrite(state, llm_chat=llm_chat))
        workflow.add_node("generate", lambda state: generate(state, llm_chat=llm_chat))
        workflow.add_node("grade_documents", lambda state: grade_documents(state, llm_chat=llm_chat))

        workflow.add_edge(START, end_key="agent")
        workflow.add_conditional_edges(source="agent", path=tools_condition, path_map={"tools": "call_tools", END: END})
        workflow.add_conditional_edges(source="call_tools", path=lambda state: route_after_tools(state, tool_config), path_map={"generate": "generate", "grade_documents": "grade_documents"})
        workflow.add_conditional_edges(source="grade_documents", path=route_after_grade, path_map={"generate": "generate", "rewrite": "rewrite"})
        workflow.add_edge(start_key="generate", end_key=END)
        workflow.add_edge(start_key="rewrite", end_key="agent")

        return workflow.compile(checkpointer=checkpointer, store=store)


def graph_response_v1(graph: StateGraph, user_input: str, config: dict, tool_config: ToolConfig, context: Context) -> None:
    try:
        events = graph.stream(
            {"messages": [{"role": "user", "content": user_input}], "rewrite_count": 0},
            config,
            context=context
        )
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
                        print(f"Tool Output [{tool_name}]: {content}")
                    else:
                        print(f"Assistant: {content}")
                        
                        if hasattr(last_message, 'content_blocks'):
                            print(f"[v1] Content blocks available: {len(last_message.content_blocks)} blocks")
                else:
                    logger.info("Message has no content, skipping")
                    print("Assistant: 未获取到相关回复")
    except ValueError as ve:
        logger.error(f"Value error in response processing: {ve}")
        print("Assistant: 处理响应时发生值错误")
    except Exception as e:
        logger.error(f"Error processing response: {e}")
        print("Assistant: 处理响应时发生未知错误")


def main():
    try:
        llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)
        tools = get_tools(llm_embedding)
        tool_config = ToolConfig(tools)

        graph = create_graph_v1(llm_chat, llm_embedding, tool_config, use_middleware=True)

        print("聊天机器人 v1 准备就绪！输入 'quit'、'exit' 或 'q' 结束对话。")
        print("使用 LangChain v1 特性: content_blocks, middleware 支持")
        config = {"configurable": {"thread_id": "1"}}
        context = Context(user_id="1")

        while True:
            user_input = input("User: ").strip()
            if user_input.lower() in {"quit", "exit", "q"}:
                print("拜拜!")
                break
            if not user_input:
                print("请输入聊天内容！")
                continue
            graph_response_v1(graph, user_input, config, tool_config, context)

    except RuntimeError as e:
        logger.error(f"Initialization error: {e}")
        print(f"错误: 初始化失败 - {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n被用户打断。再见！")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"错误: 发生未知错误 - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
