# ragAgent.py
# LangChain v1 / LangGraph v1 迁移版本
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
from langchain_core.messages import BaseMessage, AIMessage
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field
from utils.llms import get_llm
from utils.tools_config import get_tools
from utils.config import Config

os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

# LangChain v1 变更说明：
# - langgraph.runtime.Runtime 在 LangGraph v1 中可能已更名或调整
# - 尝试从新位置导入，如失败则回退到兼容方式
try:
    from langgraph.runtime import Runtime
except ImportError:
    # LangGraph v1 兼容：如果 Runtime 不在 langgraph.runtime 中，
    # 则使用 langgraph.types 中的 RunnableConfig 模式替代
    Runtime = None
    logging.getLogger(__name__).warning(
        "langgraph.runtime.Runtime not found, using fallback config pattern"
    )

# # 设置日志基本配置，级别为DEBUG或INFO
logger = logging.getLogger(__name__)
# 设置日志器级别为DEBUG
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
logger.handlers = []  # 清空默认处理器
# 使用ConcurrentRotatingFileHandler
handler = ConcurrentRotatingFileHandler(
    # 日志文件
    Config.LOG_FILE,
    # 日志文件最大允许大小为5MB，达到上限后触发轮转
    maxBytes = Config.MAX_BYTES,
    # 在轮转时，最多保留3个历史日志文件
    backupCount = Config.BACKUP_COUNT
)
# 设置处理器级别为DEBUG
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


# 定义工具配置类，用于存储和管理工具列表、名称集合和路由配置
class ToolConfig:
    def __init__(self, tools):
        # 将传入的工具列表存储到实例变量 self.tools 中
        self.tools = tools
        # 创建一个集合，包含所有工具的名称，使用集合推导式从 tools 中提取 name 属性
        self.tool_names = {tool.name for tool in tools}
        # 调用内部方法 _build_routing_config，动态生成工具路由配置并存储到 self.tool_routing_config
        self.tool_routing_config = self._build_routing_config(tools)
        # 记录日志，输出初始化完成的工具名称集合和路由配置，便于调试和验证
        logger.info(f"Initialized ToolConfig with tools: {self.tool_names}, routing: {self.tool_routing_config}")

    # 它根据工具名称是否包含 "retrieve" 来判断是否为检索类工具，根据是否为检索类工具，将其路由到 "grade_documents" 或 "generate"
    def _build_routing_config(self, tools):
        # 创建一个空字典，用于存储工具名称到目标节点的映射
        routing_config = {}
        # 遍历传入的工具列表，逐个处理每个工具
        for tool in tools:
            # 将工具名称转换为小写，确保匹配时忽略大小写
            tool_name = tool.name.lower()
            # 检查工具名称中是否包含 "retrieve"，用于判断是否为检索类工具
            if "retrieve" in tool_name:
                # 如果是检索类工具，将其路由目标设置为 "grade_documents"（需要评分）
                routing_config[tool_name] = "grade_documents"
                # 记录调试日志，说明该工具被路由到 "grade_documents"，并标注为检索工具
                logger.debug(f"Tool '{tool_name}' routed to 'grade_documents' (retrieval tool)")
            # 如果工具名称不包含 "retrieve"
            else:
                # 将其路由目标设置为 "generate"（直接生成结果）
                routing_config[tool_name] = "generate"
                # 记录调试日志，说明该工具被路由到 "generate"，并标注为非检索工具
                logger.debug(f"Tool '{tool_name}' routed to 'generate' (non-retrieval tool)")
        # 检查路由配置字典是否为空（即没有工具被处理）
        if not routing_config:
            # 如果为空，记录警告日志，提示工具列表可能为空或未正确处理
            logger.warning("No tools provided or routing config is empty")
        # 返回生成的路由配置字典
        return routing_config

    # 获取工具列表的方法，返回存储在实例中的 tools
    def get_tools(self):
        # 直接返回 self.tools，提供外部访问工具列表的接口
        return self.tools

    # 获取工具名称集合的方法，返回存储在实例中的 tool_names
    def get_tool_names(self):
        # 直接返回 self.tool_names，提供外部访问工具名称集合的接口
        return self.tool_names

    # 获取工具路由配置的方法，返回动态生成的路由配置
    def get_tool_routing_config(self):
        # 直接返回 self.tool_routing_config，提供外部访问路由配置的接口
        return self.tool_routing_config

# 定义文档相关性评分模型，用于存储文档的二进制评分（'yes' 或 'no'）
class DocumentRelevanceScore(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

# 它接收工具列表和最大线程数作为参数，初始化一个工具节点对象。
# 当调用并行工具节点时，它会从状态中提取消息列表，获取最后一个消息的工具调用列表，
# 并行执行每个工具调用，将结果返回为工具节点，将所有工具调用的结果合并为一个列表，作为图的输出。
class ParallelToolNode:
    def __init__(self, tools, max_workers: int = 5):
        from langgraph.prebuilt import ToolNode
        self.tools = tools
        self.max_workers = max_workers
        self.tool_node = ToolNode(tools)

    def _run_single_tool(self, tool_call: dict, tool_map: dict) -> ToolMessage:
        """执行单个工具调用"""
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
        """并行执行所有工具调用"""
        logger.info("ParallelToolNode processing tool calls")
        # 尝试不同的方式获取消息
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
        
        # 使用线程池并行执行工具调用
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

# 定义获取最新用户问题的函数，用于从状态中提取用户输入的最新问题。
def get_latest_question(state: AgentState) -> Optional[str]:
    """从状态中安全地获取最新用户问题。

    Args:
        state: 当前对话状态，包含消息历史。

    Returns:
        Optional[str]: 最新问题的内容，如果无法获取则返回 None。
    """
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

# 定义消息过滤函数，用于从消息列表中提取最后5条消息。
# LangChain v1 变更说明：AIMessage 现在是 ChatOpenAI invoke() 的确切返回类型
def filter_messages(messages: list) -> list:
    """过滤消息列表，仅保留 AIMessage 和 HumanMessage 类型消息"""
    filtered = [msg for msg in messages if msg.__class__.__name__ in ['AIMessage', 'HumanMessage']]
    return filtered[-5:] if len(filtered) > 5 else filtered


def store_memory(question: BaseMessage, user_id: str, store: BaseStore) -> str:
    """存储用户输入中的记忆信息。

    Args:
        question: 用户输入的消息。
        user_id: 用户ID。
        store: 数据存储实例。

    Returns:
        str: 用户相关的记忆信息字符串。
    """
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

# 该函数用于创建一个语言模型处理链，包括加载模板、绑定模型和结构化输出。
# 它使用缓存机制避免重复读取模板文件，提高效率。
# 它还支持在创建时指定可选的结构化输出模型，用于将模型输出转换为指定的结构化格式。
# 最后，它返回一个可运行的处理链对象，用于后续的对话处理。
# 该函数的参数包括语言模型实例、提示模板文件路径和可选的结构化输出模型。
def create_chain(llm_chat, template_file: str, structured_output=None):
    """创建 LLM 处理链，加载提示模板并绑定模型，使用缓存避免重复读取文件。

    LangChain v1 变更说明：
    - PromptTemplate, ChatPromptTemplate 仍在 langchain_core.prompts 中，无需修改
    - with_structured_output() 方法保持不变
    - LCEL（LangChain Expression Language）管道操作符 | 保持不变

    Args:
        llm_chat: 语言模型实例。
        template_file: 提示模板文件路径。
        structured_output: 可选的结构化输出模型。

    Returns:
        Runnable: 配置好的处理链。

    Raises:
        FileNotFoundError: 如果模板文件不存在。
    """
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

# 定义代理函数，用于处理用户查询并调用工具或生成响应。
# LangChain v1 变更说明：
# - Runtime 参数的使用方式可能需要根据 LangGraph v1 的具体实现调整
# - 使用 config 参数替代 runtime 参数进行上下文传递（v1 推荐方式）
def agent(state: AgentState, config: dict, llm_chat, tool_config: ToolConfig, store=None) -> dict:
    """代理函数，根据用户问题决定是否调用工具或结束。

    Args:
        state: 当前对话状态。
        config: 运行时配置字典，包含 configurable 信息。
        llm_chat: Chat模型。
        tool_config: 工具配置参数。
        store: 数据存储实例（可选）。

    Returns:
        dict: 更新后的对话状态。
    """
    logger.info("Agent processing user query")
    
    # LangChain v1 变更说明：
    # 从 config["configurable"] 中获取 user_id，替代原 runtime.context.user_id
    # 同时保持对旧版 runtime 模式的兼容
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    user_id = configurable.get("user_id", "unknown")
    
    try:
        question = state["messages"][-1]
        logger.info(f"agent question:{question}")

        # 如果 store 可用，存储记忆信息
        user_info = ""
        if store:
            user_info = store_memory(question, user_id, store)
        
        messages = filter_messages(state["messages"])

        llm_chat_with_tool = llm_chat.bind_tools(tool_config.get_tools())

        agent_chain = create_chain(llm_chat_with_tool, Config.PROMPT_TEMPLATE_TXT_AGENT)
        response = agent_chain.invoke({"question": question, "messages": messages, "userInfo": user_info})
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error in agent processing: {e}")
        return {"messages": [{"role": "system", "content": "处理请求时出错"}]}

# 定义文档评分函数，用于评估检索到的文档内容与问题的相关性。
def grade_documents(state: AgentState, llm_chat) -> dict:
    """评估检索到的文档内容与问题的相关性，并将评分结果存储在状态中。

    Args:
        state: 当前对话状态，包含消息历史。

    Returns:
        dict: 更新后的状态，包含评分结果。
    """
    logger.info("Grading documents for relevance")
    # 检查状态是否包含消息历史
    if not state.get("messages"):
        logger.error("Messages state is empty")
        return {
            "messages": [{"role": "system", "content": "状态为空，无法评分"}],
            "relevance_score": None
        }

    try:
        question = get_latest_question(state)
        # 从状态中提取检索到的文档内容
        context = state["messages"][-1].content

        # 检查检索到的文档内容是否为空
        # 如果为空，自动评分为 'no'
        if not context or str(context).strip() == "":
            logger.warning("Retrieved context is empty, auto-grading as 'no'")
            return {
                "messages": state["messages"],
                "relevance_score": "no"
            }        

        if hasattr(llm_chat, "model_copy"):
            # 兼容 Pydantic v2 (LangChain 较新版本)
            grader_llm = llm_chat.model_copy(update={"temperature": 0.0})
        elif hasattr(llm_chat, "copy"):
            # 兼容 Pydantic v1 (LangChain 较旧版本)
            grader_llm = llm_chat.copy(update={"temperature": 0.0})
        else:
            # 最后的退路
            grader_llm = llm_chat.bind(temperature=0.0)
            
        logger.debug(f"Grader LLM temperature set to: {getattr(grader_llm, 'temperature', 'unknown')}")
        
        grade_chain = create_chain(grader_llm, Config.PROMPT_TEMPLATE_TXT_GRADE, DocumentRelevanceScore)
        scored_result = grade_chain.invoke({"question": question, "context": context})
        score = str(scored_result.binary_score).strip().lower()
        
        # 二次校验：确保输出仅为 yes 或 no
        if score not in ["yes", "no"]:
            logger.warning(f"Unexpected score value: {score}, defaulting to 'no'")
            score = "no"

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

# 定义重写函数，用于改进用户查询。
def rewrite(state: AgentState, llm_chat) -> dict:
    """重写用户查询以改进问题。

    Args:
        state: 当前对话状态。

    Returns:
        dict: 更新后的消息状态。
    """
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

# 定义生成函数，用于基于工具返回的内容生成最终回复。
def generate(state: AgentState, llm_chat) -> dict:
    """基于工具返回的内容生成最终回复。

    Args:
        state: 当前对话状态。

    Returns:
        dict: 更新后的消息状态。
    """
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

# 定义路由函数，用于根据工具调用的结果动态决定下一步路由。
# 它会根据状态中的消息历史和工具调用结果，动态路由到生成节点或文档评分节点。
def route_after_tools(state: AgentState, tool_config: ToolConfig) -> Literal["generate", "grade_documents"]:
    """根据工具调用的结果动态决定下一步路由，使用配置字典支持多工具并包含容错处理。

    Args:
        state: 当前对话状态，包含消息历史和可能的工具调用结果。
        tool_config: 工具配置参数。

    Returns:
        Literal["generate", "grade_documents"]: 下一步的目标节点。
    """
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

# 定义路由函数，用于根据文档评分结果动态决定下一步路由。
# 它会根据状态中的消息历史和评分结果，动态路由到生成节点或重写节点。
def route_after_grade(state: AgentState) -> Literal["generate", "rewrite"]:
    """根据状态中的评分结果决定下一步路由，包含增强的状态校验和容错处理。

    Args:
        state: 当前对话状态，预期包含 messages 和 relevance_score 字段。

    Returns:
        Literal["generate", "rewrite"]: 下一步的目标节点。
    """
    if not isinstance(state, dict):
        logger.error("State is not a valid dictionary, defaulting to rewrite")
        return "rewrite"

    # 检查状态是否包含 messages 字段，若缺失则记录错误并默认路由到 rewrite
    if "messages" not in state or not isinstance(state["messages"], (list, tuple)):
        logger.error("State missing valid messages field, defaulting to rewrite")
        return "rewrite"

    # 检查 messages 是否为空，若为空则记录警告并默认路由到 rewrite
    if not state["messages"]:
        logger.warning("Messages list is empty, defaulting to rewrite")
        return "rewrite"

    # 获取状态中的 relevance_score，若不存在则返回 None
    relevance_score = state.get("relevance_score")
    # 获取状态中的 rewrite_count
    rewrite_count = state.get("rewrite_count", 0)
    logger.info(f"Routing based on relevance_score: {relevance_score}, rewrite_count: {rewrite_count}")

    # 如果重写次数超过 3 次，强制路由到 generate
    if rewrite_count >= 3:
        logger.info("Max rewrite limit reached, proceeding to generate")
        return "generate"

    try:
        # 检查 relevance_score 是否为有效字符串，若不是则视为无效评分
        if not isinstance(relevance_score, str):
            logger.warning(f"Invalid relevance_score type: {type(relevance_score)}, defaulting to rewrite")
            return "rewrite"

        # 如果评分结果为 "yes"，表示文档相关，路由到 generate 节点
        if relevance_score.lower() == "yes":
            logger.info("Documents are relevant, proceeding to generate")
            return "generate"

        # 如果评分结果为 "no" 或其他值（包括空字符串），路由到 rewrite 节点
        logger.info("Documents are not relevant or scoring failed, proceeding to rewrite")
        return "rewrite"

    except AttributeError:
        # 捕获 relevance_score 不支持 lower() 方法的异常（例如 None），默认路由到 rewrite
        logger.error("relevance_score is not a string or is None, defaulting to rewrite")
        return "rewrite"
    except Exception as e:
        # 捕获其他未预期的异常，记录详细错误信息并默认路由到 rewrite
        logger.error(f"Unexpected error in route_after_grade: {e}, defaulting to rewrite")
        return "rewrite"


# 保存状态图的可视化表示
def save_graph_visualization(graph: StateGraph, filename: str = "graph.png") -> None:
    """保存状态图的可视化表示。

    Args:
        graph: 状态图实例。
        filename: 保存文件路径。
    """
    # 尝试执行以下代码块
    try:
        # 以二进制写模式打开文件
        with open(filename, "wb") as f:
            # 将状态图转换为Mermaid格式的PNG并写入文件
            f.write(graph.get_graph().draw_mermaid_png())
        # 记录保存成功的日志
        logger.info(f"Graph visualization saved as {filename}")
    # 捕获IO错误
    except IOError as e:
        # 记录警告日志
        logger.warning(f"Failed to save graph visualization: {e}")


def create_graph(llm_chat, llm_embedding, tool_config: ToolConfig) -> StateGraph:
    """创建并配置状态图。
    
    LangChain v1 / LangGraph v1 迁移说明：
    - StateGraph 核心 API 保持不变（add_node, add_edge, add_conditional_edges, compile）
    - tools_condition 从 langgraph.prebuilt 导入（LangGraph v1 保持向后兼容）
    - PostgresSaver / PostgresStore 接口不变
    - agent 节点签名从 (state, runtime) 调整为 (state, config)，通过 config 传递上下文
    - 向量数据库已在 tools_config.py 中从 ChromaDB 迁移至 Qdrant

    Args:
        llm_chat: Chat模型。
        llm_embedding: Embedding模型。
        tool_config: 工具配置参数。

    Returns:
        StateGraph: 编译后的状态图。

    Raises:
        Exception: 如果初始化失败。
    """
    DB_URI = Config.DB_URI
    
    try:
        # 导入 PostgresSaver 和 PostgresStore
        from langgraph.checkpoint.postgres import PostgresSaver
        from langgraph.store.postgres import PostgresStore
        # LangChain v1 变更说明：
        # langgraph.prebuilt 在 v1 中已弃用，但保持向后兼容
        # tools_condition 仍可从 langgraph.prebuilt 导入
        from langgraph.prebuilt import tools_condition
        
        # 测试数据库连接（使用 psycopg2）
        import psycopg2
        conn_psycopg2 = psycopg2.connect(DB_URI)
        print("✓ 数据库连接成功!")
        conn_psycopg2.close()
        
        # 使用 psycopg-pool 创建连接池
        from psycopg_pool import ConnectionPool
        
        # 创建连接池
        pool = ConnectionPool(
            DB_URI,
            min_size=1,
            max_size=10
        )
        
        # 创建 PostgresSaver 和 PostgresStore
        checkpointer = PostgresSaver(pool)
        store = PostgresStore(pool, index={"dims": 1536, "embed": llm_embedding})
        
        # 调用 setup 方法
        checkpointer.setup()
        store.setup()
        logger.info("PostgresSaver and PostgresStore initialized successfully")

        # LangChain v1 变更说明：
        # StateGraph 构造保持不变，context_schema 用于定义静态上下文
        workflow = StateGraph(AgentState, context_schema=Context)
        
        # LangChain v1 变更说明：
        # agent 节点签名从 lambda state, runtime: agent(state, runtime, ...) 
        # 调整为 lambda state, config: agent(state, config, ...)
        # 通过 config["configurable"] 传递 user_id 等上下文信息
        workflow.add_node("agent", lambda state, config: agent(state, config, llm_chat=llm_chat, tool_config=tool_config, store=store))
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
        
        # 降级到使用内存存储
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.store.memory import InMemoryStore
        from langgraph.prebuilt import tools_condition
        
        checkpointer = MemorySaver()
        store = InMemoryStore()
        
        workflow = StateGraph(AgentState, context_schema=Context)
        
        # 内存模式下同样使用 config 传递上下文
        workflow.add_node("agent", lambda state, config: agent(state, config, llm_chat=llm_chat, tool_config=tool_config, store=store))
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


def graph_response(graph: StateGraph, user_input: str, config: dict, tool_config: ToolConfig, context: Context) -> None:
    """处理用户输入并输出响应，区分工具输出和大模型输出，支持多工具。

    LangChain v1 变更说明：
    - graph.stream() 的 context 参数在 v1 中仍受支持
    - 旧版 config["configurable"] 模式仍可工作，同时支持新的 context 参数

    Args:
        graph: 状态图实例。
        user_input: 用户输入。
        config: 运行时配置。
        tool_config: 工具配置参数。
        context: 运行时上下文。
    """
    try:
        events = graph.stream(
            {"messages": [{"role": "user", "content": user_input}], "rewrite_count": 0},
            config,
            context=context
        )
        for event in events:
            for value in event.values():
                # 检查是否包含 messages 字段
                if "messages" not in value or not isinstance(value["messages"], list):
                    logger.warning("No valid messages in response")
                    continue

                last_message = value["messages"][-1]
                
                # 检查是否包含 tool_calls 字段
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
    """主函数，初始化并运行聊天机器人。"""
    try:
        llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)
        tools = get_tools(llm_embedding)
        tool_config = ToolConfig(tools)

        graph = create_graph(llm_chat, llm_embedding, tool_config)

        print("聊天机器人准备就绪！输入 'quit'、'exit' 或 'q' 结束对话。")
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
            graph_response(graph, user_input, config, tool_config, context)

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