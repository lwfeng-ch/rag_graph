# utils/middleware.py
# Middleware 层：所有 Middleware 实例无可变状态（只存配置），
# 运行时数据全部读写 AgentState，确保多用户/多线程安全。

import logging
import time
import re
from typing import Any, Optional, Callable, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =====================================================
# 基础 Middleware 抽象类
# =====================================================
class BaseMiddleware:
    """基础 Middleware 类。
    
    设计原则：
    - 实例上只存不可变配置（如 max_calls、mode）
    - 所有运行时可变状态通过 state dict 传入传出
    - 每个 hook 返回 (state_updates: dict, should_stop: bool)
    
    子类可重写以下 hook：
    - before_model(state, node_name) → (updates, stop)
    - after_model(state, response, node_name, elapsed) → updates
    - before_tool(state, tool_call) → (updates, stop)
    - after_tool(state, tool_result, elapsed) → updates
    """

    # 声明此 Middleware 适用的节点类型，子类可覆盖
    # "model" 类节点：agent, grade_documents, rewrite, generate
    # "tool" 类节点：call_tools
    applicable_node_types: set = {"model", "tool"}

    def before_model(self, state: dict, node_name: str) -> Tuple[dict, bool]:
        """模型调用前 hook。
        
        Args:
            state: 当前 AgentState（只读，不要直接修改）
            node_name: 当前节点名称
            
        Returns:
            (state_updates, should_stop): 需要更新的状态字段 和 是否强制终止
        """
        return {}, False

    def after_model(self, state: dict, response: Any, node_name: str, elapsed: float) -> dict:
        """模型调用后 hook。
        
        Args:
            state: 当前 AgentState
            response: 模型返回的响应
            node_name: 当前节点名称
            elapsed: 本次调用耗时（秒）
            
        Returns:
            state_updates: 需要更新的状态字段
        """
        return {}

    def before_tool(self, state: dict, tool_call: dict) -> Tuple[dict, bool]:
        """工具调用前 hook。"""
        return {}, False

    def after_tool(self, state: dict, tool_result: Any, tool_name: str, elapsed: float) -> dict:
        """工具调用后 hook。"""
        return {}


# =====================================================
# 1. 日志与性能追踪 Middleware
# =====================================================
class LoggingMiddleware(BaseMiddleware):
    """日志追踪 Middleware。
    
    无实例状态：所有计数和耗时都读写 AgentState 的 mw_ 字段。
    多用户并发下每个请求有独立的 state，互不干扰。
    """
    applicable_node_types = {"model", "tool"}

    def before_model(self, state: dict, node_name: str) -> Tuple[dict, bool]:
        """记录模型调用开始，从 state 读取当前计数"""
        current_count = state.get("mw_model_call_count", 0)
        logger.info(f"[Logging] [{node_name}] 模型调用开始 (本次请求第 {current_count + 1} 次)")
        return {}, False

    def after_model(self, state: dict, response: Any, node_name: str, elapsed: float) -> dict:
        """记录模型调用结束和耗时"""
        total_time = state.get("mw_model_total_time", 0.0) + elapsed
        # 更新节点耗时记录
        timings = state.get("mw_node_timings") or {}
        if node_name not in timings:
            timings[node_name] = {"count": 0, "total_time": 0.0}
        timings[node_name]["count"] += 1
        timings[node_name]["total_time"] += elapsed

        logger.info(
            f"[Logging] [{node_name}] 模型调用完成 "
            f"耗时: {elapsed:.3f}s, 累计: {total_time:.3f}s"
        )
        return {"mw_model_total_time": total_time, "mw_node_timings": timings}

    def before_tool(self, state: dict, tool_call: dict) -> Tuple[dict, bool]:
        """记录工具调用开始"""
        tool_name = tool_call.get("name", "unknown")
        logger.info(f"[Logging] [call_tools] 工具调用开始: {tool_name}")
        return {}, False

    def after_tool(self, state: dict, tool_result: Any, tool_name: str, elapsed: float) -> dict:
        """记录工具调用耗时"""
        total_time = state.get("mw_tool_total_time", 0.0) + elapsed
        logger.info(f"[Logging] [call_tools] 工具 {tool_name} 完成, 耗时: {elapsed:.3f}s")
        return {"mw_tool_total_time": total_time}


# =====================================================
# 2. 模型调用限制 Middleware
# =====================================================
class ModelCallLimitMiddleware(BaseMiddleware):
    """模型调用次数限制 Middleware。
    
    不可变配置: max_calls（实例创建后不变）
    可变状态: mw_model_call_count（存在 AgentState 中）
    
    防止 agent 进入无限循环（rewrite → agent → rewrite → ...）
    """
    applicable_node_types = {"model"}  # 仅对模型调用节点生效

    def __init__(self, max_calls: int = 10):
        """
        Args:
            max_calls: 单次请求最大模型调用次数（跨所有节点累计）
        """
        # 只存不可变配置
        self.max_calls = max_calls

    def before_model(self, state: dict, node_name: str) -> Tuple[dict, bool]:
        """检查调用次数是否超限。从 state 读取计数，不用实例变量。"""
        current_count = state.get("mw_model_call_count", 0) + 1

        if current_count > self.max_calls:
            logger.warning(
                f"[CallLimit] [{node_name}] 模型调用次数 {current_count} "
                f"超过限制 {self.max_calls}，强制终止"
            )
            return {"mw_model_call_count": current_count, "mw_force_stop": True}, True

        logger.debug(f"[CallLimit] [{node_name}] 调用次数: {current_count}/{self.max_calls}")
        return {"mw_model_call_count": current_count}, False


# =====================================================
# 3. PII 检测 Middleware
# =====================================================
class PIIDetectionMiddleware(BaseMiddleware):
    """PII（个人身份信息）检测 Middleware。
    
    不可变配置: mode, patterns
    可变状态: mw_pii_detected（存在 AgentState 中）
    """
    applicable_node_types = {"model"}  # 对模型调用节点生效

    # PII 正则模式（类级别常量，不可变）
    PII_PATTERNS = {
        "phone": r'1[3-9]\d{9}',
        "id_card": r'\d{17}[\dXx]',
        "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "bank_card": r'(?<!\d)\d{16,19}(?!\d)',
    }

    def __init__(self, mode: str = "warn"):
        """
        Args:
            mode: 'warn'（记录告警）, 'mask'（脱敏后继续）, 'block'（拦截请求）
        """
        self.mode = mode

    def _detect_pii(self, text: str) -> list:
        """检测文本中的 PII（纯函数，无副作用）"""
        found = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, str(text))
            if matches:
                found.append({"type": pii_type, "count": len(matches)})
        return found

    def before_model(self, state: dict, node_name: str) -> Tuple[dict, bool]:
        """模型调用前检测用户输入中的 PII"""
        messages = state.get("messages", [])
        if not messages:
            return {}, False

        last_msg = messages[-1]
        content = getattr(last_msg, "content", str(last_msg))
        pii_found = self._detect_pii(content)

        if not pii_found:
            return {}, False

        logger.warning(f"[PII] [{node_name}] 检测到 PII: {pii_found}")

        if self.mode == "block":
            logger.error(f"[PII] [{node_name}] 拦截模式，阻止本次调用")
            return {"mw_pii_detected": True, "mw_force_stop": True}, True

        # warn 和 mask 模式：标记但不阻止
        return {"mw_pii_detected": True}, False

    def after_model(self, state: dict, response: Any, node_name: str, elapsed: float) -> dict:
        """模型调用后检测输出中的 PII（仅在 generate 和 agent 节点检查）"""
        if node_name not in ("generate", "agent"):
            return {}

        if hasattr(response, "content"):
            pii_found = self._detect_pii(response.content)
            if pii_found:
                logger.warning(f"[PII] [{node_name}] 模型输出中检测到 PII: {pii_found}")
        return {}


# =====================================================
# 4. 对话历史摘要 Middleware
# =====================================================
class SummarizationMiddleware(BaseMiddleware):
    """对话历史摘要 Middleware。
    
    仅在 agent 节点生效（其他节点无需处理消息截断）。
    不可变配置: max_messages, keep_recent
    无可变实例状态。
    """
    applicable_node_types = {"model"}

    def __init__(self, max_messages: int = 20, keep_recent: int = 5):
        """
        Args:
            max_messages: 消息数量阈值，超过时触发截断
            keep_recent: 保留最近的消息数量
        """
        self.max_messages = max_messages
        self.keep_recent = keep_recent
        # 指定仅在 agent 节点触发（避免在 grade/rewrite/generate 重复截断）
        self._target_nodes = {"agent"}

    def before_model(self, state: dict, node_name: str) -> Tuple[dict, bool]:
        """在 agent 节点调用前检查消息数量并截断"""
        if node_name not in self._target_nodes:
            return {}, False

        messages = state.get("messages", [])
        if len(messages) <= self.max_messages:
            return {}, False

        logger.info(
            f"[Summarization] [{node_name}] 消息数量 {len(messages)} "
            f"超过阈值 {self.max_messages}，截断保留最近 {self.keep_recent} 条"
        )
        # 注意：这里不直接修改 state["messages"]，
        # 而是通过返回 updates 让节点自己处理截断
        return {"_mw_should_truncate": True, "_mw_keep_recent": self.keep_recent}, False


# =====================================================
# 5. 工具重试 Middleware
# =====================================================
class ToolRetryMiddleware(BaseMiddleware):
    """工具调用重试 Middleware。
    
    不可变配置: max_retries, backoff_factor
    无可变实例状态（重试逻辑在 wrap_tool_call 中是纯函数式的）。
    """
    applicable_node_types = {"tool"}

    def __init__(self, max_retries: int = 2, backoff_factor: float = 0.5):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def wrap_tool_call(self, tool_func: Callable, tool_call: dict, tool_map: dict) -> Any:
        """包裹工具调用，添加重试逻辑（纯函数，无副作用到实例）"""
        last_error = None
        tool_name = tool_call.get("name", "unknown")

        for attempt in range(self.max_retries + 1):
            try:
                return tool_func(tool_call, tool_map)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor * (2 ** attempt)
                    logger.warning(
                        f"[ToolRetry] 工具 {tool_name} 第 {attempt + 1} 次失败: {e}，"
                        f"{wait_time:.1f}s 后重试"
                    )
                    time.sleep(wait_time)

        logger.error(f"[ToolRetry] 工具 {tool_name} 重试 {self.max_retries} 次后仍失败")
        raise last_error


# =====================================================
# Middleware 管理器（无状态编排器）
# =====================================================
class MiddlewareManager:
    """Middleware 管理器。
    
    本身无可变状态，只负责按顺序调度 Middleware 的 hooks。
    所有运行时数据通过 state dict 流转，确保多用户安全。
    
    执行顺序（与 Web 框架 Middleware 一致）：
    - before hooks: 正序执行 [mw1 → mw2 → mw3]
    - after hooks:  逆序执行 [mw3 → mw2 → mw1]
    """

    def __init__(self, middlewares: list = None):
        # middlewares 列表本身在初始化后不再修改（不可变引用）
        self.middlewares = middlewares or []
        # 按节点类型分组缓存，避免每次调用都过滤
        self._model_middlewares = [
            mw for mw in self.middlewares if "model" in mw.applicable_node_types
        ]
        self._tool_middlewares = [
            mw for mw in self.middlewares if "tool" in mw.applicable_node_types
        ]
        logger.info(
            f"[MiddlewareManager] 初始化: "
            f"model类={[type(m).__name__ for m in self._model_middlewares]}, "
            f"tool类={[type(m).__name__ for m in self._tool_middlewares]}"
        )

    def run_before_model(self, state: dict, node_name: str) -> Tuple[dict, bool]:
        """按正序执行所有 model 类 Middleware 的 before_model hook。
        
        Args:
            state: 当前 AgentState（只读引用）
            node_name: 节点名称
            
        Returns:
            (merged_updates, should_stop): 合并的状态更新 和 是否终止
        """
        merged_updates = {}
        # 将当前 state 和已收集的 updates 合并，供后续 middleware 读取最新值
        effective_state = {**state, **merged_updates}

        for mw in self._model_middlewares:
            try:
                updates, stop = mw.before_model(effective_state, node_name)
                if updates:
                    merged_updates.update(updates)
                    effective_state.update(updates)
                if stop:
                    logger.info(f"[MiddlewareManager] {type(mw).__name__} 在 {node_name} 触发终止")
                    return merged_updates, True
            except Exception as e:
                logger.error(f"[MiddlewareManager] {type(mw).__name__}.before_model 异常: {e}")

        return merged_updates, False

    def run_after_model(self, state: dict, response: Any, node_name: str, elapsed: float) -> dict:
        """按逆序执行所有 model 类 Middleware 的 after_model hook。"""
        merged_updates = {}
        effective_state = {**state, **merged_updates}

        for mw in reversed(self._model_middlewares):
            try:
                updates = mw.after_model(effective_state, response, node_name, elapsed)
                if updates:
                    merged_updates.update(updates)
                    effective_state.update(updates)
            except Exception as e:
                logger.error(f"[MiddlewareManager] {type(mw).__name__}.after_model 异常: {e}")

        return merged_updates

    def run_before_tool(self, state: dict, tool_call: dict) -> Tuple[dict, bool]:
        """按正序执行所有 tool 类 Middleware 的 before_tool hook。"""
        merged_updates = {}
        effective_state = {**state, **merged_updates}

        for mw in self._tool_middlewares:
            try:
                updates, stop = mw.before_tool(effective_state, tool_call)
                if updates:
                    merged_updates.update(updates)
                    effective_state.update(updates)
                if stop:
                    return merged_updates, True
            except Exception as e:
                logger.error(f"[MiddlewareManager] {type(mw).__name__}.before_tool 异常: {e}")

        return merged_updates, False

    def run_after_tool(self, state: dict, tool_result: Any, tool_name: str, elapsed: float) -> dict:
        """按逆序执行所有 tool 类 Middleware 的 after_tool hook。"""
        merged_updates = {}
        effective_state = {**state, **merged_updates}

        for mw in reversed(self._tool_middlewares):
            try:
                updates = mw.after_tool(effective_state, tool_result, tool_name, elapsed)
                if updates:
                    merged_updates.update(updates)
                    effective_state.update(updates)
            except Exception as e:
                logger.error(f"[MiddlewareManager] {type(mw).__name__}.after_tool 异常: {e}")

        return merged_updates

    def get_tool_retry_middleware(self) -> Optional['ToolRetryMiddleware']:
        """获取 ToolRetryMiddleware 实例（如果已注册）"""
        for mw in self._tool_middlewares:
            if isinstance(mw, ToolRetryMiddleware):
                return mw
        return None