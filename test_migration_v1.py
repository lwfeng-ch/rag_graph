"""
LangChain v1 迁移测试脚本

测试 ragAgent_v1.py 的核心功能，确保迁移后的代码能够正常运行。
"""

import sys
import os
import unittest
from unittest.mock import Mock, MagicMock, patch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestLangChainV1Migration(unittest.TestCase):
    """测试 LangChain v1 迁移功能"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        logger.info("开始 LangChain v1 迁移测试")
        
    def setUp(self):
        """每个测试前的初始化"""
        logger.info("\n" + "="*50)
        
    def test_import_ragAgent_v1(self):
        """测试导入 ragAgent_v1 模块"""
        try:
            import ragAgent_v1
            logger.info("✅ 成功导入 ragAgent_v1 模块")
            self.assertIsNotNone(ragAgent_v1)
        except ImportError as e:
            logger.error(f"❌ 导入 ragAgent_v1 失败: {e}")
            self.fail(f"导入失败: {e}")

    def test_import_main_v1(self):
        """测试导入 main_v1 模块"""
        try:
            import main_v1
            logger.info("✅ 成功导入 main_v1 模块")
            self.assertIsNotNone(main_v1)
        except ImportError as e:
            logger.error(f"❌ 导入 main_v1 失败: {e}")
            self.fail(f"导入失败: {e}")

    def test_agent_state_definition(self):
        """测试 AgentState 定义"""
        from ragAgent_v1 import AgentState
        logger.info("✅ AgentState 定义正确")
        self.assertTrue(hasattr(AgentState, '__annotations__'))
        self.assertIn('relevance_score', AgentState.__annotations__)
        self.assertIn('rewrite_count', AgentState.__annotations__)

    def test_context_definition(self):
        """测试 Context 定义"""
        from ragAgent_v1 import Context
        logger.info("✅ Context 定义正确")
        self.assertTrue(hasattr(Context, '__dataclass_fields__'))
        self.assertIn('user_id', Context.__dataclass_fields__)

    def test_tool_config_class(self):
        """测试 ToolConfig 类"""
        from ragAgent_v1 import ToolConfig
        
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        
        tool_config = ToolConfig([mock_tool])
        logger.info("✅ ToolConfig 初始化成功")
        self.assertEqual(tool_config.get_tools(), [mock_tool])
        self.assertIn("test_tool", tool_config.get_tool_names())

    def test_parallel_tool_node(self):
        """测试 ParallelToolNode 类"""
        from ragAgent_v1 import ParallelToolNode
        
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.invoke = Mock(return_value="test_result")
        
        tool_node = ParallelToolNode([mock_tool])
        logger.info("✅ ParallelToolNode 初始化成功")
        self.assertIsNotNone(tool_node)

    def test_document_relevance_score(self):
        """测试 DocumentRelevanceScore 模型"""
        from ragAgent_v1 import DocumentRelevanceScore
        
        score = DocumentRelevanceScore(binary_score="yes")
        logger.info("✅ DocumentRelevanceScore 创建成功")
        self.assertEqual(score.binary_score, "yes")

    def test_filter_messages(self):
        """测试 filter_messages 函数"""
        from ragAgent_v1 import filter_messages
        from langchain_core.messages import HumanMessage, AIMessage
        
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi"),
            HumanMessage(content="How are you?"),
            AIMessage(content="I'm fine"),
            HumanMessage(content="What about you?"),
        ]
        
        filtered = filter_messages(messages)
        logger.info(f"✅ filter_messages 过滤成功，保留 {len(filtered)} 条消息")
        self.assertEqual(len(filtered), 5)

    def test_route_after_tools(self):
        """测试 route_after_tools 函数"""
        from ragAgent_v1 import route_after_tools, ToolConfig
        
        mock_tool = Mock()
        mock_tool.name = "retrieve_test"
        
        tool_config = ToolConfig([mock_tool])
        
        state = {
            "messages": [
                Mock(name="retrieve_test", tool_calls=[])
            ]
        }
        
        result = route_after_tools(state, tool_config)
        logger.info(f"✅ route_after_tools 路由到: {result}")
        self.assertIn(result, ["generate", "grade_documents"])

    def test_route_after_grade(self):
        """测试 route_after_grade 函数"""
        from ragAgent_v1 import route_after_grade
        
        state = {
            "messages": [Mock()],
            "relevance_score": "yes",
            "rewrite_count": 0
        }
        
        result = route_after_grade(state)
        logger.info(f"✅ route_after_grade (relevance=yes) 路由到: {result}")
        self.assertEqual(result, "generate")
        
        state["relevance_score"] = "no"
        result = route_after_grade(state)
        logger.info(f"✅ route_after_grade (relevance=no) 路由到: {result}")
        self.assertEqual(result, "rewrite")

    def test_create_graph_v1_signature(self):
        """测试 create_graph_v1 函数签名"""
        from ragAgent_v1 import create_graph_v1
        import inspect
        
        sig = inspect.signature(create_graph_v1)
        params = list(sig.parameters.keys())
        logger.info(f"✅ create_graph_v1 参数: {params}")
        self.assertIn('llm_chat', params)
        self.assertIn('llm_embedding', params)
        self.assertIn('tool_config', params)
        self.assertIn('use_middleware', params)

    def test_agent_v1_function_exists(self):
        """测试 agent_v1 函数存在"""
        from ragAgent_v1 import agent_v1
        import inspect
        
        sig = inspect.signature(agent_v1)
        params = list(sig.parameters.keys())
        logger.info(f"✅ agent_v1 参数: {params}")
        self.assertIn('state', params)
        self.assertIn('runtime', params)
        self.assertIn('llm_chat', params)
        self.assertIn('tool_config', params)

    def test_grade_documents_function_exists(self):
        """测试 grade_documents 函数存在"""
        from ragAgent_v1 import grade_documents
        import inspect
        
        sig = inspect.signature(grade_documents)
        params = list(sig.parameters.keys())
        logger.info(f"✅ grade_documents 参数: {params}")
        self.assertIn('state', params)
        self.assertIn('llm_chat', params)

    def test_rewrite_function_exists(self):
        """测试 rewrite 函数存在"""
        from ragAgent_v1 import rewrite
        import inspect
        
        sig = inspect.signature(rewrite)
        params = list(sig.parameters.keys())
        logger.info(f"✅ rewrite 参数: {params}")
        self.assertIn('state', params)
        self.assertIn('llm_chat', params)

    def test_generate_function_exists(self):
        """测试 generate 函数存在"""
        from ragAgent_v1 import generate
        import inspect
        
        sig = inspect.signature(generate)
        params = list(sig.parameters.keys())
        logger.info(f"✅ generate 参数: {params}")
        self.assertIn('state', params)
        self.assertIn('llm_chat', params)

    def test_save_graph_visualization_v1(self):
        """测试 save_graph_visualization 函数"""
        from ragAgent_v1 import save_graph_visualization
        import inspect
        
        sig = inspect.signature(save_graph_visualization)
        params = list(sig.parameters.keys())
        logger.info(f"✅ save_graph_visualization 参数: {params}")
        self.assertIn('graph', params)
        self.assertIn('filename', params)

    def test_graph_response_v1_function_exists(self):
        """测试 graph_response_v1 函数存在"""
        from ragAgent_v1 import graph_response_v1
        import inspect
        
        sig = inspect.signature(graph_response_v1)
        params = list(sig.parameters.keys())
        logger.info(f"✅ graph_response_v1 参数: {params}")
        self.assertIn('graph', params)
        self.assertIn('user_input', params)
        self.assertIn('config', params)
        self.assertIn('tool_config', params)
        self.assertIn('context', params)

    def test_main_function_exists(self):
        """测试 main 函数存在"""
        from ragAgent_v1 import main
        logger.info("✅ main 函数存在")
        self.assertIsNotNone(main)

    def test_content_blocks_support(self):
        """测试 content_blocks 支持"""
        from ragAgent_v1 import agent_v1
        import inspect
        
        source = inspect.getsource(agent_v1)
        self.assertIn('content_blocks', source)
        logger.info("✅ agent_v1 包含 content_blocks 支持")

    def test_middleware_support(self):
        """测试 Middleware 支持"""
        from ragAgent_v1 import create_graph_v1
        import inspect
        
        source = inspect.getsource(create_graph_v1)
        self.assertIn('PIIMiddleware', source)
        self.assertIn('SummarizationMiddleware', source)
        logger.info("✅ create_graph_v1 包含 Middleware 支持")

    def test_backward_compatibility(self):
        """测试向后兼容性"""
        from ragAgent_v1 import (
            ToolConfig,
            DocumentRelevanceScore,
            ParallelToolNode,
            get_latest_question,
            filter_messages,
            store_memory,
            create_chain,
            route_after_tools,
            route_after_grade
        )
        
        logger.info("✅ 所有核心函数和类保持兼容")
        self.assertIsNotNone(ToolConfig)
        self.assertIsNotNone(DocumentRelevanceScore)
        self.assertIsNotNone(ParallelToolNode)
        self.assertIsNotNone(get_latest_question)
        self.assertIsNotNone(filter_messages)
        self.assertIsNotNone(store_memory)
        self.assertIsNotNone(create_chain)
        self.assertIsNotNone(route_after_tools)
        self.assertIsNotNone(route_after_grade)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_full_import_chain(self):
        """测试完整的导入链"""
        try:
            from ragAgent_v1 import (
                AgentState,
                Context,
                ToolConfig,
                DocumentRelevanceScore,
                ParallelToolNode,
                get_latest_question,
                filter_messages,
                store_memory,
                create_chain,
                agent_v1,
                grade_documents,
                rewrite,
                generate,
                route_after_tools,
                route_after_grade,
                save_graph_visualization,
                create_graph_v1,
                graph_response_v1,
                main
            )
            logger.info("✅ 完整导入链成功")
            self.assertTrue(True)
        except ImportError as e:
            logger.error(f"❌ 完整导入链失败: {e}")
            self.fail(f"导入失败: {e}")


def run_tests():
    """运行所有测试"""
    logger.info("\n" + "="*70)
    logger.info("开始运行 LangChain v1 迁移测试套件")
    logger.info("="*70 + "\n")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestLangChainV1Migration))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    logger.info("\n" + "="*70)
    logger.info("测试完成")
    logger.info(f"运行测试: {result.testsRun}")
    logger.info(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    logger.info(f"失败: {len(result.failures)}")
    logger.info(f"错误: {len(result.errors)}")
    logger.info("="*70 + "\n")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
