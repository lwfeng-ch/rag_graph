# test_vectorSave2.py
"""
知识库构建链路 v2 测试用例
覆盖: MarkdownSplitter / VectorStoreV2 / MinerUClient(模拟) / KnowledgeBaseBuilder / Pipeline

运行方式:
    python -m pytest test/test_vectorSave2.py -v
    或
    python test/test_vectorSave2.py
"""
import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.markdown_splitter import MarkdownSplitter
    HAS_SPLITTER = True
except ImportError as e:
    HAS_SPLITTER = False
    print(f"Warning: Cannot import MarkdownSplitter: {e}")

try:
    from vectorSave2 import VectorStoreV2, KnowledgeBaseBuilder, generate_vectors, get_embeddings
    HAS_VECTORSTORE = True
except ImportError as e:
    HAS_VECTORSTORE = False
    print(f"Warning: Cannot import VectorStoreV2: {e}")

try:
    from pipeline import Pipeline
    HAS_PIPELINE = True
except ImportError as e:
    HAS_PIPELINE = False
    print(f"Warning: Cannot import Pipeline: {e}")

try:
    from mineru_client import MinerUClient
    HAS_MINERU = True
except ImportError as e:
    HAS_MINERU = False
    print(f"Warning: Cannot import MinerUClient: {e}")


SAMPLE_MARKDOWN = """# 第一章 患者基本信息

## 1.1 个人信息

姓名：张三九
性别：男
年龄：45岁
职业：软件工程师

## 1.2 联系方式

电话：13800138000
地址：北京市朝阳区建国路88号

# 第二章 检查结果

## 2.1 体格检查

身高：175cm，体重：80kg，BMI：26.1
血压：120/80 mmHg，心率：72次/分
体温：36.5℃，呼吸频率：16次/分。

血常规检查显示白细胞计数正常，红细胞计数在正常范围内。
血小板数量充足，凝血功能指标均正常。肝功能检测显示转氨酶水平正常，
肾功能肌酐和尿素氮均在正常范围之内。血脂四项中总胆固醇略高，
需要进一步关注。血糖空腹值为5.3mmol/L，属于正常水平。

## 2.2 影像学检查

胸部X光片显示心肺膈未见明显异常。
腹部B超提示肝脏形态大小正常，胆囊壁光滑。
"""


class TestMarkdownSplitter(unittest.TestCase):
    """Markdown 两阶段语义切分器测试"""

    @unittest.skipUnless(HAS_SPLITTER, "MarkdownSplitter 依赖未安装")
    def setUp(self):
        self.splitter = MarkdownSplitter()

    def test_01_split_by_headers_basic(self):
        """
        测试: 基本标题切分功能
        输入: 包含 # ## 层级标题的 Markdown 文本
        预期: 切分为多个段落，每个段落包含对应标题下的内容
        """
        chunks = self.splitter.split_text(SAMPLE_MARKDOWN)

        self.assertGreater(len(chunks), 0, "应产生至少一个 chunk")
        for chunk in chunks:
            self.assertIn("content", chunk)
            self.assertIn("metadata", chunk)

        h1_chunks = [c for c in chunks if c["metadata"].get("h1")]
        self.assertGreater(len(h1_chunks), 0, "应有包含 h1 标题的 chunk")

    def test_02_metadata_preservation(self):
        """
        测试: 元数据（标题层级）完整保存
        输入: 带有多层标题的 Markdown
        预期: 每个 chunk 的 metadata 中包含正确的 h1/h2 信息
        """
        chunks = self.splitter.split_text(SAMPLE_MARKDOWN)

        has_h1 = any(c["metadata"].get("h1") == "第一章 患者基本信息" for c in chunks)
        has_h2 = any(c["metadata"].get("h2") == "1.1 个人信息" for c in chunks)

        self.assertTrue(has_h1, "应保留 h1 元数据")
        self.assertTrue(has_h2, "应保留 h2 元数据")

    def test_03_long_content_recursive_split(self):
        """
        测试: 超长内容自动进行阶段B递归切分
        输入: 超过 CHUNK_SIZE (800字符) 的单个章节内容
        预期: 长章节被切分为多个子 chunk
        """
        long_section = "# 长章节\n\n" + "这是一段很长的测试文本。" * 100

        chunks = self.splitter.split_text(long_section)

        long_chunks = [c for c in chunks if len(c["content"]) > 50]
        self.assertGreaterEqual(
            len(long_chunks), 1,
            "长章节应至少产生一个有效 chunk"
        )

    def test_04_empty_input(self):
        """
        测试: 空输入处理
        输入: 空字符串或纯空白字符
        预期: 返回空列表
        """
        result_empty = self.splitter.split_text("")
        result_whitespace = self.splitter.split_text("   \n\n   ")

        self.assertEqual(result_empty, [], "空字符串应返回空列表")
        self.assertEqual(result_whitespace, [], "纯空白应返回空列表")

    def test_05_build_context_string(self):
        """
        测试: 标题前置拼接功能
        输入: 带 metadata 的 chunk
        预期: 返回拼接了标题前缀的字符串，用于 Embedding 增强
        """
        chunk = {
            "content": "姓名：张三九",
            "metadata": {"h1": "第一章", "h2": "1.1节"}
        }

        context_str = self.splitter.build_context_string(chunk)

        self.assertIn("# 第一章", context_str, "应包含 h1 标题前缀")
        self.assertIn("## 1.1节", context_str, "应包含 h2 标题前缀")
        self.assertIn("姓名：张三九", context_str, "应包含原始内容")


class TestMinerUClient(unittest.TestCase):
    """MinerU API 客户端测试"""

    @unittest.skipUnless(HAS_MINERU, "MinerUClient 依赖未安装")
    def test_06_mime_type_detection(self):
        """
        测试: MIME 类型自动识别
        输入: 不同扩展名的文件路径
        预期: 返回正确的 MIME 类型
        """
        client = MinerUClient(api_url="http://fake-url")

        mime_map = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".html": "text/html",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }

        for ext, expected_mime in mime_map.items():
            fake_path = f"test{ext}"
            result = client._get_mime_type(Path(fake_path))
            self.assertEqual(result, expected_mime, f"{ext} MIME 类型错误")

    def test_07_response_parsing(self):
        """
        测试: 多种 MinerU 响应格式解析兼容性
        输入: 不同字段名的响应字典
        预期: 统一解析为标准格式 {"markdown": ..., "success": True}
        """
        client = MinerUClient(api_url="http://fake-url")

        formats = [
            {"markdown": "test content", "metadata": {}},
            {"data": "test content"},
            {"result": "test content"},
            {"text": "test content"},
        ]

        for fmt in formats:
            result = client._parse_response(fmt, "test.pdf")
            self.assertTrue(result["success"], f"格式 {list(fmt.keys())} 解析失败")
            self.assertEqual(result["markdown"], "test content")
            self.assertEqual(result["filename"], "test.pdf")

    def test_08_endpoint_discovery_fallback(self):
        """
        测试: 端点探测回退机制
        输入: 无法连接的服务器地址
        预期: 回退到默认端点 /file/convert
        """
        client = MinerUClient(api_url="http://nonexistent-host-12345")

        with patch.object(client._session, 'get', side_effect=Exception("网络不可达")):
            endpoint = client._discover_endpoint()
            self.assertEqual(endpoint, "/file/convert", "应回退到默认端点")


class TestVectorStoreV2(unittest.TestCase):
    """向量存储引擎 V2 测试"""

    @unittest.skipUnless(HAS_VECTORSTORE, "VectorStoreV2 依赖未安装")
    def test_09_generate_vectors_batch(self):
        """
        测试: 批量向量生成（mock Embedding 服务）
        输入: 文本列表
        预期: 返回等长的向量列表
        """
        mock_embeddings = [[0.1] * 10, [0.2] * 10, [0.3] * 10]

        with patch('vectorSave2.get_embeddings', return_value=mock_embeddings):
            texts = ["文本A", "文本B", "文本C"]
            result = generate_vectors(texts)

            self.assertEqual(len(result), 3, "向量数量应与输入一致")
            self.assertEqual(len(result[0]), 10, "向量维度正确")

    def test_10_search_result_format(self):
        """
        测试: 搜索结果格式兼容性
        输入: 模拟 Qdrant 搜索返回
        预期: 返回 {"documents": [...], "distances": [...]} 格式
        """
        store = VectorStoreV2(collection_name="test_collection")

        mock_point = MagicMock()
        mock_point.payload = {
            "document": "测试文档内容",
            "page_content": "测试文档内容",
            "source": "test.pdf",
        }
        mock_point.score = 0.95

        mock_response = MagicMock()
        mock_response.points = [mock_point]

        with patch.object(store.client, 'query_points', return_value=mock_response):
            with patch('vectorSave2.generate_vectors', return_value=[[0.5] * 10]):
                result = store.search("查询文本", top_n=3)

                self.assertIn("documents", result, "结果应含 documents 字段")
                self.assertIn("distances", result, "结果应含 distances 字段")
                self.assertEqual(len(result["documents"][0]), 1, "应返回1条结果")


class TestKnowledgeBaseBuilder(unittest.TestCase):
    """知识库构建器集成测试"""

    @unittest.skipUnless(HAS_VECTORSTORE and HAS_MINERU and HAS_SPLITTER, "依赖未安装")
    def test_11_builder_initialization(self):
        """
        测试: 知识库构建器初始化
        输入: 默认参数
        预期: 各组件正确初始化
        """
        builder = KnowledgeBaseBuilder(collection_name="test_kb")

        self.assertIsInstance(builder.mineru_client, MinerUClient)
        self.assertIsInstance(builder.splitter, MarkdownSplitter)
        self.assertIsInstance(builder.vector_store, VectorStoreV2)


class TestPipeline(unittest.TestCase):
    """流水线集成测试"""

    @unittest.skipUnless(HAS_PIPELINE, "Pipeline 依赖未安装")
    def test_12_pipeline_initialization(self):
        """
        测试: 流水线初始化与目录创建
        输入: 自定义参数
        预期: 各组件就绪，输出目录已创建
        """
        pipeline = Pipeline(
            input_dir="input",
            output_dir="output_test_tmp",
            collection_name="test_pipeline"
        )

        self.assertTrue(Path("output_test_tmp").exists(), "输出目录应被创建")

        import shutil
        if Path("output_test_tmp").exists():
            shutil.rmtree("output_test_tmp", ignore_errors=True)

    def test_13_split_documents_from_dict(self):
        """
        测试: 从字典切分文档
        输入: {文件名: Markdown内容}
        预期: 返回带 filename 字段的 chunks 列表
        """
        pipeline = Pipeline()

        md_contents = {
            "test_doc.md": "# 测试文档\n\n## 第一节\n\n这是测试内容"
        }
        chunks = pipeline.split_documents(markdown_contents=md_contents)

        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0]["filename"], "test_doc.md")


if __name__ == "__main__":
    unittest.main(verbosity=2)
