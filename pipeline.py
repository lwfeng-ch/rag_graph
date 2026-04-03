# pipeline.py
"""
端到端文档处理流水线 v2
完整链路: 任意格式文件 → MinerU高保真Markdown → 两阶段语义切分 → 带元数据的向量存储

使用方式:
1. 端到端处理: Pipeline().run()
2. 分步处理: convert_files() → split_documents() → vectorize()
3. 单独使用各组件: MinerUClient / MarkdownSplitter / VectorStoreV2
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from utils.config import Config
from mineru_client import MinerUClient
from utils.markdown_splitter import MarkdownSplitter
from vectorSave2 import VectorStoreV2, KnowledgeBaseBuilder

logger = logging.getLogger(__name__)


class Pipeline:
    """端到端文档处理流水线"""

    def __init__(
        self,
        input_dir: str = None,
        output_dir: str = None,
        mineru_api_url: str = None,
        collection_name: str = None,
        clear_existing: bool = False
    ):
        """
        初始化流水线。

        Args:
            input_dir: 输入文件目录
            output_dir: Markdown 缓存输出目录
            mineru_api_url: MinerU 服务地址
            collection_name: Qdrant 集合名称
            clear_existing: 是否清空已有数据后重建
        """
        self.input_dir = input_dir or Config.INPUT_DIR
        self.output_dir = output_dir or Config.OUTPUT_DIR
        self.collection_name = collection_name or Config.QDRANT_COLLECTION_NAME
        self.clear_existing = clear_existing

        self.mineru_client = MinerUClient(api_url=mineru_api_url)
        self.splitter = MarkdownSplitter()
        self.vector_store = VectorStoreV2(collection_name=self.collection_name)

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def convert_files(
        self,
        input_dir: str = None,
        output_dir: str = None,
        parse_method: str = None,
        skip_cache: bool = False
    ) -> Dict[str, str]:
        """
        步骤1: 批量转换文件为 Markdown。

        Args:
            input_dir: 输入目录
            output_dir: 输出目录（Markdown缓存）
            parse_method: 解析方法 (auto/ocr/txt)
            skip_cache: 是否跳过已有缓存

        Returns:
            Dict: {文件名: Markdown内容}
        """
        input_dir = input_dir or self.input_dir
        output_dir = output_dir or self.output_dir

        logger.info(f"扫描输入目录: {input_dir}")

        results = self.mineru_client.convert_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            parse_method=parse_method,
            skip_existing=not skip_cache
        )

        success_count = sum(1 for v in results.values() if v)
        logger.info(f"文件转换完成: {success_count}/{len(results)} 成功")

        return results

    def split_documents(
        self,
        markdown_contents: Dict[str, str] = None,
        markdown_dir: str = None
    ) -> List[Dict[str, Any]]:
        """
        步骤2: 对 Markdown 文档执行两阶段语义切分。

        Args:
            markdown_contents: {文件名: Markdown内容} 字典（优先）
            markdown_dir: Markdown 文件目录（备选）

        Returns:
            List[Dict]: 所有切分后的 chunks
        """
        documents = []
        
        # 优先处理传入的 Markdown 内容
        if markdown_contents:
            for filename, content in markdown_contents.items():
                if content:
                    documents.append({"filename": filename, "content": content})
        elif markdown_dir:
            md_dir = Path(markdown_dir) if isinstance(markdown_dir, str) else markdown_dir
            for md_file in md_dir.glob("*.md"):
                with open(md_file, "r", encoding="utf-8") as f:
                    documents.append({
                        "filename": md_file.stem,
                        "content": f.read()
                    })
        else:
            md_dir = Path(self.output_dir)
            for md_file in md_dir.glob("*.md"):
                with open(md_file, "r", encoding="utf-8") as f:
                    documents.append({
                        "filename": md_file.stem,
                        "content": f.read()
                    })

        chunks = self.splitter.split_documents(documents)
        logger.info(f"文档切分完成: {len(documents)} 个文档 → {len(chunks)} 个 chunks")

        return chunks

    def vectorize(
        self,
        chunks: List[Dict[str, Any]],
        use_context_prefix: bool = True
    ) -> List[str]:
        """
        步骤3: 向量化并存储到 Qdrant。

        Args:
            chunks: 切分后的 chunks 列表
            use_context_prefix: 是否使用标题前置拼接

        Returns:
            List[str]: 写入的 ID 列表
        """
        if not chunks:
            logger.warning("没有可向量化的 chunks")
            return []

        if self.clear_existing:
            self.vector_store.clear_collection(clear=True)

        texts = [chunk["content"] for chunk in chunks]
        metadatas = [
            {
                "filename": chunk.get("filename", ""),
                "source": chunk.get("source_filename", chunk.get("filename", "")),
                **chunk.get("metadata", {})
            }
            for chunk in chunks
        ]

        ids = self.vector_store.upsert_with_metadata(
            texts=texts,
            metadatas=metadatas,
            use_context_prefix=use_context_prefix
        )

        logger.info(f"向量化完成: {len(ids)} 条数据已写入")
        return ids

    def run(
        self,
        input_dir: str = None,
        output_dir: str = None,
        parse_method: str = None
    ) -> Dict[str, Any]:
        """
        端到端执行完整流水线。

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            parse_method: MinerU 解析方法

        Returns:
            dict: 处理结果统计
        """
        logger.info("=" * 60)
        logger.info("开始端到端知识库构建流程")
        logger.info("=" * 60)

        md_results = self.convert_files(input_dir, output_dir, parse_method)

        chunks = self.split_documents(markdown_contents=md_results)

        ids = self.vectorize(chunks)

        result = {
            "success": True,
            "files_processed": sum(1 for v in md_results.values() if v),
            "total_files": len(md_results),
            "chunks_count": len(chunks),
            "vectors_stored": len(ids),
        }

        logger.info("=" * 60)
        logger.info(f"处理完成! {result}")
        logger.info("=" * 60)

        return result

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        搜索已构建的知识库。

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            dict: 搜索结果
        """
        return self.vector_store.search(query, top_n=top_k)


def quick_build(
    file_path: str = None,
    directory: str = None,
    mineru_url: str = None,
    query: str = None
):
    """
    快速构建知识库并可选执行测试搜索。

    Args:
        file_path: 单个文件路径
        directory: 目录路径（与 file_path 二选一）
        mineru_url: MinerU 服务地址
        query: 测试查询（可选）
    """
    builder = KnowledgeBaseBuilder(
        mineru_api_url=mineru_url,
        clear_existing=True
    )

    if file_path and os.path.exists(file_path):
        result = builder.build_from_file(file_path)
        logger.info(f"单文件构建结果: {result}")
    elif directory:
        result = builder.build_from_directory(directory)
        logger.info(f"批量构建结果: {result}")
    else:
        logger.error("请指定 file_path 或 directory")
        return

    if query:
        search_result = builder.search(query, top_n=3)
        logger.info(f"测试搜索 '{query}':")
        docs = search_result.get("documents", [[]])[0]
        scores = search_result.get("distances", [[]])[0]
        for doc, score in zip(docs, scores):
            logger.info(f"  [Score: {score:.4f}] {doc[:100]}...")


if __name__ == "__main__":
    import logging
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    print("=" * 60)
    print("Pipeline 端到端流水线自测")
    print("=" * 60)

    input_dir = Config.INPUT_DIR
    output_dir = Config.OUTPUT_DIR

    if not Path(input_dir).exists():
        files = list(Path(".").rglob("*.pdf")) + list(Path(".").rglob("*.docx"))
        if files:
            print(f"⚠️ input/ 目录不存在，但发现以下文件:")
            for f in files[:5]:
                print(f"  - {f}")
            print(f"\n建议: 创建 input/ 目录并放入待处理文件，或使用以下方式指定:")
            print(f'  Pipeline().run(input_dir="{f.parent}")')
        else:
            print(f"⚠️ 未找到任何可处理的文件 (.pdf/.docx/.pptx)")
            print(f"请将文件放入 {input_dir}/ 目录后重试")

        print("\n--- 切分器独立测试 (无需 MinerU 服务) ---")
        splitter = MarkdownSplitter()
        sample_md = """# 测试文档

## 第一节 简介

这是测试内容。用于验证两阶段切分功能是否正常工作。

## 第二节 详细说明

包含更多内容的第二节。
""" * 5

        chunks = splitter.split_text(sample_md)
        print(f"切分结果: {len(chunks)} 个 chunks")
        for i, c in enumerate(chunks[:3]):
            meta = c.get("metadata", {})
            print(f"  Chunk#{i+1}: h1={meta.get('h1','-')} h2={meta.get('h2','-')} len={len(c['content'])}")

        print("\n--- 流水线分步自测 (使用模拟数据) ---")
        pipeline = Pipeline(collection_name="test_pipeline_selfcheck")

        mock_docs = {
            "test_doc.md": "# 测试\n\n## 内容\n\n这是模拟的Markdown内容用于测试流水线。" * 10,
            "sample.md": "# 样例\n\n## 数据\n\n另一份模拟文档。" * 8,
        }
        chunks = pipeline.split_documents(markdown_contents=mock_docs)
        print(f"Step2 切分完成: {len(chunks)} 个 chunks")

        print("\n" + "=" * 60)
        print("自测完成 (MinerU 服务未连接时仅测试切分模块)")
        print("=" * 60)

    else:
        pipeline = Pipeline()
        result = pipeline.run()

        print(f"\n处理结果:")
        print(f"  文件数: {result['files_processed']}/{result['total_files']}")
        print(f"  Chunks: {result['chunks_count']}")
        print(f"  向量数: {result['vectors_stored']}")

        test_query = "基本信息"
        search_result = pipeline.search(test_query, top_k=3)
        docs = search_result.get("documents", [[]])[0]
        scores = search_result.get("distances", [[]])[0]

        print(f"\n搜索测试 '{test_query}':")
        for doc, score in zip(docs, scores):
            preview = doc[:80].replace("\n", " ")
            print(f"  [Score: {score:.4f}] {preview}...")
