# markdown_splitter.py
"""
Markdown 两阶段语义切分器
阶段A: 按 Markdown 标题层级（# ## ###）进行语义切分
阶段B: 超长段落使用 RecursiveCharacterTextSplitter 进行软切分
每个 Chunk 自带标题层级元数据，用于提升检索精度
"""
import re
import logging
from typing import List, Dict, Any, Optional

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _HAS_LANGCHAIN_SPLITTER = True
except ImportError:
    _HAS_LANGCHAIN_SPLITTER = False

from .config import Config

logger = logging.getLogger(__name__)


class MarkdownSplitter:
    """Markdown 两阶段语义切分器"""

    def __init__(
        self,
        headers: List[tuple] = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: List[str] = None,
        max_chunk_length: int = None
    ):
        """
        初始化 Markdown 切分器。

        Args:
            headers: 标题层级配置列表，如 [("#", "h1"), ("##", "h2")]
            chunk_size: 阶段B最大字符数
            chunk_overlap: 阶段B重叠字符数
            separators: 阶段B分隔符列表
            max_chunk_length: 单个Chunk最大长度上限
        """
        self.headers = headers or Config.MARKDOWN_HEADERS
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.separators = separators or Config.SEPARATORS
        self.max_chunk_length = max_chunk_length or Config.MAX_CHUNK_LENGTH

        header_patterns = [re.escape(h[0]) for h in self.headers]
        self._header_regex = re.compile(
            r"^( " + "|".join(header_patterns) + r") .+$",
            re.MULTILINE
        )

        if _HAS_LANGCHAIN_SPLITTER:
            self._recursive_splitter = RecursiveCharacterTextSplitter(
                separators=self.separators,
                chunk_size=self.chunk_size,
                length_function=len,
                chunk_overlap=self.chunk_overlap,
            )
        else:
            self._recursive_splitter = None
            logger.warning("langchain_text_splitters 未安装，阶段B切分将使用内置实现")

    def split_text(self, text: str) -> List[Dict[str, Any]]:
        """
        对 Markdown 文本执行两阶段语义切分。

        Args:
            text: Markdown 格式的文本内容

        Returns:
            List[Dict]: 切分后的 chunks，每项包含:
                - content: 切分后的文本内容
                - metadata: 元数据字典，包含标题层级信息
                  {h1: "标题", h2: "子标题", ...}
        """
        if not text or not text.strip():
            return []

        sections = self._split_by_headers(text)
        chunks = []

        for section in sections:
            content = section["content"].strip()
            if not content:
                continue
            
            metadata = section.get("metadata", {})
            if len(content) <= self.chunk_size:
                chunks.append({
                    "content": content,
                    "metadata": dict(metadata)
                })
            else:
                sub_chunks = self._split_long_content(content)
                for sub in sub_chunks:
                    combined_meta = dict(metadata)
                    combined_meta.update(sub.get("metadata", {}))
                    chunks.append({
                        "content": sub["content"],
                        "metadata": combined_meta
                    })

        logger.info(f"两阶段切分完成: 原文 → {len(sections)} 个段落 → {len(chunks)} 个 chunks")
        return chunks

    def _split_by_headers(self, text: str) -> List[Dict[str, Any]]:
        """
        阶段A: 按标题层级切分。

        Args:
            text: Markdown 文本

        Returns:
            List[Dict]: 切分后的段落列表
        """
        lines = text.split("\n")
        sections = []
        current_section = {"content": "", "metadata": {}}
        header_stack = {}

        for line in lines:
            header_match = self._is_header(line)

            if header_match:
                if current_section["content"].strip():
                    sections.append({
                        "content": current_section["content"],
                        "metadata": dict(current_section["metadata"])
                    })

                level, header_text = header_match
                level_key = f"h{level}"
                header_stack[level_key] = header_text

                for k in list(header_stack.keys()):
                    k_num = int(k[1:])
                    if k_num > level:
                        del header_stack[k]
                        
                current_section = {
                    "content": line + "\n",
                    "metadata": dict(header_stack)
                }
            else:
                current_section["content"] += line + "\n"

        if current_section["content"].strip():
            sections.append({
                "content": current_section["content"],
                "metadata": dict(current_section["metadata"])
            })

        if not sections:
            sections.append({"content": text, "metadata": {}})

        return sections

    def _is_header(self, line: str) -> Optional[tuple]:
        """判断是否为标题行，返回 (级别, 标题文本) 或 None"""
        for i, (pattern, key) in enumerate(self.headers):
            if line.startswith(pattern + " ") or line.startswith(pattern + "\t"):
                header_text = line[len(pattern):].strip()
                if header_text:
                    return (i + 1, header_text)
        return None

    def _split_long_content(self, content: str) -> List[Dict[str, Any]]:
        """
        阶段B: 使用 RecursiveCharacterTextSplitter 切分超长内容。

        Args:
            content: 超长文本内容

        Returns:
            List[Dict]: 切分后的子块列表
        """
        raw_chunks = []
        if self._recursive_splitter:
            raw_chunks = self._recursive_splitter.split_text(content)
        else:
            raw_chunks = self._fallback_split(content)
        result = []

        for chunk in raw_chunks:
            if len(chunk) > self.max_chunk_length:
                sub_parts = self._force_split(chunk)
                for part in sub_parts:
                    result.append({"content": part.strip(), "metadata": {}})
            else:
                result.append({"content": chunk.strip(), "metadata": {}})

        return result

    def _force_split(self, text: str, max_len: int = None) -> List[str]:
        """
        强制按最大长度切分文本（兜底方案）。

        Args:
            text: 待切分文本
            max_len: 最大长度

        Returns:
            List[str]: 切分后的文本列表
        """
        max_len = max_len or self.max_chunk_length
        parts = []
        while len(text) > max_len:
            cut_pos = text.rfind("\n", 0, max_len)
            if cut_pos == -1:
                cut_pos = text.rfind("。", 0, max_len)
            if cut_pos == -1:
                cut_pos = text.rfind(" ", 0, max_len)
            if cut_pos == -1:
                cut_pos = max_len

            parts.append(text[:cut_pos].strip())
            text = text[cut_pos:].strip()

        if text:
            parts.append(text)

        return parts

    def _fallback_split(self, text: str) -> List[str]:
        """
        内置兜底切分实现（当 langchain_text_splitters 不可用时使用）。

        Args:
            text: 待切分文本

        Returns:
            List[str]: 切分后的文本列表
        """
        chunks = []
        current_chunk = ""
        for sep in self.separators:
            if sep == "":
                continue
            if sep in text:
                segments = text.split(sep)
                
                for seg in segments:
                    seg = seg.strip()
                    if not seg:
                        continue
                    
                    if len(current_chunk) + len(seg) < self.chunk_size:
                        current_chunk += (sep + seg if current_chunk else seg)
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = seg
                if current_chunk:
                    chunks.append(current_chunk)
                return chunks if chunks else [text]

        return self._force_split(text)

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量处理文档列表。

        Args:
            documents: 文档列表，每项为 {"filename": str, "content": str}

        Returns:
            List[Dict]: 所有文档的切分结果
        """
        all_chunks = []

        for doc in documents:
            filename = doc.get("filename", "unknown")
            content = doc.get("content", "")

            chunks = self.split_text(content)

            for chunk in chunks:
                chunk["filename"] = filename
                all_chunks.append(chunk)

        logger.info(f"批量切分完成: {len(documents)} 个文档 → {len(all_chunks)} 个 chunks")
        return all_chunks

    def build_context_string(self, chunk: Dict[str, Any]) -> str:
        """
        构建带标题上下文的检索字符串（标题前置拼接）。
        用于在 Embedding 和检索时提供更好的上下文。

        Args:
            chunk: 包含 content 和 metadata 的字典

        Returns:
            str: 拼接了标题信息的完整字符串
        """
        metadata = chunk.get("metadata", {})
        content = chunk.get("content", "")

        context_parts = []
        for _, key in self.headers:
            if key in metadata and metadata[key]:
                prefix_level = int(key[1:])
                prefix = "#" * prefix_level
                context_parts.append(f"{prefix} {metadata[key]}")
        
        if context_parts:
            return "\n".join(context_parts) + "\n\n" + content
        return content


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    SAMPLE_MD = """# 第一章 患者基本信息

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
血压：120/80 mmHg，心率：72次/分。
体温：36.5℃，呼吸频率：16次/分。血常规检查显示白细胞计数正常，
红细胞计数在正常范围内。血小板数量充足，凝血功能指标均正常。
肝功能检测显示转氨酶水平正常，肾功能肌酐和尿素氮均在正常范围之内。
血脂四项中总胆固醇略高，需要进一步关注。
血糖空腹值为5.3mmol/L，属于正常水平。

## 2.2 影像学检查

胸部X光片显示心肺膈未见明显异常。
腹部B超提示肝脏形态大小正常，胆囊壁光滑。
"""

    splitter = MarkdownSplitter()

    print("=" * 60)
    print("Markdown Splitter 自测")
    print("=" * 60)

    chunks = splitter.split_text(SAMPLE_MD)

    print(f"\n切分结果: {len(chunks)} 个 chunks\n")

    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        content = chunk.get("content", "")
        h1 = meta.get("h1", "-")
        h2 = meta.get("h2", "-")
        preview = content[:80].replace("\n", " ")
        print(f"--- Chunk #{i+1} ---")
        print(f"  标题路径: [{h1}] > [{h2}]")
        print(f"  内容长度: {len(content)} 字符")
        print(f"  内容预览: {preview}...")
        print()

    print("=" * 60)
    print("标题前置拼接测试 (build_context_string)")
    print("=" * 60)

    if chunks:
        sample = chunks[0]
        context_str = splitter.build_context_string(sample)
        print(f"\n原始内容:\n{sample['content'][:100]}...\n")
        print(f"拼接后 (用于 Embedding):\n{context_str[:200]}...")
