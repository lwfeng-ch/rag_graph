# mineru_client.py
"""
MinerU 远程 API 客户端
支持多种 API 端点自动探测，兼容 MinerU 不同版本
提供文件转换、批量处理、健康检查等功能
"""

import os
import time
import json
import logging
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List

from utils.config import Config

logger = logging.getLogger(__name__)

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# 文件扩展名到 MIME 类型的映射
MIME_TYPE_MAP = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt": "application/vnd.ms-powerpoint",
    ".html": "text/html",
    ".htm": "text/html",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


# MinerU 客户端类
class MinerUClient:
    """MinerU 远程 API 客户端"""

    # 文件转换 API 端点列表
    CONVERT_ENDPOINTS = [
        "/file/convert",
        "/v1/file/convert",
        "/convert",
        "/v1/convert",
        "/file_parse",
        "/v1/parse",
        "/api/v1/extract",
    ]

    def __init__(self, api_url: str = None, timeout: int = None):
        """
        初始化 MinerU 客户端。

        Args:
            api_url: MinerU 服务地址，默认从 Config 读取
            timeout: 请求超时秒数，默认从 Config 读取
        """
        self.api_url = (api_url or Config.MINERU_API_URL).rstrip("/")
        self.timeout = timeout or Config.MINERU_TIMEOUT
        self._convert_endpoint = None
        self._session = requests.Session()
        logger.info(f"MinerU Client 初始化: {self.api_url}")

    def health_check(self) -> bool:
        """
        检查 MinerU 服务是否可用。

        Returns:
            bool: True=可用, False=不可用
        """
        try:
            resp = self._session.get(f"{self.api_url}/health", timeout=10)
            # 如果状态码是200，则认为服务可用
            if resp.status_code == 200:
                logger.info(f"MinerU 服务健康: {resp.json()}")
                return True
        except Exception as e:
            logger.error(f"MinerU 服务不可达: {e}")
        return False

    def _discover_endpoint(self) -> str:
        """自动探测可用的文件转换 API 端点"""
        if self._convert_endpoint:
            return self._convert_endpoint

        try:
            resp = self._session.get(f"{self.api_url}/openapi.json", timeout=10)
            if resp.status_code == 200:
                schema = resp.json()
                paths = schema.get("paths", {})
                for endpoint in self.CONVERT_ENDPOINTS:
                    if endpoint in paths:
                        self._convert_endpoint = endpoint
                        logger.info(f"从 OpenAPI schema 发现端点: {endpoint}")
                        return endpoint
                for path, methods in paths.items():
                    if "post" in methods and any(
                        kw in path.lower() for kw in ["convert", "parse", "extract"]
                    ):
                        self._convert_endpoint = path
                        logger.info(f"从 OpenAPI schema 发现端点: {path}")
                        return path
        except Exception as e:
            logger.debug(f"无法获取 OpenAPI schema: {e}")

        test_file_content = b"%PDF-1.4 test"
        for endpoint in self.CONVERT_ENDPOINTS:
            try:
                resp = self._session.post(
                    f"{self.api_url}{endpoint}",
                    files={"file": ("test.pdf", test_file_content, "application/pdf")},
                    timeout=15,
                )
                if resp.status_code not in (404, 405):
                    self._convert_endpoint = endpoint
                    logger.info(
                        f"探测到可用端点: {endpoint} (status={resp.status_code})"
                    )
                    return endpoint
            except Exception:
                continue

        self._convert_endpoint = "/file/convert"
        logger.warning(f"未探测到端点，使用默认: {self._convert_endpoint}")
        return self._convert_endpoint

    def _get_mime_type(self, file_path: Path) -> str:
        """根据文件扩展名获取 MIME 类型"""
        suffix = file_path.suffix.lower()
        return MIME_TYPE_MAP.get(suffix, "application/octet-stream")

    def convert_file(
        self,
        file_path: str,
        parse_method: str = None,
        return_images: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        上传文件到 MinerU 进行转换，返回 Markdown 结果。

        Args:
            file_path: 本地文件路径
            parse_method: 解析方法 (auto/ocr/txt)
            return_images: 是否返回图片
            **kwargs: 其他 MinerU 支持的参数

        Returns:
            dict: {
                "success": True/False,
                "markdown": "转换后的MD文本",
                "images": {},
                "metadata": {},
                "filename": "原始文件名"
            }
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 自动探测可用的文件转换 API 端点
        endpoint = self._discover_endpoint()
        url = f"{self.api_url}{endpoint}"
        parse_method = parse_method or Config.MINERU_PARSE_METHOD
        mime_type = self._get_mime_type(file_path)

        logger.info(
            f"上传文件: {file_path.name} "
            f"({file_path.stat().st_size / 1024:.1f} KB), "
            f"端点: {url}, 解析方法: {parse_method}"
        )

        with open(file_path, "rb") as f:
            files = [("files", (file_path.name, f, mime_type))]
            data = {
                "backend": "hybrid-auto-engine",
                "parse_method": parse_method,
                "return_md": True,
                "return_content_list": True,
                "return_images": return_images,
                "lang_list": ["ch"],
            }
            data.update(kwargs)

            max_retries = 3
            retry_delay = 5

            for attempt in range(max_retries):
                try:
                    resp = self._session.post(
                        url, files=files, data=data, timeout=self.timeout
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    return self._parse_response(result, file_path.name)

                except requests.exceptions.HTTPError as e:
                    if resp.status_code == 409 and attempt < max_retries - 1:
                        logger.warning(
                            f"服务器繁忙 (409)，{retry_delay}s 后重试 ({attempt + 1}/{max_retries})"
                        )
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    elif resp.status_code == 202:
                        return self._poll_async_result(resp.json())
                    else:
                        logger.error(f"HTTP 错误: {e}")
                        return {"success": False, "error": str(e), "markdown": ""}

                except requests.exceptions.Timeout:
                    logger.error(f"请求超时 ({self.timeout}s)，文件可能过大")
                    return {"success": False, "error": "timeout", "markdown": ""}

                except Exception as e:
                    logger.error(f"请求失败: {e}")
                    return {"success": False, "error": str(e), "markdown": ""}

            return {"success": False, "error": "max retries exceeded", "markdown": ""}

    # 轮询异步任务结果
    def _poll_async_result(
        self, initial_response: dict, poll_interval: int = 3, max_wait: int = None
    ) -> Dict[str, Any]:
        """轮询异步任务结果"""
        task_id = initial_response.get("task_id") or initial_response.get("id")
        if not task_id:
            logger.error("未获取到 task_id")
            return {"success": False, "error": "no task_id", "markdown": ""}

        max_wait = max_wait or self.timeout
        logger.info(f"异步任务: {task_id}，开始轮询...")

        result_endpoints = [
            f"/task/{task_id}",
            f"/result/{task_id}",
            f"/file/convert/{task_id}",
            f"/v1/task/{task_id}",
        ]

        start_time = time.time()
        while time.time() - start_time < max_wait:
            for ep in result_endpoints:
                try:
                    resp = self._session.get(f"{self.api_url}{ep}", timeout=10)
                    if resp.status_code == 200:
                        result = resp.json()
                        status = result.get("status", "").lower()

                        if status in ("completed", "done", "success", "finished"):
                            logger.info(f"任务完成: {task_id}")
                            return self._parse_response(result, task_id)
                        elif status in ("failed", "error"):
                            logger.error(f"任务失败: {result}")
                            return {"success": False, "error": result, "markdown": ""}
                        break
                    elif resp.status_code == 404:
                        continue
                except Exception:
                    continue

            time.sleep(poll_interval)

        logger.error(f"异步任务超时: {task_id}")
        return {"success": False, "error": "timeout", "markdown": ""}

    def _parse_response(self, result: dict, filename: str) -> Dict[str, Any]:
        """
        解析 MinerU 响应，兼容不同版本格式。

        Args:
            result: API 响应字典
            filename: 文件名

        Returns:
            Dict: 标准化的结果字典
        """
        markdown = ""
        images = {}
        metadata = {}

        if "results" in result:
            doc_name = Path(filename).stem
            doc_result = result["results"].get(
                doc_name, result["results"].get(filename, {})
            )
            markdown = doc_result.get("md_content", "")
            images = doc_result.get("images", {})
            content_list = doc_result.get("content_list", "")
            metadata = {
                "backend": result.get("backend"),
                "version": result.get("version"),
                "content_list": content_list,
            }
        else:
            markdown = (
                result.get("markdown")
                or result.get("data")
                or result.get("result")
                or result.get("text")
                or ""
            )
            images = result.get("images", {})
            if not images and "data" in result and isinstance(result["data"], dict):
                images = result["data"].get("images", {})
            metadata = result.get("metadata", {})

        return {
            "success": True,
            "markdown": markdown,
            "images": images,
            "metadata": metadata,
            "filename": filename,
        }

    def convert_directory(
        self,
        input_dir: str = None,
        output_dir: str = None,
        parse_method: str = None,
        skip_existing: bool = True,
    ) -> Dict[str, str]:
        """
        批量转换目录下所有支持格式的文件。

        Args:
            input_dir: 输入目录路径
            output_dir: 输出目录路径（用于保存Markdown缓存）
            parse_method: 解析方法
            skip_existing: 是否跳过已有缓存的文件

        Returns:
            Dict: {文件名: Markdown内容}
        """
        input_dir = Path(input_dir or Config.INPUT_DIR)
        output_dir = Path(output_dir or Config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        files_to_process = []

        for ext in Config.SUPPORTED_EXTENSIONS:
            files_to_process.extend(input_dir.glob(f"*{ext}"))

        logger.info(f"扫描到 {len(files_to_process)} 个待处理文件")

        for file_path in files_to_process:
            md_file = output_dir / f"{file_path.stem}.md"

            # 检查是否已有缓存, 如果有则跳过
            logger.info(f"检查缓存: {md_file}")
            if skip_existing and md_file.exists():
                logger.info(f"使用缓存: {file_path.name}")
                with open(md_file, "r", encoding="utf-8") as f:
                    results[file_path.name] = f.read()
                continue

            # 转换文件, 并保存缓存
            logger.info(f"开始转换: {file_path.name}")
            result = self.convert_file(str(file_path), parse_method=parse_method)

            # 处理转换结果, 并保存缓存
            if result["success"] and result["markdown"]:
                with open(md_file, "w", encoding="utf-8") as f:
                    f.write(result["markdown"])
                results[file_path.name] = result["markdown"]
                logger.info(f"转换完成: {file_path.name}")
            else:
                logger.error(f"转换失败: {file_path.name} - {result.get('error')}")
                results[file_path.name] = ""

        return results


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    client = MinerUClient()

    print("=" * 60)
    print("MinerU Client 自测")
    print("=" * 60)

    healthy = client.health_check()
    print(f"服务健康检查: {'[OK] 可用' if healthy else '[FAIL] 不可达'}")

    if healthy:
        endpoint = client._discover_endpoint()
        print(f"探测到的转换端点: {endpoint}")

        input_dir = Config.INPUT_DIR
        if Path(input_dir).exists():
            results = client.convert_directory(input_dir=input_dir)
            success = sum(1 for v in results.values() if v)
            print(f"批量转换结果: {success}/{len(results)} 成功")
            for name, content in results.items():
                status = "[OK]" if content else "[FAIL]"
                preview = (content[:50] + "...") if content else "(空)"
                print(f"  {status} {name}: {preview}")
        else:
            print(f"输入目录不存在: {input_dir}")
            print("提示: 将待转换文件放入 input/ 目录后重试")

    print("=" * 60)
