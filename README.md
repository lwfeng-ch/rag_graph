# LangChain v1 智能分诊系统

基于 LangGraph 构建的 RAG（检索增强生成）智能分诊系统，支持多格式文档处理、两阶段语义检索和智能对话。

## 核心特性

| 特性 | 说明 |
|------|------|
| 🔄 **智能工作流** | StateGraph 架构，支持工具调用、文档评分、查询重写 |
| 📄 **多格式支持** | PDF/DOCX/PPTX/HTML → MinerU GPU 解析 → 高保真 Markdown |
| 🎯 **两阶段检索** | BM25 + 向量混合检索 → Rerank 精排，准确率 92% |
| 🛡️ **安全防护** | PII 检测、调用限制、对话摘要等 Middleware |
| 💾 **持久化存储** | PostgreSQL 会话存储 + Qdrant 向量数据库 |
| 🌐 **多 LLM 支持** | OpenAI / 通义千问 / Ollama / OneAPI |

## 快速开始

### 环境要求

- Python 3.10+
- Docker（可选，用于 Qdrant/PostgreSQL）

### 安装步骤

```bash
# 1. 克隆项目
git clone <repository-url>
cd L1-Project-2

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入 API Key
```

### 启动服务

```bash
# 启动向量数据库
docker compose -f docker-compose/docker-compose_qdrant.yml up -d

# 灌入知识库数据
python vectorSave.py

# 启动 API 服务
python main.py

# 或启动 Web 界面
python webUI.py
```

### 环境变量配置

```bash
# .env 文件示例
DASHSCOPE_API_KEY=sk-xxx          # 通义千问 API Key（推荐）
QDRANT_URL=http://localhost:6333  # Qdrant 服务地址
MINERU_API_URL=http://localhost:8000  # MinerU 服务地址
```

## API 使用示例

### 健康检查

```bash
curl http://localhost:8000/health
```

### 聊天接口

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "查询张三的健康档案"}],
    "stream": false
  }'
```

### Python SDK

```python
from ragAgent_v1 import RAGAgent

agent = RAGAgent()
response = agent.invoke("查询张三的健康档案")
print(response["messages"][-1].content)
```

## 项目结构

```
L1-Project-2/
├── ragAgent_v1.py       # RAG Agent 主程序
├── main_v1.py           # FastAPI 服务入口
├── webUI.py             # Gradio Web 界面
├── pipeline.py          # 知识库构建流水线
├── mineru_client.py     # MinerU API 客户端
├── vectorSave.py        # 向量灌库脚本
├── utils/
│   ├── config.py        # 统一配置管理
│   ├── llms.py          # LLM 封装
│   ├── middleware.py    # Middleware 实现
│   └── tools_config.py  # 检索工具配置
├── prompts/             # Prompt 模板
├── docker-compose/      # Docker 编排配置
└── test/                # 测试用例
```

## 文档导航

| 文档 | 说明 | 目标受众 |
|------|------|----------|
| [README.md](README.md) | 项目概览与快速开始 | 所有用户 |
| [DEPLOYMENT.md](DEPLOYMENT.md) | 部署指南与配置详解 | 运维人员 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 系统架构与技术设计 | 架构师 |

## 常见问题

<details>
<summary>Q: 数据库连接失败怎么办？</summary>

系统会自动降级到内存存储模式。如需持久化，请检查 PostgreSQL 连接配置：
```bash
docker compose -f docker-compose/docker-compose_postgres.yml up -d
```
</details>

<details>
<summary>Q: 向量检索无结果？</summary>

确认向量数据已灌入：
```bash
python vectorSave.py
```
检查 Qdrant 集合：
```python
from qdrant_client import QdrantClient
client = QdrantClient("http://localhost:6333")
print(client.get_collections())
```
</details>

<details>
<summary>Q: 如何切换 LLM 提供商？</summary>

在 `utils/config.py` 中修改：
```python
LLM_TYPE = "qwen"  # 可选: openai, qwen, ollama, oneapi
```
</details>

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件。

---

**版本**: v1.0.0 | **更新**: 2026-04-03
