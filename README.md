# 智能医疗分诊系统

基于 **LangGraph** 的双路由智能医疗分诊系统，融合 RAG（检索增强生成）技术与医疗专业知识图谱，支持通用问答与医疗分诊的双轨并行处理。

## 核心特性

| 特性 | 说明 |
|------|------|
| 🔄 **双路由架构** | General RAG Agent + Medical Agent 物理隔离 |
| 📄 **多格式文档处理** | PDF/DOCX/PPTX/HTML → MinerU GPU 解析 → 高保真 Markdown |
| 🎯 **两阶段语义检索** | BM25 + 向量混合检索 → DashScope Rerank 精排 |
| 🛡️ **安全防护体系** | PII 检测、调用限制、对话摘要、工具重试等 Middleware |
| 💾 **持久化存储** | PostgreSQL 会话存储（langgraph-checkpoint-postgres）+ Qdrant 向量数据库 |
| 🌐 **多 LLM 支持** | OpenAI / 通义千问 / Ollama / OneAPI 统一适配 |
| 🏥 **医疗分析工具链** | 血常规、血生化、尿常规、生命体征、症状分析 5 大专业工具 |
| 🔒 **用户文档隔离** | 基于 user_id 的医疗文档私有检索与向量集合隔离 |

## 系统架构

```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart TB
    subgraph 表现层["🎨 表现层"]
        A1["Gradio WebUI<br/>gradio_ui.py"]
        A2["FastAPI REST API<br/>main.py :8000"]
    end
    
    subgraph 应用层["⚙️ LangGraph Runtime"]
        B1["意图路由器<br/>intent_router()"]
        B2["RAG Agent<br/>agent()"]
        B3["Medical Agent<br/>medical_agent()"]
        B4["并行工具节点<br/>ParallelToolNode"]
    end
    
    subgraph 工具层["🔧 工具层"]
        C1["health_record_retriever<br/>系统知识库检索"]
        C2["user_medical_document_retriever<br/>用户文档检索"]
        C3["analyze_cbc_report<br/>血常规分析"]
        C4["analyze_biochemistry_report<br/>血生化分析"]
        C5["analyze_urinalysis_report<br/>尿常规分析"]
        C6["analyze_vital_signs<br/>生命体征分析"]
        C7["analyze_symptoms<br/>症状分析"]
    end
    
    subgraph 数据层["📦 基础设施层"]
        D1["Qdrant<br/>向量数据库 :6333"]
        D2["PostgreSQL<br/>会话持久化 :5432"]
        D3["MinerU<br/>GPU 解析服务 :8000"]
    end
    
    A2 --> B1
    B1 -->|general| B2
    B1 -->|medical| B3
    B2 --> B4
    B3 --> B4
    B4 --> C1 & C2 & C3 & C4 & C5 & C6 & C7
    C1 --> D1
    C2 --> D1
    D3 --> D1
    
    style 表现层 fill:#3498db,color:#fff
    style 应用层 fill:#9b59b6,color:#fff
    style 工具层 fill:#f39c12,color:#000
    style 数据层 fill:#27ae60,color:#fff
```

## 快速开始

### 环境要求

- Python 3.10+
- Docker（可选，用于 Qdrant/PostgreSQL）
- NVIDIA GPU（可选，用于本地 MinerU）

### 安装步骤

```bash
# 1. 克隆项目
git clone <repository-url>
cd L1-Project-2

# 2. 创建虚拟环境
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # Linux/Mac

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入 API Key
```

### 启动服务

```bash
# ① 启动向量数据库和 PostgreSQL
docker compose -f docker-compose/docker-compose_qdrant.yml up -d
docker compose -f docker-compose/docker-compose_postgres.yml up -d

# ② 灌入系统知识库数据
python vectorSave.py

# ③ 启动 FastAPI 服务
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# ④ 或启动 Gradio Web 界面
python gradio_ui.py
```

### 环境变量配置

```bash
# .env 文件示例

# ===== LLM 配置 =====
LLM_TYPE=qwen                                    # 可选: openai, qwen, ollama, oneapi
DASHSCOPE_API_KEY=sk-xxx                         # 通义千问 API Key
OPENAI_API_KEY=sk-xxx                            # OpenAI API Key
OPENAI_API_BASE=https://api.openai.com/v1        # OpenAI API Base URL

# ===== 向量数据库配置 =====
QDRANT_URL=http://localhost:6333                 # Qdrant 服务地址
QDRANT_API_KEY=                                  # Qdrant API Key（本地可空）
QDRANT_COLLECTION_NAME=knowledge_base_v2         # 集合名称

# ===== MinerU 文档解析配置 =====
MINERU_API_URL=http://localhost:8000             # MinerU 服务地址
MINERU_TIMEOUT=300                               # 超时秒数

# ===== 数据库配置 =====
DB_URI=postgresql://rag_user:rag_password@localhost:5432/rag_db

# ===== LangSmith 追踪（可选）=====
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_xxx
LANGCHAIN_PROJECT=ragAgent-Prod
```

## API 使用示例

### 健康检查

```bash
curl http://localhost:8000/health
```

### 聊天接口（非流式）

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "messages": [{"role": "user", "content": "帮我分析这份血常规报告"}],
    "stream": false,
    "conversationId": "conv_001"
  }'
```

### 聊天接口（流式 SSE）

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "messages": [{"role": "user", "content": "查询张三的健康档案"}],
    "stream": true,
    "conversationId": "conv_001"
  }'
```

### 医疗响应格式（仅 medical 路由返回）

```json
{
  "id": "chatcmpl-xxx",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "根据您的血常规报告分析..."
    }
  }],
  "medical": {
    "risk_level": "medium",
    "risk_warning": "白细胞计数偏高，建议进一步检查",
    "disclaimer": "本系统提供的医疗建议仅供参考...",
    "structured_data": {
      "recommended_departments": ["内科", "血液科"],
      "urgency_level": "routine",
      "triage_confidence": 0.85
    }
  }
}
```

## 项目结构

```
L1-Project-2/
├── main.py                          # FastAPI 服务入口（681行）
├── ragAgent.py                      # Agent 核心逻辑 / LangGraph 编译（1200+行）
├── vectorSave.py                    # 向量存储引擎 v2（混合检索）
├── pipeline.py                      # 端到端文档处理流水线
├── gradio_ui.py                     # Gradio Web 前端界面
├── config.py                        # 统一配置兼容层
├── mineru_client.py                 # MinerU 文档解析客户端
│
├── prompts/                         # 提示词模板目录
│   ├── prompt_template_intent_router.txt      # 意图路由提示词
│   ├── prompt_template_agent.txt              # RAG Agent 提示词
│   ├── prompt_template_medical_agent.txt      # Medical Agent 提示词
│   ├── prompt_template_medical_agent_cb.txt   # 断路器模式提示词
│   ├── prompt_template_generate.txt           # 响应生成提示词
│   ├── prompt_template_rewrite.txt            # 查询重写提示词
│   └── prompt_template_grade.txt              # 文档评分提示词
│
├── utils/                           # 工具模块目录
│   ├── config.py                    # 配置入口
│   ├── llms.py                      # LLM 客户端工厂（4供应商）
│   ├── tools_config.py              # 工具工厂（物理隔离）
│   ├── retriever.py                 # 两阶段混合检索器
│   ├── middleware.py                # Middleware 中间件体系（5种）
│   ├── auth.py                      # 用户认证模块（3方式）
│   ├── document_processor.py        # 用户文档处理器
│   │
│   ├── config/                      # 配置子模块（组合继承）
│   │   ├── base_config.py           # 基础配置
│   │   ├── llm_config.py            # LLM 配置
│   │   ├── vectorstore_config.py    # 向量库配置
│   │   ├── middleware_config.py     # Middleware 配置
│   │   └── service_config.py        # 服务配置
│   │
│   └── medical_analysis/            # 医疗分析子模块
│       ├── base_analyzer.py         # 分析器抽象基类
│       ├── cbc_analyzer.py          # 血常规分析器（15+指标）
│       ├── biochemistry_analyzer.py # 血生化分析器（20+指标）
│       ├── urinalysis_analyzer.py   # 尿常规分析器（10+指标）
│       ├── vital_signs_analyzer.py  # 生命体征分析器
│       ├── symptom_analyzer.py      # 症状分析器
│       └── medical_tools.py         # LangChain Tool 封装
│
├── test/                            # 测试目录
├── docker-compose/                  # Docker 编排
│   ├── docker-compose_qdrant.yml
│   ├── docker-compose_mineru.yml
│   └── docker-compose_postgres.yml
│
├── requirements.txt                 # Python 依赖清单
└── .env.example                     # 环境变量模板
```

## 核心模块说明

| 模块 | 文件 | 功能 |
|------|------|------|
| **意图路由** | `ragAgent.py` → `intent_router()` | 双层判断：关键词预检(30+) + LLM 分类 |
| **RAG Agent** | `ragAgent.py` → `agent()` | 检索→评分→重写→生成 循环 |
| **Medical Agent** | `ragAgent.py` → `medical_agent()` | 医疗工具调用 + 断路器防死循环 |
| **并行执行** | `ragAgent.py` → `ParallelToolNode` | ThreadPoolExecutor + 超时控制 |
| **混合检索** | `utils/retriever.py` | BM25+Dense 粗排 Top5 → Rerank 精排 Top3 |
| **Middleware** | `utils/middleware.py` | 日志/限流/PII/摘要/重试 5 种中间件 |
| **认证体系** | `utils/auth.py` | API Key / JWT Token / 开发模式 三重验证 |
| **文档处理** | `utils/document_processor.py` | 上传→解析→切分→向量化→存储 |

## 双路由工作流

```mermaid
%%{init: {'theme': 'dark'}}%%
stateDiagram-v2
    [*] --> intent_router: 用户输入到达
    
    state intent_router {
        [*] --> KeywordCheck
        KeywordCheck --> LLM_Classify: 无关键词匹配
        LLM_Classify --> [*]: 输出 route_domain
        KeywordCheck --> [*]: 强制 medical
    }
    
    intent_router --> general_path: route=general
    intent_router --> medical_path: route=medical
    
    state general_path {
        agent --> call_tools: 需要检索
        call_tools --> grade_documents: 检索完成
        grade_documents --> rewrite: score=no
        grade_documents --> generate: score=yes
        rewrite --> agent: 重写查询
        agent --> generate: 直接回答
    }
    
    state medical_path {
        medical_agent --> med_call_tools: 需要工具
        med_call_tools --> medical_analysis: 工具返回结果
        medical_analysis --> department_triage: 分析完成
        department_triage --> medical_safety_guard: 分诊完成
        medical_safety_guard --> generate: 安全检查通过
        medical_agent --> generate: 直接回答
    }
    
    generate --> [*]: 返回响应
```

## 文档导航

| 文档 | 说明 | 目标受众 |
|------|------|----------|
| [README.md](README.md) | 项目概览与快速开始 | 所有用户 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 系统架构与技术设计 | 架构师 / 开发者 |
| [DEPLOYMENT.md](DEPLOYMENT.md) | 部署指南与配置详解 | 运维人员 |
| [项目说明.md](项目说明.md) | 全面技术解读报告 | 技术评审 |
| [面试说明文档.md](面试说明文档.md) | 面试准备要点 | 求职者 |

## 常见问题

<details>
<summary>Q: 如何切换 LLM 提供商？</summary>

在 `.env` 文件中修改：
```bash
LLM_TYPE=openai   # 可选: qwen, ollama, oneapi, openai
```
支持的提供商：OpenAI (gpt-4o)、通义千问 (qwen-plus)、Ollama (qwen2.5:32b)、OneAPI (自定义)
</details>

<details>
<summary>Q: 向量检索无结果？</summary>

确认向量数据已灌入：
```bash
python vectorSave.py
```
检查 Qdrant 集合状态：
```bash
curl http://localhost:6333/collections/knowledge_base_v2
```
</details>

<details>
<summary>Q: 数据库连接失败怎么办？</summary>

系统会自动降级到内存存储模式。如需持久化：
```bash
docker compose -f docker-compose/docker-compose_postgres.yml up -d
```
</details>

<details>
<summary>Q: MinerU 服务如何部署？</summary>

推荐使用 Cloud Studio GPU 服务器部署，详见 [DEPLOYMENT.md](DEPLOYMENT.md) 第三章。
也可使用本地 Docker 部署（需要 NVIDIA GPU）：
```bash
docker compose -f docker-compose/docker-compose_mineru.yml up -d
```
</details>

<details>
<summary>Q: 如何上传用户的医疗文档？</summary>

通过 API 接口上传：
```bash
curl -X POST http://localhost:8000/v1/documents/upload \
  -F "file=@report.pdf" \
  -F "doc_type=blood_report" \
  -H "X-API-Key: your-api-key"
```
</details>

---

**版本**: v2.0.0 | **更新**: 2026-04-19 | **框架**: LangGraph 1.0+ / FastAPI 0.100+
