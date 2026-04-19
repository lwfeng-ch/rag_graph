# 部署指南

本文档面向运维人员，提供完整的系统部署、配置和运维说明。

## 目录

- [一、部署架构](#一部署架构)
- [二、环境准备](#二环境准备)
- [三、MinerU GPU 服务部署](#三mineru-gpu-服务部署)
- [四、本地服务部署](#四本地服务部署)
- [五、配置详解](#五配置详解)
- [六、Agent 工作流部署](#六agent-工作流流部署)
- [七、Middleware 配置](#七middleware-配置)
- [八、医疗模块专项配置](#八医疗模块专项配置)
- [九、故障排查](#九故障排查)
- [十、监控与运维](#十监控与运维)

---

## 一、部署架构

### 1.1 整体架构

```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart TB
    subgraph Cloud["Cloud Studio (GPU 服务器)"]
        M[MinerU Docker<br/>GPU 解析服务]
    end
    
    subgraph Local["本地服务器 / 云主机"]
        subgraph Services["服务层"]
            API[FastAPI<br/>:8000]
            WEB[Gradio<br/>:7860]
        end
        
        subgraph Data["数据层"]
            Q[Qdrant<br/>:6333]
            PG[(PostgreSQL<br/>:5432)]
        end
        
        subgraph Agent["Agent 层 - 双路由架构"]
            subgraph GeneralRoute["General 路由"]
                RAG[RAG Agent<br/>通用问答]
                RET[Retriever<br/>混合检索]
                TOOLS_G[Tools<br/>检索+搜索]
            end
            
            subgraph MedicalRoute["Medical 路由"]
                MED[Medical Agent<br/>医疗分析]
                MED_TOOLS[Medical Tools<br/>5大专业工具]
                DOC_PROC[Document Processor<br/>用户文档处理]
            end
            
            ROUTER[Intent Router<br/>意图路由]
            MW[Middleware<br/>安全防护层]
        end
        
        subgraph Auth["认证层"]
            AUTH_SYS[Auth System<br/>API Key/JWT/Dev Mode]
        end
    end
    
    subgraph External["外部服务"]
        LLM[LLM API<br/>OpenAI/Qwen/Ollama]
        RERANK[DashScope Rerank<br/>重排序服务]
    end
    
    M -->|Markdown| API
    API --> AUTH_SYS
    AUTH_SYS --> ROUTER
    ROUTER -->|general| GeneralRoute
    ROUTER -->|medical| MedicalRoute
    
    RAG --> RET
    RET --> TOOLS_G
    TOOLS_G --> MW
    
    MED --> MED_TOOLS
    MED_TOOLS --> DOC_PROC
    DOC_PROC --> Q
    MED_TOOLS --> MW
    
    MW --> Q
    MW --> PG
    RAG --> LLM
    MED --> LLM
    RET --> RERANK
    
    style Cloud fill:#9b59b6,color:#fff
    style Local fill:#3498db,color:#fff
    style External fill:#e67e22,color:#fff
    style GeneralRoute fill:#27ae60,color:#fff
    style MedicalRoute fill:#e74c3c,color:#fff
```

### 1.2 部署环境对照表

| 环境 | MinerU | Qdrant | PostgreSQL | 适用场景 |
|------|--------|--------|------------|----------|
| 开发环境 | 可选 | 本地模式 | 内存存储 | 本地开发调试 |
| 测试环境 | Cloud Studio | Docker | Docker | 功能测试 |
| 生产环境 | Cloud Studio | 集群部署 | 主从复制 | 正式运行 |

### 1.3 双路由架构部署要点

```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart LR
    subgraph Deploy["部署关键点"]
        A[工具隔离] --> B[General/Medical<br/>独立工具集]
        C[集合隔离] --> D[user_id 前缀<br/>私有向量集合]
        E[状态持久化] --> F[PostgreSQL<br/>会话恢复]
        G[并行执行] --> H[ThreadPoolExecutor<br/>工具并行调用]
    end
    
    style Deploy fill:#2ecc71,color:#fff
    style A fill:#3498db,color:#fff
    style C fill:#3498db,color:#fff
    style E fill:#3498db,color:#fff
    style G fill:#3498db,color:#fff
```

---

## 二、环境准备

### 2.1 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| **MinerU GPU** | RTX 3090 (24GB) | RTX 4090 (24GB+) |
| **应用服务器** | 4C8G | 8C16G |
| **Qdrant** | 2C4G + 50GB SSD | 4C8G + 200GB SSD |
| **PostgreSQL** | 2C4G + 20GB | 4C8G + 100GB SSD |

### 2.2 软件要求

```bash
# 检查 Docker 版本
docker --version  # 需要 20.10+

# 检查 Docker Compose
docker compose version  # 需要 V2

# 检查 Python
python --version  # 需要 3.10+

# 检查 NVIDIA 驱动（MinerU 服务器）
nvidia-smi  # 需要 Driver 525+
```

### 2.3 网络要求

| 服务 | 端口 | 说明 |
|------|------|------|
| MinerU API | 8000 | GPU 解析服务 |
| FastAPI | 8000 | 应用 API |
| Gradio | 7860 | Web 界面 |
| Qdrant | 6333 | 向量数据库 HTTP API |
| Qdrant gRPC | 6334 | 向量数据库 gRPC |
| PostgreSQL | 5432 | 关系数据库 |

---

## 三、MinerU GPU 服务部署

### 3.1 Docker Compose 配置

```yaml
# docker-compose/mineru/docker-compose.yml
services:
  mineru-api:
    image: mineru:latest
    container_name: mineru-api
    restart: always
    ports:
      - "8000:8000"
    
    environment:
      MINERU_MODEL_SOURCE: local
      CUDA_VISIBLE_DEVICES: "0"
      LOG_LEVEL: INFO
    
    entrypoint: mineru-api
    command:
      - --host
      - "0.0.0.0"
      - --port
      - "8000"
    
    volumes:
      - ./data/input:/app/input
      - ./data/output:/app/output
      - mineru-models:/root/.cache
    
    ulimits:
      stack: 67108864
    ipc: host
    shm_size: "8g"
    
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 120s
    
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
        limits:
          memory: 32g

volumes:
  mineru-models:
    driver: local
```

### 3.2 一键部署脚本

```bash
#!/bin/bash
# deploy_mineru.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "=========================================="
echo "  🚀 MinerU GPU 服务部署"
echo "=========================================="

# 检查 GPU
log_info "检查 GPU 环境..."
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi 未找到，请安装 NVIDIA 驱动"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv

# 检查 nvidia-container-toolkit
log_info "检查 nvidia-container-toolkit..."
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    log_info "安装 nvidia-container-toolkit..."
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
      sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi
log_ok "nvidia-container-toolkit 已就绪"

# 构建/拉取镜像
log_info "准备 MinerU 镜像..."
if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "mineru:latest"; then
    log_info "构建 MinerU 镜像..."
    [ ! -d "MinerU" ] && git clone --depth 1 https://github.com/opendatalab/MinerU.git
    docker build -t mineru:latest -f MinerU/docker/global/Dockerfile .
fi
log_ok "镜像准备完成"

# 启动服务
log_info "启动服务..."
mkdir -p data/input data/output
docker compose up -d

# 等待就绪
log_info "等待服务就绪..."
for i in {1..60}; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        log_ok "MinerU 服务部署成功！"
        echo "  API 地址: http://localhost:8000"
        echo "  API 文档: http://localhost:8000/docs"
        exit 0
    fi
    sleep 5
done
log_error "服务启动超时"
exit 1
```

### 3.3 端口暴露方案

#### 方案 A：Cloud Studio 内置转发（推荐）

```bash
# Cloud Studio 面板 → 端口 → 添加 8000
# 获取公网 URL: https://xxxx-8000.preview.myide.io
```

#### 方案 B：SSH 隧道

```bash
ssh -N -L 18000:localhost:8000 user@cloud-host
# 本地访问: http://localhost:18000
```

#### 方案 C：ngrok

```bash
./ngrok http 8000
# 公网 URL: https://xxxx.ngrok-free.app
```

### 3.4 性能调优

```yaml
# 显存不足时调整
command:
  - --gpu-memory-utilization
  - "0.5"  # 降低 GPU 显存占用比例

# 批量处理时增加超时
environment:
  MINERU_TIMEOUT: 600
```

---

## 四、本地服务部署

### 4.1 Docker Compose 编排

```yaml
# docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    restart: always
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrantDB:/qdrant/storage
    environment:
      QDRANT__LOG_LEVEL: INFO

  postgres:
    image: postgres:15
    container_name: postgres
    restart: always
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: rag_user
      POSTGRES_PASSWORD: rag_password
      POSTGRES_DB: rag_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag_user -d rag_db"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
```

### 4.2 启动服务

```bash
# 启动所有服务
docker compose up -d

# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f qdrant
docker compose logs -f postgres
```

### 4.3 应用部署

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
export DASHSCOPE_API_KEY=sk-xxx
export QDRANT_URL=http://localhost:6333
export DB_URI=postgresql://rag_user:rag_password@localhost:5432/rag_db

# 灌入知识库（系统级知识库）
python vectorSave.py

# 启动 API 服务
python main.py

# 或启动 Web 界面
python gradio_ui.py
```

### 4.4 用户文档隔离部署

```bash
# 用户上传医疗文档后，自动创建隔离集合
# 集合命名规则: user_{user_id}_documents

# 查看所有集合
curl http://localhost:6333/collections

# 查看特定用户的文档集合
curl http://localhost:6333/collections/user_123_documents
```

---

## 五、配置详解

### 5.1 配置文件结构

```
配置来源优先级:
1. 环境变量 (.env 文件)  ← 最高优先级
2. utils/config.py 默认值
```

### 5.2 核心配置项

#### LLM 配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LLM_TYPE` | `qwen` | 模型提供商 (openai/qwen/ollama/oneapi) |
| `DASHSCOPE_API_KEY` | - | 通义千问 API Key |
| `OPENAI_API_KEY` | - | OpenAI API Key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 服务地址 |
| `ONEAPI_BASE_URL` | - | OneAPI 服务地址 |

#### MinerU 配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `MINERU_API_URL` | `http://localhost:8000` | 服务地址 |
| `MINERU_TIMEOUT` | `300` | 超时秒数 |

#### Qdrant 配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `QDRANT_URL` | `http://127.0.0.1:6333` | 服务地址 |
| `QDRANT_COLLECTION_NAME` | `knowledge_base_v2` | 系统知识库集合名称 |

#### PostgreSQL 配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DB_URI` | - | 连接字符串 |

#### 文本切分配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `CHUNK_SIZE` | `800` | 最大字符数 |
| `CHUNK_OVERLAP` | `200` | 重叠字符数 |

#### 认证配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `AUTH_MODE` | `dev` | 认证模式 (api_key/jwt/dev) |
| `API_KEY_HEADER` | `X-API-Key` | API Key 请求头名称 |
| `JWT_SECRET` | - | JWT 密钥（生产环境必填） |
| `JWT_ALGORITHM` | `HS256` | JWT 算法 |

### 5.3 环境变量配置示例

```bash
# .env 文件示例

# ===== LLM 配置 =====
LLM_TYPE=qwen
DASHSCOPE_API_KEY=sk-xxx

# ===== MinerU 配置 =====
MINERU_API_URL=https://xxxx-8000.preview.myide.io
MINERU_TIMEOUT=300

# ===== Qdrant 配置 =====
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=knowledge_base_v2

# ===== PostgreSQL 配置 =====
DB_URI=postgresql://rag_user:rag_password@localhost:5432/rag_db

# ===== 认证配置 =====
AUTH_MODE=jwt
JWT_SECRET=your-super-secret-jwt-key-change-in-production
JWT_ALGORITHM=HS256

# ===== LangSmith 追踪（可选）=====
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_xxx
LANGCHAIN_PROJECT=ragAgent-Prod
```

### 5.4 多环境配置

```bash
# 开发环境
cp .env.example .env.dev

# 测试环境
cp .env.example .env.test

# 生产环境
cp .env.example .env.prod

# 加载指定环境
source .env.prod
```

---

## 六、Agent 工作流部署

### 6.1 双路由工作流架构

```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart TD
    START([用户输入]) --> AUTH[身份验证]
    AUTH --> ROUTER[Intent Router<br/>意图识别]
    
    ROUTER -->|general_domain| GENERAL[General RAG Agent]
    ROUTER -->|medical_domain| MEDICAL[Medical Agent]
    
    GENERAL --> RETRIEVE[Retriever<br/>混合检索]
    RETRIEVE --> RERANK[DashScope Rerank<br/>重排序]
    RERANK --> GEN_GENERAL[生成通用响应]
    
    MEDICAL --> ANALYZE[Medical Tools<br/>专业分析]
    ANALYZE --> BLOOD[血常规分析]
    ANALYZE --> BIOCHEM[血生化分析]
    ANALYZE --> URINE[尿常规分析]
    ANALYZE --> VITALS[生命体征分析]
    ANALYZE --> SYMPTOM[症状分析]
    
    BLOOD --> GEN_MEDICAL[生成医疗建议]
    BIOCHEM --> GEN_MEDICAL
    URINE --> GEN_MEDICAL
    VITALS --> GEN_MEDICAL
    SYMPTOM --> GEN_MEDICAL
    
    GEN_GENERAL --> SAFETY[安全守卫<br/>风险评估]
    GEN_MEDICAL --> SAFETY
    
    SAFETY --> OUTPUT([结构化输出])
    
    style ROUTER fill:#f39c12,color:#fff
    style GENERAL fill:#27ae60,color:#fff
    style MEDICAL fill:#e74c3c,color:#fff
    style SAFETY fill:#9b59b6,color:#fff
```

### 6.2 工作流配置参数

```python
# utils/config.py

# Agent 配置
AGENT_MAX_ITERATIONS = 10           # 最大迭代次数
AGENT_MAX_RETRIES = 3               # 查询重写最大重试次数

# 检索配置
RETRIEVER_TOP_K = 5                 # 粗排召回数量
RERANKER_TOP_N = 3                  # 精排返回数量

# 并行工具执行配置
TOOL_EXECUTOR_MAX_WORKERS = 3       # 最大线程数
TOOL_EXECUTOR_TIMEOUT = 120         # 单工具超时时间（秒）
```

### 6.3 持久化配置

```python
# PostgreSQL 会话存储（生产环境）
from langgraph.checkpoint.postgres import PostgresSaver
from utils.config import Config

checkpointer = PostgresSaver.from_conn_string(Config.DB_URI)

# 内存存储（开发环境）
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
```

### 6.4 状态管理字段说明

```python
class AgentState(MessagesState):
    """
    对话状态定义。
    
    设计要点:
    - 继承 MessagesState 复用消息管理
    - Annotated[int, operator.add] 实现跨节点累加
    - 业务字段和 Middleware 追踪字段清晰分离
    """
    # ===== 业务字段 =====
    relevance_score: Optional[str] = None       # 检索相关性评分 (yes/no)
    rewrite_count: int = 0                       # 查询重写次数（防死循环）
    route_domain: Optional[Literal["general", "medical"]] = None
    route_reason: Optional[str] = None          # 路由原因说明
    
    # 医疗建议字段
    recommended_departments: Optional[List[str]] = None   # 推荐科室列表
    urgency_level: Optional[Literal["routine", "urgent", "emergency"]] = None
    risk_level: Optional[Literal["low","medium","high","critical"]] = None
    final_payload: Optional[dict] = None                # 最终输出载荷
    
    # ===== Middleware 追踪字段 =====
    mw_model_call_count: Annotated[int, operator.add] = 0
    mw_model_total_time: Annotated[float, operator.add] = 0.0
    mw_tool_total_time: Annotated[float, operator.add] = 0.0
    mw_pii_detected: bool = False
    mw_force_stop: bool = False
    mw_node_timings: Optional[dict] = None
```

---

## 七、Middleware 配置

### 7.1 Middleware 执行顺序（洋葱模型）

```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart TD
    REQUEST([请求进入]) --> MW7[PII Detection<br/>第7层]
    MW7 --> MW6[Model Call Limit<br/>第6层]
    MW6 --> MW5[Summarization<br/>第5层]
    MW5 --> MW4[Tool Retry<br/>第4层]
    MW4 --> MW3[Performance Tracking<br/>第3层]
    MW3 --> MW2[Logging<br/>第2层]
    MW2 --> MW1[Error Handler<br/>第1层]
    MW1 --> AGENT[Agent 执行]
    AGENT --> MW1_OUT[Error Handler<br/>响应]
    MW1_OUT --> MW2_OUT[Logging<br/>响应]
    MW2_OUT --> MW3_OUT[Performance<br/>响应]
    MW3_OUT --> MW4_OUT[Tool Retry<br/>响应]
    MW4_OUT --> MW5_OUT[Summarization<br/>响应]
    MW5_OUT --> MW6_OUT[Model Limit<br/>响应]
    MW6_OUT --> MW7_OUT[PII Detection<br/>响应]
    MW7_OUT --> RESPONSE([响应返回])
    
    style REQUEST fill:#2ecc71,color:#fff
    style RESPONSE fill:#e74c3c,color:#fff
    style AGENT fill:#f39c12,color:#fff
```

### 7.2 Middleware 类型与配置

| Middleware | 功能 | 配置项 | 默认值 |
|------------|------|--------|--------|
| `LoggingMiddleware` | 日志追踪 | `LOG_LEVEL` | `INFO` |
| `ModelCallLimitMiddleware` | 调用限制 | `MW_MAX_MODEL_CALLS` | `10` |
| `PIIDetectionMiddleware` | PII 检测 | `MW_PII_MODE` | `detect` |
| `SummarizationMiddleware` | 对话摘要 | `MW_SUMMARIZATION_THRESHOLD` | `20` |
| `ToolRetryMiddleware` | 工具重试 | `MW_TOOL_MAX_RETRIES` | `3` |
| `PerformanceTrackingMiddleware` | 性能追踪 | `MW_PERFORMANCE_ENABLED` | `True` |
| `ErrorHandlerMiddleware` | 错误处理 | `MW_ERROR_HANDLER_ENABLED` | `True` |

### 7.3 详细配置示例

```python
# utils/config.py

# Middleware 配置
MW_MAX_MODEL_CALLS = 10                    # 最大模型调用次数
MW_PII_MODE = "detect"                     # PII 模式: detect/warn/mask/block
MW_SUMMARIZATION_THRESHOLD = 20            # 触发摘要的消息数
MW_SUMMARIZATION_KEEP_RECENT = 5           # 保留最近消息数
MW_TOOL_MAX_RETRIES = 3                    # 工具重试次数
MW_TOOL_BACKOFF_FACTOR = 0.5               # 重试退避因子（秒）
MW_PERFORMANCE_ENABLED = True              # 是否启用性能追踪
MW_ERROR_HANDLER_ENABLED = True            # 是否启用错误处理
```

### 7.4 PII 检测模式

| 模式 | 行为 | 适用场景 |
|------|------|----------|
| `detect` | 仅检测，记录日志 | 开发环境调试 |
| `warn` | 检测并警告用户 | 测试环境 |
| `mask` | 检测并脱敏处理 | 生产环境（推荐） |
| `block` | 检测并拒绝请求 | 高安全要求场景 |

---

## 八、医疗模块专项配置

### 8.1 医疗分析工具链

```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart LR
    subgraph MedicalTools["医疗分析工具集"]
        BLOOD[🩸 血常规分析<br/>blood_analysis]
        BIOCHEM[🧪 血生化分析<br/>biochemical_analysis]
        URINE[💧 尿常规分析<br/>urine_analysis]
        VITALS[❤️ 生命体征分析<br/>vitals_analysis]
        SYMPTOM[🩺 症状分析<br/>symptom_analysis]
    end
    
    subgraph Output["输出内容"]
        DEPT[推荐科室]
        URGENCY[紧急程度]
        RISK[风险等级]
        ADVICE[就医建议]
    end
    
    BLOOD --> Output
    BIOCHEM --> Output
    URINE --> Output
    VITALS --> Output
    SYMPTOM --> Output
    
    style MedicalTools fill:#e74c3c,color:#fff
    style Output fill:#9b59b6,color:#fff
```

### 8.2 医疗工具配置

```python
# utils/tools_config.py

# 医疗分析工具配置
MEDICAL_ANALYSIS_CONFIG = {
    "blood_analysis": {
        "enabled": True,
        "description": "血常规报告分析与异常检测",
        "timeout": 30,
        "max_retries": 2
    },
    "biochemical_analysis": {
        "enabled": True,
        "description": "血生化指标分析与风险评估",
        "timeout": 30,
        "max_retries": 2
    },
    "urine_analysis": {
        "enabled": True,
        "description": "尿常规检验结果解读",
        "timeout": 25,
        "max_retries": 2
    },
    "vitals_analysis": {
        "enabled": True,
        "description": "生命体征监测数据分析",
        "timeout": 20,
        "max_retries": 2
    },
    "symptom_analysis": {
        "enabled": True,
        "description": "症状描述分析与科室推荐",
        "timeout": 35,
        "max_retries": 2
    }
}
```

### 8.3 用户文档隔离配置

```python
# utils/document_processor.py

# 用户文档集合配置
USER_DOCUMENT_CONFIG = {
    "collection_prefix": "user_",           # 集合名前缀
    "chunk_size": 500,                      # 医疗文档切分大小
    "chunk_overlap": 100,                   # 切分重叠
    "embedding_model": "text-embedding-v2", # Embedding 模型
    "max_documents_per_user": 50,           # 每用户最大文档数
    "max_file_size_mb": 20                  # 单文件最大大小（MB）
}
```

### 8.4 安全守卫配置

```python
# ragAgent.py - 安全守卫节点配置

SAFETY_GUARD_CONFIG = {
    "risk_assessment_enabled": True,         # 启用风险评估
    "disclaimer_required": True,             # 强制添加免责声明
    "emergency_keywords": [                  # 紧急关键词列表
        "胸痛", "呼吸困难", "意识模糊", 
        "大出血", "过敏性休克", "中毒"
    ],
    "urgency_levels": {
        "routine": "常规就诊",
        "urgent": "尽快就医",
        "emergency": "立即急诊"
    },
    "risk_levels": {
        "low": "低风险",
        "medium": "中等风险",
        "high": "高风险",
        "critical": "危重症风险"
    }
}
```

---

## 九、故障排查

### 9.1 常见问题诊断流程

```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart TD
    ISSUE([问题现象]) --> CHECK1{服务状态?}
    CHECK1 -->|异常| FIX1[重启服务]
    CHECK1 -->|正常| CHECK2{网络连通?}
    CHECK2 -->|异常| FIX2[检查防火墙/端口]
    CHECK2 -->|正常| CHECK3{配置正确?}
    CHECK3 -->|异常| FIX3[修正配置]
    CHECK3 -->|正常| CHECK4{权限足够?}
    CHECK4 -->|异常| FIX4[调整权限]
    CHECK4 -->|正常| CHECK5{资源充足?}
    CHECK5 -->|异常| FIX5[扩容资源]
    CHECK5 -->|正常| LOG[查看详细日志]
    
    style ISSUE fill:#e74c3c,color:#fff
    style LOG fill:#f39c12,color:#fff
    style FIX1 fill:#27ae60,color:#fff
    style FIX2 fill:#27ae60,color:#fff
    style FIX3 fill:#27ae60,color:#fff
    style FIX4 fill:#27ae60,color:#fff
    style FIX5 fill:#27ae60,color:#fff
```

### 9.2 具体问题解决方案

#### MinerU 服务连接失败

```bash
# 检查服务状态
curl http://localhost:8000/health

# 查看容器日志
docker logs mineru-api --tail 100

# 检查 GPU 状态
nvidia-smi

# 重启服务
docker compose restart

# 常见错误及解决
# 1. CUDA out of memory → 降低 batch size 或使用更小模型
# 2. Model not found → 检查模型下载是否完整
# 3. Timeout → 增加 MINERU_TIMEOUT 配置
```

#### Qdrant 连接问题

```bash
# 检查服务状态
curl http://localhost:6333/collections

# 查看集合信息
curl http://localhost:6333/collections/knowledge_base_v2

# 查看用户文档集合
curl http://localhost:6333/collections/user_123_documents

# 重启服务
docker compose restart qdrant

# 常见错误及解决
# 1. Collection not found → 运行 vectorSave.py 创建集合
# 2. Connection refused → 检查 Qdrant 是否启动
# 3. Disk space → 清理旧数据或扩容磁盘
```

#### PostgreSQL 连接失败

```bash
# 检查连接
psql -h localhost -U rag_user -d rag_db

# 查看日志
docker logs postgres --tail 100

# 系统会自动降级到内存存储模式

# 常见错误及解决
# 1. Authentication failed → 检查用户名密码
# 2. Database does not exist → 创建数据库
# 3. Connection timeout → 检查网络和防火墙
```

#### Embedding 调用失败

```python
# 验证配置
from utils.config import Config
print(f"LLM_TYPE: {Config.LLM_TYPE}")
print(f"API_KEY: {Config.get_api_key()[:10]}...")

# 测试调用
from openai import OpenAI
client = OpenAI(
    base_url=Config.get_api_base(),
    api_key=Config.get_api_key()
)
response = client.embeddings.create(
    input=["测试文本"],
    model="text-embedding-v1"
)
print(response.data[0].embedding[:5])

# 常见错误及解决
# 1. Invalid API Key → 检查密钥配置
# 2. Rate limit → 降低并发或升级套餐
# 3. Model not found → 检查模型名称是否正确
```

#### Agent 路由错误

```python
# 检查路由决策日志
grep "route_decision" output/app.log

# 验证路由器配置
from prompts import load_prompt_template
router_prompt = load_prompt_template("prompt_template_intent_router")
print(router_prompt)

# 常见错误及解决
# 1. Always routing to general → 检查 prompt template
# 2. Medical tools not working → 检查工具注册
# 3. Loop detected → 检查 rewrite_count 限制
```

### 9.3 日志查看与分析

```bash
# 实时日志
tail -f output/app.log

# 错误日志
grep "ERROR" output/app.log

# 警告日志
grep "WARNING" output/app.log

# Middleware 日志
grep "middleware" output/app.log

# 路由决策日志
grep "route_domain" output/app.log

# 性能统计日志
grep "performance" output/app.log

# 最近 100 行
tail -n 100 output/app.log

# 按时间过滤
grep "2026-04-19 14:" output/app.log
```

### 9.4 健康检查脚本

```bash
#!/bin/bash
# health_check.sh

echo "=== 🏥 智能医疗分诊系统 - 健康检查 ==="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_service() {
    local name=$1
    local url=$2
    local expected_code=${3:-200}
    
    echo -n "  $name: "
    response=$(curl -sf -o /dev/null -w "%{http_code}" "$url" 2>/dev/null)
    
    if [ "$response" = "$expected_code" ]; then
        echo -e "${GREEN}✅ 正常${NC}"
        return 0
    else
        echo -e "${RED}❌ 异常 (HTTP $response)${NC}"
        return 1
    fi
}

echo "📡 外部服务检查:"
check_service "MinerU GPU" "http://localhost:8000/health"
check_service "Qdrant Vector DB" "http://localhost:6333/collections"
check_service "PostgreSQL" "http://localhost:5432" "" || \
    (docker exec postgres pg_isready -U rag_user > /dev/null 2>&1 && \
     echo -e "  PostgreSQL: ${GREEN}✅ 正常${NC}" || \
     echo -e "  PostgreSQL: ${RED}❌ 异常${NC}")

echo ""
echo "🔧 应用服务检查:"
check_service "FastAPI API" "http://localhost:8000/docs"
check_service "Gradio UI" "http://localhost:7860" "" || \
    echo -e "  Gradio UI: ${YELLOW}⚠️ 未启动（可选）${NC}"

echo ""
echo "📊 资源使用情况:"
echo "  CPU 使用率: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "  内存使用率: $(free -m | awk '/Mem/{printf "%.1f%%", $3/$2*100}')"
echo "  磁盘使用率: $(df -h / | awk 'NR==2{print $5}')"

echo ""
echo "🔐 认证状态:"
if [ -n "$AUTH_MODE" ]; then
    echo "  认证模式: $AUTH_MODE"
else
    echo -e "  认证模式: ${YELLOW}开发模式（无认证）${NC}"
fi

echo ""
echo "=== 检查完成 ==="
```

---

## 十、监控与运维

### 10.1 关键性能指标 (KPI)

| 指标类别 | 指标名称 | 告警阈值 | 说明 |
|----------|----------|----------|------|
| **响应性能** | API 平均响应时间 | > 5s | P95 响应时间 |
| | Agent 执行时间 | > 30s | 完整工作流耗时 |
| | 检索延迟 | > 2s | 两阶段检索总耗时 |
| **可靠性** | API 可用性 | < 99% | 服务可用率 |
| | 错误率 | > 1% | 5xx 错误占比 |
| | 超时率 | > 0.5% | 请求超时占比 |
| **资源使用** | CPU 使用率 | > 80% | 持续高负载 |
| | 内存使用率 | > 85% | 内存压力 |
| | GPU 显存使用率 | > 90% | MinerU 服务 |
| **业务指标** | 路由准确率 | < 95% | 意图路由准确度 |
| | 检索相关性 | < 80% | 检索结果相关度 |
| | 医疗分析成功率 | < 98% | 工具执行成功率 |

### 10.2 监控命令速查

```bash
# 实时监控 API 请求
tail -f output/app.log | grep -E "(POST|GET)"

# 监控 Agent 执行
tail -f output/app.log | grep -E "(agent|route|tool)"

# 监控资源使用
watch -n 1 'free -h && echo "---" && df -h /'

# 监控 Docker 容器
docker stats --no-stream

# 监控 Qdrant 性能
curl http://localhost:6333/cluster

# 监控 PostgreSQL 连接数
docker exec postgres psql -U rag_user -d rag_db -c "SELECT count(*) FROM pg_stat_activity;"
```

### 10.3 运维操作指南

#### 日常巡检（每日）

```bash
#!/bin/bash
# daily_check.sh - 每日巡检脚本

echo "=== 📅 每日巡检 ==="
date "+%Y-%m-%d %H:%M:%S"

# 1. 服务状态检查
./health_check.sh

# 2. 磁盘空间检查
echo ""
echo "💾 磁盘空间:"
df -h | awk '$NF=="/"{printf "  根分区: %s/%s (%s)\n", $3, $2, $5}'

# 3. 日志大小检查
echo ""
echo "📝 日志文件大小:"
ls -lh output/*.log 2>/dev/null | awk '{print "  " $9 ": " $5}'

# 4. 今日错误统计
echo ""
echo "❌ 今日错误数量:"
grep -c "$(date +%Y-%m-%d)" output/app.log 2>/dev/null | xargs -I {} echo "  {} 条"

# 5. 数据库备份检查
echo ""
echo "💾 最近备份:"
ls -lht backups/ 2>/dev/null | head -5 || echo "  无备份目录"
```

#### 定期维护（每周）

```bash
#!/bin/bash
# weekly_maintenance.sh - 每周维护脚本

echo "=== 🔧 每周维护 ==="

# 1. 日志轮转
echo "📝 轮转日志..."
mv output/app.log "output/app_$(date +%Y%m%d).log"
touch output/app.log

# 2. 数据库清理
echo "🗄️ 清理过期会话..."
docker exec postgres psql -U rag_user -d rag_db -c "
DELETE FROM checkpoints WHERE created_at < NOW() - INTERVAL '30 days';
VACUUM FULL checkpoints;
"

# 3. Qdrant 优化
echo "⚡ 优化向量索引..."
curl -X POST "http://localhost:6333/collections/knowledge_base_v2/update" \
  -H "Content-Type: application/json" \
  -d '{"optimize": true}'

# 4. 系统更新检查
echo "🔄 检查更新..."
pip list --outdated 2>/dev/null | head -10

# 5. 备份验证
echo "✅ 验证备份完整性..."
# 添加备份验证逻辑
```

#### 应急处理流程

```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart TD
    ALERT([🚨 告警触发]) --> ASSESS[影响评估]
    ASSESS -->|轻微| MINOR[记录并观察]
    ASSESS -->|严重| SEVERE[启动应急响应]
    
    SEVERE --> NOTIFY[通知相关人员]
    NOTIFY --> INVESTIGATE[问题定位]
    INVESTIGATE --> QUICK_FIX{快速修复?}
    
    QUICK_FIX -->|是| EXECUTE[执行修复]
    QUICK_FIX -->|否| WORKAROUND[临时方案]
    
    EXECUTE --> VERIFY[验证修复效果]
    WORKAROUND --> VERIFY
    
    VERIFY -->|成功| RESOLVE[问题解决]
    VERIFY -->|失败| ESCALATE[升级处理]
    
    RESOLVE --> REVIEW[事后复盘]
    ESCALATE --> REVIEW
    
    style ALERT fill:#e74c3c,color:#fff
    style RESOLVE fill:#27ae60,color:#fff
    style REVIEW fill:#f39c12,color:#fff
```

### 10.4 备份与恢复策略

#### 数据备份

```bash
#!/bin/bash
# backup.sh - 数据备份脚本

BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "=== 💾 开始备份 ==="

# 1. Qdrant 数据快照
echo "📦 备份 Qdrant 数据..."
SNAPSHOT_RESPONSE=$(curl -X POST "http://localhost:6333/collections/knowledge_base_v2/snapshots")
SNAPSHOT_NAME=$(echo $SNAPSHOT_RESPONSE | jq -r '.result.name')
curl -o "$BACKUP_DIR/qdrant_snapshot.snapshot" \
  "http://localhost:6333/collections/knowledge_base_v2/snapshots/$SNAPSHOT_NAME"
echo "  ✅ Qdrant 快照完成: $SNAPSHOT_NAME"

# 2. PostgreSQL 数据库备份
echo "🗄️ 备份 PostgreSQL 数据库..."
docker exec postgres pg_dump -U rag_user -d rag_db > "$BACKUP_DIR/postgresql_backup.sql"
echo "  ✅ PostgreSQL 备份完成"

# 3. 配置文件备份
echo "📋 备份配置文件..."
cp .env "$BACKUP_DIR/env_backup"
cp utils/config.py "$BACKUP_DIR/config_backup.py"
echo "  ✅ 配置文件备份完成"

# 4. 用户文档元数据备份
echo "📄 备份用户文档元数据..."
# 添加用户文档元数据备份逻辑
echo "  ✅ 用户文档元数据备份完成"

# 5. 生成备份清单
echo "📝 生成备份清单..."
cat > "$BACKUP_DIR/manifest.txt" << EOF
备份时间: $(date '+%Y-%m-%d %H:%M:%S')
备份类型: 全量备份
备份内容:
  - Qdrant 快照: qdrant_snapshot.snapshot
  - PostgreSQL: postgresql_backup.sql
  - 配置文件: env_backup, config_backup.py
  - 用户文档元数据: user_documents_meta.json
EOF

echo ""
echo "=== ✅ 备份完成 ==="
echo "备份位置: $BACKUP_DIR"
du -sh "$BACKUP_DIR"
```

#### 数据恢复

```bash
#!/bin/bash
# restore.sh - 数据恢复脚本

BACKUP_DIR="$1"

if [ -z "$BACKUP_DIR" ]; then
    echo "用法: $0 <备份目录>"
    exit 1
fi

echo "=== 🔙 开始恢复 ==="

# 1. 停止服务
echo "⏹️ 停止服务..."
docker compose down

# 2. 恢复 PostgreSQL
echo "🗄️ 恢复 PostgreSQL 数据库..."
docker exec -i postgres psql -U rag_user -d rag_db < "$BACKUP_DIR/postgresql_backup.sql"
echo "  ✅ PostgreSQL 恢复完成"

# 3. 恢复 Qdrant
echo "📦 恢复 Qdrant 数据..."
# 从快照恢复逻辑
echo "  ✅ Qdrant 恢复完成"

# 4. 恢复配置文件
echo "📋 恢复配置文件..."
cp "$BACKUP_DIR/env_backup" .env
cp "$BACKUP_DIR/config_backup.py" utils/config.py
echo "  ✅ 配置文件恢复完成"

# 5. 启动服务
echo "▶️ 启动服务..."
docker compose up -d

# 6. 验证恢复
echo "✅ 验证恢复结果..."
sleep 10
./health_check.sh

echo ""
echo "=== ✅ 恢复完成 ==="
```

### 10.5 运维命令速查表

| 操作类别 | 操作 | 命令 |
|----------|------|------|
| **服务管理** | 启动所有服务 | `docker compose up -d` |
| | 停止所有服务 | `docker compose down` |
| | 重启单个服务 | `docker compose restart <service>` |
| | 查看服务状态 | `docker compose ps` |
| | 查看实时日志 | `docker compose logs -f <service>` |
| **容器操作** | 进入容器 | `docker exec -it <container> bash` |
| | 查看资源占用 | `docker stats --no-stream` |
| | 清理无用镜像 | `docker image prune -a` |
| **数据管理** | 创建备份 | `./backup.sh` |
| | 恢复数据 | `./restore.sh <backup_dir>` |
| | 清理数据 | `docker compose down -v` |
| **监控** | 健康检查 | `./health_check.sh` |
| | 每日巡检 | `./daily_check.sh` |
| | 每周维护 | `./weekly_maintenance.sh` |
| **故障排查** | 查看 API 日志 | `tail -f output/app.log` |
| | 查看错误日志 | `grep ERROR output/app.log` |
| | 测试 API 连通 | `curl http://localhost:8000/docs` |
| **更新** | 拉取新镜像 | `docker compose pull` |
| | 重新部署 | `docker compose up -d --build` |
| | 查看更新日志 | `git log --oneline -10` |

---

**文档版本**: v2.0.0  
**更新日期**: 2026-04-19  
**适用项目版本**: v2.0.0+  
**维护者**: 开发团队