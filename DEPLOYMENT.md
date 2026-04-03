# 部署指南

本文档面向运维人员，提供完整的系统部署、配置和运维说明。

## 目录

- [一、部署架构](#一部署架构)
- [二、环境准备](#二环境准备)
- [三、MinerU GPU 服务部署](#三mineru-gpu-服务部署)
- [四、本地服务部署](#四本地服务部署)
- [五、配置详解](#五配置详解)
- [六、Agent 工作流部署](#六agent-工作流部署)
- [七、Middleware 配置](#七middleware-配置)
- [八、故障排查](#八故障排查)

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
        
        subgraph Agent["Agent 层"]
            RAG[RAG Agent<br/>LangGraph]
            MW[Middleware<br/>安全防护]
        end
    end
    
    subgraph External["外部服务"]
        LLM[LLM API<br/>OpenAI/Qwen/Ollama]
    end
    
    M -->|Markdown| API
    API --> RAG
    RAG --> MW
    MW --> Q
    MW --> PG
    RAG --> LLM
    
    style Cloud fill:#9b59b6,color:#fff
    style Local fill:#3498db,color:#fff
    style External fill:#e67e22,color:#fff
```

### 1.2 部署环境对照表

| 环境 | MinerU | Qdrant | PostgreSQL | 适用场景 |
|------|--------|--------|------------|----------|
| 开发环境 | 可选 | 本地模式 | 内存存储 | 本地开发调试 |
| 测试环境 | Cloud Studio | Docker | Docker | 功能测试 |
| 生产环境 | Cloud Studio | 集群部署 | 主从复制 | 正式运行 |

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
| Qdrant | 6333 | 向量数据库 |
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

# 颜色定义
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

# 灌入知识库
python vectorSave.py

# 启动 API 服务
python main_v1.py

# 或启动 Web 界面
python webUI.py
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

| 类别 | 配置项 | 默认值 | 说明 |
|------|--------|--------|------|
| **LLM** | `LLM_TYPE` | `qwen` | 模型提供商 |
| | `DASHSCOPE_API_KEY` | - | 通义千问 API Key |
| | `OPENAI_API_KEY` | - | OpenAI API Key |
| **MinerU** | `MINERU_API_URL` | `http://localhost:8000` | 服务地址 |
| | `MINERU_TIMEOUT` | `300` | 超时秒数 |
| **Qdrant** | `QDRANT_URL` | `http://127.0.0.1:6333` | 服务地址 |
| | `QDRANT_COLLECTION_NAME` | `knowledge_base_v2` | 集合名称 |
| **PostgreSQL** | `DB_URI` | - | 连接字符串 |
| **切分** | `CHUNK_SIZE` | `800` | 最大字符数 |
| | `CHUNK_OVERLAP` | `200` | 重叠字符数 |

### 5.3 环境变量配置

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

### 6.1 工作流架构

```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart TD
    START([开始]) --> AGENT[Agent 节点]
    AGENT -->|tools_condition| TOOLS{调用工具?}
    TOOLS -->|是| CALL[执行工具]
    TOOLS -->|否| END([结束])
    CALL --> ROUTE{工具类型}
    ROUTE -->|检索| GRADE[文档评分]
    ROUTE -->|其他| GEN[生成响应]
    GRADE -->|yes| GEN
    GRADE -->|no| REWRITE[重写查询]
    REWRITE --> AGENT
    GEN --> END
```

### 6.2 工作流配置

```python
# utils/config.py

# Agent 配置
AGENT_MAX_ITERATIONS = 10      # 最大迭代次数
AGENT_MAX_RETRIES = 3          # 查询重写最大重试次数

# 检索配置
RETRIEVER_TOP_K = 5            # 粗排召回数量
RERANKER_TOP_N = 3             # 精排返回数量
```

### 6.3 持久化配置

```python
# PostgreSQL 会话存储
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver(DB_URI)

# 内存存储（开发环境）
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
```

---

## 七、Middleware 配置

### 7.1 Middleware 类型

| Middleware | 功能 | 配置项 |
|------------|------|--------|
| `LoggingMiddleware` | 日志追踪 | `LOG_LEVEL` |
| `ModelCallLimitMiddleware` | 调用限制 | `MW_MAX_MODEL_CALLS` |
| `PIIDetectionMiddleware` | PII 检测 | `MW_PII_MODE` |
| `SummarizationMiddleware` | 对话摘要 | `MW_SUMMARIZATION_THRESHOLD` |
| `ToolRetryMiddleware` | 工具重试 | `MW_TOOL_MAX_RETRIES` |

### 7.2 配置示例

```python
# utils/config.py

# Middleware 配置
MW_MAX_MODEL_CALLS = 10           # 最大模型调用次数
MW_PII_MODE = "detect"            # PII 模式: detect/warn/mask/block
MW_SUMMARIZATION_THRESHOLD = 20   # 触发摘要的消息数
MW_SUMMARIZATION_KEEP_RECENT = 5  # 保留最近消息数
MW_TOOL_MAX_RETRIES = 3           # 工具重试次数
MW_TOOL_BACKOFF_FACTOR = 0.5      # 重试退避因子（秒）
```

### 7.3 PII 检测模式

| 模式 | 行为 |
|------|------|
| `detect` | 仅检测，记录日志 |
| `warn` | 检测并警告用户 |
| `mask` | 检测并脱敏处理 |
| `block` | 检测并拒绝请求 |

---

## 八、故障排查

### 8.1 常见问题

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
```

#### Qdrant 连接问题

```bash
# 检查服务状态
curl http://localhost:6333/collections

# 查看集合信息
curl http://localhost:6333/collections/knowledge_base_v2

# 重启服务
docker compose restart qdrant
```

#### PostgreSQL 连接失败

```bash
# 检查连接
psql -h localhost -U rag_user -d rag_db

# 查看日志
docker logs postgres --tail 100

# 系统会自动降级到内存存储模式
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
```

### 8.2 日志查看

```bash
# 实时日志
tail -f output/app.log

# 错误日志
grep "ERROR" output/app.log

# 最近 100 行
tail -n 100 output/app.log
```

### 8.3 健康检查脚本

```bash
#!/bin/bash
# health_check.sh

echo "=== 服务健康检查 ==="

# MinerU
echo -n "MinerU: "
curl -sf http://localhost:8000/health && echo "✅" || echo "❌"

# Qdrant
echo -n "Qdrant: "
curl -sf http://localhost:6333/collections && echo "✅" || echo "❌"

# PostgreSQL
echo -n "PostgreSQL: "
docker exec postgres pg_isready -U rag_user && echo "✅" || echo "❌"

# FastAPI
echo -n "FastAPI: "
curl -sf http://localhost:8000/health && echo "✅" || echo "❌"
```

---

## 九、运维命令速查

| 操作 | 命令 |
|------|------|
| 启动所有服务 | `docker compose up -d` |
| 停止所有服务 | `docker compose down` |
| 重启服务 | `docker compose restart <service>` |
| 查看日志 | `docker compose logs -f <service>` |
| 进入容器 | `docker exec -it <container> bash` |
| 清理数据 | `docker compose down -v` |
| 更新镜像 | `docker compose pull && docker compose up -d` |

---

**文档版本**: v1.0.0 | **更新日期**: 2026-04-03
