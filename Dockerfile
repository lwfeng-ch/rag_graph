# =========================
# 智能医疗分诊系统
# FastAPI + LangGraph + RAG
# Dockerfile - 适配优化版 requirements
# =========================

# ====== 构建阶段 ======
FROM python:3.11-bookworm AS builder

LABEL maintainer="dev-team"
LABEL description="Medical Triage System (RAG + LangGraph)"
LABEL version="2.1.0"

WORKDIR /app

# 使用国内 apt 源（稳定写法）
RUN printf "deb https://mirrors.aliyun.com/debian bookworm main\n\
deb https://mirrors.aliyun.com/debian bookworm-updates main\n\
deb https://mirrors.aliyun.com/debian-security bookworm-security main\n" > /etc/apt/sources.list

# 安装编译依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制 requirements 文件
COPY requirements.txt .

# pip 使用国内源（关键优化）
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --user -r requirements.txt


# ====== 运行阶段 ======
FROM python:3.11-bookworm AS production

WORKDIR /app

# 同样配置 apt 源（避免 runtime apt 卡死）
RUN printf "deb https://mirrors.aliyun.com/debian bookworm main\n\
deb https://mirrors.aliyun.com/debian bookworm-updates main\n\
deb https://mirrors.aliyun.com/debian-security bookworm-security main\n" > /etc/apt/sources.list

# 仅安装运行依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# 创建非 root 用户（安全）
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 拷贝 python 依赖
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# 拷贝项目代码
COPY --chown=appuser:appuser . .

# 创建运行目录
RUN mkdir -p logs output input qdrantDB user_medical_docs && \
    chown -R appuser:appuser logs output input qdrantDB user_medical_docs

USER appuser

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 端口
EXPOSE 8012

# 健康检查（避免误报建议接口存在）
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8012/v1/health || exit 1

# 启动 FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8012", "--workers", "1"]
