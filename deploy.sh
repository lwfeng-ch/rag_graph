#!/bin/bash

# ============================================
# 智能医疗分诊系统 - Docker 快速部署脚本
# ============================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查 Docker 是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose 未安装，请先安装 Docker Compose"
        exit 1
    fi
    
    log_success "Docker 和 Docker Compose 已安装"
}

# 检查 .env 文件
check_env() {
    if [ ! -f .env ]; then
        log_warning ".env 文件不存在，正在创建..."
        cp .env.example .env
        log_info "请编辑 .env 文件，配置 API Key 和数据库密码"
        read -p "按回车键继续..."
    else
        log_success ".env 文件已存在"
    fi
}

# 停止并清理旧容器
cleanup() {
    log_info "停止并清理旧容器..."
    docker-compose -f docker-compose.optimized.yml down 2>/dev/null || true
    log_success "清理完成"
}

# 构建镜像
build_image() {
    log_info "开始构建 Docker 镜像..."
    
    # 选择 requirements 文件
    echo "请选择 requirements 文件："
    echo "1) requirements-lite.txt (精简版 - 推荐生产环境)"
    echo "2) requirements.txt (完整版 - 推荐开发环境)"
    read -p "请输入选项 (1/2): " req_choice
    
    case $req_choice in
        1)
            REQ_FILE="requirements-lite.txt"
            log_info "使用精简版 requirements (requirements-lite.txt)"
            ;;
        2)
            REQ_FILE="requirements.txt"
            log_info "使用完整版 requirements (requirements.txt)"
            ;;
        *)
            log_error "无效选项，使用精简版"
            REQ_FILE="requirements-lite.txt"
            ;;
    esac
    
    # 构建镜像
    docker build \
        -f Dockerfile.optimized \
        --build-arg REQUIREMENTS_FILE=$REQ_FILE \
        -t medical-triage:$(date +%Y%m%d-%H%M%S) \
        -t medical-triage:latest \
        .
    
    log_success "镜像构建完成"
}

# 启动服务
start_services() {
    log_info "启动服务..."
    docker-compose -f docker-compose.optimized.yml up -d
    
    log_info "等待服务启动..."
    sleep 10
    
    # 检查服务状态
    docker-compose -f docker-compose.optimized.yml ps
}

# 健康检查
health_check() {
    log_info "执行健康检查..."
    
    # 等待 FastAPI 启动
    for i in {1..30}; do
        if curl -f http://localhost:8012/v1/health &> /dev/null; then
            log_success "FastAPI 服务健康检查通过"
            break
        fi
        
        if [ $i -eq 30 ]; then
            log_error "FastAPI 服务启动超时"
            exit 1
        fi
        
        echo -n "."
        sleep 2
    done
    echo ""
    
    # 检查 Qdrant
    if curl -f http://localhost:6333/ &> /dev/null; then
        log_success "Qdrant 服务健康检查通过"
    else
        log_warning "Qdrant 服务未响应，请稍后检查"
    fi
    
    # 检查 PostgreSQL
    if docker exec postgres pg_isready -U rag_user &> /dev/null; then
        log_success "PostgreSQL 服务健康检查通过"
    else
        log_warning "PostgreSQL 服务未响应，请稍后检查"
    fi
}

# 显示访问信息
show_info() {
    echo ""
    log_success "=========================================="
    log_success "    智能医疗分诊系统部署完成！"
    log_success "=========================================="
    echo ""
    echo "📌 服务访问地址："
    echo "   - FastAPI:      http://localhost:8012"
    echo "   - Gradio UI:    http://localhost:7860"
    echo "   - Qdrant:       http://localhost:6333"
    echo "   - PostgreSQL:   localhost:5432"
    echo ""
    echo "📌 常用命令："
    echo "   - 查看日志：    docker-compose -f docker-compose.optimized.yml logs -f"
    echo "   - 停止服务：    docker-compose -f docker-compose.optimized.yml down"
    echo "   - 重启服务：    docker-compose -f docker-compose.optimized.yml restart"
    echo "   - 进入容器：    docker exec -it medical-triage-api bash"
    echo ""
    echo "📌 API 测试："
    echo "   - 健康检查：    curl http://localhost:8012/v1/health"
    echo "   - API 文档：    http://localhost:8012/docs"
    echo ""
}

# 主函数
main() {
    echo ""
    log_info "=========================================="
    log_info "    智能医疗分诊系统 - Docker 部署脚本"
    log_info "=========================================="
    echo ""
    
    # 检查前置条件
    check_docker
    check_env
    
    # 清理旧容器
    read -p "是否清理旧容器？(y/n): " cleanup_choice
    if [ "$cleanup_choice" = "y" ]; then
        cleanup
    fi
    
    # 构建镜像
    read -p "是否构建新镜像？(y/n): " build_choice
    if [ "$build_choice" = "y" ]; then
        build_image
    else
        log_info "跳过镜像构建，使用现有镜像"
    fi
    
    # 启动服务
    start_services
    
    # 健康检查
    health_check
    
    # 显示信息
    show_info
}

# 执行主函数
main
