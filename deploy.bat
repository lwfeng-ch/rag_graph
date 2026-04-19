@echo off
REM ============================================
REM 智能医疗分诊系统 - Docker 快速部署脚本 (Windows 版)
REM ============================================

setlocal enabledelayedexpansion

REM 颜色定义（Windows 10+ 支持 ANSI）
set "BLUE=[94m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "NC=[0m"

REM 日志函数
:log_info
echo %BLUE%[INFO]%NC% %1
goto :eof

:log_success
echo %GREEN%[SUCCESS]%NC% %1
goto :eof

:log_warning
echo %YELLOW%[WARNING]%NC% %1
goto :eof

:log_error
echo %RED%[ERROR]%NC% %1
goto :eof

REM 检查 Docker 是否安装
:check_docker
where docker >nul 2>nul
if %errorlevel% neq 0 (
    call :log_error "Docker 未安装，请先安装 Docker Desktop"
    pause
    exit /b 1
)

where docker-compose >nul 2>nul
if %errorlevel% neq 0 (
    call :log_error "Docker Compose 未安装，请确保 Docker Desktop 已正确安装"
    pause
    exit /b 1
)

call :log_success "Docker 和 Docker Compose 已安装"
goto :check_env

REM 检查 .env 文件
:check_env
if not exist .env (
    call :log_warning ".env 文件不存在，正在创建..."
    copy .env.example .env >nul
    call :log_info "请编辑 .env 文件，配置 API Key 和数据库密码"
    call :log_info "按任意键继续..."
    pause >nul
) else (
    call :log_success ".env 文件已存在"
)
goto :cleanup

REM 停止并清理旧容器
:cleanup
call :log_info "停止并清理旧容器..."
docker-compose -f docker-compose.optimized.yml down >nul 2>&1
call :log_success "清理完成"
goto :build_image

REM 构建镜像
:build_image
echo.
echo 请选择 requirements 文件：
echo 1^) requirements-lite.txt ^(精简版 - 推荐生产环境^)
echo 2^) requirements.txt ^(完整版 - 推荐开发环境^)
set /p req_choice="请输入选项 (1/2): "

if "%req_choice%"=="1" (
    set "REQ_FILE=requirements-lite.txt"
    call :log_info "使用精简版 requirements (requirements-lite.txt)"
) else if "%req_choice%"=="2" (
    set "REQ_FILE=requirements.txt"
    call :log_info "使用完整版 requirements (requirements.txt)"
) else (
    call :log_error "无效选项，使用精简版"
    set "REQ_FILE=requirements-lite.txt"
)

call :log_info "开始构建 Docker 镜像..."
docker build ^
    -f Dockerfile.optimized ^
    --build-arg REQUIREMENTS_FILE=%REQ_FILE% ^
    -t medical-triage:latest ^
    .

call :log_success "镜像构建完成"
goto :start_services

REM 启动服务
:start_services
call :log_info "启动服务..."
docker-compose -f docker-compose.optimized.yml up -d

call :log_info "等待服务启动..."
timeout /t 10 /nobreak >nul

REM 检查服务状态
docker-compose -f docker-compose.optimized.yml ps
goto :health_check

REM 健康检查
:health_check
call :log_info "执行健康检查..."

REM 等待 FastAPI 启动
for /l %%i in (1,1,30) do (
    curl -f http://localhost:8012/v1/health >nul 2>&1 && (
        call :log_success "FastAPI 服务健康检查通过"
        goto :check_qdrant
    )
    
    if %%i==30 (
        call :log_error "FastAPI 服务启动超时"
        exit /b 1
    )
    
    echo -n .
    timeout /t 2 /nobreak >nul
)

:check_qdrant
curl -f http://localhost:6333/ >nul 2>&1 && (
    call :log_success "Qdrant 服务健康检查通过"
) || (
    call :log_warning "Qdrant 服务未响应，请稍后检查"
)

REM 检查 PostgreSQL
docker exec postgres pg_isready -U rag_user >nul 2>&1 && (
    call :log_success "PostgreSQL 服务健康检查通过"
) || (
    call :log_warning "PostgreSQL 服务未响应，请稍后检查"
)

goto :show_info

REM 显示访问信息
:show_info
echo.
call :log_success "=========================================="
call :log_success "    智能医疗分诊系统部署完成！"
call :log_success "=========================================="
echo.
echo 📌 服务访问地址：
echo    - FastAPI:      http://localhost:8012
echo    - Gradio UI:    http://localhost:7860
echo    - Qdrant:       http://localhost:6333
echo    - PostgreSQL:   localhost:5432
echo.
echo 📌 常用命令：
echo    - 查看日志：    docker-compose -f docker-compose.optimized.yml logs -f
echo    - 停止服务：    docker-compose -f docker-compose.optimized.yml down
echo    - 重启服务：    docker-compose -f docker-compose.optimized.yml restart
echo    - 进入容器：    docker exec -it medical-triage-api bash
echo.
echo 📌 API 测试：
echo    - 健康检查：    curl http://localhost:8012/v1/health
echo    - API 文档：    http://localhost:8012/docs
echo.

endlocal
exit /b 0
