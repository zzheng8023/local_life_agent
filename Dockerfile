FROM python:3.11-slim

LABEL org.opencontainers.image.title="local_life_agent"
LABEL org.opencontainers.image.description="基于 LangGraph 的本地生活智能决策助手"
LABEL org.opencontainers.image.version="3.4"

WORKDIR /app

# ── 系统依赖 ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python 依赖（分层缓存：先装依赖，后复制代码） ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 复制项目文件 ──
COPY . .

# ── 创建运行时目录 ──
RUN mkdir -p /app/data /app/logs

# ── 安全：非 root 用户运行 ──
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

ENV PYTHONUNBUFFERED=1

# 健康检查（Docker 层 + compose 层双重保障）
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
    CMD curl -f http://localhost:7860 || exit 1

# 默认启动 Web UI；可通过 CMD 覆盖
CMD ["python", "interfaces/web_ui.py"]
