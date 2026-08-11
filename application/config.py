"""
统一配置模块 (Application Configuration)

本模块集中管理应用层所有硬编码常量与可配置参数，
消除分散在各文件中的魔法值，提升可维护性与可运维性。

所有值均可通过环境变量覆盖，不配置时使用合理的默认值。
"""

from __future__ import annotations

import os
from pathlib import Path

# ── 项目根目录 ──
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


# ══════════════════════════════════════════════════════════════
# Web UI
# ══════════════════════════════════════════════════════════════

WEBUI_SERVER_NAME: str = os.getenv("WEBUI_SERVER_NAME", "0.0.0.0")
WEBUI_SERVER_PORT: int = int(os.getenv("WEBUI_SERVER_PORT", "7860"))
WEBUI_CONCURRENCY_LIMIT: int = int(os.getenv("WEBUI_CONCURRENCY_LIMIT", "3"))
WEBUI_SHARE: bool = os.getenv("WEBUI_SHARE", "").lower() == "true"

# ── Gradio 主题 ──
GRADIO_THEME: str = os.getenv("GRADIO_THEME", "soft")


# ══════════════════════════════════════════════════════════════
# Workflow
# ══════════════════════════════════════════════════════════════

MAX_WINDOW_MESSAGES: int = int(os.getenv("MAX_WINDOW_MESSAGES", "30"))
STREAM_HEARTBEAT_INTERVAL: float = float(
    os.getenv("STREAM_HEARTBEAT_INTERVAL", "2.5")
)
SAFETY_HEARTBEAT_INTERVAL: float = float(
    os.getenv("SAFETY_HEARTBEAT_INTERVAL", "0.5")
)

# ── 默认城市 ──
DEFAULT_CITY: str = os.getenv("DEFAULT_CITY", "北京")


# ══════════════════════════════════════════════════════════════
# 日志与持久化路径
# ══════════════════════════════════════════════════════════════

DATA_DIR: Path = _PROJECT_ROOT / "data"
LOGS_DIR: Path = _PROJECT_ROOT / "logs"
TRACES_DIR: Path = LOGS_DIR / "traces"
DB_FILE: Path = DATA_DIR / "local_life.db"

# ── 日志轮转：每个 trace 文件最大 5MB，保留最近 100 个 ──
TRACE_MAX_COUNT: int = int(os.getenv("TRACE_MAX_COUNT", "100"))


# ══════════════════════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════════════════════

LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def ensure_directories() -> None:
    """确保项目所需的目录结构存在。"""
    for d in (DATA_DIR, LOGS_DIR, TRACES_DIR):
        d.mkdir(parents=True, exist_ok=True)
