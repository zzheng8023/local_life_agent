"""
SQLite 持久化数据库模块 (Database Module)

统一管理项目所有持久化数据的 SQLite 存储，替换原有的 JSON 文件方案。
支持会话(sessions)、收藏(favorites)、审计日志(audit_logs)三张核心表。

特性：
- WAL 模式：支持并发读写
- 自动建表：首次连接时自动初始化表结构
- 线程安全：每个操作使用独立连接
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from application.config import DB_FILE, ensure_directories


# ── 模块级初始化 ──
_init_lock: threading.Lock = threading.Lock()
_initialized: bool = False


def _ensure_db_dir() -> None:
    """确保数据库目录存在。"""
    ensure_directories()


def _init_schema(conn: sqlite3.Connection) -> None:
    """初始化数据库表结构（幂等，仅首次执行）。"""
    global _initialized
    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                data_json    TEXT NOT NULL,
                created_at   TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                updated_at   TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            );

            CREATE TABLE IF NOT EXISTS favorites (
                id              TEXT PRIMARY KEY,
                content         TEXT NOT NULL,
                created_at      REAL NOT NULL,
                session_id      TEXT DEFAULT '',
                restaurant_name TEXT DEFAULT '',
                tags_json       TEXT DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS audit_logs (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp           REAL NOT NULL,
                timestamp_iso       TEXT NOT NULL,
                session_id          TEXT NOT NULL,
                passed              INTEGER NOT NULL,
                violations_json     TEXT DEFAULT '[]',
                prefilter_hit_count INTEGER DEFAULT 0,
                prefilter_rule_ids_json TEXT DEFAULT '[]',
                original_length     INTEGER DEFAULT 0,
                original_preview    TEXT DEFAULT '',
                rewritten           INTEGER DEFAULT 0,
                rewritten_preview   TEXT DEFAULT '',
                metadata_json       TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_favorites_created
                ON favorites(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp
                ON audit_logs(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_audit_session
                ON audit_logs(session_id);
        """)

        conn.commit()
        _initialized = True


# ── 连接管理 ──

@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """获取一个独立的数据库连接，自动初始化表结构。

    使用上下文管理器确保连接在使用后正确关闭。
    """
    _ensure_db_dir()
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    try:
        _init_schema(conn)
        yield conn
    finally:
        conn.close()


# ── 辅助函数 ──

def _to_json(obj: Any) -> str:
    """将 Python 对象序列化为 JSON 字符串。"""
    return json.dumps(obj, ensure_ascii=False)


def _from_json(text: str | None) -> Any:
    """从 JSON 字符串反序列化为 Python 对象。"""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text
