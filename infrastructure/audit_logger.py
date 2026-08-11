"""
安全审计日志 (Audit Logger)

本模块为每次安全审查决策提供持久化审计追踪。
所有安全决策（通过/拦截/改写）均写入 SQLite 数据库。

设计：
- 存储引擎: SQLite（替代原 JSONL 文件方案）
- 每条记录包含：时间戳、会话 ID、原始/改写文本摘要、预筛选命中、LLM 审查结果
"""

from __future__ import annotations

import time
from typing import Any

from infrastructure.db import get_connection, _to_json, _from_json
from loguru import logger


class AuditLogger:
    """安全审查审计日志记录器（基于 SQLite）。"""

    def log_safety_decision(
        self,
        session_id: str,
        original_text: str,
        passed: bool,
        violations: list[str],
        rewritten_text: str = "",
        prefilter_hits: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """记录一条安全审查决策。"""
        timestamp: float = time.time()
        prefilter_rule_ids: list[str] = (
            [h["rule_id"] for h in prefilter_hits] if prefilter_hits else []
        )

        try:
            with get_connection() as conn:
                conn.execute(
                    "INSERT INTO audit_logs (timestamp, timestamp_iso, session_id, "
                    "passed, violations_json, prefilter_hit_count, "
                    "prefilter_rule_ids_json, original_length, original_preview, "
                    "rewritten, rewritten_preview, metadata_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        timestamp,
                        _iso_now(),
                        session_id,
                        1 if passed else 0,
                        _to_json(violations),
                        len(prefilter_hits) if prefilter_hits else 0,
                        _to_json(prefilter_rule_ids),
                        len(original_text),
                        original_text[:200],
                        1 if rewritten_text else 0,
                        rewritten_text[:200] if rewritten_text else "",
                        _to_json(metadata or {}),
                    ),
                )
                conn.commit()
            logger.info(
                f"[Audit] 日志已写入: passed={passed}, "
                f"prefilter_hits={len(prefilter_hits) if prefilter_hits else 0}, "
                f"violations={violations}"
            )
        except Exception as exc:
            logger.error(f"[Audit] 写入审计日志失败: {exc}")

    def read_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """读取最近 N 条审计记录。"""
        try:
            with get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        except Exception as exc:
            logger.error(f"[Audit] 读取审计日志失败: {exc}")
            return []

        return [
            {
                "timestamp": row["timestamp"],
                "timestamp_iso": row["timestamp_iso"],
                "session_id": row["session_id"],
                "passed": bool(row["passed"]),
                "violations": _from_json(row["violations_json"]) or [],
                "prefilter_hit_count": row["prefilter_hit_count"],
                "prefilter_rule_ids": _from_json(row["prefilter_rule_ids_json"]) or [],
                "original_length": row["original_length"],
                "original_preview": row["original_preview"],
                "rewritten": bool(row["rewritten"]),
                "rewritten_preview": row["rewritten_preview"],
                "metadata": _from_json(row["metadata_json"]) or {},
            }
            for row in rows
        ]


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------

def _iso_now() -> str:
    """返回 ISO 8601 格式的当前时间字符串。"""
    from datetime import datetime, timezone, timedelta
    tz = timezone(timedelta(hours=8))
    return datetime.now(tz).isoformat()
