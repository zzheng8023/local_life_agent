"""
收藏夹持久化存储 (Favorites Store)

本模块提供用户收藏的 SQLite 持久化存储。
每个收藏项包含推荐文本、时间戳、来源会话 ID。

设计：
- 存储引擎: SQLite（替代原 JSON 文件方案）
- 线程安全：每次操作使用独立连接
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from infrastructure.db import get_connection, _to_json, _from_json
from loguru import logger


class FavoritesStore:
    """收藏夹持久化存储（基于 SQLite）。"""

    def add(
        self,
        content: str,
        session_id: str = "",
        restaurant_name: str = "",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """添加一条收藏。

        Returns:
            新创建的收藏项 dict。
        """
        item_id: str = uuid.uuid4().hex[:12]
        created_at: float = time.time()

        item: dict[str, Any] = {
            "id": item_id,
            "content": content,
            "created_at": created_at,
            "session_id": session_id,
            "restaurant_name": restaurant_name,
            "tags": tags or [],
        }

        with get_connection() as conn:
            conn.execute(
                "INSERT INTO favorites (id, content, created_at, session_id, "
                "restaurant_name, tags_json) VALUES (?, ?, ?, ?, ?, ?)",
                (item_id, content, created_at, session_id,
                 restaurant_name, _to_json(tags or [])),
            )
            conn.commit()

        logger.info(
            f"[Favorites] 已收藏: {restaurant_name or '未命名'} "
            f"(id={item_id})"
        )
        return item

    def remove(self, item_id: str) -> bool:
        """删除一条收藏。返回是否成功。"""
        with get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM favorites WHERE id = ?", (item_id,)
            )
            conn.commit()
            deleted: bool = cursor.rowcount > 0
            if deleted:
                logger.info(f"[Favorites] 已删除: id={item_id}")
            return deleted

    def list_all(self) -> list[dict[str, Any]]:
        """返回全部收藏项（按时间倒序）。"""
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM favorites ORDER BY created_at DESC"
            ).fetchall()

        return [
            {
                "id": row["id"],
                "content": row["content"],
                "created_at": row["created_at"],
                "session_id": row["session_id"],
                "restaurant_name": row["restaurant_name"],
                "tags": _from_json(row["tags_json"]) or [],
            }
            for row in rows
        ]
