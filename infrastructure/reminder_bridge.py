"""
提醒任务桥接 (Reminder Bridge)

本模块为"设置就餐提醒"等履约前动作生成 MCP scheduled-task 参数。
桥接层——不直接调用 MCP 工具，而是生成参数供上层使用。

设计：
- 生成结构化的提醒任务参数（含 fireAt 绝对时间戳）
- 上层（Web UI / CLI）负责实际创建 scheduled task
- 向后兼容：不传 meal_dt 时保持原有行为
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

# Beijing timezone (UTC+8)
_CST: timezone = timezone(timedelta(hours=8))


def _format_fire_at(dt: datetime) -> str:
    """Format a datetime as ISO 8601 string with timezone for fireAt."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_CST)
    return dt.isoformat()


def create_meal_reminder_params(
    restaurant_name: str,
    meal_time: str = "",
    advance_minutes: int = 30,
    note: str = "",
    meal_dt: datetime | None = None,
) -> dict[str, Any]:
    """生成用餐提醒的 scheduled-task 参数。

    Args:
        restaurant_name: 餐厅名称。
        meal_time: 用餐时间描述（如 "周六18:30"）。
        advance_minutes: 提前多少分钟提醒（默认 30）。
        note: 附加备注。
        meal_dt: 解析后的绝对用餐时间。提供时将计算 fireAt。

    Returns:
        可用于 mcp__scheduled-tasks__create_scheduled_task 的参数字典。
    """
    task_id: str = (
        f"meal-reminder-{restaurant_name[:8]}"
        .replace(" ", "-")
        .replace("·", "-")
        .lower()
    )

    description: str = f"用餐提醒: {restaurant_name}"
    if meal_time:
        description += f" @ {meal_time}"

    prompt: str = (
        f"⏰ 用餐提醒\n\n"
        f"餐厅: **{restaurant_name}**\n"
        f"时间: {meal_time or '请查看推荐方案'}\n"
        f"提前 {advance_minutes} 分钟提醒\n"
        f"{note}\n\n"
        f"请确认是否已到达餐厅，祝用餐愉快！🍽️"
    )

    result: dict[str, Any] = {
        "taskId": task_id,
        "description": description,
        "prompt": prompt,
        "restaurant_name": restaurant_name,
        "meal_time": meal_time,
        "advance_minutes": advance_minutes,
        "note": note,
    }

    # v3.1: 生成 fireAt 绝对时间戳
    if meal_dt is not None:
        fire_at_dt: datetime = meal_dt - timedelta(minutes=advance_minutes)
        result["fireAt"] = _format_fire_at(fire_at_dt)
        result["meal_dt"] = _format_fire_at(meal_dt)

    return result


def create_hotel_checkin_reminder_params(
    hotel_name: str,
    checkin_time: str = "",
) -> dict[str, Any]:
    """生成酒店入住提醒的 scheduled-task 参数。

    Args:
        hotel_name: 酒店名称。
        checkin_time: 入住时间。

    Returns:
        可用于创建 scheduled task 的参数字典。
    """
    task_id: str = (
        f"hotel-checkin-{hotel_name[:8]}"
        .replace(" ", "-")
        .replace("·", "-")
        .lower()
    )

    prompt: str = (
        f"🏨 酒店入住提醒\n\n"
        f"酒店: **{hotel_name}**\n"
        f"入住时间: {checkin_time or '请查看推荐方案'}\n\n"
        f"请确认已办理入住，祝您休息愉快！🌙"
    )

    return {
        "taskId": task_id,
        "description": f"酒店入住提醒: {hotel_name}",
        "prompt": prompt,
        "hotel_name": hotel_name,
        "checkin_time": checkin_time,
    }
