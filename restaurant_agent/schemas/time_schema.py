"""
Time Schema — Pydantic data models for time expression parsing and resolution.

Core design:
- ISO weekday: Monday=1, Tuesday=2, ..., Sunday=7
- LLM outputs TimeExpression (time semantics only, NO date computation)
- Backend computes ResolvedTime (precise date/datetime)
- TimeContext is the serializable snapshot saved in ConversationContext
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ================================================================
# TimeExpression — LLM semantic output (NO date computation)
# ================================================================


class TimeExpression(BaseModel):
    """LLM-structured time semantics extracted from natural language.

    LLM MUST only identify the TYPE and extract PARAMETERS (weekday, week_offset, etc.).
    The backend TimeResolver computes the actual date.

    ISO weekday mapping:
        Monday=1, Tuesday=2, Wednesday=3, Thursday=4,
        Friday=5, Saturday=6, Sunday=7
    """

    raw: str = Field(
        default="",
        description="Original time expression text from user input, e.g. '下周二'",
    )
    type: str = Field(
        default="none",
        description=(
            "Time expression type: "
            "'today', 'tomorrow', 'day_after_tomorrow', "
            "'relative_weekday', 'relative_days', 'relative_month', "
            "'date', 'period_only', 'reference', 'none'"
        ),
    )
    weekday: Optional[int] = Field(
        default=None,
        description="ISO weekday: 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat, 7=Sun",
        ge=1,
        le=7,
    )
    week_offset: int = Field(
        default=0,
        description="Week offset: 0=this week, 1=next week, 2=week after next, -1=last week",
    )
    day_offset: int = Field(
        default=0,
        description="Day offset for 'relative_days' type: 1=tomorrow, 2=day after, etc.",
    )
    month_offset: int = Field(
        default=0,
        description="Month offset for 'relative_month' type: 1=next month",
    )
    date_iso: str = Field(
        default="",
        description="Explicit date in YYYY-MM-DD format (only when user says exact date like '8月1号')",
    )
    period: Optional[str] = Field(
        default=None,
        description="Time period: 'morning', 'afternoon', 'evening', 'night'",
    )
    hour: Optional[int] = Field(
        default=None,
        ge=0,
        le=23,
        description="Specific hour (24h) if mentioned. 下午6点 → hour=18",
    )


# ================================================================
# ResolvedTime — Backend-computed precise time
# ================================================================


class ResolvedTime(BaseModel):
    """Backend-computed precise time result.

    Computed deterministically by TimeResolver from a TimeExpression.
    NO LLM involved in this step.
    """

    original_expression: str = Field(
        default="",
        description="User's original time expression text",
    )
    resolved_date: str = Field(
        default="",
        description="Resolved date in ISO format: '2026-08-04'",
    )
    resolved_datetime: str = Field(
        default="",
        description="Resolved datetime in ISO 8601: '2026-08-04T19:00:00+08:00'",
    )
    timezone: str = Field(
        default="Asia/Shanghai",
        description="Timezone identifier",
    )


# ================================================================
# TimeContext — Serializable snapshot for conversation memory
# ================================================================


class TimeContext(BaseModel):
    """Serializable time context snapshot saved in ConversationContext.

    Enables cross-turn time reference resolution:
        Turn 1: "下周二吃川菜" → time_context saved with resolved_date="2026-08-04"
        Turn 2: "当天会下雨吗？" → reads time_context for "当天" resolution
    """

    raw: str = Field(
        default="",
        description="Original time expression, e.g. '下周二'",
    )
    resolved_date: str = Field(
        default="",
        description="Resolved date ISO string",
    )
    resolved_datetime: str = Field(
        default="",
        description="Resolved datetime ISO 8601 string",
    )
    timezone: str = Field(
        default="Asia/Shanghai",
        description="Timezone identifier",
    )
