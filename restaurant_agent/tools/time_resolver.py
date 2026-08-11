"""
Time Resolver — Pure deterministic date computation.

Converts TimeExpression (LLM semantic output) into ResolvedTime (precise date/datetime).
ZERO LLM dependency. ZERO regex dependency. Pure arithmetic.

Core formula:
    days_offset = 7 * week_offset + target_weekday - current_weekday

Design:
- ISO weekday (Mon=1..Sun=7) input → internal Python weekday (Mon=0..Sun=6)
- All dates use Asia/Shanghai timezone
- Supports: today, tomorrow, day_after_tomorrow, relative_weekday,
            relative_days, relative_month, date, period_only, reference, none
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

from restaurant_agent.schemas.time_schema import (
    TimeExpression,
    ResolvedTime,
    TimeContext,
)

# Beijing timezone (UTC+8)
_CST: timezone = timezone(timedelta(hours=8))

# Period → default hour mapping
_PERIOD_HOUR: dict[str, int] = {
    "morning": 9,
    "afternoon": 14,
    "evening": 18,
    "night": 21,
}


class TimeResolver:
    """Pure deterministic time resolver — LLM NOT involved in date computation.

    Usage:
        resolver = TimeResolver()
        expr = TimeExpression(type="relative_weekday", weekday=2, week_offset=1)
        result = resolver.resolve(expr, current_date=date.today())
        # → ResolvedTime(resolved_date="2026-08-04", ...)
    """

    def __init__(self, timezone_name: str = "Asia/Shanghai") -> None:
        self._timezone: str = timezone_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        expr: TimeExpression,
        current_date: Optional[date] = None,
        prev_context: Optional[TimeContext] = None,
    ) -> ResolvedTime:
        """Convert TimeExpression → ResolvedTime.

        Args:
            expr: LLM-parsed time semantics (TimeExpression).
            current_date: Reference date (defaults to today).
            prev_context: Previous TimeContext for 'reference' type resolution.

        Returns:
            ResolvedTime with computed date and optional datetime.
        """
        today: date = current_date or date.today()

        if expr.type == "none":
            return ResolvedTime(
                original_expression="",
                resolved_date=today.isoformat(),
                resolved_datetime=self._make_datetime(today, None, None).isoformat(),
                timezone=self._timezone,
            )

        if expr.type == "reference":
            return self._resolve_reference(expr, prev_context, today)

        if expr.type == "today":
            dt = self._make_datetime(today, expr.period, expr.hour)
            return ResolvedTime(
                original_expression=expr.raw,
                resolved_date=today.isoformat(),
                resolved_datetime=dt.isoformat(),
                timezone=self._timezone,
            )

        if expr.type == "tomorrow":
            target = today + timedelta(days=1)
            dt = self._make_datetime(target, expr.period, expr.hour)
            return ResolvedTime(
                original_expression=expr.raw,
                resolved_date=target.isoformat(),
                resolved_datetime=dt.isoformat(),
                timezone=self._timezone,
            )

        if expr.type == "day_after_tomorrow":
            offset = expr.day_offset if expr.day_offset > 0 else 2
            target = today + timedelta(days=offset)
            dt = self._make_datetime(target, expr.period, expr.hour)
            return ResolvedTime(
                original_expression=expr.raw,
                resolved_date=target.isoformat(),
                resolved_datetime=dt.isoformat(),
                timezone=self._timezone,
            )

        if expr.type == "relative_days":
            offset = expr.day_offset if expr.day_offset > 0 else 1
            target = today + timedelta(days=offset)
            dt = self._make_datetime(target, expr.period, expr.hour)
            return ResolvedTime(
                original_expression=expr.raw,
                resolved_date=target.isoformat(),
                resolved_datetime=dt.isoformat(),
                timezone=self._timezone,
            )

        if expr.type == "relative_weekday":
            if expr.weekday is None:
                # No weekday specified → default today
                dt = self._make_datetime(today, expr.period, expr.hour)
                return ResolvedTime(
                    original_expression=expr.raw,
                    resolved_date=today.isoformat(),
                    resolved_datetime=dt.isoformat(),
                    timezone=self._timezone,
                )
            target = self._compute_weekday_date(
                weekday_iso=expr.weekday,
                week_offset=expr.week_offset,
                today=today,
            )
            dt = self._make_datetime(target, expr.period, expr.hour)
            return ResolvedTime(
                original_expression=expr.raw,
                resolved_date=target.isoformat(),
                resolved_datetime=dt.isoformat(),
                timezone=self._timezone,
            )

        if expr.type == "relative_month":
            offset_m = expr.month_offset if expr.month_offset else 1
            new_month = today.month + offset_m
            new_year = today.year + (new_month - 1) // 12
            new_month = ((new_month - 1) % 12) + 1
            day = min(today.day, 28)
            target = date(new_year, new_month, day)
            dt = self._make_datetime(target, expr.period, expr.hour)
            return ResolvedTime(
                original_expression=expr.raw,
                resolved_date=target.isoformat(),
                resolved_datetime=dt.isoformat(),
                timezone=self._timezone,
            )

        if expr.type == "date":
            if expr.date_iso:
                target = date.fromisoformat(expr.date_iso)
                dt = self._make_datetime(target, expr.period, expr.hour)
                return ResolvedTime(
                    original_expression=expr.raw,
                    resolved_date=target.isoformat(),
                    resolved_datetime=dt.isoformat(),
                    timezone=self._timezone,
                )

        if expr.type == "period_only":
            dt = self._make_datetime(today, expr.period, expr.hour)
            return ResolvedTime(
                original_expression=expr.raw,
                resolved_date=today.isoformat(),
                resolved_datetime=dt.isoformat(),
                timezone=self._timezone,
            )

        # Fallback: today
        dt = self._make_datetime(today, None, None)
        return ResolvedTime(
            original_expression=expr.raw,
            resolved_date=today.isoformat(),
            resolved_datetime=dt.isoformat(),
            timezone=self._timezone,
        )

    # ------------------------------------------------------------------
    # Weekday computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_weekday_date(
        weekday_iso: int,
        week_offset: int,
        today: date,
    ) -> date:
        """Compute target date from ISO weekday and week offset.

        Formula: days_offset = 7 * week_offset + target_weekday - current_weekday

        Args:
            weekday_iso: ISO weekday (1=Mon..7=Sun).
            week_offset: 0=this week, 1=next week, 2=week after next, -1=last week.
            today: Reference date.

        Returns:
            Computed target date.
        """
        # Convert ISO weekday (1=Mon) → Python weekday (0=Mon)
        target_py: int = weekday_iso - 1
        current_py: int = today.weekday()

        if week_offset == 0:
            # This week: find next occurrence (including today)
            delta: int = (target_py - current_py) % 7
            return today + timedelta(days=delta)
        else:
            # Offset weeks: from this week's Monday
            monday_this_week: date = today - timedelta(days=current_py)
            monday_target: date = monday_this_week + timedelta(days=week_offset * 7)
            return monday_target + timedelta(days=target_py)

    # ------------------------------------------------------------------
    # Reference resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_reference(
        expr: TimeExpression,
        prev_context: Optional[TimeContext],
        today: date,
    ) -> ResolvedTime:
        """Resolve time references ('当天', '那天') from previous context."""
        if prev_context and prev_context.resolved_date:
            resolved_dt: str = prev_context.resolved_datetime or (
                prev_context.resolved_date + "T12:00:00+08:00"
            )
            return ResolvedTime(
                original_expression=prev_context.raw or expr.raw,
                resolved_date=prev_context.resolved_date,
                resolved_datetime=resolved_dt,
                timezone=prev_context.timezone or "Asia/Shanghai",
            )
        # No previous context → default to today
        return ResolvedTime(
            original_expression=expr.raw,
            resolved_date=today.isoformat(),
            timezone="Asia/Shanghai",
        )

    # ------------------------------------------------------------------
    # Datetime combination
    # ------------------------------------------------------------------

    @staticmethod
    def _make_datetime(
        target_date: date,
        period: Optional[str],
        hour: Optional[int],
    ) -> datetime:
        """Combine date with period/hour into a timezone-aware datetime.

        Priority: explicit hour > period default > 12:00 (noon)
        """
        if hour is not None:
            t = time(hour, 0)
        elif period and period in _PERIOD_HOUR:
            t = time(_PERIOD_HOUR[period], 0)
        else:
            t = time(12, 0)  # default noon

        return datetime.combine(target_date, t, tzinfo=_CST)
