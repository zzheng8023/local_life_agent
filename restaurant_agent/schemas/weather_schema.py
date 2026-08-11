"""
Weather Schema — Pydantic data models for weather queries and results.

Weather tool accepts standardized date (resolved by TimeResolver),
NOT natural language time expressions.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ================================================================
# WeatherResult — Weather data for a specific date
# ================================================================


class WeatherResult(BaseModel):
    """Standardized weather data for a single date."""

    date: str = Field(
        ...,
        description="Date in ISO format: '2026-08-04'",
    )
    weather: str = Field(
        default="晴",
        description="Weather condition in Chinese: 晴/多云/阴/小雨/中雨/大雨/雷阵雨/雪",
    )
    temperature: str = Field(
        default="",
        description="Temperature range: '28-35℃'",
    )
    humidity: int = Field(
        default=60,
        ge=0,
        le=100,
        description="Humidity percentage: 0-100",
    )


# ================================================================
# GetWeatherInput — Weather tool input parameters
# ================================================================


class GetWeatherInput(BaseModel):
    """Input schema for the weather tool.

    Both city and date are required. Date must be pre-resolved
    (ISO format YYYY-MM-DD) — NOT natural language.
    """

    city: str = Field(
        default="上海",
        description="City name in Chinese, e.g. '上海', '北京', '深圳'",
    )
    date: str = Field(
        ...,
        description="Target date in YYYY-MM-DD format (already resolved by TimeResolver)",
    )
