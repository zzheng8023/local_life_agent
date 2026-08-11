"""
Weather Tool — Weather information by exact date (NOT natural language time).

Accepts pre-resolved dates (ISO YYYY-MM-DD format) and returns weather data.
Uses deterministic mock data based on date hash for realistic Chinese city weather.
No real weather API key required.

MCP-compatible tool schema included for future migration.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime

from restaurant_agent.schemas.weather_schema import WeatherResult, GetWeatherInput


# ================================================================
# MCP-compatible Tool Schema
# ================================================================

WEATHER_TOOL_SCHEMA: dict = {
    "name": "get_weather",
    "description": (
        "Get weather information for a specific date and city. "
        "Date must be pre-resolved to ISO format (YYYY-MM-DD) — "
        "do NOT pass natural language time expressions."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name in Chinese, e.g. '上海', '北京', '深圳'",
            },
            "date": {
                "type": "string",
                "description": "Target date in YYYY-MM-DD format",
            },
        },
        "required": ["city", "date"],
    },
    "outputSchema": {
        "type": "object",
        "properties": {
            "date": {"type": "string"},
            "weather": {"type": "string", "description": "晴/多云/阴/小雨/中雨/大雨/雷阵雨/雪"},
            "temperature": {"type": "string", "description": "e.g. '28-35℃'"},
            "humidity": {"type": "integer", "description": "0-100"},
        },
    },
}


# ================================================================
# Deterministic mock weather data
# ================================================================

# Weather patterns by season (month range)
_SEASON_WEATHER: dict[tuple[int, ...], list[dict]] = {
    (12, 1, 2): [  # Winter
        {"weather": "晴", "temp_range": (-5, 5), "humidity_range": (20, 45)},
        {"weather": "多云", "temp_range": (-3, 7), "humidity_range": (25, 50)},
        {"weather": "阴", "temp_range": (-2, 4), "humidity_range": (30, 55)},
        {"weather": "小雪", "temp_range": (-8, 0), "humidity_range": (40, 65)},
    ],
    (3, 4, 5): [  # Spring
        {"weather": "晴", "temp_range": (12, 25), "humidity_range": (30, 55)},
        {"weather": "多云", "temp_range": (15, 22), "humidity_range": (35, 60)},
        {"weather": "小雨", "temp_range": (10, 18), "humidity_range": (55, 80)},
        {"weather": "阴", "temp_range": (13, 20), "humidity_range": (45, 70)},
    ],
    (6, 7, 8): [  # Summer
        {"weather": "晴", "temp_range": (28, 38), "humidity_range": (45, 70)},
        {"weather": "多云", "temp_range": (26, 35), "humidity_range": (50, 75)},
        {"weather": "雷阵雨", "temp_range": (24, 32), "humidity_range": (65, 90)},
        {"weather": "中雨", "temp_range": (22, 28), "humidity_range": (70, 95)},
    ],
    (9, 10, 11): [  # Autumn
        {"weather": "晴", "temp_range": (15, 26), "humidity_range": (30, 50)},
        {"weather": "多云", "temp_range": (13, 23), "humidity_range": (35, 55)},
        {"weather": "小雨", "temp_range": (10, 18), "humidity_range": (50, 75)},
        {"weather": "阴", "temp_range": (12, 20), "humidity_range": (45, 65)},
    ],
}

# City temperature offsets (degrees Celsius)
_CITY_OFFSETS: dict[str, float] = {
    "北京": 0.0,
    "上海": 2.0,
    "广州": 6.0,
    "深圳": 6.0,
    "杭州": 1.5,
    "成都": -1.0,
    "武汉": 3.0,
    "西安": -2.0,
    "南京": 1.0,
    "重庆": 1.0,
    "厦门": 5.0,
    "哈尔滨": -12.0,
    "三亚": 10.0,
}


class WeatherTool:
    """Weather lookup tool — accepts standard date only, NO time parsing.

    Usage:
        tool = WeatherTool()
        result = tool.get_weather(city="上海", date="2026-08-04")
        # → WeatherResult(date="2026-08-04", weather="晴", temperature="28-35℃", humidity=60)
    """

    # ------------------------------------------------------------------
    # ITool-compatible interface
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return "get_weather"

    def get_description(self) -> str:
        return (
            "Get weather information for a specific date and city. "
            "Date must be pre-resolved ISO format (YYYY-MM-DD)."
        )

    def execute(self, **kwargs) -> dict:
        """ITool-compatible execute method."""
        city: str = kwargs.get("city", "上海")
        target_date: str = kwargs.get("date", "")
        result: WeatherResult = self.get_weather(city=city, target_date_str=target_date)
        return result.model_dump()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get_weather(
        self,
        city: str = "上海",
        target_date_str: str = "",
    ) -> WeatherResult:
        """Get weather for a specific date.

        Args:
            city: City name in Chinese.
            target_date_str: Date in ISO format YYYY-MM-DD.

        Returns:
            WeatherResult with weather, temperature, humidity.
        """
        target_date: date
        if target_date_str:
            target_date = date.fromisoformat(target_date_str)
        else:
            target_date = date.today()

        # Deterministic mock based on date hash
        result = self._generate_weather(city, target_date)
        return result

    # ------------------------------------------------------------------
    # Mock weather generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_weather(city: str, target_date: date) -> WeatherResult:
        """Generate deterministic mock weather data from date + city."""
        # Hash the date to pick a weather pattern
        date_key: str = target_date.isoformat()
        hash_val: int = int(
            hashlib.md5(date_key.encode()).hexdigest(), 16
        )

        # Find season
        month: int = target_date.month
        season_patterns = []
        for months, patterns in _SEASON_WEATHER.items():
            if month in months:
                season_patterns = patterns
                break

        if not season_patterns:
            season_patterns = _SEASON_WEATHER[(6, 7, 8)]

        # Pick weather based on hash
        idx: int = hash_val % len(season_patterns)
        pattern = season_patterns[idx]

        # Apply city offset to temperature
        city_offset = _CITY_OFFSETS.get(city, 0.0)
        temp_lo = int(pattern["temp_range"][0] + city_offset)
        temp_hi = int(pattern["temp_range"][1] + city_offset)

        # Add daily variation based on hash
        daily_variation = (hash_val // 100) % 5
        temp_lo += daily_variation
        temp_hi += daily_variation

        # Humidity
        hum_lo, hum_hi = pattern["humidity_range"]
        humidity = hum_lo + (hash_val % (hum_hi - hum_lo + 1))

        return WeatherResult(
            date=target_date.isoformat(),
            weather=pattern["weather"],
            temperature=f"{temp_lo}-{temp_hi}℃",
            humidity=humidity,
        )
