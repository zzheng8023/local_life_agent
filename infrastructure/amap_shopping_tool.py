"""
高德地图购物场所搜索工具 (Amap Shopping Search Tool)

本模块实现基于高德地图 POI 搜索 API 的真实商场/购物中心搜索。
搜索类型码 types=060100（商场）。

核心设计：
- 实现 ITool 接口，遵循 try-real-catch-fallback 三级模式
- 无 API Key 时降级到 web_tools 中的 mock 数据
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from loguru import logger

from application.ports import ITool
from infrastructure.usage_tracker import UsageTracker

_AMAP_SEARCH_URL: str = "https://restapi.amap.com/v3/place/text"


class AmapShoppingTool(ITool):
    """基于高德地图 POI 搜索的真实商场搜索工具。

    搜索类型码 types=060100（商场），覆盖购物中心、百货商场等。
    """

    def __init__(self, api_key: str | None = None, city: str = "北京") -> None:
        self._api_key: str = api_key or os.getenv("AMAP_API_KEY", "")
        self._city: str = city
        if self._api_key:
            logger.info(f"[AmapShop] 初始化完成, city={city}")
        else:
            logger.warning("[AmapShop] 未配置 AMAP_API_KEY，将使用模拟数据")

    # ------------------------------------------------------------------
    # ITool 接口实现
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return "amap_shopping_search"

    def get_description(self) -> str:
        return (
            "基于高德地图 POI 数据搜索真实商场/购物中心。"
            "参数：query(搜索关键词), city(城市)。"
        )

    def execute(self, **kwargs: Any) -> list[dict[str, Any]]:
        logger.info(f"[AmapShop] 搜索: {kwargs}")

        if not self._api_key:
            logger.warning("[AmapShop] 无 API Key，降级为模拟数据")
            return self._fallback_mock(kwargs)

        try:
            results: list[dict[str, Any]] = self._search_amap_shopping(kwargs)
            if results:
                UsageTracker().record_amap_call()
                logger.info(f"[AmapShop] 高德返回 {len(results)} 个商场")
                return results
        except Exception as exc:
            logger.error(f"[AmapShop] API 调用失败: {exc}")

        logger.warning("[AmapShop] 降级为模拟数据")
        return self._fallback_mock(kwargs)

    # ------------------------------------------------------------------
    # 高德 API 调用
    # ------------------------------------------------------------------

    def _search_amap_shopping(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        city: str = str(params.get("city") or self._city)
        raw_query: str = str(params.get("query", ""))

        keywords: str = raw_query if raw_query else "商场"

        request_params: dict[str, Any] = {
            "key": self._api_key,
            "keywords": keywords,
            "types": "060100",  # 商场大类
            "city": city,
            "offset": 20,
            "page": 1,
            "extensions": "all",
        }

        response: httpx.Response = httpx.get(
            _AMAP_SEARCH_URL,
            params=request_params,
            timeout=10.0,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()

        if data.get("status") != "1":
            logger.error(f"[AmapShop] API 返回错误: {data.get('info', 'unknown')}")
            return []

        pois: list[dict[str, Any]] = data.get("pois", [])
        if not pois:
            logger.info("[AmapShop] 未找到匹配商场")
            return []

        malls: list[dict[str, Any]] = []
        for poi in pois:
            m: dict[str, Any] = self._parse_shopping_poi(poi)
            if m:
                malls.append(m)

        return malls

    @staticmethod
    def _parse_shopping_poi(poi: dict[str, Any]) -> dict[str, Any]:
        name: str = poi.get("name", "未知商场")
        address: str = poi.get("address", "")

        # 解析坐标
        location: str = poi.get("location", "")
        longitude: float | None = None
        latitude: float | None = None
        if location and "," in location:
            try:
                parts = location.split(",")
                longitude = float(parts[0])
                latitude = float(parts[1])
            except (ValueError, IndexError):
                pass

        distance_str: str = poi.get("distance", "")
        distance_km: float | None = None
        if distance_str:
            try:
                distance_km = round(float(distance_str) / 1000, 2)
            except (ValueError, TypeError):
                pass

        biz_ext: dict[str, Any] = poi.get("biz_ext", {})
        rating_str: str = biz_ext.get("rating", "")

        rating: float = 0.0
        if rating_str:
            try:
                rating = float(rating_str)
            except (ValueError, TypeError):
                pass

        features: list[str] = []
        photos: list[dict[str, Any]] = poi.get("photos", [])
        if photos:
            features.append("有实拍图")

        tel: str = poi.get("tel", "")
        if tel:
            features.append(f"电话:{tel}")

        return {
            "name": name,
            "category": "商场",
            "rating": rating,
            "address": address or "未知地址",
            "longitude": longitude,
            "latitude": latitude,
            "distance_to_restaurant": distance_km,
            "features": features,
            "current_movies": [],
            "showtimes_after_meal": [],
            "queue_status": "营业中",
            "has_group_buy": False,
            "source": "amap_shopping",
        }

    # ------------------------------------------------------------------
    # 降级策略
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_mock(params: dict[str, Any]) -> list[dict[str, Any]]:
        from infrastructure.web_tools import HotelAndEntertainmentSearchTool
        return HotelAndEntertainmentSearchTool._mock_shopping(params)
