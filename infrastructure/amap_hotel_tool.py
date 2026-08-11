"""
高德地图酒店搜索工具 (Amap Hotel Search Tool)

本模块实现基于高德地图 POI 搜索 API 的真实酒店搜索能力。
复用与 AmapRestaurantTool 相同的 API Key 和错误处理模式。

核心设计：
- 实现 ITool 接口，遵循 try-real-catch-fallback 三级模式
- types=100000（住宿服务大类）
- 预算过滤使用"每晚价格"语义（与餐饮的"人均"区分）
- 无 API Key 时降级到 web_tools 中的 mock 数据
"""

from __future__ import annotations

import os
import re
from typing import Any

import httpx
from loguru import logger

from application.ports import ITool
from infrastructure.usage_tracker import UsageTracker

_AMAP_SEARCH_URL: str = "https://restapi.amap.com/v3/place/text"


class AmapHotelTool(ITool):
    """基于高德地图 POI 搜索的真实酒店搜索工具。

    实现 ITool 接口，调用高德 place/text API 搜索住宿服务 POI。
    搜索类型码 types=100000（住宿服务），覆盖酒店、宾馆、旅馆等。
    """

    def __init__(self, api_key: str | None = None, city: str = "北京") -> None:
        self._api_key: str = api_key or os.getenv("AMAP_API_KEY", "")
        self._city: str = city
        if self._api_key:
            logger.info(f"[AmapHotel] 初始化完成, city={city}")
        else:
            logger.warning("[AmapHotel] 未配置 AMAP_API_KEY，将使用模拟数据")

    # ------------------------------------------------------------------
    # ITool 接口实现
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return "amap_hotel_search"

    def get_description(self) -> str:
        return (
            "基于高德地图 POI 数据搜索真实酒店。"
            "参数：query(搜索关键词), budget(预算), city(城市), "
            "distance(距离约束)。"
        )

    def execute(self, **kwargs: Any) -> list[dict[str, Any]]:
        logger.info(f"[AmapHotel] 搜索: {kwargs}")

        if not self._api_key:
            logger.warning("[AmapHotel] 无 API Key，降级为模拟数据")
            return self._fallback_mock(kwargs)

        try:
            results: list[dict[str, Any]] = self._search_amap_hotels(kwargs)
            if results:
                UsageTracker().record_amap_call()
                logger.info(f"[AmapHotel] 高德返回 {len(results)} 家酒店")
                return results
        except Exception as exc:
            logger.error(f"[AmapHotel] API 调用失败: {exc}")

        logger.warning("[AmapHotel] 降级为模拟数据")
        return self._fallback_mock(kwargs)

    # ------------------------------------------------------------------
    # 高德 API 调用
    # ------------------------------------------------------------------

    def _search_amap_hotels(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        city: str = str(params.get("city") or self._city)
        raw_query: str = str(params.get("query", ""))
        budget: str = str(params.get("budget", ""))

        keywords: str = raw_query if raw_query else "酒店"

        request_params: dict[str, Any] = {
            "key": self._api_key,
            "keywords": keywords,
            "types": "100000",  # 住宿服务大类
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
            logger.error(f"[AmapHotel] API 返回错误: {data.get('info', 'unknown')}")
            return []

        pois: list[dict[str, Any]] = data.get("pois", [])
        if not pois:
            logger.info("[AmapHotel] 未找到匹配酒店")
            return []

        hotels: list[dict[str, Any]] = []
        for poi in pois:
            h: dict[str, Any] = self._parse_hotel_poi(poi, budget)
            if h:
                hotels.append(h)

        return hotels

    @staticmethod
    def _parse_hotel_poi(poi: dict[str, Any], budget: str) -> dict[str, Any] | None:
        name: str = poi.get("name", "未知酒店")
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
        cost_str: str = biz_ext.get("cost", "")

        rating: float = 0.0
        if rating_str:
            try:
                rating = float(rating_str)
            except (ValueError, TypeError):
                pass

        avg_price: str = "价格未知"
        if cost_str:
            try:
                cost_val: float = float(cost_str)
                avg_price = f"{cost_val:.0f}元起/晚"
            except (ValueError, TypeError):
                avg_price = f"{cost_str}元起/晚"

        # 预算过滤（每晚价格 > budget * 1.3 则跳过）
        if budget and "人均" not in budget and avg_price != "价格未知":
            try:
                budget_match = re.search(r"(\d+)", budget)
                price_match = re.search(r"(\d+)", avg_price)
                if budget_match and price_match:
                    budget_val: int = int(budget_match.group(1))
                    price_val: int = int(price_match.group(1))
                    if price_val > budget_val * 1.3:
                        return None
            except (ValueError, AttributeError):
                pass

        # 提取酒店类型
        poi_type: str = poi.get("type", "")
        hotel_category: str = "酒店"
        if "星级" in poi_type or "hotel" in poi_type.lower():
            hotel_category = poi_type.split(";")[-1] if ";" in poi_type else "酒店"

        features: list[str] = []
        photos: list[dict[str, Any]] = poi.get("photos", [])
        if photos:
            features.append("有实拍图")

        tel: str = poi.get("tel", "")
        if tel:
            features.append(f"电话:{tel}")

        # 从 deep_info 提取额外信息
        deep_info: dict[str, Any] = poi.get("deep_info", {})
        if deep_info.get("business_area"):
            features.append(deep_info["business_area"])

        return {
            "name": name,
            "rating": rating,
            "avg_price": avg_price,
            "address": address or "未知地址",
            "longitude": longitude,
            "latitude": latitude,
            "distance_to_restaurant": distance_km,
            "features": features,
            "queue_status": "需电话确认",
            "has_group_buy": False,
            "source": "amap_hotels",
        }

    # ------------------------------------------------------------------
    # 降级策略
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_mock(params: dict[str, Any]) -> list[dict[str, Any]]:
        from infrastructure.web_tools import HotelAndEntertainmentSearchTool
        return HotelAndEntertainmentSearchTool._mock_hotels(params)
