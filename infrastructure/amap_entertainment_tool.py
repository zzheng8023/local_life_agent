"""
高德地图娱乐场所搜索工具 (Amap Entertainment Search Tool)

本模块实现基于高德地图 POI 搜索 API 的真实娱乐场所搜索。
涵盖电影院 (types=060400) 和娱乐休闲场所 (types=080000，如 KTV、酒吧等)。

核心设计：
- 实现 ITool 接口，遵循 try-real-catch-fallback 三级模式
- 仅返回位置、评分等客观数据，不伪造电影排片/场次
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

# 娱乐场所类型码
_ENTERTAINMENT_TYPE_CODES: str = "060400|080000"  # 电影院 | 娱乐休闲


class AmapEntertainmentTool(ITool):
    """基于高德地图 POI 搜索的真实娱乐场所搜索工具。

    搜索电影院 (060400) 和娱乐休闲场所 (080000：KTV、酒吧、网吧等)。
    仅返回位置、评分等客观信息，不伪造电影排片数据。
    """

    def __init__(self, api_key: str | None = None, city: str = "北京") -> None:
        self._api_key: str = api_key or os.getenv("AMAP_API_KEY", "")
        self._city: str = city
        if self._api_key:
            logger.info(f"[AmapEnt] 初始化完成, city={city}")
        else:
            logger.warning("[AmapEnt] 未配置 AMAP_API_KEY，将使用模拟数据")

    # ------------------------------------------------------------------
    # ITool 接口实现
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return "amap_entertainment_search"

    def get_description(self) -> str:
        return (
            "基于高德地图 POI 数据搜索真实娱乐场所（电影院、KTV 等）。"
            "参数：query(搜索关键词), city(城市)。"
        )

    def execute(self, **kwargs: Any) -> list[dict[str, Any]]:
        logger.info(f"[AmapEnt] 搜索: {kwargs}")

        if not self._api_key:
            logger.warning("[AmapEnt] 无 API Key，降级为模拟数据")
            return self._fallback_mock(kwargs)

        try:
            results: list[dict[str, Any]] = self._search_amap_entertainment(kwargs)
            if results:
                UsageTracker().record_amap_call()
                logger.info(f"[AmapEnt] 高德返回 {len(results)} 个娱乐场所")
                return results
        except Exception as exc:
            logger.error(f"[AmapEnt] API 调用失败: {exc}")

        logger.warning("[AmapEnt] 降级为模拟数据")
        return self._fallback_mock(kwargs)

    # ------------------------------------------------------------------
    # 高德 API 调用
    # ------------------------------------------------------------------

    def _search_amap_entertainment(
        self, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        city: str = str(params.get("city") or self._city)
        raw_query: str = str(params.get("query", ""))

        # 根据 query 智能选择搜索关键词
        if raw_query and ("电影" in raw_query or "影院" in raw_query):
            keywords: str = "电影院"
        elif raw_query and ("KTV" in raw_query or "唱歌" in raw_query or "ktv" in raw_query.lower()):
            keywords = "KTV"
        else:
            keywords = raw_query if raw_query else "电影院"

        request_params: dict[str, Any] = {
            "key": self._api_key,
            "keywords": keywords,
            "types": _ENTERTAINMENT_TYPE_CODES,
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
            logger.error(f"[AmapEnt] API 返回错误: {data.get('info', 'unknown')}")
            return []

        pois: list[dict[str, Any]] = data.get("pois", [])
        if not pois:
            logger.info("[AmapEnt] 未找到匹配娱乐场所")
            return []

        venues: list[dict[str, Any]] = []
        for poi in pois:
            v: dict[str, Any] = self._parse_entertainment_poi(poi)
            if v:
                venues.append(v)

        return venues

    @staticmethod
    def _parse_entertainment_poi(poi: dict[str, Any]) -> dict[str, Any]:
        name: str = poi.get("name", "未知场所")
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

        # 从 POI 类型推断 category
        poi_type: str = poi.get("type", "")
        if "电影院" in poi_type or "电影" in poi_type:
            category: str = "电影院"
        elif "KTV" in poi_type or "ktv" in poi_type.lower():
            category = "KTV"
        elif "酒吧" in poi_type:
            category = "酒吧"
        elif "网吧" in poi_type:
            category = "网吧"
        else:
            category = poi_type.split(";")[-1] if ";" in poi_type else "娱乐场所"

        features: list[str] = []
        photos: list[dict[str, Any]] = poi.get("photos", [])
        if photos:
            features.append("有实拍图")

        tel: str = poi.get("tel", "")
        if tel:
            features.append(f"电话:{tel}")

        return {
            "name": name,
            "category": category,
            "rating": rating,
            "address": address or "未知地址",
            "longitude": longitude,
            "latitude": latitude,
            "distance_to_restaurant": distance_km,
            "features": features,
            "current_movies": [],        # 不伪造排片数据
            "showtimes_after_meal": [],  # 不伪造场次数据
            "queue_status": "建议提前查询",
            "has_group_buy": False,
            "source": "amap_entertainment",
        }

    # ------------------------------------------------------------------
    # 降级策略
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_mock(params: dict[str, Any]) -> list[dict[str, Any]]:
        from infrastructure.web_tools import HotelAndEntertainmentSearchTool
        return HotelAndEntertainmentSearchTool._mock_entertainments(params)
