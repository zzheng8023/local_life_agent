"""
高德地图共享单车站点搜索工具 (Amap Bike Sharing Station Search Tool)

本模块实现基于高德地图 POI 搜索 API 的共享单车站点查询。
用于"骑单车过去方便吗""附近有共享单车吗"等出行场景。

核心设计：
- 实现 ITool 接口，遵循 try-real-catch-fallback 三级模式
- types=150600（共享单车站点）
- 返回站点名称、地址、距离、运营品牌等
- 无 API Key 时降级到 mock 数据
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from loguru import logger

from application.ports import ITool
from infrastructure.usage_tracker import UsageTracker

_AMAP_SEARCH_URL: str = "https://restapi.amap.com/v3/place/text"


class AmapBikeTool(ITool):
    """基于高德地图 POI 搜索的共享单车站点查询工具。

    实现 ITool 接口，调用高德 place/text API 搜索共享单车站点 POI。
    搜索类型码 types=150600（共享单车站点），覆盖美团/青桔/哈啰等。
    """

    def __init__(self, api_key: str | None = None, city: str = "北京") -> None:
        self._api_key: str = api_key or os.getenv("AMAP_API_KEY", "")
        self._city: str = city
        if self._api_key:
            logger.info(f"[AmapBike] 初始化完成, city={city}")
        else:
            logger.warning("[AmapBike] 未配置 AMAP_API_KEY，将使用模拟数据")

    # ------------------------------------------------------------------
    # ITool 接口实现
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return "amap_bike_search"

    def get_description(self) -> str:
        return (
            "基于高德地图 POI 数据搜索附近共享单车站点。"
            "参数：query(搜索关键词，如 '共享单车'), city(城市), "
            "location(中心点坐标，格式 'lng,lat')。"
        )

    def execute(self, **kwargs: Any) -> list[dict[str, Any]]:
        logger.info(f"[AmapBike] 搜索: {kwargs}")

        if not self._api_key:
            logger.warning("[AmapBike] 无 API Key，降级为模拟数据")
            return self._fallback_mock(kwargs)

        try:
            results: list[dict[str, Any]] = self._search_amap_bike_stations(kwargs)
            if results:
                UsageTracker().record_amap_call()
                logger.info(f"[AmapBike] 高德返回 {len(results)} 个单车站点")
                return results
        except Exception as exc:
            logger.error(f"[AmapBike] API 调用失败: {exc}")

        logger.warning("[AmapBike] 降级为模拟数据")
        return self._fallback_mock(kwargs)

    # ------------------------------------------------------------------
    # 高德 API 调用
    # ------------------------------------------------------------------

    def _search_amap_bike_stations(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        city: str = str(params.get("city") or self._city)
        raw_query: str = str(params.get("query", ""))
        location: str = str(params.get("location", ""))

        keywords: str = raw_query if raw_query else "共享单车"

        request_params: dict[str, Any] = {
            "key": self._api_key,
            "keywords": keywords,
            "types": "150600",  # 共享单车站点
            "city": city,
            "offset": 15,
            "page": 1,
            "extensions": "all",
        }
        if location:
            request_params["location"] = location

        response: httpx.Response = httpx.get(
            _AMAP_SEARCH_URL,
            params=request_params,
            timeout=10.0,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()

        if data.get("status") != "1":
            logger.error(f"[AmapBike] API 返回错误: {data.get('info', 'unknown')}")
            return []

        pois: list[dict[str, Any]] = data.get("pois", [])
        if not pois:
            logger.info("[AmapBike] 未找到共享单车站点")
            return []

        stations: list[dict[str, Any]] = []
        for poi in pois:
            s: dict[str, Any] = self._parse_bike_poi(poi)
            stations.append(s)

        return stations

    @staticmethod
    def _parse_bike_poi(poi: dict[str, Any]) -> dict[str, Any]:
        """解析高德共享单车站点 POI 数据为标准格式。"""
        name: str = poi.get("name", "未知单车站点")
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

        # 提取品牌信息
        biz_ext: dict[str, Any] = poi.get("biz_ext", {})
        rating_str: str = biz_ext.get("rating", "")

        rating: float = 0.0
        if rating_str:
            try:
                rating = float(rating_str)
            except (ValueError, TypeError):
                pass

        # 从 type 判断品牌
        poi_type: str = poi.get("type", "")
        features: list[str] = []

        # 尝试从名称或类型推断品牌
        name_lower: str = name.lower()
        if "美团" in name_lower or "meituan" in poi_type.lower():
            features.append("美团单车")
        if "青桔" in name_lower or "qingju" in poi_type.lower():
            features.append("青桔单车")
        if "哈啰" in name_lower or "哈罗" in name_lower or "hello" in poi_type.lower():
            features.append("哈啰单车")

        # 从 deep_info 提取更多信息
        deep_info: dict[str, Any] = poi.get("deep_info", {})
        if deep_info.get("business_area"):
            features.append(deep_info["business_area"])

        return {
            "name": name,
            "address": address or "未知地址",
            "longitude": longitude,
            "latitude": latitude,
            "distance_km": distance_km,
            "rating": rating,
            "features": features,
            "source": "amap_bike",
        }

    # ------------------------------------------------------------------
    # 降级策略
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_mock(params: dict[str, Any]) -> list[dict[str, Any]]:
        city: str = str(params.get("city", "北京"))
        return [
            {
                "name": f"{city}朝阳大悦城东门单车站",
                "address": f"{city}朝阳区朝阳北路101号东门",
                "longitude": 116.519,
                "latitude": 39.922,
                "distance_km": 0.1,
                "rating": 0.0,
                "features": ["美团单车", "青桔单车"],
                "source": "mock_bike",
            },
            {
                "name": f"{city}国贸地铁站C口单车站",
                "address": f"{city}朝阳区建国门外大街国贸地铁站C口",
                "longitude": 116.460,
                "latitude": 39.908,
                "distance_km": 0.3,
                "rating": 0.0,
                "features": ["哈啰单车", "美团单车"],
                "source": "mock_bike",
            },
            {
                "name": f"{city}SOHO现代城单车站",
                "address": f"{city}朝阳区建国路88号南侧",
                "longitude": 116.478,
                "latitude": 39.907,
                "distance_km": 0.45,
                "rating": 0.0,
                "features": ["青桔单车"],
                "source": "mock_bike",
            },
        ]
