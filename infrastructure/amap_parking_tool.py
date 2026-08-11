"""
高德地图停车场搜索工具 (Amap Parking Search Tool)

本模块实现基于高德地图 POI 搜索 API 的停车场查询。
用于"开车去吃饭，附近有地方停车吗"等场景。

核心设计：
- 实现 ITool 接口，遵循 try-real-catch-fallback 三级模式
- types=150900（停车场大类）
- 返回停车场名称、地址、距离、空位状态等
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


class AmapParkingTool(ITool):
    """基于高德地图 POI 搜索的停车场查询工具。

    实现 ITool 接口，调用高德 place/text API 搜索停车场 POI。
    搜索类型码 types=150900（停车场），覆盖路边停车、地下车库等。
    """

    def __init__(self, api_key: str | None = None, city: str = "北京") -> None:
        self._api_key: str = api_key or os.getenv("AMAP_API_KEY", "")
        self._city: str = city
        if self._api_key:
            logger.info(f"[AmapParking] 初始化完成, city={city}")
        else:
            logger.warning("[AmapParking] 未配置 AMAP_API_KEY，将使用模拟数据")

    # ------------------------------------------------------------------
    # ITool 接口实现
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return "amap_parking_search"

    def get_description(self) -> str:
        return (
            "基于高德地图 POI 数据搜索附近停车场。"
            "参数：query(搜索关键词，如 '停车场'), city(城市), "
            "location(中心点坐标，格式 'lng,lat')。"
        )

    def execute(self, **kwargs: Any) -> list[dict[str, Any]]:
        logger.info(f"[AmapParking] 搜索: {kwargs}")

        if not self._api_key:
            logger.warning("[AmapParking] 无 API Key，降级为模拟数据")
            return self._fallback_mock(kwargs)

        try:
            results: list[dict[str, Any]] = self._search_amap_parking(kwargs)
            if results:
                UsageTracker().record_amap_call()
                logger.info(f"[AmapParking] 高德返回 {len(results)} 个停车场")
                return results
        except Exception as exc:
            logger.error(f"[AmapParking] API 调用失败: {exc}")

        logger.warning("[AmapParking] 降级为模拟数据")
        return self._fallback_mock(kwargs)

    # ------------------------------------------------------------------
    # 高德 API 调用
    # ------------------------------------------------------------------

    def _search_amap_parking(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        city: str = str(params.get("city") or self._city)
        raw_query: str = str(params.get("query", ""))
        location: str = str(params.get("location", ""))

        keywords: str = raw_query if raw_query else "停车场"

        request_params: dict[str, Any] = {
            "key": self._api_key,
            "keywords": keywords,
            "types": "150900",  # 停车场大类
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
            logger.error(f"[AmapParking] API 返回错误: {data.get('info', 'unknown')}")
            return []

        pois: list[dict[str, Any]] = data.get("pois", [])
        if not pois:
            logger.info("[AmapParking] 未找到停车场")
            return []

        parking_lots: list[dict[str, Any]] = []
        for poi in pois:
            p: dict[str, Any] = self._parse_parking_poi(poi)
            parking_lots.append(p)

        return parking_lots

    @staticmethod
    def _parse_parking_poi(poi: dict[str, Any]) -> dict[str, Any]:
        """解析高德停车场 POI 数据为标准格式。"""
        name: str = poi.get("name", "未知停车场")
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

        # 提取停车场特性
        biz_ext: dict[str, Any] = poi.get("biz_ext", {})
        parking_type: str = biz_ext.get("parking_type", "")
        rating_str: str = biz_ext.get("rating", "")

        rating: float = 0.0
        if rating_str:
            try:
                rating = float(rating_str)
            except (ValueError, TypeError):
                pass

        # 从 type 字段判断停车场类型
        poi_type: str = poi.get("type", "")
        features: list[str] = []

        if "地下" in poi_type or "地下" in name:
            features.append("地下车库")
        if "路边" in poi_type or "路边" in name:
            features.append("路边停车")
        if "室内" in poi_type:
            features.append("室内停车场")
        if parking_type:
            features.append(parking_type)

        # 从 deep_info 提取更多信息
        deep_info: dict[str, Any] = poi.get("deep_info", {})
        if deep_info.get("business_area"):
            features.append(deep_info["business_area"])

        # 费用信息
        cost: str = biz_ext.get("cost", "")
        if cost:
            features.append(f"参考费用:{cost}")

        return {
            "name": name,
            "address": address or "未知地址",
            "longitude": longitude,
            "latitude": latitude,
            "distance_km": distance_km,
            "rating": rating,
            "features": features,
            "source": "amap_parking",
        }

    # ------------------------------------------------------------------
    # 降级策略
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_mock(params: dict[str, Any]) -> list[dict[str, Any]]:
        city: str = str(params.get("city", "北京"))
        return [
            {
                "name": f"{city}朝阳大悦城地下停车场",
                "address": f"{city}朝阳区朝阳北路101号",
                "longitude": 116.518,
                "latitude": 39.921,
                "distance_km": 0.15,
                "rating": 4.2,
                "features": ["地下车库", "24小时", "参考费用:6元/小时"],
                "source": "mock_parking",
            },
            {
                "name": f"{city}国贸商城停车场",
                "address": f"{city}朝阳区建国门外大街1号",
                "longitude": 116.461,
                "latitude": 39.909,
                "distance_km": 0.35,
                "rating": 4.0,
                "features": ["室内停车场", "参考费用:10元/小时"],
                "source": "mock_parking",
            },
            {
                "name": f"{city}SOHO现代城路边停车场",
                "address": f"{city}朝阳区建国路88号",
                "longitude": 116.479,
                "latitude": 39.908,
                "distance_km": 0.5,
                "rating": 3.5,
                "features": ["路边停车", "参考费用:首小时8元"],
                "source": "mock_parking",
            },
        ]
