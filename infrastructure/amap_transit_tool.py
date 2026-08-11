"""
高德地图交通站点搜索工具 (Amap Transit Search Tool)

本模块实现基于高德地图 POI 搜索 API 的真实公交/地铁站搜索。
搜索类型码：150700（公交车站）+ 150500（地铁站）。

核心设计：
- 实现 ITool 接口，遵循 try-real-catch-fallback 三级模式
- 同时搜索公交站和地铁站，合并返回结果
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

# 交通站点类型码（公交 + 地铁）
_TRANSIT_TYPE_CODES: str = "150700|150500"


class AmapTransitTool(ITool):
    """基于高德地图 POI 搜索的真实公交/地铁站搜索工具。

    同时搜索公交车站 (150700) 和地铁站 (150500)，
    合并返回标准化的交通站点列表。
    """

    def __init__(self, api_key: str | None = None, city: str = "北京") -> None:
        self._api_key: str = api_key or os.getenv("AMAP_API_KEY", "")
        self._city: str = city
        if self._api_key:
            logger.info(f"[AmapTransit] 初始化完成, city={city}")
        else:
            logger.warning("[AmapTransit] 未配置 AMAP_API_KEY，将使用模拟数据")

    # ------------------------------------------------------------------
    # ITool 接口实现
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return "amap_transit_search"

    def get_description(self) -> str:
        return (
            "基于高德地图 POI 数据搜索真实公交站和地铁站。"
            "参数：query(搜索关键词), city(城市)。"
        )

    def execute(self, **kwargs: Any) -> list[dict[str, Any]]:
        logger.info(f"[AmapTransit] 搜索: {kwargs}")

        if not self._api_key:
            logger.warning("[AmapTransit] 无 API Key，降级为模拟数据")
            return self._fallback_mock(kwargs)

        try:
            results: list[dict[str, Any]] = self._search_amap_transit(kwargs)
            if results:
                UsageTracker().record_amap_call()
                logger.info(f"[AmapTransit] 高德返回 {len(results)} 个交通站点")
                return results
        except Exception as exc:
            logger.error(f"[AmapTransit] API 调用失败: {exc}")

        logger.warning("[AmapTransit] 降级为模拟数据")
        return self._fallback_mock(kwargs)

    # ------------------------------------------------------------------
    # 高德 API 调用
    # ------------------------------------------------------------------

    def _search_amap_transit(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        city: str = str(params.get("city") or self._city)
        raw_query: str = str(params.get("query", ""))

        # 根据 query 智能选择优先搜索类型
        if raw_query and "公交" in raw_query:
            keywords: str = "公交站"
        elif raw_query and "地铁" in raw_query:
            keywords = "地铁站"
        else:
            keywords = raw_query if raw_query else "地铁站"

        request_params: dict[str, Any] = {
            "key": self._api_key,
            "keywords": keywords,
            "types": _TRANSIT_TYPE_CODES,
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
            logger.error(f"[AmapTransit] API 返回错误: {data.get('info', 'unknown')}")
            return []

        pois: list[dict[str, Any]] = data.get("pois", [])
        if not pois:
            logger.info("[AmapTransit] 未找到匹配交通站点")
            return []

        stops: list[dict[str, Any]] = []
        for poi in pois:
            s: dict[str, Any] = self._parse_transit_poi(poi)
            if s:
                stops.append(s)

        # ── v3.1: 补全途经线路 ──
        self._enrich_lines(stops, city)

        return stops

    @staticmethod
    def _parse_transit_poi(poi: dict[str, Any]) -> dict[str, Any]:
        name: str = poi.get("name", "未知站点")
        address: str = poi.get("address", "")

        distance_str: str = poi.get("distance", "")
        distance_km: float | None = None
        if distance_str:
            try:
                distance_km = round(float(distance_str) / 1000, 2)
            except (ValueError, TypeError):
                pass

        # 从 POI 类型推断类别
        poi_type: str = poi.get("type", "")
        if "地铁" in poi_type:
            category: str = "地铁站"
        elif "公交" in poi_type:
            category = "公交站"
        else:
            category = "公交站"  # 默认

        # 坐标信息
        location: str = poi.get("location", "")
        longitude: float | None = None
        latitude: float | None = None
        if location and "," in location:
            try:
                parts: list[str] = location.split(",")
                longitude = float(parts[0])
                latitude = float(parts[1])
            except (ValueError, IndexError):
                pass

        return {
            "name": name,
            "category": category,
            "address": address or "未知地址",
            "distance_km": distance_km,
            "longitude": longitude,
            "latitude": latitude,
            "lines": [],  # 后续由 _enrich_lines 填充
            "source": "amap_transit",
        }

    def _enrich_lines(
        self, stops: list[dict[str, Any]], city: str | None = None
    ) -> None:
        """为前 5 个站点调用高德公交站点查询 API 补全途经线路。

        调用 https://restapi.amap.com/v3/bus/stopname 接口，
        提取返回的 buslines[].name 填入每个站点的 lines 列表。
        API 失败或超时时静默回退到占位文本。

        Args:
            stops: _parse_transit_poi 产出的站点列表（会被原地修改）。
            city: 城市名称。
        """
        import json as _json

        search_city: str = city or self._city
        enriched: int = 0

        for stop in stops[:5]:
            stop_name: str = stop.get("name", "")
            if not stop_name or not self._api_key:
                break

            try:
                resp: httpx.Response = httpx.get(
                    "https://restapi.amap.com/v3/bus/stopname",
                    params={
                        "key": self._api_key,
                        "keywords": stop_name,
                        "city": search_city,
                    },
                    timeout=8.0,
                )
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()

                if data.get("status") != "1":
                    continue

                bus_stops: list[dict[str, Any]] = data.get("busstops", [])
                if not bus_stops:
                    continue

                # 取第一个匹配站点，收集途经线路名
                lines_set: set[str] = set()
                for bs in bus_stops:
                    for bline in bs.get("buslines", []):
                        line_name: str = bline.get("name", "")
                        if line_name:
                            lines_set.add(line_name)

                if lines_set:
                    # 保持线路名可读排序
                    stop["lines"] = sorted(lines_set)
                    enriched += 1

            except Exception as exc:
                logger.debug(
                    f"[AmapTransit] 线路补全失败 [{stop_name}]: {exc}"
                )
                continue

        # 未补全的站点 → 占位文本
        for stop in stops:
            if not stop.get("lines"):
                stop["lines"] = ["请使用地图App查询具体线路"]

        if enriched:
            logger.info(
                f"[AmapTransit] 线路补全: {enriched}/{min(len(stops), 5)} 个站点"
            )

    # ------------------------------------------------------------------
    # 降级策略
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_mock(params: dict[str, Any]) -> list[dict[str, Any]]:
        from infrastructure.web_tools import HotelAndEntertainmentSearchTool
        return HotelAndEntertainmentSearchTool._mock_transit_stops(params)
