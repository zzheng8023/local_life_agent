"""
高德地图公交路径规划工具 (Amap Transit Direction Tool)

本模块封装高德公交路径规划 API，提供真实的两点间公交换乘方案查询。
用于行程规划阶段，为"坐公交回家"等场景提供真实路线数据。

API 文档：
https://lbs.amap.com/api/webservice/guide/api/direction

核心设计：
- 调用高德 direction/transit/integrated API
- 返回结构化路线（含步行段、公交段、地铁段）
- 无 API Key 或调用失败时返回建议类兜底信息
- 非 ITool 接口（作为工具类使用，不注入工作流）
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from loguru import logger

_AMAP_TRANSIT_DIRECTION_URL: str = (
    "https://restapi.amap.com/v3/direction/transit/integrated"
)
_AMAP_GEOCODE_URL: str = "https://restapi.amap.com/v3/geocode/geo"


class AmapTransitDirectionTool:
    """高德公交路径规划工具。

    查询两点之间的公交换乘方案，返回可读的路线步骤。
    """

    def __init__(self, api_key: str | None = None, city: str = "北京") -> None:
        self._api_key: str = api_key or os.getenv("AMAP_API_KEY", "")
        self._city: str = city
        if self._api_key:
            logger.info(f"[AmapDirection] 初始化完成, city={city}")
        else:
            logger.warning(
                "[AmapDirection] 未配置 AMAP_API_KEY，路线规划将返回兜底建议"
            )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def geocode(self, address: str, city: str | None = None) -> str | None:
        """将中文地名 geocode 为 "lng,lat" 格式。

        调用高德地理编码 API，返回第一个匹配结果的坐标。
        失败（无 Key、API 错误、无结果）时返回 None。

        Args:
            address: 中文地点名称，如 "海淀区中关村"、"西城区金融街"。
            city: 城市限定（可选，默认使用构造时的 city）。

        Returns:
            "lng,lat" 字符串，如 "116.336,39.985"；失败返回 None。
        """
        if not self._api_key:
            return None
        search_city = city or self._city

        try:
            resp = httpx.get(
                _AMAP_GEOCODE_URL,
                params={
                    "key": self._api_key,
                    "address": address,
                    "city": search_city,
                },
                timeout=5.0,
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()

            if data.get("status") != "1":
                logger.warning(
                    f"[AmapDirection] geocode 失败: {data.get('info')} "
                    f"(address={address[:40]})"
                )
                return None

            geocodes: list[dict[str, Any]] = data.get("geocodes", [])
            if not geocodes:
                logger.warning(
                    f"[AmapDirection] geocode 无结果: address={address[:40]}"
                )
                return None

            location: str = geocodes[0].get("location", "")
            if location:
                logger.debug(
                    f"[AmapDirection] geocode 成功: {address[:30]} → {location}"
                )
                return location
            return None

        except Exception as exc:
            logger.warning(
                f"[AmapDirection] geocode 异常: {exc} "
                f"(address={address[:40]})"
            )
            return None

    def get_direction(
        self,
        origin: str,
        destination: str,
        city: str | None = None,
    ) -> dict[str, Any]:
        """查询两点间的公交换乘路线。

        高德 API 的 origin/destination 接受：
        - "lng,lat" 格式（最可靠，如 "116.4749,39.9087"）
        - 地点名称（如 "海淀区中关村"，但可能返回 INVALID_PARAMS）

        因此当传入非经纬度格式时，会尝试追加 city 参数辅助定位。

        Args:
            origin: 起点（经纬度 "lng,lat" 或地点名称）。
            destination: 终点（经纬度 "lng,lat" 或地点名称）。
            city: 城市（可选，默认使用构造时的 city）。

        Returns:
            {
                "success": bool,
                "routes": [{"duration": str, "distance": str, "segments": [...]}],
                "fallback_message": str | None,
            }
        """
        search_city: str = city or self._city

        if not self._api_key:
            logger.warning("[AmapDirection] 无 API Key，返回兜底建议")
            return self._fallback_suggestion(origin, destination)

        # ── 检测 origin/destination 是否为经纬度格式 ──
        _ll_pat = __import__("re").compile(
            r"^\s*-?\d{1,3}\.\d+,\s*-?\d{1,3}\.\d+\s*$"
        )
        origin_is_ll = bool(_ll_pat.match(origin))
        dest_is_ll = bool(_ll_pat.match(destination))

        # ── 非经纬度参数先 geocode 为坐标（地名 → "lng,lat"）──
        if not origin_is_ll:
            gc = self.geocode(origin, search_city)
            if gc:
                logger.info(f"[AmapDirection] geocode origin: {origin[:30]} → {gc}")
                origin = gc
                origin_is_ll = True
            else:
                logger.warning(
                    f"[AmapDirection] 无法 geocode origin: {origin[:40]}，使用原名尝试"
                )
        if not dest_is_ll:
            gc = self.geocode(destination, search_city)
            if gc:
                logger.info(
                    f"[AmapDirection] geocode destination: {destination[:30]} → {gc}"
                )
                destination = gc
                dest_is_ll = True
            else:
                logger.warning(
                    f"[AmapDirection] 无法 geocode destination: {destination[:40]}，使用原名尝试"
                )

        both_are_ll = origin_is_ll and dest_is_ll

        try:
            params: dict[str, Any] = {
                "key": self._api_key,
                "origin": origin,
                "destination": destination,
                "city": search_city,
                "strategy": 0,  # 最快捷模式
                "extensions": "base",
            }
            # ── 当 origin 或 destination 为非经纬度时，额外指定 city1/city2 ──
            if not both_are_ll:
                params["city1"] = search_city
                params["city2"] = search_city

            response: httpx.Response = httpx.get(
                _AMAP_TRANSIT_DIRECTION_URL,
                params=params,
                timeout=10.0,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()

            if data.get("status") != "1":
                err_info = data.get("info", "unknown")
                logger.error(
                    f"[AmapDirection] API 错误: {err_info} "
                    f"(origin={origin[:40]}, dest={destination[:40]})"
                )
                # 如果是 INVALID_PARAMS 且 origin/dest 含汉字，记录调试信息
                if "INVALID_PARAMS" in str(err_info):
                    logger.warning(
                        f"[AmapDirection] 参数无效，origin_is_ll={origin_is_ll}, "
                        f"dest_is_ll={dest_is_ll}"
                    )
                return self._fallback_suggestion(origin, destination)

            route_data: dict[str, Any] = data.get("route", {})
            if not route_data:
                logger.info("[AmapDirection] 无可用路线")
                return self._fallback_suggestion(origin, destination)

            # 提取第一条路线的关键信息
            transits: list[dict[str, Any]] = route_data.get("transits", [])
            if not transits:
                return self._fallback_suggestion(origin, destination)

            first_route: dict[str, Any] = transits[0]
            segments: list[dict[str, Any]] = []

            for seg in first_route.get("segments", []):
                bus_info: dict[str, Any] = seg.get("bus", {})
                walking_info: dict[str, Any] = seg.get("walking", {})

                if bus_info.get("buslines"):
                    for line in bus_info["buslines"]:
                        segments.append({
                            "type": line.get("type", "公交"),
                            "name": line.get("name", ""),
                            "departure_stop": line.get("departure_stop", {}).get(
                                "name", ""
                            ),
                            "arrival_stop": line.get("arrival_stop", {}).get(
                                "name", ""
                            ),
                            "duration": line.get("duration", ""),
                        })
                elif walking_info:
                    segments.append({
                        "type": "步行",
                        "name": "步行",
                        "distance": walking_info.get("distance", ""),
                        "duration": walking_info.get("duration", ""),
                    })

            logger.info(
                f"[AmapDirection] 查询成功: "
                f"duration={first_route.get('duration')}s, "
                f"{len(segments)} 段"
            )

            return {
                "success": True,
                "routes": [
                    {
                        "duration": first_route.get("duration", ""),
                        "distance": first_route.get("distance", ""),
                        "segments": segments,
                    }
                ],
                "fallback_message": None,
            }

        except Exception as exc:
            logger.error(f"[AmapDirection] API 调用失败: {exc}")
            return self._fallback_suggestion(origin, destination)

    # ------------------------------------------------------------------
    # 兜底
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_suggestion(origin: str, destination: str) -> dict[str, Any]:
        return {
            "success": False,
            "routes": [],
            "fallback_message": (
                f"建议使用高德地图或百度地图 App 查询从 {origin} 到 {destination} "
                f"的实时公交路线。"
            ),
        }
