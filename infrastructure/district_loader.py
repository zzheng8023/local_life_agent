"""
高德行政区域加载器 (Amap District Loader)

本模块封装高德地图行政区域查询 API，提供省-市-县（区）三级
级联数据的加载与缓存能力，供前端下拉菜单使用。

API 文档：
https://lbs.amap.com/api/webservice/guide/api/district

核心设计：
- 首次启动时预加载全部省份列表。
- 用户选择省份后按需加载城市、区县，结果缓存于内存中。
- API Key 未配置时返回最小兜底数据，确保界面不崩溃。
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from loguru import logger

_AMAP_DISTRICT_URL: str = "https://restapi.amap.com/v3/config/district"


class DistrictLoader:
    """高德行政区域级联数据加载器。

    封装高德 district API，提供省→市→区三级联动的选项数据。
    所有 API 调用结果均缓存于内存中，避免重复请求。

    Attributes:
        _api_key: 高德 Web 服务 API Key。
        _cache: {adcode_or_keyword: [{"name": ..., "adcode": ...}, ...]} 缓存。
    """

    def __init__(self, api_key: str | None = None) -> None:
        """初始化区域加载器。

        Args:
            api_key: 高德 API Key，默认从环境变量 AMAP_API_KEY 读取。
        """
        self._api_key: str = (
            api_key or os.getenv("AMAP_API_KEY", "")
        )
        self._cache: dict[str, list[dict[str, str]]] = {}

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def get_provinces(self) -> list[str]:
        """获取全国省份列表。

        Returns:
            省份名称列表，如 ["北京市", "上海市", "广东省", ...]。
        """
        entries: list[dict[str, str]] = self._fetch_district(
            keywords="中国",
            subdistrict=1,
        )
        return [e["name"] for e in entries]

    def get_cities(self, province: str) -> list[str]:
        """获取指定省份下的城市列表。

        Args:
            province: 省份名称，如 "广东省"。

        Returns:
            城市名称列表，如 ["广州市", "深圳市", "珠海市", ...]。
        """
        entries: list[dict[str, str]] = self._fetch_district(
            keywords=province,
            subdistrict=1,
        )
        return [e["name"] for e in entries]

    def get_districts(self, city: str) -> list[str]:
        """获取指定城市下的区县列表。

        Args:
            city: 城市名称，如 "深圳市"。

        Returns:
            区县名称列表，如 ["南山区", "福田区", "罗湖区", ...]。
        """
        entries: list[dict[str, str]] = self._fetch_district(
            keywords=city,
            subdistrict=1,
        )
        return [e["name"] for e in entries]

    # ------------------------------------------------------------------
    # 底层 API 调用与缓存
    # ------------------------------------------------------------------

    def _fetch_district(
        self, keywords: str, subdistrict: int
    ) -> list[dict[str, str]]:
        """从高德 API 获取行政区域数据（带缓存）。

        Args:
            keywords: 查询关键词（如 "中国"、"广东省"）。
            subdistrict: 子级行政区层级深度（0/1/2/3）。

        Returns:
            [{"name": "名称", "adcode": "编码"}, ...] 列表。
        """
        cache_key: str = f"{keywords}:{subdistrict}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self._api_key:
            logger.warning(f"[District] 无 API Key，返回兜底数据: {keywords}")
            fallback: list[dict[str, str]] = self._fallback(keywords)
            self._cache[cache_key] = fallback
            return fallback

        try:
            response: httpx.Response = httpx.get(
                _AMAP_DISTRICT_URL,
                params={
                    "key": self._api_key,
                    "keywords": keywords,
                    "subdistrict": subdistrict,
                    "extensions": "base",
                },
                timeout=8.0,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()

            if data.get("status") != "1":
                logger.error(f"[District] API 错误: {data.get('info')}")
                fallback = self._fallback(keywords)
                self._cache[cache_key] = fallback
                return fallback

            districts: list[dict[str, Any]] = data.get("districts", [])
            if not districts:
                fallback = self._fallback(keywords)
                self._cache[cache_key] = fallback
                return fallback

            # 目标为 subdistrict 下一级的列表
            children: list[dict[str, Any]] = districts[0].get("districts", [])
            if not children:
                # 没有子级（如直辖市区的下一级），返回自身
                result: list[dict[str, str]] = [
                    {"name": districts[0]["name"], "adcode": districts[0].get("adcode", "")}
                ]
            else:
                result = [
                    {"name": c["name"], "adcode": c.get("adcode", "")}
                    for c in children
                ]

            self._cache[cache_key] = result
            logger.info(
                f"[District] 获取 {keywords} 下级: {len(result)} 条"
            )
            return result

        except Exception as exc:
            logger.error(f"[District] 请求失败: {exc}")
            fallback = self._fallback(keywords)
            self._cache[cache_key] = fallback
            return fallback

    # ------------------------------------------------------------------
    # 兜底数据 — 无 API Key 时使用
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback(keywords: str) -> list[dict[str, str]]:
        """返回最小兜底数据集，确保界面不会崩溃。

        Args:
            keywords: 查询关键词。

        Returns:
            兜底区域数据。
        """
        if keywords == "中国":
            return [
                {"name": "北京市", "adcode": "110000"},
                {"name": "上海市", "adcode": "310000"},
                {"name": "广东省", "adcode": "440000"},
                {"name": "浙江省", "adcode": "330000"},
                {"name": "四川省", "adcode": "510000"},
                {"name": "湖北省", "adcode": "420000"},
            ]
        if keywords == "北京市":
            return [
                {"name": "东城区", "adcode": "110101"},
                {"name": "西城区", "adcode": "110102"},
                {"name": "朝阳区", "adcode": "110105"},
                {"name": "海淀区", "adcode": "110108"},
                {"name": "丰台区", "adcode": "110106"},
                {"name": "通州区", "adcode": "110112"},
                {"name": "大兴区", "adcode": "110115"},
            ]
        if keywords == "北京市区":
            return [
                {"name": "东城区", "adcode": "110101"},
                {"name": "西城区", "adcode": "110102"},
                {"name": "朝阳区", "adcode": "110105"},
                {"name": "海淀区", "adcode": "110108"},
            ]
        if keywords in ("上海市", "天津市"):
            return [{"name": keywords.replace("市", "市"), "adcode": "000000"}]
        if keywords in ("广东省", "浙江省", "四川省", "湖北省"):
            prefix: str = keywords[:2]
            return [
                {"name": f"{prefix}省会城市", "adcode": "000000"},
                {"name": f"{prefix}城市2", "adcode": "000000"},
                {"name": f"{prefix}城市3", "adcode": "000000"},
            ]
        return [
            {"name": f"{keywords}区1", "adcode": "000000"},
            {"name": f"{keywords}区2", "adcode": "000000"},
        ]
