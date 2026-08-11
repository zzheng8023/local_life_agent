"""
高德地图餐饮搜索工具 (Amap Restaurant Search Tool)

本模块实现基于高德地图 POI 搜索 API 的真实餐厅搜索能力。
高德开放平台对个人开发者提供每日 5000 次免费调用额度，
适合本地生活 Agent 的生产级使用。

接入步骤：
1. 前往 https://lbs.amap.com/ 注册开发者账号
2. 创建应用 → 添加 Key（服务平台选择 "Web服务"）
3. 将获取的 Key 填入 .env 的 AMAP_API_KEY

API 文档：
https://lbs.amap.com/api/webservice/guide/api/search

核心设计：
- 实现 ITool 接口，可被 LocalLifeWorkflow 直接注入使用
- 自动将高德 POI 数据转换为领域层标准的 Restaurant 格式
- 当 API Key 未配置或调用失败时，降级返回模拟数据
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from loguru import logger

from application.ports import ITool
from infrastructure.usage_tracker import UsageTracker

# 高德 POI 搜索 API 端点
_AMAP_SEARCH_URL: str = "https://restapi.amap.com/v3/place/text"

# 高德中餐 POI 类型代码映射
_CUISINE_TYPE_MAP: dict[str, str] = {
    "川菜": "川菜",
    "粤菜": "粤菜",
    "湘菜": "湘菜",
    "东北菜": "东北菜",
    "江浙菜": "江浙菜",
    "火锅": "火锅",
    "烧烤": "烧烤",
    "日料": "日本料理",
    "韩餐": "韩国料理",
    "西餐": "西餐",
    "自助餐": "自助餐",
    "海鲜": "海鲜",
    "茶餐厅": "茶餐厅",
    "小吃快餐": "小吃快餐",
    "咖啡厅": "咖啡厅",
    "面包甜点": "面包甜点",
}


class AmapRestaurantTool(ITool):
    """基于高德地图 POI 搜索的真实餐厅搜索工具。

    实现 ITool 接口，调用高德地图 Web API 搜索周边餐饮 POI，
    返回结构化餐厅数据供推荐引擎消费。

    Attributes:
        _api_key: 高德 Web 服务 API Key。
        _city: 默认搜索城市。
    """

    def __init__(self, api_key: str | None = None, city: str = "北京") -> None:
        """初始化高德地图搜索工具。

        Args:
            api_key: 高德 API Key，默认从环境变量 AMAP_API_KEY 读取。
            city: 默认搜索城市，如 "北京"、"上海"。
        """
        self._api_key: str = (
            api_key or os.getenv("AMAP_API_KEY", "")
        )
        self._city: str = city
        if self._api_key:
            logger.info(f"[Amap] 初始化完成, city={city}")
        else:
            logger.warning("[Amap] 未配置 AMAP_API_KEY，将使用模拟数据")

    # ------------------------------------------------------------------
    # ITool 接口实现
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return "amap_restaurant_search"

    def get_description(self) -> str:
        return (
            "基于高德地图 POI 数据搜索真实餐厅。"
            "参数：query(搜索关键词), taste(菜系口味), "
            "city(城市), radius(搜索半径米)。"
        )

    def execute(self, **kwargs: Any) -> list[dict[str, Any]]:
        """执行餐厅搜索。

        先尝试调用高德 API，失败或无 Key 时降级到模拟数据。

        Args:
            **kwargs: 搜索参数。

        Returns:
            标准化候选餐厅字典列表。
        """
        logger.info(f"[Amap] 搜索: {kwargs}")

        if not self._api_key:
            logger.warning("[Amap] 无 API Key，降级为模拟数据")
            return self._fallback_mock(kwargs)

        try:
            results: list[dict[str, Any]] = self._search_amap(kwargs)
            if results:
                UsageTracker().record_amap_call()
                logger.info(f"[Amap] 高德返回 {len(results)} 家餐厅")
                return results
        except Exception as exc:
            logger.error(f"[Amap] API 调用失败: {exc}")

        logger.warning("[Amap] 降级为模拟数据")
        return self._fallback_mock(kwargs)

    # ------------------------------------------------------------------
    # 高德 API 调用
    # ------------------------------------------------------------------

    def _search_amap(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """调用高德地图 POI 搜索 API。

        Args:
            params: 搜索参数。

        Returns:
            标准化餐厅列表。
        """
        city: str = str(
            params.get("city") or self._city
        )
        taste: str = str(params.get("taste", ""))
        raw_query: str = str(params.get("query", ""))
        budget: str = str(params.get("budget", ""))

        # 构建搜索关键词
        keywords: str = self._build_keywords(taste, raw_query)

        # 调用高德 API
        request_params: dict[str, Any] = {
            "key": self._api_key,
            "keywords": keywords,
            "types": "050000",  # 餐饮大类
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
            error_info: str = data.get("info", "unknown error")
            logger.error(f"[Amap] API 返回错误: {error_info}")
            return []

        pois: list[dict[str, Any]] = data.get("pois", [])
        if not pois:
            logger.info("[Amap] 未找到匹配餐厅")
            return []

        # 转换为领域标准格式
        restaurants: list[dict[str, Any]] = []
        for poi in pois:
            r: dict[str, Any] = self._parse_poi(poi, budget)
            if r:
                restaurants.append(r)

        return restaurants

    @staticmethod
    def _build_keywords(taste: str, raw_query: str) -> str:
        """根据用户偏好构建高德搜索关键词。

        Args:
            taste: 口味/菜系偏好（LLM 提取的文本或空）。
            raw_query: 原始查询文本。

        Returns:
            高德 POI 搜索关键词字符串（≤8 字，不含标点）。
        """
        # 优先映射菜系关键词 → 短词
        if taste:
            taste_clean = str(taste).replace("\n", " ").strip()
            # 尝试直接匹配 _CUISINE_TYPE_MAP
            mapped = _CUISINE_TYPE_MAP.get(taste_clean, "")
            if mapped:
                return mapped
            # taste 可能是长文本（如 LLM 生成的复合描述 "兼顾辣与清淡的综合性菜系..."）
            # 遍历已知菜系，取第一个命中的关键词
            for cuisine, kw in _CUISINE_TYPE_MAP.items():
                if cuisine in taste_clean:
                    return kw
            # 手动截断：取前 6 个中文字符
            import re as _re
            chinese_chars = _re.findall(r"[一-鿿]", taste_clean)
            if chinese_chars:
                short = "".join(chinese_chars[:6])
                if len(short) >= 2:
                    return short
            # 回退：用 raw_query 提取菜系
            if raw_query:
                for cuisine in _CUISINE_TYPE_MAP:
                    if cuisine in raw_query:
                        return cuisine

        # 从原始查询中提取关键词
        for cuisine in _CUISINE_TYPE_MAP:
            if cuisine in raw_query:
                return cuisine

        return "餐厅"

    @staticmethod
    def _parse_poi(
        poi: dict[str, Any], budget: str
    ) -> dict[str, Any] | None:
        """将高德 POI 数据转换为领域标准 Restaurant 格式。

        Args:
            poi: 高德 POI 原始数据。
            budget: 用户预算（用于过滤）。

        Returns:
            标准化的餐厅字典，预算不匹配时返回 None。
        """
        name: str = poi.get("name", "未知餐厅")
        address: str = poi.get("address", "")
        location: str = poi.get("location", "")

        # 解析坐标
        longitude: float | None = None
        latitude: float | None = None
        if location and "," in location:
            try:
                parts = location.split(",")
                longitude = float(parts[0])
                latitude = float(parts[1])
            except (ValueError, IndexError):
                pass

        # 解析距离
        distance_str: str = poi.get("distance", "")
        distance_km: float | None = None
        if distance_str:
            try:
                distance_km = round(float(distance_str) / 1000, 2)
            except (ValueError, TypeError):
                pass

        # 扩展信息（extensions=all 时返回）
        biz_ext: dict[str, Any] = poi.get("biz_ext", {})
        rating_str: str = biz_ext.get("rating", "")
        cost_str: str = biz_ext.get("cost", "")

        # 评分
        rating: float = 0.0
        if rating_str:
            try:
                rating = float(rating_str)
            except (ValueError, TypeError):
                pass

        # 人均价格
        avg_price: str | None = None
        if cost_str:
            try:
                cost_val: float = float(cost_str)
                avg_price = f"人均{cost_val:.0f}元"
            except (ValueError, TypeError):
                avg_price = f"人均{cost_str}元"

        # 预算过滤
        if budget and "人均" in budget and avg_price:
            try:
                import re

                budget_match = re.search(r"(\d+)", budget)
                price_match = re.search(r"(\d+)", avg_price)
                if budget_match and price_match:
                    budget_val: int = int(budget_match.group(1))
                    price_val: int = int(price_match.group(1))
                    if price_val > budget_val * 1.3:
                        return None
            except (ValueError, AttributeError):
                pass

        # 菜系从分类推断
        poi_type: str = poi.get("type", "")
        cuisine: str = ""
        type_parts: list[str] = poi_type.split(";")
        if len(type_parts) >= 2:
            cuisine = type_parts[-1]

        # 特色标签
        features: list[str] = []
        photos: list[dict[str, Any]] = poi.get("photos", [])
        if photos:
            features.append("有图片")

        # 从 biz_ext 提取额外信息
        if biz_ext.get("open_time"):
            features.append("有营业时间")

        tel: str = poi.get("tel", "")
        if tel:
            features.append(f"电话:{tel}")

        return {
            "name": name,
            "rating": rating,
            "avg_price": avg_price or "人均未知",
            "cuisine": cuisine or "其他",
            "address": address or "未知地址",
            "distance_km": distance_km,
            "longitude": longitude,
            "latitude": latitude,
            "features": features,
            "source": "amap",
        }

    # ------------------------------------------------------------------
    # 降级策略：无 API Key 时使用的模拟数据
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_mock(params: dict[str, Any]) -> list[dict[str, Any]]:
        """无 API Key 或调用失败时的模拟数据降级。

        复用 web_tools.RestaurantSearchTool 中的模拟逻辑。

        Args:
            params: 搜索参数。

        Returns:
            模拟候选餐厅列表。
        """
        from infrastructure.web_tools import RestaurantSearchTool

        mock_tool: RestaurantSearchTool = RestaurantSearchTool()
        return mock_tool.execute(**params)
