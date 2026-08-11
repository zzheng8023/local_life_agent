"""
Web 工具集 (Web Tools) — v2.0

本模块实现应用层定义的 ITool 抽象接口，提供 Agent 与外部互联网交互的具体能力。

v2.0 新增：
- RestaurantSearchTool Mock 数据扩展：rating、queue_status、has_group_buy、distance。
- HotelAndEntertainmentSearchTool：模拟检索附近的酒店和电影院排片情况。
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from application.ports import ITool


class RestaurantSearchTool(ITool):
    """餐厅搜索工具。

    v2.0 扩展 Mock 数据，新增 queue_status（排队情况）、has_group_buy（团购券）、
    精确 distance 字段，支持更丰富的推荐推理。
    """

    def get_name(self) -> str:
        return "restaurant_search"

    def get_description(self) -> str:
        return (
            "搜索符合条件的餐厅。"
            "参数：query(搜索关键词), budget(预算), taste(口味), "
            "distance(距离约束), city(城市), features(特色标签列表)。"
        )

    def execute(self, **kwargs: Any) -> list[dict[str, Any]]:
        logger.info(f"[Tool] restaurant_search called, params: {kwargs}")
        return self._mock_search(kwargs)

    # ================================================================
    # Mock 数据（v2.0 扩展字段）
    # ================================================================

    @staticmethod
    def _mock_search(params: dict[str, Any]) -> list[dict[str, Any]]:
        """模拟餐厅搜索，返回带详尽业务字段的候选列表。

        v2.0 新增字段：
        - queue_status：当前排队情况（"无需排队" / "约等15分钟" / "约等30分钟+" / "需提前预约"）
        - has_group_buy：是否有团购券 / 优惠套餐 (bool)
        - distance_km：距搜索位置精确距离（公里）
        - distance_to_hotel：到最近推荐酒店的步行距离（公里），用于时空推理
        - nearby_entertainment：附近步行可达的娱乐场所列表
        """
        taste: str = str(params.get("taste", ""))
        restrictions: str = str(params.get("restrictions", ""))
        budget: str = str(params.get("budget", ""))
        need_parking: bool = bool(params.get("need_parking", False))
        features_list: list[str] = params.get("features", []) or []

        # 合并 taste + restrictions 作为完整的口味/忌口描述
        combined_taste: str = taste
        if restrictions and restrictions != "None":
            if combined_taste:
                combined_taste = f"{combined_taste}, {restrictions}"
            else:
                combined_taste = restrictions

        all_restaurants: list[dict[str, Any]] = [
            {
                "name": "川味观·臻选",
                "rating": 4.6,
                "avg_price": "人均140元",
                "cuisine": "川菜",
                "address": "朝阳区建国路88号SOHO现代城B座3层",
                "distance_km": 2.1,
                "longitude": 116.4749,
                "latitude": 39.9087,
                "queue_status": "约等15分钟",
                "has_group_buy": True,
                "features": ["有包间", "免费停车", "辣味正宗", "大厅有投影"],
                "distance_to_hotel": 0.3,
                "nearby_entertainment": ["万达影城(国贸店)", "纯K(建国路店)"],
                "source": "mock",
            },
            {
                "name": "鼎泰丰(国贸店)",
                "rating": 4.5,
                "avg_price": "人均120元",
                "cuisine": "江浙菜/小笼包",
                "address": "朝阳区建国门外大街1号国贸商城北区4层",
                "distance_km": 1.8,
                "longitude": 116.4612,
                "latitude": 39.9088,
                "queue_status": "无需排队",
                "has_group_buy": False,
                "features": ["有包间", "清淡不辣", "亲子友好", "地铁直达"],
                "distance_to_hotel": 0.6,
                "nearby_entertainment": ["英皇电影城(国贸商城店)", "pageone书店"],
                "source": "mock",
            },
            {
                "name": "大董烤鸭(工体店)",
                "rating": 4.7,
                "avg_price": "人均180元",
                "cuisine": "京菜/烤鸭",
                "address": "朝阳区工人体育场东路甲6号",
                "distance_km": 3.5,
                "longitude": 116.4506,
                "latitude": 39.9322,
                "queue_status": "需提前预约",
                "has_group_buy": True,
                "features": ["有包间", "商务宴请", "环境优雅", "代客泊车"],
                "distance_to_hotel": 1.2,
                "nearby_entertainment": ["工人体育馆", "三里屯太古里(影院)"],
                "source": "mock",
            },
            {
                "name": "海底捞火锅(望京店)",
                "rating": 4.4,
                "avg_price": "人均130元",
                "cuisine": "火锅",
                "address": "朝阳区望京街9号望京国际商业中心3层",
                "distance_km": 5.2,
                "longitude": 116.4803,
                "latitude": 39.9983,
                "queue_status": "约等30分钟+",
                "has_group_buy": False,
                "features": ["包间预约", "免费停车", "儿童乐园", "辣/不辣可选"],
                "distance_to_hotel": 0.8,
                "nearby_entertainment": ["CGV影城(望京店)", "望京SOHO音乐喷泉"],
                "source": "mock",
            },
            {
                "name": "绿茶餐厅(三里屯店)",
                "rating": 4.2,
                "avg_price": "人均75元",
                "cuisine": "创意融合菜",
                "address": "朝阳区三里屯路19号太古里南区B1",
                "distance_km": 1.3,
                "longitude": 116.4544,
                "latitude": 39.9332,
                "queue_status": "约等15分钟",
                "has_group_buy": True,
                "features": ["性价比高", "环境网红风", "无包间", "需排队"],
                "distance_to_hotel": 1.5,
                "nearby_entertainment": ["美嘉欢乐影城(三里屯店)", "三里屯纯K"],
                "source": "mock",
            },
            {
                "name": "鮨·寿司大(亮马桥店)",
                "rating": 4.8,
                "avg_price": "人均350元",
                "cuisine": "日料/Omakase",
                "address": "朝阳区亮马桥路48号燕莎友谊商城5层",
                "distance_km": 2.6,
                "longitude": 116.4665,
                "latitude": 39.9498,
                "queue_status": "需提前预约",
                "has_group_buy": False,
                "features": ["需预约", "无包间", "板前料理", "情侣约会"],
                "distance_to_hotel": 0.5,
                "nearby_entertainment": ["燕莎友谊商城", "蓝色港湾(影院)"],
                "source": "mock",
            },
        ]

        # ── 过滤逻辑 ──
        filtered: list[dict[str, Any]] = []

        for r in all_restaurants:
            # 预算过滤
            if budget and "人均" in budget:
                try:
                    price_str: str = r.get("avg_price", "")
                    match = re.search(r"(\d+)", price_str)
                    budget_match = re.search(r"(\d+)", budget)
                    if match and budget_match:
                        price_val: int = int(match.group(1))
                        budget_val: int = int(budget_match.group(1))
                        if price_val > budget_val * 1.3:
                            continue
                except (ValueError, AttributeError):
                    pass

            # 口味与忌口过滤（combined_taste = taste + restrictions）
            if combined_taste:
                r_taste: str = (
                    r.get("cuisine", "") + " " + " ".join(r.get("features", []))
                )
                t: str = combined_taste.lower()
                rt: str = r_taste.lower()

                # 先检查忌口排除（最高优先级）
                blocked = False
                # 忌辣 → 排除辣味正宗
                if any(kw in t for kw in ["忌辣", "不辣", "不吃辣", "清淡", "不喜辣"]):
                    if "辣味正宗" in rt:
                        blocked = True
                # 忌海鲜 → 排除海鲜类
                if any(kw in t for kw in ["海鲜", "不吃海鲜", "忌海鲜"]):
                    if any(kw in rt for kw in ["海鲜", "日料", "鮨", "寿司"]):
                        blocked = True
                if blocked:
                    continue

                # 再检查正面偏好匹配
                # 用户偏好辣 → 优先匹配辣/川/火锅/湘
                if any(kw in t for kw in ["辣", "川", "火锅", "湘"]) and not any(
                    kw in t for kw in ["忌辣", "不辣", "不吃辣"]
                ):
                    # 有辣偏好但没忌辣，属于正面偏好
                    pass  # 不因为缺少辣而过滤，只是偏好而已

            # 包间过滤
            if "有包间" in features_list:
                if "有包间" not in r.get("features", []):
                    continue

            # 停车过滤
            if need_parking:
                if not any(
                    kw in r.get("features", [])
                    for kw in ["免费停车", "代客泊车", "停车位"]
                ):
                    continue

            filtered.append(r)

        # 兜底：过滤结果太少时返回更多候选
        if len(filtered) < 3:
            filtered = all_restaurants[:4]

        return filtered


# ================================================================
# v2.0 新增：酒店 + 娱乐场所联合搜索
# ================================================================

class HotelAndEntertainmentSearchTool(ITool):
    """酒店与娱乐场所联合搜索工具。

    支持两个子模式：
    - mode="hotel"：检索附近酒店。
    - mode="entertainment"：检索附近电影院/娱乐场所及排片信息。

    返回结构：
    - mode="hotel" → list[dict] 酒店列表
    - mode="entertainment" → list[dict] 娱乐场所（含电影院排片）
    """

    def get_name(self) -> str:
        return "hotel_entertainment_search"

    def get_description(self) -> str:
        return (
            "检索附近的酒店、娱乐场所、商场和公交站点。"
            "参数：mode（'hotel' / 'entertainment' / 'shopping' / 'transit'）、"
            "query（搜索关键词）、budget（预算）、distance（距离约束）、city（城市）。"
        )

    def execute(self, **kwargs: Any) -> list[dict[str, Any]]:
        mode: str = str(kwargs.get("mode", "hotel"))
        logger.info(
            f"[Tool] hotel_entertainment_search called, mode={mode}, params={kwargs}"
        )
        if mode == "hotel":
            return self._mock_hotels(kwargs)
        elif mode == "entertainment":
            return self._mock_entertainments(kwargs)
        elif mode == "shopping":
            return self._mock_shopping(kwargs)
        elif mode == "transit":
            return self._mock_transit_stops(kwargs)
        else:
            return self._mock_hotels(kwargs) + self._mock_entertainments(kwargs)

    # ── 酒店 Mock 数据 ──

    @staticmethod
    def _mock_hotels(params: dict[str, Any]) -> list[dict[str, Any]]:
        """模拟酒店搜索结果。"""
        _ = params  # 保留扩展空间
        return [
            {
                "name": "全季酒店(国贸店)",
                "rating": 4.5,
                "avg_price": "358元起/晚",
                "address": "朝阳区建国路93号万达广场旁",
                "distance_to_restaurant": 0.3,
                "features": ["免费停车", "含早餐", "商务大床房", "交通便利"],
                "queue_status": "有房",
                "has_group_buy": True,
                "source": "mock",
            },
            {
                "name": "亚朵酒店(三里屯店)",
                "rating": 4.6,
                "avg_price": "428元起/晚",
                "address": "朝阳区工人体育场北路甲2号",
                "distance_to_restaurant": 0.8,
                "features": ["阅读空间", "自助洗衣", "健身房", "含早餐"],
                "queue_status": "有房",
                "has_group_buy": False,
                "source": "mock",
            },
            {
                "name": "汉庭酒店(望京店)",
                "rating": 4.2,
                "avg_price": "218元起/晚",
                "address": "朝阳区望京西路48号",
                "distance_to_restaurant": 1.2,
                "features": ["经济实惠", "免费WiFi", "地铁附近", "24小时热水"],
                "queue_status": "紧张（剩3间）",
                "has_group_buy": True,
                "source": "mock",
            },
            {
                "name": "北京瑰丽酒店",
                "rating": 4.8,
                "avg_price": "1280元起/晚",
                "address": "朝阳区呼家楼京广中心",
                "distance_to_restaurant": 1.5,
                "features": ["五星级", "行政酒廊", "游泳池", "代客泊车", "SPA"],
                "queue_status": "有房",
                "has_group_buy": False,
                "source": "mock",
            },
        ]

    # ── 娱乐场所 Mock 数据（含电影院排片）──

    @staticmethod
    def _mock_entertainments(params: dict[str, Any]) -> list[dict[str, Any]]:
        """模拟娱乐场所搜索，含电影院排片信息。"""
        _ = params  # 保留扩展空间
        return [
            {
                "name": "万达影城(国贸店)",
                "category": "电影院",
                "rating": 4.5,
                "address": "朝阳区建国路93号万达广场5层",
                "distance_to_restaurant": 0.5,
                "features": ["IMAX厅", "杜比全景声", "在线选座", "爆米花套餐"],
                "current_movies": [
                    "流浪地球3 (IMAX 2D)",
                    "封神第二部 (3D)",
                    "热辣滚烫2 (2D)",
                ],
                "showtimes_after_meal": [
                    "20:30",
                    "21:00(IMAX)",
                    "21:45(3D)",
                ],
                "queue_status": "建议提前15分钟取票",
                "has_group_buy": True,
                "source": "mock",
            },
            {
                "name": "英皇电影城(国贸商城店)",
                "category": "电影院",
                "rating": 4.6,
                "address": "朝阳区建国门外大街1号国贸商城B1",
                "distance_to_restaurant": 0.6,
                "features": ["VIP厅", "情侣座", "激光放映", "酒吧式休息区"],
                "current_movies": [
                    "封神第二部 (3D)",
                    "哪吒之魔童闹海 (IMAX)",
                    "热辣滚烫2 (2D)",
                ],
                "showtimes_after_meal": [
                    "21:00(IMAX)",
                    "21:30",
                    "22:00(VIP)",
                ],
                "queue_status": "可提前在线选座",
                "has_group_buy": True,
                "source": "mock",
            },
            {
                "name": "纯K(建国路店)",
                "category": "KTV",
                "rating": 4.4,
                "address": "朝阳区建国路88号SOHO现代城B1",
                "distance_to_restaurant": 0.4,
                "features": ["海量曲库", "包厢可容纳10人", "小吃|酒水", "深夜营业"],
                "current_movies": [],
                "showtimes_after_meal": [],
                "queue_status": "需提前2小时预约",
                "has_group_buy": True,
                "source": "mock",
            },
            {
                "name": "三里屯太古里(商业综合体)",
                "category": "商场",
                "rating": 4.3,
                "address": "朝阳区三里屯路19号",
                "distance_to_restaurant": 1.3,
                "features": ["高端品牌", "餐饮街区", "露天广场", "夜间经济"],
                "current_movies": [],
                "showtimes_after_meal": [],
                "queue_status": "无需排队",
                "has_group_buy": False,
                "source": "mock",
            },
            {
                "name": "北京欢乐谷",
                "category": "游乐园",
                "rating": 4.4,
                "address": "朝阳区东四环小武基北路",
                "distance_to_restaurant": 8.5,
                "features": ["过山车", "水上乐园", "夜场开放", "家庭套票", "亲子友好"],
                "current_movies": [],
                "showtimes_after_meal": [],
                "queue_status": "周末人流较大",
                "has_group_buy": True,
                "source": "mock",
            },
            {
                "name": "北京海洋馆",
                "category": "游乐园",
                "rating": 4.5,
                "address": "海淀区高粱桥斜街乙18号",
                "distance_to_restaurant": 6.2,
                "features": ["海洋动物表演", "海底隧道", "亲子友好", "室内场馆"],
                "current_movies": [],
                "showtimes_after_meal": [],
                "queue_status": "建议上午入园",
                "has_group_buy": True,
                "source": "mock",
            },
        ]

    # ── 商场 Mock 数据 ──

    @staticmethod
    def _mock_shopping(params: dict[str, Any]) -> list[dict[str, Any]]:
        """模拟商场/购物中心搜索结果。"""
        _ = params
        return [
            {
                "name": "朝阳大悦城",
                "category": "商场",
                "rating": 4.5,
                "address": "朝阳区朝阳北路101号",
                "distance_to_restaurant": 2.8,
                "features": ["购物中心", "IMAX影院", "餐饮街区", "儿童乐园", "免费停车"],
                "current_movies": [],
                "showtimes_after_meal": [],
                "queue_status": "无需排队",
                "has_group_buy": False,
                "source": "mock",
            },
            {
                "name": "国贸商城",
                "category": "商场",
                "rating": 4.6,
                "address": "朝阳区建国门外大街1号",
                "distance_to_restaurant": 0.8,
                "features": ["高端品牌", "国际美食", "溜冰场", "地铁直达", "停车位充足"],
                "current_movies": [],
                "showtimes_after_meal": [],
                "queue_status": "无需排队",
                "has_group_buy": False,
                "source": "mock",
            },
            {
                "name": "北京SKP",
                "category": "商场",
                "rating": 4.7,
                "address": "朝阳区建国路87号",
                "distance_to_restaurant": 1.2,
                "features": ["奢侈品旗舰", "高端餐饮", "VIP休息室", "代客泊车"],
                "current_movies": [],
                "showtimes_after_meal": [],
                "queue_status": "无需排队",
                "has_group_buy": False,
                "source": "mock",
            },
        ]

    # ── 公交/地铁站 Mock 数据 ──

    @staticmethod
    def _mock_transit_stops(params: dict[str, Any]) -> list[dict[str, Any]]:
        """模拟公交/地铁站点搜索结果。"""
        _ = params
        return [
            {
                "name": "国贸地铁站",
                "category": "地铁站",
                "address": "朝阳区建国门外大街与东三环中路交叉口",
                "distance_km": 0.5,
                "lines": ["1号线", "10号线"],
                "source": "mock",
            },
            {
                "name": "大望路地铁站",
                "category": "地铁站",
                "address": "朝阳区建国路与西大望路交叉口",
                "distance_km": 0.8,
                "lines": ["1号线", "14号线"],
                "source": "mock",
            },
            {
                "name": "国贸公交站",
                "category": "公交站",
                "address": "朝阳区建国门外大街国贸桥东",
                "distance_km": 0.3,
                "lines": ["1路", "11路", "57路", "421路", "666路"],
                "source": "mock",
            },
            {
                "name": "SOHO现代城公交站",
                "category": "公交站",
                "address": "朝阳区建国路88号",
                "distance_km": 0.2,
                "lines": ["1路", "57路", "405路"],
                "source": "mock",
            },
            {
                "name": "金台路地铁站",
                "category": "地铁站",
                "address": "朝阳区金台路与朝阳北路交叉口",
                "distance_km": 1.5,
                "lines": ["6号线", "14号线"],
                "source": "mock",
            },
        ]
