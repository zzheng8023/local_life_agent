"""
工具注册表 (Tool Registry) — 插件化架构核心

本模块定义了 ToolDefinition 数据类和 TOOL_REGISTRY 注册表，
作为系统中所有工具元数据的**单一事实来源** (Single Source of Truth)。

设计原则：
- ToolDefinition 是纯配置/元数据的 dataclass，不持有工具实例
- TOOL_REGISTRY 是一个有序列表，添加新工具只需在此追加一条目
- 所有原本逐个工具硬编码的地方改为遍历 registry 动态生成
- UserPreference / AgentState 保留具名字段（向后兼容 Pydantic 序列化）

用法：
    from domain.tool_registry import TOOL_REGISTRY, build_context_dict

    # 动态构建推荐上下文
    context = build_context_dict(TOOL_REGISTRY, state)

    # 动态构建偏好摘要
    pref_summary = build_pref_summary_dict(TOOL_REGISTRY, pref)

    # 动态检查工具触发
    triggered = get_triggered_tools(TOOL_REGISTRY, pref, raw_query)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ================================================================
# ToolDefinition 数据类
# ================================================================


@dataclass
class ToolDefinition:
    """单个工具在其完整生命周期中的不可变定义。

    每个字段描述工具在系统中的一个维度：身份、偏好映射、检索触发、
    领域元数据、Prompt 片段、UI 展示。全部通过字符串引用现有 Pydantic 字段，
    不改变数据模型。
    """

    # ── 身份 ──
    key: str  # "parking" — 唯一标识
    tool_attr: str  # "_parking_tool" — workflow 实例属性名
    candidate_key: str  # "candidate_parking" — AgentState 字段名
    tool_cls: Optional[type] = None  # AmapParkingTool — UI 自动实例化用

    # ── 偏好映射 ──
    preference_field: str = ""  # "need_parking_detail" — UserPreference 字段名
    preference_detail_field: str = ""  # 停车专用 "need_parking_detail"

    # ── 检索触发 ──
    trigger_keywords: list[str] = field(default_factory=list)
    trigger_pref_fields: list[str] = field(default_factory=list)
    always_trigger: bool = False  # 仅餐厅为 True

    # ── 领域元数据 ──
    emoji: str = ""
    display_label: str = ""  # "停车场"
    domain_category: str = ""  # "交通"（用于分析 prompt 能力范围）

    # ── 搜索参数构建器 (pref: UserPreference, state: AgentState) -> dict ──
    search_params_builder: Optional[Callable] = None

    # ── 分析 Prompt 片段 ──
    analysis_dimension_section: str = ""  # 完整维度说明章节
    analysis_json_field_name: str = ""  # aggregated_preference JSON Schema 中的字段名
    analysis_null_check_field: str = ""  # "特别注意"中列出的空值检查字段名

    # ── 推荐 Prompt / 上下文片段 ──
    recommendation_context_key: str = ""  # context dict 中的键名（如 "parking"）
    recommendation_output_section: str = ""  # 推荐输出结构中的 Markdown 章节
    recommendation_output_emoji: str = ""  # 输出章节图标

    # ── UI 候选表格 ──
    candidate_section_title: str = ""
    candidate_columns: list[str] = field(default_factory=list)

    # ── 偏好表格 ──
    preference_table_label: str = ""


# ================================================================
# 搜索参数构建器（模块级函数，供 lambda 引用）
# ================================================================


def _build_restaurant_params(pref: Any, state: Any) -> dict[str, Any]:
    """构建餐厅搜索参数。"""
    params: dict[str, Any] = {
        "query": state.raw_query,
        "budget": pref.budget,
        "taste": pref.taste,
        "restrictions": pref.restrictions,
        "distance": pref.distance,
        "city": pref.city,
        "time": pref.time,
        "has_kids": pref.has_kids,
        "need_parking": pref.need_parking,
    }
    raw = (state.raw_query or "")
    if "包间" in raw:
        params["features"] = ["有包间"]
    return params


def _build_hotel_params(pref: Any, state: Any) -> dict[str, Any]:
    return {
        "query": pref.hotel_req or state.raw_query[:80],
        "budget": pref.budget,
        "city": pref.city,
    }


def _build_entertainment_params(pref: Any, state: Any) -> dict[str, Any]:
    return {
        "query": pref.entertainment_req or state.raw_query[:80],
        "city": pref.city,
    }


def _build_shopping_params(pref: Any, state: Any) -> dict[str, Any]:
    return {
        "query": pref.shopping_req or state.raw_query[:80],
        "city": pref.city,
    }


def _build_transit_params(pref: Any, state: Any) -> dict[str, Any]:
    return {
        "query": pref.transit_req or pref.freeform_location or pref.city or "",
        "city": pref.city,
    }


def _build_parking_params(pref: Any, state: Any) -> dict[str, Any]:
    return {
        "query": pref.need_parking_detail or "停车场",
        "city": pref.city,
    }


def _build_bike_params(pref: Any, state: Any) -> dict[str, Any]:
    return {
        "query": pref.bike_req or "共享单车",
        "city": pref.city,
    }


# ================================================================
# TOOL_REGISTRY — 所有工具的单一注册点
# ================================================================

TOOL_REGISTRY: list[ToolDefinition] = [
    ToolDefinition(
        key="restaurant",
        tool_attr="_search_tool",
        candidate_key="candidate_restaurants",
        tool_cls=None,  # AmapRestaurantTool — 由调用者单独处理（始终必须注入）
        preference_field="",
        always_trigger=True,
        emoji="🍽️",
        display_label="餐厅",
        domain_category="餐饮",
        search_params_builder=_build_restaurant_params,
        recommendation_context_key="restaurants",
        recommendation_output_section="🍽️ 餐厅推荐",
        recommendation_output_emoji="🍽️",
        candidate_section_title="🍽️ 餐厅",
        candidate_columns=["name", "rating", "avg_price", "cuisine", "distance_km", "address"],
        preference_table_label="",
    ),
    ToolDefinition(
        key="hotel",
        tool_attr="_hotel_tool",
        candidate_key="candidate_hotels",
        tool_cls=None,  # AmapHotelTool
        preference_field="hotel_req",
        trigger_pref_fields=["hotel_req"],
        trigger_keywords=["住", "酒店", "住宿", "过夜", "宾馆", "民宿"],
        emoji="🏨",
        display_label="住宿",
        domain_category="住宿",
        search_params_builder=_build_hotel_params,
        analysis_dimension_section=(
            "**🏨 住宿维度**\n"
            '   - hotel_req：住宿需求描述。如果用户提到"需要住酒店"、"晚上回不去了要订房"、'
            '"4个人需要住酒店"、"帮我找个附近的酒店"等表述，则提取为具体需求文本。'
            "如果对话中完全没有住宿话题，则填 null"
        ),
        analysis_json_field_name="hotel_req",
        analysis_null_check_field="hotel_req",
        recommendation_context_key="hotels",
        recommendation_output_section="🏨 住宿推荐",
        recommendation_output_emoji="🏨",
        candidate_section_title="🏨 酒店",
        candidate_columns=["name", "rating", "avg_price", "address", "distance_to_restaurant"],
        preference_table_label="住宿需求",
    ),
    ToolDefinition(
        key="entertainment",
        tool_attr="_entertainment_tool",
        candidate_key="candidate_entertainments",
        tool_cls=None,  # AmapEntertainmentTool
        preference_field="entertainment_req",
        trigger_pref_fields=["entertainment_req"],
        trigger_keywords=["电影", "ktv", "唱歌", "影院", "娱乐"],
        emoji="🎬",
        display_label="娱乐",
        domain_category="娱乐",
        search_params_builder=_build_entertainment_params,
        analysis_dimension_section=(
            "**🎬 娱乐与购物维度**\n"
            '   - entertainment_req：饭后娱乐需求。如果用户提到以下任何表述，必须提取为具体需求文本：\n'
            '     * "看电影" / "去电影院" / "去万达" / "看电影去" / "吃完了看电影"\n'
            '     * "KTV" / "唱歌" / "去唱歌"\n'
            '     * "逛商场" / "去商场" / "购物" / "买东西"\n'
            '     * "游乐园" / "去公园" / "去XX玩"\n'
            '   - 重要：不要把"吃完看电影"的娱乐需求编造为"无"或 null。只要有相关提到就必须提取。\n'
            "如果对话中完全没有娱乐话题，则填 null"
        ),
        analysis_json_field_name="entertainment_req",
        analysis_null_check_field="entertainment_req",
        recommendation_context_key="entertainments",
        recommendation_output_section="🎬 娱乐推荐",
        recommendation_output_emoji="🎬",
        candidate_section_title="🎬 娱乐场所",
        candidate_columns=["name", "category", "rating", "address", "distance_to_restaurant"],
        preference_table_label="娱乐需求",
    ),
    ToolDefinition(
        key="shopping",
        tool_attr="_shopping_tool",
        candidate_key="candidate_shopping",
        tool_cls=None,  # AmapShoppingTool
        preference_field="shopping_req",
        trigger_pref_fields=["shopping_req"],
        trigger_keywords=["逛", "商场", "购物", "买", "游乐园"],
        emoji="🛍️",
        display_label="购物/游玩",
        domain_category="购物",
        search_params_builder=_build_shopping_params,
        analysis_dimension_section=(
            "**🛍️ 购物/游玩维度**\n"
            '   - shopping_req：购物或游玩需求。如果用户提到"逛商场"、"去购物中心"、'
            '"去游乐园"、"带小孩去乐园"、"去欢乐谷"等表述，则提取为具体需求文本。'
            "如果对话中完全没有购物/游玩话题，则填 null"
        ),
        analysis_json_field_name="shopping_req",
        analysis_null_check_field="shopping_req",
        recommendation_context_key="shopping",
        recommendation_output_section="🛍️ 购物/游玩推荐",
        recommendation_output_emoji="🛍️",
        candidate_section_title="🛍️ 购物/游玩",
        candidate_columns=["name", "category", "rating", "address"],
        preference_table_label="购物/游玩",
    ),
    ToolDefinition(
        key="transit",
        tool_attr="_transit_tool",
        candidate_key="candidate_transit_stops",
        tool_cls=None,  # AmapTransitTool
        preference_field="transit_req",
        trigger_pref_fields=["transit_req"],
        trigger_keywords=["公交", "地铁", "怎么去", "坐车", "路线", "怎么走", "从", "出发"],
        emoji="🚌",
        display_label="交通",
        domain_category="交通",
        search_params_builder=_build_transit_params,
        analysis_dimension_section=(
            "**🚌 交通出行维度**\n"
            '   - transit_req：交通出行需求。如果用户提到以下**任何**表述，必须提取为具体需求文本：\n'
            '     * "坐公交去" / "坐地铁去" / "公交路线" / "交通方式" / "怎么过去" / "怎么走"\n'
            '     * "从XX出发" / "从XX过来" / "家住XX" / "在XX上班" — 含地点出发点的表述\n'
            '     * "附近有公交站吗" / "地铁怎么坐" / "交通方便吗" / "吃完饭怎么过去"\n'
            '     * "查一下路线" / "规划路线" / "汇合" — 需要路径规划的场景\n'
            '     * 任何提到具体出发点+目的地的行程表述\n'
            '   - ⚠️ 重要：只要有人在对话中提到了自己的出发地点（如"我从海淀过来"、"在中关村出发"），\n'
            '     就应该提取 transit_req，因为后续需要为这些人规划交通路线。\n'
            "如果对话中完全没有交通话题，则填 null"
        ),
        analysis_json_field_name="transit_req",
        analysis_null_check_field="transit_req",
        recommendation_context_key="transit_stops",
        recommendation_output_section="🚌 附近交通",
        recommendation_output_emoji="🚌",
        candidate_section_title="🚌 公交/地铁站点",
        candidate_columns=["name", "category", "lines", "distance_km"],
        preference_table_label="交通需求",
    ),
    ToolDefinition(
        key="parking",
        tool_attr="_parking_tool",
        candidate_key="candidate_parking",
        tool_cls=None,  # AmapParkingTool
        preference_field="need_parking_detail",
        trigger_pref_fields=["need_parking_detail", "need_parking"],
        trigger_keywords=["停车", "停车场", "车位", "泊车"],
        emoji="🅿️",
        display_label="停车",
        domain_category="交通",
        search_params_builder=_build_parking_params,
        analysis_dimension_section=(
            "**🅿️ 停车查询维度**\n"
            '   - need_parking_detail：停车查询需求。如果用户提到以下表述，必须提取：\n'
            '     * "停车场" / "停车" / "哪里停车" / "停车方便吗" / "有没有车位"\n'
            '     * "开车去" + 表示对停车有关切（如"开车去方便停车吗"）\n'
            "如果对话中完全没有停车话题，则填 null"
        ),
        analysis_json_field_name="need_parking_detail",
        analysis_null_check_field="need_parking_detail",
        recommendation_context_key="parking",
        recommendation_output_section="🅿️ 停车场推荐",
        recommendation_output_emoji="🅿️",
        candidate_section_title="🅿️ 停车场",
        candidate_columns=["name", "address", "rating", "distance_km"],
        preference_table_label="停车需求",
    ),
    ToolDefinition(
        key="bike",
        tool_attr="_bike_tool",
        candidate_key="candidate_bike_stations",
        tool_cls=None,  # AmapBikeTool
        preference_field="bike_req",
        trigger_pref_fields=["bike_req"],
        trigger_keywords=["单车", "共享单车", "骑车", "骑行", "自行车"],
        emoji="🚲",
        display_label="共享单车",
        domain_category="交通",
        search_params_builder=_build_bike_params,
        analysis_dimension_section=(
            "**🚲 共享单车维度**\n"
            '   - bike_req：共享单车需求。如果用户提到"共享单车"、"单车"、"骑车"、"骑行"、"自行车"\n'
            "如果对话中完全没有单车话题，则填 null"
        ),
        analysis_json_field_name="bike_req",
        analysis_null_check_field="bike_req",
        recommendation_context_key="bike_stations",
        recommendation_output_section="🚲 共享单车",
        recommendation_output_emoji="🚲",
        candidate_section_title="🚲 共享单车站点",
        candidate_columns=["name", "address", "rating", "distance_km"],
        preference_table_label="单车需求",
    ),
]


# ================================================================
# 辅助函数
# ================================================================


def get_tool_by_key(registry: list[ToolDefinition], key: str) -> Optional[ToolDefinition]:
    """按 key 查找工具定义。"""
    for td in registry:
        if td.key == key:
            return td
    return None


def get_tool_by_attr(registry: list[ToolDefinition], tool_attr: str) -> Optional[ToolDefinition]:
    """按 tool_attr 查找工具定义。"""
    for td in registry:
        if td.tool_attr == tool_attr:
            return td
    return None


def _should_trigger(
    td: ToolDefinition,
    pref: Any,
    raw_lower: str,
    has_individual_prefs: bool = False,
) -> bool:
    """判定单个工具是否应在检索阶段触发。

    Args:
        td: 工具定义。
        pref: UserPreference 实例。
        raw_lower: 原始 query 的小写形式。
        has_individual_prefs: 是否有个体偏好（transit 额外条件）。

    Returns:
        True 如果应触发。
    """
    if td.always_trigger:
        return True

    # transit 特殊条件：有个人偏好时也触发
    if td.key == "transit" and has_individual_prefs:
        return True

    # 1) 检查偏好字段
    for pfield in td.trigger_pref_fields:
        # 支持 need_parking（bool 字段）
        val = getattr(pref, pfield, None)
        if val:
            return True

    # 2) 关键词兜底
    for kw in td.trigger_keywords:
        if kw in raw_lower:
            return True

    return False


def get_triggered_tools(
    registry: list[ToolDefinition],
    pref: Any,
    raw_query: str,
    has_individual_prefs: bool = False,
) -> list[ToolDefinition]:
    """返回所有应当触发的工具定义列表。

    Args:
        registry: TOOL_REGISTRY。
        pref: UserPreference 实例。
        raw_query: 当前轮 raw_query。
        has_individual_prefs: 是否有个人偏好。

    Returns:
        应当触发的 ToolDefinition 列表（按 registry 顺序）。
    """
    raw_lower: str = raw_query.lower() if raw_query else ""
    result: list[ToolDefinition] = []
    for td in registry:
        if _should_trigger(td, pref, raw_lower, has_individual_prefs):
            result.append(td)
    return result


def build_context_dict(
    registry: list[ToolDefinition],
    state: Any,
) -> dict[str, Any]:
    """从 AgentState 动态构建推荐上下文字典。

    遍历 registry，对每个有 candidate_key 的工具提取对应字段的值。

    Args:
        registry: TOOL_REGISTRY。
        state: AgentState 实例。

    Returns:
        {"restaurants": [...], "hotels": [...], "parking": [...], ...}
    """
    context: dict[str, Any] = {}
    # ── vNext：获取 previous_candidates 用于跨轮回退 ──
    prev = getattr(state, "previous_candidates", {}) or {}
    for td in registry:
        if td.candidate_key and td.recommendation_context_key:
            candidates = getattr(state, td.candidate_key, [])
            # 当前轮候选为空但有前轮数据时，回退使用前轮快照
            if not candidates and prev.get(td.candidate_key):
                candidates = prev[td.candidate_key]
            context[td.recommendation_context_key] = candidates
    # 额外上下文（非工具候选数据）
    context["user_preference"] = state.user_preference.model_dump(exclude_none=True)
    context["individual_preferences"] = [
        {
            "name": ip.name,
            "budget": ip.budget,
            "taste": ip.taste,
            "restrictions": ip.restrictions,
        }
        for ip in (getattr(state, "individual_preferences", []) or [])
    ]
    context["detected_conflicts"] = getattr(state, "detected_conflicts", []) or []
    context["conflict_strategy"] = getattr(state, "conflict_strategy", "")
    # transit_directions 不是 registry 里的工具（特殊逻辑）
    context["transit_directions"] = getattr(state, "candidate_transit_directions", [])
    return context


def build_pref_summary_dict(
    registry: list[ToolDefinition],
    pref: Any,
) -> dict[str, Any]:
    """从 UserPreference 动态构建偏好摘要字典。

    遍历 registry，提取每个工具的 preference_field 值。

    Args:
        registry: TOOL_REGISTRY。
        pref: UserPreference 实例。

    Returns:
        {"budget": ..., "hotel_req": ..., "need_parking_detail": ..., ...}
    """
    summary: dict[str, Any] = {
        "budget": pref.budget,
        "taste": pref.taste,
        "restrictions": pref.restrictions,
        "distance": pref.distance,
        "time": pref.time,
        "freeform_location": pref.freeform_location,
    }
    for td in registry:
        if td.preference_field:
            summary[td.preference_field] = getattr(pref, td.preference_field, None)
    return summary


def get_display_domains(
    registry: list[ToolDefinition],
    skip_restaurant: bool = False,
) -> list[ToolDefinition]:
    """返回需要在 UI 中展示候选表格的工具定义。

    Args:
        registry: TOOL_REGISTRY。
        skip_restaurant: 是否跳过餐厅（餐厅表格格式特殊）。

    Returns:
        需要候选表格展示的 ToolDefinition 列表。
    """
    result: list[ToolDefinition] = []
    for td in registry:
        if td.candidate_section_title and td.candidate_key:
            if skip_restaurant and td.key == "restaurant":
                continue
            result.append(td)
    return result


def get_preference_tools(
    registry: list[ToolDefinition],
) -> list[ToolDefinition]:
    """返回需要在偏好表格中展示的工具定义。

    Returns:
        有 preference_table_label 的 ToolDefinition 列表。
    """
    return [td for td in registry if td.preference_table_label]
