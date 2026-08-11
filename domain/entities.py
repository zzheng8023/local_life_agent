"""
领域实体定义 (Domain Entities)

本模块定义 local_life_agent 的核心领域实体与值对象。
v2.0 升级：引入多轮对话记忆、中断反问、多领域（餐饮+酒旅+娱乐）支持。
所有实体均基于 Pydantic BaseModel，享有自动校验、序列化与类型强制转换能力，
不包含任何持久化或框架逻辑，确保领域层的纯正性与可测试性。

核心模型：
- UserPreference：从多用户对话中提取的结构化偏好画像（扩展 hotel_req / entertainment_req）。
- Restaurant：候选餐厅数据模型（扩展时空维度字段）。
- Hotel：候选酒店数据模型。
- Entertainment / Cinema：娱乐场所 / 电影院数据模型。
- AgentState：Agent 在会话中的完整运行状态（TypedDict 风格，支持 messages 追加）。
- SafetyDecision：安全审核结果的值对象。
"""

from __future__ import annotations

import operator
from typing import Any, Optional, Union

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field
from typing_extensions import Annotated


# ============================================================
# 值对象：用户偏好（多领域扩展）
# ============================================================

class IndividualUserPreference(BaseModel):
    """单用户偏好画像 — 从多人对话中提取的每个用户的独立偏好。

    当 users[] 数组出现在 LLM 分析结果中时，每人一条。
    用于展示、冲突检测和个性化推荐。
    """

    name: str = Field(
        default="匿名用户",
        description="用户称呼，如 'A'、'小明'、'妈妈'",
    )
    budget: Optional[str] = Field(
        default=None,
        description="该用户的预算偏好",
    )
    taste: Optional[str] = Field(
        default=None,
        description="该用户的口味偏好",
    )
    restrictions: Optional[str] = Field(
        default=None,
        description="该用户的饮食限制/忌口",
    )
    distance: Optional[str] = Field(
        default=None,
        description="该用户的距离约束",
    )
    time: Optional[str] = Field(
        default=None,
        description="该用户的时间偏好",
    )
    has_kids: bool = Field(
        default=False,
        description="是否携带儿童",
    )
    need_parking: bool = Field(
        default=False,
        description="是否需要停车",
    )
    key_utterance: str = Field(
        default="",
        description="该用户的关键原话片段，用于上下文追溯",
    )
    origin_point: Optional[str] = Field(
        default=None,
        description="该用户的出发地点，如 '海淀区中关村'、'西城区金融街'、'国贸'",
    )


class UserPreference(BaseModel):
    """从用户对话中提取的结构化偏好画像。

    每个字段均为可选，表示该维度上用户是否表达了明确偏好。
    当多个用户存在冲突时，在 AgentState.conflict_strategy 中体现协调结果。

    v2.0 新增：
    - hotel_req：住宿需求（是否需要在用餐地附近订酒店、预算、房型等）。
    - entertainment_req：娱乐需求（是否饭后想看电影/唱歌/逛商场等）。
    """

    # ── 餐饮维度 ──
    budget: Optional[str] = Field(
        default=None,
        description="人均预算区间，如 '人均50元以内'、'人均100-200元'、'不限'",
    )
    taste: Optional[str] = Field(
        default=None,
        description="口味偏好，如 '川菜'、'日料'、'偏辣'、'清淡'、'无偏好'",
    )
    restrictions: Optional[str] = Field(
        default=None,
        description="饮食限制或忌口，如 '不吃香菜'、'素食'、'清真'、'海鲜过敏'",
    )
    distance: Optional[str] = Field(
        default=None,
        description="距离/区域约束，如 '3公里以内'、'朝阳区'、'公司附近'、'不限'",
    )
    city: Optional[str] = Field(
        default=None,
        description="用户指定的搜索城市，如 '北京'、'深圳'、'杭州'",
    )
    time: Optional[str] = Field(
        default=None,
        description="就餐时间约束，如 '今晚7点'、'周末中午'、'尽快'",
    )
    has_kids: bool = Field(
        default=False,
        description="是否携带儿童，影响是否需要儿童座椅、亲子友好餐厅推荐",
    )
    need_parking: bool = Field(
        default=False,
        description="是否需要停车位，影响是否推荐有停车场或代客泊车的餐厅",
    )

    # ── v2.0 新增：酒旅维度 ──
    hotel_req: Optional[str] = Field(
        default=None,
        description=(
            "住宿需求。如 '需要附近酒店，预算300以内，双人房'、"
            "'4人需要住酒店，交通便利'。为 None 表示无住宿需求。"
        ),
    )

    # ── v2.0 新增：娱乐维度 ──
    entertainment_req: Optional[str] = Field(
        default=None,
        description=(
            "娱乐需求。如 '饭后想看电影'、'附近有KTV吗'、'吃完想逛商场'。"
            "为 None 表示无娱乐需求。"
        ),
    )

    # ── v2.1 新增：购物/游玩维度 ──
    shopping_req: Optional[str] = Field(
        default=None,
        description=(
            "购物/游乐园需求。如 '想逛商场'、'去游乐园'、'附近有购物中心吗'。"
            "为 None 表示无购物/游玩需求。"
        ),
    )

    # ── v2.1 新增：交通出行维度 ──
    transit_req: Optional[str] = Field(
        default=None,
        description=(
            "交通出行需求。如 '附近有公交站吗'、'坐地铁怎么去'、'吃完饭怎么走'。"
            "为 None 表示无交通需求。"
        ),
    )

    # ── v2.1 新增：精确位置 ──
    freeform_location: Optional[str] = Field(
        default=None,
        description=(
            "用户自由输入的精确位置，如 '朝阳大悦城'、'国贸三期楼下'、"
            "'中关村软件园'。用于超越省市区的精确搜索定位。"
        ),
    )

    # ── v3.2 新增：多用户出行起点 ──
    origin_points: list[dict[str, str]] = Field(
        default_factory=list,
        description=(
            "多用户的出行起点列表。每项包含 name 和 location。"
            "用于公交路径规划的多OD对计算。"
            "如 [{'name': '老张', 'location': '海淀区中关村'}, ...]"
        ),
    )

    # ── v3.4 新增：停车 / 单车需求 ──
    need_parking_detail: Optional[Union[str, bool]] = Field(
        default=None,
        description=(
            "停车查询需求。如 '附近有停车场吗'、'开车去哪里停车'。"
            "为 None 表示无停车查询需求。"
        ),
    )
    bike_req: Optional[str] = Field(
        default=None,
        description=(
            "共享单车需求。如 '附近有共享单车吗'、'骑车过去方便吗'。"
            "为 None 表示无单车需求。"
        ),
    )


# ============================================================
# 实体：餐厅（v2.0 扩展时空维度）
# ============================================================

class Restaurant(BaseModel):
    """候选餐厅的数据模型。

    承载搜索与推荐环节中的餐厅结构化信息，作为工作流各阶段的数据契约。

    v2.0 新增：
    - distance_to_hotel：到推荐酒店的步行/驾车距离。
    - nearby_entertainment：附近可到达的娱乐场所列表。
    """

    name: str = Field(
        ...,
        description="餐厅名称",
    )
    rating: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="评分，范围 [0.0, 5.0]",
    )
    avg_price: Optional[str] = Field(
        default=None,
        description="人均价格描述，如 '人均80元'",
    )
    cuisine: Optional[str] = Field(
        default=None,
        description="菜系类型，如 '川菜'、'粤菜'、'日料'",
    )
    address: Optional[str] = Field(
        default=None,
        description="餐厅地址",
    )
    distance_km: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="距用户位置的直线距离（公里）",
    )
    features: list[str] = Field(
        default_factory=list,
        description="特色标签列表，如 ['有包间', '免费停车', '亲子友好', '夜景位']",
    )
    source: Optional[str] = Field(
        default=None,
        description="数据来源，如 'dianping'、'meituan'、'amap'",
    )

    # ── v2.0 新增：时空维度字段 ──
    distance_to_hotel: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="到推荐酒店的直线距离（公里），用于时空推理计算",
    )
    nearby_entertainment: list[str] = Field(
        default_factory=list,
        description=(
            "附近步行可达的娱乐场所名称列表，用于支持 '吃完饭去哪看电影' 等时空决策。"
            "如 ['万达影城(国贸店)', '纯K(三里屯店)']。"
        ),
    )


# ============================================================
# v2.0 新增实体：酒店
# ============================================================

class Hotel(BaseModel):
    """候选酒店的数据模型。

    支持 '聚餐后需要住宿' 场景的酒店推荐与时空距离计算。
    """

    name: str = Field(
        ...,
        description="酒店名称",
    )
    rating: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="评分，范围 [0.0, 5.0]",
    )
    avg_price: Optional[str] = Field(
        default=None,
        description="每晚价格描述，如 '298元起'",
    )
    address: Optional[str] = Field(
        default=None,
        description="酒店地址",
    )
    distance_to_restaurant: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="到推荐餐厅的直线距离（公里），用于时空推理",
    )
    features: list[str] = Field(
        default_factory=list,
        description="设施标签列表，如 ['免费停车', '含早', '接送服务', '可加床']",
    )
    source: Optional[str] = Field(
        default=None,
        description="数据来源",
    )


# ============================================================
# v2.0 新增实体：娱乐场所
# ============================================================

class Entertainment(BaseModel):
    """娱乐场所（通用）的数据模型。

    涵盖电影院、KTV、商场等多种业态。
    """

    name: str = Field(
        ...,
        description="场所名称",
    )
    category: str = Field(
        default="",
        description="业态类型，如 '电影院'、'KTV'、'商场'、'剧本杀'",
    )
    rating: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="评分",
    )
    address: Optional[str] = Field(
        default=None,
        description="地址",
    )
    distance_to_restaurant: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="到推荐餐厅的直线距离（公里）",
    )
    features: list[str] = Field(
        default_factory=list,
        description="特色标签",
    )
    source: Optional[str] = Field(
        default=None,
        description="数据来源",
    )


class Cinema(Entertainment):
    """电影院（继承 Entertainment）。

    新增排片与场次信息，支持 '吃完饭看电影' 的时空推荐。
    """

    current_movies: list[str] = Field(
        default_factory=list,
        description="当前上映的电影名称列表",
    )
    showtimes_after_meal: list[str] = Field(
        default_factory=list,
        description="饭后可选场次，如 ['21:00(原版)', '21:30(IMAX)']",
    )


# ============================================================
# v2.1 新增实体：公交/地铁站点
# ============================================================

class TransitStop(BaseModel):
    """公交/地铁站点数据模型。

    支持 '吃完饭怎么坐公交去'、'附近有地铁站吗' 等交通出行场景。
    """

    name: str = Field(
        ...,
        description="站点名称，如 '国贸地铁站'、'SOHO现代城公交站'",
    )
    category: str = Field(
        default="公交站",
        description="站点类型：'公交站' / '地铁站' / '客运站'",
    )
    address: Optional[str] = Field(
        default=None,
        description="站点地址",
    )
    distance_km: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="距搜索位置的直线距离（公里）",
    )
    lines: list[str] = Field(
        default_factory=list,
        description="途经线路，如 ['1路', '57路', '地铁1号线', '地铁10号线']",
    )
    source: Optional[str] = Field(
        default=None,
        description="数据来源",
    )


# ============================================================
# 聚合根：Agent 会话状态（v2.0 — TypedDict 风格，支持消息追加）
# ============================================================

class AgentState(BaseModel):
    """Agent 会话的完整运行状态上下文（v2.0 架构）。

    作为贯穿 Analysis → Retrieval → Recommendation → Clarification → Safety
    全流程的聚合根，承载原始输入、中间推理结果、候选集、最终输出以及 Trace 日志。

    v2.0 变更：
    - messages：使用 Annotated[list[AnyMessage], operator.add] 支持多轮对话消息追加。
    - needs_clarification / clarification_question：支持中断反问机制。
    - 新增 candidate_hotels / candidate_entertainments 多领域候选集。

    Attributes:
        messages: 多轮对话历史（Human + AI + System 消息）。
        raw_query: 当前轮次的用户输入。
        user_preference: 提取并协调后的结构化偏好。
        conflict_strategy: 多人偏好冲突时的协调策略。
        needs_clarification: 是否需要暂停并反问用户补充信息。
        clarification_question: 具体向用户反问的问题文本。
        candidate_restaurants: 候选餐厅列表（原始字典格式）。
        candidate_hotels: 候选酒店列表（v2.0 新增）。
        candidate_entertainments: 候选娱乐场所列表（v2.0 新增）。
        final_recommendation: 最终推荐文案。
        safety_passed: 安全审查是否通过。
        safety_violations: 命中的违规类型。
        trace_logs: 全流程可观测数据日志。
    """

    # ── v2.0：多轮对话记忆 ──
    messages: Annotated[list[AnyMessage], operator.add] = Field(
        default_factory=list,
        description=(
            "多轮对话历史。使用 Annotated[list[AnyMessage], operator.add] "
            "支持 LangGraph 节点间的消息追加合并。"
        ),
    )

    # ── 原始输入 ──
    raw_query: str = Field(
        default="",
        description="当前轮次的用户输入文本",
    )

    # ── v2.1：精确位置 ──
    freeform_location: str = Field(
        default="",
        description="用户自由输入的精确位置文本（与省市区域联选共存）",
    )

    # ── 用户偏好 ──
    user_preference: UserPreference = Field(
        default_factory=UserPreference,
        description="提取并协调后的结构化偏好画像",
    )
    conflict_strategy: str = Field(
        default="",
        description=(
            "当多人偏好存在冲突时的协调策略说明。"
            "例如：'A想吃辣、B忌辣，优先照顾B的忌口，推荐不辣但有风味的餐厅'"
        ),
    )

    # ── v2.1：闲聊检测 ──
    is_chitchat: bool = Field(
        default=False,
        description=(
            "当前用户输入是否为纯闲聊（问候/聊天/无业务意图）。"
            "为 True 时跳过检索-推荐-安全流程，直接返回闲聊回复。"
        ),
    )
    chitchat_reply: str = Field(
        default="",
        description="当 is_chitchat=True 时，LLM 生成的闲聊回复文本。",
    )

    # ── v3.1：领域外检测 ──
    is_out_of_domain: bool = Field(
        default=False,
        description=(
            "当前用户输入是否完全超出本地生活五大领域（餐饮/住宿/娱乐/购物/交通）。"
            "为 True 时跳过全部业务流程，直接返回礼貌拒绝对话。"
            "例如：股票、编程、翻译等与本地生活无关的话题。"
        ),
    )
    out_of_domain_reply: str = Field(
        default="",
        description="当 is_out_of_domain=True 时，LLM 生成的礼貌拒绝回复文本。",
    )

    # ── v3.1：流式模式标记 ──
    streaming_mode: bool = Field(
        default=False,
        description=(
            "v3.1 新增：是否为流式模式。为 True 时 _recommend_node 跳过同步 LLM 调用，"
            "由 run_stream() 在外部做流式生成。非流式 chat() 路径保持原有行为。"
        ),
    )

    # ── vNext：天气查询支持 ──
    is_weather_query: bool = Field(
        default=False,
        description=(
            "是否为天气查询。为 True 时跳过检索流程，走 weather_node。"
            "由 _analyze_node 中的关键词检测设置。"
        ),
    )
    weather_city: str = Field(
        default="",
        description="天气查询城市（从用户偏好或输入中提取）。",
    )
    weather_result: Optional[dict[str, Any]] = Field(
        default=None,
        description="天气工具调用结果。",
    )

    # ── vNext：跨轮次时间上下文记忆 ──
    last_resolved_time: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "上一轮解析的时间上下文（TimeContext 序列化）。"
            "用于跨轮次时间指代解析（如 '当天'、'那天' 引用前一轮中提到的日期）。"
            "每次 weather_node 或包含时间表达式的非天气查询后更新。"
        ),
    )

    # ── v2.0：中断反问机制 ──
    needs_clarification: bool = Field(
        default=False,
        description=(
            "是否需要暂停流程并反问用户补充信息。"
            "当偏好提取不足（如无预算、无口味）或存在歧义时置为 True。"
        ),
    )
    clarification_question: str = Field(
        default="",
        description=(
            "当 needs_clarification=True 时，向用户反向提问的具体内容。"
            "如 '请问你们对人均预算有什么要求吗？附近有不同价位的选择。'"
        ),
    )

    # ── v2.2：多人每人偏好 ──
    individual_preferences: list[IndividualUserPreference] = Field(
        default_factory=list,
        description=(
            "v2.2 新增：多用户场景下每个用户的独立偏好列表。"
            "当 LLM 分析结果中包含 users[] 数组时逐项解析填充。"
        ),
    )

    # ── v6.0：冲突检测与协商 ──
    detected_conflicts: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "v6.0 新增：程序化冲突检测结果列表。"
            "每项包含 dimension、severity、users_involved、suggested_resolution 等字段。"
        ),
    )
    needs_conflict_negotiation: bool = Field(
        default=False,
        description=(
            "v6.0 新增：是否需要触发冲突协商节点。"
            "当程序化冲突检测发现 severity='critical' 的冲突时置为 True。"
        ),
    )
    conflict_negotiation_question: str = Field(
        default="",
        description=(
            "v6.0 新增：冲突协商阶段向用户反问的问题文本。"
        ),
    )

    # ── 候选集：多领域 ──
    candidate_restaurants: list[dict[str, Any]] = Field(
        default_factory=list,
        description="检索阶段产出的候选餐厅列表（原始字典格式）",
    )
    candidate_hotels: list[dict[str, Any]] = Field(
        default_factory=list,
        description="v2.0 新增：候选酒店列表（原始字典格式）",
    )
    candidate_entertainments: list[dict[str, Any]] = Field(
        default_factory=list,
        description="v2.0 新增：候选娱乐场所列表（原始字典格式）",
    )
    candidate_shopping: list[dict[str, Any]] = Field(
        default_factory=list,
        description="v2.2 新增：候选商场/购物中心列表（原始字典格式）",
    )
    candidate_transit_stops: list[dict[str, Any]] = Field(
        default_factory=list,
        description="v2.1 新增：候选公交/地铁站点列表（原始字典格式）",
    )
    candidate_transit_directions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="v3.1 新增：公交路径规划结果（起点→餐厅→酒店等多段路线）",
    )
    candidate_parking: list[dict[str, Any]] = Field(
        default_factory=list,
        description="v3.4 新增：停车场查询结果",
    )
    candidate_bike_stations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="v3.4 新增：共享单车站点查询结果",
    )


    # ── 最终输出 ──
    final_recommendation: str = Field(
        default="",
        description="最终推荐文案（合并餐饮+酒旅+娱乐的综合方案）",
    )

    # ── 安全审查 ──
    safety_passed: bool = Field(
        default=True,
        description="安全审查是否通过",
    )
    safety_violations: list[str] = Field(
        default_factory=list,
        description="命中的违规类型列表",
    )

    # ── Trace ──
    trace_logs: list[dict[str, Any]] = Field(
        default_factory=list,
        description="全流程可观测数据日志，用于离线评测与回归测试",
    )


# ============================================================
# 值对象：安全审核结果
# ============================================================

class SafetyDecision(BaseModel):
    """安全审核结果的值对象。

    记录 Safety 阶段对最终推荐文案的审查结论与改写结果。
    """

    passed: bool = Field(
        default=True,
        description="审核是否通过：True 表示原始输出合规，False 表示存在违规内容",
    )
    original_text: str = Field(
        default="",
        description="进入安全审查前的原始文本",
    )
    rewritten_text: Optional[str] = Field(
        default=None,
        description="当审核不通过时，Safety 网关改写后的合规文本；通过时为 None",
    )
    violations: list[str] = Field(
        default_factory=list,
        description="检测到的违规类型列表，如 ['承诺已预约', '编造库存']",
    )
