"""
工作流编排 (Workflow Orchestrator) — v2.0 (LangGraph)

本模块基于 LangGraph StateGraph 构建具备记忆和反问能力的多轮对话图状态机。

架构升级：
- 从单体 run() 方法升级为 LangGraph StateGraph 节点图。
- 通过 MemorySaver 实现跨轮次记忆持久化（按 thread_id 隔离）。
- 条件路由：信息不足时触发 clarify_node 中断反问，信息充足时进入检索-推荐-安全流程。

核心流程：
1. START → analyze_node → [conditional]
2.   needs_clarification=true  → clarify_node → END
3.   needs_clarification=false → retrieve_node → recommend_node → safety_guard_node → END

依赖注入：
- LLM 客户端 (ILLMClient) 与工具集 (ITool) 在构造时注入，图节点通过闭包访问。
"""

from __future__ import annotations

import json
import threading
from typing import Any, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from loguru import logger

from application.ports import ILLMClient, ITool
from application.config import MAX_WINDOW_MESSAGES, STREAM_HEARTBEAT_INTERVAL
from domain.entities import AgentState, UserPreference, IndividualUserPreference
from domain.prompt_specs import PromptManager
from domain.conflict_detector import ConflictDetector
from domain.tool_registry import (
    TOOL_REGISTRY,
    ToolDefinition,
    build_context_dict,
    build_pref_summary_dict,
    get_triggered_tools,
)
from restaurant_agent.tools.time_resolver import TimeResolver
from restaurant_agent.tools.weather_tool import WeatherTool
from restaurant_agent.schemas.time_schema import TimeExpression, TimeContext
from infrastructure.tracer import TraceLogger
from infrastructure.safety_prefilter import SafetyPrefilter
from infrastructure.audit_logger import AuditLogger
from infrastructure.amap_transit_direction import AmapTransitDirectionTool


class LocalLifeWorkflow:
    """本地生活 Agent 工作流编排器 (v2.0 LangGraph)。

    基于 LangGraph 的 StateGraph 实现，支持：
    - 多轮对话记忆（MemorySaver + thread_id）
    - 中断反问（clarify_node 条件路由）
    - 多领域工具检索（餐饮 + 酒旅 + 娱乐）
    - 履约前安全审查

    Attributes:
        _llm: 注入的大语言模型客户端。
        _search_tool: 餐饮搜索工具。
        _hotel_tool: 酒店/娱乐搜索工具（可选）。
        _tracer: 追踪日志记录器。
        _app: 已编译的 LangGraph 应用（含 MemorySaver）。
    """

    def __init__(
        self,
        llm: ILLMClient,
        tools: dict[str, ITool],
        transit_direction_tool: Optional[AmapTransitDirectionTool] = None,
        max_window_messages: int = MAX_WINDOW_MESSAGES,
        fast_llm: Optional[ILLMClient] = None,
        weather_tool: Optional[WeatherTool] = None,
    ) -> None:
        """初始化工作流并编译 LangGraph 状态图。

        Args:
            llm: 大语言模型客户端（用于核心推荐生成），实现 ILLMClient 接口。
            tools: 工具字典，按 tool_attr 名索引（如 {"_search_tool": ..., "_hotel_tool": ..., ...}）。
                调用者通过遍历 TOOL_REGISTRY 构建此字典。
            transit_direction_tool: 可选，公交路径规划工具（v3.1 新增，因触发逻辑特殊而独立传入）。
            max_window_messages: 滑动窗口大小（消息条数，默认 30 = 15 轮对话）。
            fast_llm: 可选，快速/轻量级 LLM 客户端，用于分析、分类、安全审查等
                非生成任务。为 None 时回退到 llm（向后兼容）。
        """
        self._llm: ILLMClient = llm
        self._fast_llm: ILLMClient = fast_llm if fast_llm is not None else llm
        self._tools: dict[str, ITool] = tools

        # 动态设置工具属性（向后兼容 self._xxx_tool 访问模式）
        for td in TOOL_REGISTRY:
            setattr(self, td.tool_attr, tools.get(td.tool_attr))

        self._transit_direction_tool: Optional[AmapTransitDirectionTool] = (
            transit_direction_tool
        )
        self._max_window_messages: int = max_window_messages
        self._tracer: TraceLogger = TraceLogger()
        self._prefilter: SafetyPrefilter = SafetyPrefilter()
        self._audit_logger: AuditLogger = AuditLogger()
        self._conflict_detector: ConflictDetector = ConflictDetector()
        self._weather_tool: WeatherTool = weather_tool or WeatherTool()
        self._time_resolver: TimeResolver = TimeResolver()
        self._app = self._build_graph()

        # 动态构建日志字符串
        tool_names_parts: list[str] = []
        for td in TOOL_REGISTRY:
            tool = tools.get(td.tool_attr)
            if tool:
                tool_names_parts.append(type(tool).__name__)
        tool_names_str: str = " + ".join(tool_names_parts) if tool_names_parts else "(none)"
        fast_info: str = (
            f"{type(fast_llm).__name__}({getattr(fast_llm, 'model_name', '?')})"
            if fast_llm is not None and fast_llm is not llm
            else "same as llm"
        )
        logger.info(
            f"[Workflow] init: LLM={type(llm).__name__}, fast_llm={fast_info}, "
            f"Tools={tool_names_str}, Window={max_window_messages}msgs"
        )

    # ================================================================
    # 公共属性
    # ================================================================

    @property
    def tracer(self) -> TraceLogger:
        """获取 TraceLogger 实例，供外部（Web UI / 评测脚本）访问。"""
        return self._tracer

    @property
    def fast_llm(self) -> ILLMClient:
        """获取快速 LLM 客户端，供行程提取等轻量任务使用。"""
        return self._fast_llm

    @property
    def app(self) -> StateGraph:
        """获取已编译的 LangGraph 应用。"""
        return self._app

    # ================================================================
    # 图构建
    # ================================================================

    def _build_graph(self):
        """构建并编译 LangGraph StateGraph。

        Returns:
            已编译的 StateGraph 应用实例。
        """
        # 使用 AgentState (Pydantic) 作为图状态 Schema
        builder: StateGraph = StateGraph(AgentState)

        # ── 注册节点 ──
        builder.add_node("analyze", self._analyze_node)
        builder.add_node("chitchat", self._chitchat_node)
        builder.add_node("out_of_domain", self._out_of_domain_node)
        builder.add_node("clarify", self._clarify_node)
        builder.add_node("weather", self._weather_node)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("recommend", self._recommend_node)
        builder.add_node("safety_guard", self._safety_guard_node)

        # ── 注册边 ──
        builder.add_edge(START, "analyze")

        # 条件路由：根据 is_chitchat / is_out_of_domain / is_weather_query / needs_clarification 决定走向
        builder.add_conditional_edges(
            "analyze",
            self._route_after_analyze,
            {
                "chitchat": "chitchat",
                "out_of_domain": "out_of_domain",
                "weather": "weather",
                "clarify": "clarify",
                "retrieve": "retrieve",
            },
        )

        # 闲聊后直接结束
        builder.add_edge("chitchat", END)

        # 领域外查询后直接结束
        builder.add_edge("out_of_domain", END)

        # 天气查询后直接结束
        builder.add_edge("weather", END)

        # 澄清后直接结束
        builder.add_edge("clarify", END)

        # 检索 → 推荐 → 安全审查 → 结束
        builder.add_edge("retrieve", "recommend")
        builder.add_edge("recommend", "safety_guard")
        builder.add_edge("safety_guard", END)

        # 编译（含 MemorySaver 实现跨轮次记忆）
        memory: MemorySaver = MemorySaver()
        return builder.compile(checkpointer=memory)

    # ================================================================
    # 条件路由
    # ================================================================

    @staticmethod
    def _route_after_analyze(state: AgentState) -> str:
        """分析阶段后的条件路由。

        Args:
            state: 当前图状态（分析阶段已更新 is_chitchat / is_out_of_domain / needs_clarification）。

        Returns:
            "chitchat" 如果仅闲聊，
            "out_of_domain" 如果超出能力范围，
            "clarify" 如果需要反问，
            "retrieve" 继续检索。
        """
        if state.is_weather_query:
            logger.info("[Route] is_weather_query=True → weather")
            return "weather"
        if state.is_out_of_domain:
            logger.info("[Route] is_out_of_domain=True → out_of_domain")
            return "out_of_domain"
        if state.is_chitchat:
            logger.info("[Route] is_chitchat=True → chitchat")
            return "chitchat"
        if state.needs_clarification:
            logger.info("[Route] needs_clarification=True → clarify")
            return "clarify"
        logger.info("[Route] → retrieve")
        return "retrieve"

    # ================================================================
    # 节点 1：意图分析
    # ================================================================

    def _analyze_node(self, state: AgentState) -> dict[str, Any]:
        """分析节点：提取多领域偏好 + 判断信息充足性。

        读取 messages 中的多轮对话历史，调用 LLM 解析偏好，
        提取 needs_clarification 和 clarification_question。

        vNext: 天气查询短路 —— 检测到天气关键词时跳过完整 LLM 分析，
        只做最小化处理，将日期/天气推理完全交给 weather_node。
        """
        phase: str = "analyze"
        self._tracer.start_phase(phase)

        # ── vNext: 天气查询短路 ──
        # 天气查询不需要餐饮/住宿/娱乐偏好提取，直接跳过 LLM 分析。
        # LLM 负责指代理解 (weather_node 中的 _extract_time_expression)，
        # 代码负责时间计算 (TimeResolver)，API 负责天气数据 (WeatherTool)。
        raw_query: str = state.raw_query
        if self._is_weather_query(raw_query):
            logger.info(f"[Analyze] 天气查询短路 → 跳过 LLM 分析")
            self._tracer.end_phase(
                phase,
                input_snapshot={"raw_query": raw_query[:60]},
                output_snapshot={"is_weather_query": True, "short_circuit": True},
            )
            # 尝试从已有状态中提取城市信息（上一轮可能已设置）
            city: str = state.weather_city or ""
            return {
                "is_out_of_domain": False,
                "is_chitchat": False,
                "is_weather_query": True,
                "weather_city": city,
                "user_preference": UserPreference(),
                "needs_clarification": False,
                "streaming_mode": state.streaming_mode,
                "raw_query": raw_query,
                # ── 保留跨轮上下文（防止 MemorySaver 重置为 None）──
                "last_resolved_time": state.last_resolved_time,
            }

        system_prompt: str = PromptManager.build_analysis_prompt()

        # ── 滑动窗口：仅取最近 N 条消息传入 LLM ──
        all_msgs: list[Any] = state.messages
        ws: int = self._max_window_messages
        window_msgs: list[Any] = all_msgs[-ws:] if len(all_msgs) > ws else all_msgs
        if len(all_msgs) > ws:
            logger.info(
                f"[Analyze] 滑动窗口: {len(all_msgs)} → {len(window_msgs)} 条消息 "
                f"(截断 {(len(all_msgs) - len(window_msgs))} 条早期消息)"
            )

        # 从消息历史中构建上下文
        conversation_text: str = _format_conversation(window_msgs)
        if state.raw_query:
            conversation_text += f"\n\n[最新用户输入]\n{state.raw_query}"

        user_prompt: str = (
            "请分析以下对话内容，提取用户在【餐饮、住宿、娱乐、购物、交通】五个维度的偏好，"
            "检测多人冲突，并判断信息是否充足：\n\n"
            f"{conversation_text}"
        )

        try:
            analysis_result: dict[str, Any] = self._fast_llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            logger.error(f"[Analyze] LLM 调用失败: {exc}")
            self._tracer.end_phase(
                phase,
                input_snapshot={"system_prompt_len": len(system_prompt)},
                output_snapshot={"error": str(exc)},
            )
            # LLM 失败时降级：使用空偏好触发全量检索
            # 不直接返回错误，由推荐 LLM 基于完整候选数据做最终判断
            logger.warning(
                "[Analyze] 降级：分析 LLM 失败，使用空偏好触发全量检索"
            )
            self._tracer.record_system_rule_hit(
                "分析降级", category="fallback",
                detail=f"原因: {exc}",
            )
            return {
                "user_preference": UserPreference(),
                "needs_clarification": False,
                "is_chitchat": False,
                "is_out_of_domain": False,
                "streaming_mode": state.streaming_mode,
                "last_resolved_time": state.last_resolved_time,
            }

        # ── 提取 is_out_of_domain ──
        is_out_of_domain: bool = bool(analysis_result.get("is_out_of_domain"))
        out_of_domain_reply: str = analysis_result.get("out_of_domain_reply", "")

        if is_out_of_domain:
            logger.info(f"[Analyze] 领域外检测命中 → \"{out_of_domain_reply[:50]}...\"")
            self._tracer.record_system_rule_hit(
                "领域外检测", category="out_of_domain",
                detail=f"回复: {out_of_domain_reply[:60]}",
            )
            self._tracer.end_phase(
                phase,
                input_snapshot={"system_prompt_len": str(len(system_prompt))},
                output_snapshot={"is_out_of_domain": True, "out_of_domain_reply": out_of_domain_reply[:100]},
            )
            return {
                "is_out_of_domain": True,
                "out_of_domain_reply": out_of_domain_reply,
                "final_recommendation": out_of_domain_reply,
                "safety_passed": True,
                "last_resolved_time": state.last_resolved_time,
            }

        # ── 提取 is_chitchat ──
        is_chitchat: bool = bool(analysis_result.get("is_chitchat"))
        chitchat_reply: str = analysis_result.get("chitchat_reply", "")

        if is_chitchat:
            logger.info(f"[Analyze] 闲聊检测命中 → \"{chitchat_reply[:50]}...\"")
            self._tracer.record_system_rule_hit(
                "闲聊检测", category="chitchat",
                detail=f"回复: {chitchat_reply[:60]}",
            )
            self._tracer.end_phase(
                phase,
                input_snapshot={"system_prompt_len": str(len(system_prompt))},
                output_snapshot={"is_chitchat": True, "chitchat_reply": chitchat_reply[:100]},
            )
            return {
                "is_chitchat": True,
                "chitchat_reply": chitchat_reply,
                "final_recommendation": chitchat_reply,
                "safety_passed": True,
                "last_resolved_time": state.last_resolved_time,
            }

        # ── 提取 needs_clarification ──
        needs_clarification: bool = analysis_result.get("needs_clarification", False)
        clarification_question: str = analysis_result.get("clarification_question", "")

        # ── 提取偏好 ──
        aggregated: dict[str, Any] = analysis_result.get("aggregated_preference", {})

        # 从 registry 动态构建 UserPreference 参数
        pref_kwargs: dict[str, Any] = {
            "budget": aggregated.get("budget"),
            "taste": aggregated.get("taste"),
            "restrictions": aggregated.get("restrictions"),
            "distance": aggregated.get("distance"),
            "city": aggregated.get("city"),
            "time": aggregated.get("time"),
            "has_kids": aggregated.get("has_kids", False),
            "need_parking": aggregated.get("need_parking", False),
            "freeform_location": aggregated.get("freeform_location"),
        }
        for td in TOOL_REGISTRY:
            if td.preference_field and td.analysis_json_field_name:
                pref_kwargs[td.preference_field] = aggregated.get(
                    td.analysis_json_field_name
                )

        preference: UserPreference = UserPreference(**pref_kwargs)
        conflict_strategy: str = aggregated.get("conflict_strategy", "")

        # ── v2.2: 解析每人偏好 (users[]) ──
        individual_prefs: list[IndividualUserPreference] = []
        conflicts: list[dict[str, Any]] = []  # v3.1 修复：在 if 块外初始化
        users_list: list[dict[str, Any]] = analysis_result.get("users", []) or []
        for user_data in users_list:
            try:
                pref_data: dict[str, Any] = user_data.get("preference", {}) or {}
                ind_pref: IndividualUserPreference = IndividualUserPreference(
                    name=str(user_data.get("name", "匿名用户")),
                    budget=pref_data.get("budget") or user_data.get("budget"),
                    taste=pref_data.get("taste") or user_data.get("taste"),
                    restrictions=pref_data.get("restrictions") or user_data.get("restrictions"),
                    distance=pref_data.get("distance") or user_data.get("distance"),
                    time=pref_data.get("time") or user_data.get("time"),
                    has_kids=bool(pref_data.get("has_kids", False) or user_data.get("has_kids", False)),
                    need_parking=bool(pref_data.get("need_parking", False) or user_data.get("need_parking", False)),
                    key_utterance=str(user_data.get("key_utterance", "")),
                    origin_point=pref_data.get("origin_point") or user_data.get("origin_point"),
                )
                individual_prefs.append(ind_pref)
            except Exception as exc:
                logger.warning(f"[Analyze] 解析单用户偏好失败: {exc}, data={user_data}")

        if individual_prefs:
            logger.info(
                f"[Analyze] 解析到 {len(individual_prefs)} 个用户的独立偏好: "
                f"{[p.name for p in individual_prefs]}"
            )

            # ── v6.0: 程序化冲突检测 ──
            try:
                conflicts: list[dict[str, Any]] = self._conflict_detector.detect(
                    individual_prefs
                )
                if conflicts:
                    logger.warning(
                        f"[Analyze] 程序化冲突检测: {len(conflicts)} 个冲突"
                    )
                    for c in conflicts:
                        self._tracer.record_system_rule_hit(
                            rule_name=f"conflict:{c['rule_id']}",
                            category="conflict",
                            detail=c["description"],
                        )
                        logger.info(
                            f"  [{c['rule_id']}] {c['dimension']}: "
                            f"{c['description'][:80]} "
                            f"→ 建议: {c['suggested_resolution'][:60]}"
                        )
                else:
                    logger.info("[Analyze] 未检测到程序化冲突")
            except Exception as exc:
                logger.error(f"[Analyze] 冲突检测失败: {exc}")

        # 动态构建偏好日志
        pref_log_parts: list[str] = [
            f"budget={preference.budget}, taste={preference.taste}",
        ]
        for td in TOOL_REGISTRY:
            if td.preference_field:
                val = getattr(preference, td.preference_field, None)
                pref_log_parts.append(f"{td.preference_field}={'有' if val else '无'}")
        pref_log_parts.append(f"freeform_location={preference.freeform_location or '无'}")
        pref_log_parts.append(f"needs_clarification={needs_clarification}")
        logger.info(f"[Analyze] {', '.join(pref_log_parts)}")

        if needs_clarification:
            self._tracer.record_system_rule_hit(
                "反问触发",
                category="clarification",
                detail=f"问题: {clarification_question[:100]}",
            )

        self._tracer.end_phase(
            phase,
            input_snapshot={
                "system_prompt_len": str(len(system_prompt)),
                "user_prompt": user_prompt[:500],
            },
            output_snapshot=analysis_result,
        )

        return {
            "is_out_of_domain": False,
            "is_chitchat": False,
            "user_preference": preference,
            "individual_preferences": individual_prefs,
            "conflict_strategy": conflict_strategy,
            "detected_conflicts": conflicts,
            "needs_clarification": needs_clarification,
            "clarification_question": clarification_question,
            "freeform_location": preference.freeform_location or "",
            "is_weather_query": False,  # already short-circuited above for weather
            "weather_city": preference.city or "",
            # ── 跨轮次时间上下文：独立于 LLM，直接扫描 raw_query ──
            # 代码负责"时间计算" — 不依赖 LLM 的 aggregated_preference.time
            "last_resolved_time": self._try_resolve_time_context(state.raw_query),
        }

    @staticmethod
    def _has_time_keywords(raw_query: str) -> bool:
        """Detect if user input contains explicit time expressions.

        Independent of LLM output — code detects time keywords directly.
        Follows: 代码负责"时间计算" — the code decides when to extract time,
        not the LLM's aggregated_preference.time field.

        Returns:
            True if the query contains Chinese time expressions worth
            saving for cross-turn reference (e.g. "这周六" → save as context
            for later "当天天气怎么样").
        """
        # 只检测 [用户需求] 部分的原始用户输入，
        # 避免 [位置信息] 前缀中的城市名干扰
        intent: str = LocalLifeWorkflow._extract_user_intent(raw_query)
        time_keywords: list[str] = [
            "今天", "明天", "后天", "大后天",
            "这周", "下周", "下下周", "本周",
            "周一", "周二", "周三", "周四", "周五", "周六", "周日",
            "星期", "周末",
            "今晚", "明晚", "后天晚上",
            "这个月", "下个月", "本月", "月底", "月初",
        ]
        return any(kw in intent for kw in time_keywords)

    @staticmethod
    def _extract_user_intent(raw_query: str) -> str:
        """Strip the [位置信息] wrapper to get the original user input.

        The enhanced_query injected by web_ui.py wraps user text with:
          [位置信息] 搜索城市: XXX
          [用户需求] <original user input>

        LLM time extraction and keyword detection should operate on the
        original user input, not the wrapped form.
        """
        if "[用户需求]" in raw_query:
            # Split on the [用户需求] marker and take everything after it
            parts: list[str] = raw_query.split("[用户需求]", 1)
            if len(parts) > 1:
                return parts[1].strip()
        return raw_query

    def _try_resolve_time_context(
        self, raw_query: str
    ) -> Optional[dict[str, Any]]:
        """Detect and resolve time expressions from raw user input.

        Independent of LLM output — uses keyword detection to decide
        when to extract time, then calls _extract_time_expression (LLM for
        semantic parsing) → TimeResolver (deterministic date computation).

        Saves the resolved TimeContext so "当天天气怎么样" can reference
        the previously discussed date.

        Args:
            raw_query: Original user input text.

        Returns:
            Serialized TimeContext dict or None if no time keywords found.
        """
        if not raw_query or not self._has_time_keywords(raw_query):
            return None
        from datetime import date as date_type
        today: date_type = date_type.today()
        try:
            # 提取 [用户需求] 部分的纯文本，避免 [位置信息] 前缀干扰 LLM 时间提取
            intent: str = self._extract_user_intent(raw_query)
            time_expr: TimeExpression = self._extract_time_expression(
                intent, today.isoformat()
            )
            # Skip trivial/unresolvable types — no useful date to save
            if time_expr.type in ("none",):
                return None
            resolved = self._time_resolver.resolve(time_expr, current_date=today)
            tc: dict[str, Any] = {
                "raw": time_expr.raw,
                "resolved_date": resolved.resolved_date,
                "resolved_datetime": resolved.resolved_datetime,
                "timezone": resolved.timezone,
            }
            logger.info(
                f"[Analyze] time context saved: '{tc['raw']}' → {tc['resolved_date']}"
            )
            return tc
        except Exception as exc:
            logger.warning(f"[Analyze] time context extraction failed: {exc}")
            return None

    # ================================================================
    # 节点 1.5：天气查询
    # ================================================================

    @staticmethod
    def _is_weather_query(raw_query: str) -> bool:
        """Detect if user input is asking about weather via keyword matching.

        Simple and fast — avoids an extra LLM call during the analyze phase.
        """
        weather_keywords: list[str] = [
            "天气", "下雨", "雨", "温度", "气温", "晴", "多云",
            "热不热", "冷不冷", "刮风", "下雪", "湿度", "空气质量",
            "穿什么", "带伞", "伞",
        ]
        return any(kw in raw_query for kw in weather_keywords)

    def _weather_node(self, state: AgentState) -> dict[str, Any]:
        """Weather query node: parse time, call WeatherTool, format response.

        Flow:
        1. Detect if user input has time reference keywords ("当天"/"那天")
        2. If reference detected + last_resolved_time exists → use as prev_context
        3. Extract time semantics from user input (via fast_llm, not date computation)
        4. TimeResolver.resolve() computes actual date (purely deterministic)
        5. WeatherTool.execute() fetches weather data for resolved date
        6. Format natural-language response with weather info
        7. Save resolved TimeContext for cross-turn reference
        """
        from datetime import date as date_type

        today: date_type = date_type.today()
        city: str = state.weather_city or state.user_preference.city or "上海"
        raw_query: str = state.raw_query

        logger.info(
            f"[Weather] query='{raw_query[:60]}', city={city}, today={today.isoformat()}"
        )

        # ── Detect reference keywords ("当天", "那天") ──
        _reference_keywords: list[str] = ["当天", "那天", "这一天", "那天晚上"]
        intent: str = self._extract_user_intent(raw_query)
        has_reference: bool = any(kw in intent for kw in _reference_keywords)

        # ── Build prev_context from last_resolved_time ──
        prev_context: Optional[TimeContext] = None
        if has_reference and state.last_resolved_time:
            try:
                prev_context = TimeContext(**state.last_resolved_time)
                logger.info(
                    f"[Weather] reference detected → using prev_context: "
                    f"date={prev_context.resolved_date}"
                )
            except Exception as exc:
                logger.warning(f"[Weather] failed to parse last_resolved_time: {exc}")

        # ── Step 1: Extract time expression from user input ──
        # 使用纯用户意图文本（不含 [位置信息] 前缀），
        # 确保"当天"等指代词能被 LLM 正确识别为 type="reference"
        time_expr: TimeExpression = self._extract_time_expression(
            intent, today.isoformat()
        )

        # ── Step 1.5: Override with reference type if detected ──
        if has_reference and prev_context:
            # Force the time expression type to "reference" so the
            # TimeResolver uses prev_context instead of computing from today
            time_expr = TimeExpression(
                raw=raw_query,
                type="reference",
            )

        # ── Step 2: Resolve to actual date (backend, NO LLM) ──
        resolved = self._time_resolver.resolve(
            time_expr, current_date=today, prev_context=prev_context
        )
        resolved_date: str = resolved.resolved_date

        logger.info(
            f"[Weather] time: '{time_expr.raw}' ({time_expr.type}) "
            f"→ {resolved_date}"
        )

        # ── Step 3: Call weather tool ──
        weather: dict[str, Any] = self._weather_tool.execute(
            city=city, date=resolved_date
        )

        logger.info(
            f"[Weather] result: {city} {resolved_date} → "
            f"{weather.get('weather', '?')} {weather.get('temperature', '?')}"
        )

        # ── Step 4: Format response ──
        response: str = self._format_weather_response(
            weather=weather,
            resolved_date=resolved_date,
            time_expr=time_expr,
            city=city,
            today=today,
        )

        # ── Step 5: Save TimeContext for cross-turn reference ──
        last_resolved_time: dict[str, Any] = {
            "raw": time_expr.raw,
            "resolved_date": resolved_date,
            "resolved_datetime": resolved.resolved_datetime,
            "timezone": resolved.timezone,
        }

        return {
            "final_recommendation": response,
            "safety_passed": True,
            "weather_result": weather,
            "weather_city": city,
            "last_resolved_time": last_resolved_time,
        }

    def _extract_time_expression(
        self,
        user_input: str,
        today_iso: str,
    ) -> TimeExpression:
        """Use fast_llm to extract time SEMANTICS from user input.

        LLM extracts: type, weekday (ISO), week_offset, day_offset, date_iso, period, hour.
        LLM does NOT compute dates — the backend TimeResolver does that.

        Falls back to TimeExpression(type="today") if LLM call fails.
        """
        prompt: str = (
            f"从用户输入中提取时间语义参数。**不要计算日期**，只输出语义类型和参数。\n\n"
            f"当前日期（参考用）: {today_iso}\n"
            f"用户输入: {user_input}\n\n"
            '输出 JSON 格式:\n'
            '{\n'
            '  "raw": "原始时间表达文本",\n'
            '  "type": "today|tomorrow|day_after_tomorrow|relative_weekday|relative_days|date|period_only|reference|none",\n'
            '  "weekday": ISO星期 (1=周一..7=周日), 仅 relative_weekday,\n'
            '  "week_offset": 0=本周, 1=下周, 2=下下周, 仅 relative_weekday,\n'
            '  "day_offset": N天后, 仅 relative_days,\n'
            '  "date_iso": "YYYY-MM-DD", 仅用户明确说出日期时,\n'
            '  "period": "morning|afternoon|evening|night",\n'
            '  "hour": 0-23 (24h制), 仅用户提到具体时间点\n'
            '}\n\n'
            '**重要规则：**\n'
            '- 如果用户说"当天"、"那天"、"这一天"、"那天晚上"等指代词 → type="reference"\n'
            '- "reference" 表示该时间指代对话上下文中之前提到的某个日期，不要计算\n'
            '只输出 JSON，不要输出其他内容。'
        )

        try:
            result: dict[str, Any] = self._fast_llm.generate_json(
                system_prompt="你是一个时间语义提取器，只提取结构化参数，不计算日期。",
                user_prompt=prompt,
            )
            return TimeExpression(
                raw=result.get("raw", user_input),
                type=result.get("type", "today"),
                weekday=result.get("weekday"),
                week_offset=result.get("week_offset", 0),
                day_offset=result.get("day_offset", 0),
                date_iso=result.get("date_iso", ""),
                period=result.get("period"),
                hour=result.get("hour"),
            )
        except Exception as exc:
            logger.warning(f"[Weather] time extraction failed: {exc}, falling back to today")
            return TimeExpression(raw=user_input, type="today")

    @staticmethod
    def _format_weather_response(
        weather: dict[str, Any],
        resolved_date: str,
        time_expr: TimeExpression,
        city: str,
        today: Any,
    ) -> str:
        """Build a natural Chinese weather response from tool results.

        Uses explicit date formatting (no LLM dependency for the core response).
        """
        from datetime import date as date_type

        weekday_names: list[str] = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

        # Format the resolved date for display
        try:
            d = date_type.fromisoformat(resolved_date)
            date_display: str = f"{d.month}月{d.day}日（{weekday_names[d.weekday()]}）"
        except (ValueError, TypeError):
            date_display = resolved_date

        weather_desc: str = weather.get("weather", "晴")
        temperature: str = weather.get("temperature", "")
        humidity: int = weather.get("humidity", 60)

        # Build weather advice based on conditions
        advice: str = ""
        if "雨" in weather_desc:
            advice = "出门记得带伞哦～"
        elif weather_desc == "晴":
            advice = "阳光不错，适合出行！"
        elif "雪" in weather_desc:
            advice = "天气寒冷，注意保暖～"

        return (
            f"📅 {date_display} {city}天气：**{weather_desc}**，"
            f"气温 {temperature}，湿度 {humidity}%。{advice}"
        )

    # ================================================================
    # 节点 2：闲聊回复
    # ================================================================

    def _chitchat_node(self, state: AgentState) -> dict[str, Any]:
        """闲聊节点：输出 LLM 生成的闲聊回复，跳过全部业务流程。

        仅追加一条 AIMessage 到对话历史。
        """
        reply: str = state.chitchat_reply or state.final_recommendation or (
            "你好！我是本地生活决策助手，可以帮你规划聚餐、推荐餐厅、查找酒店和娱乐场所。"
            "请告诉我你的需求吧～"
        )
        logger.info(f"[Chitchat] → {reply[:60]}...")
        return {
            "messages": [AIMessage(content=reply)],
            "final_recommendation": reply,
            "safety_passed": True,
        }

    # ================================================================
    # 节点 2.5：领域外拒绝
    # ================================================================

    def _out_of_domain_node(self, state: AgentState) -> dict[str, Any]:
        """领域外节点：输出礼貌拒绝回复，跳过全部业务流程。"""
        reply: str = state.out_of_domain_reply or state.final_recommendation or (
            "抱歉，我目前专注于本地生活服务领域（餐饮、住宿、娱乐、购物、交通）。"
            "无法回答与本地生活无关的问题。"
            "请问有什么本地生活方面的需求我可以帮您吗？"
        )
        logger.info(f"[OutOfDomain] → {reply[:60]}...")
        return {
            "messages": [AIMessage(content=reply)],
            "final_recommendation": reply,
            "safety_passed": True,
        }

    # ================================================================
    # 节点 3：中断反问
    # ================================================================

    def _clarify_node(self, state: AgentState) -> dict[str, Any]:
        """澄清节点：将反问问题追加到消息历史，暂停本轮流转。

        不执行后续检索/推荐/安全流程，直接返回澄清问题让用户回答。
        """
        question: str = state.clarification_question or (
            "请您补充更多信息，例如预算范围、口味偏好或出行人数。"
        )
        logger.info(f"[Clarify] 反问用户: {question[:80]}...")

        ai_message: AIMessage = AIMessage(content=question)
        return {
            "messages": [ai_message],
            "final_recommendation": question,
            "safety_passed": True,
        }

    # ================================================================
    # 节点 4：工具检索 (v3.1 — 并行化)
    # ================================================================

    def _retrieve_node(self, state: AgentState) -> dict[str, Any]:
        """检索节点：并行调用注册的工具获取候选数据。

        v3.1 升级：使用 ThreadPoolExecutor 并行执行多个 I/O 密集型
        工具调用（餐厅 / 酒店 / 娱乐 / 购物 / 交通），总耗时 ≈ 最慢的
        单次 API 调用，而非串行累加。

        vNext：跨轮上下文保留 — 检索前保存前轮快照到 previous_candidates。
        """
        phase: str = "retrieve"
        self._tracer.start_phase(phase)

        pref: UserPreference = state.user_preference
        updates: dict[str, Any] = {}

        # ── vNext：保存前轮候选快照（跨轮上下文回退用）──
        previous_snapshot: dict[str, list[dict[str, Any]]] = {}
        for td in TOOL_REGISTRY:
            current_val = getattr(state, td.candidate_key, [])
            if current_val:
                previous_snapshot[td.candidate_key] = list(current_val)
        if state.candidate_transit_directions:
            previous_snapshot["candidate_transit_directions"] = list(
                state.candidate_transit_directions
            )
        updates["previous_candidates"] = previous_snapshot

        # ── v3.3 容错：分析失败降级时，从原始 query 做关键词触发 ──
        # 避免因偏好提取出错导致所有非餐饮领域被跳过
        # vNext：触发逻辑由 ToolRegistry 统一管理
        raw_lower: str = state.raw_query.lower() if state.raw_query else ""
        has_ind_prefs: bool = len(state.individual_preferences or []) >= 1
        triggered_tools: list[ToolDefinition] = get_triggered_tools(
            TOOL_REGISTRY, pref, state.raw_query, has_ind_prefs
        )

        # ── 构建并行任务列表 ──
        from concurrent.futures import Future, ThreadPoolExecutor, as_completed

        # 每个任务：(tool, task_key, keyword_args)
        tasks: list[tuple[Optional[ITool], str, dict[str, Any]]] = []

        # 遍历 registry，动态构建所有工具任务
        for td in TOOL_REGISTRY:
            tool: Optional[ITool] = self._tools.get(td.tool_attr)
            should_run: bool = td in triggered_tools

            if should_run and tool:
                params: dict[str, Any] = (
                    td.search_params_builder(pref, state)
                    if td.search_params_builder
                    else {}
                )
                tasks.append((tool, td.candidate_key, params))
            else:
                reason: str = (
                    f"无{td.display_label}需求"
                    if not should_run
                    else "工具未注入"
                )
                self._tracer.record_phase_skip(
                    f"retrieve_{td.key}",
                    reason=reason,
                )

        # ── 并行执行 ──
        logger.info(f"[Retrieve] 启动 {len(tasks)} 个并行检索任务")
        with ThreadPoolExecutor(max_workers=max(1, len(tasks))) as executor:
            future_map: dict[Future, tuple[str, str]] = {}
            for tool, key, kwargs in tasks:
                tool_name: str = tool.get_name() if tool else "unknown"
                f: Future = executor.submit(tool.execute, **kwargs) if tool else None  # type: ignore[union-attr]
                if isinstance(f, Future):
                    future_map[f] = (key, tool_name)

            for future in as_completed(future_map):
                key, tool_name = future_map[future]
                try:
                    result: list[dict[str, Any]] = future.result()
                    updates[key] = result
                    logger.info(
                        f"[Retrieve] {key}: {len(result)} 个候选 "
                        f"(source={result[0].get('source', '?') if result else 'none'})"
                    )
                    self._tracer.record_tool_call(
                        tool_name=tool_name,
                        params={"parallel_batch": len(tasks)},
                        result_summary={"count": len(result)},
                    )
                except Exception as exc:
                    logger.error(f"[Retrieve] {key} 并行搜索失败: {exc}")
                    updates[key] = []

        # ── v3.3：公交路径规划（有餐厅坐标 + 交通站点时触发）──
        # 修复：origin 使用 transit stop 经纬度，destination 使用餐厅经纬度
        # 用户需求"从X到餐厅怎么坐公交"意味着 origin=公交站/出发点, dest=餐厅
        directions: list[dict[str, Any]] = []
        if self._transit_direction_tool:
            restaurants: list[dict[str, Any]] = updates.get(
                "candidate_restaurants", []
            )
            transit_stops: list[dict[str, Any]] = updates.get(
                "candidate_transit_stops", []
            )
            if restaurants and transit_stops:
                top_restaurant: dict[str, Any] = restaurants[0]
                dest_name: str = top_restaurant.get("name", "")
                dest_lon = top_restaurant.get("longitude")
                dest_lat = top_restaurant.get("latitude")
                # 高德要求 origin=起点经纬度, destination=终点经纬度
                dest_loc: str = f"{dest_lon},{dest_lat}" if dest_lon and dest_lat else ""

                # ── 为 transit stops 有经纬度的站点生成路线 ──
                for ts in transit_stops[:3]:
                    stop_name: str = ts.get("name", "")
                    s_lon = ts.get("longitude")
                    s_lat = ts.get("latitude")
                    stop_loc: str = f"{s_lon},{s_lat}" if s_lon and s_lat else ""

                    if not stop_loc or not dest_loc:
                        continue

                    # ── 尝试 origin=站点, destination=餐厅 ──
                    try:
                        result: dict[str, Any] = self._transit_direction_tool.get_direction(
                            origin=stop_loc,
                            destination=dest_loc,
                            city=pref.city or getattr(self._transit_direction_tool, '_city', '北京'),
                        )
                        if result.get("success"):
                            result["_from"] = stop_name
                            result["_to"] = dest_name
                            directions.append(result)
                            logger.info(
                                f"[Retrieve] 公交路径规划: {stop_name} → {dest_name}"
                            )
                        else:
                            logger.debug(
                                f"[Retrieve] 路径规划无结果: {stop_name} → {dest_name}"
                            )
                    except Exception as exc:
                        logger.debug(f"[Retrieve] 路径规划异常: {exc}")

                # ── 如果有 individual_preferences 且含 origin_point，也尝试生成 ──
                for ip in (state.individual_preferences or []):
                    origin_pt = ip.origin_point
                    if not origin_pt or not dest_loc:
                        continue
                    # 尝试用地点名作为 origin（高德 API 同时支持名称和坐标）
                    try:
                        result = self._transit_direction_tool.get_direction(
                            origin=origin_pt,
                            destination=dest_loc,
                            city=pref.city or getattr(self._transit_direction_tool, '_city', '北京'),
                        )
                        if result.get("success"):
                            result["_from"] = f"{ip.name}({origin_pt})"
                            result["_to"] = dest_name
                            directions.append(result)
                            logger.info(
                                f"[Retrieve] 用户路径规划: {ip.name}({origin_pt}) → {dest_name}"
                            )
                    except Exception as exc:
                        logger.debug(f"[Retrieve] 用户路径规划异常 ({ip.name}): {exc}")
            updates["candidate_transit_directions"] = directions

        self._tracer.end_phase(
            phase,
            input_snapshot={"parallel_tasks": len(tasks)},
            output_snapshot={
                "restaurants": len(updates.get("candidate_restaurants", [])),
                "hotels": len(updates.get("candidate_hotels", [])),
                "entertainments": len(updates.get("candidate_entertainments", [])),
            },
        )

        return updates

    # ================================================================
    # 节点 5：推荐生成
    # ================================================================

    def _recommend_node(self, state: AgentState) -> dict[str, Any]:
        """推荐节点：基于候选数据和用户偏好生成时空推理推荐。

        v3.1：当 state._streaming_mode=True 时跳过 LLM 调用（由 run_stream() 外部处理）。
        """
        phase: str = "recommend"
        self._tracer.start_phase(phase)

        if not state.candidate_restaurants:
            logger.warning("[Recommend] 无候选餐厅")
            self._tracer.end_phase(
                phase,
                input_snapshot={"candidates": 0},
                output_snapshot={"fallback": True},
            )
            return {
                "final_recommendation": "未能找到符合条件的餐厅，请尝试调整搜索条件。"
            }

        # ── v3.1：流式模式 → 跳过同步 LLM 调用，由 run_stream() 处理 ──
        if getattr(state, "streaming_mode", False):
            logger.info("[Recommend] 流式模式，跳过同步 LLM 调用")
            self._tracer.end_phase(
                phase,
                input_snapshot={"candidates": len(state.candidate_restaurants)},
                output_snapshot={"streaming_delegated": True},
            )
            return {
                "final_recommendation": ""  # 占位，run_stream() 负责填充
            }

        # 构建 v2.1 上下文（vNext：由 ToolRegistry 动态构建）
        # v3.2: 纳入个人偏好和冲突信息供推荐决策
        individual_budgets: list[dict[str, Any]] = []
        for ip in (state.individual_preferences or []):
            individual_budgets.append({
                "name": ip.name,
                "budget": ip.budget,
                "taste": ip.taste,
                "restrictions": ip.restrictions,
            })
        context: dict[str, Any] = build_context_dict(TOOL_REGISTRY, state)

        system_prompt: str = PromptManager.build_recommendation_prompt(context)

        pref: UserPreference = state.user_preference
        pref_dict: dict[str, Any] = build_pref_summary_dict(TOOL_REGISTRY, pref)
        pref_dict["conflict_strategy"] = state.conflict_strategy
        pref_summary: str = json.dumps(pref_dict, ensure_ascii=False)
        # v3.2: 显式传入每人预算
        budget_note: str = ""
        if individual_budgets:
            budget_note = (
                "\n\n## 💰 每人预算（必须据此计算分摊方案）\n"
                + json.dumps(individual_budgets, ensure_ascii=False)
            )
        user_prompt: str = (
            f"## 用户偏好\n```json\n{pref_summary}\n```\n\n"
            "请基于以上用户偏好和候选数据，生成个性化推荐。"
            + budget_note
        )

        try:
            recommendation: str = self._llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            logger.error(f"[Recommend] LLM 调用失败: {exc}")
            recommendation = "抱歉，生成推荐时遇到了技术问题，请稍后再试。"

        logger.info(f"[Recommend] response len: {len(recommendation)}")

        self._tracer.end_phase(
            phase,
            input_snapshot={
                "restaurants": len(state.candidate_restaurants),
                "hotels": len(state.candidate_hotels),
                "entertainments": len(state.candidate_entertainments),
                "preference": pref_summary[:300],
            },
            output_snapshot={
                "recommendation_length": len(recommendation),
                "recommendation_preview": recommendation[:200],
            },
        )

        return {"final_recommendation": recommendation}

    # ================================================================
    # 节点 6：安全审查
    # ================================================================

    def _safety_guard_node(self, state: AgentState) -> dict[str, Any]:
        """安全审查节点：履约前合规审查与话术改写。

        v3.1：当 state._streaming_mode=True 时跳过（由 run_stream() 外部处理）。
        """
        phase: str = "safety"
        self._tracer.start_phase(phase)

        # ── v3.1：流式模式 → 跳过，由 run_stream() 处理 ──
        if getattr(state, "streaming_mode", False):
            logger.info("[Safety] 流式模式，跳过（由 run_stream 处理）")
            self._tracer.end_phase(
                phase,
                input_snapshot={"recommendation": ""},
                output_snapshot={"streaming_delegated": True},
            )
            return {"safety_passed": True}

        if not state.final_recommendation:
            logger.warning("[Safety] 无推荐文本可审查")
            self._tracer.end_phase(
                phase,
                input_snapshot={"recommendation": ""},
                output_snapshot={"passed": True, "reason": "empty_input"},
            )
            return {"safety_passed": True}

        # ── v4.0: 程序化预筛选 ──
        prefilter_hits: list[dict[str, Any]] = SafetyPrefilter.scan(
            state.final_recommendation
        )
        if prefilter_hits:
            logger.warning(
                f"[Safety] 预筛选命中 {len(prefilter_hits)} 条规则: "
                f"{[h['rule_id'] for h in prefilter_hits]}"
            )
            for hit in prefilter_hits:
                self._tracer.record_safety_hit(
                    rule_name=f"prefilter:{hit['rule_id']}",
                    detail=f"{hit['category']}: 匹配 \"{hit['match_text']}\"",
                )

        system_prompt: str = PromptManager.build_safety_guard_prompt()
        user_prompt: str = (
            "请审查以下推荐文本是否触碰安全红线，并按要求输出 JSON：\n\n"
            f"{state.final_recommendation}"
        )

        # 注入预筛选命中上下文
        prefilter_context: str = SafetyPrefilter.build_context_hint(prefilter_hits)
        if prefilter_context:
            user_prompt = prefilter_context + "\n" + user_prompt

        try:
            safety_result: dict[str, Any] = self._fast_llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            logger.error(f"[Safety] LLM 调用失败: {exc}")
            self._tracer.end_phase(
                phase,
                input_snapshot={"recommendation_len": len(state.final_recommendation)},
                output_snapshot={"error": str(exc)},
            )
            return {"safety_passed": True}

        passed: bool = safety_result.get("passed", True)
        violations: list[str] = safety_result.get("violations", [])
        safe_output: str = safety_result.get(
            "output", state.final_recommendation
        )

        if not passed:
            logger.warning(f"[Safety] violations: {violations}")
            for violation in violations:
                self._tracer.record_safety_hit(
                    rule_name=violation,
                    detail=f"推荐文本触发安全规则: {violation}",
                )
        else:
            logger.info("[Safety] 审查通过")

        self._tracer.end_phase(
            phase,
            input_snapshot={
                "original_length": len(state.final_recommendation)
            },
            output_snapshot={
                "passed": passed,
                "violations": violations,
                "was_rewritten": not passed,
                "prefilter_hits": len(prefilter_hits),
            },
        )

        # ── v4.0: 审计日志持久化 ──
        try:
            # 提取 thread_id（从 state 中不可直接获取，用占位）
            self._audit_logger.log_safety_decision(
                session_id=getattr(state, "raw_query", "")[:40] or "unknown",
                original_text=state.final_recommendation,
                passed=passed,
                violations=violations,
                rewritten_text=safe_output if not passed else "",
                prefilter_hits=prefilter_hits,
            )
        except Exception as exc:
            logger.error(f"[Safety] 审计日志写入失败: {exc}")

        # 构建 AI 回复消息
        ai_message: AIMessage = AIMessage(content=safe_output)

        return {
            "final_recommendation": safe_output,
            "safety_passed": passed,
            "safety_violations": violations,
            "messages": [ai_message],
        }

    # ================================================================
    # 公共对话接口
    # ================================================================

    def chat(self, user_input: str, thread_id: str = "default") -> AgentState:
        """向图引擎输入新消息，利用 thread_id 提取历史上下文进行连续对话。

        Args:
            user_input: 用户当前轮次的输入文本。
            thread_id: 对话线程 ID。相同 thread_id 共享 MemorySaver 中的历史状态。
                       不同 thread_id 的对话完全隔离。

        Returns:
            本轮执行结束后的最终 AgentState（包含回复文案、偏好、候选等）。
        """
        logger.info(
            f"[Workflow] chat: thread={thread_id}, input={user_input[:80]}..."
        )
        self._tracer.set_query(user_input)

        # 构建 HumanMessage 输入
        human_message: HumanMessage = HumanMessage(content=user_input)

        # 配置：指定 thread_id 用于 MemorySaver 的 checkpoint 存取
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id}
        }

        # ── 调用 LangGraph 图引擎 ──
        try:
            result: dict[str, Any] = self._app.invoke(
                {"messages": [human_message], "raw_query": user_input},
                config=config,
            )
        except Exception as exc:
            logger.error(f"[Workflow] 图执行异常: {exc}")
            self._tracer.record_failure(
                phase="workflow",
                error_type=type(exc).__name__,
                detail=str(exc),
            )
            return AgentState(
                raw_query=user_input,
                final_recommendation="抱歉，系统在处理您的问题时遇到了技术问题，请稍后再试。",
                safety_passed=True,
            )

        # 构建返回的 AgentState
        state: AgentState = AgentState(**result)
        state.trace_logs = self._tracer.get_logs()

        self._tracer.dump_report()
        logger.info(
            f"[Workflow] chat done: needs_clarification={state.needs_clarification}, "
            f"candidates={len(state.candidate_restaurants)}, "
            f"reply_len={len(state.final_recommendation)}"
        )
        return state

    def run_stream(
        self,
        user_input: str,
        thread_id: str = "default",
    ) -> Generator[tuple[str, AgentState | None], None, None]:
        """流式运行工作流 —— 推荐阶段逐 Token yield（v3.1 新增）。

        执行流程：
        1. 同步运行 analyze → route → (chitchat|clarify|out_of_domain) → retrieve
        2. 若进入推荐阶段，使用 generate_stream() 逐 Token yield
           yield 格式：(token, None)
        3. 安全审查（同步）
        4. yield 最终结果：(safe_output, final_state)

        简化词/非同义词/领域外查询会走快捷路径，不会进入推荐阶段。
        此时 yield 一条完整回复，然后 yield (finished, state)。

        Args:
            user_input: 用户输入文本。
            thread_id: 对话线程 ID。

        Yields:
            (text_chunk, None) —— 增量 token，或
            (full_text, state) —— 最终完整回复 + 已完成的 AgentState
        """
        logger.info(
            f"[Workflow] run_stream: thread={thread_id}, input={user_input[:80]}..."
        )
        self._tracer.set_query(user_input)

        human_message: HumanMessage = HumanMessage(content=user_input)
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

        # ── 第一阶段：后台线程执行 analyze → retrieve ──
        # invoke() 可能耗时 30s+（分析 LLM），期间前端 SSE 连接需要心跳防止超时
        invoke_result: dict[str, Any] | None = None
        invoke_error: Exception | None = None
        invoke_done: threading.Event = threading.Event()

        def _blocking_invoke() -> None:
            nonlocal invoke_result, invoke_error
            try:
                invoke_result = self._app.invoke(
                    {
                        "messages": [human_message],
                        "raw_query": user_input,
                        "streaming_mode": True,
                    },
                    config=config,
                )
            except Exception as exc:
                invoke_error = exc
            finally:
                invoke_done.set()

        bg_thread: threading.Thread = threading.Thread(
            target=_blocking_invoke, daemon=True
        )
        bg_thread.start()

        # ── 心跳循环：每 HEARTBEAT_INTERVAL 秒 yield 一个心跳 token，保持 SSE 连接活跃 ──
        while not invoke_done.wait(timeout=STREAM_HEARTBEAT_INTERVAL):
            yield ("", None)  # 空 token = 心跳，Gradio 保持连接但不显示文字

        # 后台线程完成，检查结果
        if invoke_error is not None:
            exc = invoke_error
            logger.error(f"[Workflow] 图执行异常: {exc}")
            self._tracer.record_failure(
                phase="workflow",
                error_type=type(exc).__name__,
                detail=str(exc),
            )
            fallback: str = "抱歉，系统在处理您的问题时遇到了技术问题，请稍后再试。"
            yield (fallback, None)
            yield ("", AgentState(
                raw_query=user_input,
                final_recommendation=fallback,
                safety_passed=True,
            ))
            return

        result = invoke_result

        state: AgentState = AgentState(**result)

        # 闲聊 / 领域外 / 澄清 / 天气 —— 直接返回完整文本，无需流式
        if (
            state.is_chitchat
            or state.is_out_of_domain
            or state.needs_clarification
            or state.is_weather_query
        ):
            reply: str = state.final_recommendation or ""
            # 追加 AI 消息到历史（chitchat/ood/clarify 节点已经生成了回复）
            state.trace_logs = self._tracer.get_logs()
            self._tracer.dump_report()
            yield (reply, None)
            yield ("", state)
            return

        # ── 第二阶段：流式推荐生成 ──
        if not state.candidate_restaurants:
            logger.warning("[Workflow] 流式 —— 无候选餐厅")
            no_result = "未能找到符合条件的餐厅，请尝试调整搜索条件。"
            state.final_recommendation = no_result
            state.trace_logs = self._tracer.get_logs()
            self._tracer.dump_report()
            yield (no_result, None)
            yield ("", state)
            return

        # 构建推荐上下文（vNext：由 ToolRegistry 动态构建）
        individual_budgets: list[dict[str, Any]] = []
        for ip in (state.individual_preferences or []):
            individual_budgets.append({
                "name": ip.name,
                "budget": ip.budget,
                "taste": ip.taste,
                "restrictions": ip.restrictions,
            })
        context: dict[str, Any] = build_context_dict(TOOL_REGISTRY, state)
        rec_system_prompt: str = PromptManager.build_recommendation_prompt(context)

        pref_dict: dict[str, Any] = build_pref_summary_dict(
            TOOL_REGISTRY, state.user_preference
        )
        pref_dict["conflict_strategy"] = state.conflict_strategy
        pref_summary: str = json.dumps(pref_dict, ensure_ascii=False)
        # v3.2: 显式传入每人预算
        budget_note: str = ""
        if individual_budgets:
            budget_note = (
                "\n\n## 💰 每人预算（必须据此计算分摊方案）\n"
                + json.dumps(individual_budgets, ensure_ascii=False)
            )
        rec_user_prompt: str = (
            f"## 用户偏好\n```json\n{pref_summary}\n```\n\n"
            "请基于以上用户偏好和候选数据，生成个性化推荐。"
            + budget_note
        )

        # ── 流式生成推荐文本 ──
        recommendation_parts: list[str] = []
        try:
            for token in self._llm.generate_stream(
                system_prompt=rec_system_prompt,
                user_prompt=rec_user_prompt,
            ):
                recommendation_parts.append(token)
                yield (token, None)
        except Exception as exc:
            logger.error(f"[Workflow] 流式推荐生成失败: {exc}")
            err_msg: str = "抱歉，生成推荐时遇到了技术问题，请稍后再试。"
            yield (err_msg, None)
            state.final_recommendation = err_msg
            state.trace_logs = self._tracer.get_logs()
            self._tracer.dump_report()
            yield ("", state)
            return

        recommendation: str = "".join(recommendation_parts)
        logger.info(f"[Workflow] 流式推荐长度: {len(recommendation)}")

        # ── 第三阶段：安全审查（后台线程 + 心跳）──
        # 流式 token 结束后，安全审查 generate_json() 可能阻塞 5-10s，
        # 期间如果不 yield 任何内容，浏览器 SSE 会超时断开。
        # 解决方案：generate_json() 在后台线程运行，主线程定期 yield 心跳。
        safe_output: str = recommendation
        safety_passed: bool = True
        safety_violations: list[str] = []
        safety_done: bool = False
        safety_error: str | None = None

        def _run_safety_check() -> None:
            """后台执行安全审查（程序化预筛选 + LLM JSON 审查）。"""
            nonlocal safe_output, safety_passed, safety_violations, safety_done, safety_error
            try:
                # 程序化预筛选
                prefilter_hits: list[dict[str, Any]] = SafetyPrefilter.scan(recommendation)
                if prefilter_hits:
                    logger.warning(
                        f"[Safety] 预筛选命中 {len(prefilter_hits)} 条规则: "
                        f"{[h['rule_id'] for h in prefilter_hits]}"
                    )
                    for hit in prefilter_hits:
                        self._tracer.record_safety_hit(
                            rule_name=f"prefilter:{hit['rule_id']}",
                            detail=f"{hit['category']}: 匹配 \"{hit['match_text']}\"",
                        )

                safety_system_prompt: str = PromptManager.build_safety_guard_prompt()
                safety_user_prompt: str = (
                    "请审查以下推荐文本是否触碰安全红线，并按要求输出 JSON：\n\n"
                    f"{recommendation}"
                )
                prefilter_context: str = SafetyPrefilter.build_context_hint(prefilter_hits)
                if prefilter_context:
                    safety_user_prompt = prefilter_context + "\n" + safety_user_prompt

                safety_result: dict[str, Any] = self._fast_llm.generate_json(
                    system_prompt=safety_system_prompt,
                    user_prompt=safety_user_prompt,
                )
                safety_passed = safety_result.get("passed", True)
                safety_violations = safety_result.get("violations", [])
                if not safety_passed:
                    safe_output = safety_result.get("output", recommendation)
                    logger.warning(f"[Safety] violations: {safety_violations}")
            except Exception as exc:
                logger.error(f"[Safety] LLM 调用失败: {exc}")
                safety_error = str(exc)
            finally:
                safety_done = True

        safety_thread = threading.Thread(target=_run_safety_check, daemon=True)
        safety_thread.start()

        # ── 心跳循环：每 500ms yield 心跳，防止 SSE 超时 ──
        while not safety_done:
            safety_thread.join(timeout=0.5)
            if not safety_done:
                yield ("", None)  # 心跳

        # 构建最终状态
        ai_message: AIMessage = AIMessage(content=safe_output)
        state.final_recommendation = safe_output
        state.safety_passed = safety_passed
        state.safety_violations = safety_violations
        state.messages = [ai_message]  # type: ignore[assignment]
        state.trace_logs = self._tracer.get_logs()

        self._tracer.dump_report()
        logger.info(
            f"[Workflow] run_stream done: candidates={len(state.candidate_restaurants)}, "
            f"reply_len={len(safe_output)}"
        )

        # 如果安全审查改写了文本，需要告知 UI 最后的改写
        if not safety_passed and safe_output != recommendation:
            # yield 改写后的最终版本
            yield (f"\n\n---\n⚠️ 安全审查后改写：\n\n{safe_output}", None)

        yield ("", state)

    # ================================================================
    # 兼容旧接口
    # ================================================================

    def run(
        self, user_query: str, city: str | None = None,
        location_detail: str | None = None, thread_id: str = "default"
    ) -> AgentState:
        """兼容旧版 run() 接口，内部委托 chat() 实现。

        Args:
            user_query: 用户查询文本。
            city: 用户从 Web UI 下拉菜单选择的城市（将在 analyze 后覆盖）。
            location_detail: v2.1 新增：用户自由输入的精确位置（建筑名、地标等）。
            thread_id: 对话线程 ID。

        Returns:
            包含全流程结果的 AgentState。
        """
        if location_detail and location_detail.strip():
            if city:
                enhanced_query: str = (
                    f"[位置信息] 搜索城市: {city}, "
                    f"精确位置: {location_detail.strip()}\n"
                    f"[用户需求] {user_query}"
                )
            else:
                enhanced_query: str = (
                    f"[位置信息] 精确位置: {location_detail.strip()}\n"
                    f"[用户需求] {user_query}"
                )
        elif city:
            # 将城市前缀拼入查询，同时保留位置信息给 analyze 节点提取
            enhanced_query: str = (
                f"[位置信息] 搜索城市: {city}\n[用户需求] {user_query}"
            )
        else:
            enhanced_query = user_query

        return self.chat(enhanced_query, thread_id=thread_id)


# ================================================================
# 工具函数
# ================================================================

def _format_conversation(messages: list[Any]) -> str:
    """将 LangChain 消息列表格式化为可读的对话文本。

    Args:
        messages: LangChain message 对象列表（HumanMessage / AIMessage / SystemMessage）。

    Returns:
        格式化的多轮对话文本，用于 LLM 分析 Prompt。
    """
    if not messages:
        return "（新对话，无历史记录）"

    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role: str = "用户"
        elif isinstance(msg, AIMessage):
            role = "Agent"
        elif isinstance(msg, SystemMessage):
            role = "系统"
        else:
            role = type(msg).__name__

        content: str = msg.content if hasattr(msg, "content") else str(msg)
        lines.append(f"[{role}]: {content}")

    return "\n".join(lines)
