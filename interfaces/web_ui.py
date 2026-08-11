"""
Web UI 界面 (Gradio Web Interface) — v5.0 极简可靠版

本模块为 local_life_agent 提供基于 Gradio 的 Web 图形界面。
v5.0 设计原则：
- 零 Gradio 内部 class 选择器依赖（兼容 Gradio 5/6 及 Safari）
- 暖调配色通过 CSS 变量注入
- 三栏布局完全依赖 Gradio Row/Column API（elem_id 仅用于配色）
- 移除隐藏桥接组件（visible=False，不渲染到 DOM）
- 会话管理通过 Gradio 原生 Dropdown change 事件实现

运行方式：
    cd /path/to/local_life_agent
    python interfaces/web_ui.py
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Generator

# 将项目根目录加入 Python 搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
from dotenv import load_dotenv
from loguru import logger

from application.ports import ILLMClient, ITool
from application.workflow import LocalLifeWorkflow
from application.config import (
    WEBUI_SERVER_NAME, WEBUI_SERVER_PORT, WEBUI_CONCURRENCY_LIMIT,
    WEBUI_SHARE, GRADIO_THEME, MAX_WINDOW_MESSAGES,
    ensure_directories,
)
from domain.entities import AgentState
from domain.tool_registry import TOOL_REGISTRY, build_context_dict, get_display_domains, get_preference_tools
from domain.sharing import generate_share_card
from infrastructure.district_loader import DistrictLoader
from infrastructure.llm_adapter import OpenAILikeClient
from infrastructure.amap_tool import AmapRestaurantTool
from infrastructure.amap_hotel_tool import AmapHotelTool
from infrastructure.amap_entertainment_tool import AmapEntertainmentTool
from infrastructure.amap_shopping_tool import AmapShoppingTool
from infrastructure.amap_transit_tool import AmapTransitTool
from infrastructure.amap_parking_tool import AmapParkingTool
from infrastructure.amap_bike_tool import AmapBikeTool
from infrastructure.amap_transit_direction import AmapTransitDirectionTool
from restaurant_agent.tools.weather_tool import WeatherTool
from infrastructure.itinerary_extractor import ItineraryExtractor
from infrastructure.reminder_bridge import create_meal_reminder_params
from infrastructure.db import get_connection, _to_json, _from_json

# 配置日志
logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>"
    ),
    level="INFO",
    colorize=True,
)


# ══════════════════════════════════════════════════════════════
# 全局实例
# ══════════════════════════════════════════════════════════════

_workflow: LocalLifeWorkflow | None = None
_district_loader: DistrictLoader | None = None


def _get_workflow() -> LocalLifeWorkflow:
    """获取或创建全局工作流单例。

    v3.1 升级：创建双 LLM 客户端（smart + fast）用于模型分级路由。
    smart_llm：用于推荐生成（核心质量任务）
    fast_llm：用于分析、分类、安全审查（轻量分类任务）
    """
    global _workflow
    if _workflow is None:
        load_dotenv()
        # ── 智能模型：推荐生成（核心质量） ──
        smart_llm: ILLMClient = OpenAILikeClient()
        # ── 快速模型：分析 / 分类 / 安全审查（使用阿里百炼 Qwen）──
        fast_llm: ILLMClient = OpenAILikeClient(
            model=os.getenv("FAST_MODEL", "qwen-turbo"),
            api_key=os.getenv("FAST_API_KEY", os.getenv("QWEN_API_KEY", "")),
            base_url=os.getenv("FAST_BASE_URL", os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")),
        )
        # ── 工具类映射（registry key → 具体类）──
        _tool_cls_map: dict[str, type] = {
            "restaurant": AmapRestaurantTool,
            "hotel": AmapHotelTool,
            "entertainment": AmapEntertainmentTool,
            "shopping": AmapShoppingTool,
            "transit": AmapTransitTool,
            "parking": AmapParkingTool,
            "bike": AmapBikeTool,
        }
        # ── 遍历 registry 自动构建 tools dict ──
        tools: dict[str, ITool] = {}
        tool_names: list[str] = []
        for td in TOOL_REGISTRY:
            cls = _tool_cls_map.get(td.key)
            if cls is not None:
                tools[td.tool_attr] = cls()
                tool_names.append(td.display_label)
        transit_direction_tool = AmapTransitDirectionTool()
        weather_tool = WeatherTool()
        _workflow = LocalLifeWorkflow(
            llm=smart_llm,
            fast_llm=fast_llm,
            tools=tools,
            transit_direction_tool=transit_direction_tool,
            weather_tool=weather_tool,
            max_window_messages=MAX_WINDOW_MESSAGES,
        )
        logger.info(
            f"[WebUI] 工作流实例已初始化 "
            f"(smart={smart_llm.model_name}, fast={fast_llm.model_name}"
            f" + {' + '.join(tool_names)}, Window={MAX_WINDOW_MESSAGES})"
        )
    return _workflow


def _get_district_loader() -> DistrictLoader:
    """获取或创建全局区域数据加载器单例。"""
    global _district_loader
    if _district_loader is None:
        load_dotenv()
        _district_loader = DistrictLoader()
        logger.info("[WebUI] 区域加载器已初始化")
    return _district_loader


# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
# 多会话管理
# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

def _generate_thread_id() -> str:
    """生成唯一的对话线程 ID。"""
    return f"session_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"


def _build_dropdown_choices(store: dict[str, dict]) -> list[tuple[str, str]]:
    """从会话存储构建 Dropdown 选项列表。"""
    if not store:
        return []
    sorted_items = sorted(
        store.items(),
        key=lambda kv: kv[1].get("created_at", 0),
        reverse=True,
    )
    return [(data.get("title", "未命名")[:30], tid) for tid, data in sorted_items]


def _ensure_active_conversation(
    store: dict[str, dict],
    current_tid: str,
) -> tuple[dict[str, dict], str]:
    """确保存在一个活跃会话。无会话时自动创建。"""
    store = dict(store) if store else {}
    if not store:
        new_id = _generate_thread_id()
        store[new_id] = {
            "title": "新对话",
            "messages": [],
            "created_at": time.time(),
        }
        return store, new_id
    if current_tid not in store:
        latest = max(store.keys(), key=lambda k: store[k].get("created_at", 0))
        return store, latest
    return store, current_tid


def _auto_title(store: dict[str, dict], tid: str) -> dict[str, dict]:
    """从第一条用户消息自动生成会话标题。"""
    store = dict(store)
    if tid in store and store[tid].get("title") == "新对话":
        messages = store[tid].get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                store[tid]["title"] = content[:20] + (
                    "..." if len(content) > 20 else ""
                )
                break
    return store


# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
# v3.1：会话持久化
# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

def _load_sessions() -> dict[str, dict]:
    """从 SQLite 加载所有会话。失败时返回空字典。"""
    try:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT session_id, data_json FROM sessions"
            ).fetchall()
        store = {
            row["session_id"]: _from_json(row["data_json"])
            for row in rows
        }
        logger.info(f"[Persistence] 加载 {len(store)} 个会话")
        return store
    except Exception as exc:
        logger.warning(f"[Persistence] 加载失败: {exc}")
    return {}


def _save_sessions(store: dict[str, dict]) -> None:
    """将会话存储保存到 SQLite。"""
    try:
        with get_connection() as conn:
            for session_id, data in store.items():
                conn.execute(
                    "INSERT INTO sessions (session_id, data_json, updated_at) "
                    "VALUES (?, ?, datetime('now', 'localtime')) "
                    "ON CONFLICT(session_id) DO UPDATE SET "
                    "data_json = excluded.data_json, "
                    "updated_at = excluded.updated_at",
                    (session_id, _to_json(data)),
                )
            conn.commit()
    except Exception as exc:
        logger.warning(f"[Persistence] 保存失败: {exc}")


# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
# 会话管理回调
# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

def _on_new_conversation(
    store_state: dict[str, dict] | None,
    _current_tid: str,
) -> tuple[dict[str, dict], str, list[dict[str, str]], gr.Dropdown]:
    """新建对话。"""
    store = dict(store_state) if store_state else {}

    # 若已有空白对话，直接切换
    for tid, data in store.items():
        if not data.get("messages"):
            logger.info(f"[WebUI] 存在空白对话，直接切换: {tid}")
            choices = _build_dropdown_choices(store)
            return store, tid, [], gr.update(choices=choices, value=tid)

    new_id = _generate_thread_id()
    store[new_id] = {
        "title": "新对话",
        "messages": [],
        "created_at": time.time(),
    }
    logger.info(f"[WebUI] 新建对话: {new_id}")
    _save_sessions(store)
    choices = _build_dropdown_choices(store)
    return store, new_id, [], gr.update(choices=choices, value=new_id)


def _on_select_conversation(
    selected_id: str,
    store_state: dict[str, dict] | None,
) -> tuple[dict[str, dict], str, list[dict[str, str]], gr.Dropdown]:
    """切换对话。"""
    if not selected_id:
        return _on_new_conversation(store_state, "")

    store = dict(store_state) if store_state else {}
    if selected_id not in store:
        return _on_new_conversation(store, "")

    conv = store[selected_id]
    messages = list(conv.get("messages", []))
    logger.info(
        f"[WebUI] 切换对话 → {conv.get('title', '?')} ({len(messages)} 条消息)"
    )
    choices = _build_dropdown_choices(store)
    return store, selected_id, messages, gr.update(choices=choices, value=selected_id)


def _on_delete_and_new(
    store_state: dict[str, dict] | None,
    current_tid: str,
) -> tuple[dict[str, dict], str, list[dict[str, str]], gr.Dropdown]:
    """删除当前对话并切换到下一个。"""
    store = dict(store_state) if store_state else {}
    if current_tid and current_tid in store:
        logger.info(f"[WebUI] 删除对话: {store[current_tid].get('title', '?')}")
        del store[current_tid]
    _save_sessions(store)

    if not store:
        return _on_new_conversation(store, "")

    latest = max(store.keys(), key=lambda k: store[k].get("created_at", 0))
    return _on_select_conversation(latest, store)


# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
# 级联下拉菜单回调
# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

def _on_province_change(province: str) -> gr.Dropdown:
    """省份变更 → 刷新城市列表。"""
    if not province:
        return gr.Dropdown(choices=[], value=None, interactive=False)
    loader: DistrictLoader = _get_district_loader()
    cities: list[str] = loader.get_cities(province)
    return gr.Dropdown(
        choices=cities, value=cities[0] if cities else None, interactive=True
    )


def _on_city_change(city: str) -> gr.Dropdown:
    """城市变更 → 刷新区县列表。"""
    if not city:
        return gr.Dropdown(choices=[], value=None, interactive=False)
    loader: DistrictLoader = _get_district_loader()
    districts: list[str] = loader.get_districts(city)
    return gr.Dropdown(
        choices=districts,
        value=districts[0] if districts else None,
        interactive=True,
    )


# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
# 核心回调：处理用户输入，驱动工作流
# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝


def _handle_chat(
    message: str,
    province: str,
    city: str,
    district: str,
    location_detail: str,
    history: list[dict[str, str]],
    current_tid: str,
    store_state: dict[str, dict] | None,
) -> Generator[tuple, None, None]:
    """处理用户聊天消息，流式输出推荐文本（v3.1 升级）。

    v3.1 变更：使用 workflow.run_stream() 替代 workflow.run()，
    实现 Token 级逐字输出打字机效果。
    """
    history = list(history) if history else []
    store = dict(store_state) if store_state else {}
    store, current_tid = _ensure_active_conversation(store, current_tid)

    if not message or not message.strip():
        yield from _noop_yield(history, store, current_tid, "请输入内容后再发送。")
        return

    location_parts: list[str] = [p for p in [province, city, district] if p]
    if location_detail and location_detail.strip():
        location_parts.append(location_detail.strip())
    location_str: str = " > ".join(location_parts) if location_parts else ""
    logger.info(
        f"[WebUI] thread={current_tid[:20]}..., "
        f"位置={location_str or '未选择'}, query={message[:60]}..."
    )

    history.append({"role": "user", "content": message})

    # 第一步：显示"思考中"
    thinking_history = history + [
        {"role": "assistant", "content": "⏳ 正在分析您的需求..."}
    ]
    choices = _build_dropdown_choices(store)
    yield (
        thinking_history,
        "⏳ 分析中...",
        "<p style='color:#888;text-align:center'>分析中...</p>",
        "*处理中...*",
        "*处理中...*",
        "*处理中...*",
        "*处理中...*",
        "",          # ← 仅第一个 yield 清空输入框（发送后清除）
        store,
        current_tid,
        gr.update(choices=choices, value=current_tid),
    )

    workflow: LocalLifeWorkflow = _get_workflow()

    # ── 构建增强查询（含位置信息） ──
    enhanced_query: str = message.strip()
    if location_detail and location_detail.strip():
        if city:
            enhanced_query = (
                f"[位置信息] 搜索城市: {city}, "
                f"精确位置: {location_detail.strip()}\n"
                f"[用户需求] {enhanced_query}"
            )
        else:
            enhanced_query = (
                f"[位置信息] 精确位置: {location_detail.strip()}\n"
                f"[用户需求] {enhanced_query}"
            )
    elif city:
        enhanced_query = (
            f"[位置信息] 搜索城市: {city}\n[用户需求] {enhanced_query}"
        )

    # ── 流式执行工作流 ──
    accumulated_text: str = ""
    final_state: AgentState | None = None
    heartbeat_count: int = 0

    try:
        for token, state in workflow.run_stream(
            user_input=enhanced_query,
            thread_id=current_tid,
        ):
            if token:
                # 增量 token → 追加到累积文本，实时更新 chatbot
                accumulated_text += token
                stream_history = history + [
                    {"role": "assistant", "content": accumulated_text}
                ]
                choices = _build_dropdown_choices(store)
                yield (
                    stream_history,
                    "⏳ 生成中...",
                    "<p style='color:#888;text-align:center'>生成中...</p>",
                    "*生成中...*",
                    "*生成中...*",
                    "*生成中...*",
                    "*生成中...*",
                    gr.update(),  # 流式 token yield：不清空输入框
                    store,
                    current_tid,
                    gr.update(choices=choices, value=current_tid),
                )
            elif state is not None:
                # 最终状态 → 保存到 history，构建面板
                final_state = state
                break
            else:
                # ── 心跳 token：保持 SSE 连接活跃（空 token，无 state）──
                # 关键：已流式输出的 accumulated_text 必须保留显示，
                # 不能覆盖为"⏳ 正在分析..."，否则用户看到文本闪烁消失。
                heartbeat_count += 1
                dot_str: str = "." * (heartbeat_count % 6)
                if accumulated_text:
                    display_text = accumulated_text
                else:
                    display_text = f"⏳ 正在分析您的需求{dot_str}"
                heartbeat_history = history + [
                    {"role": "assistant", "content": display_text}
                ]
                choices = _build_dropdown_choices(store)
                # 地图等在流式过程中也显示状态提示，而非"处理中"
                map_placeholder = (
                    "<p style='color:#888;text-align:center;padding:40px'>生成中...</p>"
                    if not accumulated_text else
                    "<p style='color:#888;text-align:center;padding:40px'>⏳ 安全审查中...</p>"
                )
                yield (
                    heartbeat_history,
                    f"⏳ 分析中{dot_str}" if not accumulated_text else "⏳ 安全审查中...",
                    map_placeholder,
                    "*处理中...*",
                    "*处理中...*",
                    "*处理中...*",
                    "*处理中...*",
                    gr.update(),  # 心跳 yield：不清空输入框
                    store,
                    current_tid,
                    gr.update(choices=choices, value=current_tid),
                )
    except Exception as exc:
        logger.error(f"[WebUI] 工作流执行失败: {exc}")
        error_msg: str = (
            f"抱歉，处理您的请求时遇到了技术问题。\n\n"
            f"**错误类型**: `{type(exc).__name__}`\n\n"
            f"请检查 API Key 配置是否正确。"
        )
        history.append({"role": "assistant", "content": error_msg})
        store[current_tid]["messages"] = list(history)
        store = _auto_title(store, current_tid)
        _save_sessions(store)
        choices = _build_dropdown_choices(store)
        yield (
            history,
            "❌ 执行失败",
            "<p style='color:#888;text-align:center'>错误</p>",
            str(exc),
            "",
            "",
            "",
            gr.update(),  # 异常路径：不清空输入框
            store,
            current_tid,
            gr.update(choices=choices, value=current_tid),
        )
        return

    # ── 使用最终状态构建输出 ──
    if final_state is None:
        # 兜底：流式生成未返回最终状态
        reply = accumulated_text or "（未生成回复）"
        history.append({"role": "assistant", "content": reply})
        store[current_tid]["messages"] = list(history)
        store = _auto_title(store, current_tid)
        _save_sessions(store)
        choices = _build_dropdown_choices(store)
        yield (
            history,
            "⚠️ 未返回状态",
            "<p style='color:#888;text-align:center'>未返回状态</p>",
            "",
            "",
            "",
            "",
            gr.update(),  # 兜底路径：不清空输入框
            store,
            current_tid,
            gr.update(choices=choices, value=current_tid),
        )
        return

    # 使用 final_state 中的 final_recommendation（可能被安全审查改写）
    reply = final_state.final_recommendation or accumulated_text or "（未生成回复）"
    history.append({"role": "assistant", "content": reply})
    store[current_tid]["messages"] = list(history)
    store = _auto_title(store, current_tid)
    _save_sessions(store)

    safety_status: str = _build_safety_status(final_state)
    preference_md: str = _build_preference_table(final_state)
    candidates_md: str = _build_candidates_table(final_state)
    trace_md: str = _build_trace_table(final_state)
    itinerary_md: str = _build_itinerary(final_state, llm_client=_get_workflow().fast_llm)
    map_html: str = _build_map_html(final_state)

    choices = _build_dropdown_choices(store)
    yield (
        history,
        safety_status,
        map_html,
        itinerary_md,
        preference_md,
        candidates_md,
        trace_md,
        "",
        store,
        current_tid,
        gr.update(choices=choices, value=current_tid),
    )


def _noop_yield(
    history: list[dict[str, str]],
    store: dict[str, dict],
    current_tid: str,
    reason: str,
) -> Generator[tuple, None, None]:
    """空操作 yield 生成器。"""
    h = list(history) if history else []
    st = dict(store) if store else {}
    st, current_tid = _ensure_active_conversation(st, current_tid)
    choices = _build_dropdown_choices(st)
    yield (
        h,
        reason,
        "<p style='color:#888;text-align:center'>" + reason + "</p>",
        "",
        "",
        "",
        "",
        "",
        st,
        current_tid,
        gr.update(choices=choices, value=current_tid),
    )


# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
# 右侧日志区内容构建
# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

def _build_safety_status(state: AgentState) -> str:
    """构建安全审查状态显示。"""
    if state.safety_passed:
        return "✅ 安全审查通过"
    violations: list[str] = getattr(state, "safety_violations", []) or []
    if violations:
        lines: list[str] = ["### 🛡️ 安全审查不通过\n", "以下违规类型被检测到：\n"]
        violation_map: dict[str, str] = {
            "禁止承诺已履约": "🚫 承诺已履约",
            "禁止编造库存": "🚫 编造库存/空位",
            "禁止越权交易": "🚫 越权下单/支付",
            "禁止伪造商家承诺": "🚫 伪造商家承诺/优惠",
        }
        for v in violations:
            lines.append(f"- {violation_map.get(v, v)}")
        return "\n".join(lines)
    return "🛡️ 安全审查不通过 — 检测到违规内容"


def _build_preference_table(state: AgentState) -> str:
    """构建提取的用户偏好表格（含每人明细）。"""
    pref = state.user_preference
    lines: list[str] = [
        "| 维度 | 值 |",
        "|---|---|",
        f"| 预算 | {pref.budget or '未指定'} |",
        f"| 口味 | {pref.taste or '未指定'} |",
        f"| 忌口/限制 | {pref.restrictions or '无'} |",
        f"| 距离 | {pref.distance or '未指定'} |",
        f"| 城市 | {pref.city or '未指定'} |",
        f"| 时间 | {pref.time or '未指定'} |",
        f"| 携带儿童 | {'是' if pref.has_kids else '否'} |",
        f"| 需要停车 | {'是' if pref.need_parking else '否'} |",
        f"| 精确位置 | {pref.freeform_location or '未指定'} |",
    ]
    # ── 动态生成领域偏好行（来自 registry）──
    for td in get_preference_tools(TOOL_REGISTRY):
        val = getattr(pref, td.preference_field, None)
        if val:
            lines.append(f"| {td.preference_table_label} | {val} |")
    if state.conflict_strategy:
        lines.append(f"| 冲突策略 | {state.conflict_strategy[:60]}... |")

    # ── v2.2: 每人偏好明细 ──
    ind_prefs = getattr(state, "individual_preferences", []) or []
    if ind_prefs:
        lines.append("\n### 👤 个人偏好明细\n")
        lines.append("| 用户 | 预算 | 口味 | 忌口 | 关键原话 |")
        lines.append("|---|---|---|---|---|")
        for ip in ind_prefs:
            lines.append(
                f"| {ip.name} | {ip.budget or '—'} | {ip.taste or '—'} | "
                f"{ip.restrictions or '—'} | {ip.key_utterance[:30] if ip.key_utterance else '—'} |"
            )

    return "\n".join(lines)


# ── 候选表格每领域最大展示条数 ──
_MAX_ITEMS: dict[str, int] = {
    "restaurant": 6,
    "transit": 5,
}


def _build_candidates_table(state: AgentState) -> str:
    """构建候选数据表格 — 餐厅手动渲染，其余领域由 registry 动态驱动。"""
    parts: list[str] = []

    # ── 餐厅（特殊格式：含排队/团购列）──
    candidates: list[dict[str, Any]] = state.candidate_restaurants
    if candidates:
        lines: list[str] = [
            "### 🍽️ 候选餐厅\n",
            "| # | 餐厅 | 评分 | 人均 | 距离 | 排队 | 团购 |",
            "|---|---|---|---|---|---|---|",
        ]
        for i, r in enumerate(candidates[:6], 1):
            name: str = r.get("name", "?")
            rating: Any = r.get("rating", "-")
            avg_price: str = r.get("avg_price", "-")
            dist: Any = r.get("distance_km", "-")
            queue: str = r.get("queue_status", "-")
            gb: str = "有" if r.get("has_group_buy") else "—"
            lines.append(
                f"| {i} | {name} | ★{rating} | {avg_price} | "
                f"{dist}km | {queue} | {gb} |"
            )
        parts.append("\n".join(lines))
    else:
        parts.append("### 🍽️ 候选餐厅\n\n（未检索到）")

    # ── 其余领域：从 registry 动态驱动 ──
    for td in get_display_domains(TOOL_REGISTRY, skip_restaurant=True):
        candidates = getattr(state, td.candidate_key, [])
        if not candidates:
            continue
        cols = td.candidate_columns
        header = "| # | " + " | ".join(c.title() for c in cols) + " |"
        sep = "|---" * (len(cols) + 1) + "|"
        section_lines: list[str] = [
            f"\n### {td.candidate_section_title}\n",
            header,
            sep,
        ]
        max_items = _MAX_ITEMS.get(td.key, 4)
        for i, item in enumerate(candidates[:max_items], 1):
            row_parts = [str(i)]
            for col in cols:
                val = item.get(col, "—")
                if isinstance(val, list):
                    val = ", ".join(val[:4]) if val else "—"
                row_parts.append(str(val))
            section_lines.append("| " + " | ".join(row_parts) + " |")
        parts.append("\n".join(section_lines))

    return "\n".join(parts) if parts else "（未检索到候选数据）"


def _build_trace_table(state: AgentState) -> str:
    """构建 Trace 评测日志表格。"""
    trace_logs: list[dict[str, Any]] = state.trace_logs
    if not trace_logs:
        return "（暂无 Trace 日志）"

    lines: list[str] = [
        "| 阶段 | 耗时 | 状态摘要 |",
        "|---|---|---|",
    ]
    for entry in trace_logs:
        phase: str = entry.get("phase", "?")
        elapsed: float = entry.get("elapsed_ms", 0)
        skipped: bool = entry.get("skipped", False)

        if skipped:
            summary: str = f"跳过 ({entry.get('skip_reason', '')})"
        else:
            out: Any = entry.get("output", {})
            summary = ""
            if isinstance(out, dict):
                if phase == "analyze":
                    nc = out.get("needs_clarification", False)
                    summary = "需要反问" if nc else "偏好提取完成"
                elif phase == "retrieve":
                    summary = f"获取 {out.get('count', out.get('restaurants', '?'))} 个候选"
                elif phase == "recommend":
                    summary = f"生成长度 {out.get('recommendation_length', '?')} 回复"
                elif phase == "safety":
                    passed = out.get("passed", True)
                    summary = "审查通过" if passed else "拦截并改写"
            elif isinstance(out, str):
                summary = out[:50]

        lines.append(f"| {phase} | {elapsed:.0f} ms | {summary} |")

    violations: list[str] = getattr(state, "safety_violations", []) or []
    if violations:
        lines.append(f"| 🛡️ 违规 | — | {', '.join(violations)} |")

    return "\n".join(lines)


# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
# v3.1：地图可视化
# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

def _build_map_html(state: AgentState) -> str:
    """基于候选餐厅构建高德地图 HTML。

    使用高德 JS API v2.0 在地图上标注候选餐厅。
    无 AMAP_API_KEY 或无边间数据时展示占位提示。
    """
    amap_key: str = os.getenv("AMAP_API_KEY", "")
    security_code: str = os.getenv("AMAP_JS_SECURITY_CODE", "")
    candidates: list[dict[str, Any]] = (
        state.candidate_restaurants
        or state.candidate_hotels
        or state.candidate_entertainments
        or []
    )

    if not amap_key:
        return (
            "<div style='text-align:center;color:#888;padding:40px;font-size:14px'>"
            "⚠️ 未配置高德地图 API Key<br>"
            "<small>请在 <code>.env</code> 中设置 <code>AMAP_API_KEY</code> 后重启</small>"
            "</div>"
        )

    if not candidates:
        return (
            "<div style='text-align:center;color:#888;padding:40px;font-size:14px'>"
            "📍 暂无候选地点数据<br>"
            "<small>获取推荐后可在地图上查看候选位置</small>"
            "</div>"
        )

    # ── 构建 Marker JS ──
    marker_entries: list[str] = []
    default_lng, default_lat = 116.397, 39.909  # 北京天安门
    first_lng, first_lat = default_lng, default_lat

    for i, c in enumerate(candidates):
        title: str = (c.get("name", "?")).replace("'", "\\'")
        lng: float = float(c.get("longitude", 0))
        lat: float = float(c.get("latitude", 0))
        rating: Any = c.get("rating", "—")
        price: str = c.get("avg_price", "—")
        category: str = c.get("category", c.get("cuisine", ""))

        if lng and lat:
            if i == 0:
                first_lng, first_lat = lng, lat
            info_str: str = f"{title}|★{rating}|{price}"
            if category:
                info_str += f"|{category}"
            marker_entries.append(
                f"{{lng:{lng}, lat:{lat}, title:'{title}', "
                f"info:'{info_str}'}}"
            )

    if not marker_entries:
        return (
            "<div style='text-align:center;color:#888;padding:40px;font-size:14px'>"
            "⚠️ 候选数据缺少经纬度信息<br>"
            "<small>高德 API 返回的 POI 数据不含坐标，无法标注地图</small>"
            "</div>"
        )

    markers_js: str = ",\n        ".join(marker_entries)

    # ── 构建交通站点 Marker JS（绿色标记）──
    transit_markers_js: str = ""
    transit_stops: list[dict[str, Any]] = getattr(state, "candidate_transit_stops", []) or []
    if transit_stops:
        transit_entries: list[str] = []
        for stop in transit_stops:
            stop_name: str = (stop.get("name", "?")).replace("'", "\\'")[:30]
            stop_lng: float = float(stop.get("longitude", 0))
            stop_lat: float = float(stop.get("latitude", 0))
            if stop_lng and stop_lat:
                stop_type: str = stop.get("type", "地铁站")
                emoji: str = "🚇" if "地铁" in stop_type else "🚌"
                transit_entries.append(
                    f"{{lng:{stop_lng}, lat:{stop_lat}, "
                    f"title:'{emoji} {stop_name}', "
                    f"info:'{stop_name}|{stop_type}'}}"
                )
        if transit_entries:
            transit_markers_js = ",\n        ".join(transit_entries)

    # ── 构建交通路线 Polyline JS（在地图上画出公交/地铁走向）──
    polylines_js: str = ""
    transit_dirs: list[dict[str, Any]] = getattr(state, "candidate_transit_directions", []) or []
    # 收集有坐标的 direction 起点/终点对
    dir_endpoints: list[dict[str, Any]] = []
    for d in transit_dirs:
        if not d.get("success"):
            continue
        origin_name = d.get("_from", "")
        dest_name = d.get("_to", "")
        # 从 transit_stops 查找 origin 坐标
        origin_coord = None
        dest_coord = None
        for stop in transit_stops:
            sn = stop.get("name", "")
            if origin_name and origin_name in sn:
                origin_coord = (float(stop.get("longitude", 0)), float(stop.get("latitude", 0)))
            if dest_name and dest_name in sn:
                dest_coord = (float(stop.get("longitude", 0)), float(stop.get("latitude", 0)))
        # 从餐厅 candidates 查找 destination 坐标
        if not dest_coord:
            for c in candidates:
                cn = c.get("name", "")
                if dest_name and dest_name in cn:
                    dest_coord = (float(c.get("longitude", 0)), float(c.get("latitude", 0)))
        if origin_coord and dest_coord:
            dir_endpoints.append({
                "from": origin_coord, "to": dest_coord,
                "fromName": origin_name, "toName": dest_name,
            })

    if dir_endpoints:
        poly_lines: list[str] = []
        route_colors: list[str] = ["#4A90D9", "#E8734A", "#50B86C", "#F5A623", "#7B68EE"]
        for ri, ep in enumerate(dir_endpoints):
            color = route_colors[ri % len(route_colors)]
            poly_lines.append(
                f"""var line_{ri} = new AMap.Polyline({{
          path: [[{ep['from'][0]}, {ep['from'][1]}], [{ep['to'][0]}, {ep['to'][1]}]],
          strokeColor: '{color}', strokeWeight: 3, strokeOpacity: 0.6,
          strokeStyle: 'dashed', map: map,
          showDir: true
        }});"""
            )
            # 起点小圆点
            poly_lines.insert(0,
                f"""new AMap.CircleMarker({{
          center: [{ep['from'][0]}, {ep['from'][1]}],
          radius: 5, fillColor: '{color}', fillOpacity: 0.8,
          strokeColor: '#fff', strokeWeight: 1,
          map: map, zIndex: 100
        }});"""
            )
        polylines_js = "\n        ".join(poly_lines)

    # ── 构建完整 HTML 文档（内嵌在 iframe 中）──
    # 关键：Gradio gr.HTML 通过 innerHTML 注入内容，浏览器不会执行 innerHTML
    # 中的 <script> 标签。因此必须用 iframe srcdoc 包裹完整文档，让 AMap JS SDK
    # 在隔离的文档上下文中正常加载和执行。
    inner_html: str = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body, html {{ margin:0; padding:0; width:100%; height:100%; }}
  #map-container {{ width:100%; min-height:350px; border-radius:8px; }}
  #map-container .map-error {{ text-align:center; color:#888; padding:40px; font-size:14px; }}
  .amap-info {{ font-size:12px; line-height:1.5; }}
</style>
</head>
<body>
<div id="map-container"></div>
<script>
  window._AMapSecurityConfig = {{ securityJsCode: "{security_code}" }};
  // ── 定义 initMap：由 SDK script 的 onload 回调触发 ──
  function initMap() {{
    try {{
      var container = document.getElementById("map-container");
      if (!container || container.offsetHeight === 0) {{
        container.style.height = "350px";
      }}

      var map = new AMap.Map("map-container", {{
        zoom: 14,
        center: [{first_lng}, {first_lat}],
        resizeEnable: true,
      }});

      var markers = [
        {markers_js}
      ];

      markers.forEach(function(m) {{
        var marker = new AMap.Marker({{
          position: [m.lng, m.lat],
          title: m.title,
          map: map,
        }});
        var parts = m.info.split('|');
        var infoContent = '<div class="amap-info">'
          + '<b>' + parts[0] + '</b><br>'
          + '评分: ' + (parts[1] || '—') + '<br>'
          + '人均: ' + (parts[2] || '—');
        if (parts[3]) infoContent += '<br>类型: ' + parts[3];
        infoContent += '</div>';

        marker.on('click', function() {{
          var info = new AMap.InfoWindow({{
            content: infoContent,
            offset: new AMap.Pixel(0, -30),
          }});
          info.open(map, marker.getPosition());
        }});
      }});

      // ── 交通站点标记（绿色）──
      var transitMarkers = [
        {transit_markers_js}
      ];
      transitMarkers.forEach(function(m) {{
        if (!m.lng || !m.lat) return;
        // 绿色圆形标记，带 emoji 文字
        var label = m.title.substring(0, 2);
        var svg = '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28">'
          + '<circle cx="14" cy="14" r="12" fill="#4CAF50" stroke="#fff" stroke-width="2"/>'
          + '<text x="14" y="19" text-anchor="middle" font-size="14" fill="#fff">' + label + '</text>'
          + '</svg>';
        var icon = new AMap.Icon({{
          size: new AMap.Size(28, 28),
          imageSize: new AMap.Size(28, 28),
          image: 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg)
        }});
        var marker = new AMap.Marker({{
          position: [m.lng, m.lat],
          title: m.title,
          map: map,
          icon: icon,
          offset: new AMap.Pixel(-14, -14)
        }});
        var parts = m.info.split('|');
        var infoContent = '<div class="amap-info">'
          + '<b>' + parts[0] + '</b><br>'
          + '类型: ' + (parts[1] || '—')
          + '</div>';
        marker.on('click', function() {{
          var info = new AMap.InfoWindow({{
            content: infoContent,
            offset: new AMap.Pixel(0, -20),
          }});
          info.open(map, marker.getPosition());
        }});
      }});

      // ── 交通路线折线（站点→餐厅）──
      {polylines_js}

      // ── 延迟 resize：修复 Accordion 折叠后展开时尺寸为 0 ──
      setTimeout(function() {{ map.resize(); }}, 200);
    }} catch(e) {{
      document.getElementById("map-container").innerHTML =
        '<div class="map-error">⚠️ 地图初始化失败: ' + e.message + '</div>';
    }}
  }}
  // ── SDK 加载失败时的错误处理（函数引用避免 srcdoc 属性嵌套引号问题）──
  function onAMapLoadError() {{
    document.getElementById("map-container").innerHTML =
      '<div class="map-error">⚠️ 高德地图 JS API 加载失败<br><small>请检查 AMAP_API_KEY 是否为 <b>Web端(JS API)</b> 类型，而非 Web服务 类型</small></div>';
  }}
</script>
<script src="https://webapi.amap.com/maps?v=2.0&key={amap_key}&plugin=AMap.Marker,AMap.Polyline,AMap.CircleMarker,AMap.GeometryUtil"
  onload="initMap()"
  onerror="onAMapLoadError()">
</script>
</body>
</html>"""

    # ── 用 iframe srcdoc 包裹，确保 <script> 标签正常执行 ──
    # 注意: inner_html 中已有 &amp; (来自 Python f-string 的 {{ }} 转义)
    # 所以只转义双引号为 &quot;，不转义 &（避免 &amp; → &amp;amp; 双重转义）
    escaped: str = inner_html.replace('"', "&quot;")
    html: str = (
        '<iframe srcdoc="' + escaped + '" '
        'style="width:100%;height:370px;border:none;border-radius:8px;" '
        'sandbox="allow-scripts allow-same-origin" '
        'title="高德地图" loading="lazy"></iframe>'
    )

    return html


# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
# v3.0：行程提取 & 快捷动作回调
# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

def _build_itinerary(state: AgentState, llm_client: Any = None) -> str:
    """从推荐结果提取行程规划 Markdown（v3.1：支持 LLM 提取）。

    Args:
        state: Agent 状态（含 final_recommendation）。
        llm_client: 可选，LLM 客户端（用于智能行程提取）。传入 None 时回退关键词扫描。
    """
    rec: str = state.final_recommendation
    if not rec:
        return "（获取推荐后可提取行程）"

    try:
        steps = ItineraryExtractor.extract(rec, llm_client=llm_client)
    except Exception:
        steps = []

    if not steps:
        return "（未能从推荐中提取到明确行程，请查看推荐文本手动规划）"

    lines: list[str] = ["| 时间 | 地点 | 行动 | 备注 |", "|---|---|---|---|"]
    for s in steps:
        lines.append(
            f"| {s.get('time', '')} | {s.get('location', '')[:25]} | "
            f"{s.get('action', '')} | {s.get('note', '')[:30]} |"
        )
    return "\n".join(lines)


def _extract_content_text(content: Any) -> str:
    """从 Gradio chatbot 消息中提取纯文本内容。

    兼容 Gradio 5+ 的 OpenAI 格式（content 可能是 str 或 list[dict]）。

    Args:
        content: 消息的 content 字段，可以是 str 或 list[dict]。

    Returns:
        提取出的纯文本字符串。
    """
    if isinstance(content, list):
        # OpenAI 格式: [{"text": "...", "type": "text"}, ...]
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                text_parts.append(part)
        return "".join(text_parts)
    return str(content)


def _get_last_recommendation_text(history: list[dict[str, str]]) -> str:
    """从聊天历史中获取最后一条 Agent 推荐文本。"""
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            content: str = _extract_content_text(msg.get("content", ""))
            # 跳过"思考中"等状态消息
            if content and "⏳" not in content and "处理中" not in content:
                return content
    return ""



def _on_share(
    history: list[dict[str, str]],
    current_tid: str,
    store_state: dict[str, dict] | None,
) -> str:
    """生成分享卡片。"""
    text: str = _get_last_recommendation_text(history)
    if not text:
        return "⚠️ 暂无推荐内容可分享"

    try:
        # 构建最小 AgentState 用于分享
        from domain.entities import UserPreference
        state = AgentState(
            final_recommendation=text,
            user_preference=UserPreference(),
        )
        card: str = generate_share_card(state)
        # 返回前 500 字符预览
        return f"📤 **分享卡片已生成** (共 {len(card)} 字符)\n\n```markdown\n{card[:600]}\n```"
    except Exception as exc:
        return f"❌ 生成分享卡片失败: {exc}"


def _on_reminder(history: list[dict[str, str]]) -> str:
    """生成提醒参数（v3.1：解析真实日期和餐厅名称）。

    1. 从最后一条用户消息中解析用餐时间（中文时间 → 绝对 datetime）
    2. 从最后一条 Agent 推荐中提取餐厅名称
    3. 生成含 fireAt 的结构化参数
    """
    text: str = _get_last_recommendation_text(history)
    if not text:
        return "⚠️ 暂无推荐内容，无法设置提醒"

    try:
        # ── 1. 提取最后一条用户消息 ──
        last_user_msg: str = ""
        for msg in reversed(history):
            if msg.get("role") == "user":
                last_user_msg = _extract_content_text(msg.get("content", ""))
                break

        # ── 2. 解析用餐时间 ──
        meal_dt = _parse_meal_time(last_user_msg) if last_user_msg else None

        # ── 3. 从推荐文本中提取餐厅名称 ──
        restaurant_name: str = _extract_restaurant_name(text)

        # ── 4. 构建提醒参数 ──
        meal_time_str: str = ""
        if meal_dt:
            meal_time_str = meal_dt.strftime("%m月%d日 %H:%M")
        else:
            meal_time_str = "请根据推荐文本指定"

        params = create_meal_reminder_params(
            restaurant_name=restaurant_name,
            meal_time=meal_time_str,
            note=text[:100],
            meal_dt=meal_dt,
        )

        # ── 5. 创建系统级提醒任务 ──
        # 直接写入 scheduled-tasks SKILL.md 文件（与 MCP 工具等效）
        mcp_success: bool = False
        mcp_error: str = ""
        try:
            from pathlib import Path as _Path

            _tasks_dir: _Path = _Path.home() / ".claude" / "scheduled-tasks" / params["taskId"]
            _tasks_dir.mkdir(parents=True, exist_ok=True)
            _skill_path: _Path = _tasks_dir / "SKILL.md"

            # 构建 SKILL.md 内容（含 frontmatter）
            _fire_at: str = params.get("fireAt", "")
            _frontmatter: str = (
                "---\n"
                f"description: {params['description']}\n"
            )
            if _fire_at:
                _frontmatter += f"fireAt: {_fire_at}\n"
            _frontmatter += "enabled: true\n"
            _frontmatter += "---\n\n"

            _skill_path.write_text(_frontmatter + params["prompt"], encoding="utf-8")
            mcp_success = True
            logger.info(f"[WebUI] 提醒任务已创建: {params['taskId']} → {_skill_path}")
        except Exception as exc:
            mcp_error = str(exc)
            logger.warning(f"[WebUI] 提醒创建失败: {exc}")

        # ── 6. 构建用户友好的响应 ──
        if mcp_success:
            lines: list[str] = ["✅ **提醒已创建成功！**\n"]
        else:
            lines: list[str] = [f"⚠️ 自动创建提醒失败（{mcp_error}），以下为手动参数：\n"]
        if meal_dt:
            snippet: str = last_user_msg[:30] if last_user_msg else ""
            lines.append(
                f"📅 用餐时间: **{meal_time_str}**"
                f"（从 “{snippet}...” 解析）\n"
            )
            if "fireAt" in params:
                lines.append(f"🔔 将在 **{params['fireAt']}** 提醒您\n")
        else:
            lines.append("⚠️ 未能从输入中解析出具体时间，请手动设置\n")
        lines.append(f"🍽️ 餐厅: **{restaurant_name}**\n")
        lines.append("\n> 💡 提醒触发时系统将显示用餐时间和餐厅信息。")
        return "\n".join(lines)

    except Exception as exc:
        return f"❌ 生成提醒失败: {exc}"


def _parse_meal_time(text: str) -> datetime | None:
    """Parse a meal time from user message text using simple heuristics.

    Falls back to just using the current time + 2 hours if no time
    can be parsed from the text.
    """
    import re

    now = datetime.now()
    # Try to find time patterns like "18:30", "18：30", "晚上6点"
    time_patterns = [
        r"(\d{1,2}):(\d{2})",
        r"(\d{1,2})：(\d{2})",
        r"(晚上|下午|中午|早上|上午)?(\d{1,2})点(半|(\d{1,2})分)?",
    ]

    for pattern in time_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                if ":" in match.group(0) or "：" in match.group(0):
                    hour = int(match.group(1))
                    minute = int(match.group(2))
                else:
                    hour = int(match.group(2))
                    minute = 30 if "半" in match.group(0) else 0
                    period = match.group(1) or ""
                    if "下午" in period and hour < 12:
                        hour += 12
                    elif "晚上" in period and hour < 12:
                        hour += 12

                result = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                return result
            except (ValueError, IndexError):
                pass

    return now + timedelta(hours=2)


def _extract_restaurant_name(recommendation_text: str) -> str:
    """从推荐文本中用启发式规则提取餐厅名称。

    优先级：
    1. 查找 "推荐餐厅" / "餐厅名称" 等标记后的名称
    2. 查找 **粗体** 包裹的名称
    3. 查找以 "去" 开头的句子中的餐厅名
    4. 回退为 "推荐餐厅"
    """
    import re

    # 模式 1：明确标记 "推荐餐厅: XXX" 或 "餐厅: XXX"
    for pat in [
        r"推荐餐厅[：:]\s*[\*\*]*([^*\n]+?)[\*\*]*(?:$|\n|，|。)",
        r"餐厅[：:]\s*[\*\*]*([^*\n]+?)[\*\*]*(?:$|\n|，|。)",
        r"「([^」]+)」",
    ]:
        m = re.search(pat, recommendation_text)
        if m:
            name = m.group(1).strip()
            # 过滤掉太通用的名称
            if name and len(name) >= 2 and name not in ("推荐", "餐厅", "酒店"):
                return name[:30]

    # 模式 2：Markdown 粗体 **餐厅名**
    bold_matches = re.findall(r"\*\*([^*]+)\*\*", recommendation_text)
    for bm in bold_matches:
        bm = bm.strip()
        # 跳过非餐厅名称的粗体（如标题、数字等）
        if (
            len(bm) >= 2
            and not bm.isdigit()
            and not re.match(r"^[\d.]+$", bm)
            and "元" not in bm
            and "人" not in bm
            and bm not in ("推荐", "注意事项", "温馨提示")
        ):
            return bm[:30]

    # 模式 3："去 XXX" — 首选用餐场景中的推荐
    m = re.search(r"(?:推荐|建议|可以|不妨)[去前往到]?\s*[\*\*]*([^*\n，。]{2,20}?)[\*\*]*(?:餐厅|酒店|用餐|吃饭|品尝|试试)", recommendation_text)
    if m:
        name = m.group(1).strip()
        # 去掉可能的动词前缀
        name = re.sub(r'^[去前往到]\s*', '', name)
        return name[:30]

    return "推荐餐厅"


# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
# 极简 CSS：只设置颜色/字体，不碰 Gradio 内部布局类
# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

_CUSTOM_CSS: str = """
/* ---- 暖调配色变量 ---- */
:root {
  --ll-canvas: #faf9f5;
  --ll-surface-soft: #f5f0e8;
  --ll-surface-card: #efe9de;
  --ll-ink: #141413;
  --ll-body: #3d3d3a;
  --ll-muted: #6c6a64;
  --ll-muted-soft: #8e8b82;
  --ll-hairline: #e6dfd8;
  --ll-primary: #cc785c;
  --ll-primary-active: #a9583e;
  --ll-on-primary: #ffffff;
  --ll-sidebar-bg: #efe9de;
  --ll-sidebar-text: #3d3d3a;
  --ll-sidebar-item-hover: #e8e0d2;
  --ll-sidebar-item-active: rgba(204, 120, 92, 0.15);
  --ll-success: #5db872;
  --ll-radius-sm: 6px;
  --ll-radius-md: 8px;
  --ll-radius-lg: 12px;
  --ll-radius-xl: 16px;
}

/* ---- 全局画布 ---- */
body {
  background: var(--ll-canvas) !important;
}
.gradio-container {
  max-width: 1440px !important;
  margin: 0 auto !important;
  background: var(--ll-canvas) !important;
  font-family: 'Noto Sans SC', Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  color: var(--ll-body) !important;
}
footer { display: none !important; }

/* ---- 标题 ---- */
#app-title-block h1 {
  font-family: 'Noto Serif SC', Georgia, 'Times New Roman', serif !important;
  color: var(--ll-ink) !important;
  font-weight: 400 !important;
}
#app-title-block h3 {
  color: var(--ll-muted) !important;
  font-weight: 400 !important;
}

/* ---- 左侧边栏 ---- */
#sidebar-col {
  background: var(--ll-sidebar-bg) !important;
  border: 1px solid var(--ll-surface-card) !important;
  border-radius: var(--ll-radius-xl) !important;
  padding: 16px 12px !important;
  color: var(--ll-sidebar-text) !important;
}

/* ---- 位置选择行 ---- */
#location-row {
  background: var(--ll-surface-soft) !important;
  border-radius: var(--ll-radius-lg) !important;
  padding: 12px !important;
  margin-bottom: 12px !important;
  border: 1px solid var(--ll-hairline) !important;
}

/* ---- 输入行 ---- */
#input-row {
  background: var(--ll-surface-soft) !important;
  border-radius: var(--ll-radius-lg) !important;
  padding: 10px 12px !important;
  margin-top: 10px !important;
  border: 1px solid var(--ll-hairline) !important;
}

/* ---- 新建对话按钮 ---- */
#new-chat-btn {
  background: var(--ll-primary) !important;
  color: var(--ll-on-primary) !important;
}

/* ---- 发送按钮 ---- */
#send-btn {
  background: var(--ll-primary) !important;
  color: var(--ll-on-primary) !important;
}

/* ---- 删除对话按钮 ---- */
#delete-conv-btn {
  background: transparent !important;
  color: var(--ll-muted) !important;
  border: 1px solid var(--ll-hairline) !important;
}

/* ---- 右侧面板 ---- */
#right-panel {
  gap: 12px;
}

/* ---- Markdown 标题 ---- */
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
  font-family: 'Noto Serif SC', Georgia, 'Times New Roman', serif !important;
  color: var(--ll-ink) !important;
  font-weight: 500 !important;
}
.gr-markdown p { color: var(--ll-body) !important; line-height: 1.6 !important; }

/* ---- 滚动条 ---- */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: var(--ll-hairline);
  border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: var(--ll-muted-soft); }

/* ---- 隐藏 Chatbot 消息右上角的分享/删除/复制按钮 ---- */
.message-buttons-right.bubble {
  display: none !important;
}
"""


def build_ui() -> gr.Blocks:
    """构建三栏布局的 Gradio Web UI。

    设计策略：
    - 布局完全依赖 Gradio Row/Column/scale（不使用 CSS 控制布局）
    - elem_id 仅用于颜色/装饰性 CSS（不用于布局选择器）
    - 所有组件 visible=True（无隐藏桥接组件）
    - 会话列表用 Dropdown 替代 HTML+JS 桥接
    """
    loader: DistrictLoader = _get_district_loader()
    provinces: list[str] = loader.get_provinces()

    # 默认省/市/区设为北京朝阳区
    first_province: str = "北京市" if "北京市" in provinces else (provinces[0] if provinces else "")
    first_cities: list[str] = loader.get_cities("北京市") if first_province == "北京市" else []
    first_city: str = first_cities[0] if first_cities else ""
    first_districts: list[str] = loader.get_districts(first_city) if first_city else []
    first_district: str = (
        "朝阳区" if "朝阳区" in first_districts
        else (first_districts[0] if first_districts else "")
    )

    with gr.Blocks(
        title="local_life_agent — 本地生活决策助手",
    ) as app:

        # 初始化默认会话（v3.1：优先从文件加载）
        initial_store: dict[str, dict] = _load_sessions()
        if not initial_store:
            initial_tid: str = _generate_thread_id()
            initial_store[initial_tid] = {
                "title": "新对话",
                "messages": [],
                "created_at": time.time(),
            }
        else:
            # 取最近活跃会话作为当前线程
            latest_tid: str = max(
                initial_store.keys(),
                key=lambda k: initial_store[k].get("created_at", 0),
            )
            initial_tid = latest_tid
        initial_choices = _build_dropdown_choices(initial_store)

        # 隐藏状态
        conversations_state: gr.State = gr.State(initial_store)
        thread_id_state: gr.State = gr.State(initial_tid)

        # ── 标题区 ──
        gr.Markdown(
            "# local_life_agent\n"
            "### 本地生活决策助手 — 多人聚餐 · 酒店 · 娱乐智能推荐",
            elem_id="app-title-block",
        )

        with gr.Row(equal_height=False):

            # ──── 左侧：对话管理栏 ────
            with gr.Column(scale=1, min_width=180, elem_id="sidebar-col"):
                gr.Markdown("**📁 对话列表**")

                new_chat_btn: gr.Button = gr.Button(
                    "＋ 新建对话",
                    size="sm",
                    elem_id="new-chat-btn",
                )

                conversation_dropdown: gr.Dropdown = gr.Dropdown(
                    choices=initial_choices,
                    value=initial_tid,
                    label="历史对话",
                    interactive=True,
                )

                delete_conv_btn: gr.Button = gr.Button(
                    "🗑 删除当前对话",
                    size="sm",
                    elem_id="delete-conv-btn",
                )

            # ──── 中间：对话区 ────
            with gr.Column(scale=4, elem_id="chatbot-column"):

                with gr.Row(elem_id="location-row"):
                    province_dd: gr.Dropdown = gr.Dropdown(
                        label="省",
                        choices=provinces,
                        value=first_province or None,
                        scale=1,
                    )
                    city_dd: gr.Dropdown = gr.Dropdown(
                        label="市",
                        choices=first_cities,
                        value=first_city or None,
                        scale=1,
                    )
                    district_dd: gr.Dropdown = gr.Dropdown(
                        label="区/县",
                        choices=first_districts,
                        value=first_district or None,
                        scale=1,
                    )
                    location_input: gr.Textbox = gr.Textbox(
                        label="精确位置",
                        placeholder="如：朝阳大悦城、国贸三期...",
                        scale=2,
                    )

                chatbot: gr.Chatbot = gr.Chatbot(
                    label="对话",
                    height=480,
                    autoscroll=False,
                )

                with gr.Row(elem_id="input-row"):
                    msg_input: gr.Textbox = gr.Textbox(
                        placeholder=(
                            "例如：周六晚上4人聚餐，人均150，A想吃辣、B忌辣，需要有包间..."
                        ),
                        scale=10,
                        lines=2,
                        show_label=False,
                    )
                    send_btn: gr.Button = gr.Button(
                        "发送",
                        variant="primary",
                        scale=1,
                        size="sm",
                        elem_id="send-btn",
                    )

            # ──── 右侧：信息区 ────
            with gr.Column(scale=1, min_width=200, elem_id="right-panel"):

                with gr.Accordion("🛡️ 安全审查状态", open=False):
                    safety_status_display: gr.Markdown = gr.Markdown("等待输入...")

                with gr.Accordion("🗺️ 地图", open=False):
                    map_display: gr.HTML = gr.HTML("<p style='color:#888;text-align:center;padding:40px'>（获取推荐后可查看地图）</p>")

                with gr.Accordion("🗺️ 行程规划", open=False):
                    itinerary_display: gr.Markdown = gr.Markdown("（获取推荐后可提取行程）")

                with gr.Accordion("🔍 详情（偏好 / 候选 / Trace）", open=False):
                    preference_output: gr.Markdown = gr.Markdown("")
                    candidates_output: gr.Markdown = gr.Markdown("")
                    trace_output: gr.Markdown = gr.Markdown("")

                # ── v3.0 动作按钮 ──
                gr.Markdown("**⚡ 快捷动作**")
                with gr.Row():
                    share_btn: gr.Button = gr.Button("📤 分享", size="sm", scale=1)
                reminder_btn: gr.Button = gr.Button("⏰ 设置提醒", size="sm")
                action_output: gr.Markdown = gr.Markdown("")

        # ── 级联下拉绑定 ──
        province_dd.change(
            fn=_on_province_change,
            inputs=[province_dd],
            outputs=[city_dd],
        )
        city_dd.change(
            fn=_on_city_change,
            inputs=[city_dd],
            outputs=[district_dd],
        )

        # ── 对话管理回调 ──
        conv_mgmt_outputs = [
            conversations_state,
            thread_id_state,
            chatbot,
            conversation_dropdown,
        ]

        new_chat_btn.click(
            fn=_on_new_conversation,
            inputs=[conversations_state, thread_id_state],
            outputs=conv_mgmt_outputs,
        )

        conversation_dropdown.change(
            fn=_on_select_conversation,
            inputs=[conversation_dropdown, conversations_state],
            outputs=conv_mgmt_outputs,
        )

        delete_conv_btn.click(
            fn=_on_delete_and_new,
            inputs=[conversations_state, thread_id_state],
            outputs=conv_mgmt_outputs,
        )

        # ── 聊天回调 ──
        chat_outputs: list = [
            chatbot,
            safety_status_display,
            map_display,
            itinerary_display,
            preference_output,
            candidates_output,
            trace_output,
            msg_input,
            conversations_state,
            thread_id_state,
            conversation_dropdown,
        ]

        chat_inputs: list = [
            msg_input,
            province_dd,
            city_dd,
            district_dd,
            location_input,
            chatbot,
            thread_id_state,
            conversations_state,
        ]

        send_btn.click(
            fn=_handle_chat,
            inputs=chat_inputs,
            outputs=chat_outputs,
        )
        msg_input.submit(
            fn=_handle_chat,
            inputs=chat_inputs,
            outputs=chat_outputs,
        )

        # ── v3.0 动作按钮回调 ──
        share_btn.click(
            fn=_on_share,
            inputs=[chatbot, thread_id_state, conversations_state],
            outputs=[action_output],
        )
        reminder_btn.click(
            fn=_on_reminder,
            inputs=[chatbot],
            outputs=[action_output],
        )


    return app


# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
# 启动入口
# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

def main() -> None:
    """启动 Gradio Web 服务器。"""
    ensure_directories()
    _check_env()

    app: gr.Blocks = build_ui()
    app.queue(default_concurrency_limit=WEBUI_CONCURRENCY_LIMIT)
    app.launch(
        server_name=WEBUI_SERVER_NAME,
        server_port=WEBUI_SERVER_PORT,
        share=WEBUI_SHARE,
        show_error=True,
        theme=GRADIO_THEME,
        css=_CUSTOM_CSS,
    )


def _check_env() -> None:
    """启动前环境检查。"""
    load_dotenv()
    key: str = os.getenv("DEEPSEEK_API_KEY", "")
    if not key or "your_" in key:
        logger.warning(
            "[WebUI] 未检测到有效的 DEEPSEEK_API_KEY，"
            "LLM 调用可能失败。请编辑 .env 文件配置 API Key。"
        )
    else:
        logger.info(f"[WebUI] API Key 已配置 (长度={len(key)})")


if __name__ == "__main__":
    main()
