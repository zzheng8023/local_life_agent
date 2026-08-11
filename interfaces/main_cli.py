"""
主程序入口 (Main CLI Entry Point) — v2.0

本模块是 local_life_agent 系统的启动入口，负责：
1. 加载环境变量与系统配置。
2. 执行依赖注入 (Dependency Injection)：
   - 实例化基础设施层的具体实现（LLM 适配器、餐饮工具、酒店/娱乐工具）。
   - 将它们注入到应用层的 LangGraph 工作流编排器中。
3. 启动连续对话交互循环：
   - 接收用户输入，通过 chat() 方法驱动 LangGraph 图引擎。
   - 自动识别反问（Clarification）与最终推荐，展示对应输出。
   - 利用 thread_id 保持跨轮次上下文记忆。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 将项目根目录加入 Python 搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from application.ports import ILLMClient, ITool
from application.workflow import LocalLifeWorkflow
from domain.tool_registry import TOOL_REGISTRY, get_display_domains, get_preference_tools
from infrastructure.llm_adapter import OpenAILikeClient
from infrastructure.amap_tool import AmapRestaurantTool
from infrastructure.amap_hotel_tool import AmapHotelTool
from infrastructure.amap_entertainment_tool import AmapEntertainmentTool
from infrastructure.amap_shopping_tool import AmapShoppingTool
from infrastructure.amap_transit_tool import AmapTransitTool
from infrastructure.amap_transit_direction import AmapTransitDirectionTool
from infrastructure.amap_parking_tool import AmapParkingTool
from infrastructure.amap_bike_tool import AmapBikeTool

# ── 日志配置 ──
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


# ================================================================
# 依赖注入
# ================================================================

def bootstrap() -> LocalLifeWorkflow:
    """执行依赖注入组装。

    遵循 DDD 依赖倒置原则：
    1. 从环境变量加载配置。
    2. 实例化基础设施层的具体实现。
    3. 将实现注入到应用层的编排器中（绑定到端口接口）。

    Returns:
        配置完成的 LocalLifeWorkflow 实例，包含 LLM + 餐饮工具 + 酒店/娱乐工具。
    """
    load_dotenv()
    logger.info("[Bootstrap] 环境变量已加载")

    # ── 基础设施层实例化 ──
    smart_llm: ILLMClient = OpenAILikeClient()
    fast_llm: ILLMClient = OpenAILikeClient(
        model=os.getenv("FAST_MODEL", os.getenv("DEFAULT_MODEL", "deepseek-v4-flash")),
        api_key=os.getenv("FAST_API_KEY", os.getenv("DEEPSEEK_API_KEY", "")),
        base_url=os.getenv("FAST_BASE_URL", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")),
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
    logger.info(
        "[Bootstrap] 基础设施层: "
        f"LLM(smart={smart_llm.model_name}, fast={fast_llm.model_name}), "
        + ", ".join(tool_names)
    )

    # ── 依赖注入：实现 → 端口 ──
    workflow: LocalLifeWorkflow = LocalLifeWorkflow(
        llm=smart_llm,
        fast_llm=fast_llm,
        tools=tools,
        transit_direction_tool=transit_direction_tool,
    )
    logger.info("[Bootstrap] 依赖注入完成，LangGraph 工作流就绪")

    return workflow


# ================================================================
# 主循环
# ================================================================

def main() -> None:
    """主程序入口：连续对话交互循环。

    流程：
    1. 检查 API Key → 组装工作流。
    2. 进入 while True 循环，读取用户输入。
    3. 调用 workflow.chat(input, thread_id) 驱动图引擎。
    4. 根据 state.needs_clarification 决定展示反问或推荐结果。
    5. 输入 'quit' / 'exit' 退出。
    """
    console: Console = Console()
    THREAD_ID: str = "session_001"

    # ── 打印欢迎横幅 ──
    console.print()
    console.print(
        Panel(
            "[bold cyan]local_life_agent v2.0[/]\n\n"
            "基于 LangGraph 的多轮对话本地生活决策助手\n\n"
            "[dim]覆盖领域：[/]🍽️ 餐饮  🏨 酒旅  🎬 娱乐\n"
            "[dim]核心能力：[/]多用户冲突协调 · 时空连续推理 · 安全履约审计\n"
            "[dim]对话记忆：[/]thread_id = [yellow]session_001[/]",
            title="Welcome",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print(
        "[dim]提示：输入 [bold]quit[/] 或 [bold]exit[/] 退出对话[/]\n"
    )

    # ── API Key 检查 ──
    api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key or "your_" in api_key:
        console.print(
            Panel(
                "[bold yellow]⚠ 未检测到有效的 API Key[/]\n\n"
                "请在 [cyan].env[/] 文件中配置 [bold]DEEPSEEK_API_KEY[/] 后重试。\n"
                "可前往 https://platform.deepseek.com/api_keys 获取 Key。",
                title="配置提示",
                border_style="yellow",
            )
        )
        logger.warning("[Main] 未配置有效的 API Key，程序退出")
        return

    # ── 依赖注入 ──
    workflow: LocalLifeWorkflow = bootstrap()

    # ── 对话循环 ──
    while True:
        try:
            user_input: str = console.input(
                "[bold cyan]You > [/]"
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]对话结束，再见。[/]")
            break

        # 空输入跳过
        if not user_input:
            continue

        # 退出指令
        if user_input.lower() in ("quit", "exit", "退出"):
            console.print("[dim]对话结束，再见。[/]")
            break

        # ── v3.1：流式运行工作流 ──
        console.print()
        state = None
        accumulated = ""

        with Live(
            Markdown("⏳ 思考中..."),
            console=console,
            refresh_per_second=10,
            transient=False,
        ) as live:
            for token, maybe_state in workflow.run_stream(
                user_input, thread_id=THREAD_ID
            ):
                if maybe_state is not None:
                    # 最终状态（chitchat / clarify / ood / 完成后）
                    state = maybe_state
                else:
                    accumulated += token
                    live.update(Markdown(accumulated))

        # ── 处理结果 ──
        if state is None:
            console.print("[yellow]⚠️ 未获取到有效状态[/]")
            continue

        if state.needs_clarification:
            # === 反问场景 ===
            question: str = (
                state.clarification_question or state.final_recommendation
            )
            if not accumulated:
                # 澄清路径不走流式，直接展示 Panel
                console.print(
                    Panel(
                        Markdown(question),
                        title="[bold yellow]🔍 Agent 需要确认更多信息[/]",
                        border_style="yellow",
                        padding=(1, 2),
                    )
                )
            _print_analysis_meta(console, state)

        else:
            # === 推荐场景 ===
            _print_recommendation(console, state)

    logger.info("[Main] 程序正常退出")


# ================================================================
# 输出辅助
# ================================================================

def _print_recommendation(console: Console, state) -> None:
    """打印最终推荐结果及元数据。"""
    # ── 安全审查状态 ──
    safety_icon: str = "🛡️ 通过"
    safety_style: str = "green"
    if state.safety_violations:
        safety_icon = "⚠️ 已有违规改写"
        safety_style = "yellow"
    elif not state.safety_passed:
        safety_icon = "🛑 拦截"
        safety_style = "red"

    # ── 主业推荐文案 ──
    console.print(
        Panel(
            Markdown(state.final_recommendation or "（无推荐内容）"),
            title=f"[bold green]🤖 Agent 回复（安全审查: {safety_icon}）[/]",
            border_style="green",
            padding=(1, 2),
        )
    )

    console.print()

    # ── 偏好 → 候选 → Trace 横向三栏 ──
    _print_preference_table(console, state)
    console.print()
    _print_candidates_table(console, state)
    console.print()
    _print_trace_summary(console, state)
    console.print()


def _print_analysis_meta(console: Console, state) -> None:
    """反问场景下展示已提取的偏好与分析元数据。"""
    pref = state.user_preference
    table = Table(title="已提取信息", border_style="dim blue")
    table.add_column("维度", style="cyan")
    table.add_column("当前值", style="white")

    if pref.budget:
        table.add_row("预算", pref.budget)
    if pref.taste:
        table.add_row("口味", pref.taste)
    if pref.restrictions:
        table.add_row("忌口", pref.restrictions)
    if pref.distance:
        table.add_row("距离", pref.distance)
    if pref.city:
        table.add_row("城市", pref.city)
    if pref.time:
        table.add_row("时间", pref.time)
    # ── 动态生成领域偏好行（来自 registry）──
    for td in get_preference_tools(TOOL_REGISTRY):
        val = getattr(pref, td.preference_field, None)
        if val:
            table.add_row(td.preference_table_label, val)
    if state.conflict_strategy:
        table.add_row("冲突策略", Text(state.conflict_strategy[:60] + "..."))

    if table.row_count == 0:
        table.add_row("—", "[dim](尚未提取到有效偏好)[/]")

    console.print(table)


def _print_preference_table(console: Console, state) -> None:
    """打印提取的用户偏好表格（含每人明细）。"""
    pref = state.user_preference
    table = Table(
        title="🧑‍🤝‍🧑 提取的用户偏好",
        title_style="bold cyan",
        border_style="cyan",
    )
    table.add_column("维度", style="cyan")
    table.add_column("值", style="white")

    rows = [
        ("预算", pref.budget),
        ("口味", pref.taste),
        ("忌口", pref.restrictions),
        ("距离", pref.distance),
        ("城市", pref.city),
        ("时间", pref.time),
        ("儿童", "是" if pref.has_kids else "否"),
        ("停车", "是" if pref.need_parking else "否"),
        ("精确位置", pref.freeform_location or "—"),
    ]
    for dim, val in rows:
        if val:
            table.add_row(dim, val)
    # ── 动态生成领域偏好行（来自 registry）──
    for td in get_preference_tools(TOOL_REGISTRY):
        val = getattr(pref, td.preference_field, None)
        if val:
            table.add_row(td.preference_table_label, val)
    console.print(table)

    # ── v2.2: 每人偏好明细 ──
    ind_prefs = getattr(state, "individual_preferences", []) or []
    if ind_prefs:
        console.print()
        ind_table = Table(
            title="👤 个人偏好明细",
            title_style="bold yellow",
            border_style="yellow",
        )
        ind_table.add_column("用户", style="cyan")
        ind_table.add_column("预算", style="white")
        ind_table.add_column("口味", style="white")
        ind_table.add_column("忌口", style="white")
        ind_table.add_column("关键原话", style="dim white")
        for ip in ind_prefs:
            ind_table.add_row(
                ip.name,
                ip.budget or "—",
                ip.taste or "—",
                ip.restrictions or "—",
                ip.key_utterance[:40] if ip.key_utterance else "—",
            )
        console.print(ind_table)


def _print_candidates_table(console: Console, state) -> None:
    """打印候选数据表格 — 餐厅手动渲染，其余领域由 registry 动态驱动。"""
    has_restaurants: bool = bool(state.candidate_restaurants)

    # 检查是否有任何领域有数据
    any_data = has_restaurants
    if not any_data:
        for td in get_display_domains(TOOL_REGISTRY, skip_restaurant=True):
            if getattr(state, td.candidate_key, []):
                any_data = True
                break
    if not any_data:
        return

    table = Table(
        title="🏪 候选数据概览",
        title_style="bold magenta",
        border_style="magenta",
    )
    table.add_column("领域", style="cyan")
    table.add_column("数量", style="white")

    if has_restaurants:
        table.add_row(
            "🍽️ 餐厅",
            str(len(state.candidate_restaurants)),
        )
        # 列出前 3 家
        for i, r in enumerate(state.candidate_restaurants[:3], 1):
            dist: str = (
                f"{r.get('distance_km')}km" if r.get("distance_km") else "—"
            )
            table.add_row(
                f"  {i}. {r.get('name', '?')}",
                f"★{r.get('rating', '—')} | 人均{r.get('avg_price', '')} | {dist}",
            )

    # ── 其余领域：从 registry 动态驱动 ──
    for td in get_display_domains(TOOL_REGISTRY, skip_restaurant=True):
        candidates = getattr(state, td.candidate_key, [])
        if not candidates:
            continue
        label = td.candidate_section_title
        table.add_row(label, str(len(candidates)))
        max_items = 3
        for item in candidates[:max_items]:
            name = item.get("name", "?")
            # 收集要展示的子字段
            extras: list[str] = []
            for col in td.candidate_columns[1:3]:  # 跳过 name，取 2 个额外字段
                val = item.get(col, "—")
                if isinstance(val, list):
                    val = ", ".join(val[:2]) if val else "—"
                extras.append(str(val))
            extra_str = " | ".join(extras)
            table.add_row(f"  {name}", extra_str)

    console.print(table)


def _print_trace_summary(console: Console, state) -> None:
    """打印 Trace 评测摘要。"""
    metrics = state.trace_logs
    if not metrics:
        return

    table = Table(
        title="📊 执行追踪",
        title_style="bold blue",
        border_style="blue",
    )
    table.add_column("阶段", style="cyan")
    table.add_column("耗时", style="white")

    for entry in metrics:
        phase: str = entry.get("phase", "?")
        elapsed: float = entry.get("elapsed_ms", 0)
        skipped: bool = entry.get("skipped", False)
        status_icon: str = "⏭" if skipped else "✓"
        table.add_row(
            f"{status_icon} {phase}",
            f"{elapsed:.2f} ms" if not skipped else f"跳过 ({entry.get('skip_reason', '')})",
        )

    # 安全违规
    violations: list[str] = getattr(state, "safety_violations", []) or []
    if violations:
        table.add_row(
            "🛡️ 安全违规",
            ", ".join(violations),
        )

    console.print(table)


# ================================================================
# 入口
# ================================================================

if __name__ == "__main__":
    main()
