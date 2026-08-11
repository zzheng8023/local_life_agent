"""
分享卡片生成器 (Share Card Generator)

本模块将 Agent 的推荐结果生成为可转发的 Markdown 格式分享卡片。
适用于微信/钉钉等 IM 工具的文本分享场景。

设计：
- 纯函数，零外部依赖
- 输出适合移动端阅读的紧凑格式
"""

from __future__ import annotations

from typing import Any

from domain.entities import AgentState


def generate_share_card(state: AgentState) -> str:
    """从 Agent 运行状态生成可转发的 Markdown 分享卡片。

    Args:
        state: 工作流执行完成后的 AgentState。

    Returns:
        适合移动端分享的 Markdown 字符串。
    """
    lines: list[str] = []
    pref = state.user_preference

    # ── 标题 ──
    lines.append("## 🍽️ 本地生活决策助手 · 推荐方案")
    lines.append("")

    # ── 偏好摘要 ──
    meta_parts: list[str] = []
    if pref.city:
        meta_parts.append(f"📍 {pref.city}")
    if pref.budget:
        meta_parts.append(f"💰 {pref.budget}")
    if pref.taste:
        meta_parts.append(f"🍜 {pref.taste}")
    if pref.time:
        meta_parts.append(f"🕐 {pref.time}")
    if meta_parts:
        lines.append(" | ".join(meta_parts))
        lines.append("")

    # ── 推荐内容 ──
    recommendation: str = state.final_recommendation
    if recommendation:
        lines.append("### 📋 推荐方案")
        lines.append("")
        # 截断过长的推荐文本（移动端友好）
        if len(recommendation) > 1500:
            lines.append(recommendation[:1500] + "\n\n...(更多内容请打开 App 查看)")
        else:
            lines.append(recommendation)
        lines.append("")

    # ── 候选概览 ──
    if state.candidate_restaurants:
        lines.append("### 🏪 候选餐厅")
        lines.append("")
        for i, r in enumerate(state.candidate_restaurants[:5], 1):
            name: str = r.get("name", "?")
            rating: Any = r.get("rating", "—")
            price: str = r.get("avg_price", "—")
            lines.append(f"{i}. **{name}** — ★{rating} | {price}")

    if state.candidate_hotels:
        lines.append("")
        lines.append("### 🏨 附近酒店")
        lines.append("")
        for i, h in enumerate(state.candidate_hotels[:3], 1):
            lines.append(
                f"{i}. **{h.get('name', '?')}** — "
                f"★{h.get('rating', '—')} | {h.get('avg_price', '—')}"
            )

    if state.candidate_entertainments:
        lines.append("")
        lines.append("### 🎬 附近娱乐")
        lines.append("")
        for i, e in enumerate(state.candidate_entertainments[:3], 1):
            cat: str = e.get("category", "")
            lines.append(
                f"{i}. **{e.get('name', '?')}** [{cat}] — "
                f"★{e.get('rating', '—')}"
            )

    # ── 页脚 ──
    lines.append("")
    lines.append("---")
    lines.append(
        "🤖 由 [local_life_agent](https://github.com) 生成 · "
        "数据来源：高德地图 POI"
    )

    return "\n".join(lines)
