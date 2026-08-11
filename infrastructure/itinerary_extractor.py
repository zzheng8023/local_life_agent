"""
行程提取器 (Itinerary Extractor)

本模块从推荐文本中提取结构化行程，供 Web UI 右侧面板展示。

设计：
- 优先从 Markdown 表格解析（推荐 prompt 要求输出的标准格式）
- 次选 LLM 结构化提取（generate_json）
- 兜底正则关键词扫描
- 失败时返回空列表（不阻塞主流程）
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

# ── 行程表格解析正则 ──
# 匹配 Markdown 表格行：| 时间 | 地点 | 行动 | 备注 |
# 捕获四个字段
_TABLE_ROW_RE = re.compile(
    r"^\|\s*"
    r"(\d{1,2}[:：]\d{2})\s*\|"   # 时间
    r"\s*(.+?)\s*\|"               # 地点
    r"\s*(.+?)\s*\|"               # 行动
    r"\s*(.*?)\s*\|"               # 备注
    r"\s*$",
    re.MULTILINE,
)


class ItineraryExtractor:
    """从推荐文本提取结构化行程。

    将 LLM 自然语言推荐解析为时间线格式:
    [
        {"time": "18:30", "location": "川味观·臻选", "action": "聚餐",
         "note": "人均140元，有包间"},
        {"time": "20:30", "location": "万达影城(国贸店)", "action": "看电影",
         "note": "步行5分钟可达"},
    ]
    """

    @staticmethod
    def extract(
        recommendation_text: str,
        llm_client: Any | None = None,
    ) -> list[dict[str, Any]]:
        """从推荐文本提取行程步骤。

        优先级：
        1. Markdown 表格解析（推荐 prompt 要求的标准格式，零 LLM 开销）
        2. LLM 结构化提取（如果提供了 llm_client）
        3. 正则关键词扫描兜底

        Args:
            recommendation_text: Agent 生成的完整推荐文本。
            llm_client: 可选，LLM 客户端（需实现 generate_json 方法）。
                        为 None 时跳过 LLM 提取步骤。

        Returns:
            结构化的行程步骤列表。
        """
        if not recommendation_text or not recommendation_text.strip():
            return []

        # ── 1. 优先从 Markdown 表格解析（最直接、最可靠）──
        table_steps = ItineraryExtractor._parse_table(recommendation_text)
        if table_steps:
            logger.info(
                f"[Itinerary] 表格解析提取 {len(table_steps)} 个行程步骤"
            )
            return table_steps

        # ── 2. LLM 结构化提取 ──
        if llm_client and hasattr(llm_client, "generate_json"):
            llm_steps = ItineraryExtractor._llm_extract(
                recommendation_text, llm_client
            )
            if llm_steps:
                return llm_steps

        # ── 3. 正则关键词扫描兜底 ──
        return ItineraryExtractor._keyword_extract(recommendation_text)

    # ------------------------------------------------------------------
    # 策略 1：Markdown 表格解析
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_table(text: str) -> list[dict[str, Any]]:
        """从推荐文本解析 Markdown 行程表格。

        匹配格式：
        | 时间 | 地点 | 行动 | 备注 |
        |---|---|---|---|
        | 18:30 | 大董烤鸭(工体店) | 🍽️ 聚餐 | 人均180元 |
        """
        # 先找到表格区域：以 "| 时间 |" 开头的行
        header_match = re.search(r"^\|\s*时间\s*\|", text, re.MULTILINE)
        if not header_match:
            return []

        # 从表头之后开始匹配数据行
        tail = text[header_match.end():]

        # 跳过表头行剩余部分和分隔行
        next_newline = tail.find("\n")
        if next_newline != -1:
            tail = tail[next_newline + 1:]
        # 跳过分隔行 (|---|---|...)
        sep_newline = tail.find("\n")
        if sep_newline != -1 and re.match(r"^\s*\|[\s\-|]+\|", tail):
            tail = tail[sep_newline + 1:]

        steps: list[dict[str, Any]] = []
        action_map: dict[str, str] = {
            "聚餐": "🍽️ 聚餐",
            "看电影": "🎬 看电影",
            "住宿": "🏨 住宿",
            "购物": "🛍️ 购物",
            "出行": "🚌 出行",
            "唱歌": "🎤 唱歌",
            "游乐园": "🎢 游乐园",
        }

        seen_locations: set[str] = set()
        for match in _TABLE_ROW_RE.finditer(tail):
            time_str = match.group(1).replace("：", ":").strip()
            location = match.group(2).strip()
            raw_action = match.group(3).strip()
            note = match.group(4).strip()

            # 跳过明显不是数据行的匹配
            if not time_str or time_str.lower() in ("时间", "time"):
                continue
            if location.lower() in ("地点", "location"):
                continue
            if not location:
                continue

            # 去重：同地点只保留第一次出现的行
            if location in seen_locations:
                continue
            seen_locations.add(location)

            # 规范化行动类型（移除 emoji 前缀，纯文本）
            clean_action = re.sub(
                r"^[\U0001F300-\U0001FAFF]\s*", "", raw_action
            ).strip()
            # 反向查找：如果文本包含关键词，映射回规范动作
            for key, emoji_label in action_map.items():
                if key in clean_action or key in raw_action:
                    clean_action = key
                    break

            if not clean_action:
                clean_action = "活动"

            note = note.rstrip("|").strip()

            steps.append({
                "time": time_str,
                "location": location[:60],
                "action": clean_action,
                "note": note[:60],
            })

        return steps

    # ------------------------------------------------------------------
    # 策略 2：LLM 提取
    # ------------------------------------------------------------------

    @staticmethod
    def _llm_extract(
        text: str, llm_client: Any
    ) -> list[dict[str, Any]]:
        from domain.prompt_specs import PromptManager

        system_prompt: str = PromptManager.build_itinerary_extraction_prompt()
        user_prompt: str = f"请从以下推荐文本中提取行程：\n\n{text}"

        try:
            result: dict[str, Any] = llm_client.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            steps: list[dict[str, Any]] = result.get("steps", [])
            if steps:
                logger.info(f"[Itinerary] LLM 提取 {len(steps)} 个行程步骤")
                return steps
        except Exception as exc:
            logger.error(f"[Itinerary] LLM 提取失败: {exc}")

        return []

    # ------------------------------------------------------------------
    # 策略 3：正则关键词扫描兜底
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_extract(text: str) -> list[dict[str, Any]]:
        """基于关键词的简单行程提取（不依赖 LLM）。

        扫描文本中的时间/地点模式，构建基础行程。
        """
        steps: list[dict[str, Any]] = []
        lines: list[str] = text.split("\n")

        time_pattern = re.compile(
            r"(\d{1,2}[:：]\d{2})\s*(.{0,30})"
        )
        location_patterns: list[str] = [
            "餐厅", "酒店", "电影院", "KTV", "商场", "影院",
            "馆", "店", "城", "广场",
        ]

        step_index: int = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue

            time_match = time_pattern.search(line)
            has_location: bool = any(kw in line for kw in location_patterns)

            if time_match or has_location:
                step_index += 1
                time_str: str = time_match.group(1) if time_match else ""
                rest: str = line[time_match.end():].strip() if time_match else line

                action: str = "活动"
                if any(kw in rest for kw in ["吃", "餐", "饭", "菜"]):
                    action = "🍽️ 聚餐"
                elif any(kw in rest for kw in ["电影", "影院", "KTV", "唱"]):
                    action = "🎬 娱乐"
                elif any(kw in rest for kw in ["酒店", "住宿", "入住"]):
                    action = "🏨 住宿"
                elif any(kw in rest for kw in ["逛", "购物", "商场", "买"]):
                    action = "🛍️ 购物"
                elif any(kw in rest for kw in ["公交", "地铁", "出行", "交通"]):
                    action = "🚌 出行"

                steps.append({
                    "time": time_str,
                    "location": rest[:60] if rest else line[:60],
                    "action": action,
                    "note": "",
                })

        if steps:
            logger.info(
                f"[Itinerary] 关键词扫描提取 {len(steps)} 个步骤"
            )
        return steps
