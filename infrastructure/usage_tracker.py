"""
用量追踪器 (Usage Tracker) — v3.1

本模块提供全局线程安全的 API 用量与 Token 统计单例，
用于追踪 LLM 调用和高德 API 调用的次数与 Token 消耗。

核心设计：
- 线程安全的单例模式（threading.Lock）
- 按模型分组记录 Token 用量（prompt / completion / total）
- 高德 API 调用次数独立计数
- generate_report() 返回可读的 Markdown 表格
"""

from __future__ import annotations

import threading
from typing import Any


class UsageTracker:
    """线程安全的 API 用量追踪器（单例）。"""

    _instance: UsageTracker | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> UsageTracker:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._llm_usage: dict[str, dict[str, Any]] = {}
                    obj._amap_calls: int = 0
                    obj._usage_lock = threading.Lock()
                    cls._instance = obj
        return cls._instance

    # ------------------------------------------------------------------
    # LLM 用量
    # ------------------------------------------------------------------

    def record_llm_call(
        self,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        """记录一次 LLM 调用的 Token 消耗。"""
        with self._usage_lock:
            if model not in self._llm_usage:
                self._llm_usage[model] = {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            entry = self._llm_usage[model]
            entry["calls"] += 1
            entry["prompt_tokens"] += prompt_tokens
            entry["completion_tokens"] += completion_tokens
            entry["total_tokens"] += total_tokens

    # ------------------------------------------------------------------
    # 高德 API
    # ------------------------------------------------------------------

    def record_amap_call(self) -> None:
        """记录一次高德 API 调用。"""
        with self._usage_lock:
            self._amap_calls += 1

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get_llm_usage(self) -> dict[str, dict[str, Any]]:
        """获取 LLM 用量统计快照。"""
        with self._usage_lock:
            return dict(self._llm_usage)

    def get_amap_calls(self) -> int:
        """获取高德 API 调用总数。"""
        with self._usage_lock:
            return self._amap_calls

    def generate_report(self) -> str:
        """生成可读的用量报告（Markdown 表格）。"""
        with self._usage_lock:
            parts: list[str] = ["## 📊 API 用量统计\n"]

            if self._llm_usage:
                parts.append("### 🤖 LLM 调用\n")
                parts.append("| 模型 | 调用次数 | Prompt Tokens | Completion Tokens | Total Tokens |")
                parts.append("|---|---|---|---|---|")
                for model, stats in sorted(self._llm_usage.items()):
                    parts.append(
                        f"| {model} | {stats['calls']} | "
                        f"{stats['prompt_tokens']:,} | "
                        f"{stats['completion_tokens']:,} | "
                        f"{stats['total_tokens']:,} |"
                    )
            else:
                parts.append("（暂无 LLM 调用记录）\n")

            parts.append(f"\n### 🗺️ 高德 API\n\n调用次数：**{self._amap_calls}**\n")

            # 粗略成本估算（按常见价格）
            if self._llm_usage:
                parts.append("\n### 💰 估算成本\n")
                total_cost: float = 0.0
                for model, stats in self._llm_usage.items():
                    # 假设价格（元/1M tokens）— 大致参考
                    if "deepseek" in model.lower():
                        prompt_price = 1.0  # $0.14/1M ≈ ¥1.0/1M
                        completion_price = 2.0
                    elif "qwen" in model.lower():
                        prompt_price = 2.0
                        completion_price = 6.0
                    else:
                        prompt_price = 2.0
                        completion_price = 6.0

                    cost = (
                        stats["prompt_tokens"] / 1_000_000 * prompt_price
                        + stats["completion_tokens"] / 1_000_000 * completion_price
                    )
                    total_cost += cost
                    if cost > 0.001:
                        parts.append(f"- {model}: ~¥{cost:.4f}")
                if total_cost > 0.001:
                    parts.append(f"\n**总计: ~¥{total_cost:.4f}**")
                else:
                    parts.append("（调用量太小，成本可忽略）")

            return "\n".join(parts)
