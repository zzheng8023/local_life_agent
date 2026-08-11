"""
追踪与评测日志 (Tracer & Evaluation Logger) — v2.0

本模块负责拦截并记录 Agent 系统在运行过程中产生的全部可观测数据。
采用非侵入式设计，通过依赖注入或装饰器模式挂载到工作流的关键节点。

v2.0 增强：
- 规则命中追踪：记录每轮对话中命中的所有系统规则（安全拦截、反问触发、兜底策略等）。
- 任务成功率指标：按轮次和阶段维度计算加权成功率，包含规则命中分析。
- 评测报告：强化 Markdown 报告，展示规则命中明细与成功率分解。

核心职责：
- Trace 记录：全程记录 LLM 调用（Prompt / Response）、工具执行、工作流状态转移。
- 规则命中追踪：安全拦截、反问触发、兜底策略、预算过滤等所有系统规则命中。
- 评测日志：为离线评测 (Offline Evaluation) 提供结构化的输入输出对。
- 指标计算：基于各阶段执行结果与规则命中情况，计算加权成功率。
- 报告生成：输出人类可读的 Markdown 格式评测报告（含规则命中明细）。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from loguru import logger

_TRACE_DIR: Path = Path(__file__).resolve().parent.parent / "logs" / "traces"


class TraceLogger:
    """工作流追踪与评测日志记录器 (v2.0)。

    为每轮对话提供完整的可观测能力，记录：
    - 阶段级 Trace（LLM 输入/输出、耗时）
    - 工具调用（名称、参数、结果摘要）
    - 安全规则命中（类型、详情）
    - 系统规则命中（反问触发、兜底触发等）
    - 失败/异常（阶段、错误类型、明细）
    - 加权成功率指标
    """

    def __init__(self) -> None:
        self._logs: list[dict[str, Any]] = []
        self._phase_timers: dict[str, float] = {}
        self._query: str = ""
        self._failures: list[dict[str, Any]] = []
        self._safety_rule_hits: list[dict[str, Any]] = []
        self._system_rule_hits: list[dict[str, Any]] = []
        self._tool_calls: list[dict[str, Any]] = []
        self._session_start: float = 0.0
        self._phase_status: dict[str, bool] = {}
        self._round_index: int = 0

    # ================================================================
    # 会话级
    # ================================================================

    def set_query(self, query: str) -> None:
        """标记新一轮对话开始，重置阶段状态并记录用户 Query。"""
        self._query = query
        self._session_start = time.perf_counter()
        self._round_index += 1
        self._phase_status = {}
        logger.info(f"[TRACE] round {self._round_index} start: {query[:80]}...")

    # ================================================================
    # 阶段计时
    # ================================================================

    def start_phase(self, phase_name: str) -> None:
        self._phase_timers[phase_name] = time.perf_counter()
        logger.info(f"[TRACE] >>> phase start: {phase_name}")

    def end_phase(
        self,
        phase_name: str,
        *,
        input_snapshot: Any = None,
        output_snapshot: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """结束阶段计时并记录 Trace 条目。"""
        start_time: float = self._phase_timers.pop(phase_name, 0.0)
        elapsed_ms: float = (time.perf_counter() - start_time) * 1000
        entry: dict[str, Any] = {
            "phase": phase_name,
            "round": self._round_index,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_ms": round(elapsed_ms, 2),
            "input": input_snapshot,
            "output": output_snapshot,
            **(extra or {}),
        }
        self._logs.append(entry)
        self._phase_status[phase_name] = True
        logger.info(
            f"[TRACE] <<< phase end: {phase_name} ({elapsed_ms:.2f}ms)"
        )
        return entry

    def record_phase_skip(self, phase_name: str, reason: str = "") -> None:
        """记录某阶段被跳过（如无住宿需求时跳过酒店搜索）。"""
        entry: dict[str, Any] = {
            "phase": phase_name,
            "round": self._round_index,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_ms": 0,
            "input": None,
            "output": None,
            "skipped": True,
            "skip_reason": reason,
        }
        self._logs.append(entry)
        self._phase_status[phase_name] = False
        logger.info(f"[TRACE] ── phase skip: {phase_name} ({reason})")

    # ================================================================
    # 工具调用记录
    # ================================================================

    def record_tool_call(
        self, tool_name: str, params: dict[str, Any], result_summary: Any
    ) -> None:
        entry: dict[str, Any] = {
            "tool_name": tool_name,
            "round": self._round_index,
            "params": params,
            "result_summary": result_summary,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._tool_calls.append(entry)
        logger.info(f"[TRACE] tool call: {tool_name}")

    # ================================================================
    # 规则命中记录（v2.0 增强）
    # ================================================================

    def record_safety_hit(self, rule_name: str, detail: str) -> None:
        """记录安全网关拦截规则命中。"""
        entry: dict[str, Any] = {
            "rule_category": "safety",
            "rule": rule_name,
            "detail": detail,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "round": self._round_index,
        }
        self._safety_rule_hits.append(entry)
        logger.warning(f"[TRACE] safety hit: {rule_name}")

    def record_system_rule_hit(
        self,
        rule_name: str,
        category: str = "system",
        detail: str = "",
    ) -> None:
        """记录通用系统规则命中（反问触发、兜底策略、降级逻辑等）。

        Args:
            rule_name: 规则名称，如 "反问触发"、"兜底推荐"、"降级模拟数据"。
            category: 规则分类（system / clarification / fallback / filter）。
            detail: 命中详情描述。
        """
        entry: dict[str, Any] = {
            "rule_category": category,
            "rule": rule_name,
            "detail": detail,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "round": self._round_index,
        }
        self._system_rule_hits.append(entry)
        logger.info(f"[TRACE] system rule hit [{category}]: {rule_name}")

    # ================================================================
    # 异常记录
    # ================================================================

    def record_failure(self, phase: str, error_type: str, detail: str) -> None:
        entry: dict[str, Any] = {
            "phase": phase,
            "error_type": error_type,
            "detail": detail,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "round": self._round_index,
        }
        self._failures.append(entry)
        logger.error(f"[TRACE] failure [{phase}]: {error_type}")

    # ================================================================
    # 指标计算（v2.0 增强）
    # ================================================================

    def calculate_metrics(self) -> dict[str, Any]:
        """计算当前会话的加权成功率与规则命中统计。

        权重分配（v2.0 重平衡）：
        - 偏好提取 (analyze)：30%
        - 工具检索 (retrieve)：25%
        - 推荐生成 (recommend)：20%
        - 安全审查 (safety)：15%
        - 规则合规加分：10%

        Returns:
            包含 overall_score、details、rule_hits_summary 的指标字典。
        """
        details: dict[str, dict[str, Any]] = {
            "analyze": {"score": 0.0, "max": 30.0, "status": "skip"},
            "retrieve": {"score": 0.0, "max": 25.0, "status": "skip"},
            "recommend": {"score": 0.0, "max": 20.0, "status": "skip"},
            "safety": {"score": 0.0, "max": 15.0, "status": "skip"},
            "rule_compliance": {"score": 0.0, "max": 10.0, "status": "skip"},
        }

        phase_outputs: dict[str, Any] = {}
        for entry in self._logs:
            phase: str = entry.get("phase", "")
            if phase in ("analyze", "retrieve", "recommend", "safety"):
                phase_outputs[phase] = entry.get("output", {})

        # ── Analyze：偏好提取（30 分）──
        analyze_output = phase_outputs.get("analyze", {})
        if isinstance(analyze_output, dict):
            agg = analyze_output.get("aggregated_preference", analyze_output)
            needs_clarification: bool = bool(
                analyze_output.get("needs_clarification", False)
            )
            if bool(agg.get("budget")) or bool(agg.get("taste")):
                details["analyze"]["score"] = 30.0
                details["analyze"]["status"] = "pass"
            elif needs_clarification:
                # 信息不足触发反问，这是正确的行为
                details["analyze"]["score"] = 20.0
                details["analyze"]["status"] = "clarify"
            else:
                details["analyze"]["status"] = "fail"

        # ── Retrieve：工具检索（25 分）──
        retrieve_output = phase_outputs.get("retrieve", {})
        if isinstance(retrieve_output, dict):
            count = retrieve_output.get("count", 0)
            if isinstance(count, (int, float)) and count > 0:
                details["retrieve"]["score"] = 25.0
                details["retrieve"]["status"] = "pass"
            else:
                details["retrieve"]["status"] = "fail"
        elif len(self._tool_calls) > 0:
            # 无阶段输出但有工具调用记录
            details["retrieve"]["score"] = 20.0
            details["retrieve"]["status"] = "pass"

        # ── Recommend：推荐生成（20 分）──
        recommend_output = phase_outputs.get("recommend", {})
        if isinstance(recommend_output, dict):
            has_rec: bool = not recommend_output.get("fallback", False)
            if has_rec:
                details["recommend"]["score"] = 20.0
                details["recommend"]["status"] = "pass"
            else:
                details["recommend"]["status"] = "fail"

        # ── Safety：安全审查（15 分）──
        safety_output = phase_outputs.get("safety", {})
        if isinstance(safety_output, dict):
            passed: bool = safety_output.get("passed", True)
            violations: list = safety_output.get("violations", [])
            if passed and not violations:
                details["safety"]["score"] = 15.0
                details["safety"]["status"] = "pass"
            elif passed and violations:
                # 有违规但已改写（通过了安全审查）
                details["safety"]["score"] = 10.0
                details["safety"]["status"] = "rewritten"
            else:
                details["safety"]["status"] = "fail"

        # ── 规则合规加分（10 分）──
        # 安全拦截命中 = 合规执行（不扣分）
        # 正确触发反问 = 合规执行
        # 正确触发兜底 = 合规执行
        safety_hit_count: int = len(self._safety_rule_hits)
        system_hit_count: int = len(self._system_rule_hits)
        failure_count: int = len(self._failures)

        if failure_count == 0:
            details["rule_compliance"]["score"] = 10.0
            details["rule_compliance"]["status"] = "pass"
        elif failure_count <= 2:
            details["rule_compliance"]["score"] = 5.0
            details["rule_compliance"]["status"] = "partial"
        else:
            details["rule_compliance"]["status"] = "fail"

        overall_score: float = sum(d["score"] for d in details.values())

        # ── 规则命中汇总 ──
        all_rule_hits: list[dict[str, Any]] = (
            self._safety_rule_hits + self._system_rule_hits
        )
        rule_hits_by_category: dict[str, int] = {}
        for hit in all_rule_hits:
            cat: str = hit.get("rule_category", "unknown")
            rule_hits_by_category[cat] = rule_hits_by_category.get(cat, 0) + 1

        return {
            "overall_score": overall_score,
            "max_score": 100.0,
            "details": details,
            "safety_hits": safety_hit_count,
            "system_rule_hits": system_hit_count,
            "failure_count": failure_count,
            "tool_calls": len(self._tool_calls),
            "rule_hits_by_category": rule_hits_by_category,
        }

    # ================================================================
    # 报告生成（v2.0 增强）
    # ================================================================

    def generate_report(self) -> str:
        """生成完整的 Markdown 评测报告。

        包含：会话概览、用户 Query、阶段执行详情、规则命中明细、失败记录、加权成功率。
        """
        metrics: dict[str, Any] = self.calculate_metrics()
        details: dict[str, Any] = metrics["details"]
        total_elapsed: float = sum(
            entry.get("elapsed_ms", 0.0) for entry in self._logs
        )

        def _bar(score: float, max_s: float, width: int = 24) -> str:
            if max_s == 0:
                return "▁" * width
            ratio: float = min(score / max_s, 1.0)
            filled: int = int(ratio * width)
            return "█" * filled + "▁" * (width - filled)

        lines: list[str] = [
            "",
            "# Trace 评测报告 (v2.0)",
            "",
            "| 项目 | 值 |",
            "|---|---|",
            f"| 会话时间 | {time.strftime('%Y-%m-%d %H:%M:%S')} |",
            f"| 对话轮次 | Round {self._round_index} |",
            f"| 总耗时 | {total_elapsed:.2f} ms |",
            f"| 阶段数 | {len(self._logs)} |",
            f"| 工具调用 | {metrics['tool_calls']} 次 |",
            f"| 安全规则命中 | {metrics['safety_hits']} 次 |",
            f"| 系统规则命中 | {metrics['system_rule_hits']} 次 |",
            f"| 异常次数 | {metrics['failure_count']} 次 |",
            "",
            "## 用户 Query",
            "",
            "> " + (self._query or "(未记录)"),
            "",
        ]

        # ── 各阶段执行详情表 ──
        if self._logs:
            lines.extend([
                "## 各阶段执行详情",
                "",
                "| 阶段 | 轮次 | 耗时 | 状态 | 输出摘要 |",
                "|---|---|---|---|---|",
            ])
            for entry in self._logs:
                phase: str = entry.get("phase", "?")
                round_n: int = entry.get("round", 0)
                elapsed: float = entry.get("elapsed_ms", 0.0)
                skipped: bool = entry.get("skipped", False)
                out: Any = entry.get("output")
                out_str: str = ""
                if skipped:
                    out_str = f"⏭ {entry.get('skip_reason', '')}"
                elif isinstance(out, dict):
                    out_keys: list[str] = list(out.keys())[:2]
                    out_str = ", ".join(
                        f"{k}={str(out[k])[:30]}" for k in out_keys
                    )
                else:
                    out_str = str(out)[:45] if out is not None else "-"
                status_str: str = "⏭ 跳过" if skipped else "✓"
                lines.append(
                    f"| {phase} | {round_n} | {elapsed:.2f} ms | {status_str} | {out_str} |"
                )

        # ── 系统规则命中明细 ──
        all_system_hits: list[dict[str, Any]] = self._system_rule_hits
        if all_system_hits:
            lines.append("")
            lines.append("## 系统规则命中明细")
            lines.append("")
            lines.append("| # | 分类 | 规则名称 | 详情 |")
            lines.append("|---|---|---|---|")
            for i, hit in enumerate(all_system_hits, 1):
                cat: str = hit.get("rule_category", "?")
                rule: str = hit.get("rule", "?")
                detail: str = str(hit.get("detail", ""))[:60]
                lines.append(f"| {i} | {cat} | {rule} | {detail} |")

        # ── 安全规则命中明细 ──
        if self._safety_rule_hits:
            lines.append("")
            lines.append("## 安全规则命中明细")
            lines.append("")
            lines.append("| # | 规则 | 详情 |")
            lines.append("|---|---|---|")
            for i, hit in enumerate(self._safety_rule_hits, 1):
                rule: str = hit.get("rule", "?")
                detail: str = str(hit.get("detail", ""))[:80]
                lines.append(f"| {i} | {rule} | {detail} |")

        # ── 失败记录 ──
        if self._failures:
            lines.append("")
            lines.append("## 失败与异常记录")
            for i, f_entry in enumerate(self._failures, 1):
                lines.append(
                    f"**{i}. [{f_entry['phase']}] `{f_entry['error_type']}`**"
                )
                lines.append(f"> {f_entry['detail'][:300]}")
                lines.append("")

        # ── 加权成功率 ──
        lines.append("")
        lines.append("## 加权成功率")
        lines.append("")
        for phase_key, label, description in [
            ("analyze", "偏好提取", "是否正确提取或合理触发反问"),
            ("retrieve", "工具检索", "是否成功获取候选数据"),
            ("recommend", "推荐生成", "是否产出有效推荐文案"),
            ("safety", "安全审查", "是否合规输出或正确拦截"),
            ("rule_compliance", "规则合规", "系统规则命中无异常"),
        ]:
            d: dict[str, Any] = details[phase_key]
            score: float = d["score"]
            max_s: float = d["max"]
            status: str = d["status"]
            icon: str = {
                "pass": "✅",
                "fail": "❌",
                "skip": "⏭️",
                "partial": "⚠️",
                "clarify": "🔍",
                "rewritten": "✏️",
            }.get(status, "❓")
            lines.append(
                f"- {icon} **{label}** `{_bar(score, max_s)}` {score:.1f} / {max_s:.1f}  "
                f"({description})"
            )
        lines.append("")
        overall: float = metrics["overall_score"]
        lines.append(
            f"### 综合得分: **{overall:.1f} / 100.0** `{_bar(overall, 100.0)}`"
        )
        lines.append("")

        # ── 规则命中统计 ──
        rule_hits_cat: dict[str, int] = metrics.get("rule_hits_by_category", {})
        if rule_hits_cat:
            lines.append("## 规则命中统计")
            lines.append("")
            lines.append("| 分类 | 命中次数 |")
            lines.append("|---|---|")
            for cat, count in sorted(rule_hits_cat.items()):
                lines.append(f"| {cat} | {count} |")
            lines.append("")

        # ── v3.1: API 用量摘要 ──
        try:
            from infrastructure.usage_tracker import UsageTracker
            usage_report: str = UsageTracker().generate_report()
            lines.append(usage_report)
        except Exception:
            pass

        report: str = "\n".join(lines)
        logger.info(f"\n{report}")
        return report

    # ================================================================
    # 便捷方法
    # ================================================================

    def dump_report(self) -> str:
        report: str = self.generate_report()
        self._persist_if_enabled()
        return report

    def get_logs(self) -> list[dict[str, Any]]:
        return list(self._logs)

    def get_full_snapshot(self) -> dict[str, Any]:
        """获取当前会话的完整可观测快照。"""
        return {
            "round": self._round_index,
            "query": self._query,
            "logs": self._logs,
            "failures": self._failures,
            "safety_rule_hits": self._safety_rule_hits,
            "system_rule_hits": self._system_rule_hits,
            "tool_calls": self._tool_calls,
            "metrics": self.calculate_metrics(),
        }

    # ================================================================
    # v5.0：文件持久化
    # ================================================================

    def _persist_if_enabled(self) -> None:
        """将当前会话的完整 Trace 快照持久化到 logs/traces/ 目录。

        文件名格式: trace_{round_index}_{timestamp}.json
        """
        try:
            _TRACE_DIR.mkdir(parents=True, exist_ok=True)
            snapshot: dict[str, Any] = self.get_full_snapshot()
            ts: str = time.strftime("%Y%m%d_%H%M%S")
            filename: str = f"trace_{self._round_index}_{ts}.json"
            filepath: Path = _TRACE_DIR / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)

            logger.info(f"[Trace] 持久化: {filepath}")

            # ── 日志轮转：保留最近 TRACE_MAX_COUNT 个文件 ──
            self._rotate_traces()
        except Exception as exc:
            logger.error(f"[Trace] 持久化失败: {exc}")

    @staticmethod
    def _rotate_traces() -> None:
        """清理旧的 trace 文件，仅保留最近 TRACE_MAX_COUNT 个。"""
        try:
            from application.config import TRACE_MAX_COUNT

            trace_files: list[Path] = sorted(
                _TRACE_DIR.glob("trace_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for old_file in trace_files[TRACE_MAX_COUNT:]:
                try:
                    old_file.unlink()
                    logger.debug(f"[Trace] 轮转删除: {old_file.name}")
                except OSError:
                    pass
        except Exception as exc:
            logger.warning(f"[Trace] 轮转失败: {exc}")
