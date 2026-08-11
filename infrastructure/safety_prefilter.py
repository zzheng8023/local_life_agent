"""
安全预筛选器 (Safety Pre-filter)

本模块在 LLM 安全审查前对推荐文本做程序化预筛选。
通过预编译正则表达式快速扫描文本，确保即使 LLM 安全审查失败也有兜底保护。

设计：
- 5 条正则规则，预编译为 Pattern 对象
- scan() 返回命中列表，每条含 rule_id、category、match_position
- 扫描结果注入 LLM 安全审查 Prompt 上下文，提升审查精度
- 所有规则均为"高精度"匹配（避免误报）
"""

from __future__ import annotations

import re
from typing import Any


class SafetyPrefilter:
    """程序化安全预筛选器。

    在 LLM 安全审查之前先做正则扫描。
    5 条规则覆盖四大红线领域。
    """

    # ── 预编译正则规则 ──
    _RULES: list[dict[str, Any]] = [
        {
            "id": "PRE-001",
            "category": "禁止承诺已履约",
            "severity": "critical",
            "pattern": re.compile(
                r"已(为(您|你)|帮(您|你))?(预订|预约|下单|排号|留位|支付|扣款)"
            ),
            "description": "检测承诺已完成的履约动作",
        },
        {
            "id": "PRE-002",
            "category": "禁止编造库存",
            "severity": "critical",
            "pattern": re.compile(
                r"(仅剩|还剩|最后)\s*\d+\s*(间|桌|位|张|套|个)"
            ),
            "description": "检测编造的精确库存数字",
        },
        {
            "id": "PRE-003",
            "category": "禁止越权交易",
            "severity": "critical",
            "pattern": re.compile(
                r"(已从|将从).{0,10}(账户|银行卡|支付宝|微信|余额).{0,10}(扣|支付|转账)"
            ),
            "description": "检测越权交易/支付话术",
        },
        {
            "id": "PRE-004",
            "category": "禁止伪造商家承诺",
            "severity": "high",
            "pattern": re.compile(
                r"(商家|老板|经理|店长).{0,15}(承诺|保证|答应|说可以)"
            ),
            "description": "检测伪造商家承诺话术",
        },
        {
            "id": "PRE-005",
            "category": "禁止伪造商家承诺",
            "severity": "high",
            "pattern": re.compile(
                r"出示\s*(本|此|该)(消息|信息|优惠码|折扣码).{0,10}(即可|直接).{0,10}(享受|获得|抵扣|减免)"
            ),
            "description": "检测伪造凭证式承诺",
        },
    ]

    # ── 误报豁免（白名单短语，命中也不报）──
    _ALLOWLIST: list[re.Pattern] = [
        re.compile(r"建议.{0,20}(预订|预约)"),   # "建议提前预订" 不违规
        re.compile(r"(可以|可|请).{0,20}(预订|预约)"),  # "您可以在大众点评预订"
    ]

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    @classmethod
    def scan(cls, text: str) -> list[dict[str, Any]]:
        """扫描文本，返回所有命中的规则。

        Args:
            text: 待扫描的推荐文本。

        Returns:
            命中列表，每项: {rule_id, category, severity, description, match_text, position}
            若未命中则返回空 list。
        """
        if not text or not text.strip():
            return []

        hits: list[dict[str, Any]] = []

        for rule in cls._RULES:
            for match in rule["pattern"].finditer(text):
                matched_text: str = match.group()

                # 检查是否在白名单中
                if cls._is_allowlisted(text, match.start(), match.end()):
                    continue

                hits.append({
                    "rule_id": rule["id"],
                    "category": rule["category"],
                    "severity": rule["severity"],
                    "description": rule["description"],
                    "match_text": matched_text,
                    "position": match.start(),
                })

        # 按 severity 排序: critical 优先
        severity_order: dict[str, int] = {"critical": 0, "high": 1, "medium": 2}
        hits.sort(key=lambda h: severity_order.get(h["severity"], 9))

        return hits

    @classmethod
    def build_context_hint(cls, hits: list[dict[str, Any]]) -> str:
        """将预筛选命中结果转换为可注入 LLM Prompt 的文本上下文。

        Args:
            hits: scan() 返回的命中列表。

        Returns:
            适合嵌入安全审查 user_prompt 的提示文本。
        """
        if not hits:
            return ""

        lines: list[str] = [
            "[系统指令] 程序化预筛选在以下位置检测到潜在违规（请重点审查）：",
            "",
        ]
        for h in hits:
            lines.append(
                f"  - [{h['rule_id']}] {h['category']}: "
                f"匹配到 \"{h['match_text']}\" (位置 {h['position']})"
            )
        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @classmethod
    def _is_allowlisted(cls, text: str, start: int, end: int) -> bool:
        """检查匹配位置是否在白名单豁免范围内。

        扩展匹配窗口的前后 20 字符，检测是否被白名单覆盖。
        """
        window_start: int = max(0, start - 20)
        window_end: int = min(len(text), end + 20)
        window: str = text[window_start:window_end]

        for pattern in cls._ALLOWLIST:
            if pattern.search(window):
                return True
        return False
