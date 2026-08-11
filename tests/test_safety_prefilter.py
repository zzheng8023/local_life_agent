"""
Safety Prefilter 单元测试

测试 5 条安全审查正则规则与白名单豁免机制。
"""

from __future__ import annotations

import pytest

from infrastructure.safety_prefilter import SafetyPrefilter


class TestSafetyPrefilter:
    """安全预筛选规则测试。"""

    # ── PRE-001：禁止承诺已履约 ──

    def test_detect_booking_claim(self) -> None:
        """检测承诺已完成的履约动作。"""
        text: str = "已为您预订了靠窗座位"
        hits = SafetyPrefilter.scan(text)
        assert len(hits) >= 1
        assert any(h["rule_id"] == "PRE-001" for h in hits)

    def test_detect_queue_claim(self) -> None:
        """检测声称已排号的违规。"""
        text: str = "已帮您排号，前面还有3桌"
        hits = SafetyPrefilter.scan(text)
        assert len(hits) >= 1
        assert any(h["rule_id"] == "PRE-001" for h in hits)

    def test_detect_payment_claim(self) -> None:
        """检测声称已支付的违规。"""
        text: str = "已为您支付了定金"
        hits = SafetyPrefilter.scan(text)
        assert len(hits) >= 1
        assert any(h["rule_id"] == "PRE-001" for h in hits)

    def test_allowlist_booking_suggestion(self) -> None:
        """白名单：建议预订不应触发违规。"""
        text: str = "建议提前预订以确保有座位"
        hits = SafetyPrefilter.scan(text)
        # 应被白名单豁免
        booking_hits = [h for h in hits if h["rule_id"] == "PRE-001"]
        assert len(booking_hits) == 0

    def test_allowlist_can_book(self) -> None:
        """白名单：引导用户自行预订不应触发违规。"""
        text: str = "您可以在大众点评上直接预订"
        hits = SafetyPrefilter.scan(text)
        booking_hits = [h for h in hits if h["rule_id"] == "PRE-001"]
        assert len(booking_hits) == 0

    # ── PRE-002：禁止编造库存 ──

    def test_detect_stock_fabrication(self) -> None:
        """检测编造的精确库存数字。"""
        text: str = "仅剩2桌，预订从速"
        hits = SafetyPrefilter.scan(text)
        assert len(hits) >= 1
        assert any(h["rule_id"] == "PRE-002" for h in hits)

    def test_detect_last_room_claim(self) -> None:
        """检测‘最后X间’编造。"""
        text: str = "还剩3间空房"
        hits = SafetyPrefilter.scan(text)
        assert len(hits) >= 1
        assert any(h["rule_id"] == "PRE-002" for h in hits)

    # ── PRE-003：禁止越权交易 ──

    def test_detect_unauthorized_payment(self) -> None:
        """检测越权支付话术。"""
        text: str = "已从您的账户扣款200元"
        hits = SafetyPrefilter.scan(text)
        assert len(hits) >= 1
        assert any(h["rule_id"] == "PRE-003" for h in hits)

    def test_detect_alipay_payment(self) -> None:
        """检测支付宝扣款话术。"""
        text: str = "将从您的支付宝支付定金"
        hits = SafetyPrefilter.scan(text)
        assert len(hits) >= 1
        assert any(h["rule_id"] == "PRE-003" for h in hits)

    # ── PRE-004：禁止伪造商家承诺 ──

    def test_detect_merchant_promise(self) -> None:
        """检测伪造商家承诺话术。"""
        text: str = "老板说可以给您打8折"
        hits = SafetyPrefilter.scan(text)
        assert len(hits) >= 1
        assert any(h["rule_id"] == "PRE-004" for h in hits)

    def test_detect_manager_guarantee(self) -> None:
        """检测经理保证类的违规。"""
        text: str = "商家承诺给您免排队"
        hits = SafetyPrefilter.scan(text)
        assert len(hits) >= 1
        assert any(h["rule_id"] == "PRE-004" for h in hits)

    # ── PRE-005：禁止伪造凭证 ──

    def test_detect_fake_voucher(self) -> None:
        """检测伪造凭证式承诺。"""
        text: str = "出示此消息即可享受8折优惠"
        hits = SafetyPrefilter.scan(text)
        assert len(hits) >= 1
        assert any(h["rule_id"] == "PRE-005" for h in hits)

    def test_detect_fake_discount_code(self) -> None:
        """检测伪造折扣码。"""
        text: str = "出示该优惠码即可直接抵扣50元"
        hits = SafetyPrefilter.scan(text)
        assert len(hits) >= 1
        assert any(h["rule_id"] == "PRE-005" for h in hits)

    # ── 边界情况 ──

    def test_empty_text(self) -> None:
        """空文本不应产生命中。"""
        assert SafetyPrefilter.scan("") == []

    def test_clean_text(self) -> None:
        """正常推荐文本不应触发任何规则。"""
        text: str = (
            "为您推荐以下餐厅：\n"
            "1. 川味观·臻选 — ★4.5，人均140元\n"
            "建议提前致电确认是否有座位。"
        )
        hits = SafetyPrefilter.scan(text)
        assert len(hits) == 0

    def test_multiple_violations(self) -> None:
        """同时包含多个违规的文本应全部命中。"""
        text: str = "已为您预订了座位，仅剩3桌，老板说可以打8折"
        hits = SafetyPrefilter.scan(text)
        rule_ids = {h["rule_id"] for h in hits}
        assert len(rule_ids) >= 2

    # ── 上下文提示构建 ──

    def test_build_context_hint_empty(self) -> None:
        """空命中列表应返回空字符串。"""
        assert SafetyPrefilter.build_context_hint([]) == ""

    def test_build_context_hint_with_hits(self) -> None:
        """非空命中列表应包含规则ID和匹配文本。"""
        hits = [{
            "rule_id": "PRE-001",
            "category": "禁止承诺已履约",
            "severity": "critical",
            "description": "test",
            "match_text": "已预订",
            "position": 0,
        }]
        hint = SafetyPrefilter.build_context_hint(hits)
        assert "PRE-001" in hint
        assert "已预订" in hint
