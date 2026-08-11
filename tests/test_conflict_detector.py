"""
Conflict Detector 单元测试

测试 5 条程序化多人偏好冲突检测规则。
"""

from __future__ import annotations

import pytest

from domain.conflict_detector import ConflictDetector
from domain.entities import IndividualUserPreference


def _make_user(
    name: str = "A",
    taste: str = "",
    restrictions: str = "",
    budget: str = "",
    distance: str = "",
    time: str = "",
) -> IndividualUserPreference:
    """创建测试用 IndividualUserPreference。"""
    return IndividualUserPreference(
        name=name,
        taste=taste,
        restrictions=restrictions,
        budget=budget,
        distance=distance,
        time=time,
    )


class TestConflictDetector:
    """冲突检测器测试。"""

    def setup_method(self) -> None:
        self._detector = ConflictDetector()

    # ── C-001: 口味冲突 ──

    def test_taste_conflict_spicy_vs_light(self) -> None:
        """A 喜辣 B 忌辣。"""
        users = [
            _make_user(name="A", taste="川菜，喜欢辣"),
            _make_user(name="B", taste="不辣，清淡"),
        ]
        conflicts = self._detector.detect(users)
        taste_conflicts = [c for c in conflicts if c["rule_id"] == "C-001"]
        assert len(taste_conflicts) >= 1
        assert taste_conflicts[0]["severity"] == "high"

    def test_taste_conflict_hotpot_vs_light(self) -> None:
        """A 要火锅 B 要清淡。"""
        users = [
            _make_user(name="A", taste="火锅"),
            _make_user(name="B", taste="清淡粤菜"),
        ]
        conflicts = self._detector.detect(users)
        assert any(c["rule_id"] == "C-001" for c in conflicts)

    def test_no_taste_conflict_same_preference(self) -> None:
        """相同偏好不触发冲突。"""
        users = [
            _make_user(name="A", taste="川菜"),
            _make_user(name="B", taste="川菜，喜欢辣"),
        ]
        conflicts = self._detector.detect(users)
        taste_conflicts = [c for c in conflicts if c["rule_id"] == "C-001"]
        assert len(taste_conflicts) == 0

    # ── C-002: 预算冲突 ──

    def test_budget_conflict_2x_difference(self) -> None:
        """预算差异超过 2 倍触发冲突。"""
        users = [
            _make_user(name="A", budget="人均100元"),
            _make_user(name="B", budget="人均300元"),
        ]
        conflicts = self._detector.detect(users)
        budget_conflicts = [c for c in conflicts if c["rule_id"] == "C-002"]
        assert len(budget_conflicts) >= 1

    def test_no_budget_conflict_similar(self) -> None:
        """预算相近不触发冲突。"""
        users = [
            _make_user(name="A", budget="人均150元"),
            _make_user(name="B", budget="人均200元"),
        ]
        conflicts = self._detector.detect(users)
        budget_conflicts = [c for c in conflicts if c["rule_id"] == "C-002"]
        assert len(budget_conflicts) == 0

    def test_no_budget_when_missing(self) -> None:
        """无预算信息时不触发。"""
        users = [
            _make_user(name="A"),
            _make_user(name="B"),
        ]
        conflicts = self._detector.detect(users)
        budget_conflicts = [c for c in conflicts if c["rule_id"] == "C-002"]
        assert len(budget_conflicts) == 0

    # ── C-003: 饮食限制冲突 ──

    def test_restriction_halal_vs_bbq(self) -> None:
        """清真 vs 烧烤。"""
        users = [
            _make_user(name="A", restrictions="清真"),
            _make_user(name="B", taste="烧烤"),
        ]
        conflicts = self._detector.detect(users)
        restriction_conflicts = [c for c in conflicts if c["rule_id"] == "C-003"]
        assert len(restriction_conflicts) >= 1
        assert restriction_conflicts[0]["severity"] == "critical"

    def test_restriction_vegetarian_vs_meat(self) -> None:
        """素食 vs 肉食。"""
        users = [
            _make_user(name="A", taste="素食"),
            _make_user(name="B", taste="肉"),
        ]
        conflicts = self._detector.detect(users)
        assert any(c["rule_id"] == "C-003" for c in conflicts)

    def test_restriction_seafood_allergy(self) -> None:
        """海鲜过敏 vs 海鲜。"""
        users = [
            _make_user(name="A", restrictions="海鲜过敏"),
            _make_user(name="B", taste="海鲜"),
        ]
        conflicts = self._detector.detect(users)
        restriction_conflicts = [c for c in conflicts if c["rule_id"] == "C-003"]
        assert len(restriction_conflicts) >= 1

    # ── C-004: 距离冲突 ──

    def test_distance_conflict_large_diff(self) -> None:
        """距离期望差异 > 5km。"""
        users = [
            _make_user(name="A", distance="3公里以内"),
            _make_user(name="B", distance="10公里以内"),
        ]
        conflicts = self._detector.detect(users)
        distance_conflicts = [c for c in conflicts if c["rule_id"] == "C-004"]
        assert len(distance_conflicts) >= 1
        assert distance_conflicts[0]["severity"] == "low"

    def test_no_distance_conflict_small_diff(self) -> None:
        """距离差异 ≤ 5km 不触发。"""
        users = [
            _make_user(name="A", distance="3公里以内"),
            _make_user(name="B", distance="5公里以内"),
        ]
        conflicts = self._detector.detect(users)
        distance_conflicts = [c for c in conflicts if c["rule_id"] == "C-004"]
        assert len(distance_conflicts) == 0

    # ── C-005: 时间冲突 ──

    def test_time_conflict_large_gap(self) -> None:
        """时间差 > 1.5 小时。"""
        users = [
            _make_user(name="A", time="18:00"),
            _make_user(name="B", time="20:00"),
        ]
        conflicts = self._detector.detect(users)
        time_conflicts = [c for c in conflicts if c["rule_id"] == "C-005"]
        assert len(time_conflicts) >= 1

    def test_no_time_conflict_small_gap(self) -> None:
        """时间差 ≤ 1.5 小时不触发。"""
        users = [
            _make_user(name="A", time="19:00"),
            _make_user(name="B", time="19:30"),
        ]
        conflicts = self._detector.detect(users)
        time_conflicts = [c for c in conflicts if c["rule_id"] == "C-005"]
        assert len(time_conflicts) == 0

    # ── 边界情况 ──

    def test_single_user_no_conflict(self) -> None:
        """单用户场景不产生冲突。"""
        users = [_make_user(name="A", taste="川菜")]
        conflicts = self._detector.detect(users)
        assert len(conflicts) == 0

    def test_empty_list(self) -> None:
        """空列表不产生冲突。"""
        conflicts = self._detector.detect([])
        assert len(conflicts) == 0

    def test_deduplication(self) -> None:
        """完全相同的用户间冲突去重。"""
        users = [
            _make_user(name="A", restrictions="清真"),
            _make_user(name="B", taste="烧烤"),
            _make_user(name="C", taste="猪肉"),
        ]
        conflicts = self._detector.detect(users)
        rule_ids = [c["rule_id"] for c in conflicts]
        # A vs B: C-003 (清真 vs 烧烤), A vs C: C-003 (清真 vs 猪肉)
        assert rule_ids.count("C-003") == 2
