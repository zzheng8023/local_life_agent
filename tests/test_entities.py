"""
领域实体 (Entities) 单元测试

测试 AgentState、UserPreference 等 Pydantic 模型的
创建、默认值、序列化/反序列化。
"""

from __future__ import annotations

import pytest

from domain.entities import (
    AgentState,
    UserPreference,
    IndividualUserPreference,
    SafetyDecision,
    Restaurant,
    Hotel,
    Entertainment,
    TransitStop,
)


class TestUserPreference:
    """UserPreference 模型测试。"""

    def test_default_values(self) -> None:
        """默认构造所有可选字段为 None。"""
        pref = UserPreference()
        assert pref.budget is None
        assert pref.taste is None
        assert pref.restrictions is None
        assert pref.has_kids is False
        assert pref.need_parking is False
        assert pref.hotel_req is None
        assert pref.entertainment_req is None
        assert pref.origin_points == []

    def test_with_partial_preferences(self) -> None:
        """部分偏好设置测试。"""
        pref = UserPreference(
            budget="人均150元",
            taste="川菜",
            city="北京",
        )
        assert pref.budget == "人均150元"
        assert pref.taste == "川菜"
        assert pref.city == "北京"
        assert pref.restrictions is None

    def test_feature_fields(self) -> None:
        """v2.1+ 新增字段测试。"""
        pref = UserPreference(
            freeform_location="朝阳大悦城",
            need_parking_detail="附近有停车场吗",
            bike_req="共享单车",
        )
        assert pref.freeform_location == "朝阳大悦城"
        assert pref.need_parking_detail == "附近有停车场吗"
        assert pref.bike_req == "共享单车"


class TestIndividualUserPreference:
    """IndividualUserPreference 模型测试。"""

    def test_create_with_all_fields(self) -> None:
        """完整字段创建。"""
        ip = IndividualUserPreference(
            name="老张",
            budget="人均150元",
            taste="川菜",
            restrictions="不吃香菜",
            distance="朝阳区",
            time="今晚7点",
            has_kids=True,
            need_parking=True,
            key_utterance="我想吃辣的",
            origin_point="海淀区中关村",
        )
        assert ip.name == "老张"
        assert ip.has_kids is True
        assert ip.need_parking is True
        assert ip.origin_point == "海淀区中关村"


class TestAgentState:
    """AgentState 聚合根测试。"""

    def test_default_state(self) -> None:
        """默认初始状态。"""
        state = AgentState()
        assert state.raw_query == ""
        assert isinstance(state.user_preference, UserPreference)
        assert state.candidate_restaurants == []
        assert state.candidate_hotels == []
        assert state.safety_passed is True
        assert state.is_chitchat is False
        assert state.is_out_of_domain is False
        assert state.trace_logs == []

    def test_with_query(self) -> None:
        """带查询创建。"""
        state = AgentState(raw_query="周六晚上聚餐")
        assert state.raw_query == "周六晚上聚餐"

    def test_candidate_fields(self) -> None:
        """多领域候选集字段。"""
        state = AgentState(
            candidate_restaurants=[{"name": "测试餐厅", "rating": 4.5}],
            candidate_hotels=[{"name": "测试酒店"}],
            candidate_parking=[{"name": "地下停车场"}],
        )
        assert len(state.candidate_restaurants) == 1
        assert len(state.candidate_hotels) == 1
        assert len(state.candidate_parking) == 1

    def test_individual_preferences_storage(self) -> None:
        """个人偏好列表存储。"""
        ip = IndividualUserPreference(name="A", taste="川菜")
        state = AgentState(individual_preferences=[ip])
        assert len(state.individual_preferences) == 1
        assert state.individual_preferences[0].name == "A"

    def test_detected_conflicts_storage(self) -> None:
        """冲突检测结果存储。"""
        state = AgentState(
            detected_conflicts=[
                {"rule_id": "C-001", "severity": "high"},
            ]
        )
        assert len(state.detected_conflicts) == 1
        assert state.detected_conflicts[0]["rule_id"] == "C-001"


class TestRestaurant:
    """Restaurant 实体测试。"""

    def test_minimal_creation(self) -> None:
        """最小字段创建。"""
        r = Restaurant(name="测试餐厅")
        assert r.name == "测试餐厅"
        assert r.rating == 0.0
        assert r.features == []

    def test_full_creation(self) -> None:
        """完整字段创建。"""
        r = Restaurant(
            name="川味观",
            rating=4.5,
            avg_price="人均120元",
            cuisine="川菜",
            address="朝阳区建国路88号",
            distance_km=2.3,
            features=["有包间", "免费停车"],
        )
        assert r.name == "川味观"
        assert r.rating == 4.5
        assert r.cuisine == "川菜"
        assert "有包间" in r.features

    def test_rating_bounds(self) -> None:
        """评分边界检验。"""
        with pytest.raises(Exception):
            Restaurant(name="x", rating=6.0)

        with pytest.raises(Exception):
            Restaurant(name="x", rating=-1.0)


class TestHotel:
    """Hotel 实体测试。"""

    def test_creation(self) -> None:
        h = Hotel(
            name="全季酒店",
            rating=4.2,
            avg_price="298元起",
            features=["含早", "免费停车"],
        )
        assert h.name == "全季酒店"
        assert h.rating == 4.2
        assert "含早" in h.features


class TestEntertainment:
    """Entertainment 实体测试。"""

    def test_creation(self) -> None:
        e = Entertainment(
            name="万达影城",
            category="电影院",
            rating=4.3,
        )
        assert e.name == "万达影城"
        assert e.category == "电影院"


class TestTransitStop:
    """TransitStop 实体测试。"""

    def test_creation(self) -> None:
        ts = TransitStop(
            name="国贸地铁站",
            category="地铁站",
            lines=["1号线", "10号线"],
        )
        assert ts.name == "国贸地铁站"
        assert ts.category == "地铁站"
        assert "1号线" in ts.lines


class TestSafetyDecision:
    """SafetyDecision 值对象测试。"""

    def test_passed(self) -> None:
        sd = SafetyDecision(passed=True, original_text="合规文本")
        assert sd.passed is True
        assert sd.rewritten_text is None

    def test_failed_with_rewrite(self) -> None:
        sd = SafetyDecision(
            passed=False,
            original_text="已预订",
            rewritten_text="建议您自行预订",
            violations=["禁止承诺已履约"],
        )
        assert sd.passed is False
        assert sd.rewritten_text == "建议您自行预订"
        assert "禁止承诺已履约" in sd.violations
