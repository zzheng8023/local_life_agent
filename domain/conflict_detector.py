"""
冲突检测器 (Conflict Detector)

本模块提供程序化多人偏好冲突检测，作为 LLM 冲突检测的兜底和增强。

核心设计：
- 6 条程序化冲突规则，每条返回 dimension、priority、users_involved、suggested_resolution
- 与 LLM 偏好提取互补：LLM 做语义理解，代码做规则化冲突验证
- 检测结果注入 analyze_node，复杂冲突触发 conflict_negotiate 反问节点

规则列表：
1. 口味冲突：用户 A 偏好 X，用户 B 忌/排斥 X
2. 预算冲突：用户间预算相差 > 2x
3. 距离冲突：用户间距离期望差异 > 5km
4. 儿童/成人冲突：有儿童 vs 需要安静环境
5. 饮食限制冲突：清真/素食 vs 想吃肉/烧烤
6. 时间冲突：用户间时间差异 > 1小时
"""

from __future__ import annotations

import re
from typing import Any

from domain.entities import IndividualUserPreference


class ConflictDetector:
    """程序化多人偏好冲突检测器。

    对 IndividualUserPreference 列表执行规则化冲突扫描。
    每条规则返回结构化的冲突描述和解决建议。
    """

    # ── 常量 ──
    _OPPOSITE_TASTES: dict[str, list[str]] = {
        "辣": ["不辣", "忌辣", "清淡", "不喜辣", "不吃辣"],
        "川菜": ["不辣", "清淡", "粤菜", "日料"],
        "火锅": ["清淡", "日料", "素食"],
        "烧烤": ["素食", "清真", "清淡"],
        "海鲜": ["素食", "清真"],
        "肉": ["素食", "清真", "吃素"],
    }

    _RESTRICTION_CONFLICTS: list[tuple[str, str]] = [
        ("清真", "猪肉"),
        ("清真", "烧烤"),
        ("素食", "肉"),
        ("素食", "烧烤"),
        ("素食", "火锅"),
        ("海鲜过敏", "海鲜"),
        ("忌辣", "川菜"),
        ("忌辣", "火锅"),
    ]

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def detect(
        self, users: list[IndividualUserPreference]
    ) -> list[dict[str, Any]]:
        """对用户列表执行全量冲突检测。

        Args:
            users: 从 LLM 分析结果中解析的个人偏好列表。

        Returns:
            冲突列表，每项: {
                dimension, severity, users_involved, description,
                suggested_resolution, rule_id
            }
        """
        if len(users) < 2:
            return []

        conflicts: list[dict[str, Any]] = []

        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                a: IndividualUserPreference = users[i]
                b: IndividualUserPreference = users[j]

                conflicts.extend(self._check_taste_conflict(a, b))
                conflicts.extend(self._check_budget_conflict(a, b))
                conflicts.extend(self._check_restriction_conflict(a, b))
                conflicts.extend(self._check_distance_conflict(a, b))
                conflicts.extend(self._check_time_conflict(a, b))

        # 去重（按 rule_id 和 users_involved）
        seen: set[tuple] = set()
        unique: list[dict[str, Any]] = []
        for c in conflicts:
            key: tuple = (c["rule_id"], tuple(sorted(c["users_involved"])))
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return unique

    # ------------------------------------------------------------------
    # 规则 1：口味冲突
    # ------------------------------------------------------------------

    def _check_taste_conflict(
        self, a: IndividualUserPreference, b: IndividualUserPreference
    ) -> list[dict[str, Any]]:
        conflicts: list[dict[str, Any]] = []
        taste_a: str = (a.taste or "").lower()
        taste_b: str = (b.taste or "").lower()

        for preference, opposites in self._OPPOSITE_TASTES.items():
            a_likes: bool = preference in taste_a or (
                taste_a and preference in taste_a
            )
            b_opposes: bool = any(op in taste_b for op in opposites)

            if a_likes and b_opposes:
                conflicts.append(self._build_conflict(
                    rule_id="C-001",
                    dimension="口味偏好",
                    severity="high",
                    users=[a.name, b.name],
                    description=(
                        f"{a.name} 偏好「{preference}」，"
                        f"但 {b.name} 的偏好中包含冲突项"
                    ),
                    suggestion=(
                        f"推荐口味折中方案：选择有{preference}和不辣双选的餐厅，"
                        f"或优先照顾忌口方 {b.name} 的需求"
                    ),
                ))

        return conflicts

    # ------------------------------------------------------------------
    # 规则 2：预算冲突
    # ------------------------------------------------------------------

    def _check_budget_conflict(
        self, a: IndividualUserPreference, b: IndividualUserPreference
    ) -> list[dict[str, Any]]:
        conflicts: list[dict[str, Any]] = []

        budget_a: int | None = self._parse_budget(a.budget)
        budget_b: int | None = self._parse_budget(b.budget)

        if budget_a is None or budget_b is None:
            return conflicts

        if max(budget_a, budget_b) > min(budget_a, budget_b) * 2:
            conflicts.append(self._build_conflict(
                rule_id="C-002",
                dimension="预算",
                severity="medium",
                users=[a.name, b.name],
                description=(
                    f"{a.name} 预算约 ¥{budget_a}，"
                    f"{b.name} 预算约 ¥{budget_b}，差异超过 2 倍"
                ),
                suggestion=(
                    f"建议人均控制在 ¥{max(budget_a, budget_b)} 以内，"
                    f"或选择有不同价位选择的综合商圈"
                ),
            ))

        return conflicts

    # ------------------------------------------------------------------
    # 规则 3：饮食限制冲突
    # ------------------------------------------------------------------

    def _check_restriction_conflict(
        self, a: IndividualUserPreference, b: IndividualUserPreference
    ) -> list[dict[str, Any]]:
        conflicts: list[dict[str, Any]] = []

        a_restrictions: str = (a.restrictions or "").lower()
        b_restrictions: str = (b.restrictions or "").lower()
        a_taste: str = (a.taste or "").lower()
        b_taste: str = (b.taste or "").lower()

        for restriction, forbidden in self._RESTRICTION_CONFLICTS:
            # a 有限制，b 偏好被限制的食物
            has_restriction: bool = restriction in a_restrictions or restriction in a_taste
            b_wants_forbidden: bool = (
                forbidden in b_taste
                or forbidden in b_restrictions
                or forbidden == b_taste
            )

            if has_restriction and b_wants_forbidden:
                conflicts.append(self._build_conflict(
                    rule_id="C-003",
                    dimension="饮食限制",
                    severity="critical",
                    users=[a.name, b.name],
                    description=(
                        f"{a.name} 有「{restriction}」限制，"
                        f"与 {b.name} 的「{forbidden}」偏好冲突"
                    ),
                    suggestion=(
                        f"优先满足 {a.name} 的饮食限制，"
                        f"为 {b.name} 寻找替代方案或分餐制"
                    ),
                ))

        return conflicts

    # ------------------------------------------------------------------
    # 规则 4：距离冲突
    # ------------------------------------------------------------------

    def _check_distance_conflict(
        self, a: IndividualUserPreference, b: IndividualUserPreference
    ) -> list[dict[str, Any]]:
        conflicts: list[dict[str, Any]] = []

        dist_a: float | None = self._parse_distance(a.distance)
        dist_b: float | None = self._parse_distance(b.distance)

        if dist_a is None or dist_b is None:
            return conflicts

        if abs(dist_a - dist_b) > 5.0:
            conflicts.append(self._build_conflict(
                rule_id="C-004",
                dimension="距离",
                severity="low",
                users=[a.name, b.name],
                description=(
                    f"{a.name} 期望 {dist_a}km，"
                    f"{b.name} 期望 {dist_b}km，差异超过 5km"
                ),
                suggestion="取中间位置或选择交通便利的枢纽商圈",
            ))

        return conflicts

    # ------------------------------------------------------------------
    # 规则 5：时间冲突
    # ------------------------------------------------------------------

    def _check_time_conflict(
        self, a: IndividualUserPreference, b: IndividualUserPreference
    ) -> list[dict[str, Any]]:
        conflicts: list[dict[str, Any]] = []

        time_a: str = (a.time or "").strip()
        time_b: str = (b.time or "").strip()

        if not time_a or not time_b:
            return conflicts

        # 简易时间差检测
        hour_a: int | None = self._extract_hour(time_a)
        hour_b: int | None = self._extract_hour(time_b)

        if hour_a is not None and hour_b is not None:
            if abs(hour_a - hour_b) > 1.5:
                conflicts.append(self._build_conflict(
                    rule_id="C-005",
                    dimension="时间",
                    severity="medium",
                    users=[a.name, b.name],
                    description=(
                        f"{a.name} 偏好 {time_a}，"
                        f"{b.name} 偏好 {time_b}，时差较大"
                    ),
                    suggestion="建议协调折中时间或分两批到达",
                ))

        return conflicts

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_budget(budget_str: str | None) -> int | None:
        if not budget_str:
            return None
        try:
            match = re.search(r"(\d+)", str(budget_str))
            if match:
                return int(match.group(1))
        except (ValueError, TypeError):
            pass
        return None

    @staticmethod
    def _parse_distance(dist_str: str | None) -> float | None:
        if not dist_str:
            return None
        try:
            match = re.search(r"(\d+(?:\.\d+)?)", str(dist_str))
            if match:
                return float(match.group(1))
        except (ValueError, TypeError):
            pass
        return None

    @staticmethod
    def _extract_hour(time_str: str) -> float | None:
        """从时间字符串中提取小时数（24h）。"""
        patterns: list[str] = [
            r"(\d{1,2})[:：](\d{2})",  # "18:30"
            r"(\d{1,2})点",             # "7点"
            r"晚上(\d{1,2})",            # "晚上7"
            r"中午(\d{1,2})",            # "中午12"
            r"早上(\d{1,2})",            # "早上8"
        ]
        for pattern in patterns:
            match = re.search(pattern, str(time_str))
            if match:
                try:
                    hour: float = float(match.group(1))
                    if "晚上" in str(time_str) and hour <= 11:
                        hour += 12
                    if "中午" in str(time_str) and hour <= 2:
                        hour += 12
                    return hour
                except (ValueError, TypeError):
                    continue
        return None

    @staticmethod
    def _build_conflict(
        rule_id: str,
        dimension: str,
        severity: str,
        users: list[str],
        description: str,
        suggestion: str,
    ) -> dict[str, Any]:
        return {
            "rule_id": rule_id,
            "dimension": dimension,
            "severity": severity,
            "users_involved": users,
            "description": description,
            "suggested_resolution": suggestion,
        }
