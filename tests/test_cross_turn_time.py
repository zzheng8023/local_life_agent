#!/usr/bin/env python3
"""
End-to-end verification tests for weather cross-turn context fix.

Validates the four Agent Principles:
  - LLM 负责"指代理解"  (TimeExpression extraction only, no date math)
  - Context 负责"事实记忆" (last_resolved_time persisted via MemorySaver)
  - 代码负责"时间计算"    (TimeResolver.resolve() deterministic arithmetic)
  - API 负责"真实数据"    (WeatherTool is sole weather data source)

Uses a mock LLM client so no real API keys are needed.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OPENAI_API_KEY", "sk-test-mock-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-mock-key")

from typing import Any

from application.ports import ILLMClient
from application.workflow import LocalLifeWorkflow


# ================================================================
# Mock LLM Client — simulates LLM responses for testing
# ================================================================

class MockLLMClient(ILLMClient):
    """Mock LLM that returns sensible responses for each phase of the workflow.

    Routing is determined by system_prompt content:
    - "时间语义提取器" → time extraction
    - "安全审查网关" → safety guard
    - "智能决策中枢" → analysis
    """

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if "推荐" in system_prompt or "推荐" in user_prompt:
            return (
                "根据您的偏好，为您推荐以下方案：\n\n"
                "🍽️ **蜀九香火锅** — 人均120元，评分4.5，川味正宗\n"
                "📍 朝阳区建国路88号 | 🅿️ 有停车位\n"
                "祝您用餐愉快！"
            )
        return "好的，我来帮您分析一下。"

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        sp = system_prompt or ""
        up = user_prompt or ""

        # Time extraction — system prompt: "你是一个时间语义提取器..."
        if "时间语义提取器" in sp:
            return self._mock_time_extraction(up)

        # Safety guard — system prompt: "...安全审查网关..."
        if "安全审查网关" in sp:
            return {"passed": True, "violations": [], "output": ""}

        # Analysis — system prompt: "...智能决策中枢..."
        if "智能决策中枢" in sp:
            return self._mock_analysis(up)

        # Fallback
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Time extraction mocks
    # ------------------------------------------------------------------
    def _mock_time_extraction(self, prompt: str) -> dict[str, Any]:
        # Only inspect the "用户输入:" line to avoid matching instruction text
        # The prompt template is:
        #   从用户输入中提取时间语义参数...
        #   当前日期（参考用）: ...
        #   用户输入: <actual user query>
        #   输出 JSON 格式:...
        user_input: str = ""
        for line in (prompt or "").split("\n"):
            if line.startswith("用户输入:"):
                user_input = line[len("用户输入:"):].strip()
                break
        if not user_input:
            user_input = prompt or ""

        # Reference keywords — check user input only, NOT instructions
        if any(kw in user_input for kw in ["当天", "那天", "这一天", "那天晚上"]):
            return {"raw": "当天", "type": "reference"}

        # This Saturday / this weekend Saturday
        if "这周六" in user_input or "本周六" in user_input:
            return {"raw": "这周六", "type": "relative_weekday", "weekday": 6, "week_offset": 0}

        # Next Wednesday
        if "下周三" in user_input:
            return {"raw": "下周三", "type": "relative_weekday", "weekday": 3, "week_offset": 1}

        # Tomorrow
        if "明天" in user_input:
            return {"raw": "明天", "type": "tomorrow"}

        # Today
        if "今天" in user_input:
            return {"raw": "今天", "type": "today"}

        # Day after tomorrow
        if "后天" in user_input:
            return {"raw": "后天", "type": "day_after_tomorrow"}

        # Default: today
        return {"raw": "", "type": "today"}

    # ------------------------------------------------------------------
    # Analysis mocks
    # ------------------------------------------------------------------
    def _mock_analysis(self, prompt: str) -> dict[str, Any]:
        p = prompt or ""

        # Out-of-domain: coding / scripting
        if any(kw in p for kw in ["写个Python", "Python脚本", "帮我写"]):
            return {
                "is_out_of_domain": True,
                "is_chitchat": False,
                "out_of_domain_reply": (
                    "抱歉，我是本地生活助手，无法帮您编写代码。"
                    "请问有什么本地生活方面的需求我可以帮您吗？"
                ),
                "aggregated_preference": {},
                "needs_clarification": False,
                "clarification_question": "",
                "users": [],
            }

        # Chitchat
        if any(kw in p for kw in ["你好", "谢谢", "再见"]):
            return {
                "is_out_of_domain": False,
                "is_chitchat": True,
                "chitchat_reply": "你好！请问有什么可以帮您的？",
                "aggregated_preference": {},
                "needs_clarification": False,
                "clarification_question": "",
                "users": [],
            }

        # Sichuan cuisine query
        if "川菜" in p:
            return {
                "is_out_of_domain": False,
                "is_chitchat": False,
                "aggregated_preference": {
                    "taste": "川菜",
                    "time": "这周六",
                    "budget": "人均100元左右",
                    "city": "北京",
                },
                "needs_clarification": False,
                "clarification_question": "",
                "users": [],
            }

        # Hotpot query — with time
        if "火锅" in p:
            time_val: Optional[str] = None
            if "下周三" in p:
                time_val = "下周三"
            pref: dict[str, Any] = {
                "taste": "火锅",
                "budget": "人均100元左右",
                "city": "北京",
            }
            if time_val:
                pref["time"] = time_val
            return {
                "is_out_of_domain": False,
                "is_chitchat": False,
                "aggregated_preference": pref,
                "needs_clarification": False,
                "clarification_question": "",
                "users": [],
            }

        # Default restaurant query
        return {
            "is_out_of_domain": False,
            "is_chitchat": False,
            "aggregated_preference": {"city": "北京"},
            "needs_clarification": False,
            "clarification_question": "",
            "users": [],
        }


# ================================================================
# Color helpers
# ================================================================

def green(s: str) -> str:
    return f"\033[92m{s}\033[0m"

def red(s: str) -> str:
    return f"\033[91m{s}\033[0m"

def bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


# ================================================================
# Test runner
# ================================================================

def run_tests() -> bool:
    mock_llm = MockLLMClient()
    workflow = LocalLifeWorkflow(llm=mock_llm, tools={}, fast_llm=mock_llm)

    failures: list[str] = []

    # ============================================================
    # Test 1: Cross-turn time reference
    # "这周六我要吃川菜" → "当天天气怎么样"
    # Expected: resolved_date = 2026-08-01 (Saturday), NOT 2026-07-28 (today)
    # ============================================================
    print(bold("\n=== Test 1: Cross-Turn Time Reference ==="))
    print("Turn 1: '这周六我要吃川菜'")

    thread_id: str = "test-cross-turn-001"
    state1 = workflow.chat("这周六我要吃川菜", thread_id=thread_id)

    lrt = getattr(state1, "last_resolved_time", None)
    print(f"  taste: {getattr(state1.user_preference, 'taste', 'N/A')}")
    print(f"  is_weather_query: {getattr(state1, 'is_weather_query', 'N/A')}")
    print(f"  is_out_of_domain: {getattr(state1, 'is_out_of_domain', 'N/A')}")
    print(f"  last_resolved_time: {lrt}")

    if lrt:
        resolved_date: str = lrt.get("resolved_date", "")
        print(f"  → Saved context: raw='{lrt.get('raw')}', date={resolved_date}")
        if resolved_date == "2026-08-01":
            print(green("  ✓ Turn 1 saved correct date: 2026-08-01 (Saturday)"))
        elif resolved_date == "2026-07-28":
            msg = f"Turn 1 saved TODAY instead of Saturday: {resolved_date}"
            print(red(f"  ✗ {msg}"))
            failures.append(msg)
        else:
            msg = f"Turn 1 saved unexpected date: {resolved_date}"
            print(red(f"  ✗ {msg}"))
            failures.append(msg)
    else:
        msg = "Turn 1 did NOT save last_resolved_time (required for cross-turn reference)"
        print(red(f"  ✗ {msg}"))
        failures.append(msg)
        # Can't continue with cross-turn test if turn 1 didn't save context
        print(red("  → SKIPPING Turn 2 test (no context to reference)"))
        # But still run other tests

    # Only run turn 2 if turn 1 saved context
    if lrt:
        print("\nTurn 2: '当天天气怎么样'")
        state2 = workflow.chat("当天天气怎么样", thread_id=thread_id)

        final_rec: str = getattr(state2, "final_recommendation", "")
        weather_result = getattr(state2, "weather_result", None)
        lrt2 = getattr(state2, "last_resolved_time", None)

        print(f"  is_weather_query: {getattr(state2, 'is_weather_query', 'N/A')}")
        print(f"  is_out_of_domain: {getattr(state2, 'is_out_of_domain', 'N/A')}")
        print(f"  weather_result: {weather_result}")
        print(f"  last_resolved_time: {lrt2}")
        print(f"  final_recommendation: {final_rec[:300]}")

        has_aug1 = "8月1日" in final_rec
        has_today_wrong = "7月28日" in final_rec

        if has_aug1 and not has_today_wrong:
            print(green("  ✓ PASS: '当天' references Saturday (Aug 1) — cross-turn context works!"))
        elif has_today_wrong:
            msg = "'当天' resolved to TODAY (Jul 28) instead of Saturday (Aug 1)"
            print(red(f"  ✗ FAIL: {msg}"))
            failures.append(msg)
        else:
            msg = f"Neither Aug 1 nor Jul 28 found in: {final_rec[:200]}"
            print(red(f"  ✗ FAIL: {msg}"))
            failures.append(msg)

        # Verify WeatherTool was called with the right date
        if weather_result:
            wdate: str = weather_result.get("date", "")
            if wdate == "2026-08-01":
                print(green(f"  ✓ WeatherTool called with correct date: {wdate}"))
            else:
                msg = f"WeatherTool called with wrong date: {wdate}, expected 2026-08-01"
                print(red(f"  ✗ {msg}"))
                failures.append(msg)

        # Verify turn 2 saved resolved time
        if lrt2 and lrt2.get("resolved_date") == "2026-08-01":
            print(green(f"  ✓ Turn 2 saved resolved time: {lrt2['resolved_date']}"))
        else:
            msg = f"Turn 2 didn't save correct resolved time: {lrt2}"
            print(red(f"  ✗ {msg}"))
            failures.append(msg)

    # ============================================================
    # Test 2: Standalone weather query still works
    # ============================================================
    print(bold("\n=== Test 2: Standalone Weather Query ==="))
    state3 = workflow.chat("本周六天气如何", thread_id="test-standalone-002")

    final_rec3: str = getattr(state3, "final_recommendation", "")
    is_ood3: bool = getattr(state3, "is_out_of_domain", False)
    is_wq3: bool = getattr(state3, "is_weather_query", False)
    print(f"  is_weather_query: {is_wq3}")
    print(f"  is_out_of_domain: {is_ood3}")
    print(f"  final_recommendation: {final_rec3[:200]}")

    if "8月1日" in final_rec3:
        print(green("  ✓ '本周六' resolves to Aug 1"))
    else:
        msg = f"Standalone weather query did not resolve to Aug 1: {final_rec3[:150]}"
        print(red(f"  ✗ {msg}"))
        failures.append(msg)

    if not is_ood3:
        print(green("  ✓ Weather NOT classified as out-of-domain"))
    else:
        msg = "Weather query incorrectly classified as out-of-domain"
        print(red(f"  ✗ {msg}"))
        failures.append(msg)

    if is_wq3:
        print(green("  ✓ Weather query flagged as is_weather_query=True"))
    else:
        msg = "Weather query NOT flagged as is_weather_query"
        print(red(f"  ✗ {msg}"))
        failures.append(msg)

    # ============================================================
    # Test 3: Normal restaurant query still works
    # ============================================================
    print(bold("\n=== Test 3: Normal Restaurant Query ==="))
    state4 = workflow.chat("推荐一家火锅店", thread_id="test-restaurant-003")

    is_ood4: bool = getattr(state4, "is_out_of_domain", False)
    is_wq4: bool = getattr(state4, "is_weather_query", False)
    rec4: str = getattr(state4, "final_recommendation", "")
    print(f"  is_out_of_domain: {is_ood4}")
    print(f"  is_weather_query: {is_wq4}")
    print(f"  final_recommendation: {rec4[:200]}")

    if not is_ood4:
        print(green("  ✓ Restaurant query NOT out-of-domain"))
    else:
        msg = "Restaurant query incorrectly classified as out-of-domain"
        print(red(f"  ✗ {msg}"))
        failures.append(msg)

    if not is_wq4:
        print(green("  ✓ Restaurant query NOT flagged as weather"))
    else:
        msg = "Restaurant query incorrectly flagged as weather"
        print(red(f"  ✗ {msg}"))
        failures.append(msg)

    # ============================================================
    # Test 4: Out-of-domain routing still works
    # ============================================================
    print(bold("\n=== Test 4: Out-of-Domain Routing ==="))
    state5 = workflow.chat("帮我写个Python脚本", thread_id="test-ood-004")

    is_ood5: bool = getattr(state5, "is_out_of_domain", False)
    rec5: str = getattr(state5, "final_recommendation", "")
    print(f"  is_out_of_domain: {is_ood5}")
    print(f"  final_recommendation: {rec5[:200]}")

    if is_ood5:
        print(green("  ✓ Out-of-domain correctly routed"))
    else:
        msg = "Out-of-domain query NOT routed to out_of_domain"
        print(red(f"  ✗ {msg}"))
        failures.append(msg)

    # Ensure no weather leak in out-of-domain response
    weather_leak_keywords: list[str] = ["天气", "下雨", "气温", "温度", "湿度"]
    leaked: list[str] = [kw for kw in weather_leak_keywords if kw in rec5]
    if not leaked:
        print(green("  ✓ Out-of-domain response has NO weather leak (LLM not overstepping)"))
    else:
        msg = f"Out-of-domain response leaks weather keywords: {leaked}"
        print(red(f"  ✗ {msg}"))
        failures.append(msg)

    # ============================================================
    # Test 5: "那天" cross-turn reference (next Wednesday → rain)
    # ============================================================
    print(bold("\n=== Test 5: '那天' Cross-Turn Reference ==="))
    thread_id2: str = "test-cross-turn-005"
    state6 = workflow.chat("下周三吃火锅", thread_id=thread_id2)
    lrt6 = getattr(state6, "last_resolved_time", None)
    print(f"  Turn 1 last_resolved_time: {lrt6}")

    if lrt6:
        expected_date: str = "2026-08-05"  # Next Wednesday from Jul 28 (Tue)
        actual_date: str = lrt6.get("resolved_date", "")
        if actual_date == expected_date:
            print(green(f"  ✓ '下周三' resolved to {expected_date}"))
        else:
            msg = f"'下周三' should be {expected_date} but got {actual_date}"
            print(red(f"  ✗ {msg}"))
            failures.append(msg)

        state7 = workflow.chat("那天会下雨吗", thread_id=thread_id2)
        rec7: str = getattr(state7, "final_recommendation", "")
        weather7 = getattr(state7, "weather_result", None)
        print(f"  Turn 2 ('那天会下雨吗'): {rec7[:300]}")

        if weather7:
            wdate7: str = weather7.get("date", "")
            if wdate7 == expected_date:
                print(green(f"  ✓ '那天' references next Wednesday ({expected_date}): {wdate7}"))
            else:
                msg = f"Expected {expected_date} but WeatherTool date was {wdate7}"
                print(red(f"  ✗ {msg}"))
                failures.append(msg)
        elif "8月5日" in rec7:
            print(green(f"  ✓ '那天' references next Wednesday ({expected_date}) in response"))
        else:
            msg = f"Could not verify '那天' reference resolution in: {rec7[:200]}"
            print(red(f"  ✗ {msg}"))
            failures.append(msg)
    else:
        msg = "Turn 1 ('下周三吃火锅') did NOT save last_resolved_time"
        print(red(f"  ✗ {msg}"))
        failures.append(msg)

    # ============================================================
    # Test 6: "今天天气怎么样" (fresh thread) — should work standalone
    # ============================================================
    print(bold("\n=== Test 6: Standalone '今天天气怎么样' (Fresh Thread) ==="))
    state8 = workflow.chat("今天天气怎么样", thread_id="test-fresh-today-006")
    rec8: str = getattr(state8, "final_recommendation", "")
    weather8 = getattr(state8, "weather_result", None)
    is_ood8: bool = getattr(state8, "is_out_of_domain", False)
    print(f"  is_out_of_domain: {is_ood8}")
    print(f"  weather_result: {weather8}")
    print(f"  final_recommendation: {rec8[:200]}")

    if not is_ood8:
        print(green("  ✓ '今天天气怎么样' NOT out-of-domain"))
    else:
        msg = "'今天天气怎么样' incorrectly classified as out-of-domain"
        print(red(f"  ✗ {msg}"))
        failures.append(msg)

    if "7月28日" in rec8:
        print(green("  ✓ '今天' resolves to July 28 (today)"))
    else:
        msg = f"'今天' did not resolve to today's date: {rec8[:150]}"
        print(red(f"  ✗ {msg}"))
        failures.append(msg)

    # ============================================================
    # Test 7: Verify agent principle architecture
    # ============================================================
    print(bold("\n=== Test 7: Agent Principle Architecture ==="))
    graph = workflow._app
    nodes = list(graph.nodes.keys()) if hasattr(graph, 'nodes') else []
    print(f"  Graph nodes: {nodes}")

    for expected_node in ["weather", "out_of_domain", "chitchat"]:
        if expected_node in nodes:
            print(green(f"  ✓ '{expected_node}' node exists"))
        else:
            msg = f"'{expected_node}' node missing from graph"
            print(red(f"  ✗ {msg}"))
            failures.append(msg)

    # ============================================================
    # Test 8: last_resolved_time survives weather short-circuit
    # (Was Bug: weather short-circuit return dict lacked last_resolved_time)
    # ============================================================
    print(bold("\n=== Test 8: last_resolved_time Survives Weather Short-Circuit ==="))
    thread_id8: str = "test-survive-short-008"
    # Turn 1: restaurant query with time → saves context
    state_a = workflow.chat("这周六我要吃川菜", thread_id=thread_id8)
    lrt_a = getattr(state_a, "last_resolved_time", None)
    print(f"  Turn 1 last_resolved_time: {lrt_a}")
    if lrt_a and lrt_a.get("resolved_date") == "2026-08-01":
        print(green("  ✓ Turn 1 saved time context"))
    else:
        msg = "Turn 1 did NOT save time context"
        print(red(f"  ✗ {msg}"))
        failures.append(msg)

    # Turn 2: weather query → short-circuit must NOT wipe last_resolved_time
    state_b = workflow.chat("当天天气怎么样", thread_id=thread_id8)
    lrt_b = getattr(state_b, "last_resolved_time", None)
    print(f"  Turn 2 last_resolved_time after weather short-circuit: {lrt_b}")
    # The crucial test: weather query's short-circuit should NOT destroy context
    if lrt_b:
        print(green(f"  ✓ last_resolved_time survived weather short-circuit: {lrt_b.get('resolved_date')}"))
    else:
        msg = "CRITICAL BUG: weather short-circuit wiped last_resolved_time!"
        print(red(f"  ✗ {msg}"))
        failures.append(msg)

    # Turn 3: another weather query → should still have context from Turn 2's update
    state_c = workflow.chat("那天会下雨吗", thread_id=thread_id8)
    rec_c: str = getattr(state_c, "final_recommendation", "")
    print(f"  Turn 3 ('那天会下雨吗'): {rec_c[:200]}")
    if "8月1日" in rec_c:
        print(green("  ✓ '那天' in Turn 3 still references Aug 1 — context chain intact"))
    else:
        msg = f"Turn 3 lost the context chain: {rec_c[:150]}"
        print(red(f"  ✗ {msg}"))
        failures.append(msg)

    # ============================================================
    # Summary
    # ============================================================
    print(bold("\n" + "=" * 60))
    if failures:
        print(red(f"{len(failures)} TEST(S) FAILED ✗"))
        print("=" * 60)
        for i, f in enumerate(failures, 1):
            print(red(f"  {i}. {f}"))
        return False
    else:
        print(green("ALL TESTS PASSED ✓"))
        print("=" * 60)
        print()
        print("Agent Principle compliance verified:")
        print("  1. LLM 负责'指代理解'  — TimeExpression extraction only, no date math")
        print("  2. Context 负责'事实记忆' — last_resolved_time persisted via MemorySaver")
        print("  3. 代码负责'时间计算'    — TimeResolver.resolve() deterministic arithmetic")
        print("  4. API 负责'真实数据'    — WeatherTool is sole weather data source")
        return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
