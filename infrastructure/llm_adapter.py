"""
LLM 适配器 (LLM Adapter)

本模块实现应用层定义的 ILLMClient 抽象接口，负责与具体的大语言模型服务进行通信。
支持对接中国主流开源大模型 API，如 DeepSeek、Qwen (通义千问) 等。

核心职责：
- 封装 API 认证、请求构造、超时重试等底层技术细节。
- 将模型返回的原始响应转换为领域层可消费的标准化结构。
- 支持多种模型后端的热切换（通过配置驱动）。
- v3.1：新增流式输出（SSE / Token 级 generate_stream）。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Generator

from loguru import logger
from openai import OpenAI

from application.ports import ILLMClient
from infrastructure.usage_tracker import UsageTracker


class OpenAILikeClient(ILLMClient):
    """OpenAI 兼容协议的 LLM 客户端适配器。

    支持标准同步调用和流式 Token 级输出。
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        resolved_key: str = (
            api_key or os.getenv("DEEPSEEK_API_KEY", "")
        )
        resolved_url: str = (
            base_url
            or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        )
        resolved_model: str = (
            model or os.getenv("DEFAULT_MODEL", "deepseek-v4-pro")
        )
        self._model: str = resolved_model
        self._client: OpenAI = OpenAI(
            api_key=resolved_key,
            base_url=resolved_url,
        )
        logger.info(
            f"[LLM] init: model={self._model}, base_url={resolved_url}"
        )

    @property
    def model_name(self) -> str:
        """返回当前实例使用的模型名称。"""
        return self._model

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=4096,
        )
        # ── v3.1: 记录 Token 用量 ──
        if response.usage:
            tracker = UsageTracker()
            tracker.record_llm_call(
                model=self._model,
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )
        return response.choices[0].message.content or ""

    def _call_api_stream(
        self, system_prompt: str, user_prompt: str
    ) -> Generator[str, None, None]:
        """流式调用 LLM API，逐 Token yield 文本增量。

        使用 OpenAI 兼容的 stream=True 模式，
        底层通过 SSE (Server-Sent Events) 协议接收增量响应。
        """
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=4096,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
            # ── v3.1: 流末 chunk 含 usage，记录 Token 用量 ──
            if hasattr(chunk, "usage") and chunk.usage:
                tracker = UsageTracker()
                tracker.record_llm_call(
                    model=self._model,
                    prompt_tokens=chunk.usage.prompt_tokens or 0,
                    completion_tokens=chunk.usage.completion_tokens or 0,
                    total_tokens=chunk.usage.total_tokens or 0,
                )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        logger.info("[LLM] generate() called")
        result: str = self._call_api(system_prompt, user_prompt)
        logger.info(f"[LLM] generate() response len: {len(result)}")
        return result

    def generate_stream(
        self, system_prompt: str, user_prompt: str
    ) -> Generator[str, None, None]:
        """流式生成：逐 Token yield 文本块（v3.1 新增）。

        使用 SSE 协议实时接收 LLM 响应，适合前端打字机效果展示。
        """
        logger.info("[LLM] generate_stream() called")
        token_count: int = 0
        for token in self._call_api_stream(system_prompt, user_prompt):
            token_count += 1
            yield token
        logger.info(f"[LLM] generate_stream() done, {token_count} tokens")

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        logger.info("[LLM] generate_json() called")
        raw: str = self._call_api(system_prompt, user_prompt)
        json_text: str = self._extract_json(raw)
        try:
            result: dict[str, Any] = json.loads(json_text)
            logger.info("[LLM] generate_json() parsed OK")
            return result
        except json.JSONDecodeError as exc:
            logger.warning(f"[LLM] JSON parse failed, attempting repair: {exc}")
            result = self._repair_json(json_text)
            if result:
                logger.info("[LLM] generate_json() repaired OK")
                return result
            raise ValueError(f"LLM returned invalid JSON: {exc}") from exc

    @staticmethod
    def _extract_json(raw: str) -> str:
        """从 LLM 原始响应中提取 JSON 文本。

        尝试策略：
        1. Markdown code block (```json ... ```)
        2. Markdown code block (``` ... ```)
        3. 花括号对 { ... }
        4. 直接返回原始文本
        """
        # 策略 1: ```json ... ```
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if match:
            return match.group(1).strip()

        # 策略 2: 提取第一个 { 到最后一个 }
        first_brace: int = raw.find("{")
        last_brace: int = raw.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return raw[first_brace : last_brace + 1]

        return raw

    @staticmethod
    def _repair_json(json_text: str) -> dict[str, Any] | None:
        """尝试修复常见 JSON 格式错误。

        修复策略：
        1. 移除尾部多余逗号 (trailing commas)
        2. 尝试逐行修复
        3. 尝试提取含 null/NULL 值的字段并修补
        """
        import re as _re

        repaired: str = json_text

        # 修复 1: 移除尾部逗号
        repaired = _re.sub(r",\s*([}\]])", r"\1", repaired)

        # 修复 2: 替换可能的 SQL NULL 为 JSON null
        repaired = _re.sub(r':\s*NULL\b', ': null', repaired)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        # 修复 3: 尝试找到最后一个完整对象，截断多余内容
        last_closing: int = repaired.rfind("}")
        if last_closing > 0:
            try:
                candidate: str = repaired[: last_closing + 1]
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        return None
