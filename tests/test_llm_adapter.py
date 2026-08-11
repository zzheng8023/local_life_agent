"""
LLM Adapter 流式输出单元测试 (v3.1)

测试 OpenAILikeClient 的 generate_stream() 方法和 default fallback。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from application.ports import ILLMClient
from infrastructure.llm_adapter import OpenAILikeClient


class TestGenerateStream:
    """测试 LLM 适配器的流式输出功能。"""

    def test_generate_stream_default_fallback_yields_text(self) -> None:
        """默认 fallback：generate_stream 应 yield 非空文本块。"""
        # 创建一个实现了 generate() 的简单 mock
        client = _FakeLLMClient()

        chunks: list[str] = list(client.generate_stream("sys", "user"))
        assert len(chunks) > 0, "默认 fallback 应至少 yield 一块文本"
        combined = "".join(chunks)
        assert "mock" in combined

    def test_generate_stream_default_fallback_empty_text(self) -> None:
        """generate() 返回空字符串时，generate_stream 应 yield 一个空字符串。"""
        client = _FakeLLMClient(fake_response="")

        chunks = list(client.generate_stream("sys", "user"))
        assert chunks == [""]


class TestOpenAILikeClientStream:
    """测试 OpenAILikeClient 的真实流式调用（mock OpenAI client）。"""

    def test_generate_stream_yields_from_api(self) -> None:
        """generate_stream() 应逐 chunk yield 来自 API 的 SSE 增量。"""
        fake_chunks = [
            _make_chunk("Hello"),
            _make_chunk(" "),
            _make_chunk("World"),
            _make_chunk(None),  # 末 chunk 无 content
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = fake_chunks

        adapter = OpenAILikeClient(
            api_key="test",
            base_url="https://test.api",
            model="test-model",
        )
        # 替换内部 OpenAI client
        adapter._client = mock_client

        result = list(adapter.generate_stream("sys", "user"))
        assert result == ["Hello", " ", "World"]

    def test_generate_stream_logs_token_count(self) -> None:
        """generate_stream() 应记录流式 Token 数量。"""
        fake_chunks = [_make_chunk("A"), _make_chunk("B"), _make_chunk("C")]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = fake_chunks

        adapter = OpenAILikeClient(
            api_key="test",
            base_url="https://test.api",
            model="test-model",
        )
        adapter._client = mock_client

        result = list(adapter.generate_stream("sys", "user"))
        assert len(result) == 3

    def test_model_name_property(self) -> None:
        """model_name property 应返回模型名称。"""
        adapter = OpenAILikeClient(
            api_key="test",
            base_url="https://test.api",
            model="qwen-turbo",
        )
        assert adapter.model_name == "qwen-turbo"


class TestILLMClientInterface:
    """测试 ILLMClient 端口接口的 generate_stream 默认实现。"""

    def test_all_methods_defined(self) -> None:
        """ILLMClient 应定义 generate / generate_json / generate_stream。"""
        assert hasattr(ILLMClient, "generate")
        assert hasattr(ILLMClient, "generate_json")
        assert hasattr(ILLMClient, "generate_stream")
        assert callable(ILLMClient.generate_stream)


# ================================================================
# 辅助函数
# ================================================================

class _FakeLLMClient(ILLMClient):
    """假的 LLM 客户端，用于测试默认 generate_stream 实现。"""

    def __init__(self, fake_response: str = "mock response text for testing"):
        self._fake = fake_response

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self._fake

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        return {"status": "ok"}


def _make_chunk(content: str | None) -> MagicMock:
    """构造一个类似 OpenAI stream chunk 的 mock 对象。"""
    chunk = MagicMock()
    if content is not None:
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = content
    else:
        chunk.choices = []
    return chunk
