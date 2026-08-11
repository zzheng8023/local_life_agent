"""
端口接口定义 (Ports / Abstract Interfaces)

本模块遵循依赖倒置原则 (Dependency Inversion Principle)，
定义应用层所需的外部服务抽象接口。领域层和应用层仅依赖这些抽象，
具体实现由基础设施层 (infrastructure/) 提供并在运行时注入。

核心端口：
- ILLMClient：大语言模型客户端的抽象接口，定义统一的 LLM 调用契约。
- ITool：工具能力抽象接口，定义搜索、信息收集等外部动作的统一契约。
"""

from abc import ABC, abstractmethod
from typing import Any, Generator


class ILLMClient(ABC):
    """大语言模型客户端抽象接口。

    所有 LLM 后端的适配器（DeepSeek、Qwen、OpenAI 等）均需实现此接口，
    确保上层业务逻辑与具体模型实现解耦。

    采用三种输出模式：
    - generate：返回自由文本，适用于非结构化生成场景。
    - generate_json：返回结构化 JSON，适用于偏好提取、推荐输出等需要机读结果的场景。
    - generate_stream：流式返回自由文本，逐 Token yield，适用于需要打字机效果的实时展示。
    """

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """以自由文本模式调用 LLM。

        Args:
            system_prompt: 系统级指令，定义模型的角色、行为边界和输出约束。
            user_prompt: 用户级输入，包含当前轮次的具体查询或上下文。

        Returns:
            模型生成的原始文本响应。
        """
        ...

    @abstractmethod
    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        """以结构化 JSON 模式调用 LLM。

        Args:
            system_prompt: 系统级指令，须包含 JSON Schema 约束说明。
            user_prompt: 用户级输入，包含需要结构化解析的文本内容。

        Returns:
            模型返回的已解析 JSON 字典。

        Raises:
            ValueError: 当模型返回的内容无法解析为合法 JSON 时抛出。
        """
        ...

    def generate_stream(
        self, system_prompt: str, user_prompt: str
    ) -> Generator[str, None, None]:
        """以流式模式调用 LLM，逐 Token yield 文本块。

        默认实现：fallback 到 generate() 后一次性 yield 全部文本。
        子类可重写以实现真正的 SSE 流式输出。

        Args:
            system_prompt: 系统级指令。
            user_prompt: 用户级输入。

        Yields:
            每次产生一小段文本（通常为单 Token），适合前端打字机效果。
        """
        full: str = self.generate(system_prompt, user_prompt)
        if full:
            # 按字符分块输出，模拟逐 Token 效果
            chunk_size: int = max(1, len(full) // 20)
            for i in range(0, len(full), chunk_size):
                yield full[i : i + chunk_size]
        else:
            yield ""


class ITool(ABC):
    """工具能力抽象接口。

    定义 Agent 可调用的外部工具的统一契约。每个具体工具（搜索、地图、交通等）
    均需实现此接口，以支持工作流中的动态调度与组合。

    工具设计遵循"自描述"原则：每个工具通过 get_name 和 get_description
    暴露其身份与能力边界，使 LLM 可以在 Function Calling 场景下自动选择工具。
    """

    @abstractmethod
    def get_name(self) -> str:
        """返回工具的唯一标识名称。

        Returns:
            工具名称，如 "web_search"、"map_nearby"。
        """
        ...

    @abstractmethod
    def get_description(self) -> str:
        """返回工具的功能描述与参数说明。

        Returns:
            人类可读的工具描述文本，用于 LLM 的 Function Calling 上下文。
        """
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """执行工具的核心逻辑。

        Args:
            **kwargs: 工具所需的运行时参数，具体由子类定义。

        Returns:
            工具执行结果，类型取决于具体工具的实现。
        """
        ...
