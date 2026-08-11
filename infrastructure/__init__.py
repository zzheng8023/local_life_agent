"""
基础设施层 (Infrastructure Layer)

DDD 架构中的最外层，负责与外部系统、框架、第三方服务进行具体交互。
本层实现应用层定义的端口接口 (Ports)，将技术细节与业务逻辑完全隔离。

核心职责：
- LLM 适配器 (llm_adapter)：对接中国开源大模型 API（如 DeepSeek、Qwen 等）。
- Web 工具 (web_tools)：执行搜索、网页抓取等具体的信息收集动作。
- 追踪器 (tracer)：拦截并记录系统运行过程中的 Trace 数据与评测日志。
"""
