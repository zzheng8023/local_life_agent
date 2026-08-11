"""
应用层 (Application Layer)

DDD 架构中的编排层，负责调度领域层的业务逻辑以完成具体的用例 (Use Case)。
本层定义对外部服务的抽象接口（端口），并通过依赖倒置原则 (DIP) 解耦领域层与基础设施层。

核心职责：
- 编排工作流 (Workflow)：将 Analyze → Retrieve → Recommend → Safety 串联为完整用例。
- 定义端口接口 (Ports)：声明 ILLMClient、ITool 等抽象基类，
  由基础设施层提供具体实现。
"""
