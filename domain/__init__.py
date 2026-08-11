"""
领域层 (Domain Layer)

DDD 架构中的核心层，承载系统的全部业务逻辑与规则。
本层遵循"依赖无环"原则 —— 不依赖任何外部框架、数据库或 UI 层，
所有模型均使用纯 Python 定义，确保业务逻辑的可测试性与可迁移性。

核心职责：
- 定义领域实体 (Entity) 与值对象 (Value Object)：例如 AgentState、用户偏好等核心业务概念。
- 管理 Prompt 规范的领域规则 (PromptSpecs)：Prompt 是本 Agent 系统的核心业务逻辑，
  所有模板定义、变量约束、组合规则均在此层声明。
"""
