# local_life_agent

> 基于 LangGraph 的多轮对话本地生活智能决策助手

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-57%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 概述

local_life_agent 是一个面向本地生活场景的智能决策助手，基于 **LangGraph 状态机架构 + DDD 四层分层**构建。它能够理解多用户之间的复杂偏好，进行跨领域（餐饮 + 住宿 + 娱乐 + 购物 + 交通）的时空联动推理，并生成安全的个性化推荐方案。

### 核心能力

- **多用户协调**：自动检测多人偏好的冲突点，按安全优先级（忌口 > 口味 > 预算）协调
- **时空连续推理**：同时考虑吃饭 → 看电影 → 住宿的空间距离和时间衔接
- **多领域覆盖**：餐饮 · 住宿 · 娱乐 · 购物 · 交通
- **双层安全审查**：程序化正则预筛选 + LLM 安全审查，四层红线规则确保 Agent 不编造履约信息
- **多轮对话记忆**：基于 LangGraph MemorySaver 的跨轮次上下文保持
- **流式输出**：Token 级实时流式生成，SSE 心跳保活
- **双模型分级路由**：smart_llm（推荐推理）+ fast_llm（分析/分类/安全审查）

## 项目结构

```
local_life_agent/
├── application/              # 应用层
│   ├── ports.py              #   抽象接口（ILLMClient, ITool）
│   ├── workflow.py           #   LangGraph 状态图编排
│   └── config.py             #   统一配置模块（环境变量 + 路径管理）
├── domain/                   # 领域层
│   ├── entities.py           #   Pydantic 实体模型
│   ├── prompt_specs.py       #   Prompt 模板管理器（核心业务逻辑）
│   ├── conflict_detector.py  #   程序化冲突检测器（6 条规则）
│   ├── tool_registry.py      #   插件化工具注册表
│   └── sharing.py            #   分享卡片生成
├── infrastructure/           # 基础设施层
│   ├── db.py                 #   SQLite 持久化（WAL 模式，三表设计）
│   ├── llm_adapter.py        #   OpenAI 兼容客户端（同步 + 流式 + JSON 修复）
│   ├── favorites_store.py    #   用户收藏存储
│   ├── audit_logger.py       #   安全审计日志
│   ├── safety_prefilter.py   #   安全预筛选（5 条预编译正则 + 白名单）
│   ├── tracer.py             #   可观测性追踪（日志轮转）
│   ├── usage_tracker.py      #   API 用量追踪
│   ├── district_loader.py    #   高德行政区划加载
│   ├── itinerary_extractor.py    # 行程结构化提取
│   ├── reminder_bridge.py        # 提醒参数构建
│   ├── amap_tool.py              # 餐饮 POI 搜索
│   ├── amap_hotel_tool.py        # 酒店搜索
│   ├── amap_entertainment_tool.py    # 娱乐场所搜索
│   ├── amap_shopping_tool.py     # 购物中心搜索
│   ├── amap_transit_tool.py      # 公交/地铁站搜索
│   ├── amap_transit_direction.py # 公交路径规划
│   ├── amap_parking_tool.py      # 停车场搜索
│   └── amap_bike_tool.py         # 共享单车搜索
├── interfaces/               # 接口层
│   ├── web_ui.py             #   Gradio Web 界面
│   └── main_cli.py           #   命令行交互（Rich + Live）
├── restaurant_agent/         # 餐厅子域
│   ├── schemas/              #   时间/天气数据模型
│   └── tools/                #   时间解析/天气查询工具
├── tests/                    # 测试（57 个用例，全部通过）
│   ├── test_conflict_detector.py      # 冲突检测（16 个）
│   ├── test_safety_prefilter.py       # 安全预筛选（17 个）
│   ├── test_entities.py               # 实体模型（16 个）
│   ├── test_llm_adapter.py            # LLM 适配器（6 个）
│   ├── test_cross_turn_time.py        # 跨轮次时间解析
│   └── eval_runner.py                 # 评测运行器
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example             # 环境变量模板
├── .gitignore
└── README.md
```

## 快速开始

### 环境要求

- Python 3.10+
- DeepSeek 或 Qwen（通义千问）API Key
- （可选）高德地图开放平台 API Key

### 安装

```bash
git clone https://github.com/zzheng8023/local_life_agent.git
cd local_life_agent
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，填入 API Key
```

### 配置

编辑 `.env`：

```ini
# LLM 配置（必填 — 至少配一个）
DEEPSEEK_API_KEY=sk-your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEFAULT_MODEL=deepseek-chat

# 快速模型（分析/分类/安全审查）
FAST_MODEL=qwen-turbo
FAST_API_KEY=sk-your-key
FAST_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 高德地图（可选 — 不配则使用模拟数据）
AMAP_API_KEY=your-amap-key
DEFAULT_CITY=北京
```

### 启动

```bash
# Web 界面（推荐）
python interfaces/web_ui.py
# → http://localhost:7860

# 命令行界面
python interfaces/main_cli.py
```

### Docker

```bash
docker compose up -d
# → http://localhost:7860
```

## 架构

```
用户输入
  ↓
[Analyze] 意图分析 ──→ 闲聊/领域外/反问 → END
  ↓ (正常检索)
[Retrieve] 并行调用 8 个高德 API
  ↓
[Recommend] 时空推理 + 多用户协调
  ↓
[Safety] 预筛选（正则）→ LLM 审查 → 改写/放行
  ↓
END → 流式 SSE 输出
```

- **图引擎**：LangGraph `StateGraph` + `MemorySaver`（按 `thread_id` 隔离多轮对话）
- **模型路由**：`smart_llm` 处理推荐推理，`fast_llm` 处理分析/分类/安全审查
- **工具注册**：`ToolRegistry` 插件化注册，支持动态发现和上下文注入

## 使用示例

### 多用户偏好协调

```
> A想吃火锅，B想吃日料，C不想太贵，3人聚餐

系统自动：
1. 检测口味冲突（火锅 vs 日料）
2. 检测预算约束
3. 提出折中方案 → 日式涮涮锅、性价比居酒屋
```

### 跨领域时空联动

```
> 4人聚餐，人均150，要有包间，吃完看电影，附近有酒店更好

系统并行检索：餐厅 + 电影院 + 酒店，输出步行距离和时间衔接方案。
```

## 运行测试

```bash
# 全部测试（57 个）
python -m pytest tests/ -v

# 带覆盖率
python -m pytest tests/ -v --cov=. --cov-report=term-missing
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 图引擎 | LangGraph StateGraph + MemorySaver |
| LLM | OpenAI 兼容 API（DeepSeek / Qwen） |
| 地图搜索 | 高德 POI / 公交 / 路径规划 API |
| 数据模型 | Pydantic v2 |
| 持久化 | SQLite（WAL 模式） |
| Web UI | Gradio 5+ |
| CLI | Rich + Live |
| 流式输出 | SSE + 心跳保活 |
| 安全审查 | 正则预筛选 + LLM 审查（双层） |
| 测试 | pytest（57 个用例） |
| CI | GitHub Actions |
| 容器 | Docker + Compose |

## 架构设计原则

- **DDD 四层分层**：Interface / Application / Domain / Infrastructure，依赖倒置
- **Prompt-as-Business-Logic**：Prompt 模板集中在 `domain/prompt_specs.py`，修改业务逻辑无需改其他文件
- **插件化工具**：新增领域工具只需实现 `ITool` 接口并注册到 `ToolRegistry`
- **统一配置**：所有魔法值集中在 `application/config.py`，均可通过环境变量覆盖

## License

MIT
