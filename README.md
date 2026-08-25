# Accrete

Accrete 是一个面向可检查、可验证执行过程的 Python Agent Harness。它的目标是观测一个 target Agent 的可观测数据，基于其提出优化建议并进行实验性改进。它保持主循环简单，显式管理工具与执行环境，持久化会话和轨迹，并为实验分析与受限候选改进提供基础能力。

## 能力

- **Agent 执行契约** — `infra/core` 定义类型化消息、工具调用、限额、hooks、状态与轨迹事件。
- **受控运行时** — `infra/runtime` 提供本地与 Docker 环境、工作区文件工具、网页搜索/抓取、OpenAI 兼容 Provider、持久会话与会话感知 Runner。
- **可观测运行** — 每次运行都写入会话记录和 JSONL 轨迹，使失败可追溯到模型响应、工具调用或环境结果。
- **实验分析** — 适配器读取 target （当前是 Mini-SWE Agent）轨迹；摄取与调查阶段将其转化为可供后续诊断和立项使用的结构化证据。
- **候选演化基元** — 冻结目标 revision、隔离执行、按策略导出 patch 与归档 candidate ref，使改进实验可被复核。

## 架构

```text
task
  -> provider + core loop
  -> tool registry
  -> local or Docker environment
  -> session + JSONL trajectory
  -> analysis evidence
  -> frozen candidate patch
```

运行时负责执行与可观测性。分析和演化消费产生的证据，不进入普通 Agent 运行的关键路径。

## 快速开始

需要 Python 3.10+ 和 [uv](https://docs.astral.sh/uv/)。

```powershell
uv sync --group dev
Copy-Item .env.example .env
```

在 `.env` 中设置所用模型系列的 API key（`OPENAI_API_KEY`、`DASHSCOPE_API_KEY`、`DEEPSEEK_API_KEY` 或 `XAI_API_KEY`）。只有使用网页搜索工具时才需要 `TAVILY_API_KEY`。

在 PowerShell 中运行测试：

```powershell
New-Item -ItemType Directory -Force .\.pytest_tmp | Out-Null
icacls .\.pytest_tmp /inheritance:e /grant '*S-1-5-11:(OI)(CI)M' /T /C /Q
uv run python -m pytest -q --basetemp=.\.pytest_tmp
```

## 项目结构

```text
infra/
  core/          Agent 主循环、契约、状态、hooks、轨迹与工具注册表
  runtime/       环境、Provider、会话、Runner 与内置工具
analyzer/        轨迹适配、摄取、调查、诊断和立项阶段
evolver/         冻结输入与按策略受限的候选 patch 导出
grader/          测评相关流程的包边界
tests/           契约与组件测试
```

## 公开范围

当前公开内容即为现阶段可用范围；其余组件会在完成整理后陆续发布。

## 量化记录

在同一固定的 SWE-bench Verified 50 题子集上，0810 与 0817 的最终 resolved 分别为 27/50（54%）和 26/50（52%），相差 1 题，处于可接受的运行波动范围。过程数据与成本则直接来自保留的轨迹和逐调用 usage 记录：

| 指标 | baseline | evolved candidate |
|---|---:|---:|
| 完整轨迹 / 预测 | 50 / 50 | 50 / 50 |
| resolved | 27 / 50 | 26 / 50 |
| 模型调用 | 4,667 | 3,288 |
| 平均调用 / 题 | 93.36 | 65.76 |
| 输入 token | 248.6M | 86.1M |
| 输出 token | 1.34M | 0.70M |
| 按官方阶梯价重算的模型成本 | ¥130.17 | ¥18.63 |

0817 比 0810 少用 1,379 次模型调用（约 30%），按轨迹 usage 与 `qwen3.5-flash` 的官方阶梯价格重算，模型成本减少约 ¥111.55。成本不包含控制台优惠、未记录的账务调整或外部基础设施费用。

## 状态

Accrete Agent 仍是持续演化的工程项目，而非稳定框架发行版。受测试覆盖的接口是预期集成面；实验 assets 和 benchmark 工作流会在能够提供可复现配置与清晰契约后再公开。

