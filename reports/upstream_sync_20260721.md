# UCAgent 上游同步审计（2026-07-21）

## 1. 同步对象与原则

- 上游快照：`UCAgent-main`，文件时间截至 2026-07-16；该目录不包含 Git 元数据，
  因此不能给出可靠的 upstream commit SHA。
- 研究工作树：`UCAgent`，同步前 HEAD 为
  `b9c259b35d048e70cf4d0429d3b20a0077fed15d`，且包含未提交研究改动。
- 发布工作树：`CO-UCAgent`，同步前 HEAD 为
  `52000a9f3a351d0fa36e6fd2ca98f915fe28bcc8`。
- 原则：上游新增功能优先迁移；研究运行时采用手工合并；不回退或覆盖无法确认归属的
  本地改动。同步前备份保存在 `/tmp/`，不属于发布产物。

## 2. 迁移结果

自动比较记录共 368 项：

- 310 个上游新增文件已迁移。
- 58 个“上游变化且本地未改”的文件已更新。
- Agent、StageManager、Checker、消息生命周期、工具与配置等研究冲突文件采用手工合并。

最终逐文件复核（排除缓存和 `.pyc`）覆盖上游 602 个源文件：研究工作树中 570
个与上游字节一致，30 个因研究适配而不同；其余 2 个可执行探测脚本内容完整保留，但
分别改名为 `tests/manual_mcp_probe.py` 和 `tests/legacy_unity_env_and_bundle.py`，避免被
默认 pytest 收集为稳定单元测试。公开仓库为 569 个一致、31 个适配/脱敏版本和相同的
2 个重命名脚本。

迁移的主要上游能力包括 Formal/static-bug 工作流、Skill、Textual TUI、Web console、
master/worker API、Docker/Kubernetes 支持、workspace archive/URL 输入、新 CLI/config
override、stage journal、LLM Checker suggestion、新示例、文档和测试。

## 3. 研究功能适配

- 保留 `create_react_agent + pre_model_hook`，继续支持精确 LLM 输入 dump、TTFT/TPS
  采集、结构化/分层摘要、failure-aware context 和 context reuse。上游
  `create_agent + middleware` 迁移被推迟到独立实验，避免同时改变 Agent 执行语义和
  context-reuse 实验变量。
- 将结构化修改事件、六类测试终态、RunTestCases/Check/Complete 观察、trace tree、
  role-typed Episode credit、跨 DUT 检索和 v2.2 semantic/risk gate 接入新版 StageManager。
- 修复 Checker 回调列表跨实例泄漏、SearchText 忽略 workspace 根目录、StageManager
  轻量测试配置兼容和 `UCMessagesNode.reset_chat` 生命周期兼容。
- `RunTestCases` 改用当前解释器的 `python -m pytest`，减少 Conda 环境错配。

## 4. 验证结果

- `UCAgent`：358 passed，6 skipped；2 条 LangGraph deprecation warning。
- `CO-UCAgent`：357 passed，7 skipped；2 条相同 warning。
- 归因、测试终态、策略库整理与基线冻结专项：55 passed。
- 两个仓库的 MkDocs strict build 均通过。
- 发布研究数据：30 个 JSON 和 20 个 JSONL 均可解析，JSONL 共 3767 行。
- 当前工作树和 `CO-UCAgent` 全部 Git 历史未发现真实 API token；示例测试中的
  `sk-example-secret` 是脱敏单测夹具。

## 5. 实验冻结与公开边界

内部冻结目录：
`benchmark/ucagent_experiments/20260721_upstream_sync_v22_candidate/`。它定义 B0
（关闭复用）、B1（v1 检索）、B2（v2.1 utility gate）和 B3（v2.2 semantic/risk
gate）四个实验臂。37 个 snapshot 文件与 manifest 哈希一致。

内部策略包仍含历史日志绝对路径，因此公开仓库不发布该原始 snapshot，只发布脱敏后的
研究文件、manifest 和运行说明。公开版本不包含 `log/`、`output/`、模型权重、私有
RTL 或大规模 LLM input dump。

## 6. 尚未完成与风险

- 尚未将上游 `create_agent + middleware` 纳入研究运行时；这不是当前同步遗漏，而是
  为控制实验变量保留的独立兼容任务。
- 当前端到端数据不能证明 context reuse 提速。v2.1/v2.2 的历史单次运行均未超过近期
  Adder 运行；必须补齐固定模型、硬件、seed 和 checkpoint 的 B0-B3 配对实验。
- `res.csv` 混合了不同代码、模型与硬件版本，只能使用可追溯到日志和冻结配置的记录。
- GitHub 发布前应由维护者检查大文件政策，并轮换曾在对话或终端中暴露过的外部 token，
  即使它们当前不在仓库中。
