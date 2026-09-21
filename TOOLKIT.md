# CO-UCAgent Toolkit

CO-UCAgent 将长流程芯片验证中的性能观测、上下文构建、过程记忆、轨迹审计、策略工程、
实验复现和轨迹数据构建整理为七项可独立调用的工具。它们共享同一套结构化事件和验证
状态语义，因此可以从一次模型请求逐步连接到跨 DUT 的端到端实验。

## 安装与入口

```bash
git clone https://github.com/spadenoheart/CO-UCAgent.git
cd CO-UCAgent
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

安装后提供以下命令：

| 命令 | 作用 | 在线/离线 |
|---|---|---|
| `co-llm-profiler` | 分析和重放真实 Agent 的 TTFT、TPS 与 Token | 离线分析/在线重放 |
| `co-context` | 汇总结构化上下文事件、Stage 驻留和运行成本 | 离线分析 |
| `co-memory-cache` | 检查长期记忆条目、阶段分布和缓存指标 | 离线分析 |
| `co-trace` | 编译 trace tree 并生成静态或实时轨迹图 | 离线/实时 |
| `co-strategy` | 构建、去重、审计和准入跨 DUT 修复策略 | 离线构建/回放 |
| `co-bench` | 冻结配置并运行可恢复的多 DUT 实验 | 在线实验 |
| `co-trajectory-data` | 从真实轨迹构建可审计 SFT 数据 | 离线构建 |

所有入口均支持 `--help`。源码脚本仍可直接运行，便于二次开发和单步调试。

## 1. `co-llm-profiler`

该工具把“Agent 很慢”拆分为输入长度、TTFT、prefill、decode 和请求轮次，并允许将生产
轨迹中的精确输入重放到另一个推理后端。

```bash
# 汇总一份运行日志，并导出请求级 CSV
co-llm-profiler analyze log/Adder+*/ucagent-log.log --csv llm_requests.csv

# 从精确 llm_input_dumps 构建延迟用例
co-llm-profiler extract --log-root log --out-dir benchmark/llm_latency_cases

# 在 OpenAI-compatible/Ollama 接口上重放
co-llm-profiler replay benchmark/llm_latency_cases \
  --url http://127.0.0.1:11434/api/chat --num-predict 512
```

源码入口：`ucagent/abackend/langchain/message/performance.py`、
`scripts/analyze_llm_performance.py`、`scripts/extract_llm_latency_cases.py`、
`scripts/run_llm_latency_cases.py`。

## 2. `co-context`

在线 Context Engine 位于 Agent 主循环中；公开命令负责分析它生成的事件和运行效果。

```bash
co-context events output/workspace_Adder/unity_test/structured_events.jsonl --json
co-context runtime log/Adder+*/ucagent-log.log --markdown runtime_report.md
```

在线实现：`ucagent/abackend/langchain/message/conversation.py`、
`ucagent/abackend/langchain/middleware/messages.py`、`ucagent/stage/vmanager.py`。
运行时通过 `conversation_summary` 和 `context_upgrade` 配置结构化摘要、状态包、
failure-aware context 与 observation masking。

## 3. `co-memory-cache`

`co-memory-cache` 无需启动模型即可审计 workspace 或实验目录中的记忆文件。

```bash
co-memory-cache inspect output/workspace_Adder/.ucagent_memory --json
```

输出包含文件级条目数、memory type、Stage、DUT 和累计缓存指标。在线实现位于
`ucagent/memory/long_term.py`，记忆文件默认保存在
`<workspace>/.ucagent_memory/<DUT>/`。

## 4. `co-trace`

轨迹工具将文件读取、修改、测试、Checker、阶段迁移和失败事件组织为可查询的执行路径。

```bash
co-trace build output/workspace_Adder/unity_test/structured_events.jsonl \
  --out trace_tree.json --nodes-jsonl trace_nodes.jsonl

co-trace visualize benchmark/ucagent_experiments \
  --serve --port 8765 --refresh-seconds 20
```

浏览器中的运行和 Stage 选择器可用于比较不同 DUT、seed、baseline 与 CO-UCAgent 轨迹。
核心实现位于 `scripts/build_trace_tree.py`、`scripts/visualize_agent_trajectory.py` 和
`ucagent/util/test_result.py`。

## 5. `co-strategy`

策略工具将“失败前状态—行动—验证后状态”提取为 Repair Episode，执行角色化归因、结构
去重和风险约束准入。

```bash
co-strategy build trace_tree.json \
  --out-json context_reuse.json --out-md context_reuse.md

co-strategy curate --input context_reuse.json \
  --out-json context_reuse_curated.json --out-md curation_report.md

co-strategy analyze trace_tree.json \
  --out-json strategy_metrics.json --out-md strategy_metrics.md
```

对冻结 checkpoint 的注入/不注入结果可进一步执行准入回放：

```bash
co-strategy gate paired_decisions.jsonl \
  --pack context_reuse_curated.json --dut Adder \
  --out-json admission.json --out-md admission.md
```

在线检索位于 `ucagent/memory/context_reuse.py` 和
`ucagent/memory/trajectory_contract.py`；离线构建入口位于 `scripts/` 下对应模块。

## 6. `co-bench`

`co-bench` 用 suite JSON 固定源码、模型、上下文长度、环境变量、DUT、seed 和停止策略。

```bash
# 只验证矩阵和计划运行，不调用模型
co-bench run --suite path/to/suite.json --run-root benchmark/runs --dry-run

# 可恢复的正式运行
co-bench run --suite path/to/suite.json --run-root benchmark/runs

# 冻结关键配置和代码哈希
co-bench freeze --output-dir benchmark/frozen \
  --file config.yaml --file ucagent/setting.yaml
```

CodeBridge 通知在公开版本中默认关闭；已配置通知组件的本地环境可显式增加 `--notify`。

运行目录包含 ledger、日志、workspace 和有效性标记。clean end-to-end、checkpoint replay
与 patched resume 必须分开报告。实现入口为 `scripts/run_multi_dut_experiments.py`、
`scripts/freeze_experiment_baseline.py` 和 `scripts/run_adder_stage_benchmarks.py`。

## 7. `co-trajectory-data`

该工具优先读取精确 LLM input dump，并在不可用时从消息日志重建可见上下文；每条样本
保留 DUT、run、Stage、来源文件和质量分数。

```bash
co-trajectory-data build --log-roots log benchmark/ucagent_experiments \
  --out-dir benchmark/ucagent_finetune_dataset \
  --holdout-dut uart_tx

co-trajectory-data export \
  --source-dir benchmark/ucagent_finetune_dataset \
  --out-dir benchmark/ucagent_finetune_dataset/llamafactory_data
```

实现入口为 `scripts/build_ucagent_finetune_dataset.py` 和
`scripts/prepare_llamafactory_ucagent_dataset.py`。

## 推荐组合

性能优化闭环：

```text
co-llm-profiler -> co-context -> co-trace -> co-bench
```

跨 DUT 策略闭环：

```text
co-trace -> co-memory-cache -> co-strategy -> co-bench
```

轨迹训练闭环：

```text
co-trace -> co-trajectory-data -> SFT/LoRA -> co-llm-profiler -> co-bench
```

## 输出边界

公开仓库保留源码、配置模板、示例 DUT、冻结实验协议和经过筛选的统计结果。模型权重、
原始私有 RTL、访问令牌、完整运行 workspace、进程文件和大规模 Prompt dump 不进入发布
包。公开实验结论以 `res_public.csv` 和 README 中标注证据等级的结果为准。
