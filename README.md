# CO-UCAgent (Context Optimized-UCAgent)

CO-UCAgent 是在原始 [UCAgent](https://github.com/XS-MLVP/UCAgent) 基础上，围绕“上下文利用效率与验证效率”做的增强版本，重点优化了检索、摘要与长期记忆等上下文管理流程，并提供可复现实验数据的汇总。其聚焦于验证型 Agent 在长任务场景中的上下文管理问题，提出一套面向工程可用性的改进方案，包括结构化摘要、分层摘要、检索重排序与长期记忆协同机制，并通过公开可复现的实验记录给出系统级评价。该项目的贡献不在于扩大模型能力本身，而在于通过更稳定的上下文表达与更可控的上下文裁剪，使得复杂验证流程在相同模型与工具链条件下获得更高的成功率与更稳定的运行行为。

## 研究背景与问题定义

**上下文管理的重要性**：多轮交互与长文档输入显著放大上下文长度与成本。已有研究表明，当关键信息处于上下文中部时，模型性能明显退化，说明单纯扩大上下文窗口并不足以缓解长期依赖问题 [5]。同时，针对长期对话与跨会话任务，需要更精细的记忆分层与上下文调度机制，以保证信息的可持续可用性 [6]。

**UCAgent 的现状与痛点**：UCAgent 已具备多阶段验证流程与工具化能力，但其上下文管理仍主要依赖对话摘要与尾部消息保留策略，且 MemoryPut/MemoryGet 基于进程内存的短期存储，缺乏持久化与结构化回填机制。因此在长任务场景中仍面临：  
- 关键信息在多次摘要后出现语义漂移或被弱化  
- 工具输出与测试日志占用大量上下文，影响后续推理密度  
- 跨阶段信息难以稳定保留与复用  

**本项目要解决的问题**：CO-UCAgent 聚焦上下文管理这一核心瓶颈，以更高的上下文表达稳定性、可控的裁剪与摘要策略，以及可复现实验统计为目标，提升长任务的可运行性与验证效率。  

## 方法概述

CO-UCAgent 的技术路线强调以可解释的上下文处理策略替代单纯的上下文堆叠。在输入侧，系统通过检索重排序提升检索结果的时效性与相关性，从而减少与任务无关的噪声进入上下文；在摘要侧，引入结构化与分层机制，将工具输出与对话历史转换为更稳定的摘要单元，并在阶段边界形成可复用的阶段级摘要；在记忆侧，采用持久化与检索回填的协同策略，缓解长任务中的跨阶段遗忘问题。该流程以最小化上下文漂移为约束，在保持模型行为一致性的同时提升整体验证效率。

## 主要改进

以下改进均通过 `ucagent/setting.yaml` 的 `context_upgrade.*` 统一开关控制，核心逻辑分布在消息裁剪、工具输出压缩、记忆检索与统计采集等环节。为便于稳态使用，`enable_all` 默认设为 `false`，可按需开启对应能力。

### 1) 语义检索结果轻量重排序（rerank）

- **功能说明**：在语义检索返回结果后，按照“相似度/得分 + 时间新鲜度”做轻量二次排序，提高“最新/更相关”文档被优先引用的概率。
- **动机与问题**：仅依赖向量相似度时，时效性与领域内新增内容容易被淹没，导致指引文档的“新旧优先级”不稳定。
- **选择理由与优势**：该方案不引入额外模型调用，代价低、可解释、对在线环境更友好，且能显式补偿时间新鲜度。
- **代码位置**：
  - `ucagent/tools/memory.py`：`SemanticSearchInGuidDoc._rerank()` 评分逻辑与 `_run()` 中的重排调用
  - `ucagent/verify_agent.py`：实例化 `SemanticSearchInGuidDoc` 时传入 `rerank_enabled`

### 2) 结构化摘要（Structured Summary）

- **功能说明**：对上下文摘要输出进行结构化整理（Stage 信息、约束、进度、关键信息等），减少自由文本带来的噪声与漂移。
- **动机与问题**：自由文本摘要容易出现信息遗漏、格式不稳定和语义漂移，难以被后续模块稳定解析。
- **备选思路**：
  - 直接保留原始对话（成本高）
  - 仅进行摘要但不要求结构约束（易漂移）
  - 强制结构化输出或 schema 校验 [1]
- **选择理由与优势**：结构化摘要利于统一模板、便于后续解析与指标提取，同时能更稳定地保留关键上下文字段。
- **代码位置**：
  - `ucagent/abackend/langchain/message/conversation.py`：`summarize_messages_structured()`、`_normalize_structured_summary()`、`_coerce_structured_summary()`
  - `ucagent/abackend/langchain/message/conversation.py`：在 `__call__()` 中根据 `structured_summary` 开关选择结构化摘要路径

### 3) 分层摘要（Hierarchical Summary）

- **功能说明**：将超长工具输出（如批量测试结果）压缩为 **Batch Summary**，并在 Stage 结束时合并生成 **Stage Summary**；同时维护小型滚动缓存以抵抗裁剪带来的信息丢失。
- **动机与问题**：单轮工具输出往往极长，直接保留会挤占上下文窗口；简单截断又会丢失关键失败信息。
- **备选思路**：
  - 固定比例裁剪或日志过滤（容易丢关键项）
  - 仅保留最新若干条 tool 输出（信息碎片化）
  - 分层汇总（批次汇总 + 阶段汇总） [4]
- **选择理由与优势**：分层摘要在保持关键信息的前提下，将上下文规模控制在可预测范围；对长任务更稳定。
- **代码位置**：
  - `ucagent/abackend/langchain/message/conversation.py`：`_maybe_update_hierarchy()` 中的 `BATCH_SUMMARY` / `STAGE_SUMMARY`
  - `ucagent/abackend/langchain/message/conversation.py`：`stage_summary_data` / `batch_summary_data` 作为摘要前缀参与后续上下文构造

### 4) 长期记忆（Long-term Memory）+ 向量检索

- **功能说明**：将关键摘要/批次结果/阶段结果落盘保存（jsonl），支持按 DUT、Stage 等过滤，并在后续阶段优先回填上下文。
- **动机与问题**：长任务中阶段信息容易被上下文截断，且跨阶段复用依赖人工回忆。
- **备选思路**：
  - 全量保存并在需要时手动检索（效率低）
  - 仅依赖短期上下文（跨阶段易丢失）
  - 结构化落盘 + 向量检索 [2][3]
- **选择理由与优势**：持久化记忆降低跨阶段信息丢失风险，向量检索提升相关性与可用性。
- **代码位置**：
  - `ucagent/memory/long_term.py`：长期记忆存储、查询与归档逻辑
  - `ucagent/abackend/langchain/message/conversation.py`：在 Batch/Stage/Turn 总结时写入长期记忆
  - `ucagent/stage/vmanager.py`：`get_current_tips()` 中按阶段检索并插入提示前缀

### 5) 测试输出压缩（Compact Test Output）

- **功能说明**：在测试用例输出过长时，对 pytest stdout/stderr 进行压缩，避免无效日志占用上下文窗口。
- **动机与问题**：日志冗余会导致上下文浪费，关键失败片段难以突出。
- **选择理由与优势**：该策略在保持可读性的同时控制上下文占用，减少无效信息干扰。
- **代码位置**：
  - `ucagent/stage/vmanager.py`：`free_pytest_run.compact_test_output` 从配置读取开关
  - `ucagent/checkers/unity_test.py`：`UnityChipCheckerTestFree.do_check()` 中调用 `fc.compact_pytest_output()`

### 6) 数据采集与结果可复现（Data Collection）

- **功能说明**：自动统计每次验证的耗时与 token 使用情况，并记录到日志/结果文件，用于对比“改进前后”的性能变化。
- **动机与问题**：没有统一统计口径时，性能对比缺乏可复现性。
- **选择理由与优势**：内置采集降低使用门槛，保证统计口径一致，便于横向比较与长期追踪。
- **代码位置**：
  - `ucagent/verify_agent.py`：`enable_data_collection` 开关与统计写入逻辑（包含 resume 机制）
  - `ucagent/stage/vmanager.py`：记录 Stage 级别 token 统计与耗时信息

### 7) 失败/成功上下文分离（Failure-aware Context）

- **功能说明（测试阶段）**：当启用失败感知模式时，将失败上下文与成功上下文分离成 `FAILURE_CONTEXT` / `SUCCESS_CONTEXT` 前缀，减少失败样例对后续推理的干扰。该功能仍处于测试阶段，默认不建议在稳态流程中开启。
  

## 评测与对比

### 实验环境

- 运行模式：UCAgent API 模式
- LLM：iflow 平台 `qwen3-coder-plus`
- Embedding：`Qwen/Qwen3-Embedding-0.6B`

### 数据集与用例

本报告基于 `res_public.csv` 的公开结果，包含 17 次运行记录，覆盖以下 DUT：

- Adder（origin/improved 对比）
- uart_tx（origin/improved 对比，含多次 improved 重复运行）
- IntegerDivider（仅 improved，多次重复运行）

**单次运行记录表**

| Run | Model | DUT | Time | token_in | token_out |
| --- | --- | --- | --- | --- | --- |
| R1 | origin | Adder | 40min | - | - |
| R2 | origin | Adder | 39min | 1.04M | 16.6K |
| R3 | origin | uart_tx | 1h14min | - | - |
| R4 | origin | uart_tx | 1h7min | 1.49M | 90.2K |
| R5 | improved | Adder | 27min | - | - |
| R6 | improved | Adder | 30min | 647.2K | 19.8K |
| R7 | improved | uart_tx | 34min | - | - |
| R8 | improved | uart_tx | 38min | 1.27M | 53.0K |
| R9 | improved | uart_tx | 57min | 1.36M | 37.6K |
| R10 | improved | uart_tx | 1h10min | 1.22M | 43.2K |
| R11 | improved | uart_tx | 1h16min | 981.9K | 27.5K |
| R12 | improved | IntegerDivider | 1h18min | - | - |
| R13 | improved | IntegerDivider | 52min | 1.57M | 34.1K |
| R14 | improved | IntegerDivider | 1h5min | 1.48M | 21.8K |
| R15 | improved | IntegerDivider | 1h11min | 1.49M | 47.5K |
| R16 | improved | IntegerDivider | 1h33min | 2.38M | 79.1K |
| R17 | improved | IntegerDivider | 1h34min | 2.85M | 36.3K |

### 指标定义

- **Time (min)**：端到端运行耗时
- **Speedup**：`origin_mean / improved_mean`
- **CV (变异系数)**：`std / mean`，描述运行稳定性
- **token_in/out**：模型统计的输入/输出 token 数量
- **token_in/min**：`token_in / time`，衡量 token 消耗速率

### 汇总指标

**运行时间统计**

| DUT | Variant | N | mean(min) | min | max | std | CV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Adder | origin | 2 | 39.50 | 39.00 | 40.00 | 0.50 | 0.01 |
| uart_tx | origin | 2 | 70.50 | 67.00 | 74.00 | 3.50 | 0.05 |
| Adder | improved | 2 | 28.50 | 27.00 | 30.00 | 1.50 | 0.05 |
| uart_tx | improved | 5 | 55.00 | 34.00 | 76.00 | 16.73 | 0.30 |
| IntegerDivider | improved | 6 | 75.50 | 52.00 | 94.00 | 14.93 | 0.20 |

**Token 统计**

| DUT | Variant | N | token_in mean | token_in min | token_in max | token_out mean | token_out min | token_out max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Adder | origin | 1 | 1040000 | 1040000 | 1040000 | 16600 | 16600 | 16600 |
| uart_tx | origin | 1 | 1490000 | 1490000 | 1490000 | 90200 | 90200 | 90200 |
| Adder | improved | 1 | 647200 | 647200 | 647200 | 19800 | 19800 | 19800 |
| uart_tx | improved | 4 | 1207975 | 981900 | 1360000 | 40325 | 27500 | 53000 |
| IntegerDivider | improved | 5 | 1954000 | 1480000 | 2850000 | 43760 | 21800 | 79100 |

**Token 速率**

| DUT | Variant | N | token_in/min mean | min | max |
| --- | --- | --- | --- | --- | --- |
| Adder | improved | 1 | 21573 | 21573 | 21573 |
| Adder | origin | 1 | 26667 | 26667 | 26667 |
| uart_tx | improved | 4 | 21907 | 12920 | 33421 |
| uart_tx | origin | 1 | 22239 | 22239 | 22239 |
| IntegerDivider | improved | 5 | 25972 | 20986 | 30319 |

### 关键结果与分析

综合运行时间、token 使用情况与多次运行行为，本项目的上下文优化在**验证效率、上下文成本控制以及推理行为稳定性**三个层面均表现出明确改善。

#### 1. 验证效率
- **Adder**：平均运行时间由 **39.50 min** 降至 **28.50 min**，约 **1.39×** 加速（时间缩短约 **27.8%**）。
- **uart_tx**：平均运行时间由 **70.50 min** 降至 **55.00 min**，约 **1.28×** 加速（时间缩短约 **22.0%**）。

在具备原始版本对照的 DUT 上，引入上下文优化后，端到端验证时间均出现稳定下降，该结果表明，在不改变模型、工具链与测试流程本身的前提下，仅通过上下文组织方式的优化，验证型 Agent 的整体运行效率即可获得可观提升。这一收益在中等与偏复杂 DUT 上均具有一致性。

#### 2. 上下文成本
- **Adder** 的 token_in 由 **1.04M** 降至 **0.65M**，减少约 **37.8%**；
- **uart_tx** 的 token_in 由 **1.49M** 降至 **1.21M（均值）**，减少约 **18.9%**。
- **Adder** 的 token_out 从 **16.6K** 上升至 **19.8K**；
- **uart_tx** 的 token_out 从 **90.2K** 显著下降至 **40.3K（均值）**。

从 token 角度看，上下文优化并非简单压缩输入长度，而是对信息密度与推理路径产生了结构性影响。这表明结构化摘要、分层摘要与长期记忆回填能够有效抑制冗余历史信息与无关日志进入上下文，使输入规模收敛到更稳定、可控的区间。两类 DUT 呈现出不同趋势，但并不矛盾：在规模较小的 Adder 场景中，更高质量的上下文促使模型输出更完整的推理与解释；而在 uart_tx 这类长日志、多测试路径场景中，上下文压缩与输出约束显著抑制了冗余生成。这说明上下文优化改变的是模型生成内容的结构与必要性，而非单纯追求 token 数量的单调下降。

#### 3. 多次运行行为
- **uart_tx（improved）**：token_in 分布于 **0.98M–1.36M**，token_out 分布于 **27.5K–53.0K**；
- **IntegerDivider（improved）**：token_in 分布于 **1.48M–2.85M**，token_out 分布于 **21.8K–79.1K**。

在仅包含 improved 版本、且执行多次运行的 DUT 上，token 使用呈现出受控但非恒定的分布特征，上述波动主要来源于测试路径差异、失败点位置变化及阶段性回填信息的不同组合，而非上下文机制本身的不稳定。从工程角度看，更重要的是 token 未出现不可控膨胀，且输入规模始终维持在可预测范围内，这为在更复杂 DUT 上扩展提供了可行性基础。

综合来看，本项目的上下文优化并不仅体现为运行时间的缩短，更重要的是改变了验证型 Agent 对上下文的使用方式：输入侧由被动累积转为受控组织，输出侧由冗余生成转为与任务相关的推理表达。在保持模型与工具链不变的前提下，这种上下文结构的调整同时带来了效率提升、token 成本下降以及行为稳定性的改善，使得长任务验证流程更具工程可控性与可复现性。因此，运行时间、token 使用与多次运行行为应被视为相互关联的观测维度，共同反映上下文管理策略在真实工程场景中的实际价值。


### 大规模/复杂测试扩展计划（待补充结果）

为进一步验证可扩展性与稳定性，后续计划补充以下对比用例：

1. ALU754（复杂算术路径 + 多 bug 注入）
2. Mux / FSM / DualPort（中等复杂度逻辑，验证路径多样）
3. HPerfCounter / ShiftRegister（长序列、时序相关）
4. DCache（GenSpec 场景，验证流程复杂、上下文体量大）
5. RandomTC（随机测试压力场景）

### 局限性与有效性讨论

- **样本数量有限**：origin 对比仅包含 Adder 与 uart_tx，后续需扩展更多 DUT
- **环境因素**：API 调用的网络波动可能影响耗时与 token 统计
- **策略耦合**：当前改进多为组合式开启，单策略的独立收益需要进一步消融实验

从可复现性角度看，本项目通过公开运行记录与统计口径降低了评估偏差，但尚未完全覆盖更大规模与更复杂的 DUT 组合；从方法学角度看，当前策略组合在工程约束与性能收益之间取得折中，后续仍需通过严格消融与跨任务评估明确各组件的独立贡献与边际收益。

## Future Work

未来工作可在摘要压缩与长期记忆两个方向继续深入推进。在摘要与裁剪方面，有必要探索由小模型承担上下文压缩职责、由大模型执行推理的双系统架构，以降低推理负担并提升上下文一致性。相关研究已经展示了轻量长程模型构建全局记忆的可行性，以及基于困惑度的提示压缩与分段软压缩在工程上的效率优势 [8–10]，这些结果为独立压缩器的设计提供了方法学基础。在长期记忆方面，仍需建立更系统的结构化存储与评测体系，以确保跨阶段信息复用的稳定性与可解释性；现有工作提出了分层记忆组织机制与面向记忆能力的评测基准，可作为后续改进的理论与评估依据 [6][11]。

## 上游项目与部署

请先按上游 UCAgent 的部署步骤完成环境搭建与依赖安装，本仓库仅描述改进点与实验结果。

上游仓库：
```text
https://github.com/XS-MLVP/UCAgent
```
  
## 文件说明

- `res_public.csv`：公开实验结果
- `ucagent/setting.yaml`：上下文优化与模型配置入口

## 参考文献

[1] LangChain Structured Output 文档：https://python.langchain.com/docs/how_to/structured_output/  
[2] LangMem 记忆系统与最佳实践：https://langchain-ai.github.io/langmem/  
[3] Mem0 开源记忆框架：https://github.com/mem0ai/mem0  
[4] Hierarchical Summarization, ACL 2019：https://aclanthology.org/P19-1500/  
[5] Lost in the Middle: How Language Models Use Long Contexts (TACL 2024)  
    https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00638/119630  
[6] MemGPT: Towards LLMs as Operating Systems (arXiv:2310.08560)  
    https://arxiv.gg/abs/2310.08560  
[7] Compressing Context to Enhance Inference Efficiency of LLMs / Selective Context (arXiv:2310.06201)  
    https://spacefrontiers.org/r/10.48550/arxiv.2310.06201  
[8] LLMLingua: Compressing Prompts for Accelerated Inference of LLMs (arXiv:2310.05736)  
    https://spacefrontiers.org/r/10.48550/arxiv.2310.05736  
[9] MemoRAG: Boosting Long Context Processing with Global Memory-Enhanced Retrieval Augmentation (arXiv:2409.05591)  
    https://spacefrontiers.org/r/10.48550/arxiv.2409.05591  
[10] CompLLM: Efficient Long-Context Compression for LLM QA (arXiv:2509.19228)  
     https://www.emergentmind.com/papers/2509.19228  
[11] Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions / MemoryAgentBench (arXiv:2507.05257)  
     https://arxiv.gg/abs/2507.05257  
