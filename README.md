# CO-UCAgent (Context Optimized-UCAgent)

CO-UCAgent 是在原始 UCAgent 基础上，围绕“上下文利用效率与验证效率”做的增强版本，重点优化了检索、摘要与长期记忆等上下文管理流程，并提供可复现实验数据的汇总。

## 主要改进（逐项说明）

以下改进均通过 `ucagent/setting.yaml` 的 `context_upgrade.*` 统一开关控制，核心逻辑分布在消息裁剪、工具输出压缩、记忆检索与统计采集等环节。为便于稳态使用，`enable_all` 默认设为 `false`，可按需开启对应能力。

### 1) 语义检索结果轻量重排序（rerank）

- **功能说明**：在语义检索返回结果后，按照“相似度/得分 + 时间新鲜度”做轻量二次排序，提高“最新/更相关”文档被优先引用的概率。  
- **代码位置**：
  - `ucagent/tools/memory.py`：`SemanticSearchInGuidDoc._rerank()` 对每条记忆项进行评分（相似度 + 时间衰减加分），并在 `_run()` 中根据开关启用重排，同时记录前后对比日志。
  - `ucagent/verify_agent.py`：实例化 `SemanticSearchInGuidDoc` 时传入 `rerank_enabled`。

### 2) 结构化摘要（Structured Summary）

- **功能说明**：对上下文摘要输出进行结构化整理（Stage 信息、约束、进度、关键信息等），减少自由文本带来的噪声与漂移。  
- **代码位置**：
  - `ucagent/abackend/langchain/message/conversation.py`：`summarize_messages_structured()`、`_normalize_structured_summary()`、`_coerce_structured_summary()` 等用于生成/修复结构化摘要。
  - `ucagent/abackend/langchain/message/conversation.py`：在 `__call__()` 中根据 `structured_summary` 开关选择结构化摘要路径。

### 3) 分层摘要（Hierarchical Summary）

- **功能说明**：将超长工具输出（如批量测试结果）压缩为 **Batch Summary**，并在 Stage 结束时合并生成 **Stage Summary**；同时维护一个小型滚动缓存以抵抗裁剪带来的信息丢失。  
- **代码位置**：
  - `ucagent/abackend/langchain/message/conversation.py`：`_maybe_update_hierarchy()`  
    - 当 `RunTestCases` 输出过长时，生成 `BATCH_SUMMARY` 并替换原始 tool message。  
    - 在 Stage 切换时生成 `STAGE_SUMMARY` 并写入缓存。  
  - `ucagent/abackend/langchain/message/conversation.py`：`stage_summary_data` / `batch_summary_data` 作为摘要前缀参与后续上下文构造。

### 4) 长期记忆（Long-term Memory）+ 向量检索

- **功能说明**：将关键摘要/批次结果/阶段结果落盘保存（jsonl），支持按 DUT、Stage 等过滤，并在后续阶段优先回填上下文。  
- **代码位置**：
  - `ucagent/memory/long_term.py`：长期记忆存储、查询与归档逻辑。
  - `ucagent/abackend/langchain/message/conversation.py`：在 Batch/Stage/Turn 总结时写入长期记忆。
  - `ucagent/stage/vmanager.py`：在 `get_current_tips()` 中对当前 Stage 进行相关记忆检索，并插入到提示前缀。

### 5) 测试输出压缩（Compact Test Output）

- **功能说明**：在测试用例输出过长时，对 pytest stdout/stderr 进行压缩，避免无效日志占用上下文窗口。  
- **代码位置**：
  - `ucagent/stage/vmanager.py`：`free_pytest_run.compact_test_output` 从配置读取开关。
  - `ucagent/checkers/unity_test.py`：`UnityChipCheckerTestFree.do_check()` 中调用 `fc.compact_pytest_output()` 对输出做精简。

### 6) 数据采集与结果可复现（Data Collection）

- **功能说明**：自动统计每次验证的耗时与 token 使用情况，并记录到日志/结果文件，用于对比“改进前后”的性能变化。  
- **代码位置**：
  - `ucagent/verify_agent.py`：`enable_data_collection` 开关与统计写入逻辑（包含 resume 机制）。
  - `ucagent/stage/vmanager.py`：记录 Stage 级别 token 统计与耗时信息。

### 7) 失败/成功上下文分离（Failure-aware Context）

- **功能说明**：当启用失败感知模式时，将失败上下文与成功上下文分离成 `FAILURE_CONTEXT` / `SUCCESS_CONTEXT` 前缀，减少失败样例对后续推理的干扰。  
- **代码位置**：
  - `ucagent/abackend/langchain/message/conversation.py`：`_build_failure_context_prefix()` 根据摘要标签拆分上下文。
  - `ucagent/stage/vmanager.py`：`get_current_tips()` 在插入长期记忆时区分失败/非失败记忆。



## 性能结果汇总（基于 `res_public.csv`）

实验环境说明（用于对比参考）：
- 运行模式：UCAgent API 模式
- LLM：iflow 平台 `qwen3-coder-plus`
- Embedding：`Qwen/Qwen3-Embedding-0.6B`

- **Adder**
  - 原始：40 min
  - 改进后：27 min
  - 约 **32.5%** 时间缩短（约 **1.48x** 加速）
- **uart_tx**
  - 原始：74 min
  - 改进后：34–70 min，平均 **49.8 min**
  - 约 **32.8%** 时间缩短（约 **1.49x** 加速）
- **IntegerDivider**
  - 改进后：52–94 min，平均 **75.5 min**
  - 原始基线暂无（原始 UCAgent 在该 DUT 上由于上下文压缩策略过于粗粒度、且多次压缩引入语义失真，导致流程无法稳定完成；本工作通过改进后的上下文管理策略使其可稳定运行）

Token 使用情况（仅统计含 token 记录的样本）：

- **uart_tx（3 次记录）**
  - token_in：1.22M–1.36M（平均 1.283M）
  - token_out：37.6K–53.0K（平均 44.6K）
- **IntegerDivider（5 次记录）**
  - token_in：1.48M–2.85M（平均 1.954M）
  - token_out：21.8K–79.1K（平均 43.8K）

## 上游项目与部署

请先按上游 UCAgent 的部署步骤完成环境搭建与依赖安装，本仓库仅描述改进点与实验结果。

上游仓库：
```text
https://github.com/XS-MLVP/UCAgent
```

## 文件说明

- `res_public.csv`：已过滤后的公开实验结果
- `ucagent/setting.yaml`：上下文优化与模型配置入口

 
