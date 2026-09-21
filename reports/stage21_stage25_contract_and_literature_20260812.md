# Stage 21/25 验证契约修复与近期文献定位

日期：2026-08-12

## 1. 问题与修复边界

本次修改针对两个已在 Adder 长时运行中复现的问题：

1. Stage 21 `basic_api_functional_test` 在修复失败用例时反复重写测试集合，曾出现已通过用例被删除、改名或重新失败，甚至整份测试文件被清空。仅靠 Prompt 无法阻止这种状态回退。
2. Stage 25 `refine_test_cases_based_on_functional_points` 的 Checker 把 pytest 非零退出直接视为阶段失败。对于“测试正确、失败真实暴露 DUT bug、bug 文档证据链完整”的任务，这与 UCAgent 要求保留失败测试的目标冲突。

本次采用“Prompt 引导 + 确定性 Checker 约束”，而不是只增加 Prompt：

- **测试执行契约**：将结果明确区分为 `pass`、`test_failure`、`infrastructure_error`、`timeout`、`crash` 和 `no_tests_collected`。只有正常通过，或仅包含可被 bug 文档验证的显式 `FAILED` 断言时，才进入后续语义检查。
- **Stage 25 证据门控**：pytest 非零退出不再直接否决阶段。真实 `FAILED` 必须继续通过 `check_report`，与 FG/FC/CK/BG/TC 文档链精确匹配；`ERROR`、`XFAIL`、`SKIPPED`、超时、崩溃、导入/收集错误和零测试仍被拒绝。
- **Stage 21 单调通过集合**：每次 Check 后持久化已经通过的 API test nodeid。后续 Check 禁止这些测试消失、改名或回退为失败；源码行号变化不会被误认为测试改名。
- **Prompt 约束**：把上述契约和已锁定测试集合显式传给模型，要求只增量修复当前失败函数或共用 API 根因，禁止整文件重写和缩减测试集合。
- **超时污染修复**：pytest 插件列表中的 `timeout-2.4.0` 不再被正则误判为真实 timeout。

## 2. 当前验证证据

- 相关单元与回归测试：`83 passed, 6 skipped`。
- 使用历史 Stage 25 的真实 Adder 工作区离线重放：44 个测试中 41 个通过、3 个真实 DUT 失败，35 个 CK 全部覆盖，bug 文档证据一致。旧 Checker 会因 pytest 非零退出提前失败；新 Checker 能继续验证证据链并通过。
- 当前修复只能保证 Checker 阻止测试集合回退，并不能保证模型第一次修改就正确。是否降低 Stage 21/25 耗时及总 Token，仍需相同模型、硬件、seed 和上下文配置下的全量 A/B 实验确认。

## 3. 建议实验

### 3.1 最小对比

| 组别 | 系统 | 用途 |
| --- | --- | --- |
| B0 | 同步后的原始 UCAgent | Prompt-only / 原 Checker 基线 |
| B1 | B0 + Stage 21/25 Prompt 约束 | 区分 Prompt 本身的收益 |
| B2 | B0 + 新 Checker 契约和单调集合 | 区分确定性约束的收益 |
| B3 | 完整 CO-UCAgent | 测量与 context reuse、failure-aware context 的组合效果 |

先用历史相同 seed 做一次故障复现，再对重要组 B0/B3 至少运行 3 个相同 seed。不能用单次最快结果声明击败 baseline。

### 3.2 主指标

- DUT 完成率与总墙钟时间。
- Stage 21、Stage 25 墙钟时间和模型调用次数。
- 输入/输出 Token，尤其是两个目标阶段的输入 Token。
- Checker 失败次数、无效测试执行次数。
- 测试集合缩减次数、已通过测试回退次数。
- 已确认 DUT bug 的 precision/recall，以及 FG/FC/CK/BG/TC 证据链通过率。

“击败 baseline”的最低条件应为：完成率不下降，B3 的 Stage 21/25 中位耗时和无效循环显著下降；总耗时与 Token 至少一项下降，且另一项没有明显退化。

## 4. 近期文献与本项目关系

### 4.1 直接支撑当前实现

1. **Coding-agents can replicate scientific machine learning papers**（2026-07）指出 Prompt 本身不能可靠保存进展或验证证据，完成条件应依赖持久化 workspace 状态、证据来源和 validation gate。这直接支持本次“已通过测试集合持久化 + Checker 证据门控”，而不是继续堆 Prompt。  
   https://arxiv.org/abs/2607.02134

2. **Verifiable Process Rewards for Agentic Reasoning**（2026-05）把可靠中间验证器转化为稠密过程信号，并强调收益依赖 oracle 质量。它说明在用 Episode 做归因、检索或训练前，必须先修复 Checker 误分类；否则错误 oracle 会污染策略库。  
   https://arxiv.org/abs/2605.10325

3. **TRIAGE**（2026-06）将行为分为 progress、useful exploration、no-progress 和 regression，并报告可靠识别成功轨迹中的 regression 是主要收益来源。本项目已有的 Episode role 和这次“已通过测试回退”检测，可以形成硬件验证域的可执行 regression 定义。  
   https://arxiv.org/abs/2606.32017

### 4.2 适合下一步实现的新方向

1. **Coding Agents as Test-Suite Auditors**（2026-08）不把单个测试失败直接当作 bug，而是使用多实现共识、暴力求解和输入合法性检查组成 certification chain。本项目可把 DUT bug 证据链升级为：合法 stimulus 检查 + reference model/metamorphic oracle + 重复复现 + RTL 静态根因 + FG/FC/CK/BG/TC 映射。当前代码只完成了测试执行有效性和文档映射，尚未实现完整认证链。  
   https://arxiv.org/abs/2608.01715

2. **Latent Programming Horizons in Coding Agents**（2026-07）表明模型隐状态可预测未来修改是否减少失败或引入回归。Ollama API 通常不能暴露 122B 模型逐层隐状态，因此不能直接复现；可先用 trace 中可观测特征训练轻量级 edit-risk classifier，在写文件前预测“整文件重写、测试集合缩减、已通过用例回退”的风险。  
   https://arxiv.org/abs/2607.05188

3. **SWE-MeM**（2026-06）让 Agent 根据轨迹状态、任务进展和上下文预算决定何时、压缩什么以及如何压缩，而非固定阈值摘要。对 CO-UCAgent 的实际启发是：摘要时必须保护当前 Checker 契约、锁定测试集合和最新失败签名，并在 stage 推进或信息密度下降时触发压缩。该论文依赖训练和 memory-aware GRPO，不能直接视为当前 context reuse 已复现。  
   https://arxiv.org/abs/2606.28434

4. **Plans Don't Persist**（2026-06）显示关键计划被朴素裁剪会显著损害长程任务成功率，且简单重新注入未必恢复。它支持把 Checker 契约和当前单调状态作为不可压缩的结构化状态，而不是普通历史消息。  
   https://arxiv.org/abs/2606.22953

5. **TRACE**（2026-06）把 rollout 预算分配到最可能产生结果分歧的中间前缀。对当前非 RL 系统，更现实的迁移方式不是直接构建大量 rollout，而是只在高不确定、高回归风险的修改点使用额外验证或候选分支，避免所有步骤平均增加模型调用。  
   https://arxiv.org/abs/2606.11119

## 5. 可形成论文的主线

建议把研究问题收敛为 **Verifier-Grounded Monotonic Memory for Hardware Verification Agents**：

1. 用结构化验证状态表示轨迹：stage、collected tests、passed set、failed set、CK coverage、bug evidence。
2. 用可验证的状态差定义 progress、diagnostic、no-progress、regression 和 invalid，而不是仅看最终成功。
3. 只把验证器认证的 progress/diagnostic Episode 写入跨 DUT 策略库；检索时同时匹配失败签名、修改对象和回归风险。
4. 把当前契约、锁定通过集合和失败签名作为不可压缩记忆；其余历史由事件驱动摘要。
5. 对高风险修改实施预执行风险门控，必要时分配 TRACE 式额外候选/验证预算。

当前代码已完成第 1 点的一部分和第 2 点的确定性基础，但第 3 至第 5 点仍需要实现和消融实验。近期最值得优先做的是完整 DUT bug certification chain 和写文件前的测试集合差分风险门控；它们比继续添加 Stage-specific Prompt 更有研究价值，也更容易形成可测量的独立贡献。
