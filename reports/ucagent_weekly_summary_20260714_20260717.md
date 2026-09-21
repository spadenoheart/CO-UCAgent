# UCAgent 周工作与 Adder v2.1 实验分析（2026-07-14 至 2026-07-17）

## 1. 最新 Adder 运行结论

本次完整运行位于 `log/Adder+20260717_104648`，使用
`context_reuse_credit_v2_rule_v2_curated_preview.json`、Prompt v1 和
Verifier-grounded utility gate（阈值 0.05）。运行完成全部 26 个阶段，最终
Checker 执行 52 个测试、覆盖 27 个检查点并记录 1 个 DUT bug。

| 指标 | 结果 |
|---|---:|
| 总耗时 | 3 h 40 min 19 s |
| 输入 / 输出 token | 1.30 M / 32.7 K |
| 主模型请求 | 474，0 个错误 |
| TTFT p50 / p95 | 10.12 s / 81.94 s |
| context-reuse 查询 / 注入 | 32 / 32 |
| 注入条目 | 85 |
| utility 候选通过 / 拒绝 | 188 / 47（通过率 80.0%） |

不能把该结果解释为 v2.1 提速。历史完整 Adder 运行约为 2 h 52 min 至
3 h 03 min，本次慢约 20% 至 27%，但运行 seed、服务尾延迟和代码版本并未
严格配对，因此这只是退化信号，不是因果结论。当前 TTFT 中位数与历史运行
接近，说明总耗时增加不能只归因于模型推理变慢；stage 19 和 21 的修复循环
仍是主要流程瓶颈。

| stage | 名称 | 耗时 | Checker 失败次数 |
|---:|---|---:|---:|
| 16 | evaluate_env_fixture | 20.6 min | 5 |
| 19 | basic_api_functional_test | 39.9 min | 10 |
| 20 | create_test_case_templates | 12.5 min | 0 |
| 21 | test_case_implementation_in_batch | 57.7 min | 10 |
| 22 | comprehensive_verification_and_bug_analysis | 19.0 min | 6 |
| 24 | generate_random_test_cases | 15.2 min | 2 |
| 25 | verification_review_and_summary | 13.2 min | 1 |

积极结果是 stage 20 未再次出现两小时停滞，表明此前 Checker 契约和 Prompt
阶段化约束至少没有破坏该阶段，且本次 fail_count 为 0。未解决的问题集中在
stage 19/21 的测试修复循环和 stage 22 的 bug 文档一致性循环。

## 2. v2.1 策略库效果

`Adder_20260717_v21_utility_gate_decisions.json` 将注入与下一次验证结果连接。
24 次可评估注入中，规则得到 5 次 diagnostic、11 次 no_progress、6 次
regression、2 次 invalid、0 次 immediate progress。该分析是观察性归因，且
当前规则对延迟推进较保守，不能把每个结果都归因给注入；但 25% 的观察回归率
仍然说明策略准入过松。

主要失配包括：

1. `reference_files_unread` 检索到修改测试代码的 Episode，而正确动作只是先读取
   Checker 要求的参考文件。
2. `generic_checker_failure` 依靠同阶段或文本相似度匹配到具体 API/测试修复，缺少
   足够的错误身份信息。
3. bug 文档的 passed-TC、incomplete-TC 和通用 schema 错误只因属于同一大类而
   相互匹配，导致一种 schema 错误被替换成另一种。
4. stage 24 的 bug 文档重复标签错误匹配到 stage 2 的 specification 文档修改；
   历史效用为正，但当前动作对象不一致。
5. 低支持的 `delete_file/write` 整文件改写策略被注入，风险未被 utility 分数充分
   反映。

归因规则本身也尚未通过预设验收门槛。40 条双人独立标注的时间留出集上，v2.1
准确率为 0.80、macro-F1 为 0.8205、标注者 Cohen's kappa 为 0.9659；其中
regression F1 为 0.9524，但 progress recall 只有 0.4167。因此当前 regression
告警相对可信，而“0 progress”可能存在低估。目标 0.90 准确率尚未达到。

## 3. 本次立即完成的 v2.2 修复

本次没有针对 Adder 策略 ID 建黑名单，而是在运行时增加可消融的通用语义准入：

- 流程证据型失败（当前为 `reference_files_unread`）禁止注入文件修复 Episode。
- `generic_checker_failure` 只有失败签名完全一致时才能复用。
- bug 文档高风险子类型要求 failure pattern 精确一致。
- 当前 verifier 证据推断的动作类别必须与 Episode 修改对象一致。
- 低支持的 `delete_file/write` 只有失败签名一致时允许进入 Prompt。
- compression hint 必须同签名，或同阶段且同精确 pattern；默认禁止 hints-only 注入。
- 新增 hard-gate 查询数、拒绝候选数及拒绝原因日志。

在本次 32 次历史注入上的反事实回放中，新门控改变 23 次、完全抑制 15 次，
Prompt 条目从 85 降至 42，减少 50.6%。该结果只证明门控能挡住已发现的失配，
不证明真实运行会提速。相关 55 个单元测试全部通过。候选快照位于
`benchmark/ucagent_experiments/20260717_v22_semantic_gate_candidate/`。

## 4. 本周完成工作

1. **先修正实验标签可信度。** 将测试结果明确区分为 pass、test_failure、
   infrastructure_error、timeout、crash 和 no_tests_collected，禁止“未执行测试”
   被写成成功 Episode，并冻结可复现实验快照。
2. **从终局成功启发式升级为角色化 Episode 归因。** Episode 包含
   failure_before、action_sequence、observation_after、stage_advanced、
   failure_set_delta、regression_count、information_gain、action_cost、
   credit_role 和 confidence；完成 40 条平衡样本的独立标注、规则修订和新的
   时间留出盲测。
3. **把策略质量接入运行时决策。** 实现历史 verifier utility、Prompt token 成本
   扣除、结构化注入决策日志和运行后 observational credit；最新完整 Adder 运行
   暴露了“历史效用高但当前语义不适配”的问题，并据此形成 v2.2 语义准入门控。

## 5. 与近期文献的关系

| 工作 | 可借鉴思想 | 当前 UCAgent 的关系与差距 |
|---|---|---|
| [TRIAGE (2026)](https://arxiv.org/abs/2606.32017) | 将行动分成 progress、exploration/diagnostic、no-progress、regression，再做局部信用分配 | 已复用“角色化归因”思想，但当前是 verifier 规则和离线策略筛选，不是 RL 训练；需先把盲测准确率提高到验收线。 |
| [Beyond Similarity / MemGate (2026)](https://arxiv.org/abs/2606.06054) | 相似记忆不等于适合当前任务，检索后需要 query-conditioned admission gate | v2.2 是面向芯片验证失败/动作契约的可解释规则门控；尚未学习化，但比单纯加权相似度更贴合当前退化原因。 |
| [TRACE (2026)](https://arxiv.org/abs/2606.11119) | 在固定预算下把 rollout 分配到更有信息量的中间前缀，形成树状探索 | 当前 trace tree 已提供节点结构，但尚未实现预算分配；应在策略污染受控后再做，否则会扩大错误分支。 |
| [DIM-WAM (2026)](https://arxiv.org/abs/2606.27677) | 多尺度历史事件、阶段进度和不同 memory bank 共同影响后续行动 | 当前结构化事件和 stage-aware 检索对应其思想层面；尚无学习的世界模型或 progress supervision。 |
| [VerificAgent (2025)](https://arxiv.org/abs/2506.02539) | 对轨迹记忆进行事实核验和部署前净化 | UCAgent 用可执行 Checker 替代纯人工事实核验，潜在优势是领域 verifier 更强；尚缺严格的跨 DUT 成功率实验。 |
| [Agent Workflow Memory (2024)](https://arxiv.org/abs/2409.07429) | 从历史轨迹抽取可复用 workflow 并选择性注入 | 可作为“通用 workflow memory”基线；UCAgent 的研究差异应是 failure-conditioned、verifier-grounded、role-typed 和 cost-aware。 |

较合适的论文主线不是“实现了一个策略库”，而是：**将相似度驱动的 workflow
memory 改造成 verifier-grounded、role-typed、utility- and risk-gated repair
memory，并验证其能否降低长流程芯片验证 Agent 的无效修复与回归。**

## 6. 下周对比实验

### 第一阶段：低成本 checkpoint replay

从 stage 16/19/21/22/24 保存的失败状态中选 20 至 30 个 checkpoint，每个状态在
相同输入和预算下比较：

- B0：Prompt v1，关闭 context reuse。
- B1：质量阈值 + 加权相似度检索，不使用 utility/hard gate。
- B2：v2.1 historical utility gate，不使用 semantic hard gate。
- B3：v2.2 utility + semantic/risk gate。

主指标为固定 K 轮内 verifier progress rate、regression rate、invalid rate、
stage advance、失败集合净减少、LLM token、修复动作数和 wall time。报告均值、
中位数、bootstrap 95% CI，并保留同一 checkpoint 的配对差值。优先证明 B3
相对 B1/B2 降低回归，而不是先追求总耗时显著性。

### 第二阶段：归因规则再验证

从新 Adder、uart_tx、FSM/ALU754 轨迹中抽取新的 40 至 60 条时间与 DUT 均隔离的
Episode，双盲标注。验收线：accuracy >= 0.90、progress precision >= 0.90、
invalid-to-progress = 0；同时单独报告 progress recall，避免规则通过“全部判为
no_progress”获得表面准确率。

### 第三阶段：端到端实验

checkpoint 结果成立后，只比较 B0、B1 和 B3。最低配置为 Adder 3 个 seed，另选
uart_tx 与 FSM 各 1 至 3 个 seed验证跨 DUT；论文级配置应达到每个 DUT/方法至少
3 个 seed。端到端主指标为 DUT 完成率、总时间、stage 19/21/22 时间、失败循环数、
输入/输出 token 和 context-reuse precision。模型、GPU、Prompt、温度、DUT 和
超时必须固定。

当前不建议立即加入 TRACE 式分支预算分配。先证明 B3 的记忆准入质量，否则树状
扩展只会把错误 Episode 的影响复制到更多分支。若 B3 在 checkpoint replay 上
显著降低回归，下一步再将 utility 用于“是否继续当前修复分支/是否回退并探索新
分支”的预算决策，才形成从 memory gate 到 trace allocation 的完整研究链。
