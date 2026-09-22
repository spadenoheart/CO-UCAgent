# CO-UCAgent：面向本地大模型芯片验证的证据驱动 Context Engineering

CO-UCAgent is a verification-aware agent harness for evidence-grounded context
engineering with local LLMs.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](pyproject.toml)
[![Release: v0.2.0](https://img.shields.io/badge/Release-v0.2.0-2ea44f.svg)](CHANGELOG.md)
[![Upstream: UCAgent](https://img.shields.io/badge/Upstream-UCAgent-6f42c1.svg)](https://github.com/XS-MLVP/UCAgent)

**项目入口：** [快速开始](#快速开始) · [七项开源工具](#4-open-source-toolkit) ·
[工具命令手册](TOOLKIT.md) · [实验结果](#6-results) ·
[复现说明](#9-artifact-and-reproducibility)

## 摘要

大语言模型 Agent 正在被用于规范理解、验证环境构建、测试生成和缺陷归因。与单轮代码
生成不同，完整芯片验证会持续数小时，并在规范、RTL、测试、覆盖率和 Checker 反馈之间
形成不断增长的异构上下文。本地部署进一步放大了这一问题：有限的推理吞吐使重复预填充
和无效修复具有较高代价，而测试未执行、基础设施异常和验证范围变化又会使错误经验进入
摘要或长期记忆。现有上下文压缩与相似度检索难以同时保证信息效率、证据可靠性和跨任务
适用性。

本文提出 CO-UCAgent，一个面向本地大模型芯片验证的 verification-aware agent harness。
其核心方法由可执行验证反馈驱动。Harness 将代码修改、测试执行、Checker 反馈和阶段
迁移规范化为具有来源的结构化证据，并在每次模型调用前从阶段契约、近期交互、层次化
状态和长期记忆中构造工作上下文。对于跨任务经验，CO-UCAgent 从轨迹中提取修复片段
（repair episode），根据失败集合变化、阶段推进、诊断信息和回归风险评估其作用，仅在
历史行动与当前失败具有可验证适用性时将其用于后续推理。该设计将 Prompt、memory、
tool feedback 和运行时控制纳入同一个 Context Engineering 闭环。

同 seed 的 clean Adder 实验中，CO-UCAgent 将 30-Stage active time 从 12h59min 降至
5h37min（-56.7%），Checker 失败事件由 70 次降至 41 次；HPerfCounter 的运行
耗时下降 27.0%，Prompt token 下降 37.4%。FSM、ShiftRegister 和 Mux 的实验结果分别观察到 67.8%、68.5% 和 59.0% 的 active-time 降幅。验证差分归因在两个包含
40 条样本的人工标注集上均取得 0.80 accuracy，经验适用性过滤使历史回放中的 Prompt
条目减少 50.6%。ALU754 的 checkpoint-composed 完成链仍高于 baseline，说明现有机制
已经在部分 DUT 上形成明显收益，但跨 DUT 稳定性仍需通过更多重复实验验证。

**关键词：** 芯片验证智能体；本地大语言模型；Context Engineering；Agent Harness；
过程记忆；验证反馈

## 快速开始

CO-UCAgent 支持任意 OpenAI-compatible 对话模型服务。芯片仿真示例还需要 Python 3.11、
Verilator、C++ 工具链和 [Picker](https://github.com/XS-MLVP/picker)；Dockerfile 基于
Picker 官方镜像，适合希望复用完整验证环境的用户。

```bash
git clone https://github.com/spadenoheart/CO-UCAgent.git
cd CO-UCAgent

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

配置本地模型。下面以 Ollama 的 OpenAI-compatible 接口为例，模型名称可以替换为服务端
实际提供的模型：

```bash
export OPENAI_MODEL='mdq100/qwen3.5-coder:122b-64k'
export OPENAI_API_KEY='ollama'
export OPENAI_API_BASE='http://127.0.0.1:11434/v1'
```

运行仓库内的 Adder 示例。`run_Adder` 是无交互、可记录日志的公开运行入口；
`test_Adder` 启动 TUI 与人工交互模式。

```bash
make run_Adder
# 或：make test_Adder
```

也可以绕过 Makefile，直接使用安装后的 CLI：

```bash
make init_Adder CWD=output/adder_demo
co-ucagent output/adder_demo Adder --config config.yaml \
  --interaction-mode standard --loop --exit-on-completion --stream-output
```

安装会同时提供七项研究工具。以下命令可查看稳定子命令；完整示例见
[TOOLKIT.md](TOOLKIT.md)。

```bash
co-llm-profiler --help
co-context --help
co-memory-cache --help
co-trace --help
co-strategy --help
co-bench --help
co-trajectory-data --help
```

## 1. Motivation

### 1.1 本地大模型使芯片验证成为系统问题

现代芯片验证包含规范理解、接口建模、测试生成、仿真执行、覆盖率分析与缺陷归因等相互
依赖的任务。基于大语言模型的智能体能够调用文件、仿真和检查工具，并在多阶段工作流中
迭代生成验证环境与测试程序。然而，与代码补全或单轮 RTL 生成不同，完整验证任务通常
持续数小时，包含数百次模型调用和工具交互，并反复引用规范、RTL 接口、历史修改与测试
结果。此时，模型能力只是系统性能的一部分；上下文如何形成、验证、压缩和复用同样决定
智能体能否完成任务。

本地部署是这一场景中的重要系统约束。长时智能体可能产生百万级输入 token，商业 API
费用会随验证轮次和重复上下文持续增长；未公开 RTL、设计规范、缺陷记录和内部验证脚本
又通常不适合离开组织的数据边界；企业验证环境还要求模型版本、推理参数、工具接口与
运行日志可冻结、可审计且可离线复现。因此，本地模型不仅用于降低外部服务依赖，也为
数据治理和实验控制提供基础。

本地部署同时将计算瓶颈直接暴露给 Agent 工作流。122B 级模型在多 GPU 环境中的首
token 延迟和解码吞吐会放大每一次无效调用；较小模型虽然推理更快，却更容易违反端口、
API、检查点和文档 schema 等验证契约。单纯提高 TPS 只能缩短一次调用，无法消除同一
错误上的重复修复；单纯缩短 Prompt 则可能删除决定行动正确性的证据。由此产生的核心
系统问题是：**如何在有限的本地推理预算下，提高每个 token 和每次行动产生有效验证
进展的概率。**

### 1.2 长流程瓶颈来自上下文质量，而非上下文长度本身

给定 DUT 规范、RTL 实现、当前验证阶段和历史交互轨迹，智能体需要选择下一步修改或
验证行动。其上下文至少包含四类信息：相对稳定的规范与接口契约、随阶段变化的验证目标、
近期失败与修改记录，以及可能跨任务复用的历史经验。这些信息具有不同的有效期、粒度和
可信度，不能由统一的 token 截断策略处理。

保留完整历史会重复预填充规范、代码和工具输出，并使关键错误被大量已过期信息稀释。
自由文本摘要虽然能够降低长度，却可能丢失端口名、函数签名、检查点层级和断言结果。
长期记忆能够恢复跨阶段信息，但若写入条件只依赖“最终成功”，诊断步骤、偶然成功和
回归行动也会被作为正向经验保存。语义检索能够找到文本相似的错误，却无法保证历史行动
适用于当前阶段、文件类型和验证契约。因此，问题不在于智能体是否拥有摘要或记忆，而在
于这些上下文是否保留了必要证据，以及是否应在当前决策中出现。

这一观察将 Context Engineering 从单一压缩问题扩展为三个相互依赖的问题：

1. **表示：** 如何用适合验证流程的结构保存规范、阶段状态、失败和修改，而不反复携带
   原始对话？
2. **可信度：** 如何判断测试与 Checker 观察是否真正执行、验证范围是否可比，以及
   某次行动是否获得了有效信用？
3. **选择：** 如何根据当前失败和风险选择摘要、近期证据与跨 DUT 经验，而不是无条件
   注入所有相似内容？

### 1.3 经验性观察揭示了现有方法的失效模式

对 Adder、uart_tx、IntegerDivider、FSM 和 ALU754 等 DUT 的长时轨迹分析显示，性能
损失并非均匀分布在所有阶段，而是集中于少量持续失败的修复循环。这些循环呈现四类可
跨 DUT 观察的失效模式。

**无效验证污染上下文。** `tests_total=0`、collection error、进程崩溃或超时曾可能被
聚合为“全部测试通过”。一旦这种观察进入摘要或长期记忆，后续系统就可能将未被验证的
修改作为成功经验复用。

**终局成功掩盖行动差异。** 一个最终成功的轨迹通常混合了定位根因、错误修改、回退、
最小测试和真正修复。将整条轨迹赋予统一正信用，无法区分产生进展的行动与仅增加成本的
行动。

**相似错误需要不同修复对象。** Checker 消息可能具有相似措辞，但错误可能分别来自
测试代码、API 封装、bug 文档 schema 或未读取的规范。只依赖文本相似度会把一种修复
策略迁移到另一种错误，造成 negative transfer。

**更多记忆不必然带来更高效率。** 历史实验中，单独启用长期记忆可能增加输入 token，
而过严的经验过滤虽然减少注入，也没有自动改善端到端耗时。这说明优化目标不能是记忆
数量或命中率，而应是注入后产生验证进展且不引入回归的条件概率。

### 1.4 Research Gap 与设计要求

现有长上下文压缩方法主要优化通用问答中的 token 保留，通用 Agent memory 主要关注
信息召回或工作流复用，而 Agent credit assignment 则通常依赖训练阶段的终局奖励。
芯片验证需要把三者连接到可执行的验证器信号：上下文项应当具有来源，历史行动需要由
测试或 Checker 证据归因，跨任务记忆必须在修改对象和验证契约上与当前失败兼容。

据此，CO-UCAgent 遵循四项设计要求：

- **Evidence grounding：** 所有可复用成功与修复经验都必须连接到有效测试、Checker
  观察或阶段迁移，未执行验证不能获得正信用。
- **Task-aware compression：** 压缩必须区分稳定契约、阶段摘要、近期失败和工具噪声，
  而不是只依据文本位置或长度。
- **Utility-aware reuse：** 经验选择不仅考虑相似度，还应考虑验证收益、动作成本、回归风险
  和当前修改对象的兼容性。
- **End-to-end measurability：** 系统必须同时观测 token、推理延迟、失败循环、阶段推进
  和回归，避免用单一检索命中率代替任务收益。

本文围绕以下研究问题展开：

- **RQ1：** 任务感知的 Context Engineering 能否在保留验证契约的同时减少重复 token
  与失败循环？
- **RQ2：** 验证结果归一化和局部差分能否可靠识别有效修复、诊断、无进展与回归行动？
- **RQ3：** 风险约束的跨 DUT 经验复用能否提高有效注入率并降低 negative transfer？
- **RQ4：** 上述机制能否在固定本地模型与硬件条件下提高完成率并降低端到端成本？

## 2. Contributions

本文作出以下贡献：

1. **提出验证反馈驱动的 Context Engineering Harness。** CO-UCAgent 将 Prompt 构造、
   状态压缩、长期记忆和跨任务经验复用统一到可执行验证闭环中，使 Context Engineering
   不再依赖未经校验的自然语言历史。
2. **设计多时间尺度、可追踪的验证状态。** 结构化事件和六类测试终态为上下文提供可靠
   证据；近期窗口、结构化摘要、Batch/Stage 状态、failure-aware context 和阶段预取在
   不同时间尺度上维护模型工作状态。
3. **提出验证条件化的跨 DUT 过程记忆。** 系统根据前后验证差分对 repair episode 进行
   角色化归因，并联合失败签名、阶段职责、修改对象、跨任务支持度、历史效用与回归风险
   判断经验是否适用于当前失败。
4. **建立面向长时本地 Agent 的评测与复现基础。** 指标体系同时覆盖任务完成、推理成本、
   行动质量、归因准确率和记忆污染；配套工具支持精确 Prompt dump、性能采集、checkpoint
   replay 和实验配置冻结。

## 3. System Design

### 3.1 框架概述

CO-UCAgent 将语言模型与其运行基础设施明确分离。模型负责基于当前输入进行推理并选择
工具；Agent Harness 则负责维护任务状态、暴露文件与验证工具、执行阶段转换、构造每轮
模型输入并记录运行证据。CO-UCAgent 在保持模型参数不变的条件下扩展 Harness，因此
同一套方法可以作用于不同规模或不同部署方式的本地模型。

<p align="center">
  <img src=".github/assets/co-ucagent-framework.svg" width="100%" alt="CO-UCAgent 系统架构图">
  <br>
  <strong>图 1：CO-UCAgent 的系统架构。</strong>实线表示在线的上下文构建与验证闭环，
  虚线表示运行后的经验归因、过程记忆更新与后续检索。
</p>

图 1 左侧给出三类原始证据。设计契约描述 DUT 规范、RTL 端口和验证 API；阶段目标规定
当前步骤允许修改的对象与完成条件；运行时轨迹保存模型行动及测试、Checker 返回的环境
观察。Context Engine 不直接拼接这些原始内容，而是依次完成证据规范化、多时间尺度状态
维护和验证条件化的经验选择，最终在给定 token 预算内形成模型工作上下文。模型据此生成
工具调用，Harness 执行修改或验证，并把新的可执行结果送回 Context Engine，形成在线
闭环。

图中的跨 DUT 过程记忆位于在线循环之外。一次运行结束后，系统把“失败观察—行动序列—
后续验证”切分为可审计的修复片段，并根据验证差分评估其作用。通过质量筛选的片段被合并
到过程记忆；后续运行只有在失败语义、阶段职责和修改对象均匹配时才会检索这些经验。因而，
该模块区别于保存用户事实或对话片段的通用记忆，更接近由验证器反馈监督的 procedural
memory。

这一架构包含两个相互补充的研究对象。**Context Engineering** 研究每次推理前应向模型
提供哪些信息、如何转换这些信息以及如何控制其规模；**Agent Harness** 提供工具接口、
持久状态、验证反馈和执行控制，使 Context Engineering 能够获得可靠信号并影响后续行动。

### 3.2 验证感知的结构化事件

Context Engineering 的可靠性首先取决于输入证据是否可信。自然语言运行日志通常混合
模型解释、工具输出和控制信息，难以判断某段文本对应哪一次修改，也无法区分功能失败与
测试进程异常。CO-UCAgent 因此在 Harness 内建立独立的事件流，将行动和观察记录到
`structured_events.jsonl`。每条事件包含 DUT、线程、模型轮次、阶段编号、时间戳和事件
类型，并使用这些字段恢复局部的行动—观察关系。

事件模型覆盖三类对象。`file_mutation` 描述写入、追加、局部替换、复制、移动和删除等
操作，记录目标路径、文件类别、修改规模与内容摘要，而不在事件流中重复保存完整代码。
`test_run` 和 `check_result` 描述测试或 Checker 实际验证了什么，包括测试选择范围、
通过与失败集合、错误类别、阻塞项和执行耗时。`stage_transition` 则记录阶段是否推进，
防止把“测试数量下降”与“工作流目标完成”混为同一种进展信号。

原始工具返回在进入事件流之前经过结果归一化。测试终态被划分为六个互斥类别：

**表 1：测试执行结果的规范化终态。**

| 终态 | 含义 |
|---|---|
| `pass` | 测试被实际收集并全部通过 |
| `test_failure` | 测试执行完成且存在断言或功能失败 |
| `infrastructure_error` | 收集、依赖或运行环境异常 |
| `timeout` | 测试在预算内未完成 |
| `crash` | 测试进程非正常退出 |
| `no_tests_collected` | 未收集到可执行测试 |

`pass` 要求至少收集并实际执行一个测试，且不存在测试失败或运行异常。其他五类结果仍然
是有意义的环境观察，但不能证明此前修改正确。例如，`infrastructure_error` 可以提示依赖
或导入问题，`no_tests_collected` 可以暴露测试发现错误；二者都不应获得成功信用。这一
区分阻止了“没有运行测试”等价于“所有测试通过”的污染路径。

连续两次验证还需要满足**验证范围可比性**。系统根据测试选择集合将前后观察标记为
`same`、`narrowed`、`expanded`、`disjoint` 或 `incompatible`。只有在相同范围内，或在
能够明确映射失败集合的范围变化下，失败数量差才具有直接意义。例如，从完整测试集切换到
单一测试得到零失败，只能说明该目标测试通过，不能推断其他失败已经被修复。结构化终态与
范围关系共同构成后续归因和记忆写入的证据基础。

### 3.3 多时间尺度状态管理与动态上下文构建

在第 $t$ 次模型调用前，Context Engine 从五类来源构造工作上下文：设计与 API 契约、
当前阶段状态、近期行动—观察窗口、跨阶段长期状态，以及与当前失败相关的过程经验。这里的
“构造”不仅是文本拼接，还包含选择、结构转换、去重、排序和预算分配。其目标是在不丢失
当前决策所需约束的前提下，减少对早期原始消息和重复工具输出的依赖。

#### 3.3.1 近期工作状态与结构化压缩

系统始终保留最近若干条模型消息和工具观察，保证模型能够看到导致当前状态的直接行动。
当历史长度超过预算时，较早消息不会被简单截断，而是转换为固定 schema 的状态摘要。
该 schema 包含 `stage_info`、`test_report`、`coverage_status`、`bug_tracking` 和 `next`
五个区域，分别保存阶段契约、最近测试统计、覆盖率缺口、缺陷证据和未完成事项。端口、
测试用例、检查点与报告路径以短引用保留，原始代码块和重复 pytest 输出则被删除。

结构化摘要的输出还经过格式与内容保护。若摘要模型返回工具调用、空内容或不完整 JSON，
Harness 会尝试规范化字段，并从最近的 Stage/Batch/Check 观察中恢复可确定的信息；无法
恢复的字段保留为空，而不是由模型补写未知事实。摘要调用使用独立的性能统计，并可配置
较小的摘要模型；当摘要模型不可用或输入超出其上下文限制时，系统回退到主模型。

#### 3.3.2 Batch、Stage 与失败状态的分层表示

单轮摘要解决消息长度问题，但不足以表达验证任务的层次结构。CO-UCAgent 进一步维护
Batch 和 Stage 两种聚合状态。Batch 状态对应一次测试批次，保留测试总数、失败用例、
失败检查点和报告位置；Stage 状态汇总同一阶段内的多个 Batch，记录阶段完成度以及尚未
解除的阻塞项。当工作流进入新阶段时，上一阶段的低层工具输出可被 Stage 状态替代，而
接口约束和未解决失败仍然保留。

Failure-aware state 用于处理局部修复循环。系统把失败、混合结果和可信成功检查点分开
存储，并优先选择与当前阶段一致的少量条目。模型因此能够同时看到“当前仍然失败什么”、
“最近修改了什么”和“最后一个可信状态是什么”，而不会被更早的同签名错误反复占用
上下文。这个缓冲区服务于短程因果连续性，不承担跨运行知识复用。

#### 3.3.3 长期状态与阶段预取

跨阶段信息由持久化长期记忆维护。候选条目首先根据事件类型、测试状态、失败证据完整性
和写入质量计算保留分数；低于阈值的条目只进入候选记录，不直接参与 Prompt。相似条目
按内容和阶段信息合并，并通过多次支持逐步从 candidate 提升为 episode 或 semantic
memory，避免单次偶然结果立即获得较高权重。

进入阶段时，Harness 根据 DUT、阶段标题和阶段编号预取少量相关条目。检索分数综合词项
重合、可选 embedding 相似度、阶段距离、历史支持度、已观测 useful/pollution 次数和
staleness；相同阶段的检索结果使用版本化缓存，减少每轮重复搜索。预取只是候选准备，
实际写入 Prompt 仍受最低分数、相对分数和来源多样性约束。下一次验证完成后，系统根据
失败是否减少或原签名是否消失更新 useful/pollution 统计，使长期状态具有运行后反馈。

#### 3.3.4 阶段契约与最终组装

Prompt v1 为规范分组、检查点设计、Bundle/API 构建、测试实现、bug 文档和随机测试等
阶段定义各自的操作契约。阶段契约规定当前目标、允许依赖的接口、必须满足的断言与文档
schema，并把最近 Checker 阻塞项压缩为可执行的 blocker summary。这些约束属于稳定的
控制上下文，不会在每次摘要时被重新生成。

最终工作上下文按照“阶段契约与必读证据、近期行动—观察、结构化状态、相关长期状态、
修复经验”的顺序组装。每类信息具有独立数量上限和来源标签，模型能够区分当前事实与历史
类比。该过程可概括为：稳定契约用于限定行动空间，近期窗口维持局部因果，层次化状态保持
跨阶段连续性，过程经验提供可验证的修复先验。它们共同组成模型当轮可见的 working state。

### 3.4 基于验证差分的 Repair Episode 归因

语言模型 Agent 以“行动—观察”循环运行：模型修改文件或调用工具，环境返回新的测试或
Checker 结果，模型再据此决定下一步。完整 DUT 轨迹可能包含数百个循环，无法作为一个
整体直接写入记忆。CO-UCAgent 将其中具有局部因果边界的片段定义为 **repair episode**。

一个 episode 从可信失败观察开始。例如，观察 $o_i^-$ 表示当前测试集合中 A、B 两个
用例失败；随后模型读取 API、修改测试并运行最小目标，这些连续行动构成
$a_{i:j}$；下一次有效测试结果 $o_i^+$ 表示 A 已通过而 B 仍失败。系统将这三部分记为
$e_i=(o_i^-, a_{i:j}, o_i^+)$。上标 “-” 和 “+” 分别表示行动序列之前和之后，并不
表示负样本或正样本。若后一次测试没有正常执行，$o_i^+$ 仍会被记录，但该片段不能获得
修复成功信用。

系统从这个基本转移中派生用于归因的属性。`failure_before` 保存行动前的失败签名与验证
范围；`action_sequence` 保存工具类型、修改对象和行动顺序；`observation_after` 保存
规范化测试终态及 Checker 结果。`failure_set_delta` 将失败划分为已消除、仍保留和新引入
三部分；`stage_advanced` 表示 Harness 是否确认阶段推进；`information_gain` 表示行动
是否产生新的根因、接口位置或最小复现证据；`action_cost` 近似修改规模、工具调用和验证
时间；`confidence` 则由结果完整性和前后验证范围可比性确定。

归因的目的不是评价模型生成文本的语言质量，而是回答“这段行动在验证流程中起到了什么
作用”。根据上述局部证据，每个 episode 被分配一个 credit role：

**表 2：Repair episode 的 credit role 及其记忆语义。**

| 角色 | 判定语义 | 在过程记忆中的用途 |
|---|---|---|
| `progress` | 在可比较验证范围内消除失败，或由 Harness 确认阶段完成 | 可作为候选修复经验 |
| `diagnostic` | 尚未减少失败，但定位了新的根因、接口或可复现条件 | 可作为诊断经验，不能表述为已修复 |
| `no_progress` | 核心失败集合与已知信息基本不变 | 不进入高质量过程记忆 |
| `regression` | 引入新失败、破坏既有通过项或违反新的验证契约 | 作为风险证据保留 |
| `invalid` | 测试未执行、崩溃、超时或前后范围不可比较 | 排除正向信用 |

角色判定优先使用强证据。有效阶段推进或可比较范围内的失败净减少支持 `progress`；没有
直接修复但产生新的可操作证据时判为 `diagnostic`；新增失败优先判为 `regression`；缺少
有效后验观察时判为 `invalid`。只有其余情况才进入 `no_progress`。这种局部归因避免把
最终完成 DUT 的奖励平均分配给轨迹中的每个中间行动，也为人工抽查提供了明确证据入口。

### 3.5 跨 DUT 经验抽取与风险约束检索

过程记忆的目标不是保存某个 DUT 的具体补丁，而是提取能够跨设计迁移的“失败模式—行动
类型—验证方式”关系。离线构建首先把 Checker 类别、异常文本、失败用例和失败检查点
规范化为 failure pattern，并将修改路径映射为测试代码、覆盖率代码、API/环境、规范文档
或 bug 文档等 action category。随后，系统按规范化失败签名、阶段职责和行动模式合并
相似 episode，累计其 DUT 来源、支持次数、progress/diagnostic 证据和回归记录。只有
具有有效后验观察且满足质量阈值的片段进入可检索集合；`invalid` 和低置信度片段保留用于
审计，但不会被当作正向修复经验。

在线阶段只在 `RunTestCases`、`Check` 或 `Complete` 返回当前阶段的失败后触发经验查询。
候选生成综合精确 failure pattern、阶段角色、词项重合、同 DUT 偏好、跨 DUT 支持度和
历史质量。该排序用于提高召回，但排名靠前并不意味着经验适用于当前任务。因此，系统在
候选排序之后执行 verifier-conditioned applicability filtering，其判定维度如表 3 所示。

**表 3：历史经验相对于当前失败的适用性判定。**

| 适用性维度 | 判定问题 | 对经验选择的影响 |
|---|---|---|
| Evidence readiness | 当前失败是否首先要求读取规范或建立缺失证据？ | 在证据前置条件未满足时，抑制代码修改经验 |
| Failure semantics | 历史与当前错误是否属于相同的具体契约？ | generic Checker 错误和 bug 文档子类型要求更严格的签名一致性 |
| Action-target compatibility | 历史行动修改的对象是否与当前错误来源一致？ | API、测试、覆盖率与文档经验不能仅因文本相似而互换 |
| Support and mutation risk | 该行动是否具有跨轨迹支持，且是否包含整文件写入或删除？ | 对低支持、高影响修改要求更强证据，否则降低优先级或拒绝 |
| Historical verifier utility | 该经验过去被使用后是否带来进展，是否曾产生污染？ | 提高具有稳定 useful hit 的候选分数，降低 regression、pollution 与 stale episode 的优先级 |

上述判定把“语义相关”与“行动可迁移”分开处理。例如，`reference_files_unread` 描述的
是缺失工作流证据，正确下一步是读取指定文件，而不是复用某个历史代码补丁；两个 bug
文档错误虽然都包含相似标签，却可能分别要求删除已通过用例和补全缺失 TC，其行动不能
互换。对于整文件 `write` 或 `delete_file` 等高影响操作，系统还要求更高的签名一致性和
历史支持度，从而降低跨 DUT 迁移造成大范围回归的概率。

通过适用性判定的 episode 被压缩为四部分：触发该经验的失败模式、可迁移的行动原则、
应执行的最小验证以及需要避免的风险。DUT 专有信号和值不会作为可直接复制的结论；Prompt
明确要求模型先核对当前文件。相同失败签名只注入有限次数，相同阶段/模式也具有总量限制，
防止失败循环反复追加相同经验。缺少具体行动的 compression hint 默认不能单独注入。

每次查询都会记录候选、分数分解、适用性拒绝原因、最终注入内容和下一次验证结果。若后续
失败减少或原签名消失，相关 episode 增加 useful 统计；若失败持续或出现新回归，则增加
pollution/risk 统计。该反馈不会立即改变模型权重，而是更新后续 Context Engineering 的
经验选择先验。

### 3.6 性能观测与可复现实验支持

CO-UCAgent 同时观测模型推理与 Harness 行为，避免把端到端性能变化简单归因于模型 TPS。
推理侧记录每次调用的输入/输出 token、首 token 延迟、prompt evaluation 时间、decode
时间和估计吞吐；Harness 侧记录阶段耗时、工具调用、失败循环、摘要调用、memory query、
episode injection 及其后验结果。两类指标通过模型轮次和阶段编号关联，可以区分“单次
推理变慢”“Prompt 变长”和“Agent 采取了更多无效行动”等不同原因。

精确 `llm_input_messages` dump 保存实际发送给模型的消息序列，用于把长流程中的代表性
调用重放为独立延迟测试，也支持 Prompt v0/v1 和不同 Context Engineering 组件的受控
比较。实验冻结工具进一步保存配置、过程记忆快照和关键代码文件哈希，使一次运行能够追溯
到具体模型服务、Prompt 与检索策略。该机制既服务于性能分析，也为后续 checkpoint replay
和跨 DUT 消融提供一致输入。

## 4. Open-Source Toolkit

CO-UCAgent Toolkit 是在 UCAgent 基础上构建的一组芯片验证 Agent 工具，覆盖本地大模型
长时间运行时的性能分析、上下文管理、经验复用、执行审计和实验评测。七个模块从一次
模型请求延伸到完整多 DUT 实验，既可独立使用，也可组合成验证反馈驱动的闭环。

### 4.1 七项工具概览

| 工具 | 解决的问题 | 核心能力 | 已形成的代表性结果 |
|---|---|---|---|
| `co-llm-profiler` | 长耗时是否主要来自模型推理 | 真实上下文 TTFT/TPS 采集与回放 | 建立 10 个 843–102K token 的轨迹用例 |
| `co-context` | 长对话重复携带无效信息 | Verifier-aware 摘要、状态包和 observation masking | 被选中旧观察的字符量减少 97.0% |
| `co-memory-cache` | 历史经验难以持续复用 | 结构化记忆、阶段缓存、预取和质量反馈 | 区分命中、有效、陈旧和污染四类指标 |
| `co-trace` | Agent 执行过程不可解释 | 六类测试终态、轨迹树和实时可视化 | 阻止空测试和执行异常形成成功证据 |
| `co-strategy` | 相似轨迹可能产生负迁移 | 状态条件策略、动作归因和成对准入 | 构建 36 条结构去重的策略候选 |
| `co-bench` | 长实验难复现、难比较 | 冻结配置、断点回放、多 DUT 与 Token 统计 | 建立 8 个 DUT 的 clean baseline 结果 |
| `co-trajectory-data` | 真实 Agent 训练数据缺少来源审计 | 轨迹筛选、数据切分和 LLaMA-Factory 导出 | 生成 2,196 条带 provenance 的 SFT 样本 |

安装 `co-ucagent` 后，上表中的工具名就是可直接调用的命令。每个命令采用稳定的
“工具名 + 子命令”形式，并继续兼容 `python scripts/<name>.py` 的源码调用方式：

| 公开命令 | 子命令 | 典型输入与产物 |
|---|---|---|
| `co-llm-profiler` | `analyze`, `extract`, `replay` | Agent 日志、精确 Prompt dump、延迟结果 |
| `co-context` | `events`, `runtime` | 结构化事件、Stage/Token/耗时报告 |
| `co-memory-cache` | `inspect` | memory JSONL、缓存命中与污染统计 |
| `co-trace` | `build`, `visualize` | trace tree JSON、交互式 HTML/实时页面 |
| `co-strategy` | `build`, `curate`, `analyze`, `gate` | Repair Episode、策略库和成对准入报告 |
| `co-bench` | `run`, `freeze`, `stage` | 冻结 manifest、多 DUT ledger、阶段回放结果 |
| `co-trajectory-data` | `build`, `export` | 带 provenance 的 SFT JSONL 与 LLaMA-Factory 数据 |

命令参数、源码映射和端到端组合示例统一维护在 [TOOLKIT.md](TOOLKIT.md)，避免 README、
Python 脚本和安装后的命令入口发生偏移。

### 4.2 `co-llm-profiler`：真实 Agent 上下文性能分析

芯片验证 Agent 的输入包含规范、RTL 摘要、Python 测试、Checker 结果和历史工具输出。
短 Prompt 测得的吞吐不能代表真实运行成本。`co-llm-profiler` 在 Agent 内部记录主模型
与摘要模型的请求级指标，并将生产轨迹转换为独立推理用例。

**主要能力**

- 记录 latency、TTFT、Prompt/Completion token；
- 分别估算 prefill TPS 和 decode TPS；
- 区分主模型、fallback 和 summary 请求；
- 导出精确 LLM 输入消息和模型参数；
- 在 Ollama 兼容接口上重放相同类型的长上下文请求；
- 按请求、Stage 和完整运行统计性能分布。

当前基准包含 10 个来自真实 Adder 轨迹的上下文，覆盖约 843、10K、30K、50K 和 100K
token 等输入规模。测试揭示了接近上下文上限时 TTFT 和 decode 性能的显著退化，使
“模型慢”可以进一步分解为上下文长度、请求轮次和服务长尾问题。该工具可用于比较
Ollama、llama.cpp、vLLM 等推理后端，以及量化等级、GPU 数量和上下文长度。

**实现入口**

- `ucagent/abackend/langchain/message/performance.py`
- `scripts/analyze_llm_performance.py`
- `scripts/extract_llm_latency_cases.py`
- `scripts/run_llm_latency_cases.py`
- `tests/test_llm_performance.py`

### 4.3 `co-context`：面向验证状态的上下文编译器

长流程 Agent 会积累大量已经失效的测试输出和重复背景信息。完整携带历史会持续增加
推理成本；仅按长度截断则可能删除当前 Checker 契约或关键失败证据。`co-context` 将
对话历史转换为面向当前验证阶段的紧凑输入。

**主要能力**

- 结构化摘要：保留 Stage、待办、测试状态、根因假设和下一步动作；
- 分层摘要：分别维护全局、Stage 和测试 Batch 信息；
- 失败感知上下文：区分失败、成功和恢复期信息；
- 双阈值 Token 控制：高水位触发摘要，hard cap 保证最终请求不超过预算；
- Stage state package：集中保存当前契约、未决失败、最近验证和剩余动作预算；
- Observation masking：压缩已被最新状态取代的旧测试和 Checker 输出；
- 独立 summary model 与失败回退机制。

Observation masking 依据验证事件的新旧关系选择压缩对象。在已有轨迹中，573 次 masking
覆盖 3,250 条旧观察，被选择内容从 23.38M 字符压缩为 0.70M 字符，局部压缩率达到
97.0%，同时保留最近验证结果和当前 Stage 约束。状态包也为策略检索和控制器提供稳定
输入，降低不同模块重复解析自然语言日志的成本。

**实现入口**

- `ucagent/abackend/langchain/message/conversation.py`
- `ucagent/abackend/langchain/agent.py`
- `ucagent/stage/vmanager.py`

### 4.4 `co-memory-cache`：Cache-like 长期记忆

芯片验证经验具有明显的阶段性和重复性，例如接口绑定错误、测试集合错误、覆盖标记缺失
和 Bug 文档契约冲突。简单保存历史对话会产生大量重复与过期信息。`co-memory-cache`
将经验组织为结构化 memory line，并引入缓存系统中的准入、命中、晋级、预取和失效思想。

**主要能力**

- 从 Stage、测试和 Checker 事件生成结构化记忆条目；
- 依据内容哈希、失败集合和根因信息合并近似条目；
- 根据支持次数将记忆从 candidate 晋级为 episode 或 semantic memory；
- 组合词法、Embedding、Stage 接近度与历史支持度进行检索；
- 在进入下一 Stage 前预取，并在 Stage 内复用查询结果；
- 记录 useful、stale 和 pollution 反馈并调整后续排序；
- 将已完成运行中的记忆归档并预热到后续任务。

该工具建立了区别于普通 hit rate 的评价体系：`retrieval_hit_rate` 记录候选命中，
`useful_hit_rate` 记录注入后的有效推进，`stale_hit_rate` 记录过期信息，
`memory_pollution_rate` 记录负面影响，`stage_first_turn_hit_rate` 则衡量阶段预取效果。
这些指标使长期记忆能够被调试和消融，并为学习式 memory policy 提供监督信号。

**实现入口**

- `ucagent/memory/long_term.py`
- `ucagent/stage/vmanager.py`

### 4.5 `co-trace`：验证状态轨迹编译与可视化

一次运行可能包含上千次模型和工具交互。最终完成状态无法解释 Agent 在哪个阶段反复
读取或修改了哪些对象、测试是否真实执行，以及失败后是否改变策略。`co-trace` 将原始
日志编译为结构化执行轨迹，并提供面向实验运行的可视化。

**主要能力**

- 记录文件读取、搜索、写入、删除和移动；
- 记录 `RunTestCases`、`Check`、`Complete` 及 Stage 转移；
- 将测试执行统一分类为六种终态；
- 使用摘要哈希记录修改规模，避免在事件流中复制完整代码；
- 构建 action-observation-stage trace tree；
- 按实验、运行、DUT 和 Stage 筛选轨迹；
- 首次加载历史实验，随后增量解析新日志；
- 展示失败事件、区域切换、重复动作和策略转折。

测试终态统一为：

```text
pass / test_failure / infrastructure_error /
timeout / crash / no_tests_collected
```

该分类保证 `tests_total=0`、pytest collection error、超时和进程异常不会被解释为成功，
从数据源头保护轨迹分析、策略抽取和微调样本。轨迹图进一步将“未完成”分解为过度调查、
过早修改、测试范围过宽、Checker 契约阻塞和同一失败下的重复写入等可研究行为。

**实现入口**

- `ucagent/util/test_result.py`
- `scripts/build_trace_tree.py`
- `scripts/analyze_structured_events.py`
- `scripts/visualize_agent_trajectory.py`
- `tests/test_test_result_classification.py`
- `tests/test_agent_trajectory_visualizer.py`

### 4.6 `co-strategy`：Verifier-grounded 策略工程

不同 DUT 可能出现相似错误，但错误文本相似不代表修复动作可以直接迁移。例如两个 Bug
文档都出现 Checker failure，其根因可能分别是标签层级错误和错误引用已通过测试。
`co-strategy` 使用当前验证状态约束历史轨迹的检索、注入和执行。

**主要能力**

- 标准化动作前后的测试集合、Checker 状态和 Stage 转移；
- 将动作收益划分为 `progress`、`diagnostic`、`no_progress`、`regression` 和 `invalid`；
- 把有效轨迹编译为包含前置条件、动作边界、预期转移和安全约束的 strategy contract；
- 根据 Stage role、失败模式、失败签名、修改对象和动作类别执行硬门控；
- 对低支持度写入、删除和移动操作使用更严格的签名约束；
- 使用 off、shadow 和 enforce 三种模式评估策略；
- 在冻结 checkpoint 上进行同 seed 的注入/不注入成对回放；
- 使用进展控制器限制同一失败状态下的连续修改，并在停滞时触发诊断或策略转向。

策略检索与策略准入是两个独立步骤。检索发现可能相关的历史经验；准入依据真实 Checker
状态迁移判断候选是否具有收益。生产策略具有 candidate、held、admitted、quarantined
和 archived 生命周期，为跨 DUT 经验复用提供可追踪、可撤销的治理机制。当前策略包
包含 36 条结构去重候选，来源覆盖 Adder、uart_tx、FSM 和 ALU754。

**实现入口**

- `ucagent/memory/context_reuse.py`
- `ucagent/control/progress.py`
- `scripts/build_context_reuse_pack.py`
- `scripts/curate_context_reuse_pack.py`
- `scripts/replay_context_reuse_gate.py`
- `scripts/run_adder_stage_benchmarks.py`
- `tests/test_context_reuse_curation.py`
- `tests/test_episode_credit.py`
- `tests/test_progress_controller.py`

### 4.7 `co-bench`：可复现的多 DUT 实验 Harness

芯片验证 Agent 的完整实验成本高，单次运行可持续十小时以上。配置、随机种子、恢复位置
或模型服务发生变化，都可能使结果失去可比性。`co-bench` 为 UCAgent 与 CO-UCAgent
提供统一实验入口和运行账本。

**主要能力**

- 冻结模型、Prompt、上下文、策略库和 Checker 配置；
- 运行前检查模型端点、Embedding 服务、DUT 和依赖环境；
- 支持单 DUT、多 DUT 和多 seed 队列；
- 保存 ledger、Stage 结果、Token、LLM 请求、测试和 Checker 指标；
- 支持失败重试、断点恢复和 Stage checkpoint 回放；
- 显式标记 clean、interrupted、patched-resume 和 invalid；
- 为原版 UCAgent 注入独立 Token meter，使 baseline 与候选使用相同统计口径；
- 接入通知和轨迹可视化。

实验有效性属于账本的一部分。恢复运行可用于完成链验证、阶段研究和故障定位，但不会被
自动改写为 clean end-to-end 结果。当前已建立八个 DUT 的 clean baseline；代表性的
同 seed Adder 实验中，30-Stage active time 从 12 h 59 min 降至 5 h 37 min，Checker
失败事件从 70 次降至 41 次。

**实现入口**

- `scripts/run_adder_experiments.py`
- `scripts/run_multi_dut_experiments.py`
- `scripts/run_upstream_baseline_multi_dut.py`
- `scripts/prepare_multi_dut_stage_resume.py`
- `scripts/freeze_experiment_baseline.py`
- `scripts/analyze_ucagent_runtime.py`
- `tests/test_multi_dut_experiment_runner.py`

### 4.8 `co-trajectory-data`：可审计的 Agent 轨迹数据构建

芯片验证 Agent 的微调数据需要同时包含 Stage 任务、工具输出、Checker 反馈和下一步动作。
`co-trajectory-data` 从真实运行中提取多轮决策样本，并保留完整来源信息。

**主要能力**

- 优先读取精确 `llm_input_messages` dump；
- 对历史日志重建可见上下文并标记重建来源；
- 根据完成状态、验证反馈、工具类型和错误信息筛选候选；
- 控制每个 run 和 Stage 的样本数量，减少高频阶段垄断数据；
- 按 train/validation/test 和 held-out DUT 划分；
- 输出 chat-SFT JSONL；
- 转换为 LLaMA-Factory Alpaca 格式并生成 LoRA 训练配置；
- 汇总 RTL debug、VerilogEval、RTLLM 和 CircuitNet 等公开数据源。

每条样本保留 run、DUT、Stage、选择分数、选择原因、输入是否精确和是否截断等 provenance
字段。当前数据管线从 118 个历史运行目录中识别 92 个完成运行，生成 2,196 条轨迹样本，
其中训练集 1,974 条，验证集和测试集各 111 条。该工具可用于监督微调、轨迹蒸馏、动作
分类和 memory policy 学习，并允许研究者追溯样本对应的原始验证过程。

**实现入口**

- `scripts/build_ucagent_finetune_dataset.py`
- `scripts/prepare_llamafactory_ucagent_dataset.py`
- `benchmark/ucagent_finetune_dataset/manifest.json`
- `benchmark/ucagent_finetune_dataset/public_dataset_catalog.json`

### 4.9 组合使用方式

**性能诊断闭环**

```text
co-llm-profiler -> co-context -> co-trace -> co-bench
```

先区分输入长度、生成速度和请求数量，再通过上下文编译减少重复内容，使用轨迹检查行为
变化，最后在冻结实验中比较 wall time、Token 和完成率。

**跨 DUT 经验复用闭环**

```text
co-trace -> co-memory-cache -> co-strategy -> co-bench
```

将运行日志转化为验证状态，沉淀候选经验，对动作收益和适用范围进行归因，通过 checkpoint
pair 验证后进入策略库，再使用 held-out DUT 检查泛化效果。

**模型适配闭环**

```text
co-trace -> co-trajectory-data -> LoRA/SFT -> co-llm-profiler -> co-bench
```

从真实执行中构建训练样本，完成模型适配后先测试推理性能与局部任务，再进入完整 DUT
评测，避免仅依据训练 loss 判断模型是否改善 Agent 行为。

## 5. Evaluation

### 5.1 Experimental Setup

- **主要模型：** 本地 `mdq100/qwen3.5-coder:122b-64k`（Q4_K_M）
- **推理后端：** Ollama OpenAI-compatible API
- **硬件：** 每个模型实例使用 5 张 NVIDIA RTX 3090
- **上下文窗口：** 65,536 token
- **工作流：** 30-Stage UCAgent
- **DUT：** Adder、Mux、FSM、ShiftRegister、DualPort、HPerfCounter、uart_tx、
  ALU754 和 IntegerDivider
- **主要指标：** 完成状态、active time、Prompt/Completion token、LLM 请求、Checker
  迁移、测试终态和失败循环

实验冻结模型权重与量化方式、GPU 数量、推理服务参数、Prompt、超时、策略包、代码版本
和随机种子。主结果只采用能够追溯到日志、账本和冻结配置的记录。`clean end-to-end`、
`resume aggregate` 与 `checkpoint-composed` 分别报告，不用恢复链替代 clean 证据。

### 5.2 Baselines and Ablations

为分别度量证据建模、上下文管理与经验复用的独立贡献，本文计划采用以下递进配置：

| ID | 配置 | 研究目的 | 状态 |
|---|---|---|---|
| C0 | 原始 UCAgent 上下文路径 | 上游系统基线 | 8 个 DUT 已完成，IntegerDivider 未完成 |
| C1 | C0 + 结构化/层次摘要 | 摘要贡献 | 有历史数据，待重跑 |
| C2 | C1 + failure-aware context + 输出压缩 | 局部失败上下文贡献 | 有历史数据，待重跑 |
| C3 | C2 + 长期记忆与阶段预取 | 跨阶段复用贡献 | 有历史数据，待重跑 |
| C4 | C3 + 阶段契约 Prompt v1 | Prompt 与上下文协同贡献 | 已有探索性结果 |
| C5 | C4 + 跨 DUT Episode 检索 | 经验复用贡献 | 已有历史运行 |
| C6 | C5 + verifier-utility selection | 归因与历史效用贡献 | 1 次完整 Adder |
| C7 | C6 + semantic/action-risk filtering | 经验适用性约束贡献 | 1 次完整 Adder |

进一步消融包括：去除验证范围建模、去除 role typing、允许 invalid Episode、去除动作对象
兼容性、去除回归风险约束，以及关闭后验效用回填。

### 5.3 Metrics

**任务级指标：** DUT completion rate、总耗时、阶段耗时、失败循环长度、LLM 调用数、
输入/输出 token、TTFT p50/p95 和 prefill/decode TPS。

**行动级指标：** 固定预算内的 stage advance、失败净减少量、progress/diagnostic/
no-progress/regression/invalid 比例、action cost 和 time-to-recovery。

**记忆级指标：** retrieval precision、injection precision、useful hit rate、memory
pollution rate、negative-transfer rate、注入 token 开销和 token ROI。

**归因指标：** accuracy、macro-F1、各角色 precision/recall/F1、标注者一致性，以及
`invalid -> progress` 等高风险混淆率。

完整 DUT 实验至少采用 3 个随机种子并报告均值、标准差、配对差值与 bootstrap 95% CI；
Episode 与 checkpoint 实验按 DUT 和时间划分训练/验证集合。

## 6. Results

### 6.1 Context Engineering 与 Prompt

> **版本口径说明：** 本节数据来自 2026 年 6 月早期实验，运行于上游
> [UCAgent v26.04.17](https://github.com/XS-MLVP/UCAgent/releases/tag/v26.04.17)
> 版本线的旧工作流，早于 `v26.06.24` 之后采用的 30-Stage 流程。该版本执行的阶段、
> Checker 契约及上下文采集范围均较少，因此 Adder 的 `3h06min` 和对应 token 规模
> 不能作为 6.4 节 30-Stage 实验的绝对时间或 token baseline。本节仅用于比较同一旧版
> 工作流下 Prompt v0/v1 与早期 Context Engineering 的相对变化。

| DUT | 配置 | N | 总耗时 | 输入 token | 输出 token |
|---|---|---:|---:|---:|---:|
| Adder | 122B, Prompt v0 + input dump | 1 | 3h06min | 1.30M | 16.4K |
| Adder | 122B, Prompt v1 + input dump | 1 | 2h27min | 990.8K | 16.2K |
| uart_tx | 122B, Prompt v1 + input dump | 2 | 7h16min / 5h32min | 2.23M / 1.59M | 24.2K / 27.4K |

Adder 的单次观察中，总耗时下降 21.0%，输入 token 下降 23.8%。但该结果仅有一次运行，
且 uart_tx 的两次运行相差 1h44min，表明模型采样、失败路径和推理服务长尾均可能产生
较大方差。

### 6.2 Repair Episode 归因

| 数据集 | N | Accuracy | Macro-F1 | Progress P/R/F1 | Regression P/R/F1 |
|---|---:|---:|---:|---:|---:|
| 独立平衡标注集 | 40 | 0.8000 | 0.7976 | .750/.750/.750 | .500/.667/.571 |
| Adder 时间留出集 | 40 | 0.8000 | 0.8205* | .833/.417/.556 | 1.000/.909/.952 |

`*` 时间留出集未包含 `invalid`，Macro-F1 仅对出现的四类计算。时间留出集的双标注者
一致率为 97.5%，但 progress recall 为 0.4167，表明当前规则对真实进展采取了较保守的
判定。

### 6.3 验证条件化的经验检索

在 v2.1 历史注入回放上，语义与风险约束改变 23/32 次注入决策，完全抑制 15 次注入，
Prompt 中的经验条目由 85 减少至 42，下降 50.6%。该结果说明适用性过滤能够识别一部分
已知不兼容候选，但不能单独证明端到端收益。

| 配置 | N | Adder 耗时 | token in/out | 检索行为 |
|---|---:|---:|---:|---|
| 近期历史运行（非严格配对） | 4 | 2h52–3h03 | 1.01–1.27M / 16.7–19.2K | 未启用 episode credit |
| verifier-utility selection | 1 | 3h40min | 1.30M / 32.7K | 32 次注入，85 条目 |
| semantic/action-risk filtering | 1 | 4h58min | 1.60M / 73.9K | 9 次注入，25 条目 |

风险约束将注入次数由 32 次降至 9 次，但对应完整运行的耗时与 token 未改善。这一结果
揭示了“减少检索污染”与“提高任务性能”之间并非等价关系：被保留 Episode 的质量、
未检索到的正向经验、模型随机性以及独立失败循环均可能影响端到端结果。后续实验需要在
相同 checkpoint 上比较行动结果，再扩展到完整 DUT。

### 6.4 多 DUT 端到端与完成链结果

> **版本口径说明：** 本节采用以
> [UCAgent v26.06.24](https://github.com/XS-MLVP/UCAgent/releases/tag/v26.06.24)
> 为上游基线、并完成必要兼容性修补的 30-Stage 工作流。相较 6.1 节的旧版流程，
> 该版本增加了阶段任务、Checker 检查与交付约束，baseline 和 CO-UCAgent 均需完成
> 更长的验证链，因此耗时与 Prompt token 总量整体更高。下表的时间和 token 变化只在
> 相同 30-Stage 任务口径内比较，不与 6.1 节的绝对数值交叉比较。

下表以 2026-09-14 已冻结的统计表为主口径，并追加截至 2026-09-18 已确认的新结果。
`Base (h)` 为 baseline 端到端 active time；`CO-UCAgent (h)` 对 clean 运行同样采用端到端
active time。


| DUT | Base (h) | CO-UCAgent (h) | Δtime | Base→CO Prompt | ΔToken | Status |
|---|---:|---:|---:|---:|---:|---|
| FSM | 9.91 | 3.19 | **-67.8%** | 25.76M → 22.60M | **-12.3%** | 完成 |
| ShiftRegister | 12.90 | 4.06 | **-68.5%** | 34.44M → 16.72M | **-51.4%** | 完成 |
| Adder | 12.98 | 5.62 | **-56.7%** |  25.83M → 11.63M | **-55.0%** | 完成 |
| Mux | 12.72 | 5.21 | **-59.0%** | 30.69M → 18.64M | **-39.3%** | 完成 |
| DualPort | 11.21 | 6.67 | **-40.5%** | 29.90M → 33.52M | +12.1% | 完成 |
| HPerfCounter | 15.33 | 11.18 | **-27.1%** | 38.42M → 24.03M | **-37.4%** | 完成 |
| uart_tx | 16.51 | — | — | 42.07M → — | — | 未完成 |
| ALU754 | 12.86 | 15.72 | +22.2% | 26.61M → 33.57M | +26.2% | 完成 |
| IntegerDivider | baseline 未完成 | — | — | 首次 50.88M；后续 79.997M | — | 未完成 |
#### 6.4.1 Clean 结果

Adder 和 HPerfCounter 构成当前最严格的正向证据。Adder 在相同 seed、相同 30-Stage
任务和相同五卡本地模型条件下，将 active time 从 12.98h 降至 5.62h，降幅 56.7%；
Stage 23 从 7h04min 降至 46min，Stage 24 从 1h21min 降至 6min57s，最终 Checker
通过并识别到与 baseline 一致的 RTL 位宽根因。HPerfCounter 从 15.33h 降至 11.18h，
同时 Prompt token 从 38.42M 降至 24.03M。两项 clean 结果累计耗时由 28.31h 降至
16.80h，描述性降幅为 40.7%。Adder clean baseline Token为25.83M ，CO-UCAgent 的Token为11.63M 
该组的 ΔToken计算为-55.0%。

FSM、ShiftRegister 和 Mux 的修复后成功续跑段分别为 3.19h、4.06h 和 5.21h；相对于
baseline 的描述性差值分别为 -67.8%、-68.5% 和 -59.0%。相应有效完成链的 Prompt
Token 由 90.89M 降至 57.96M，下降 36.2%。

DualPort 重新审计后，原 `39.92M` 被确认是包含废弃 Stage 28 尝试的 resume aggregate，
去除该失败尝试后的有效完成链 Prompt Token 为 33.52M，仍较 baseline 增加 12.1%。
修复后成功续跑段耗时为 6.67h，较 baseline 低 40.5%，但该时间不包含修复前已完成前缀，
因此 DualPort 只能说明断点恢复后的完成成本，不能作为 clean 端到端时间收益证据。

在七个具有同 seed、同任务口径且 Prompt Token 可比的 DUT 上（FSM、ShiftRegister、Adder、
Mux、DualPort、HPerfCounter 和 ALU754），有效完成链 Prompt Token 从 211.65M 降至
160.71M（-24.1%）。该汇总保留 DualPort 与 ALU754 的负收益。

#### 6.4.2 新增完成链与未完成任务

ALU754 已由三个可追溯片段形成完整验证链：Stage 0–22 的有效前缀、独立通过的 Stage 23
回放，以及连续完成的 Stage 24–30。复核后的 33.57M 仅累加这三个被采用片段，未计入
被替代的 Stage 23/24 失败尝试。总 active time 为 15.72h，时间和 Prompt Token 分别
高于 baseline 22.2% 和 26.2%。该结果证明复杂 DUT 可以借助检查点恢复完成全部阶段，
同时也定位出 Stage 23 与后半程的显著成本，当前版本尚未在 ALU754 上取得性能优势。

uart_tx 的近期 clean 尝试分别在 Stage 24 和 Stage 25 停滞，尚无可用于主表的 CO 完成
结果。IntegerDivider baseline 首次在 22.89h 后失败，后续 35.98h 尝试超时并消耗
79.997M Prompt token；截至本文更新时，新一轮 baseline 重试仍在运行。因此这两个 DUT
只报告完成状态和已观测成本，不计算性能提升。


## 7. Discussion and Future Directions

### 7.1 学习式经验选择

当前检索采用可解释规则和加权打分，便于审计但难以刻画多特征交互。后续可使用人工归因
和运行后验构建轻量学习排序器，预测候选 repair episode 的 progress probability 与 regression
risk。模型训练和评价必须采用 held-out DUT，才能检验跨设计泛化，而非记忆具体任务。

### 7.2 不确定性感知的验证预算分配

结构化 trace tree 已提供行动、观察、成本和局部收益。下一步可将线性重试扩展为预算约束
下的分支选择：当当前路径持续 no-progress 时，在继续修复、回退到可信检查点和探索替代
动作之间分配验证预算。该方向需要独立于检索模块进行消融，以区分记忆质量与搜索策略的
贡献。

### 7.3 模型适配与系统协同

领域微调可用于提高接口理解、测试断言生成和错误归因能力，但微调数据必须来源于可信
repair episode，避免将无效测试与回归行动固化到模型参数中。模型量化、推理并行和
KV cache 优化主要改变单次调用成本，而 Context Engineering 改变调用长度与修复轮次；
二者构成互补的系统级优化维度。

## 8. Related Work

**Context Engineering。** [近期综述](https://arxiv.org/abs/2507.13334)将 Context
Engineering 定义为对推理时信息负载的系统性优化，并将其分解为 context
retrieval/generation、processing 和 management。
CO-UCAgent 采用这一问题边界，但面向芯片验证进一步要求摘要保留端口、API、检查点和
验证范围等可执行契约。[Agentic Context Engineering](https://arxiv.org/abs/2510.04618)
将上下文视为通过 generation、reflection 与 curation 持续演化的 playbook；本文与其
共同关注 execution feedback 和增量更新，但使用测试与 Checker 差分约束上下文写入，
而不是依赖自然语言反思判断经验是否有效。

**Agent Harness 与交互接口。** [SWE-agent](https://arxiv.org/abs/2405.15793) 表明，为
语言模型设计专用 Agent-Computer Interface 能显著改变其软件工程行为。2026 年的
[Natural-Language Agent Harnesses](https://arxiv.org/abs/2603.25723) 进一步把显式契约、
持久化 artifact 和运行时 adapter 作为可迁移 Harness 的核心组成；
[Code as Agent Harness](https://arxiv.org/abs/2605.18747) 则强调代码在行动、状态维护、
环境建模和 execution-based verification 中的基础设施作用。CO-UCAgent 延续这一视角，
但研究对象是包含多阶段 Checker、覆盖率和 bug 文档契约的芯片验证 Harness，其 Context
Engine 是 Harness 内负责推理时信息构造的一个模块，而非 Harness 的同义词。

**过程记忆与行动归因。** Agent Workflow Memory 通过复用历史 workflow 改善后续任务，
CO-UCAgent 则将工作流切分为具有前后验证观察的 repair episode。角色化归因借鉴 TRIAGE
对中间行动差异化分配信用的思想，但不复现其强化学习过程，而是使用无需训练的验证器差分。
经验选择与 Beyond Similarity/MemGate 的“相似性不足以判断记忆价值”结论一致，并加入
阶段职责、修改对象和文档契约约束。TRACE 对 trace node 进行 rollout 预算分配，为后续
不确定性感知搜索提供了参考；当前方法只构建其所需的 trace、成本与局部收益信号，尚未
实现学习式预算分配。

## 9. Artifact and Reproducibility

### 9.1 与 UCAgent 的兼容性

当前版本已同步本地 UCAgent 2026-07-16 快照中的 Formal/static-bug 工作流、Skill、
Textual TUI、Web console、master/worker API、workspace archive、CLI/config override、
stage journal、LLM Checker suggestion 以及相关示例和测试。研究运行时保留
`create_react_agent + pre_model_hook`，用于精确输入采集和上下文实验；新版
`create_agent + middleware` 将作为独立兼容变量评估。

### 9.2 代码结构

| 路径 | 功能 |
|---|---|
| `ucagent/abackend/langchain/message/` | 摘要、消息生命周期与性能采集 |
| `ucagent/memory/` | 长期记忆、阶段预取与 repair episode 检索 |
| `ucagent/stage/` | 阶段管理、Checker 观察与上下文注入 |
| `ucagent/util/test_result.py` | 六类测试终态归一化 |
| `scripts/build_trace_tree.py` | 结构化轨迹与 trace tree 构建 |
| `scripts/build_context_reuse_pack.py` | repair episode 抽取与角色化归因 |
| `scripts/curate_context_reuse_pack.py` | 过程记忆去重与质量筛选 |
| `scripts/analyze_context_reuse_decisions.py` | 检索后验与污染率分析 |
| `benchmark/ucagent_experiments/` | 冻结配置、manifest 与实验协议 |

### 9.3 环境与运行

```bash
git clone https://github.com/spadenoheart/CO-UCAgent.git
cd CO-UCAgent
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

export OPENAI_MODEL='mdq100/qwen3.5-coder:122b-64k'
export OPENAI_API_KEY='ollama'
export OPENAI_API_BASE='http://127.0.0.1:11434/v1'

make run_Adder
make run_uart_tx
```

`make run_<DUT>` 使用 `output/workspace_<DUT>` 作为独立工作区并持续写入 `log/`；
`make continue_<DUT>` 从同一工作区恢复。需要 TUI 和人工确认时使用 `make test_<DUT>`。

Prompt 对比实验：

```bash
UCAGENT_PROMPT_VARIANT=v0 \
  make run_Adder CWD=output/adder_prompt_v0

UCAGENT_PROMPT_VARIANT=v1 UCAGENT_LLM_INPUT_DUMP=true \
  make run_Adder CWD=output/adder_prompt_v1
```

核心回归测试：

```bash
PYTHONPATH="$PWD" python -m pytest -q \
  tests/test_test_result_classification.py \
  tests/test_context_reuse_curation.py \
  tests/test_episode_credit.py \
  tests/test_experiment_baseline.py \
  tests/test_message_lifecycle.py
```

当前冻结实验配置位于
`benchmark/ucagent_experiments/20260721_upstream_sync_v22_candidate/`。原始日志、模型
权重、私有 RTL 与大规模输入 dump 不属于公开发布数据。

## 10. Evaluation Completion Matrix

下表区分已经获得的证据与仍需补充的论文实验，避免用恢复链替代 clean 重复实验。

| 实验 | DUT / 划分 | 当前重复 | 主要指标 | 当前状态 |
|---|---|---:|---|---|
| 上游 baseline | 9 个目标 DUT | 8 completed | completion, time, token, requests | IntegerDivider 尚未完成 |
| Clean end-to-end 对比 | Adder, HPerfCounter | 各 1 seed | completion, time, token, Checker | 已获得两项正向结果 |
| Resume/composed 完成链 | FSM, ShiftRegister, Mux, DualPort, ALU754 | 各 1 chain | active time, token, stage bottleneck | 已完成，证据等级单独标注 |
| C0-C4 上下文管理消融 | Adder, uart_tx, FSM | 历史探索数据 | completion, time, token, loop length | 需统一配置后补跑 |
| C4-C7 经验复用消融 | paired checkpoints | 已有局部 pair | progress, regression, token ROI | 需扩展到每 DUT >= 20 pairs |
| 跨 DUT 泛化 | held-out DUT | 未形成完整划分 | retrieval precision, negative transfer | 待完成 |
| 归因外部验证 | balanced + time holdout | 80 episodes | macro-F1, high-risk confusion | 两组 accuracy 均为 0.80 |
| 多 seed 主实验 | 优先 Adder、HPerfCounter、FSM、ShiftRegister、Mux | 当前 1 seed | mean/std, paired delta, bootstrap CI | 待扩展至 >= 3 seeds |

## 11. References

1. [UCAgent](https://github.com/XS-MLVP/UCAgent)
2. [Agent Workflow Memory](https://arxiv.org/abs/2409.07429)
3. [TRIAGE: Role-Typed Credit Assignment for Agentic Reinforcement Learning](https://arxiv.org/abs/2606.32017)
4. [Beyond Similarity: Trustworthy Memory Search for Personal AI Agents](https://arxiv.org/abs/2606.06054)
5. [TRACE: A Unified Rollout Budget Allocation Framework for Efficient Agentic Reinforcement Learning](https://arxiv.org/abs/2606.11119)
6. [Lost in the Middle: How Language Models Use Long Contexts](https://doi.org/10.1162/tacl_a_00638)
7. [LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models](https://arxiv.org/abs/2310.05736)
8. [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
9. [A Survey of Context Engineering for Large Language Models](https://arxiv.org/abs/2507.13334)
10. [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618)
11. [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793)
12. [Natural-Language Agent Harnesses](https://arxiv.org/abs/2603.25723)
13. [Code as Agent Harness](https://arxiv.org/abs/2605.18747)

## 12. Authors and License

- Jiabao Wang，北京邮电大学
- Yuzhong Sun，中国科学院计算技术研究所
- Li Xiao，北京邮电大学

本项目基于 UCAgent 开发并遵循 MIT License。发布与再分发时应保留上游版权及 NOTICE。
