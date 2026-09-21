# v2.1 Literature Positioning and Research Claim

## Closest 2026 work

| Work | Relevant idea | What UCAgent borrows | Remaining difference |
|---|---|---|---|
| TRIAGE (arXiv:2606.32017) | Role-typed segment credit separates progress, exploration, no-progress, and regression | Role taxonomy and conservative precedence | UCAgent uses deterministic compiler/test/Checker deltas instead of an LLM judge and is not yet an RL method |
| Memory-R2 (arXiv:2605.21768) | Local rerollouts from the same intermediate memory state give fairer memory-operation credit | Paired replay from the same UCAgent stage checkpoint | UCAgent evaluates retrieval/injection utility without LoGo-GRPO training |
| Useful Memories Become Faulty (arXiv:2605.12978) | Repeated LLM consolidation can corrupt useful memory; raw Episodes and explicit gates matter | Preserve raw trace evidence and curate without overwriting it | UCAgent studies verifier-grounded repair Episodes and cross-DUT transfer |
| Decision-Aware Memory Cards (arXiv:2606.08151) | Rank context by expected decision effect and negative-transfer risk, not similarity alone | Record retrieval decisions and measure next-verifier utility | UCAgent targets multi-stage hardware verification and can measure failure-set delta and stage advance directly |
| DIM-WAM (arXiv:2606.27677) | Historical events should retain task-progress identity and remain separated by memory type | Keep stage and role identity in Episodes | Its robotics model and training method are not reproduced |
| TRACE (arXiv:2606.11119) | Allocate rollout budget by node value in a thought-action-observation tree | Trace nodes and later budget-aware retry allocation | Current v2.1 does not implement TRACE prediction or agentic RL |

Primary links:

- https://arxiv.org/abs/2606.32017
- https://arxiv.org/abs/2605.21768
- https://arxiv.org/abs/2605.12978
- https://arxiv.org/abs/2606.08151
- https://arxiv.org/abs/2606.27677
- https://arxiv.org/abs/2606.11119

## Objective novelty assessment

Verifier-grounded role classification alone is not a strong paper contribution.
It is a domain-specific adaptation of TRIAGE and can be presented as a component,
not as the full novelty. Likewise, replacing semantic similarity with another
hand-weighted score would overlap strongly with decision-aware context-selection
work and would be difficult to defend without causal evidence.

The more defensible central claim is:

> A hardware-verification agent can build and reuse cross-DUT repair memory more
> reliably when memory credit is assigned from verifier deltas, retrieval
> decisions are linked to the next validation outcome, and strategy utility is
> estimated through paired stage-checkpoint replay rather than retrieval hit rate.

This claim combines three measurable contributions:

1. A five-role Episode attribution schema grounded in test/Checker evidence,
   including invalid execution and negative transfer.
2. An auditable retrieval-to-verifier feedback path that measures progress,
   diagnostic gain, no progress, regression, action cost, and stage advance for
   each injection.
3. A cross-DUT evaluation protocol using held-out DUTs and paired stage
   checkpoints, with token/wall-time efficiency and regression rate reported
   alongside completion.

## Baselines to beat

- B0: no context reuse.
- B1: v0 `failure -> later success` Episode extraction.
- B2: current v1 scalar quality plus rule-weighted retrieval.
- B3: v2.1 attribution and curation, but no measured retrieval-utility feedback.
- Ours: v2.1 plus retrieval-decision credit and a utility gate learned or
  calibrated from paired checkpoint outcomes.

The paper should claim improvement over these adapted UCAgent baselines, not
over TRIAGE, Memory-R2, or TRACE as complete systems. Those methods operate in
different environments and, in some cases, train policies with RL.

## Required evidence

- Fresh blind attribution accuracy >= 0.90 and progress precision >= 0.90.
- Leave-one-DUT-out evaluation on at least Adder, uart_tx, FSM, and ALU754.
- Paired stage-checkpoint A/B at stages 16, 20, 21, and 22 with at least three
  repeated runs per condition.
- Primary outcomes: stage recovery, failure-set reduction, regression/negative
  transfer, invalid-test rate, LLM calls, tokens, mutations, and wall time.
- Ablations: remove role typing, remove invalid/regression gate, remove paired
  utility feedback, and remove stage/action compatibility constraints.

Until these experiments are complete, the work is a credible research
prototype, not yet a validated paper result.
