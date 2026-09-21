# Verifier-Contract Trajectory Memory: Research Plan

Date: 2026-08-17

## 1. Objective conclusion

Failure-state-aware trajectory retrieval is not sufficient as the main novelty.
Recent work already covers failure-typed repair memory, phase/subtask-aligned
memory, selective intervention, abstention, contextual-bandit routing, and
performance-cost control. The defensible research question is therefore not
"can a failed trajectory be retrieved?", but:

> Can trajectory memory be maintained as a regression-tested policy artifact,
> where each strategy must pass paired verifier replay on held-out failure-stage
> checkpoints before admission, and remains deployed only while its observed
> transition contract continues to hold?

The proposed method is tentatively named **VECTRA-CI**: Verifier-grounded
Episode Contracts and Trajectory Replay Allocation with Continuous Integration.
The transition contract is a supporting representation. The primary novelty is
replay-validated memory admission and regression testing.

## 2. Novelty audit

| Existing direction | Representative work | Already covered | Remaining gap used here |
|---|---|---|---|
| Self-optimizing memory policy | [SelfMem](https://arxiv.org/abs/2607.03726), [AutoMem](https://arxiv.org/abs/2607.01224) | Agents/meta-LLMs refine memory strategies or memory skills | A deterministic verifier can supervise memory without a judge LLM or policy fine-tuning |
| Trainable memory construction/retrieval | [MemoryCPT](https://arxiv.org/abs/2608.04843), [Memory-R1](https://arxiv.org/abs/2508.19828) | Learned memory construction, retrieval, and memory operations | Local-model verification has sparse runs and expensive training; exploit native Checker signals instead |
| Failure-aware repair memory | [MERIT](https://arxiv.org/abs/2608.05906), [ReasoningBank](https://arxiv.org/abs/2509.25140) | Positive/negative episodes and failure-typed retrieval | Similarity/type match does not specify what observable transition the reused action must cause |
| Planning/progress coupled memory | [PMCoder](https://arxiv.org/abs/2608.06811), [Subtask-Level Memory](https://openreview.net/forum?id=2CoRS45Ucj) | Phase-aware retrieval, stuck detection, subtask granularity | Retrieved memory is still guidance rather than a runtime-verifiable action contract |
| Selective and budget-aware memory | [RSCB-MC](https://arxiv.org/abs/2604.27283), [BudgetMem](https://arxiv.org/abs/2602.06025), [Proactive Memory Agent](https://arxiv.org/abs/2607.08716) | Abstention, risk-sensitive routing, tier allocation, selective reminders | Decide value using expected and observed verifier-state transitions, not only proxy reward or relevance |
| Role-typed trajectory credit | [TRIAGE](https://arxiv.org/abs/2606.32017) | Progress/exploration/no-progress/regression process credit | Apply role credit to a memory item's lifecycle rather than model-policy training |
| Rollout budget allocation | [TRACE](https://arxiv.org/abs/2606.11119) | Allocate rollout budget to informative roots and prefixes | Allocate memory interventions and verification budget to uncertain contracts |
| Transition-graph repair memory | [AgentTether](https://arxiv.org/abs/2607.06273) | Critical Transition Graph, Repair Memory, and guarded runtime intervention | Admission is not defined as a paired memory-policy regression test over frozen failure-stage checkpoints |
| Trace-guided harness repair | [HarnessFix](https://arxiv.org/abs/2606.06324) | Failure attribution, repair specifications, held-out harness validation | Optimize memory-policy negative transfer rather than patching the agent harness |
| Verifier-gated executable policy knowledge | [Kintsugi](https://arxiv.org/abs/2605.09487) | Typed KB edits and verifier-gated policy execution | Treat each memory strategy and retrieval-policy revision as a CI-tested artifact across DUT checkpoints |

Important evidence qualification: many 2026 references above are recent
preprints. The ICML subtask-memory paper is stronger publication evidence than
single-author or smoke-scale preprints. Claims must be based on our own paired
experiments, not on the novelty language of those preprints.

These closest works make a contract-only claim too weak. VECTRA-CI must be
evaluated as a **memory policy testing and admission system**, not presented as
the first use of verifier gates or transition graphs.

## 3. Proposed method

### 3.1 Contract-carrying trajectory memory

Each reusable Episode is converted to:

```text
precondition:
  stage/subtask
  normalized failure pattern and failure-set types
  compatible modification object
action contract:
  action role, operations, path categories, mutation budget
expected transition:
  failure-set reduction or information gain
  whether stage advancement is acceptable evidence
safety constraints:
  maximum regression count
  invalid/no-test/timeout rejection
verification:
  required verifier event and comparable validation scope
```

This is not an executable patch and does not copy DUT-specific values. It is a
falsifiable claim about which kind of state transition should follow a strategy.

### 3.2 Memory-policy continuous integration

Each new Episode or retrieval-policy revision creates a candidate artifact and
an automatically generated test set:

1. positive checkpoints whose preconditions match the candidate contract;
2. hard-negative checkpoints with high textual similarity but incompatible
   failure/action contracts;
3. prior regression checkpoints on which related memory caused no progress,
   invalid execution, or new failures.

For each checkpoint, run paired arms from the same frozen stage-entry state and
seed: `no candidate memory` versus `candidate memory/policy`. Admit the candidate
only when its lower-confidence utility is positive, its regression bound is
below a fixed threshold, and quality remains non-inferior. Every harmful online
intervention is added as a new regression checkpoint. This is memory test-driven
development rather than one-time memory curation.

### 3.3 Automatic closed-loop lifecycle

1. Build a normalized failure state from RunTestCases/Check/Complete.
2. Retrieve candidates using the existing failure/stage/action hard gates.
3. Evaluate each candidate's contract applicability and CI status.
4. Select `abstain`, `diagnostic contract`, or `repair contract` under a token
   and mutation budget.
5. Observe the next authoritative verifier result.
6. Compare expected and observed failure-set transitions.
7. Promote fulfilled contracts; retain diagnostic contracts only when they add
   actionable evidence; demote no-progress contracts; quarantine regression or
   invalid contracts.

The first implementation runs step 3 in shadow mode. It does not change Prompt
or retrieval order. Enforcement starts only after a held-out annotation/replay
audit.

### 3.4 What is and is not causal

One normal run only provides observational attribution. A successful validation
after injection does not prove that the memory caused success. Causal policy
uplift must be estimated with paired stage-checkpoint replay:

- same DUT, seed, model, stage state, and failure signature;
- treatment: enable one candidate memory/policy revision;
- control: no candidate memory or the current frozen retrieval policy;
- compare stage advancement, failure-set delta, regressions, tokens, and time.

The paper should use "verifier-grounded observational credit" before paired
replay, and reserve "causal effect" for randomized or matched treatment-control
comparisons.

## 4. Research questions and hypotheses

- **RQ1:** Does a contract applicability gate reduce harmful memory injection on
  held-out DUTs compared with semantic and failure-typed retrieval?
- **RQ2:** Does verifier-driven contract lifecycle management reduce repeated
  Checker signatures, mutation loops, wall time, and tokens without reducing
  completion or defect-detection quality?
- **RQ3:** Do expected transition types transfer across DUTs better than exact
  failure strings or whole-episode similarity?
- **RQ4:** Can uncertainty-aware abstention and verification allocation improve
  the quality-cost frontier under fixed local-model budgets?
- **RQ5:** Does replay-validated admission outperform one-pass heuristic or
  judge-based curation in preventing memory regressions over repeated updates?

Primary hypothesis: VECTRA-CI reduces total tokens and wall time while remaining
non-inferior in DUT completion and confirmed defect quality. A pure speedup that
misses bugs is not a successful result.

## 5. Experimental protocol

### 5.1 Data split

- Partition the usable 10 DUTs before building the final strategy library.
- Recommended split: 6 memory-source DUTs, 2 development DUTs, 2 strictly
  held-out DUTs.
- Stratify by interface/protocol and task complexity; do not randomly place all
  similar arithmetic DUTs in both source and held-out sets.
- DUTs too large for full validation may be used only for bounded stage replay,
  and must not be counted as full-flow completion results.
- Use 3 seeds for the primary baseline-versus-VECTRA-CI comparison and at least 2
  seeds for major ablations. Cross-model validation is deferred to the final
  experiment stage.

### 5.2 Baselines

1. Upstream UCAgent without CO-UCAgent memory/control changes.
2. CO-UCAgent without context reuse.
3. Semantic top-k Episode retrieval.
4. Failure-typed fixed retrieval, representing the current MERIT/ReasoningBank-
   like baseline.
5. Stage/subtask-aligned retrieval.
6. Selective abstention/risk gate baseline, reproducing the core RSCB-MC idea in
   the same harness.
7. VECTRA contract gate without replay admission.
8. VECTRA-CI with paired replay admission, regression checkpoint mining, and
   lifecycle policy.

We may claim to reproduce or adapt ideas from prior work, but can only claim to
beat these baselines inside the same CO-UCAgent harness. Results on different
benchmarks are not direct wins over the original papers.

### 5.3 Metrics

Quality constraints:

- full-flow completion rate and completed stages;
- confirmed DUT-bug precision/recall or an auditable proxy;
- test/check validity and regression rate.

Efficiency:

- wall time, total input/output tokens, LLM calls;
- repeated failure-signature count and no-progress mutation rounds;
- failure reduction and stage advances per 1K injected tokens.

Memory-specific:

- contract applicability precision/recall from manual or paired-replay labels;
- contract fulfillment rate;
- harmful-injection and invalid-injection rate;
- abstention precision and missed-reuse rate;
- expected-versus-observed transition calibration;
- cross-DUT support and held-out transfer rate.

Report paired bootstrap confidence intervals and paired tests across DUT-seed
units. Do not treat multiple calls from one run as independent samples.

## 6. Engineering status

Implemented in this increment:

- `ucagent/memory/trajectory_contract.py`: automatic contract construction,
  applicability scoring, and verifier outcome evaluation;
- `ucagent/memory/context_reuse.py`: shadow contract policy and metrics;
- `ucagent/stage/vmanager.py`: auditable contract and shadow decision fields in
  structured events;
- `scripts/analyze_context_reuse_decisions.py`: links injected contracts to the
  next verifier observation and reports fulfillment outcomes;
- `tests/test_trajectory_contract.py`: progress, regression, invalid, stage
  advance, and shadow non-interference tests.
- `scripts/audit_contract_shadow.py`: balanced blinded applicability sampling,
  independent-label merge, precision/recall, Cohen kappa, and Wilson intervals;
- `scripts/run_adder_stage_benchmarks.py` in the orchestration repository:
  resumable matched control/treatment replay with alternating run order;
- `scripts/evaluate_contract_policy_admission.py`: quality-constrained
  bootstrap-LCB admission and admitted/held/quarantined registry state;
- `benchmark/ucagent_experiments/adder_vectra_ci_stage_pair_20260818_matrix.json`:
  frozen contract-off, shadow, and enforce pilot arms.

Not implemented yet:

- automatic promotion/demotion/quarantine persistence;
- enforcement mode validation and prompt schema experiment;
- uncertainty-aware contract selection and verification budget allocation;
- automatic positive/hard-negative/regression checkpoint selection;
- runtime consumption of the policy registry and automatic champion rollback;
- final cross-model evaluation.

### 6.1 Retrospective sanity check

The 2026-07-17 Adder utility-gate trace was re-analysed with contracts backfilled
from `context_reuse_v1.json`:

- 32 injection decisions and 40 throttled decisions;
- 24 injections linkable to a subsequent verifier observation;
- observational credit: 11 no-progress, 4 diagnostic, 6 regression, 3 invalid,
  and 0 progress;
- 30 strategy-contract outcomes: 18 inconclusive, 8 regression, 4 invalid,
  and 0 fulfilled.

This is not a result for VECTRA-CI because the trace predates contract selection.
It is evidence that the current utility gate is not a strong baseline: historical
utility did not reliably transfer to the current failure transition. The 18
inconclusive cases also show why event/scope compatibility must be explicit.
The audit is stored in
`benchmark/ucagent_context_reuse/Adder_20260717_contract_retrospective.{json,md}`.

## 7. Acceptance gates before enforcement

1. Shadow applicability decisions are audited on at least 50 balanced cases.
2. Precision for `applicable` is at least 0.90; report confidence intervals.
3. Regression and invalid cases are never admitted in the audited set.
4. Shadow mode produces byte-equivalent LLM input to contract-policy-off mode.
5. Paired stage replay shows positive median utility before any full DUT run.

If these gates fail, revise the contract schema. Do not compensate by lowering
the threshold until injections become frequent.
