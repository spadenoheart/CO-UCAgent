# Episode Credit v2 Research Protocol

## Research question

Can verifier-grounded, role-typed credit assignment identify reusable RTL
verification repairs more accurately than outcome-only Episode extraction, and
thereby reduce repeated repair loops without increasing cross-DUT regressions?

## Method lineage

- Agent Lightning decomposes arbitrary agent trajectories into training
  transitions through a unified data interface. We reuse the transition view,
  but keep training out of scope for this phase.
- TRIAGE adds semantic roles to outcome credit and shows why uniform terminal
  credit cannot distinguish progress, exploration, no-progress work, and
  regression. We adapt this idea to deterministic verifier observations.
- TRACE models thought-action-observation turns as tree nodes. The v2 Episode
  schema can later supply node utility for rollout-budget allocation, but the
  current work does not claim to implement TRACE's predictor or RL algorithm.
- Memory-R2 shows that memory operations should be compared from the same
  intermediate memory state. We adapt this as paired stage-checkpoint replay,
  not as an implementation of LoGo-GRPO.
- Useful Memories Become Faulty When Continuously Updated by LLMs motivates
  preserving raw verifier Episodes and gating consolidation instead of
  overwriting evidence after every run.
- Decision-Aware Memory Cards motivates evaluating retrieved context by its
  effect on the next action and verifier outcome, rather than reporting
  semantic retrieval hit rate as utility.
- Skill-Pro verifies and maintains compact procedural skills. It motivates the
  separate reuse gate after credit attribution rather than injecting every
  mined Episode.
- Insights from Verification motivates using compiler/simulator/testbench
  outcomes as domain-grounded supervision for hardware tasks.

Primary sources:

- https://arxiv.org/abs/2508.03680
- https://arxiv.org/abs/2606.32017
- https://arxiv.org/abs/2606.11119
- https://arxiv.org/abs/2602.01869
- https://arxiv.org/abs/2504.15804
- https://arxiv.org/abs/2605.21768
- https://arxiv.org/abs/2605.12978
- https://arxiv.org/abs/2606.08151

## Attribution algorithm

1. Partition events by `run_id`, stage index, and stage name. Episodes never
   cross a restart or stage boundary.
2. Use a failing `RunTestCases` or `Check` observation as `failure_before` and
   the immediately following validation observation as `observation_after`.
3. Record successful file mutations and the final validation call as
   `action_sequence`.
4. Reject invalid execution evidence using the explicit outcomes `timeout`,
   `crash`, `infrastructure_error`, and `no_tests_collected`.
5. Classify validation scope as `same`, `narrowed`, `expanded`, `disjoint`, or
   `incompatible`. For narrowed/expanded test targets, compare only the shared
   target subset; newly observed tests outside that subset are not regressions.
   Generic signatures are not treated as concrete failure-set changes. Within
   the same Checker scope, a new specific checker error class is a regression.
   Cross-validator success counts as progress only when a stage transition
   confirms it.
6. Assign roles with conservative precedence: invalid observation, regression,
   verified progress, diagnostic evidence gain, then no progress.
7. Compute confidence from validation-scope agreement, structured evidence,
   action locality, and stage-transition confirmation. The `information_gain`
   score is an operational evidence-specificity proxy, not Shannon mutual
   information.
8. Mark an Episode `reuse_eligible` only when it is progress, has no regression,
   contains a file mutation, matches the validation scope, and reaches the
   confidence threshold. Runtime deployment remains a separate decision.
9. Truncate every run at its first `all_completed` transition. Cleanup actions
   or tool retries emitted after terminal completion cannot become Episodes.

## Baselines and claims

- B0: UCAgent v0 `failure -> later success` Episode extraction.
- B1: outcome-only uniform credit over all actions in a successful segment.
- B2: current scalar quality-gated v1 strategy pack.
- Ours: verifier-grounded role-typed v2 attribution plus quality curation.

The defensible framing is: borrow transition decomposition from Agent Lightning
and role typing from TRIAGE, then outperform adapted outcome-only UCAgent
baselines on the same DUT splits. Do not claim to outperform those papers as
complete systems unless their full methods are reproduced on a shared benchmark.

## Offline acceptance gate

- The original role-stratified 40-item sample is now a development set because
  its eight disagreements were used to revise v2.1. Its refreshed score is
  1.0000 accuracy and macro-F1, but this is a resubstitution result and is not
  evidence of independent generalization.
- Use `episode_credit_v2_holdout_adder_20260716_blind.jsonl` as the first
  temporal holdout. It contains 40 Episodes from completed run `703598` and
  withholds all rule roles and confidences. Its natural roles predicted by the
  frozen rule are progress=6, diagnostic=5, no_progress=19, regression=10.
  It contains no invalid execution, so a separate unseen invalid-execution
  challenge set is still required.
- Prefer two independent annotators; resolve disagreements only after the first
  labels are frozen.
- Report accuracy, macro-F1, per-role precision/recall, confusion matrix, and
  Cohen's kappa. The 90% attribution accuracy is a target, not a current result.
- Because the 40-item sample is role-balanced, accuracy estimates classification
  quality but not natural role prevalence. Add a separate random sample for a
  population-weighted estimate before paper submission.
- Pilot acceptance: accuracy >= 0.90, no `invalid` Episode accepted as progress,
  and progress precision >= 0.90. A paper-quality evaluation should expand to
  at least 150-300 independently labeled Episodes.

Development regression command:

```bash
python scripts/evaluate_episode_credit_annotations.py \
  benchmark/ucagent_context_reuse/episode_credit_v2_annotation_40.jsonl \
  --predictions-pack benchmark/ucagent_context_reuse/context_reuse_credit_v2_rule_v2_candidates.json \
  --refreshed-out benchmark/ucagent_context_reuse/episode_credit_v2_annotation_40_rule_v2.jsonl \
  --out benchmark/ucagent_context_reuse/episode_credit_v2_rule_v2_annotation_metrics.json
```

## Online experiment

- Use leave-one-DUT-out construction: build the library on three DUTs and test
  on the held-out DUT.
- Compare B0/B1/B2/v2 under the same model, prompt variant, hardware, endpoint,
  temperature, and stage checkpoint. Run at least three seeds per condition.
- Primary metrics: stage recovery rate, failed-set reduction per repair,
  repeated-failure count, regression rate, invalid-test rate, and negative
  transfer rate.
- Efficiency metrics: LLM calls, prompt/completion tokens, TTFT, wall time, and
  file mutations per stage advance.
- End-to-end DUT completion is secondary until checkpoint replay establishes
  attribution and retrieval causality at stages 16, 20, and 21.
- Every retrieval decision is now emitted as `context_reuse_decision` with the
  selected strategy IDs, scores, and failure evidence. After a new run, build a
  trace and link injections to the next verifier result with:

```bash
python scripts/analyze_context_reuse_decisions.py TRACE.json \
  --out-json decision_credit.json --out-md decision_credit.md
```

The resulting progress/regression rate is observational. Causal uplift requires
paired checkpoint replay with identical model/configuration and injection on/off.

## Current status

Rule v2.1 fixes the known development-set errors in scope expansion/narrowing,
generic-signature regression, and specific diagnostic evidence. The 40-item
development set now reaches 1.0000 accuracy and macro-F1, while the original
blind pass was 0.8000/0.7976. This improvement is useful as a regression test
but is optimistically biased because the labels informed the revision.

The runtime configuration still points to `context_reuse_v1.json`; v2.1 is an
offline candidate. Online deployment remains blocked on fresh holdout labels,
the invalid-execution challenge set, and paired injection on/off evaluation.
