# Episode Credit v2.1 Holdout Annotation Guide

## Blind procedure

1. Annotator A uses only `episode_credit_v2_holdout_adder_20260716_blind_a.jsonl`.
2. Annotator B independently uses only `episode_credit_v2_holdout_adder_20260716_blind_b.jsonl`.
3. Do not open the candidate pack, prediction-bearing JSONL, development-set
   metrics, or the other annotator's file before labels are frozen.
4. Fill the first `annotations` entry with a nonempty annotator ID, one role,
   and a short evidence note. Leave rule fields absent.

The blind files contain only raw compact observations, actions, and the observed
stage-transition flag. Rule-derived fields such as `failure_set_delta`,
`regression_count`, `information_gain`, failure pattern, and model confidence
are intentionally removed. Annotators must derive the role from the raw fields.

## Role precedence

Apply the first matching rule:

1. `invalid`: the after-observation is timeout, crash, infrastructure error, or
   no tests collected. Invalid execution can never be progress.
2. `regression`: comparable validation introduces a concrete new failed case,
   checkpoint, category, or specific Checker error class without net progress.
3. `progress`: the stage advances, the comparable target passes, or the shared
   failure set is reduced without regression.
4. `diagnostic`: no direct progress, but the result adds actionable evidence
   that changes the next probe, such as a missing signal/attribute/import.
5. `no_progress`: none of the above; the failure is unchanged, evidence is only
   restated, or the validation scope is insufficient to prove improvement.

For a file-to-single-test narrowing, unexecuted tests are unknown, not resolved.
For a single-test-to-file expansion, newly observed out-of-scope failures are
not automatically regressions against the original test. A changed generic hash
or signature is not itself a concrete regression.

## Merge and evaluate

```bash
python scripts/evaluate_episode_credit_annotations.py \
  benchmark/ucagent_context_reuse/episode_credit_v2_holdout_adder_20260716_blind_a.jsonl \
  --merge-annotations benchmark/ucagent_context_reuse/episode_credit_v2_holdout_adder_20260716_blind_b.jsonl \
  --merged-out benchmark/ucagent_context_reuse/episode_credit_v2_holdout_adder_20260716_merged.jsonl \
  --predictions-pack benchmark/ucagent_context_reuse/episode_credit_v2_holdout_adder_20260716_candidates.json \
  --refreshed-out benchmark/ucagent_context_reuse/episode_credit_v2_holdout_adder_20260716_scored.jsonl \
  --out benchmark/ucagent_context_reuse/episode_credit_v2_holdout_adder_20260716_metrics.json
```

Acceptance requires accuracy >= 0.90, progress precision >= 0.90, and no
invalid Episode predicted as progress. Because this holdout contains no natural
invalid observation, invalid handling is additionally checked by parser/rule
unit tests and must later receive an unseen invalid-execution challenge set.
