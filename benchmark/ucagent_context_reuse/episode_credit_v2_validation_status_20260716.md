# Episode Credit v2.1 Validation Status

## Latest Adder evidence

- Run: `log/Adder+20260715_114538`, thread/run id `703598`, seed `417085`.
- Completion: `all_completed=true`, stage index 26.
- Total wall time: 9072.03 seconds, approximately 2 h 31 min 12 s.
- Stage 20: 355.92 seconds; the previous two-hour stall did not recur.
- Remaining long stages: stage 21 = 2897.22 seconds, stage 22 = 2836.01 seconds.
- This run used runtime `context_reuse_v1.json`. It validates the current
  Checker/Prompt/retrieval guard changes as a runnable baseline, but it does not
  establish v2 attribution or v2 strategy efficacy.

## v2.1 rule changes

- Distinguish same, narrowed, expanded, disjoint, and incompatible validation
  scopes and compare only the shared test subset.
- Do not turn generic signature changes into concrete regressions.
- Treat a new specific error class in the same Checker scope as regression.
- Credit a disjoint validation only as diagnostic when it yields new actionable
  evidence such as a missing signal, attribute, name, module, or import.
- Reject post-terminal events after the first `all_completed` transition.
- Export rule version `verifier_grounded_role_typed_v2.1` in every Episode.

## Current evidence boundary

- Development set: 40 labeled Episodes; initial accuracy 0.8000 and macro-F1
  0.7976. After error-driven revision, resubstitution accuracy/macro-F1 are both
  1.0000. This is not an independent test result.
- Temporal holdout: 40 blind Episodes from run `703598`, stored in
  `episode_credit_v2_holdout_adder_20260716_blind.jsonl`. Independent labels are
  still pending; the set does not contain invalid executions.
- Curated v2.1 preview: 36 strategies from 62 validated and deduplicated
  progress Episodes. It is not enabled in `setting.yaml`.

## New feedback instrumentation

Runtime retrieval now emits `context_reuse_decision` events for inject, miss,
and throttled decisions. Each event preserves strategy IDs, scores, quality,
match reasons, and current verifier failure evidence. The analyzer
`scripts/analyze_context_reuse_decisions.py` links an injection to the next
mutation/validation segment and assigns v2.1 observational credit.

This instrumentation enables the next causal experiment: replay the same stage
checkpoint with context reuse disabled and enabled, then compare verifier delta,
regressions, action cost, tokens, and stage advance. Whole-run completion alone
is too noisy to identify retrieval utility.

## Go/no-go gate

Do not deploy v2.1 into normal Agent runs yet. Proceed to controlled comparison
only after a fresh blind holdout reaches accuracy >= 0.90, progress precision >=
0.90, and zero invalid-to-progress errors. Then run paired stage-level A/B before
the more expensive full-DUT experiment.
