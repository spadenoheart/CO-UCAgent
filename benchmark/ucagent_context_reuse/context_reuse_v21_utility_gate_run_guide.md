# v2.1 Verifier-Grounded Utility Gate Run Guide

## Scope

This is a simple, interpretable runtime gate. It is not a learned reward model
and it does not establish causal improvement from a single run.

Offline curation computes a historical utility prior:

```text
U_history =
    0.45 * failure_reduction
  + 0.35 * stage_advance
  + 0.20 * information_gain
  - 0.60 * regression
  - 0.15 * action_cost
  - 0.10 * historical_token_cost
  - uncertainty_penalty
```

Runtime retrieval first applies the existing stage, failure-pattern, action,
and quality matching. It then estimates the candidate's Prompt token cost:

```text
U_net = U_history - 0.05 * min(1, estimated_prompt_tokens / 512)
```

Only candidates with `U_net > UCAGENT_CONTEXT_REUSE_MIN_UTILITY` are injected.
Strategies without utility evidence are rejected. Compression hints remain a
separate repetition-control mechanism and are not treated as repair actions.

## Adder Effect-Observation Run

Use a `0.05` safety margin for this first run. This threshold is not claimed to
be optimal. It is deliberately stricter than zero because the current utility
is heuristic and historical token-cost coverage in the v2.1 pack is zero.

```bash
UCAGENT_CONTEXT_REUSE_PACK=benchmark/ucagent_context_reuse/context_reuse_credit_v2_rule_v2_curated_preview.json \
UCAGENT_CONTEXT_REUSE_MIN_QUALITY=0.68 \
UCAGENT_CONTEXT_REUSE_UTILITY_GATE=true \
UCAGENT_CONTEXT_REUSE_MIN_UTILITY=0.05 \
make test_Adder
```

Keep the model endpoint, model weights, Prompt variant, hardware, and all other
environment variables unchanged from the latest completed Adder run.

## Post-Run Analysis

```bash
RUN_ID=$(tail -n 1 output/unity_test/structured_events.jsonl | jq -r '.thread_id')

python scripts/build_trace_tree.py \
  output/unity_test/structured_events.jsonl \
  --run-id "$RUN_ID" \
  --out benchmark/ucagent_trace_tree_v2/Adder_v21_utility_gate_trace_tree.json

python scripts/analyze_context_reuse_decisions.py \
  benchmark/ucagent_trace_tree_v2/Adder_v21_utility_gate_trace_tree.json \
  --out-json benchmark/ucagent_context_reuse/Adder_v21_utility_gate_decisions.json \
  --out-md benchmark/ucagent_context_reuse/Adder_v21_utility_gate_decisions.md
```

The primary implementation checks are:

- `utility_gate_counts.rejected_candidates > 0`: the gate actually filtered candidates.
- `decision_counts` and per-event `search_summary`: filtering changed injection decisions.
- `credit_role_counts`: the next verifier result after injection was progress,
  diagnostic, no-progress, regression, or invalid.
- Stage 19/21/22 loop count, total injected Prompt items, total tokens, and total
  runtime versus the frozen gate-off comparison.

The progress/regression rates from one run are observational. A paper-level
claim requires paired runs or stage-checkpoint replay with gate on/off.

## Preflight Replay

Replaying the latest completed Adder trace (`run_703598`) against the current
v2.1 pack produced the following non-causal implementation forecast:

| Threshold | Unique failure queries | Queries with changed top-2 | Injected items, off -> on |
|---:|---:|---:|---:|
| 0.00 | 26 | 0 | 48 -> 48 |
| 0.05 | 26 | 8 | 48 -> 45 |

At `0.05`, changed queries occurred at stage 19, stage 21, and stage 22. This
replay only verifies that the gate is strong enough to alter retrieval; it does
not predict that the new decisions will improve completion time.
