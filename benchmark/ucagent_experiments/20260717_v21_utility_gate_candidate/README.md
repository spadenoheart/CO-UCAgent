# Experiment Baseline: 20260717_v21_utility_gate_candidate

This directory freezes the v2.1 strategy pack, utility-gate implementation,
configuration, analysis scripts, and focused tests. The worktree was dirty, so
the files under `snapshot/` are the source of truth for this experiment.

## v2.1 gate-off comparison

```bash
UCAGENT_CONTEXT_REUSE_PACK=benchmark/ucagent_context_reuse/context_reuse_credit_v2_rule_v2_curated_preview.json \
UCAGENT_CONTEXT_REUSE_MIN_QUALITY=0.68 \
UCAGENT_CONTEXT_REUSE_UTILITY_GATE=false \
make test_Adder
```

## v2.1 utility-gate candidate

```bash
UCAGENT_CONTEXT_REUSE_PACK=benchmark/ucagent_context_reuse/context_reuse_credit_v2_rule_v2_curated_preview.json \
UCAGENT_CONTEXT_REUSE_MIN_QUALITY=0.68 \
UCAGENT_CONTEXT_REUSE_UTILITY_GATE=true \
UCAGENT_CONTEXT_REUSE_MIN_UTILITY=0.05 \
make test_Adder
```

Keep all other environment variables, model settings, hardware, DUT inputs,
and prompt variants identical between paired runs.

The candidate is enabled only through environment overrides. A single run is
used to verify that the gate changes retrieval and to collect observational
credit; causal uplift requires paired gate-on/gate-off runs or checkpoint replay.

See `snapshot/benchmark/ucagent_context_reuse/context_reuse_v21_utility_gate_run_guide.md`
for the formula, preflight replay, and post-run analysis commands.
