# Experiment Baseline: 20260715_test_outcome_context_reuse

This directory freezes the code/configuration used to compare the original
context-reuse pack with the curated v1 pack. The worktree was dirty, so the
files under `snapshot/` are the source of truth for this experiment.

## v0 baseline

```bash
UCAGENT_CONTEXT_REUSE_PACK=benchmark/ucagent_context_reuse/context_reuse_v0.json \
UCAGENT_CONTEXT_REUSE_MIN_QUALITY=0 \
make test_Adder
```

## v1 candidate

```bash
UCAGENT_CONTEXT_REUSE_PACK=benchmark/ucagent_context_reuse/context_reuse_v1.json \
UCAGENT_CONTEXT_REUSE_MIN_QUALITY=0.68 \
make test_Adder
```

Keep all other environment variables, model settings, hardware, DUT inputs,
and prompt variants identical between paired runs.
