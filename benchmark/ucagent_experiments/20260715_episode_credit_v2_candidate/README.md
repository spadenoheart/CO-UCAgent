# Experiment Baseline: 20260715_episode_credit_v2_candidate

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

## v2 offline candidate

The role-typed v2 pack is frozen for annotation and analysis only. Do not point
`UCAGENT_CONTEXT_REUSE_PACK` at the v2 preview until the manual attribution gate
in `episode_credit_v2_research_protocol.md` passes.
