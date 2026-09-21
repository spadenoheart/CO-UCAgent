# Experiment Baseline: 20260721_upstream_sync_v22_candidate

This directory publishes the manifest and commands for the post-upstream-sync
B0-B3 context-reuse comparison. The internal worktree was dirty, so the
manifest records hashes of the exact frozen files. The original `snapshot/`
contains local paths from historical traces and is intentionally omitted from
the public repository; the corresponding source files in this repository are
the path-redacted publication copy.

## B0: no context reuse

```bash
make test_Adder ARGS='--override context_upgrade.enable_context_reuse=false'
```

## B1: v1 retrieval

```bash
UCAGENT_CONTEXT_REUSE_PACK=benchmark/ucagent_context_reuse/context_reuse_v1.json \
UCAGENT_CONTEXT_REUSE_MIN_QUALITY=0.68 \
make test_Adder
```

## B2: v2.1 utility gate

Use the `B2_v21_utility_gate.environment` mapping in `manifest.json`. This arm
uses the role-typed pack and utility gate, while disabling the v2.2 semantic
and destructive-action hard gates to reproduce the observed v2.1 behavior.

## B3: v2.2 semantic/risk gate

```bash
UCAGENT_CONTEXT_REUSE_PACK=benchmark/ucagent_context_reuse/context_reuse_credit_v2_rule_v2_curated_preview.json \
UCAGENT_CONTEXT_REUSE_MIN_QUALITY=0.68 \
UCAGENT_CONTEXT_REUSE_UTILITY_GATE=true \
UCAGENT_CONTEXT_REUSE_MIN_UTILITY=0.05 \
make test_Adder
```

Keep the model, hardware, prompt variant, DUT input, seed, timeout, and all
non-context-reuse settings identical between paired runs. Record unsuccessful
and interrupted runs rather than dropping them. The historical gate replay is
retrieval evidence only; it is not a causal end-to-end performance result.

Before a full-DUT run, execute the parser, curation, and retrieval unit tests
from this snapshot. The role attribution still requires holdout validation;
the frozen v2.2 arm is a research candidate, not a proven improvement.
