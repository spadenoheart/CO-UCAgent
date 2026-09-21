# Experiment Baseline: 20260717_v22_semantic_gate_candidate

This snapshot freezes the semantic-admission gates added after the completed
Adder v2.1 utility-gate run. The worktree was dirty, so `snapshot/` and the
hashes in `manifest.json` define the candidate implementation.

## Observed v2.1 configuration

The following disables the new semantic gates while retaining the historical
verifier-utility gate used by `Adder+20260717_104648`:

```bash
UCAGENT_CONTEXT_REUSE_PACK=benchmark/ucagent_context_reuse/context_reuse_credit_v2_rule_v2_curated_preview.json \
UCAGENT_CONTEXT_REUSE_MIN_QUALITY=0.68 \
UCAGENT_CONTEXT_REUSE_UTILITY_GATE=true \
UCAGENT_CONTEXT_REUSE_MIN_UTILITY=0.05 \
UCAGENT_CONTEXT_REUSE_STRICT_ACTION_MATCH=false \
UCAGENT_CONTEXT_REUSE_EXACT_PATTERN_GROUPS=false \
UCAGENT_CONTEXT_REUSE_GENERIC_SIGNATURE=false \
UCAGENT_CONTEXT_REUSE_SKIP_NON_REUSABLE=false \
UCAGENT_CONTEXT_REUSE_STRICT_HINT_MATCH=false \
UCAGENT_CONTEXT_REUSE_ALLOW_HINTS_ONLY=true \
UCAGENT_CONTEXT_REUSE_DESTRUCTIVE_SUPPORT_GATE=false \
make test_Adder
```

## v2.2 semantic-gate candidate

```bash
UCAGENT_CONTEXT_REUSE_PACK=benchmark/ucagent_context_reuse/context_reuse_credit_v2_rule_v2_curated_preview.json \
UCAGENT_CONTEXT_REUSE_MIN_QUALITY=0.68 \
UCAGENT_CONTEXT_REUSE_UTILITY_GATE=true \
UCAGENT_CONTEXT_REUSE_MIN_UTILITY=0.05 \
make test_Adder
```

The candidate adds five independently switchable controls: exact failure
subtype matching, current-action compatibility, generic-failure signature
matching, non-repairable workflow-failure suppression, and low-support
destructive-action suppression. Compression hints also require a strict match,
and hints alone cannot trigger an injection.

## Evidence boundary

`Adder_20260717_v21_gate_replay.json` is a counterfactual retrieval replay. It
shows which historical Prompt injections would change, but it is not a causal
runtime or task-quality result. Use paired stage-checkpoint replay before a full
DUT claim.
