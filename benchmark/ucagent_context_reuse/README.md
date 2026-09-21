# Context-Reuse Research Artifacts

This directory contains compact repair packs, Episode-credit annotations,
historical gate decisions, and replay metrics used by the research prototype.

The public copy replaces machine-specific repository roots with
`<REPO_ROOT>`. Raw Agent logs, full LLM inputs, generated DUT workspaces, model
weights, and private RTL are intentionally excluded. Historical replay results
measure retrieval behavior only; they are not causal evidence of end-to-end
speedup.

Key files:

- `context_reuse_v0.json`: default runtime-compatible repair pack.
- `context_reuse_credit_v2_rule_v2_curated_preview.json`: verifier-credit
  curated candidate pack.
- `episode_credit_v2_research_protocol.md`: attribution protocol.
- `episode_credit_v2_annotation_metrics.json`: initial independent audit.
- `episode_credit_v2_holdout_adder_20260716_metrics.json`: temporal holdout
  audit.
- `Adder_20260717_v21_gate_replay.json`: v2.2 counterfactual gate replay.
