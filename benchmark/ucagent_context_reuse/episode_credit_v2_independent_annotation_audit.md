# Episode Credit v2 Independent Annotation Audit

## Blind annotation procedure

- The formal annotation used a prediction-free export:
  `episode_credit_v2_annotation_40_blind.jsonl`.
- The annotator was an isolated sub-agent with no conversation context and was
  allowed to read only the blind sample and research protocol.
- Its raw output is `episode_credit_v2_independent_agent_b.jsonl` and has been
  merged into the first `annotations` slot of
  `episode_credit_v2_annotation_40.jsonl`.
- An earlier trial is excluded because `evidence.credit_role` and
  `evidence.confidence` remained in the first blind export. The excluded output
  is retained as `episode_credit_v2_independent_agent_a_contaminated.jsonl`.

## Label distribution

| Role | Count |
|---|---:|
| progress | 8 |
| diagnostic | 8 |
| no_progress | 10 |
| regression | 6 |
| invalid | 8 |

## Agreement with the rule-based attribution

- Records: 40/40 labeled
- Accuracy: 0.8000
- Macro-F1: 0.7976
- Progress precision/recall/F1: 0.7500 / 0.7500 / 0.7500
- Diagnostic precision/recall/F1: 1.0000 / 1.0000 / 1.0000
- No-progress precision/recall/F1: 0.7500 / 0.6000 / 0.6667
- Regression precision/recall/F1: 0.5000 / 0.6667 / 0.5714
- Invalid precision/recall/F1: 1.0000 / 1.0000 / 1.0000
- Cohen's kappa: unavailable because only one valid independent annotator has
  completed the sample.

The 0.90 acceptance target is not met. The v2 pack must remain offline.

## Disagreements

| Episode | Rule | Blind label | Main reason |
|---|---|---|---|
| `credit-dd020203b0c11d1b` | no_progress | regression | Same Check scope retained all five failures but introduced a new document parse failure. |
| `credit-1c0cd4b7b769ecc8` | progress | no_progress | Validation narrowed from a full file to one still-failing test; unexecuted failures cannot be counted as resolved. |
| `credit-3da9a53cced062bb` | progress | no_progress | Same target-narrowing problem, with no intervening file mutation. |
| `credit-d246fe0c420ded65` | regression | progress | Validation expanded from one test to a full file; newly observed failures are not proven regressions. |
| `credit-719b24b0430c98d1` | no_progress | regression | Same Check scope retained failures and changed to a new incomplete-TC parse error. |
| `credit-5168b89694c307c8` | regression | no_progress | Only a generic signature changed; no concrete new failure establishes regression. |
| `credit-c647499516b7de12` | regression | no_progress | Core pytest failure stayed unchanged; a new generic signature is insufficient regression evidence. |
| `credit-3fac9d7c33a61142` | regression | progress | Validation expanded from one test to a full file; additional failures are scope expansion, not demonstrated regressions. |

## Rule defects exposed

1. Test-scope comparability is too coarse. File-level and single-test targets
   are currently treated as comparable even when one strictly contains the
   other.
2. Regression detection over-trusts normalized signatures. A new generic
   signature should not count as a regression without a concrete new case,
   checkpoint, category, or error class.
3. Check-result regression needs semantic error deltas. Equal checkpoint counts
   can still hide a new document/schema failure and should not automatically be
   treated as no progress.

