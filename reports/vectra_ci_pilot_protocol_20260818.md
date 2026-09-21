# VECTRA-CI Pilot Protocol

Date: 2026-08-18

## Purpose

This pilot validates three claims in order. It must not be reported as a
held-out cross-DUT result because the current strategy pack contains historical
Adder episodes.

1. `shadow` instrumentation does not change retrieval order or the Prompt.
2. Contract applicability agrees with blinded human labels.
3. Contract enforcement improves matched-stage efficiency without losing stage
   completion or adding regression/invalid verifier executions.

## 1. Shadow data collection

Run from the `UCAgent` repository. Start with the reference seed; add the other
two continuation seeds only when the first run is valid.

```bash
PY=${PYTHON:-python3.11}
MATRIX=benchmark/ucagent_experiments/adder_vectra_ci_stage_pair_20260818_matrix.json

$PY scripts/run_adder_stage_benchmarks.py run \
  --matrix "$MATRIX" \
  --arm CONTRACT_SHADOW \
  --label vectra_shadow_s20260721 \
  --stage 21 --stage 23 --stage 28 \
  --seed 20260721 \
  --timeout-hours 6
```

For seeds `20260722` and `20260723`, use a distinct label and add
`--allow-seed-override`. These are continuation-seed repeats from the same
frozen checkpoint, not three independently generated full-run histories.

Build one trace and one observational analysis from the shadow run root:

```bash
cd "$(git rev-parse --show-toplevel)"

$PY scripts/build_trace_tree.py \
  ../UCAgent/benchmark/ucagent_experiments/adder_stage_benchmark_runs \
  --out benchmark/ucagent_context_reuse/vectra_shadow_trace.json

$PY scripts/analyze_context_reuse_decisions.py \
  benchmark/ucagent_context_reuse/vectra_shadow_trace.json \
  --pack benchmark/ucagent_context_reuse/context_reuse_credit_v2_rule_v2_curated_preview.json \
  --out-json benchmark/ucagent_context_reuse/vectra_shadow_analysis.json \
  --out-md benchmark/ucagent_context_reuse/vectra_shadow_analysis.md
```

## 2. Balanced blinded audit

```bash
$PY scripts/audit_contract_shadow.py sample \
  benchmark/ucagent_context_reuse/vectra_shadow_analysis.json \
  --pack benchmark/ucagent_context_reuse/context_reuse_credit_v2_rule_v2_curated_preview.json \
  --size 50 --seed 20260818 \
  --blind-out benchmark/ucagent_context_reuse/vectra_contract_audit_50_blind.jsonl \
  --key-out benchmark/ucagent_context_reuse/vectra_contract_audit_50_key.jsonl
```

Make two copies of the blind file. Annotators independently fill
`annotations[].applicability` with `applicable`, `inapplicable`, or `uncertain`.
They must not see the key file or subsequent verifier outcome. Evaluate both
files together:

```bash
$PY scripts/audit_contract_shadow.py evaluate \
  benchmark/ucagent_context_reuse/vectra_contract_audit_agent_a.jsonl \
  benchmark/ucagent_context_reuse/vectra_contract_audit_agent_b.jsonl \
  --key benchmark/ucagent_context_reuse/vectra_contract_audit_50_key.jsonl \
  --out benchmark/ucagent_context_reuse/vectra_contract_audit_metrics.json
```

Enforcement gate: at least 50 binary-labeled cases, applicable precision point
estimate at least 0.90, and no known regression/invalid item classified as safe.
The Wilson interval is reported separately; a 0.90 point estimate does not mean
its lower confidence bound is 0.90.

## 3. Matched checkpoint replay

First test instrumentation non-interference (`off` versus `shadow`). Then test
the actual intervention (`off` versus `enforce`). The runner alternates arm
order and resumes completed pairs automatically.

```bash
cd "${UPSTREAM_UCAGENT_ROOT:?Set UPSTREAM_UCAGENT_ROOT to an upstream UCAgent checkout}"

$PY scripts/run_adder_stage_benchmarks.py run-pair \
  --matrix "$MATRIX" \
  --control-arm TYPED_REUSE_CONTRACT_OFF \
  --treatment-arm CONTRACT_ENFORCE \
  --pair-label vectra_contract_v0 \
  --stage 21 --stage 23 --stage 28 \
  --seed 20260721 --seed 20260722 --seed 20260723 \
  --allow-seed-override \
  --order alternating \
  --run-root benchmark/ucagent_experiments/adder_vectra_ci_stage_pair_runs \
  --timeout-hours 6
```

Outputs are updated after every completed pair:

- `stage_results.csv`: raw arm results;
- `paired_results.csv`: matched deltas;
- `paired_results.json`: input to policy admission;
- `ledger.json`: resumable execution state.

## 4. Candidate admission

Run from `CO-UCAgent`:

```bash
$PY scripts/evaluate_contract_policy_admission.py \
  ../UCAgent/benchmark/ucagent_experiments/adder_vectra_ci_stage_pair_runs/paired_results.json \
  --policy-id vectra_contract_v0 \
  --min-pairs 9 \
  --min-utility-lcb 0 \
  --max-harmful-rate 0.10 \
  --out benchmark/ucagent_context_reuse/vectra_contract_v0_admission.json \
  --registry benchmark/ucagent_context_reuse/vectra_policy_registry.json
```

Possible decisions:

- `admit`: quality passes, utility lower bound is positive, and the harmful-rate
  Wilson upper bound is below threshold;
- `hold`: no observed quality failure, but evidence is insufficient;
- `reject`: completion, regression, invalid-test, or harmful-rate constraint is
  violated.

The registry does not silently replace the deployed champion. Promotion is an
explicit frozen-matrix change so every reported run remains reproducible.

## 5. Claim boundary

Adder replay is a debugging pilot. Paper-level evaluation requires rebuilding
the pack from source DUTs only, freezing a 6/2/2 source/dev/held-out split, and
running primary comparisons on held-out DUTs with three seeds. Multiple stage
checkpoints from one DUT are correlated diagnostic cases, not independent DUT
samples.
