# Test Layout

The automated release gate covers current upstream functionality and the
context-reuse research modules. Historical interactive probes and tests for
removed APIs are retained under `manual_*.py` or `legacy_*.py`, so pytest does
not collect them as current contract tests.

Some legacy checker tests require generated `output/` data or the removed
`examples/ALU` fixture. They explicitly skip when those local artifacts are not
available. This is distinct from passing the test.

Run the research-critical suite with:

```bash
PYTHONPATH="$PWD" python -m pytest -q \
  tests/test_test_result_classification.py \
  tests/test_context_reuse_curation.py \
  tests/test_episode_credit.py \
  tests/test_experiment_baseline.py \
  tests/test_message_lifecycle.py
```

The synchronized publication tree was verified with `python -m pytest -q`:
357 tests passed and 7 fixture-dependent tests were skipped on 2026-07-21.
