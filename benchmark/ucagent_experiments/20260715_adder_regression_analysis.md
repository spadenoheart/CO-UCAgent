# Adder July 15 Regression Analysis

## Conclusion

One core regression introduced before the July 15 runs is confirmed: the new
test-result classifier treated expected `XFAIL` cases as failed cases during
`create_test_case_templates`. This materially prolonged stage 20. It does not,
however, explain every delay in the two attempts.

## Confirmed regression

- The 11:14 frozen snapshot classified every status other than `PASS/PASSED` as
  a failed case and had no stage-aware exception.
- Stage 20 explicitly requires template tests to end with `assert False, "Not
  implemented"` and the reporter records these expected placeholders as XFAIL.
- At 14:50:42 the runner reported 15 passed, 9 xfailed, zero failed tests, and
  `run_test_success=true`, but the summary still emitted
  `test_outcome=test_failure`.
- At 15:06:19 the same occurred for 3 xfailed tests. The false failure was fed
  back into failure-aware context and classified as
  `test_template_not_implemented`, extending a loop that should have stopped.

The fix allows XFAIL only in `create_test_case_templates`; XPASS and XFAIL in
implementation stages remain failures. The runtime v1 strategy pack itself was
not replaced by the new v2 offline pack.

## Independent causes

- Workload expansion: the completed July 10 run produced 17 checkpoints; the
  11:58 July 15 attempt produced 44, and the 15:08 attempt produced 32. The 44
  checkpoint run therefore created roughly 2.6 times the downstream test-point
  workload of the July 10 run.
- Initial service failure: the first July 15 start repeatedly returned
  `Connection error` before useful stage work began.
- Stage 16 test discovery: the 15:08 attempt generated pytest tests as class
  methods. Pytest executed them, but the UCAgent checker searches module-level
  functions and therefore reported zero env fixture test functions.
- Summary latency: after a 53,551-token history, structured summarization spent
  244.4 seconds on an invalid-JSON response and another 137.4 seconds in the
  main-model fallback.
- The log records repeated SIGINT/PDB interruption events. Those are explicit
  external interrupts handled by `VerifyPDB`, not evidence of an autonomous
  Agent crash.

## Attribution

- Stage 20 delay in the 11:58 attempt: directly aggravated by the classifier
  regression, then amplified by stale/generic context feedback and the larger
  44-checkpoint workload.
- Stage 16 delay in the 15:08 attempt: primarily a generated-test structure
  mismatch plus very long model/summary latency. No evidence shows that the
  context-reuse store made the checker unable to pass; the earlier 11:58 attempt
  passed stage 16 under the same runtime feature set.
- Overall non-completion: mixed cause. It is inaccurate to attribute both runs
  solely to context reuse or solely to model inference.

## Safeguards added

- Stage-aware expected-XFAIL classification and explicit status counts.
- Dedicated `test_structure_or_discovery` failure pattern.
- Prompt guidance requiring module-level env fixture tests with the checker
  naming convention.
- Unit tests for pass/failure/infrastructure/timeout/crash/no-tests outcomes and
  role-typed Episode attribution.
