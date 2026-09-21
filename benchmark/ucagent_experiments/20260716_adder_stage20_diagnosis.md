# Adder Stage 20 interruption diagnosis (2026-07-16)

## Scope

- Log directory: `log/Adder+20260715_114538`
- Target thread: `207655`
- Target run started at 2026-07-16 11:01 and was interrupted at 14:58:25.
- Stage 20 (`create_test_case_templates`) started at 12:35:55 and never advanced before interruption.
- Events after 14:59 belong to a new thread (`854089`) and are excluded.

## Stage 20 measurements

| Metric | Interrupted run | Successful reference (`Adder+20260710_140927`) |
| --- | ---: | ---: |
| Approximate stage wall time | 2 h 22 min | 5 min 12 s |
| LLM requests | 192 | 14 |
| Sum of LLM latency | 140.87 min | 6.07 min |
| Sum of TTFT | 107.83 min | 3.98 min |
| Prompt tokens | 6,584,904 | 200,594 |
| Maximum prompt tokens/request | 101,345 | 34,977 |
| File mutations | 71 | not used for this comparison |
| Test runs | 25 (20 failure, 4 pass, 1 no-tests) | completed normally |
| Check calls | 8, all failed | passed |

The long wall time was therefore amplified by model inference and prefill, but the
request explosion was caused by a repair loop rather than by one abnormally slow
inference call.

## Causal evidence

1. The original Stage 20 task and Prompt v1 stage rule correctly required skeletons
   with `assert False, "Not implemented"`, and explicitly prohibited real test logic.
2. The first `Check` ran those templates, pytest exited with code 1, and the checker
   rejected the result before accepting the template structure.
3. Context reuse classified this expected template state as
   `test_template_not_implemented`, then injected two Stage 21
   (`test_case_implementation_in_batch`) episodes whose successful actions replaced
   test code and made tests pass.
4. Prompt v1's checker summary then instructed the model to remove placeholders and
   implement the current batch. The next model response explicitly adopted this
   interpretation.
5. The model subsequently oscillated between making all tests pass and restoring
   expected failures. Full-file runs reached 10/10 pass twice, but `Check` still
   failed because Stage 20 requires templates. Later rewrites regressed to 10/10
   failures without satisfying the checker contract.
6. The model also attempted one write to `Adder/Adder.v`; the write guard rejected it.

## Attribution

| Candidate cause | Assessment | Evidence |
| --- | --- | --- |
| Prompt v1 checker summary | Primary direct cause | Stage-agnostic `Not implemented` advice contradicted the Stage 20 rule and was immediately followed by real test implementation. |
| Context-reuse strategy retrieval | Secondary direct cause | Stage 21 repair episodes were injected into Stage 20 because same-stage matching is only a score bonus, not a compatibility constraint. |
| Template checker/runner contract | Underlying enabling defect | The checker first rejects a non-zero pytest exit, but later requires template tests to fail with `Not implemented`. |
| Model randomness/seed | Trigger and variance source only | It influenced the initial template form and subsequent choices, but cannot explain the deterministic contradictory feedback. |
| LLM inference latency | Runtime multiplier, not semantic root cause | 192 requests accumulated 140.87 minutes of model latency and 107.83 minutes of TTFT. |

## Additional contamination

After interruption, the new Stage 2 run received long-term-memory failure entries
from Stage 20. This did not cause the analyzed Stage 20 loop, but it shows that the
current memory injection path can leak stage-incompatible context into later runs.

## Required fixes before another full Adder run

1. Make checker summaries stage-aware. In Stage 20, `Not implemented` is expected and
   must never produce an instruction to implement test logic.
2. Add a hard stage-role compatibility gate to context reuse. Template-generation
   failures must not retrieve test-implementation episodes.
3. Reconcile the template checker with pytest exit semantics: validate a generated
   structured report containing expected placeholder failures before rejecting the
   non-zero process exit.
4. Apply the same stage-role gate to long-term-memory final injection.
5. Validate Stage 20 with a focused replay/A-B matrix before spending another full
   Adder run: Prompt v0/v1 x context reuse off/on, using the same Stage 20 input.

