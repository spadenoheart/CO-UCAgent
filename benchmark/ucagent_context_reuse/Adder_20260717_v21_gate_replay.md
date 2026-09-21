# Context Reuse Gate Replay

This is a counterfactual retrieval audit, not a causal agent-quality result.

- Injections replayed: 32
- Changed injections: 23
- Fully suppressed injections: 15
- Prompt items: 85 -> 42
- Prompt-item reduction rate: 0.5059
- Hard-gate reasons: `{"action_category_mismatch": 128, "exact_failure_pattern_mismatch": 284, "generic_signature_mismatch": 264, "low_support_destructive_action": 55, "non_reusable_failure": 108}`

## Changed Queries

- stage 2 `dut_function_grouping` / `duplicate_label_definition`: strategies ['strategy-b38ed156ba205f17', 'strategy-515d94b32a14f390'] -> ['strategy-b38ed156ba205f17']; hints 0 -> 0
- stage 16 `evaluate_env_fixture` / `api_or_env_interface_mismatch`: strategies ['strategy-2ec888516858a1a4', 'strategy-9e5938bcad3f5e77'] -> ['strategy-2ec888516858a1a4', 'strategy-7cfa75646098dfcb']; hints 1 -> 1
- stage 16 `evaluate_env_fixture` / `reference_files_unread`: strategies ['strategy-2ec888516858a1a4', 'strategy-3b80ef5e23c03bd0'] -> []; hints 1 -> 0
- stage 16 `evaluate_env_fixture` / `api_or_env_interface_mismatch`: strategies ['strategy-2ec888516858a1a4', 'strategy-9e5938bcad3f5e77'] -> ['strategy-2ec888516858a1a4', 'strategy-7cfa75646098dfcb']; hints 1 -> 1
- stage 16 `evaluate_env_fixture` / `generic_checker_failure`: strategies ['strategy-2ec888516858a1a4', 'strategy-9e5938bcad3f5e77'] -> []; hints 1 -> 0
- stage 19 `basic_api_functional_test` / `reference_files_unread`: strategies ['strategy-ff57f9d87efe3ebc', 'strategy-2ec888516858a1a4'] -> []; hints 1 -> 0
- stage 19 `basic_api_functional_test` / `generic_checker_failure`: strategies ['strategy-ff57f9d87efe3ebc'] -> []; hints 1 -> 0
- stage 19 `basic_api_functional_test` / `generic_checker_failure`: strategies ['strategy-ff57f9d87efe3ebc'] -> []; hints 1 -> 0
- stage 19 `basic_api_functional_test` / `bug_doc_passed_tc_marked_as_bug`: strategies ['strategy-515d94b32a14f390', 'strategy-9b9cc99fbf6f4ba4'] -> []; hints 1 -> 0
- stage 19 `basic_api_functional_test` / `bug_doc_incomplete_tc_label`: strategies ['strategy-515d94b32a14f390', 'strategy-9b9cc99fbf6f4ba4'] -> []; hints 1 -> 0
- stage 21 `test_case_implementation_in_batch` / `generic_checker_failure`: strategies ['strategy-adc1351791d43323'] -> []; hints 1 -> 0
- stage 21 `test_case_implementation_in_batch` / `label_or_report_mismatch`: strategies ['strategy-061c267e6fc8cb5e', 'strategy-515d94b32a14f390'] -> ['strategy-061c267e6fc8cb5e']; hints 1 -> 1
- stage 21 `test_case_implementation_in_batch` / `generic_checker_failure`: strategies ['strategy-adc1351791d43323'] -> []; hints 1 -> 0
- stage 22 `comprehensive_verification_and_bug_analysis` / `generic_checker_failure`: strategies ['strategy-515d94b32a14f390', 'strategy-9b9cc99fbf6f4ba4'] -> []; hints 1 -> 0
- stage 22 `comprehensive_verification_and_bug_analysis` / `bug_doc_passed_tc_marked_as_bug`: strategies ['strategy-515d94b32a14f390', 'strategy-9b9cc99fbf6f4ba4'] -> []; hints 1 -> 0
- stage 22 `comprehensive_verification_and_bug_analysis` / `bug_doc_schema_or_marking`: strategies ['strategy-9b9cc99fbf6f4ba4', 'strategy-0d36510279b1f76d'] -> ['strategy-0d36510279b1f76d', 'strategy-9174d99e9137c8d1']; hints 1 -> 1
- stage 22 `comprehensive_verification_and_bug_analysis` / `bug_doc_incomplete_tc_label`: strategies ['strategy-515d94b32a14f390', 'strategy-9b9cc99fbf6f4ba4'] -> []; hints 1 -> 1
- stage 22 `comprehensive_verification_and_bug_analysis` / `generic_checker_failure`: strategies ['strategy-515d94b32a14f390', 'strategy-9b9cc99fbf6f4ba4'] -> []; hints 1 -> 0
- stage 22 `comprehensive_verification_and_bug_analysis` / `bug_doc_schema_or_marking`: strategies ['strategy-9b9cc99fbf6f4ba4', 'strategy-0d36510279b1f76d'] -> ['strategy-0d36510279b1f76d', 'strategy-9174d99e9137c8d1']; hints 1 -> 1
- stage 24 `generate_random_test_cases` / `test_logic_or_case_implementation`: strategies ['strategy-f18405e114f991e8', 'strategy-adc1351791d43323'] -> ['strategy-f18405e114f991e8', 'strategy-adc1351791d43323']; hints 1 -> 0
- stage 24 `generate_random_test_cases` / `generic_checker_failure`: strategies [] -> []; hints 1 -> 0
- stage 24 `generate_random_test_cases` / `duplicate_label_definition`: strategies ['strategy-b38ed156ba205f17', 'strategy-515d94b32a14f390'] -> []; hints 1 -> 0
- stage 25 `verification_review_and_summary` / `reference_files_unread`: strategies ['strategy-2ec888516858a1a4', 'strategy-e379477472827c18'] -> []; hints 0 -> 0
