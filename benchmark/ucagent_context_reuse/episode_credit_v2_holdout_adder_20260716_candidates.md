# UCAgent Context Reuse Pack (ucagent_episode_credit_v2_temporal_holdout)

- Trace count: 1
- Run count: 1
- DUTs: Adder
- Candidate episodes: 40
- Effective repair items: 5
- Compression hints: 5
- Credit roles: `{"diagnostic": 5, "no_progress": 19, "progress": 6, "regression": 10}`

## Effective Repair Items
- Adder stage 19 `basic_api_functional_test` pattern=duplicate_label_definition sig=4842ff758a2cd46a action_categories=['bug_document'] actions=[delete_file:unity_test/Adder_bug_analysis.md, write:unity_test/Adder_bug_analysis.md]
- Adder stage 21 `test_case_implementation_in_batch` pattern=coverage_or_test_marks_missing sig=0370663c7a722f4a action_categories=['test_code'] actions=[replace_string:unity_test/tests/test_Adder_templates.py]
- Adder stage 21 `test_case_implementation_in_batch` pattern=test_logic_or_case_implementation sig=4aedbde100d2eefb action_categories=['test_code'] actions=[replace_string:unity_test/tests/test_Adder_templates.py]
- Adder stage 22 `comprehensive_verification_and_bug_analysis` pattern=bug_doc_incomplete_tc_label sig=0a76112a37a9edfc action_categories=['bug_document'] actions=[delete_file:unity_test/Adder_bug_analysis.md, write:unity_test/Adder_bug_analysis.md, delete_file:unity_test/Adder_bug_analysis.md, write:unity_test/Adder_bug_analysis.md]
- Adder stage 24 `generate_random_test_cases` pattern=test_logic_or_case_implementation sig=b3d610a84d7f3386 action_categories=['test_code'] actions=[replace_string:unity_test/tests/test_Adder_random.py]

## Compression Hints
- Adder stage 21 `test_case_implementation_in_batch` pattern=generic_checker_failure sig=dfb3b85227181fa8 repeats=3: keep_latest_failure_summary_and_the_adjacent_action_observation_transition; drop older raw outputs with the same signature
- Adder stage 21 `test_case_implementation_in_batch` pattern=generic_checker_failure sig=a2e35ad08b30a4d6 repeats=2: keep_latest_failure_summary_and_the_adjacent_action_observation_transition; drop older raw outputs with the same signature
- Adder stage 21 `test_case_implementation_in_batch` pattern=bug_doc_incomplete_tc_label sig=b31f4c30bd518ff7 repeats=2: keep_latest_failure_summary_and_the_adjacent_action_observation_transition; drop older raw outputs with the same signature
- Adder stage 22 `comprehensive_verification_and_bug_analysis` pattern=bug_doc_incomplete_tc_label sig=0a76112a37a9edfc repeats=7: keep_latest_failure_summary_and_the_adjacent_action_observation_transition; drop older raw outputs with the same signature
- Adder stage 22 `comprehensive_verification_and_bug_analysis` pattern=bug_doc_passed_tc_marked_as_bug sig=643e593ccc73cf25 repeats=6: keep_latest_failure_summary_and_the_adjacent_action_observation_transition; drop older raw outputs with the same signature
