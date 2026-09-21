# UCAgent Failure Pattern Report

- Trace count: 4
- DUTs: ALU754, Adder, FSM, uart_tx

## Pattern Counts
- api_or_env_interface_mismatch: 63
- test_template_not_implemented: 52
- bug_doc_schema_or_marking: 47
- test_logic_or_case_implementation: 43
- coverage_or_test_marks_missing: 40
- label_or_report_mismatch: 9
- missing_assert: 4
- bug_doc_incomplete_tc_label: 4
- duplicate_label_definition: 2
- test_import_error: 1
- generic_checker_failure: 1
- bug_doc_passed_tc_marked_as_bug: 1

## High-Failure Stages
- FSM stage 21 `test_case_implementation_in_batch`: score=299, mutations=117, test_runs=28, check_failures=24, failure_signatures=10
- uart_tx stage 21 `test_case_implementation_in_batch`: score=249, mutations=101, test_runs=25, check_failures=17, failure_signatures=10
- ALU754 stage 22 `comprehensive_verification_and_bug_analysis`: score=157, mutations=71, test_runs=5, check_failures=16, failure_signatures=4
- uart_tx stage 19 `basic_api_functional_test`: score=144, mutations=62, test_runs=4, check_failures=11, failure_signatures=10
- Adder stage 21 `test_case_implementation_in_batch`: score=126, mutations=49, test_runs=7, check_failures=9, failure_signatures=9
- ALU754 stage 19 `basic_api_functional_test`: score=114, mutations=28, test_runs=14, check_failures=7, failure_signatures=10
- FSM stage 16 `evaluate_env_fixture`: score=109, mutations=43, test_runs=18, check_failures=0, failure_signatures=10
- uart_tx stage 16 `evaluate_env_fixture`: score=106, mutations=19, test_runs=14, check_failures=8, failure_signatures=9
- uart_tx stage 22 `comprehensive_verification_and_bug_analysis`: score=102, mutations=20, test_runs=18, check_failures=4, failure_signatures=10
- ALU754 stage 16 `evaluate_env_fixture`: score=94, mutations=21, test_runs=12, check_failures=7, failure_signatures=7
- FSM stage 19 `basic_api_functional_test`: score=79, mutations=19, test_runs=11, check_failures=2, failure_signatures=10
- Adder stage 19 `basic_api_functional_test`: score=62, mutations=16, test_runs=5, check_failures=3, failure_signatures=8
- FSM stage 25 `verification_review_and_summary`: score=57, mutations=11, test_runs=3, check_failures=7, failure_signatures=4
- FSM stage 24 `generate_random_test_cases`: score=55, mutations=12, test_runs=8, check_failures=3, failure_signatures=5
- ALU754 stage 20 `create_test_case_templates`: score=50, mutations=17, test_runs=3, check_failures=3, failure_signatures=5
- uart_tx stage 24 `generate_random_test_cases`: score=47, mutations=16, test_runs=6, check_failures=1, failure_signatures=5
- FSM stage 20 `create_test_case_templates`: score=45, mutations=20, test_runs=2, check_failures=3, failure_signatures=3
- uart_tx stage 20 `create_test_case_templates`: score=35, mutations=14, test_runs=2, check_failures=2, failure_signatures=3
- Adder stage 16 `evaluate_env_fixture`: score=30, mutations=5, test_runs=4, check_failures=2, failure_signatures=3
- ALU754 stage 18 `basic_api_implementation`: score=30, mutations=14, test_runs=3, check_failures=1, failure_signatures=2

## Cross-DUT Common Stages
- `basic_api_functional_test`: duts=ALU754,Adder,FSM,uart_tx, score=399, check_failures=23, mutations=125
- `evaluate_env_fixture`: duts=ALU754,Adder,FSM,uart_tx, score=339, check_failures=17, mutations=88
- `comprehensive_verification_and_bug_analysis`: duts=ALU754,Adder,FSM,uart_tx, score=303, check_failures=24, mutations=108
- `generate_random_test_cases`: duts=ALU754,Adder,FSM,uart_tx, score=142, check_failures=5, mutations=39
- `create_test_case_templates`: duts=ALU754,Adder,FSM,uart_tx, score=132, check_failures=8, mutations=53
- `verification_review_and_summary`: duts=ALU754,Adder,FSM,uart_tx, score=65, check_failures=7, mutations=14
- `basic_api_implementation`: duts=ALU754,Adder,FSM,uart_tx, score=58, check_failures=2, mutations=26
- `coverage_group_creation`: duts=ALU754,Adder,FSM,uart_tx, score=42, check_failures=3, mutations=21
- `dut_function_grouping`: duts=ALU754,Adder,FSM,uart_tx, score=34, check_failures=2, mutations=20
- `bundle_wrapper_design`: duts=ALU754,Adder,FSM,uart_tx, score=23, check_failures=1, mutations=16
- `requirement_analysis_and_planning`: duts=ALU754,Adder,FSM,uart_tx, score=9, check_failures=0, mutations=9
- `dut_function_understanding`: duts=ALU754,Adder,FSM,uart_tx, score=5, check_failures=0, mutations=5
- `test_case_implementation_in_batch`: duts=Adder,FSM,uart_tx, score=674, check_failures=50, mutations=267
- `dut_creation_implementation`: duts=ALU754,FSM,uart_tx, score=32, check_failures=2, mutations=18
- `implemente_function_checks_in_batch`: duts=ALU754,Adder, score=37, check_failures=4, mutations=7
- `pytest_fixture_dut_implementation`: duts=ALU754,uart_tx, score=5, check_failures=0, mutations=3

## Failure Examples
### coverage_or_test_marks_missing
- Adder stage 10 `implemente_function_checks_in_batch` check_result sig=8ec1ff58145dede1; cases=; error=Please fix the issues reported in 'check_info.last_msg.error' according to the suggestions, and then use the `Check` tool again to re-validate your work.
- Adder stage 19 `basic_api_functional_test` test_run sig=c35b38452bbdb805; cases=unity_test/tests/test_Adder_api_coverage.py:94-120::test_api_Adder_step_falling, unity_test/tests/test_Adder_api_coverage.py:123-148::test_api_Adder_carry_in_1, unity_test/tests/test_Adder_api_coverage.py:205-223::test_api_Adder_carry_out_carry_output; error=FAILED
- Adder stage 19 `basic_api_functional_test` test_run sig=856c6dcff9e60fdf; cases=unity_test/tests/test_Adder_api_coverage.py:94-120::test_api_Adder_step_falling; error=FAILED; =================================== FAILURES =================================== _________________________ test_api_Adder_step_falling __________________________ env = <Adder_api.A
- uart_tx stage 9 `coverage_group_creation` check_result sig=51dcf863397e7fc5; cases=; error=Error occurred during check: Failed to import and process <REPO_ROOT>/output/unity_test/tests/uart_tx_function_coverage_def.py: invalid synt; Please fix the issues reported in 'check_info.last_msg.error' according to the suggestions, and then use the `Check` tool again to re-validate your work.
- uart_tx stage 16 `evaluate_env_fixture` check_result sig=854528cef187ccee; cases=; error=Insufficient env fixture test coverage: 0 env test functions found, minimum required is 1. You have defined 0 env test functions:  in file 'unity_test/tests/test_uart_tx_env_fixtur; Please fix the issues reported in 'check_info.last_msg.error' according to the suggestions, and then use the `Check` tool again to re-validate your work.

### api_or_env_interface_mismatch
- Adder stage 16 `evaluate_env_fixture` test_run sig=029e5ca51d7d638b; cases=unity_test/tests/test_Adder_env_fixture.py:15-36::test_env_inputs_initialization, unity_test/tests/test_Adder_env_fixture.py:39-53::test_env_outputs_access, unity_test/tests/test_Adder_env_fixture.py:56-75::test_env_write_inputs; error=FAILED
- Adder stage 16 `evaluate_env_fixture` check_result sig=b6e0a9b84dd1f35f; cases=; error=You need use tool `ReadTextFile` to read and understand the reference files; Please fix the issues reported in 'check_info.last_msg.error' according to the suggestions, and then use the `Check` tool again to re-validate your work.
- Adder stage 16 `evaluate_env_fixture` check_result sig=631936568068a6f5; cases=; error=The Env test function 'test_env_cin_toggle' name must start with 'test_api_Adder_env_', but got 'test_env_cin_toggle'.; Please fix the issues reported in 'check_info.last_msg.error' according to the suggestions, and then use the `Check` tool again to re-validate your work.
- Adder stage 18 `basic_api_implementation` test_run sig=e67b1702c7f12bfb; cases=unity_test/tests/test_Adder_api_basic.py:7-34::test_api_Adder_add_basic, unity_test/tests/test_Adder_api_basic.py:37-58::test_api_Adder_add_small_numbers, unity_test/tests/test_Adder_api_basic.py:61-83::test_api_Adder_add_identity; error=FAILED; =================================== FAILURES =================================== ___________________________ test_api_Adder_add_basic ___________________________ env = <Adder_api.A
- Adder stage 19 `basic_api_functional_test` test_run sig=e9e54f23012dc105; cases=unity_test/tests/test_Adder_api_basic.py:114-136::test_api_Adder_add_large_numbers; error=FAILED; =================================== FAILURES =================================== _______________________ test_api_Adder_add_large_numbers _______________________ env = <Adder_api.A

### bug_doc_schema_or_marking
- Adder stage 19 `basic_api_functional_test` check_result sig=dcf36334510c98ab; cases=unity_test/tests/test_Adder_api_basic.py:114-136::test_api_Adder_add_large_numbers; error=Bug analysis documentation parsing failed for file 'unity_test/Adder_bug_analysis.md': At line (11): Found TC tag '<TC-' but its parent BG tag '<BG-' was not found in previous line; You must use the format <FG-GROUP>, <FC-FUNCTION>, <CK-CHECK>, <BG-NAME-XX>, <TC-FAILEDTESTCASE>.
- Adder stage 21 `test_case_implementation_in_batch` check_result sig=3a4096c0d3187c55; cases=unity_test/tests/test_Adder_templates.py:161-190::test_api_step_falling, unity_test/tests/test_Adder_templates.py:281-300::test_arithmetic_add_large_numbers, unity_test/tests/test_Adder_templates.py:326-348::test_arithmetic_cin_1; error============================== test session starts ============================== platform linux -- Python 3.11.14, pytest-8.4.2, pluggy-1.6.0 rootdir: <REDACTED_MOUNT>/9E4E6CA64E6C7943/d; FAILED
- Adder stage 22 `comprehensive_verification_and_bug_analysis` check_result sig=dc42f0231a3afd54; cases=; error============================== test session starts ============================== platform linux -- Python 3.11.14, pytest-8.4.2, pluggy-1.6.0 rootdir: <REDACTED_MOUNT>/9E4E6CA64E6C7943/d; FAILED
- Adder stage 22 `comprehensive_verification_and_bug_analysis` check_result sig=dc42f0231a3afd54; cases=; error============================== test session starts ============================== platform linux -- Python 3.11.14, pytest-8.4.2, pluggy-1.6.0 rootdir: <REDACTED_MOUNT>/9E4E6CA64E6C7943/d; FAILED
- uart_tx stage 19 `basic_api_functional_test` check_result sig=4c2235e70d7c9f06; cases=; error=Bug analysis documentation parsing failed for file 'unity_test/uart_tx_bug_analysis.md': At line (7): Found CK tag '<CK-' but its parent FC tag '<FC-' was not found in previous lin; You must use the format <FG-GROUP>, <FC-FUNCTION>, <CK-CHECK>, <BG-NAME-XX>, <TC-FAILEDTESTCASE>.

### missing_assert
- Adder stage 19 `basic_api_functional_test` check_result sig=5a89e68d9ed0c753; cases=unity_test/tests/test_Adder_api_basic.py:114-136::test_api_Adder_add_large_numbers, unity_test/tests/test_Adder_api_coverage.py:94-120::test_api_Adder_step_falling, unity_test/tests/test_Adder_api_coverage.py:123-148::test_api_Adder_carry_in_1; error=The following 1 test cases do not contain assert statements: unity_test/tests/test_Adder_api_coverage.py:391-411::test_api_Adder_special_all_ones. Note: A test case MUST contain at; FAILED
- ALU754 stage 16 `evaluate_env_fixture` check_result sig=ba05b90f8c2432da; cases=; error=The following 1 test cases do not contain assert statements: unity_test/tests/test_ALU754_env_fixture.py:115-128::test_api_ALU754_env_basic_operation. Note: A test case MUST contai; Please fix the issues reported in 'last_check_result.check_info.last_msg.error' according to the suggestions, and then use the `Complete` tool again to complete this stage.
- FSM stage 24 `generate_random_test_cases` check_result sig=5ab83078b48ac656; cases=; error=The following 5 test cases do not contain assert statements: unity_test/tests/test_FSM_random_state_management.py:152-230::test_random_pulse_timing, unity_test/tests/test_FSM_rando; Please fix the issues reported in 'check_info.last_msg.error' according to the suggestions, and then use the `Check` tool again to re-validate your work.
- FSM stage 24 `generate_random_test_cases` check_result sig=54b3577174011569; cases=; error=The following 1 test cases do not contain assert statements: unity_test/tests/test_FSM_random_edge_cases.py:106-182::test_random_key_release_stuck. Note: A test case MUST contain a; Please fix the issues reported in 'check_info.last_msg.error' according to the suggestions, and then use the `Check` tool again to re-validate your work.

### test_template_not_implemented
- Adder stage 21 `test_case_implementation_in_batch` test_run sig=26a60e2eebdcfbde; cases=unity_test/tests/test_Adder_templates.py:7-23::test_api_initialize_basic, unity_test/tests/test_Adder_templates.py:26-43::test_api_port_write_input, unity_test/tests/test_Adder_templates.py:46-65::test_api_port_read_output; error=FAILED
- Adder stage 21 `test_case_implementation_in_batch` test_run sig=1b6b892a679618c0; cases=unity_test/tests/test_Adder_templates.py:7-23::test_api_initialize_basic; error=FAILED; =================================== FAILURES =================================== __________________________ test_api_initialize_basic ___________________________ env = <Adder_api.A
- Adder stage 21 `test_case_implementation_in_batch` test_run sig=0ccb5fba739c6479; cases=unity_test/tests/test_Adder_templates.py:29-52::test_api_port_write_input, unity_test/tests/test_Adder_templates.py:55-86::test_api_port_read_output, unity_test/tests/test_Adder_templates.py:89-112::test_api_step_basic; error=FAILED
- Adder stage 21 `test_case_implementation_in_batch` test_run sig=0c6bf9c32760e9d8; cases=unity_test/tests/test_Adder_templates.py:60-97::test_api_port_read_output, unity_test/tests/test_Adder_templates.py:129-158::test_api_step_rising, unity_test/tests/test_Adder_templates.py:161-190::test_api_step_falling; error=FAILED
- Adder stage 21 `test_case_implementation_in_batch` test_run sig=8d93e980d1d98d47; cases=unity_test/tests/test_Adder_templates.py:60-97::test_api_port_read_output; error=FAILED; =================================== FAILURES =================================== __________________________ test_api_port_read_output ___________________________ env = <Adder_api.A

### label_or_report_mismatch
- Adder stage 21 `test_case_implementation_in_batch` check_result sig=3a4096c0d3187c55; cases=unity_test/tests/test_Adder_templates.py:161-190::test_api_step_falling, unity_test/tests/test_Adder_templates.py:281-300::test_arithmetic_add_large_numbers, unity_test/tests/test_Adder_templates.py:326-348::test_arithmetic_cin_1; error============================== test session starts ============================== platform linux -- Python 3.11.14, pytest-8.4.2, pluggy-1.6.0 rootdir: <REDACTED_MOUNT>/9E4E6CA64E6C7943/d; FAILED
- ALU754 stage 22 `comprehensive_verification_and_bug_analysis` check_result sig=b0e55fa6a93a03ad; cases=; error============================== test session starts ============================== platform linux -- Python 3.11.14, pytest-8.4.2, pluggy-1.6.0 rootdir: <REDACTED_MOUNT>/9E4E6CA64E6C7943/d; FAILED
- ALU754 stage 22 `comprehensive_verification_and_bug_analysis` check_result sig=5f72080b9f0f78cd; cases=; error============================== test session starts ============================== platform linux -- Python 3.11.14, pytest-8.4.2, pluggy-1.6.0 rootdir: <REDACTED_MOUNT>/9E4E6CA64E6C7943/d; FAILED
- FSM stage 21 `test_case_implementation_in_batch` check_result sig=9d1a1ab2e597586d; cases=; error=Bug analysis documentation 'unity_test/FSM_bug_analysis.md' contains 1 check points (FG-FUNCTYPE-A/FC-A1/CK-NAME1) which are not found in the test report. Ensure the check point la; Please fix the issues reported in 'check_info.last_msg.error' according to the suggestions, and then use the `Check` tool again to re-validate your work.
- FSM stage 21 `test_case_implementation_in_batch` check_result sig=9d1a1ab2e597586d; cases=; error=Bug analysis documentation 'unity_test/FSM_bug_analysis.md' contains 1 check points (FG-FUNCTYPE-A/FC-A1/CK-NAME1) which are not found in the test report. Ensure the check point la; Please fix the issues reported in 'check_info.last_msg.error' according to the suggestions, and then use the `Check` tool again to re-validate your work.

### test_logic_or_case_implementation
- Adder stage 21 `test_case_implementation_in_batch` check_result sig=c1982634711400c4; cases=; error=Please fix the issues reported in 'check_info.last_msg.error' according to the suggestions, and then use the `Check` tool again to re-validate your work.
- Adder stage 24 `generate_random_test_cases` test_run sig=6ff4882db136a2d0; cases=unity_test/tests/test_Adder_random_extended.py:39-68::test_random_Adder_arithmetic_large_numbers; error=FAILED; =================================== FAILURES =================================== __________________ test_random_Adder_arithmetic_large_numbers __________________ env = <Adder_api.A
- Adder stage 24 `generate_random_test_cases` check_result sig=013e4d4f1c2dedd4; cases=; error=You need use tool `ReadTextFile` to read and understand the reference files; Please fix the issues reported in 'check_info.last_msg.error' according to the suggestions, and then use the `Check` tool again to re-validate your work.
- uart_tx stage 21 `test_case_implementation_in_batch` test_run sig=88f64ac6dec2f85a; cases=unity_test/tests/test_uart_tx_api_operation.py:84-112::test_api_step_multi, unity_test/tests/test_uart_tx_api_operation.py:115-154::test_api_step_during_send, unity_test/tests/test_uart_tx_config.py:14-48::test_config_data_width_5; error=FAILED; [/tmp/build-via-sdist-uyo5jzlp/xspcomm-0.0.1/src/xsignal_cfg.cpp:95] [error] signal name: tx_state not found [/tmp/build-via-sdist-uyo5jzlp/xspcomm-0.0.1/src/xsignal_cfg.cpp:95] [e
- uart_tx stage 21 `test_case_implementation_in_batch` test_run sig=500cbe49616de7ad; cases=unity_test/tests/test_uart_tx_api_operation.py:115-154::test_api_step_during_send, unity_test/tests/test_uart_tx_config.py:14-48::test_config_data_width_5, unity_test/tests/test_uart_tx_config.py:272-322::test_config_dynamic_change; error=FAILED; =================================== FAILURES =================================== __________________________ test_api_step_during_send ___________________________ env = <uart_tx_api

### test_import_error
- uart_tx stage 6 `dut_creation_implementation` check_result sig=06e6e5b7758cd3dd; cases=; error=Error occurred during check: cannot import name 'DUTUartTx' from 'uart_tx' (<REPO_ROOT>/output/uart_tx/__init__.py)  Traceback (most recent ; Please fix the issues reported in 'check_info.last_msg.error' according to the suggestions, and then use the `Check` tool again to re-validate your work.

### duplicate_label_definition
- uart_tx stage 19 `basic_api_functional_test` check_result sig=7f8af6c664b088ca; cases=; error=Bug analysis documentation parsing failed for file 'unity_test/uart_tx_bug_analysis.md': At line (34): 'FG-API' is defined multiple times.. Common issues:; You must use the format <FG-GROUP>, <FC-FUNCTION>, <CK-CHECK>, <BG-NAME-XX>, <TC-FAILEDTESTCASE>.
- FSM stage 2 `dut_function_grouping` check_result sig=6c50b84281dbb679; cases=; error=Documentation parsing failed for file 'unity_test/FSM_functions_and_checks.md': At line (299): 'FG-API' is defined multiple times..; Please fix the issues reported in 'check_info.last_msg.error' according to the suggestions, and then use the `Check` tool again to re-validate your work.

### bug_doc_incomplete_tc_label
- uart_tx stage 19 `basic_api_functional_test` check_result sig=bb88ac0ba8c8ab8a; cases=; error=Bug analysis documentation parsing failed for file 'unity_test/uart_tx_bug_analysis.md': Incomplete label '<TC-*>' detected: `FG-API/FC-INIT/CK-INIT-CLOCK at line 3 need sub node '; You must use the format <FG-GROUP>, <FC-FUNCTION>, <CK-CHECK>, <BG-NAME-XX>, <TC-FAILEDTESTCASE>.
- uart_tx stage 19 `basic_api_functional_test` check_result sig=924e50d0800ec07a; cases=; error=Bug analysis documentation parsing failed for file 'unity_test/uart_tx_bug_analysis.md': Incomplete label '<TC-*>' detected: `FG-PROTOCOL/FC-DATA-BITS/CK-DATA-6BIT/BG-TODO-SAMPLE-0; You must use the format <FG-GROUP>, <FC-FUNCTION>, <CK-CHECK>, <BG-NAME-XX>, <TC-FAILEDTESTCASE>.
- FSM stage 21 `test_case_implementation_in_batch` check_result sig=711171751be4bb50; cases=; error=Bug analysis documentation parsing failed for file 'unity_test/FSM_bug_analysis.md': Incomplete label '<TC-*>' detected: `FG-STATE-MGMT/FC-PRESSED-MANAGEMENT/CK-PRESSED-KEEP/BG-FSM; You must use the format <FG-GROUP>, <FC-FUNCTION>, <CK-CHECK>, <BG-NAME-XX>, <TC-FAILEDTESTCASE>.
- FSM stage 21 `test_case_implementation_in_batch` check_result sig=7d0c99f191886b2c; cases=; error=Bug analysis documentation parsing failed for file 'unity_test/FSM_bug_analysis.md': Incomplete label '<TC-*>' detected: `FG-STATE-MGMT/FC-PRESSED-MANAGEMENT/CK-PRESSED-KEEP/BG-FSM; You must use the format <FG-GROUP>, <FC-FUNCTION>, <CK-CHECK>, <BG-NAME-XX>, <TC-FAILEDTESTCASE>.

### generic_checker_failure
- ALU754 stage 2 `dut_function_grouping` check_result sig=c42e9ce20f72d2aa; cases=; error=Documentation parsing failed for file 'unity_test/ALU754_functions_and_checks.md': Invalid characters FG-* found in keys: FG-*.; Please fix the issues reported in 'check_info.last_msg.error' according to the suggestions, and then use the `Check` tool again to re-validate your work.

### bug_doc_passed_tc_marked_as_bug
- FSM stage 21 `test_case_implementation_in_batch` check_result sig=7087a743cb130ee7; cases=; error=Bug analysis documentation 'unity_test/FSM_bug_analysis.md' contains 2 test cases (<TC-test_FSM_reset_templates.py::test_FSM_reset_while_pressed>(location: unity_test/tests/templat; 2. Ensure the test cases is working correctly and 'FAILED' as expected.
