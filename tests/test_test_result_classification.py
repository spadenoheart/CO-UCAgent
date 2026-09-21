from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ucagent.stage.vmanager import StageManager
from ucagent.verify_agent import VerifyAgent
from ucagent.memory.context_reuse import (
    ContextReuseStore,
    classify_failure_pattern,
    stage_roles_compatible,
)
from ucagent.tools.testops import RunPyTest
from ucagent.util.test_result import (
    TEST_OUTCOME_CRASH,
    TEST_OUTCOME_FAILURE,
    TEST_OUTCOME_INFRASTRUCTURE_ERROR,
    TEST_OUTCOME_NO_TESTS,
    TEST_OUTCOME_PASS,
    TEST_OUTCOME_TIMEOUT,
    classify_test_outcome,
)


def summarize(payload):
    manager = StageManager.__new__(StageManager)
    return manager._summarize_test_result(payload)


def summarize_for_stage(payload, stage_name: str):
    manager = StageManager.__new__(StageManager)
    manager.stages = [SimpleNamespace(name=stage_name)]
    manager.stage_index = 0
    return manager._summarize_test_result(payload)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {
                "REPORT": {
                    "run_test_success": True,
                    "tests": {
                        "total": 2,
                        "fails": 0,
                        "test_cases": {"test_a": "PASSED", "test_b": "PASS"},
                    },
                }
            },
            TEST_OUTCOME_PASS,
        ),
        (
            {
                "REPORT": {
                    "run_test_success": False,
                    "tests": {
                        "total": 2,
                        "fails": 1,
                        "test_cases": {"test_a": "FAILED", "test_b": "PASSED"},
                    },
                }
            },
            TEST_OUTCOME_FAILURE,
        ),
        (
            {
                "error": "Run test cases/generate report fail!",
                "REPORT": {"run_test_success": False},
                "STDERR": "ERROR collecting tests/test_api.py\nImportError: cannot import name DUT",
            },
            TEST_OUTCOME_INFRASTRUCTURE_ERROR,
        ),
        (
            {
                "error": "Run test cases/generate report fail!",
                "REPORT": {"run_test_success": False},
                "STDERR": "Test run timed out after 30 seconds.",
            },
            TEST_OUTCOME_TIMEOUT,
        ),
        (
            {
                "error": "Run test cases/generate report fail!",
                "REPORT": {"run_test_success": False},
                "STDERR": "Fatal Python error: Segmentation fault\nPytest process exited with code -11.",
            },
            TEST_OUTCOME_CRASH,
        ),
        (
            {
                "REPORT": {
                    "run_test_success": False,
                    "tests": {"total": 0, "fails": 0, "test_cases": {}},
                },
                "STDOUT": "collected 0 items\nno tests ran",
            },
            TEST_OUTCOME_NO_TESTS,
        ),
    ],
)
def test_summarize_test_result_has_explicit_outcome(payload, expected):
    summary = summarize(payload)

    assert summary["test_outcome"] == expected
    assert summary["tests_passed_all"] is (expected == TEST_OUTCOME_PASS)


def test_zero_tests_can_never_be_a_pass():
    summary = summarize({
        "REPORT": {
            "run_test_success": True,
            "tests": {"total": 0, "fails": 0, "test_cases": {}},
        }
    })

    assert summary["test_outcome"] == TEST_OUTCOME_NO_TESTS
    assert summary["tests_passed_all"] is False


def test_no_report_no_tests_output_is_still_classified_as_no_tests():
    summary = summarize({
        "error": "Run test cases/generate report fail!",
        "REPORT": {"run_test_success": False},
        "STDOUT": "collected 0 items\nno tests ran",
        "STDERR": "Pytest process exited with code 5.",
    })

    assert summary["test_outcome"] == TEST_OUTCOME_NO_TESTS
    assert summary["tests_passed_all"] is False


def test_pytest_timeout_plugin_name_does_not_turn_pass_into_timeout():
    summary = summarize({
        "REPORT": {
            "run_test_success": True,
            "tests": {
                "total": 2,
                "fails": 0,
                "test_cases": {"test_a": "PASSED", "test_b": "PASSED"},
            },
        },
        "STDOUT": (
            "plugins: asyncio-1.2.0, timeout-2.4.0, xdist-3.8.0\n"
            "============================== 2 passed in 0.31s ==============================\n"
        ),
    })

    assert summary["test_outcome"] == TEST_OUTCOME_PASS
    assert summary["test_outcome_reason"] == "verified_nonempty_pass"


def test_pytest_timeout_plugin_name_does_not_hide_test_failure():
    summary = summarize({
        "REPORT": {
            "run_test_success": False,
            "tests": {
                "total": 2,
                "fails": 1,
                "test_cases": {"test_a": "PASSED", "test_b": "FAILED"},
            },
        },
        "STDOUT": "plugins: asyncio-1.2.0, timeout-2.4.0, xdist-3.8.0\n",
    })

    assert summary["test_outcome"] == TEST_OUTCOME_FAILURE
    assert summary["test_outcome_reason"] == "reported_non_pass_status"


def test_structured_error_status_is_an_infrastructure_error():
    summary = summarize({
        "REPORT": {
            "run_test_success": True,
            "tests": {
                "total": 1,
                "fails": 0,
                "test_cases": {"test_setup": "ERROR"},
            },
        }
    })

    assert summary["test_outcome"] == TEST_OUTCOME_INFRASTRUCTURE_ERROR
    assert summary["test_outcome_reason"] == "reported_infrastructure_status"
    assert summary["failed_cases_top"] == ["test_setup"]


def test_ambiguous_result_is_conservatively_infrastructure_error():
    outcome, reason = classify_test_outcome({"message": "unexpected runner response"})

    assert outcome == TEST_OUTCOME_INFRASTRUCTURE_ERROR
    assert reason == "insufficient_success_evidence"


def test_diagnostic_crash_marker_overrides_stale_explicit_pass():
    outcome, reason = classify_test_outcome({
        "test_outcome": TEST_OUTCOME_PASS,
        "tests_total": 1,
        "tests_failed": 0,
        "failed_cases_top": [],
        "error_top": ["Fatal Python error: Segmentation fault"],
    })

    assert outcome == TEST_OUTCOME_CRASH
    assert reason == "crash_marker"


def test_template_stage_accepts_expected_xfail_without_recording_failed_case():
    payload = {
        "REPORT": {
            "run_test_success": True,
            "tests": {
                "total": 2,
                "fails": 0,
                "test_cases": {"test_a": "XFAIL", "test_b": "PASSED"},
            },
        },
        "STDOUT": "1 passed, 1 xfailed",
    }

    summary = summarize_for_stage(payload, "create_test_case_templates")

    assert summary["test_outcome"] == TEST_OUTCOME_PASS
    assert summary["test_outcome_reason"] == "expected_template_xfail"
    assert summary["failed_cases_top"] == []


def test_expected_xfail_is_not_a_pass_outside_template_stage():
    payload = {
        "REPORT": {
            "run_test_success": True,
            "tests": {
                "total": 1,
                "fails": 0,
                "test_cases": {"test_a": "XFAIL"},
            },
        },
        "STDOUT": "1 xfailed",
    }

    summary = summarize_for_stage(payload, "test_case_implementation_in_batch")

    assert summary["test_outcome"] == TEST_OUTCOME_FAILURE
    assert summary["failed_cases_top"] == ["test_a"]


def test_template_stage_rejects_unexpected_xpass():
    payload = {
        "REPORT": {
            "run_test_success": True,
            "tests": {
                "total": 1,
                "fails": 0,
                "test_cases": {"test_a": "XPASS"},
            },
        }
    }

    summary = summarize_for_stage(payload, "create_test_case_templates")

    assert summary["test_outcome"] == TEST_OUTCOME_FAILURE


def test_interface_assertion_is_not_misclassified_as_template_placeholder():
    pattern = classify_failure_pattern({
        "failed_cases_top": ["test_api_Adder_env_clear_cbs"],
        "error_top": ["assert hasattr(env, 'clear_cbs')"],
    }, stage_name="evaluate_env_fixture", event_type="test_run")

    assert pattern == "api_or_env_interface_mismatch"


def test_prompt_v1_classifies_coverage_callback_attribute_error_before_env_api():
    manager = StageManager.__new__(StageManager)
    manager._is_prompt_v1_enabled = lambda: True
    manager.stages = [SimpleNamespace(name="evaluate_env_fixture")]
    manager.stage_index = 0

    summary = manager._checker_error_summary({
        "STDERR": (
            "unity_test/tests/DualPort_function_coverage_def.py:143 in sample\n"
            "AttributeError: 'DUTDualPort' object has no attribute 'busy'"
        )
    })

    assert summary["categories"][0] == "coverage_runtime_contract"
    assert "api_or_env_interface_mismatch" not in summary["categories"]
    assert "coverage" in summary["primary_actions"][0]


def test_env_checker_discovery_error_has_dedicated_pattern():
    pattern = classify_failure_pattern({
        "error_top": ["Insufficient env fixture test coverage: 0 env test functions found, minimum required 4"],
    }, stage_name="evaluate_env_fixture", event_type="check_result")

    assert pattern == "test_structure_or_discovery"


def test_template_stage_uses_contract_pattern_not_implementation_pattern():
    pattern = classify_failure_pattern({
        "error_top": ["AssertionError: Not implemented"],
    }, stage_name="create_test_case_templates", event_type="check_result")

    assert pattern == "template_validation_or_runner_contract"


def test_undocumented_failed_cases_use_classification_pattern():
    error = (
        "[Undocumented Failed Cases] Found 2 failed test case(s) not documented "
        "in the bug analysis file: test_a, test_b. All failed test cases must be documented."
    )

    pattern = classify_failure_pattern(
        {"error_top": [error]},
        stage_name="test_case_implementation_in_batch",
        event_type="check_result",
    )

    assert pattern == "failed_cases_need_classification"


def test_prompt_v1_undocumented_failed_cases_are_not_generic():
    manager = StageManager.__new__(StageManager)
    manager._is_prompt_v1_enabled = lambda: True
    manager.stages = [SimpleNamespace(name="test_case_implementation_in_batch")]
    manager.stage_index = 0

    summary = manager._checker_error_summary({
        "check_info": {
            "error": "[Undocumented Failed Cases] Found failed test cases not documented in the bug analysis file."
        },
    })

    assert summary["categories"] == ["failed_cases_need_classification"]


def test_prompt_v1_template_summary_preserves_placeholders():
    manager = StageManager.__new__(StageManager)
    manager._is_prompt_v1_enabled = lambda: True
    manager.stages = [SimpleNamespace(name="create_test_case_templates")]
    manager.stage_index = 0

    summary = manager._checker_error_summary({
        "check_info": {"error": "AssertionError: Not implemented"},
    })

    assert summary["categories"] == ["template_validation_or_runner_contract"]
    assert "禁止实现真实测试逻辑" in summary["primary_actions"][0]


def test_prompt_v1_implementation_summary_still_removes_placeholders():
    manager = StageManager.__new__(StageManager)
    manager._is_prompt_v1_enabled = lambda: True
    manager.stages = [SimpleNamespace(name="test_case_implementation_in_batch")]
    manager.stage_index = 0

    summary = manager._checker_error_summary({
        "check_info": {"error": "AssertionError: Not implemented"},
    })

    assert summary["categories"] == ["test_template_not_implemented"]
    assert "实现当前batch要求的测试逻辑" in summary["primary_actions"][0]


def test_prompt_v1_skill_usage_contract_is_not_generic():
    manager = StageManager.__new__(StageManager)
    manager._is_prompt_v1_enabled = lambda: True
    manager.stages = [SimpleNamespace(name="test_case_implementation_in_batch")]
    manager.stage_index = 0

    checker_message = (
        "Please use tool 'SetSkillUsage' to check and set the skill usage "
        "of this stage before completing it."
    )
    summary = manager._checker_error_summary({
        "check_info": checker_message,
        "check_pass": False,
        "action": "Please fix the issues reported in check_info.",
    })

    assert summary["categories"] == ["stage_skill_usage_required"]
    assert summary["raw_error_top"][0] == checker_message
    assert "不要修改测试或文档" in summary["primary_actions"][0]


def test_prompt_v1_bug_doc_parent_error_has_precise_repair_contract():
    manager = StageManager.__new__(StageManager)
    manager._is_prompt_v1_enabled = lambda: True
    manager.stages = [SimpleNamespace(name="comprehensive_verification_and_bug_analysis")]
    manager.stage_index = 0

    summary = manager._checker_error_summary({
        "check_info": {
            "error": (
                "Bug analysis documentation parsing failed: Found TC tag "
                "'<TC-test_Adder.py::test_overflow>' but its parent BG tag "
                "'<BG-' was not found. Please ensure proper nesting."
            )
        },
    })

    assert "bug_doc_hierarchy_violation" in summary["categories"]
    action_index = summary["categories"].index("bug_doc_hierarchy_violation")
    assert "只保留一个BG" in summary["primary_actions"][action_index]
    assert "不要追加第二份同名祖先" in summary["primary_actions"][action_index]


def test_prompt_v1_bug_doc_tc_parse_error_requires_pytest_node_id():
    manager = StageManager.__new__(StageManager)
    manager._is_prompt_v1_enabled = lambda: True
    manager.stages = [SimpleNamespace(name="test_case_implementation_in_batch")]
    manager.stage_index = 0

    summary = manager._checker_error_summary({
        "check_info": {
            "error": (
                "Test case (<TC-test_boundary_max_input>) parse fail, its format "
                "should be: <TC-test_file.py::[ClassName::]test_case_name>."
            )
        },
    })

    assert summary["categories"] == ["bug_doc_tc_identifier_format"]
    assert "真实FAILED node id" in summary["primary_actions"][0]


def test_prompt_v1_bug_doc_bare_tc_placeholder_is_precisely_classified():
    manager = StageManager.__new__(StageManager)
    manager._is_prompt_v1_enabled = lambda: True
    manager.stages = [SimpleNamespace(name="test_case_implementation_in_batch")]
    manager.stage_index = 0

    summary = manager._checker_error_summary({
        "check_info": {
            "error": (
                "[Test Case Format Error] '<TC->' has incorrect format. "
                "[Correct Format] <TC-test_file.py::test_case_name>."
            )
        },
    })

    assert summary["categories"] == ["bug_doc_tc_identifier_format"]
    assert "标题、说明文字或占位内容" in summary["primary_actions"][0]
    assert "<TC->" in summary["primary_actions"][0]


def test_prompt_v1_checkpoint_not_found_uses_exact_bug_doc_contract():
    manager = StageManager.__new__(StageManager)
    manager._is_prompt_v1_enabled = lambda: True
    manager.stages = [SimpleNamespace(name="test_case_implementation_in_batch")]
    manager.stage_index = 0

    exact_error = (
        "[Checkpoint Not Found] Bug analysis document references 1 checkpoint(s) "
        "(FG-BOUNDARY/FC-SPECIAL-VALUES/CK-SINGLE-BIT-MSB) that do not exist "
        "in the test report."
    )
    summary = manager._checker_error_summary({
        "check_info": {
            "last_msg": {
                "STDOUT": "pytest output that must not hide the contract error",
                "error": exact_error,
            }
        }
    })

    assert summary["categories"] == ["bug_doc_checkpoint_not_found"]
    assert "TEST_REPORT.failed_test_case_with_check_point_list" in summary["primary_actions"][0]
    assert summary["raw_error_top"][0] == exact_error


def test_prompt_v1_random_ck_records_require_explicit_generated_tool_argument():
    manager = StageManager.__new__(StageManager)
    manager._is_prompt_v1_enabled = lambda: True
    manager.stages = [SimpleNamespace(name="generate_random_test_cases")]
    manager.stage_index = 0

    summary = manager._checker_error_summary({
        "check_info": {
            "error": (
                "No valid CK labels were recorded in the current random-test batch "
                "(need use args `generated: dict` to pass CK processing records). "
                "Please analyze FG-A/FC-A/CK-A."
            )
        }
    })

    assert summary["categories"] == ["random_test_generated_records_missing"]
    assert "顶层generated对象" in summary["primary_actions"][0]
    assert "SetCurrentStageJournal" in summary["primary_actions"][0]
    assert "不要为此修改已通过的测试" in summary["primary_actions"][0]


def test_prompt_v1_classifies_api_collection_regression_exactly():
    manager = StageManager.__new__(StageManager)
    manager._is_prompt_v1_enabled = lambda: True
    manager.stages = [SimpleNamespace(name="basic_api_functional_test")]
    manager.stage_index = 0

    summary = manager._checker_error_summary({
        "check_info": {
            "error": "[API Test Collection Regression] Missing or renamed previously passing tests: tests/test_api.py::test_reset"
        }
    })

    assert "api_test_collection_regression" in summary["categories"]
    assert "整文件重写" in summary["primary_actions"][0]


def test_prompt_v1_aggregates_random_source_contract_category():
    manager = StageManager.__new__(StageManager)
    manager._is_prompt_v1_enabled = lambda: True
    manager.stages = [SimpleNamespace(name="generate_random_test_cases")]
    manager.stage_index = 0

    summary = manager._checker_error_summary({
        "check_info": {
            "error": [
                "Random-test source contract violations were found.",
                "test_a.py:test_a: missing 'ucagent.repeat_count'",
            ]
        }
    })

    assert "random_test_source_contract_violation" in summary["categories"]
    assert "一次性修复" in summary["primary_actions"][0]


def test_prompt_v1_classifies_incomplete_test_marking_contract():
    manager = StageManager.__new__(StageManager)
    manager._is_prompt_v1_enabled = lambda: True
    manager.stages = [SimpleNamespace(name="basic_api_functional_test")]
    manager.stage_index = 0

    summary = manager._checker_error_summary({
        "check_info": {
            "error": (
                "Find 24 functions do not have correct check point marks. "
                "Test cases not marked with 'mark_function': test_api.py::test_reset"
            )
        }
    })

    assert "test_marking_contract_incomplete" in summary["categories"]
    assert "一次性" in summary["primary_actions"][0]
    assert "_fixed" in summary["primary_actions"][0]
    assert classify_failure_pattern(
        {"checker_categories": summary["categories"]},
        stage_name="basic_api_functional_test",
    ) == "test_marking_contract_incomplete"


def test_prompt_v1_classifies_template_source_contract_before_runner_error():
    manager = StageManager.__new__(StageManager)
    manager._is_prompt_v1_enabled = lambda: True
    manager.stages = [SimpleNamespace(name="create_test_case_templates")]
    manager.stage_index = 0

    summary = manager._checker_error_summary({
        "check_info": {
            "error": [
                "[Template Source Contract] Placeholder tests require active mark_function calls.",
                "test_pipe.py:4::test_stage1 (missing_active_mark_function_for_current_test)",
            ],
            "STDERR": "Invalid data file: tests/data/toffee_tmp/master/test_stage1.dat",
        }
    })

    assert summary["categories"][0] == "template_source_contract_incomplete"
    assert summary["raw_error_top"][0].startswith("[Template Source Contract]")
    assert classify_failure_pattern(
        {"checker_categories": summary["categories"]},
        stage_name="create_test_case_templates",
    ) == "template_source_contract_incomplete"


def test_prompt_v1_classifies_bug_doc_checkpoint_not_marked_exactly():
    manager = StageManager.__new__(StageManager)
    manager._is_prompt_v1_enabled = lambda: True
    manager.stages = [SimpleNamespace(name="static_bug_validation")]
    manager.stage_index = 0

    exact_error = (
        "[Checkpoint Not Marked] Bug analysis document contains 8 test case(s) "
        "that have not called mark_function for their associated checkpoints."
    )
    summary = manager._checker_error_summary({
        "check_info": {
            "last_msg": {
                "STDOUT": "UnityChipCheckerStaticBugValidation passed",
                "error": [exact_error, "[Cause] incorrect TC to CK association"],
            }
        }
    })

    assert summary["categories"] == ["bug_doc_checkpoint_not_marked"]
    assert summary["raw_error_top"][0] == exact_error
    assert classify_failure_pattern(
        {"checker_categories": summary["categories"]},
        stage_name="static_bug_validation",
    ) == "bug_doc_checkpoint_not_marked"


def test_prompt_v1_classifies_passed_tc_status_mismatch_exactly():
    manager = StageManager.__new__(StageManager)
    manager._is_prompt_v1_enabled = lambda: True
    manager.stages = [SimpleNamespace(name="test_case_implementation_in_batch")]
    manager.stage_index = 0

    summary = manager._checker_error_summary({
        "check_info": {
            "error": [
                "[Test Case Status Mismatch] Bug analysis document contains 3 test case(s) whose actual status is PASSED.",
                "[Cause] Test cases marked in bug analysis must be FAILED (failure proves the bug exists).",
            ]
        }
    })

    assert summary["categories"] == ["bug_doc_passed_tc_marked_as_bug"]
    assert "prunebug.py" in summary["primary_actions"][0]
    assert classify_failure_pattern(
        {"checker_categories": summary["categories"]},
        stage_name="test_case_implementation_in_batch",
    ) == "bug_doc_passed_tc_marked_as_bug"


def test_prompt_v1_static_bug_hierarchy_has_checker_only_incremental_action():
    manager = StageManager.__new__(StageManager)
    manager._is_prompt_v1_enabled = lambda: True
    manager.stages = [SimpleNamespace(name="static_bug_analysis")]
    manager.stage_index = 0

    summary = manager._checker_error_summary({
        "check_info": {
            "error": (
                "Tag hierarchy parse error: Found BG-STATIC tag but its parent CK tag "
                "was not found in previous lines."
            )
        },
    })

    assert summary["categories"] == ["static_bug_tag_hierarchy"]
    assert "增量修正" in summary["primary_actions"][0]
    assert "不要删除" in summary["primary_actions"][0]


def test_prompt_v1_static_bug_batch_progress_forbids_pytest():
    manager = StageManager.__new__(StageManager)
    manager._is_prompt_v1_enabled = lambda: True
    manager.stages = [SimpleNamespace(name="static_bug_analysis")]
    manager.stage_index = 0

    summary = manager._checker_error_summary({
        "check_info": {
            "error": (
                "Not all 'RTL_file_to_analyze' in this batch have been completed (0/1). "
                "Please use tool CurrentFileTips."
            )
        },
    })

    assert summary["categories"] == ["static_bug_batch_progress"]
    assert "不要运行pytest" in summary["primary_actions"][0]


def test_context_reuse_hard_gates_template_and_implementation_roles(tmp_path: Path):
    pack = {
        "effective_items": [
            {
                "strategy_id": "template",
                "dut": "Adder",
                "stage_name": "create_test_case_templates",
                "stage_index": 20,
                "failure_pattern": "template_validation_or_runner_contract",
                "failure": {"failure_pattern": "template_validation_or_runner_contract"},
                "quality_score": 1.0,
                "actions": [{"operation": "replace_string", "path_category": "test_code"}],
            },
            {
                "strategy_id": "implementation",
                "dut": "Adder",
                "stage_name": "test_case_implementation_in_batch",
                "stage_index": 21,
                "failure_pattern": "template_validation_or_runner_contract",
                "failure": {"failure_pattern": "template_validation_or_runner_contract"},
                "quality_score": 1.0,
                "actions": [{"operation": "replace_string", "path_category": "test_code"}],
            },
        ],
        "compression_hints": [
            {
                "dut": "Adder",
                "stage_name": "create_test_case_templates",
                "stage_index": 20,
                "failure_pattern": "template_validation_or_runner_contract",
                "failure_group": "test_logic",
                "repeat_count": 2,
            },
            {
                "dut": "Adder",
                "stage_name": "test_case_implementation_in_batch",
                "stage_index": 21,
                "failure_pattern": "template_validation_or_runner_contract",
                "failure_group": "test_logic",
                "repeat_count": 3,
            },
        ],
    }
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(__import__("json").dumps(pack), encoding="utf-8")
    store = ContextReuseStore(str(tmp_path), "Adder", {
        "pack_path": str(pack_path),
        "min_score": 0.0,
        "strict_stage_role_match": True,
    })

    results = store.search(
        {"failure_pattern": "template_validation_or_runner_contract"},
        stage_index=20,
        stage_name="create_test_case_templates",
        limit=5,
    )
    hints = store.compression_hints_for(
        {"failure_pattern": "template_validation_or_runner_contract"},
        stage_index=20,
        stage_name="create_test_case_templates",
        limit=5,
    )

    assert [item["strategy_id"] for item in results] == ["template"]
    assert [item["stage_name"] for item in hints] == ["create_test_case_templates"]
    assert stage_roles_compatible("create_test_case_templates", "test_case_implementation_in_batch") is False


def test_context_reuse_verifier_utility_gate_rejects_non_positive_and_unknown(tmp_path: Path):
    def strategy(strategy_id: str, gate_score):
        item = {
            "strategy_id": strategy_id,
            "dut": "Adder",
            "stage_name": "evaluate_env_fixture",
            "stage_index": 16,
            "failure_pattern": "api_or_env_interface_mismatch",
            "failure": {"failure_pattern": "api_or_env_interface_mismatch"},
            "quality_score": 1.0,
            "actions": [{
                "operation": "replace_string",
                "path": "tests/test_Adder_env.py",
                "path_category": "test_code",
            }],
        }
        if gate_score is not None:
            item["verifier_utility"] = {
                "policy_version": "verifier_grounded_utility_v0",
                "gate_score": gate_score,
            }
        return item

    pack = {
        "effective_items": [strategy("positive", 0.4), strategy("negative", -0.1), strategy("unknown", None)],
        "compression_hints": [],
    }
    pack_path = tmp_path / "utility_pack.json"
    pack_path.write_text(__import__("json").dumps(pack), encoding="utf-8")
    store = ContextReuseStore(str(tmp_path), "Adder", {
        "pack_path": str(pack_path),
        "min_score": 0.0,
        "enable_utility_gate": "true",
        "min_utility_score": 0.0,
        "utility_unknown_policy": "reject",
        "utility_prompt_token_weight": 0.05,
    })

    results = store.search(
        {"failure_pattern": "api_or_env_interface_mismatch"},
        stage_index=16,
        stage_name="evaluate_env_fixture",
        limit=5,
    )

    assert [item["strategy_id"] for item in results] == ["positive"]
    assert results[0]["__context_reuse__"]["utility"]["passed"] is True
    assert results[0]["__context_reuse__"]["utility"]["net_score"] > 0
    assert store.get_last_search_summary() == {
        "utility_gate_enabled": True,
        "quality_candidates": 3,
        "match_candidates": 3,
        "utility_passed": 1,
        "utility_rejected": 2,
        "utility_unknown": 1,
        "hard_gate_rejected": 0,
        "hard_gate_reasons": {},
        "returned": 1,
    }
    metrics = store.get_runtime_metrics()
    assert metrics["utility_candidates"] == 3
    assert metrics["utility_passed"] == 1
    assert metrics["utility_rejected"] == 2
    assert metrics["utility_unknown"] == 1


def test_context_reuse_skips_workflow_evidence_failures(tmp_path: Path):
    pack = {
        "effective_items": [{
            "strategy_id": "edit-test",
            "dut": "uart_tx",
            "stage_name": "evaluate_env_fixture",
            "failure_pattern": "api_or_env_interface_mismatch",
            "quality_score": 1.0,
            "actions": [{"path_category": "test_code"}],
        }],
        "compression_hints": [{
            "dut": "uart_tx",
            "stage_name": "evaluate_env_fixture",
            "failure_pattern": "reference_files_unread",
            "failure_signature": "same",
        }],
    }
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(__import__("json").dumps(pack), encoding="utf-8")
    store = ContextReuseStore(str(tmp_path), "Adder", {
        "pack_path": str(pack_path),
        "min_score": 0.0,
        "skip_non_reusable_failures": True,
    })
    failure = {
        "failure_pattern": "reference_files_unread",
        "failure_signature": "same",
        "error_top": ["Required reference files have not been read"],
    }

    assert store.search(failure, 25, "verification_review_and_summary", limit=5) == []
    assert store.compression_hints_for(failure, 25, "verification_review_and_summary", limit=5) == []
    summary = store.get_last_search_summary()
    assert summary["hard_gate_reasons"] == {"non_reusable_failure": 1}


def test_context_reuse_requires_exact_bug_subtype_and_action_category(tmp_path: Path):
    pack = {
        "effective_items": [
            {
                "strategy_id": "wrong-bug-subtype",
                "dut": "Adder",
                "stage_name": "comprehensive_verification_and_bug_analysis",
                "failure_pattern": "bug_doc_incomplete_tc_label",
                "quality_score": 1.0,
                "actions": [{"path_category": "bug_document"}],
            },
            {
                "strategy_id": "wrong-document",
                "dut": "FSM",
                "stage_name": "dut_function_grouping",
                "failure_pattern": "duplicate_label_definition",
                "quality_score": 1.0,
                "actions": [{"path_category": "spec_check_document"}],
            },
            {
                "strategy_id": "matching-bug-document",
                "dut": "uart_tx",
                "stage_name": "comprehensive_verification_and_bug_analysis",
                "failure_pattern": "duplicate_label_definition",
                "quality_score": 1.0,
                "actions": [{"path_category": "bug_document"}],
            },
        ],
        "compression_hints": [],
    }
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(__import__("json").dumps(pack), encoding="utf-8")
    store = ContextReuseStore(str(tmp_path), "Adder", {
        "pack_path": str(pack_path),
        "min_score": 0.0,
        "strict_exact_pattern_groups": True,
        "strict_action_category_match": True,
    })
    failure = {
        "failure_pattern": "duplicate_label_definition",
        "error_top": [
            "Documentation parsing failed for unity_test/Adder_bug_analysis.md: "
            "<BG-BOUNDARY> is defined multiple times"
        ],
    }

    results = store.search(failure, 24, "generate_random_test_cases", limit=5)

    assert [item["strategy_id"] for item in results] == ["matching-bug-document"]
    assert store.get_last_search_summary()["hard_gate_reasons"] == {
        "exact_failure_pattern_mismatch": 1,
        "action_category_mismatch": 1,
    }


def test_context_reuse_generic_failure_requires_same_signature(tmp_path: Path):
    pack = {
        "effective_items": [
            {
                "strategy_id": "same-signature",
                "dut": "FSM",
                "stage_name": "test_case_implementation_in_batch",
                "failure_pattern": "generic_checker_failure",
                "failure": {"failure_signature": "sig-a"},
                "quality_score": 1.0,
                "actions": [{"path_category": "test_code"}],
            },
            {
                "strategy_id": "different-signature",
                "dut": "FSM",
                "stage_name": "test_case_implementation_in_batch",
                "failure_pattern": "generic_checker_failure",
                "failure": {"failure_signature": "sig-b"},
                "quality_score": 1.0,
                "actions": [{"path_category": "test_code"}],
            },
        ],
        "compression_hints": [],
    }
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(__import__("json").dumps(pack), encoding="utf-8")
    store = ContextReuseStore(str(tmp_path), "Adder", {
        "pack_path": str(pack_path),
        "min_score": 0.0,
        "generic_requires_signature": True,
    })

    results = store.search({
        "failure_pattern": "generic_checker_failure",
        "failure_signature": "sig-a",
    }, 21, "test_case_implementation_in_batch", limit=5)

    assert [item["strategy_id"] for item in results] == ["same-signature"]
    assert store.get_last_search_summary()["hard_gate_reasons"] == {
        "generic_signature_mismatch": 1,
    }


def test_context_reuse_low_support_destructive_action_requires_same_signature(tmp_path: Path):
    pack = {
        "effective_items": [{
            "strategy_id": "rewrite-document",
            "dut": "Adder",
            "stage_name": "comprehensive_verification_and_bug_analysis",
            "failure_pattern": "bug_doc_incomplete_tc_label",
            "failure": {"failure_signature": "historical"},
            "quality_score": 1.0,
            "support_count": 1,
            "actions": [
                {"operation": "delete_file", "path_category": "bug_document"},
                {"operation": "write", "path_category": "bug_document"},
            ],
        }],
        "compression_hints": [],
    }
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(__import__("json").dumps(pack), encoding="utf-8")
    store = ContextReuseStore(str(tmp_path), "Adder", {
        "pack_path": str(pack_path),
        "min_score": 0.0,
        "destructive_requires_support": True,
        "min_destructive_support": 2,
    })

    mismatch = store.search({
        "failure_pattern": "bug_doc_incomplete_tc_label",
        "failure_signature": "current",
    }, 22, "comprehensive_verification_and_bug_analysis", limit=5)
    exact = store.search({
        "failure_pattern": "bug_doc_incomplete_tc_label",
        "failure_signature": "historical",
    }, 22, "comprehensive_verification_and_bug_analysis", limit=5)

    assert mismatch == []
    assert [item["strategy_id"] for item in exact] == ["rewrite-document"]


def test_context_reuse_low_support_file_move_is_destructive(tmp_path: Path):
    pack = {
        "effective_items": [{
            "strategy_id": "move-test-file",
            "dut": "ALU754",
            "stage_name": "evaluate_env_fixture",
            "failure_pattern": "api_or_env_interface_mismatch",
            "failure": {"failure_signature": "historical"},
            "quality_score": 1.0,
            "support_count": 1,
            "actions": [{"operation": "move", "path_category": "test_code"}],
        }],
        "compression_hints": [],
    }
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(__import__("json").dumps(pack), encoding="utf-8")
    store = ContextReuseStore(str(tmp_path), "Adder", {
        "pack_path": str(pack_path),
        "min_score": 0.0,
        "destructive_requires_support": True,
        "min_destructive_support": 2,
    })

    results = store.search({
        "failure_pattern": "api_or_env_interface_mismatch",
        "failure_signature": "new-failure",
    }, 16, "evaluate_env_fixture", limit=5)

    assert results == []
    assert store.get_last_search_summary()["hard_gate_reasons"] == {
        "low_support_destructive_action": 1,
    }


def test_context_reuse_hint_requires_signature_or_exact_pattern_and_stage(tmp_path: Path):
    pack = {
        "effective_items": [],
        "compression_hints": [
            {
                "dut": "FSM",
                "stage_name": "test_case_implementation_in_batch",
                "failure_pattern": "test_logic_or_case_implementation",
                "failure_signature": "other",
                "repeat_count": 5,
            },
            {
                "dut": "uart_tx",
                "stage_name": "generate_random_test_cases",
                "failure_pattern": "test_logic_or_case_implementation",
                "failure_signature": "other-2",
                "repeat_count": 3,
            },
        ],
    }
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(__import__("json").dumps(pack), encoding="utf-8")
    store = ContextReuseStore(str(tmp_path), "Adder", {
        "pack_path": str(pack_path),
        "strict_compression_hint_match": True,
    })

    hints = store.compression_hints_for({
        "failure_pattern": "test_logic_or_case_implementation",
        "failure_signature": "current",
    }, 24, "generate_random_test_cases", limit=5)

    assert [item["stage_name"] for item in hints] == ["generate_random_test_cases"]


def test_context_reuse_decision_records_utility_gate_audit_fields():
    events = []
    manager = StageManager.__new__(StageManager)
    manager.enable_data_collection = True
    manager.agent = SimpleNamespace(
        record_structured_event=lambda event_type, payload: events.append((event_type, payload))
    )
    episode = {
        "strategy_id": "strategy-a",
        "dut": "uart_tx",
        "source_duts": ["uart_tx", "Adder"],
        "stage_name": "evaluate_env_fixture",
        "failure_pattern": "api_or_env_interface_mismatch",
        "action_categories": ["test_code"],
        "quality_score": 0.9,
        "support_count": 2,
        "__context_reuse__": {
            "score": 1.2,
            "reasons": ["same_failure_pattern"],
            "utility": {
                "enabled": True,
                "passed": True,
                "reason": "positive_net_utility",
                "historical_score": 0.2,
                "prompt_tokens_estimated": 160,
                "prompt_cost": 0.0156,
                "net_score": 0.1844,
                "threshold": 0.05,
                "policy_version": "verifier_grounded_utility_v0",
            },
        },
    }

    manager._record_context_reuse_decision(
        {"failure_signature": "sig-a", "failure_pattern": "api_or_env_interface_mismatch"},
        SimpleNamespace(name="evaluate_env_fixture"),
        decision="inject",
        episodes=[episode],
        search_summary={"match_candidates": 2, "utility_passed": 1, "utility_rejected": 1},
    )

    assert events[0][0] == "context_reuse_decision"
    payload = events[0][1]
    assert payload["search_summary"]["utility_rejected"] == 1
    assert payload["strategies"][0]["utility_gate"]["net_score"] == 0.1844
    assert payload["strategies"][0]["utility_gate"]["policy_version"] == "verifier_grounded_utility_v0"


def test_run_summary_collects_context_reuse_utility_metrics():
    agent = VerifyAgent.__new__(VerifyAgent)
    agent.stage_manager = SimpleNamespace(get_memory_metrics=lambda: {"context_reuse_queries": 2})
    agent.long_term_memory = None
    agent.context_reuse = SimpleNamespace(get_runtime_metrics=lambda: {
        "queries": 2,
        "utility_candidates": 5,
        "utility_passed": 3,
        "utility_rejected": 2,
        "utility_unknown": 0,
        "utility_prompt_tokens_estimated": 640,
        "utility_pass_rate": 0.6,
    })

    metrics = agent._collect_memory_cache_metrics({})

    assert metrics["context_reuse_utility_candidates"] == 5
    assert metrics["context_reuse_utility_rejected"] == 2
    assert metrics["context_reuse_utility_prompt_tokens_estimated"] == 640
    assert metrics["context_reuse_utility_pass_rate"] == 0.6


@pytest.mark.parametrize(
    ("source", "expected_success"),
    [
        ("def test_ok():\n    assert 1 + 1 == 2\n", True),
        ("def test_bad():\n    assert 1 + 1 == 3\n", False),
    ],
)
def test_run_pytest_uses_process_exit_code(tmp_path: Path, source: str, expected_success: bool):
    (tmp_path / "test_sample.py").write_text(source, encoding="utf-8")
    runner = RunPyTest()

    success, stdout, stderr = runner.do(
        str(tmp_path),
        return_stdout=True,
        return_stderr=True,
        timeout=30,
    )

    assert success is expected_success
    if not expected_success:
        assert "exited with code 1" in stderr
        assert "1 failed" in stdout


def test_run_pytest_reports_no_tests_as_unsuccessful(tmp_path: Path):
    runner = RunPyTest()

    success, stdout, stderr = runner.do(
        str(tmp_path),
        return_stdout=True,
        return_stderr=True,
        timeout=30,
    )

    assert success is False
    assert "no tests ran" in stdout
    assert "exited with code 5" in stderr
