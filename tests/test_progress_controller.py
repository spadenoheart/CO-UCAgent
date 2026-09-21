from types import SimpleNamespace

from ucagent.control import ProgressAwareLoopController
from ucagent.verify_agent import VerifyAgent


def _mutation(path="unity_test/tests/test_adder.py"):
    return {
        "event_type": "file_mutation",
        "stage_index": 21,
        "tool": "WriteTextFile",
        "operation": "write",
        "path": path,
        "path_category": "test",
        "success": True,
    }


def _test_result(
    outcome,
    failed_cases=None,
    target="",
    *,
    stage_name="evaluate_env_fixture",
    outcome_reason="",
    status_counts=None,
):
    return {
        "event_type": "test_run",
        "stage_index": 21,
        "stage_name": stage_name,
        "tool": "RunTestCases",
        "target": target,
        "result": {
            "test_outcome": outcome,
            "test_outcome_reason": outcome_reason,
            "test_status_counts": dict(status_counts or {}),
            "failed_cases_top": list(failed_cases or []),
            "failed_checkpoints_top": [],
            "checker_categories": [],
        },
    }


def _checker_result(
    passed,
    categories=None,
    tool="Check",
    *,
    failed_cases=None,
    stage_name="basic_api_functional_test",
    error_top=None,
):
    return {
        "event_type": "check_result",
        "stage_index": 21,
        "stage_name": stage_name,
        "tool": tool,
        "result": {
            "check_pass": passed,
            "failed_cases_top": list(failed_cases or []),
            "failed_checkpoints_top": [],
            "checker_categories": list(categories or []),
            "error_top": list(error_top or []),
        },
    }


def test_decreasing_checker_count_is_progress_not_stagnation():
    controller = ProgressAwareLoopController({"no_progress_patience": 2})

    first = controller.observe(_checker_result(
        False,
        ["test_marking_contract_incomplete"],
        error_top=["Find 24 functions do not have correct check point marks"],
    ))
    second = controller.observe(_checker_result(
        False,
        ["test_marking_contract_incomplete"],
        error_top=["Find 23 functions do not have correct check point marks"],
    ))
    third = controller.observe(_checker_result(
        False,
        ["test_marking_contract_incomplete"],
        error_top=["Find 23 functions do not have correct check point marks"],
    ))

    assert first["credit_role"] == "diagnostic"
    assert second["credit_role"] == "progress"
    assert second["policy"] == "exploit"
    assert second["checker_no_progress_streak"] == 0
    assert second["observation_after"]["progress_metrics"] == {
        "unmarked_test_functions": 23,
    }
    assert third["credit_role"] == "no_progress"
    assert third["checker_no_progress_streak"] == 1


def test_marking_contract_gets_atomic_batch_budget_before_pivot():
    controller = ProgressAwareLoopController({
        "hard_enforce_mutation_budget": True,
        "max_mutations_marking_contract": 16,
    })

    decision = controller.observe(_checker_result(
        False,
        ["test_marking_contract_incomplete"],
        error_top=["Find 13 functions do not have correct check point marks"],
    ))

    assert decision["policy"] == "repair_marking_contract"
    assert decision["mutation_budget"]["limit"] == 16
    assert "one repair batch" in decision["control"]["constraints"][0]


def test_checkpoint_evidence_mismatch_gets_full_reconciliation_budget():
    controller = ProgressAwareLoopController({
        "hard_enforce_mutation_budget": True,
        "max_mutations_checkpoint_evidence": 12,
    })

    decision = controller.observe(_checker_result(
        False,
        ["bug_doc_checkpoint_not_marked"],
        stage_name="static_bug_validation",
        error_top=["[Checkpoint Not Marked] 8 test cases need marks"],
    ))

    assert decision["policy"] == "reconcile_checkpoint_evidence"
    assert decision["mutation_budget"]["limit"] == 12
    assert "TC-to-FG/FC/CK" in decision["control"]["constraints"][0]


def test_repeated_failure_automatically_pivots_after_patience():
    controller = ProgressAwareLoopController({"no_progress_patience": 2})
    controller.observe(_mutation())
    first = controller.observe(_test_result("test_failure", ["test_sum"]))
    controller.observe(_mutation())
    second = controller.observe(_test_result("test_failure", ["test_sum"]))
    controller.observe(_mutation())
    third = controller.observe(_test_result("test_failure", ["test_sum"]))

    assert first["credit_role"] == "diagnostic"
    assert second["credit_role"] == "no_progress"
    assert second["policy"] == "diagnose"
    assert third["credit_role"] == "no_progress"
    assert third["policy"] == "pivot"
    assert third["control"]["max_next_file_mutations"] == 1


def test_failure_delta_distinguishes_progress_and_regression():
    controller = ProgressAwareLoopController()
    controller.observe(_test_result("test_failure", ["test_a", "test_b"]))
    controller.observe(_mutation())
    progress = controller.observe(_test_result("test_failure", ["test_b"]))
    controller.observe(_mutation())
    regression = controller.observe(_test_result("test_failure", ["test_b", "test_c"]))

    assert progress["credit_role"] == "progress"
    assert regression["credit_role"] == "regression"
    assert regression["policy"] == "contain"


def test_invalid_test_execution_blocks_semantic_repair():
    controller = ProgressAwareLoopController()
    decision = controller.observe(_test_result(
        "no_tests_collected",
        stage_name="basic_api_functional_test",
    ))

    assert decision["credit_role"] == "invalid"
    assert decision["policy"] == "repair_execution"
    assert decision["control"]["max_next_file_mutations"] == 3
    assert decision["control"]["stage_contract"] == "documentable_dut_failure"


def test_template_stage_plain_pass_is_a_contract_violation():
    controller = ProgressAwareLoopController({"hard_enforce_mutation_budget": True})

    decision = controller.observe(_test_result(
        "pass",
        stage_name="create_test_case_templates",
        outcome_reason="verified_nonempty_pass",
    ))

    assert decision["credit_role"] == "diagnostic"
    assert decision["policy"] == "restore_template_contract"
    assert decision["mutation_budget"]["limit"] == 8
    assert "contract:template_tests_must_expected_fail" in decision["observation_after"]["failure_set"]


def test_repeated_template_checker_failure_keeps_batch_sized_pivot_budget():
    controller = ProgressAwareLoopController({
        "hard_enforce_mutation_budget": True,
        "no_progress_patience": 2,
        "max_mutations_template_contract": 8,
    })
    event = _checker_result(False, ["template_validation_or_runner_contract"])
    event["stage_name"] = "create_test_case_templates"

    controller.observe(event)
    controller.observe(event)
    decision = controller.observe(event)

    assert decision["policy"] == "pivot"
    assert decision["mutation_budget"]["limit"] == 8
    assert decision["control"]["max_next_file_mutations"] == 8
    assert any("template batch" in item for item in decision["control"]["constraints"])


def test_template_stage_expected_placeholder_failure_can_verify_stage():
    controller = ProgressAwareLoopController()

    decision = controller.observe(_test_result(
        "pass",
        stage_name="create_test_case_templates",
        outcome_reason="expected_template_xfail",
        status_counts={"XFAIL": 4},
    ))

    assert decision["policy"] == "verify_stage"
    assert decision["control"]["stage_contract"] == "template_expected_failure"


def test_documentable_failure_is_classified_without_forcing_pass_or_xfail():
    controller = ProgressAwareLoopController()

    decision = controller.observe(_test_result(
        "test_failure",
        ["test_width"],
        stage_name="comprehensive_verification_and_bug_analysis",
        outcome_reason="reported_non_pass_status",
        status_counts={"PASSED": 3, "XFAIL": 1},
    ))

    assert decision["policy"] == "classify_failure"
    assert decision["control"]["stage_contract"] == "documentable_dut_failure"
    assert "remove the expected-failure marker" in decision["control"]["constraints"][0]
    assert decision["control"]["max_next_file_mutations"] == 3


def test_repeated_documentable_failure_still_pivots():
    controller = ProgressAwareLoopController({"no_progress_patience": 2})
    event = _test_result(
        "test_failure",
        ["test_width"],
        stage_name="comprehensive_verification_and_bug_analysis",
    )

    first = controller.observe(event)
    second = controller.observe(event)
    third = controller.observe(event)

    assert first["policy"] == "classify_failure"
    assert second["policy"] == "classify_failure"
    assert third["policy"] == "pivot"
    assert any("ordinary FAILED" in item for item in third["control"]["constraints"])


def test_string_false_disables_controller():
    controller = ProgressAwareLoopController({"enabled": "false"})

    assert controller.observe(_test_result("test_failure", ["test_a"])) is None


def test_targeted_test_pass_does_not_clear_repeated_checker_state():
    controller = ProgressAwareLoopController({"no_progress_patience": 2})

    controller.observe(_checker_result(False, ["bug_doc_schema"]))
    second = controller.observe(_checker_result(False, ["bug_doc_schema"], tool="Complete"))
    local_pass = controller.observe(_test_result("pass"))
    third = controller.observe(_checker_result(False, ["bug_doc_schema"]))

    assert second["checker_no_progress_streak"] == 1
    assert local_pass["observation_after"]["failure_set"] == []
    assert local_pass["policy"] == "verify_stage"
    assert third["checker_no_progress_streak"] == 2
    assert third["policy"] == "pivot"


def test_checker_category_churn_does_not_reset_documentable_failure_stall():
    controller = ProgressAwareLoopController({"no_progress_patience": 2})
    stage_name = "test_case_implementation_in_batch"

    first = controller.observe(_checker_result(
        False,
        ["failed_cases_need_classification"],
        failed_cases=["unity_test/tests/test_Adder.py:10-20::test_msb"],
        stage_name=stage_name,
    ))
    second = controller.observe(_checker_result(
        False,
        ["bug_doc_checkpoint_not_found"],
        failed_cases=["unity_test/tests/test_Adder.py:14-24::test_msb"],
        stage_name=stage_name,
    ))
    third = controller.observe(_checker_result(
        False,
        ["generic_checker_failure"],
        failed_cases=["unity_test/tests/test_Adder.py:18-28::test_msb"],
        stage_name=stage_name,
    ))

    assert first["checker_no_progress_streak"] == 0
    assert second["checker_no_progress_streak"] == 1
    assert third["checker_no_progress_streak"] == 2
    assert third["policy"] == "pivot"


def test_hard_mutation_budget_rejects_calls_beyond_policy_limit():
    controller = ProgressAwareLoopController({
        "hard_enforce_mutation_budget": True,
        "max_mutations_diagnostic": 2,
    })
    controller.observe(_checker_result(False, ["missing_assert"]))

    first, first_detail = controller.authorize_mutation(21, tool="EditTextFile", path="a.py")
    second, second_detail = controller.authorize_mutation(21, tool="EditTextFile", path="b.py")
    third, third_detail = controller.authorize_mutation(21, tool="EditTextFile", path="c.py")

    assert first is True and first_detail["remaining"] == 1
    assert second is True and second_detail["remaining"] == 0
    assert third is False
    assert third_detail["reason"] == "mutation_budget_exhausted"


def test_failed_mutation_can_refund_pre_authorized_budget():
    controller = ProgressAwareLoopController({
        "hard_enforce_mutation_budget": True,
        "max_mutations_diagnostic": 1,
    })
    controller.observe(_checker_result(False, ["missing_assert"]))

    allowed, _ = controller.authorize_mutation(21, tool="EditTextFile", path="a.py")
    refunded = controller.refund_mutation(21)

    assert allowed is True
    assert refunded["consumed"] == 0
    assert refunded["remaining"] == 1


def test_required_exact_stage_output_cannot_be_deleted():
    agent = VerifyAgent.__new__(VerifyAgent)
    agent.workspace = "/tmp/workspace"
    agent.stage_manager = SimpleNamespace(
        stage_index=0,
        stages=[SimpleNamespace(output_files=["unity_test/report.md", "unity_test/tests/test_*.py"])],
    )
    agent.progress_controller = ProgressAwareLoopController({
        "hard_enforce_mutation_budget": True,
    })
    events = []
    agent.record_structured_event = lambda event_type, detail: events.append((event_type, detail))

    result = agent._guard_file_mutation_call(
        SimpleNamespace(name="DeleteFile"),
        {"path": "unity_test/report.md"},
    )

    assert "MUTATION_CONTRACT_BLOCKED" in result
    assert events[0][0] == "mutation_contract_blocked"
    assert agent._protected_required_output(0, "DeleteFile", "unity_test/tests/test_old.py") == ""


def test_template_stage_blocks_conftest_and_reporter_data_deletion():
    agent = VerifyAgent.__new__(VerifyAgent)
    agent.dut_name = "DUT"
    agent.stage_manager = SimpleNamespace(
        stage_index=22,
        stages=[SimpleNamespace(name="unused")] * 22
        + [SimpleNamespace(name="create_test_case_templates")],
    )
    agent.progress_controller = ProgressAwareLoopController({
        "hard_enforce_mutation_budget": True,
    })

    conftest = agent._protected_stage_mutation_contract(
        22,
        "EditTextFile",
        "unity_test/tests/conftest.py",
        {"path": "unity_test/tests/conftest.py", "mode": "write", "data": "# workaround"},
    )
    reporter_data = agent._protected_stage_mutation_contract(
        22,
        "DeleteFile",
        "unity_test/tests/data",
        {"path": "unity_test/tests/data", "recursive": True},
    )

    assert "conftest.py" in conftest["reason"]
    assert "generated evidence" in reporter_data["reason"]


def test_static_bug_analysis_uses_checker_only_contract():
    controller = ProgressAwareLoopController({"no_progress_patience": 1})
    event = _checker_result(False, ["static_bug_tag_hierarchy"])
    event["stage_name"] = "static_bug_analysis"

    decision = controller.observe(event)

    assert decision["control"]["stage_contract"] == "checker_only_document"
    assert "do not run pytest" in decision["control"]["required_validation"]


def test_local_pass_hard_blocks_more_edits_until_stage_verification():
    controller = ProgressAwareLoopController({"hard_enforce_mutation_budget": True})

    decision = controller.observe(_test_result("pass"))
    allowed, detail = controller.authorize_mutation(21, tool="EditTextFile", path="late.py")

    assert decision["policy"] == "verify_stage"
    assert decision["control"]["max_next_file_mutations"] == 0
    assert allowed is False
    assert detail["remaining"] == 0


def test_different_test_targets_do_not_create_false_regression():
    controller = ProgressAwareLoopController()
    controller.observe(_test_result("pass", target="test_a.py::test_a"))

    decision = controller.observe(_test_result(
        "test_failure",
        ["test_b"],
        target="test_b.py::test_b",
    ))

    assert decision["credit_role"] == "diagnostic"
    assert decision["failure_before"]["failure_set"] == []
    assert decision["observation_scope"].startswith("test:")


def test_same_pytest_node_line_churn_is_not_progress():
    controller = ProgressAwareLoopController()
    first = _test_result("test_failure", ["tests/test_chip.py:10-20::test_case"])
    second = _test_result("test_failure", ["tests/test_chip.py:30-45::test_case"])

    controller.observe(first)
    decision = controller.observe(second)

    assert decision["failure_set_delta"]["added"] == []
    assert decision["failure_set_delta"]["removed"] == []
    assert decision["credit_role"] == "no_progress"


def test_stage21_blocks_destructive_api_test_rewrite():
    agent = VerifyAgent.__new__(VerifyAgent)
    agent.workspace = "/tmp/workspace"
    agent.stage_manager = SimpleNamespace(
        stage_index=0,
        stages=[SimpleNamespace(name="basic_api_functional_test", output_files=[])],
    )
    agent.progress_controller = ProgressAwareLoopController({"hard_enforce_mutation_budget": True})
    events = []
    agent.record_structured_event = lambda event_type, detail: events.append((event_type, detail))

    result = agent._guard_file_mutation_call(
        SimpleNamespace(name="EditTextFile"),
        {
            "path": "unity_test/tests/test_Adder_api_basic.py",
            "mode": "replace",
            "start": 0,
            "count": -1,
        },
    )

    assert "MUTATION_CONTRACT_BLOCKED" in result
    assert "PASSED node id" in result
    assert events[0][1]["reason"].startswith("Stage 21")


def test_stage21_blocks_renamed_api_test_duplicate():
    agent = VerifyAgent.__new__(VerifyAgent)
    agent.workspace = "/tmp/workspace"
    agent.stage_manager = SimpleNamespace(
        stage_index=0,
        stages=[SimpleNamespace(name="basic_api_functional_test", output_files=[])],
    )
    agent.progress_controller = ProgressAwareLoopController({"hard_enforce_mutation_budget": True})
    agent.record_structured_event = lambda *_args: None

    result = agent._guard_file_mutation_call(
        SimpleNamespace(name="EditFileDialogs"),
        {
            "path": "unity_test/tests/test_Mux_api_basic_fixed.py",
            "mode": "write",
            "data": "def test_duplicate(): pass",
        },
    )

    assert "MUTATION_CONTRACT_BLOCKED" in result
    assert "renamed duplicate" in result


def test_stage21_blocks_mark_function_name_string():
    agent = VerifyAgent.__new__(VerifyAgent)
    agent.workspace = "/tmp/workspace"
    agent.stage_manager = SimpleNamespace(
        stage_index=0,
        stages=[SimpleNamespace(name="basic_api_functional_test", output_files=[])],
    )
    agent.progress_controller = ProgressAwareLoopController({"hard_enforce_mutation_budget": True})
    agent.record_structured_event = lambda *_args: None

    result = agent._guard_file_mutation_call(
        SimpleNamespace(name="ReplaceStringInFile"),
        {
            "path": "unity_test/tests/test_Mux_api_basic.py",
            "new_string": (
                "env.dut.fc_cover['FG-API'].mark_function(\n"
                "    env, test_api_Mux_set_input.__name__, ['FC-SET/CK-BASIC'])"
            ),
        },
    )

    assert "MUTATION_CONTRACT_BLOCKED" in result
    assert "function object" in result


def test_stage23_requires_schema_aware_bug_recording():
    agent = VerifyAgent.__new__(VerifyAgent)
    agent.workspace = "/tmp/workspace"
    agent.stage_manager = SimpleNamespace(
        stage_index=0,
        stages=[SimpleNamespace(name="test_case_implementation_in_batch", output_files=[])],
    )
    agent.progress_controller = ProgressAwareLoopController({"hard_enforce_mutation_budget": True})
    agent.record_structured_event = lambda *_args: None

    result = agent._guard_file_mutation_call(
        SimpleNamespace(name="ReplaceStringInFile"),
        {"path": "unity_test/Adder_bug_analysis.md"},
    )

    assert "MUTATION_CONTRACT_BLOCKED" in result
    assert "recordbug.py" in result


def test_stage23_blocks_workspace_local_recordbug_helper():
    agent = VerifyAgent.__new__(VerifyAgent)
    agent.workspace = "/tmp/workspace"
    agent.stage_manager = SimpleNamespace(
        stage_index=0,
        stages=[SimpleNamespace(name="test_case_implementation_in_batch", output_files=[])],
    )
    agent.progress_controller = ProgressAwareLoopController({"hard_enforce_mutation_budget": True})
    agent.record_structured_event = lambda *_args: None

    result = agent._guard_file_mutation_call(
        SimpleNamespace(name="EditFileDialogs"),
        {
            "path": "unity_test/recordbug.py",
            "mode": "write",
            "data": "def main(): pass",
        },
    )

    assert "MUTATION_CONTRACT_BLOCKED" in result
    assert "RunSkillScript" in result


def test_stage23_blocks_dut_and_rtl_mutations_with_bug_record_redirect():
    agent = VerifyAgent.__new__(VerifyAgent)
    agent.workspace = "/tmp/workspace"
    agent.dut_name = "Adder"
    agent.stage_manager = SimpleNamespace(
        stage_index=0,
        stages=[SimpleNamespace(name="test_case_implementation_in_batch", output_files=[])],
    )
    agent.progress_controller = ProgressAwareLoopController({"hard_enforce_mutation_budget": True})
    agent.record_structured_event = lambda *_args: None

    for path in ("Adder/__init__.py", "Adder_RTL/Adder.v"):
        result = agent._guard_file_mutation_call(
            SimpleNamespace(name="ReplaceStringInFile"),
            {"path": path, "old_string": "bad", "new_string": "fixed"},
        )

        assert "MUTATION_CONTRACT_BLOCKED" in result
        assert "immutable DUT implementation" in result
        assert "RunSkillScript" in result
        assert "recordbug.py" in result
        assert "do not rerun the identical test" in result


def test_stage23_blocks_api_output_mask_that_hides_dut_failure():
    agent = VerifyAgent.__new__(VerifyAgent)
    agent.workspace = "/tmp/workspace"
    agent.dut_name = "Adder"
    agent.stage_manager = SimpleNamespace(
        stage_index=0,
        stages=[SimpleNamespace(name="test_case_implementation_in_batch", output_files=[])],
    )
    agent.progress_controller = ProgressAwareLoopController({"hard_enforce_mutation_budget": True})
    agent.progress_controller._latest_decision[0] = {"policy": "classify_failure"}
    agent.record_structured_event = lambda *_args: None

    result = agent._guard_file_mutation_call(
        SimpleNamespace(name="ReplaceStringInFile"),
        {
            "path": "unity_test/tests/Adder_api.py",
            "old_string": "sum_val = self.sum.value",
            "new_string": (
                "# Due to DUT Bug, mask the bad high bit\n"
                "sum_val = self.sum.value & 0x7fffffffffffffff"
            ),
        },
    )

    assert "MUTATION_CONTRACT_BLOCKED" in result
    assert "hides the defect" in result
    assert "recordbug.py" in result
