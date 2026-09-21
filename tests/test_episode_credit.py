from __future__ import annotations

from scripts.build_context_reuse_pack import (
    CREDIT_ROLES,
    annotation_record,
    build_credit_episode,
    build_pack,
    select_annotation_sample,
)
from scripts.evaluate_episode_credit_annotations import (
    evaluate_records,
    merge_annotation_records,
    refresh_predictions,
)
from scripts.analyze_context_reuse_decisions import analyze_trace
from scripts.build_trace_tree import node_from_event


REQUIRED_CREDIT_FIELDS = {
    "credit_rule_version",
    "failure_before",
    "action_sequence",
    "observation_after",
    "stage_advanced",
    "failure_set_delta",
    "regression_count",
    "information_gain",
    "action_cost",
    "credit_role",
    "confidence",
}


def make_test_node(node_id: str, outcome: str, *, failed_cases=None, target="test_a.py", time=0.0):
    failed_cases = list(failed_cases or [])
    failed = len(failed_cases)
    return {
        "node_id": node_id,
        "event_type": "test_run",
        "stage_index": 16,
        "stage_name": "evaluate_env_fixture",
        "run_id": "run-a",
        "time_unix": time,
        "observation": {
            "tool": "RunTestCases",
            "target": target,
            "tests_total": max(1, failed),
            "tests_failed": failed,
            "run_test_success": outcome == "pass",
            "test_outcome": outcome,
            "tests_passed_all": outcome == "pass",
            "failed_cases_top": failed_cases,
            "failure_signature": f"sig-{node_id}",
            "error_top": ["AssertionError"] if failed else [],
        },
    }


def mutation_node(node_id="m1", time=1.0):
    return {
        "node_id": node_id,
        "event_type": "file_mutation",
        "stage_index": 16,
        "stage_name": "evaluate_env_fixture",
        "run_id": "run-a",
        "time_unix": time,
        "success": True,
        "action": {
            "tool": "WriteTextFile",
            "operation": "write",
            "path": "output/tests/test_api_Adder_env.py",
            "path_category": "test_code",
            "detail": {"text": {"chars": 120, "lines": 5, "sha256": "abc"}},
        },
    }


def transition_node(time=3.0):
    return {
        "node_id": "t1",
        "event_type": "stage_transition",
        "stage_index": 16,
        "stage_name": "evaluate_env_fixture",
        "run_id": "run-a",
        "time_unix": time,
        "reward": {"advanced": True, "all_completed": False},
    }


def check_node(node_id: str, passed: bool, *, time=2.0):
    return {
        "node_id": node_id,
        "event_type": "check_result",
        "stage_index": 16,
        "stage_name": "evaluate_env_fixture",
        "run_id": "run-a",
        "time_unix": time,
        "observation": {
            "tool": "Check",
            "check_pass": passed,
            "failure_signature": f"sig-{node_id}",
            "error_top": [] if passed else ["checker failure"],
        },
    }


def build_episode(after, actions=None, include_transition=False):
    before = make_test_node("f1", "test_failure", failed_cases=["test_a.py::test_a"], time=0.0)
    actions = list(actions or [])
    nodes = [before, *actions, after]
    if include_transition:
        nodes.append(transition_node())
    return build_credit_episode(
        dut="Adder",
        run_id="run-a",
        stage_index=16,
        stage_name="evaluate_env_fixture",
        failure_node=before,
        after_node=after,
        after_index=1 + len(actions),
        stage_nodes=nodes,
        action_nodes=actions,
    )


def test_progress_episode_has_required_schema_and_stage_credit():
    episode = build_episode(
        make_test_node("p1", "pass", time=2.0),
        actions=[mutation_node()],
        include_transition=True,
    )

    assert REQUIRED_CREDIT_FIELDS <= episode.keys()
    assert episode["credit_role"] == "progress"
    assert episode["stage_advanced"] is True
    assert episode["failure_set_delta"]["net_reduction"] > 0
    assert episode["reuse_eligible"] is True


def test_invalid_test_execution_never_receives_progress_credit():
    after = make_test_node("n1", "no_tests_collected", time=2.0)
    after["observation"].update(tests_total=0, tests_failed=0, run_test_success=False)

    episode = build_episode(after, actions=[mutation_node()])

    assert episode["credit_role"] == "invalid"
    assert episode["reuse_eligible"] is False


def test_unchanged_failure_without_actions_is_no_progress():
    after = make_test_node("f2", "test_failure", failed_cases=["test_a.py::test_a"], time=2.0)

    episode = build_episode(after)

    assert episode["credit_role"] == "no_progress"


def test_new_failure_without_reduction_is_regression():
    after = make_test_node(
        "r1",
        "test_failure",
        failed_cases=["test_a.py::test_a", "test_b.py::test_b"],
        time=2.0,
    )

    episode = build_episode(after, actions=[mutation_node()])

    assert episode["credit_role"] == "regression"
    assert episode["regression_count"] == 1


def test_narrowed_scope_with_same_failure_is_not_false_progress():
    before = make_test_node(
        "f-file",
        "test_failure",
        failed_cases=["test_a.py::test_a", "test_a.py::test_b"],
        target="test_a.py",
        time=0.0,
    )
    after = make_test_node(
        "f-one",
        "test_failure",
        failed_cases=["test_a.py::test_a"],
        target="test_a.py::test_a",
        time=2.0,
    )
    episode = build_credit_episode(
        dut="Adder",
        run_id="run-a",
        stage_index=16,
        stage_name="evaluate_env_fixture",
        failure_node=before,
        after_node=after,
        after_index=1,
        stage_nodes=[before, after],
        action_nodes=[],
    )

    assert episode["validation_scope_relation"] == "narrowed"
    assert episode["failure_set_delta"]["unobserved_before"] == ["case:test_a.py::test_b"]
    assert episode["failure_set_delta"]["net_reduction"] == 0
    assert episode["credit_role"] == "no_progress"


def test_expanded_scope_does_not_treat_newly_observed_failures_as_regression():
    before = make_test_node(
        "f-one",
        "test_failure",
        failed_cases=["test_a.py::test_a"],
        target="test_a.py::test_a",
        time=0.0,
    )
    after = make_test_node(
        "f-file",
        "test_failure",
        failed_cases=["test_a.py::test_b", "test_a.py::test_c"],
        target="test_a.py",
        time=2.0,
    )
    episode = build_credit_episode(
        dut="Adder",
        run_id="run-a",
        stage_index=16,
        stage_name="evaluate_env_fixture",
        failure_node=before,
        after_node=after,
        after_index=2,
        stage_nodes=[before, mutation_node(), after],
        action_nodes=[mutation_node()],
    )

    assert episode["validation_scope_relation"] == "expanded"
    assert episode["failure_set_delta"]["scope_expansion_items"] == [
        "case:test_a.py::test_b",
        "case:test_a.py::test_c",
    ]
    assert episode["regression_count"] == 0
    assert episode["credit_role"] == "progress"


def test_generic_signature_change_is_not_concrete_regression():
    before = check_node("before", False, time=0.0)
    after = check_node("after", False, time=2.0)
    before["observation"]["failure_pattern"] = "generic_checker_failure"
    after["observation"]["failure_pattern"] = "generic_checker_failure"
    episode = build_credit_episode(
        dut="Adder",
        run_id="run-a",
        stage_index=16,
        stage_name="evaluate_env_fixture",
        failure_node=before,
        after_node=after,
        after_index=2,
        stage_nodes=[before, mutation_node(), after],
        action_nodes=[mutation_node()],
    )

    assert episode["regression_count"] == 0
    assert episode["credit_role"] == "no_progress"


def test_new_specific_error_class_in_same_checker_is_regression():
    before = check_node("before", False, time=0.0)
    after = check_node("after", False, time=2.0)
    before["observation"].update(
        error_top=["Bug analysis document has an invalid BG/TC hierarchy"],
        failed_checkpoints_top=["FG-A/FC-A/CK-A"],
    )
    after["observation"].update(
        error_top=["FG-A is defined multiple times"],
        failed_checkpoints_top=["FG-A/FC-A/CK-A"],
    )
    episode = build_credit_episode(
        dut="Adder",
        run_id="run-a",
        stage_index=19,
        stage_name="basic_api_functional_test",
        failure_node=before,
        after_node=after,
        after_index=2,
        stage_nodes=[before, mutation_node(), after],
        action_nodes=[mutation_node()],
    )

    assert episode["failure_set_delta"]["semantic_regressions"] == [
        "error_class:duplicate_label_definition"
    ]
    assert episode["credit_role"] == "regression"


def test_disjoint_scope_with_new_actionable_error_is_diagnostic_not_regression():
    before = make_test_node(
        "before",
        "test_failure",
        failed_cases=["test_a.py::test_reset"],
        target="test_a.py::test_reset",
        time=0.0,
    )
    after = make_test_node(
        "after",
        "test_failure",
        failed_cases=["test_a.py::test_press"],
        target="test_a.py::test_press",
        time=2.0,
    )
    after["observation"]["error_top"] = ["signal name: current_state not found"]
    episode = build_credit_episode(
        dut="FSM",
        run_id="run-a",
        stage_index=19,
        stage_name="basic_api_functional_test",
        failure_node=before,
        after_node=after,
        after_index=2,
        stage_nodes=[before, mutation_node(), after],
        action_nodes=[mutation_node()],
    )

    assert episode["validation_scope_relation"] == "disjoint"
    assert episode["regression_count"] == 0
    assert episode["information_gain"]["new_actionable_evidence"] == [
        "signal_not_found:current_state"
    ]
    assert episode["credit_role"] == "diagnostic"


def test_cross_validator_success_without_stage_advance_is_not_repair_progress():
    episode = build_episode(check_node("c1", True))

    assert episode["failure_set_delta"]["comparable"] is False
    assert episode["credit_role"] == "no_progress"
    assert episode["reuse_eligible"] is False


def test_cross_validator_success_can_receive_progress_only_when_stage_advances():
    episode = build_episode(check_node("c2", True), include_transition=True)

    assert episode["failure_set_delta"]["comparable"] is False
    assert episode["credit_role"] == "progress"
    assert episode["stage_advanced"] is True
    assert episode["reuse_eligible"] is False


def test_build_pack_does_not_pair_observations_across_runs():
    failure = make_test_node("f1", "test_failure", failed_cases=["test_a.py::test_a"], time=0.0)
    other_run_pass = make_test_node("p1", "pass", time=1.0)
    other_run_pass["run_id"] = "run-b"
    trace = {
        "trace_type": "ucagent_structured_trajectory",
        "dut": "Adder",
        "nodes": [failure, other_run_pass],
    }

    pack = build_pack([trace])

    assert pack["run_count"] == 2
    assert pack["episodes"] == []


def test_build_pack_ignores_events_after_terminal_transition():
    terminal = transition_node(time=1.0)
    terminal["reward"] = {"advanced": True, "all_completed": True}
    post_failure = make_test_node(
        "post-failure",
        "test_failure",
        failed_cases=["test_a.py::test_a"],
        time=2.0,
    )
    post_failure.update(stage_index=26, stage_name="")
    post_pass = make_test_node("post-pass", "pass", time=3.0)
    post_pass.update(stage_index=26, stage_name="")
    trace = {
        "trace_type": "ucagent_structured_trajectory",
        "dut": "Adder",
        "nodes": [terminal, post_failure, post_pass],
    }

    pack = build_pack([trace])

    assert pack["episodes"] == []
    assert all(summary["stage_index"] != 26 for summary in pack["stage_summaries"])


def test_annotation_sample_and_metrics_cover_credit_roles():
    episodes = [
        {"episode_id": f"{role}-{index}", "credit_role": role}
        for role in CREDIT_ROLES
        for index in range(3)
    ]
    sample = select_annotation_sample(episodes, size=10, seed=7)
    sampled_roles = {item["credit_role"] for item in sample}
    records = [
        {
            "model_credit_role": role,
            "annotations": [
                {"annotator": "a", "credit_role": role},
                {"annotator": "b", "credit_role": role},
            ],
        }
        for role in CREDIT_ROLES
    ]

    metrics = evaluate_records(records)

    assert sampled_roles == set(CREDIT_ROLES)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1_present_roles"] == 1.0
    assert metrics["cohen_kappa"] == 1.0


def test_blind_annotation_record_withholds_rule_prediction():
    episode = build_episode(make_test_node("p1", "pass", time=2.0), actions=[mutation_node()])

    record = annotation_record(episode, blind=True)

    assert "model_credit_role" not in record
    assert "model_confidence" not in record
    assert "credit_role" not in record["evidence"]
    assert "confidence" not in record["evidence"]
    assert "failure_set_delta" not in record["evidence"]
    assert "regression_count" not in record["evidence"]
    assert "information_gain" not in record["evidence"]
    assert "action_cost" not in record["evidence"]
    assert "failure_pattern" not in record["evidence"]["failure_before"]
    assert "failure_signature" not in record["evidence"]["failure_before"]
    assert "test_outcome" not in record["evidence"]["failure_before"]


def test_refresh_predictions_preserves_independent_labels():
    records = [{
        "episode_id": "ep-1",
        "model_credit_role": "no_progress",
        "model_confidence": 0.4,
        "evidence": {"old": True},
        "annotations": [{"annotator": "blind-a", "credit_role": "progress", "notes": "kept"}],
    }]
    pack = {"episodes": [{
        "episode_id": "ep-1",
        "credit_role": "progress",
        "confidence": 0.9,
        "failure_before": {"node_id": "before"},
    }]}

    refreshed, summary = refresh_predictions(records, pack)

    assert summary == {
        "records_total": 1,
        "matched": 1,
        "missing": 0,
        "missing_episode_ids": [],
    }
    assert refreshed[0]["model_credit_role"] == "progress"
    assert refreshed[0]["model_confidence"] == 0.9
    assert refreshed[0]["annotations"] == records[0]["annotations"]
    assert records[0]["model_credit_role"] == "no_progress"


def test_context_reuse_decision_links_injection_to_next_validation():
    failure = make_test_node("f1", "test_failure", failed_cases=["test_a.py::test_a"], time=0.0)
    decision = {
        "node_id": "d1",
        "event_type": "context_reuse_decision",
        "stage_index": 16,
        "stage_name": "evaluate_env_fixture",
        "run_id": "run-a",
        "time_unix": 0.5,
        "memory_decision": {
            "decision": "inject",
            "failure": {"signature": "sig-f1"},
            "strategies": [{
                "strategy_id": "strategy-a",
                "transition_contract": {
                    "contract_version": "verifier_transition_contract_v0",
                    "expected_transition": {
                        "credit_role": "progress",
                        "min_failure_net_reduction": 1,
                    },
                    "safety_constraints": {"max_regression_count": 0},
                    "verification": {"event_type": "test_run"},
                },
            }],
            "search_summary": {
                "match_candidates": 2,
                "utility_passed": 1,
                "utility_rejected": 1,
                "contract_candidates": 1,
                "contract_applicable": 1,
            },
            "prompt_item_count": 1,
        },
    }
    passed = make_test_node("p1", "pass", time=2.0)
    trace = {
        "dut": "Adder",
        "nodes": [failure, decision, mutation_node(time=1.0), passed, transition_node(time=3.0)],
    }

    result = analyze_trace(trace)

    assert result["decision_counts"] == {"inject": 1}
    assert result["credit_role_counts"] == {"progress": 1}
    assert result["observed_progress_rate"] == 1.0
    assert result["strategy_outcomes"] == {"strategy-a": {"progress": 1}}
    assert result["utility_gate_counts"] == {
        "match_candidates": 2,
        "passed_candidates": 1,
        "rejected_candidates": 1,
        "unknown_candidates": 0,
    }
    assert result["records"][0]["search_summary"]["utility_rejected"] == 1
    assert result["contract_applicability_counts"]["candidates"] == 1
    assert result["contract_outcome_counts"] == {"fulfilled": 1}
    assert result["observed_contract_fulfillment_rate"] == 1.0


def test_trace_tree_preserves_context_reuse_decision_payload():
    event = {
        "event_type": "context_reuse_decision",
        "thread_id": 42,
        "stage_index": 21,
        "stage_name": "test_case_implementation_in_batch",
        "decision": "inject",
        "failure": {"signature": "sig-a", "pattern": "test_logic_or_case_implementation"},
        "strategies": [{"strategy_id": "strategy-a", "score": 1.2}],
        "search_summary": {"match_candidates": 2, "utility_passed": 1, "utility_rejected": 1},
        "hint_count": 1,
        "prompt_item_count": 2,
    }

    node = node_from_event(event, "n1", None)

    assert node["role"] == "memory_action"
    assert node["memory_decision"]["decision"] == "inject"
    assert node["memory_decision"]["failure"]["signature"] == "sig-a"
    assert node["memory_decision"]["strategies"][0]["strategy_id"] == "strategy-a"
    assert node["memory_decision"]["search_summary"]["utility_rejected"] == 1


def test_merge_independent_annotations_by_episode_id():
    base = [{
        "episode_id": "ep-1",
        "annotations": [{"annotator": "a", "credit_role": "progress", "notes": ""}],
    }]
    independent = [{
        "episode_id": "ep-1",
        "annotations": [{"annotator": "b", "credit_role": "diagnostic", "notes": "uncertain"}],
    }]

    merged, summary = merge_annotation_records(base, [independent])

    assert [item["credit_role"] for item in merged[0]["annotations"]] == ["progress", "diagnostic"]
    assert summary["source_sets"] == 2
    assert summary["records_double_labeled"] == 1
    assert summary["missing"] == 0
