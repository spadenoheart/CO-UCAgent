from __future__ import annotations

import copy

from scripts.build_context_reuse_pack import is_success_observation
from scripts.curate_context_reuse_pack import (
    aggregate_strategy_utility,
    curate_pack,
    episode_verifier_utility,
)


def make_episode(dut: str, action_path: str, success: dict | None = None) -> dict:
    return {
        "dut": dut,
        "stage_index": 16,
        "stage_name": "evaluate_env_fixture",
        "failure_pattern": "api_or_env_interface_mismatch",
        "failure_group": "interface",
        "action_categories": ["test_code"],
        "preferred_action_categories": ["test_code", "code"],
        "failure": {
            "event_type": "check_result",
            "check_pass": False,
            "failure_signature": "same-signature",
            "failure_pattern": "api_or_env_interface_mismatch",
            "checker_categories": ["api_or_env_interface_mismatch"],
            "error_top": ["env has no attribute output"],
        },
        "actions": [{
            "operation": "replace_string",
            "path": action_path,
            "path_category": "test_code",
        }],
        "success": success or {
            "event_type": "test_run",
            "run_test_success": True,
            "tests_total": 2,
            "tests_failed": 0,
            "tests_passed_all": True,
            "failed_cases_top": [],
            "error_top": [],
        },
    }


def test_trace_pack_rejects_zero_test_and_inconsistent_success():
    zero_test_node = {
        "event_type": "test_run",
        "observation": {
            "run_test_success": True,
            "tests_total": 0,
            "tests_failed": 0,
            "tests_passed_all": True,
        },
    }
    inconsistent_node = {
        "event_type": "test_run",
        "observation": {
            "tests_total": 1,
            "tests_failed": 0,
            "tests_passed_all": True,
            "failed_cases_top": ["test_api.py::test_setup"],
        },
    }

    assert is_success_observation(zero_test_node) is False
    assert is_success_observation(inconsistent_node) is False


def test_curator_filters_false_success_deduplicates_and_aggregates_cross_dut():
    adder = make_episode("Adder", "tests/test_Adder_env.py")
    adder_duplicate = copy.deepcopy(adder)
    adder_duplicate["actions"][0]["path"] = "tests/test_Adder_env_fixture.py"
    uart = make_episode("uart_tx", "tests/test_uart_tx_env.py")
    invalid = make_episode(
        "FSM",
        "tests/test_FSM_env.py",
        success={
            "event_type": "test_run",
            "run_test_success": True,
            "tests_total": 0,
            "tests_failed": 0,
            "tests_passed_all": True,
            "failed_cases_top": [],
            "error_top": [],
        },
    )
    source = {
        "pack_type": "ucagent_context_reuse_v0",
        "trace_count": 3,
        "dut_list": ["Adder", "FSM", "uart_tx"],
        "effective_items": [adder, adder_duplicate, uart, invalid],
        "compression_hints": [],
        "stage_summaries": [],
    }

    curated = curate_pack(source, min_quality=0.68)

    assert curated["curation"]["validated_episode_count"] == 3
    assert curated["curation"]["deduplicated_episode_count"] == 2
    assert curated["curation"]["strategy_count"] == 1
    assert curated["curation"]["drop_reason_counts"]["test_success_is_no_tests_collected"] == 1
    assert curated["curation"]["drop_reason_counts"]["near_duplicate_episode"] == 1

    strategy = curated["effective_items"][0]
    assert strategy["support_count"] == 2
    assert strategy["source_duts"] == ["Adder", "uart_tx"]
    assert strategy["quality_tier"] == "high"
    assert strategy["strategy_id"].startswith("strategy-")
    assert curated["utility_policy"]["policy_version"] == "verifier_grounded_utility_v0"
    assert strategy["verifier_utility"]["gate_score"] > 0


def test_curator_accepts_role_typed_partial_progress():
    item = make_episode("Adder", "tests/test_Adder_env.py")
    item.update({
        "credit_role": "progress",
        "reuse_eligible": True,
        "confidence": 0.8,
        "validation_scope_match": True,
        "stage_advanced": False,
        "regression_count": 0,
        "failure_before": item["failure"],
        "action_sequence": [*item["actions"], {"action_type": "validation", "tool": "RunTestCases"}],
        "observation_after": {
            "event_type": "test_run",
            "test_outcome": "test_failure",
            "tests_total": 3,
            "tests_failed": 1,
            "failed_cases_top": ["test_b"],
        },
        "failure_set_delta": {
            "net_reduction": 1,
            "regression_items": [],
        },
    })
    source = {
        "pack_type": "ucagent_context_reuse_credit_v2",
        "credit_schema": "verifier_grounded_role_typed_v2",
        "trace_count": 1,
        "dut_list": ["Adder"],
        "effective_items": [item],
        "compression_hints": [],
        "stage_summaries": [],
    }

    curated = curate_pack(source, min_quality=0.68)

    assert curated["curation"]["validated_episode_count"] == 1
    assert curated["effective_items"][0]["quality"]["success_reason"] == "verified_partial_failure_reduction"
    assert curated["effective_items"][0]["verifier_utility"]["gate_score"] > 0


def test_verifier_utility_rewards_progress_and_penalizes_regression_and_cost():
    progress = make_episode("Adder", "tests/test_Adder_env.py")
    progress.update({
        "stage_advanced": True,
        "failure_set_delta": {"net_reduction": 4},
        "regression_count": 0,
        "information_gain": {"score": 0.5},
        "action_cost": {"file_mutations": 1, "changed_lines_proxy": 10, "elapsed_ms": 1000},
    })
    regression = copy.deepcopy(progress)
    regression.update({
        "stage_advanced": False,
        "failure_set_delta": {"net_reduction": 0},
        "regression_count": 2,
        "information_gain": {"score": 0.0},
        "action_cost": {"file_mutations": 4, "changed_lines_proxy": 200, "elapsed_ms": 600000},
    })

    progress_utility = episode_verifier_utility(progress)
    regression_utility = episode_verifier_utility(regression)
    aggregate = aggregate_strategy_utility([progress])

    assert progress_utility["score"] > 0
    assert regression_utility["score"] < 0
    assert aggregate["gate_score"] < progress_utility["score"]
    assert aggregate["uncertainty_penalty"] > 0
