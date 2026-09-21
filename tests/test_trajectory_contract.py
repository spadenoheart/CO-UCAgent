from __future__ import annotations

import json

from ucagent.memory.context_reuse import ContextReuseStore
from ucagent.memory.trajectory_contract import (
    build_transition_contract,
    contract_applicability,
    evaluate_transition_contract,
)


def _episode() -> dict:
    return {
        "stage_name": "test_case_implementation_in_batch",
        "failure_pattern": "test_logic_or_case_implementation",
        "failure_group": "test_logic",
        "credit_role": "progress",
        "action_categories": ["test_code"],
        "preferred_action_categories": ["test_code"],
        "failure_before": {
            "event_type": "test_run",
            "test_outcome": "test_failure",
            "failure_pattern": "test_logic_or_case_implementation",
            "failure_group": "test_logic",
            "failure_signature": "sig-before",
            "tests_total": 2,
            "failed_cases_top": ["tests/test_uart.py::test_busy", "tests/test_uart.py::test_stop"],
        },
        "actions": [{
            "operation": "replace_string",
            "path": "tests/test_uart.py",
            "path_category": "test_code",
        }],
        "observation_after": {
            "event_type": "test_run",
            "test_outcome": "test_failure",
            "tests_total": 2,
            "failed_cases_top": ["tests/test_uart.py::test_stop"],
        },
        "failure_set_delta": {"net_reduction": 1},
        "regression_count": 0,
        "confidence": 0.8,
    }


def test_build_contract_captures_precondition_action_and_expected_transition():
    contract = build_transition_contract(_episode())

    assert contract["evidence_type"] == "observational_verifier_transition"
    assert contract["preconditions"]["failure_pattern"] == "test_logic_or_case_implementation"
    assert contract["action_contract"]["action_categories"] == ["test_code"]
    assert contract["expected_transition"]["min_failure_net_reduction"] == 1
    assert contract["safety_constraints"]["max_regression_count"] == 0


def test_contract_applicability_requires_failure_and_action_compatibility():
    contract = build_transition_contract(_episode())
    compatible = contract_applicability(contract, {
        "failure_pattern": "test_logic_or_case_implementation",
        "failure_group": "test_logic",
        "failure_signature": "new-signature",
        "failed_cases_top": ["tests/test_adder.py::test_overflow"],
        "preferred_action_categories": ["test_code"],
    }, stage_name="test_case_implementation_in_batch")
    incompatible = contract_applicability(contract, {
        "failure_pattern": "bug_doc_schema_or_marking",
        "failure_group": "bug_document",
        "failed_checkpoints_top": ["FG-A/FC-B/CK-C"],
        "preferred_action_categories": ["bug_document"],
    }, stage_name="test_case_implementation_in_batch")

    assert compatible["applicable"] is True
    assert incompatible["applicable"] is False
    assert "failure_pattern_mismatch" in incompatible["hard_mismatches"]
    assert "action_category_mismatch" in incompatible["hard_mismatches"]


def test_contract_evaluation_distinguishes_progress_regression_and_invalid():
    episode = _episode()
    contract = build_transition_contract(episode)
    before = episode["failure_before"]
    progress = evaluate_transition_contract(contract, before, episode["observation_after"])
    regression = evaluate_transition_contract(contract, before, {
        "event_type": "test_run",
        "test_outcome": "test_failure",
        "tests_total": 3,
        "failed_cases_top": [
            "tests/test_uart.py::test_busy",
            "tests/test_uart.py::test_stop",
            "tests/test_uart.py::test_new_regression",
        ],
    })
    invalid = evaluate_transition_contract(contract, before, {
        "event_type": "test_run",
        "test_outcome": "no_tests_collected",
        "tests_total": 0,
    })

    assert progress["status"] == "fulfilled"
    assert progress["failure_set_delta"]["net_reduction"] == 1
    assert regression["status"] == "regression"
    assert regression["failure_set_delta"]["regression_count"] == 1
    assert invalid["status"] == "invalid"


def test_stage_advance_can_fulfill_contract_without_direct_failure_reduction():
    episode = _episode()
    contract = build_transition_contract(episode)
    outcome = evaluate_transition_contract(
        contract,
        episode["failure_before"],
        {
            "event_type": "test_run",
            "test_outcome": "test_failure",
            "tests_total": 2,
            "failed_cases_top": episode["failure_before"]["failed_cases_top"],
        },
        stage_advanced=True,
    )

    assert outcome["status"] == "fulfilled"
    assert outcome["stage_advanced"] is True


def test_scope_expansion_does_not_treat_newly_observed_failures_as_regressions():
    episode = _episode()
    contract = build_transition_contract(episode)
    outcome = evaluate_transition_contract(
        contract,
        episode["failure_before"],
        {
            "event_type": "test_run",
            "test_outcome": "test_failure",
            "tests_total": 3,
            "failed_cases_top": [
                "tests/test_uart.py::test_stop",
                "tests/test_uart.py::test_newly_observed",
            ],
        },
        scope_relation="expanded",
    )

    assert outcome["status"] == "fulfilled"
    assert outcome["failure_set_delta"]["regression_count"] == 0
    assert outcome["failure_set_delta"]["scope_relation"] == "expanded"


def test_shadow_contract_policy_records_advice_without_filtering_or_prompt_change(tmp_path):
    episode = _episode()
    episode.update({"quality_score": 0.9, "support_count": 2})
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(json.dumps({
        "effective_items": [episode],
        "compression_hints": [],
    }), encoding="utf-8")
    store = ContextReuseStore(str(tmp_path), "Adder", {
        "pack_path": str(pack_path),
        "contract_policy_mode": "shadow",
        "min_score": -1.0,
        "strict_action_category_match": False,
        "strict_exact_pattern_groups": False,
        "strict_failure_pattern_match": False,
        "prefer_action_category_match": False,
    })
    failure = {
        "failure_pattern": "bug_doc_schema_or_marking",
        "failure_group": "bug_document",
        "failed_checkpoints_top": ["FG-A/FC-B/CK-C"],
        "preferred_action_categories": ["bug_document"],
    }

    results = store.search(
        failure,
        stage_index=21,
        stage_name="test_case_implementation_in_batch",
        limit=1,
    )

    assert len(results) == 1
    advice = results[0]["__context_reuse__"]["contract_policy"]
    assert advice["mode"] == "shadow"
    assert advice["applicable"] is False
    assert "transition_contract" not in store.render_episode(results[0])
