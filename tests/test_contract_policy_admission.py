from __future__ import annotations

import json

from scripts.evaluate_contract_policy_admission import (
    evaluate_admission,
    update_registry,
)


def _pair(index: int, utility: float = 0.2, **changes) -> dict:
    row = {
        "pair_id": f"p{index}",
        "quality_noninferior": True,
        "regression_delta": 0,
        "invalid_test_delta": 0,
        "control_status": "completed",
        "treatment_status": "completed",
        "utility_score": utility,
        "winner": "treatment",
        "comparison_valid": True,
    }
    row.update(changes)
    return row


def test_admission_rejects_quality_regression_even_when_faster():
    pairs = [_pair(index) for index in range(8)]
    pairs.append(_pair(8, utility=0.9, quality_noninferior=False, regression_delta=1))

    result = evaluate_admission(
        pairs,
        policy_id="contract-v1",
        min_pairs=3,
        max_harmful_rate=0.5,
        bootstrap_samples=500,
    )

    assert result["decision"] == "reject"
    assert result["reason"] == "quality_constraint_violated"


def test_admission_holds_when_harm_bound_is_too_wide():
    result = evaluate_admission(
        [_pair(index) for index in range(9)],
        policy_id="contract-v1",
        min_pairs=9,
        max_harmful_rate=0.10,
        bootstrap_samples=500,
    )

    assert result["decision"] == "hold"
    assert "harm_upper_bound_too_wide" in result["reason"]
    assert result["evidence"]["utility_bootstrap_95"][0] > 0


def test_admission_accepts_sufficient_consistently_positive_pairs():
    result = evaluate_admission(
        [_pair(index, utility=0.1 + index / 1000) for index in range(50)],
        policy_id="contract-v1",
        min_pairs=20,
        max_harmful_rate=0.10,
        bootstrap_samples=500,
    )

    assert result["decision"] == "admit"
    assert result["evidence"]["harmful_rate_wilson_95"][1] < 0.10
    assert result["evidence"]["utility_bootstrap_95"][0] > 0


def test_registry_quarantines_rejected_candidate_without_promoting_champion(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"schema_version": 1, "champion": "old", "candidates": {}}))
    admission = {
        "policy_id": "bad",
        "decision": "reject",
        "reason": "quality_constraint_violated",
    }

    update_registry(path, admission)
    registry = json.loads(path.read_text())

    assert registry["champion"] == "old"
    assert registry["candidates"]["bad"]["state"] == "quarantined"


def test_admission_holds_incomplete_pair_without_using_censored_utility():
    pair = _pair(
        1,
        utility=None,
        comparison_valid=False,
        control_status="failed",
        treatment_status="completed",
        winner="treatment_completion",
    )

    result = evaluate_admission(
        [pair],
        policy_id="contract-v1",
        min_pairs=1,
        bootstrap_samples=100,
    )

    assert result["decision"] == "hold"
    assert result["evidence"]["valid_pairs"] == 0
    assert result["evidence"]["invalid_pairs"] == 1
    assert result["evidence"]["mean_utility"] is None
