from __future__ import annotations

from scripts.audit_contract_shadow import (
    balanced_sample,
    blind_record,
    collect_candidates,
    evaluate_annotations,
    key_record,
    merge_annotation_sets,
)
from ucagent.memory.trajectory_contract import build_transition_contract


def _source_episode(strategy_id: str = "s1") -> dict:
    episode = {
        "strategy_id": strategy_id,
        "episode_id": "e1",
        "dut": "uart_tx",
        "stage_index": 21,
        "stage_name": "test_case_implementation_in_batch",
        "failure_pattern": "bug_doc_schema_or_marking",
        "failure_group": "bug_document",
        "action_categories": ["bug_document"],
        "failure_before": {
            "event_type": "check_result",
            "failure_pattern": "bug_doc_schema_or_marking",
            "failure_group": "bug_document",
            "failed_checkpoints_top": ["FG-A/FC-B/CK-C"],
        },
        "action_sequence": [
            {"operation": "write", "path_category": "bug_document", "path": "uart_tx_bug.md"}
        ],
        "observation_after": {"event_type": "check_result", "check_pass": True},
        "stage_advanced": True,
    }
    episode["transition_contract"] = build_transition_contract(episode)
    return episode


def test_collect_candidates_backfills_contract_and_keeps_prediction_separate():
    source = _source_episode()
    analysis = {
        "records": [{
            "node_id": "d1",
            "run_id": "r1",
            "stage_index": 21,
            "stage_name": "test_case_implementation_in_batch",
            "failure": {
                "pattern": "bug_doc_schema_or_marking",
                "group": "bug_document",
                "failed_checkpoints_top": ["FG-A/FC-B/CK-C"],
            },
            "strategies": [{"strategy_id": "s1", "source_dut": "uart_tx"}],
            "credit_role": "progress",
            "contract_outcomes": [{"strategy_id": "s1", "status": "fulfilled"}],
        }]
    }

    rows = collect_candidates(analysis, {"s1": source})

    assert len(rows) == 1
    assert rows[0]["prediction"]["label"] == "applicable"
    assert rows[0]["observation"]["contract_status"] == "fulfilled"
    blinded = blind_record(rows[0])
    assert "prediction" not in blinded
    assert "observation" not in blinded
    assert key_record(rows[0])["prediction"]["label"] == "applicable"


def test_balanced_sample_round_robins_prediction_outcome_strata():
    candidates = []
    for stratum in ("applicable:fulfilled", "inapplicable:regression"):
        for index in range(4):
            candidates.append({
                "audit_id": f"{stratum}-{index}",
                "stratum": stratum,
                "candidate": {"strategy_id": f"s-{index}"},
            })

    sample = balanced_sample(candidates, 6, seed=7)

    counts = {}
    for item in sample:
        counts[item["stratum"]] = counts.get(item["stratum"], 0) + 1
    assert counts == {"applicable:fulfilled": 3, "inapplicable:regression": 3}


def test_evaluate_annotations_reports_precision_and_double_label_kappa():
    annotations = [
        {
            "audit_id": "a1",
            "annotations": [
                {"applicability": "applicable"},
                {"applicability": "applicable"},
            ],
        },
        {
            "audit_id": "a2",
            "annotations": [
                {"applicability": "inapplicable"},
                {"applicability": "inapplicable"},
            ],
        },
        {"audit_id": "a3", "annotations": [{"applicability": "inapplicable"}]},
    ]
    keys = [
        {"audit_id": "a1", "prediction": {"label": "applicable"}},
        {"audit_id": "a2", "prediction": {"label": "inapplicable"}},
        {"audit_id": "a3", "prediction": {"label": "applicable"}},
    ]

    result = evaluate_annotations(annotations, keys)

    assert result["records_binary_labeled"] == 3
    assert result["applicable_precision"] == 0.5
    assert result["applicable_recall"] == 1.0
    assert result["double_labeled_records"] == 2
    assert result["cohen_kappa"] == 1.0


def test_merge_annotation_sets_combines_independent_blind_labels():
    base = [{"audit_id": "a1", "annotations": [{
        "annotator": "agent-a", "applicability": "applicable", "rationale": "match"
    }]}]
    other = [{"audit_id": "a1", "annotations": [{
        "annotator": "agent-b", "applicability": "inapplicable", "rationale": "scope"
    }]}]

    merged = merge_annotation_sets([base, other])

    assert len(merged) == 1
    assert len(merged[0]["annotations"]) == 2
