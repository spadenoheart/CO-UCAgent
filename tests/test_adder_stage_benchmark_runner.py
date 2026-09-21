import csv
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_adder_stage_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("run_adder_stage_benchmarks", SCRIPT)
assert SPEC and SPEC.loader
stage_runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage_runner)


def test_resolve_python_accepts_explicit_virtualenv_interpreter():
    assert stage_runner.resolve_python(Namespace(python=sys.executable, env="missing")) == str(
        Path(sys.executable).resolve()
    )


def test_select_boundary_matches_successful_stage_transition():
    transitions = {
        21: {
            "time_unix": 100.2,
            "timestamp": "2026-08-12T15:00:22",
            "to_stage": {"stage_index": 21, "stage_name": "api_test", "stage_title": "API test"},
            "from_stage": {"stage_index": 20},
        }
    }
    commits = [
        {"commit": "before", "time_unix": 90.0, "subject": "old"},
        {"commit": "boundary", "time_unix": 100.0, "subject": "stage 20"},
    ]

    result = stage_runner.select_boundary(21, transitions, commits)

    assert result["commit"] == "boundary"
    assert result["stage_name"] == "api_test"
    assert result["alignment_delta_seconds"] < 1


def test_select_boundary_rejects_stage_zero_without_pre_stage_snapshot():
    try:
        stage_runner.select_boundary(
            0,
            {},
            [{"commit": "already-completed-stage-zero", "time_unix": 100.0}],
        )
    except ValueError as exc:
        assert "leak stage-0 outputs" in str(exc)
    else:
        raise AssertionError("stage 0 must not use a post-stage commit as its entry checkpoint")


def test_sanitize_stage_state_removes_future_progress_and_uses_snapshot_report(tmp_path):
    output = tmp_path / "unity_test"
    output.mkdir()
    report = {"source_ck_list": ["CK-1"], "refine_result": {}}
    (output / ".TEST_TEMPLATE_IMP_REPORT.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    final_state = {
        "stage_index": 30,
        "all_completed": True,
        "is_agent_exit": True,
        "stages_info": {
            "20": {"is_completed": True},
            "21": {"is_completed": True},
            "22": {"is_completed": True},
            "23": {"is_completed": True},
        },
        "stage_data": {
            "COVER_GROUP_DOC_CK_LIST": ["CK-1"],
            "_BASIC_API_PASSED_TEST_NODEIDS": ["test_ok"],
            "TEST_TEMPLATE_IMP_REPORT": {"future": True},
            "_TC_REFINE_RESULT": {"future": True},
            "_RANDOM_TEST_CASES_RESULT": {"future": True},
        },
    }

    state = stage_runner.sanitize_stage_state(final_state, 23, tmp_path)

    assert state["stage_index"] == 23
    assert state["all_completed"] is False
    assert set(state["stages_info"]) == {"20", "21", "22"}
    assert state["stage_data"]["TEST_TEMPLATE_IMP_REPORT"] == report
    assert "_TC_REFINE_RESULT" not in state["stage_data"]
    assert "_RANDOM_TEST_CASES_RESULT" not in state["stage_data"]


def test_copy_memory_at_boundary_filters_future_rows_and_embeddings(tmp_path):
    source = tmp_path / "source" / ".ucagent_memory" / "Adder"
    source.mkdir(parents=True)
    rows = [
        {"hash": "old", "meta": {"timestamp": 10.0}},
        {"hash": "future", "meta": {"timestamp": 20.0}},
    ]
    for name in ("memory.candidate.jsonl", "memory.jsonl"):
        (source / name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    (source / "memory.emb.jsonl").write_text(
        json.dumps({"hash": "old", "vec": [1]})
        + "\n"
        + json.dumps({"hash": "future", "vec": [2]})
        + "\n",
        encoding="utf-8",
    )
    (source / "metrics.jsonl").write_text(
        json.dumps({"ts": 10.0, "event": "old"})
        + "\n"
        + json.dumps({"ts": 20.0, "event": "future"})
        + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "destination"

    count = stage_runner.copy_memory_at_boundary(tmp_path / "source", destination, 15.0)

    assert count == 1
    memory = (destination / ".ucagent_memory" / "Adder" / "memory.jsonl").read_text()
    embeddings = (destination / ".ucagent_memory" / "Adder" / "memory.emb.jsonl").read_text()
    metrics = (destination / ".ucagent_memory" / "Adder" / "metrics.jsonl").read_text()
    assert "old" in memory and "future" not in memory
    assert "old" in embeddings and "future" not in embeddings
    assert "old" in metrics and "future" not in metrics


def test_target_stage_completed_requires_saved_completion_and_advance(tmp_path):
    state_dir = tmp_path / ".ucagent"
    state_dir.mkdir()
    state_path = state_dir / "ucagent_info.json"
    state_path.write_text(
        json.dumps({"stage_index": 21, "stages_info": {"21": {"is_completed": True}}}),
        encoding="utf-8",
    )
    assert stage_runner.target_stage_completed(tmp_path, 21) is False
    state_path.write_text(
        json.dumps({"stage_index": 22, "stages_info": {"21": {"is_completed": True}}}),
        encoding="utf-8",
    )
    assert stage_runner.target_stage_completed(tmp_path, 21) is True


def test_parse_stage_metrics_counts_only_target_stage(tmp_path):
    (tmp_path / "workspace" / "unity_test").mkdir(parents=True)
    (tmp_path / "ucagent-log.log").write_text(
        "2026-08-12 12:00:00,000 INFO [data_collection][llm_request] stage=21 prompt_tokens=100 completion_tokens=10\n"
        "2026-08-12 12:01:00,000 INFO [data_collection][llm_request] stage=22 prompt_tokens=999 completion_tokens=99\n",
        encoding="utf-8",
    )
    events = [
        {"event_type": "check_result", "stage_index": 21, "success": False},
        {
            "event_type": "test_run",
            "stage_index": 21,
            "result": {"test_outcome": "test_failure"},
        },
        {
            "event_type": "context_reuse_decision",
            "stage_index": 21,
            "decision": "inject",
            "search_summary": {
                "contract_candidates": 3,
                "contract_applicable": 2,
                "contract_inapplicable": 1,
                "contract_enforced_rejected": 1,
            },
        },
        {
            "event_type": "progress_control_decision",
            "stage_index": 21,
            "credit_role": "no_progress",
            "policy": "pivot",
        },
        {
            "event_type": "progress_control_decision",
            "stage_index": 21,
            "credit_role": "progress",
            "policy": "verify_stage",
        },
        {"event_type": "mutation_budget_blocked", "stage_index": 21},
        {"event_type": "stage_state_package_update", "stage_index": 21},
        {
            "event_type": "observation_masking",
            "stage_index": 21,
            "masked_observations": 3,
            "saved_chars": 1200,
        },
        {"event_type": "check_result", "stage_index": 22, "success": False},
    ]
    (tmp_path / "workspace" / "unity_test" / "structured_events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
    )

    result = stage_runner.parse_stage_metrics(tmp_path, 21)

    assert result["prompt_tokens"] == 100
    assert result["completion_tokens"] == 10
    assert result["llm_requests"] == 1
    assert result["failed_checks"] == 1
    assert result["passed_checks"] == 0
    assert result["test_runs"] == 1
    assert result["context_injections"] == 1
    assert result["contract_candidates"] == 3
    assert result["contract_applicable"] == 2
    assert result["contract_inapplicable"] == 1
    assert result["contract_enforced_rejected"] == 1
    assert result["progress_decisions"] == 2
    assert result["progress_roles"] == {"no_progress": 1, "progress": 1}
    assert result["progress_policies"] == {"pivot": 1, "verify_stage": 1}
    assert result["pivot_decisions"] == 1
    assert result["verify_stage_decisions"] == 1
    assert result["mutation_budget_blocks"] == 1
    assert result["stage_state_updates"] == 1
    assert result["observation_masking_events"] == 1
    assert result["max_masked_observations"] == 3
    assert result["max_masked_chars_saved"] == 1200


def test_append_result_csv_is_idempotent(tmp_path):
    path = tmp_path / "stage_results.csv"
    row = {field: "" for field in stage_runner.RESULT_FIELDS}
    row.update({"run_id": "candidate__stage_21", "status": "completed"})

    stage_runner.append_result_csv(path, row)
    updated = dict(row, status="timeout")
    stage_runner.append_result_csv(path, updated)

    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["run_id"] == "candidate__stage_21"
    assert rows[0]["status"] == "timeout"


def test_compare_stage_pair_applies_quality_gate_before_efficiency():
    base = {
        "run_id": "control",
        "stage_index": 21,
        "seed": 7,
        "status": "completed",
        "duration_seconds": 100,
        "prompt_tokens": 900,
        "completion_tokens": 100,
        "progress_roles": {"regression": 0},
        "test_outcomes": {"pass": 2},
        "checkpoint_sha256": "same",
    }
    faster_but_harmful = {
        **base,
        "run_id": "treatment",
        "duration_seconds": 50,
        "prompt_tokens": 400,
        "progress_roles": {"regression": 1},
    }

    comparison = stage_runner.compare_stage_pair(
        base,
        faster_but_harmful,
        pair_label="contract-v1",
        control_arm="off",
        treatment_arm="enforce",
    )

    assert comparison["quality_noninferior"] is False
    assert comparison["comparison_valid"] is True
    assert comparison["winner"] == "control_quality"
    assert comparison["utility_score"] == -1.0


def test_compare_stage_pair_reports_matched_efficiency_gain():
    control = {
        "run_id": "control",
        "stage_index": 23,
        "seed": 9,
        "status": "completed",
        "duration_seconds": 100,
        "prompt_tokens": 900,
        "completion_tokens": 100,
        "progress_roles": {},
        "test_outcomes": {"pass": 1},
        "checkpoint_sha256": "same",
    }
    treatment = {
        **control,
        "run_id": "treatment",
        "duration_seconds": 80,
        "prompt_tokens": 700,
        "completion_tokens": 100,
    }

    comparison = stage_runner.compare_stage_pair(
        control,
        treatment,
        pair_label="contract-v1",
        control_arm="off",
        treatment_arm="enforce",
    )

    assert comparison["quality_noninferior"] is True
    assert comparison["comparison_valid"] is True
    assert comparison["time_reduction_ratio"] == 0.2
    assert comparison["token_reduction_ratio"] == 0.2
    assert comparison["utility_score"] == 0.2
    assert comparison["winner"] == "treatment"


def test_compare_stage_pair_does_not_report_speedup_against_crashed_control():
    control = {
        "run_id": "control",
        "stage_index": 23,
        "seed": 9,
        "status": "failed",
        "duration_seconds": 100,
        "prompt_tokens": 900,
        "completion_tokens": 100,
        "progress_roles": {},
        "test_outcomes": {"infrastructure_error": 1},
        "checkpoint_sha256": "same",
    }
    treatment = {
        **control,
        "run_id": "treatment",
        "status": "completed",
        "duration_seconds": 80,
        "prompt_tokens": 700,
        "completion_tokens": 100,
        "test_outcomes": {"pass": 1},
    }

    comparison = stage_runner.compare_stage_pair(
        control,
        treatment,
        pair_label="contract-v1",
        control_arm="off",
        treatment_arm="enforce",
    )

    assert comparison["comparison_valid"] is False
    assert comparison["comparison_invalid_reason"] == "control_not_completed"
    assert comparison["winner"] == "treatment_completion"
    assert comparison["time_reduction_ratio"] is None
    assert comparison["token_reduction_ratio"] is None
    assert comparison["utility_score"] is None


def test_append_pair_result_csv_is_idempotent(tmp_path):
    path = tmp_path / "paired_results.csv"
    row = {field: "" for field in stage_runner.PAIR_RESULT_FIELDS}
    row.update({"pair_id": "p1", "winner": "control"})

    stage_runner.append_pair_result_csv(path, row)
    stage_runner.append_pair_result_csv(path, dict(row, winner="treatment"))

    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["winner"] == "treatment"
