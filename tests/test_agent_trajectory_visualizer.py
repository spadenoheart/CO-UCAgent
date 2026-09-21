from __future__ import annotations

import json
from pathlib import Path

import scripts.visualize_agent_trajectory as trajectory_visualizer
from scripts.visualize_agent_trajectory import (
    PayloadCache,
    build_metrics,
    build_stage_metrics,
    build_payload,
    build_trace,
    classify_area,
    discover_sources,
    parse_baseline_token_actions,
    payload_index,
    render_html,
    source_from_event_path,
    structured_actions,
    trace_transport,
    tool_verdict,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _event(event_type: str, time_unix: float, **values) -> dict:
    return {
        "event_type": event_type,
        "time_unix": time_unix,
        "timestamp": "2026-08-21T10:00:00",
        "dut": "Adder",
        "stage_index": 21,
        "stage_name": "test_case_implementation_in_batch",
        "stage_title": "test implementation",
        **values,
    }


def _make_run(tmp_path: Path, name: str = "run-a") -> Path:
    run_dir = tmp_path / name
    event_path = run_dir / "workspace" / "unity_test" / "structured_events.jsonl"
    failure = {
        "test_outcome": "test_failure",
        "test_outcome_reason": "pytest_failures",
        "tests_total": 2,
        "tests_failed": 1,
        "tests_passed_all": False,
        "failed_cases_top": ["tests/test_Adder.py::test_overflow"],
        "error_top": ["assert 0 == 1"],
    }
    rows = [
        _event(
            "file_mutation",
            1_787_300_020,
            success=True,
            tool="EditTextFile",
            operation="write",
            path="unity_test/tests/test_Adder.py",
        ),
        _event(
            "test_run",
            1_787_300_030,
            success=False,
            tool="RunTestCases",
            target="test_Adder.py::test_overflow",
            result=failure,
        ),
        _event(
            "test_run",
            1_787_300_040,
            success=False,
            tool="RunTestCases",
            target="test_Adder.py::test_overflow",
            result=failure,
        ),
        _event(
            "check_result",
            1_787_300_050,
            success=True,
            tool="Complete",
            result={"check_pass": True, "checker_categories": []},
        ),
    ]
    _write_jsonl(event_path, rows)
    (run_dir / "ucagent-log.log").write_text(
        "2026-08-21 10:00:45,000 - INFO - [data_collection][llm_request] "
        "role=main model=test-model stage=21 status=success stream=true "
        "latency_ms=2000 ttft_ms=500 prompt_tokens=1000 completion_tokens=50\n",
        encoding="utf-8",
    )
    dump_dir = event_path.parent / "llm_input_dumps"
    dump_dir.mkdir()
    (dump_dir / "req_000001_stage_21.json").write_text(
        json.dumps(
            {
                "created_at": "2026-08-21T09:59:50",
                "stage": 21,
                "messages": [
                    {
                        "type": "ai",
                        "content": "先读取 API 契约。",
                        "tool_calls": [
                            {
                                "id": "call-read-1",
                                "name": "ReadTextFile",
                                "args": {"path": "unity_test/tests/Adder_api.py"},
                            }
                        ],
                    },
                    {
                        "type": "tool",
                        "tool_call_id": "call-read-1",
                        "name": "ReadTextFile",
                        "status": "success",
                        "content": "def add(a, b): ...",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return run_dir


def test_area_classifier_prioritizes_specialized_test_artifacts() -> None:
    assert classify_area(path="unity_test/tests/Adder_api.py") == "Env/API"
    assert classify_area(path="unity_test/tests/Adder_function_coverage.py") == "覆盖率模型"
    assert classify_area(path="unity_test/tests/test_Adder_random.py") == "随机测试"
    assert classify_area(path="unity_test/tests/test_Adder.py") == "定向测试"
    assert classify_area(path="Adder/Adder.v") == "RTL/DUT"
    assert classify_area(path="unity_test/Adder_bug_analysis.md") == "Bug分析"


def test_zero_test_invocations_are_never_visualized_as_passes() -> None:
    explicit_invalid = _event(
        "test_run",
        10,
        success=True,
        result={
            "test_outcome": "no_tests_collected",
            "tests_total": 0,
            "tests_failed": 0,
            "tests_passed_all": True,
        },
    )
    legacy_invalid = _event(
        "test_run",
        20,
        success=True,
        result={"tests_total": 0, "tests_failed": 0, "tests_passed_all": True},
    )
    valid = _event(
        "test_run",
        30,
        success=True,
        result={"test_outcome": "pass", "tests_total": 1, "tests_failed": 0},
    )

    actions, _, _ = structured_actions([explicit_invalid, legacy_invalid, valid])

    assert [action["verdict"] for action in actions] == ["failure", "failure", "success"]
    assert "no_tests_collected" in actions[0]["label"]


def test_successful_read_of_error_text_is_not_a_tool_failure() -> None:
    assert tool_verdict("ReadTextFile", "success", "ERROR: expected DUT evidence") == "success"
    assert tool_verdict("ReadTextFile", "error", "No such file") == "failure"
    assert tool_verdict("SearchText", "success", "No matches found") == "empty"


def test_trace_combines_events_tool_history_llm_metrics_and_retries(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    source = source_from_event_path(
        run_dir / "workspace" / "unity_test" / "structured_events.jsonl",
        "candidate",
    )

    trace = build_trace(source)

    assert trace["label"] == "candidate"
    assert trace["tool_history_recovery"]["tool_calls_recovered"] == 1
    assert trace["metrics"]["reads"] == 1
    assert trace["metrics"]["mutations"] == 1
    assert trace["metrics"]["blind_retries"] == 1
    assert trace["metrics"]["failures"] == 2
    assert trace["metrics"]["prompt_tokens"] == 1000
    assert trace["metrics"]["median_ttft_ms"] == 500
    assert trace["failure_loops"][0]["count"] == 2
    assert all(action["id"].startswith(trace["trace_id"] + "-") for action in trace["actions"])


def test_discovery_and_html_support_multiple_runs_without_node_id_collisions(tmp_path: Path) -> None:
    run_a = _make_run(tmp_path, "run-a")
    run_b = _make_run(tmp_path, "run-b")
    sources = discover_sources([f"baseline={run_a}", f"candidate={run_b}"])

    payload = build_payload(sources, title="comparison")
    document = render_html(payload)

    assert [trace["label"] for trace in payload["traces"]] == ["baseline", "candidate"]
    assert payload["traces"][0]["trace_id"] != payload["traces"][1]["trace_id"]
    assert "<svg class=\"trajectory\"" in document
    assert 'id="run-filter"' in document
    assert "selectedTraces()" in document
    assert "UCAgent Trajectory Atlas" in document
    assert "累计输入 Token" in document
    assert "最近完成 Prompt" in document
    assert "不代表当前单次 Prompt" in document
    assert "__TRACE_DATA__" not in document


def test_metrics_tolerate_actions_without_timestamps() -> None:
    metrics = build_metrics(
        [
            {
                "kind": "read",
                "verdict": "success",
                "area": "需求与规范",
                "time_unix": 0,
                "detail": {},
            }
        ],
        [],
    )

    assert metrics["duration_seconds"] == 0
    assert metrics["observable_actions"] == 1


def test_stage_metrics_exclude_offline_resume_gap() -> None:
    actions = [
        {"stage_index": 23, "stage_name": "implementation", "time_unix": 100.0, "kind": "test", "verdict": "success"},
        {"stage_index": 23, "stage_name": "implementation", "time_unix": 1300.0, "kind": "llm", "verdict": "success"},
        {"stage_index": 23, "stage_name": "implementation", "time_unix": 68_500.0, "kind": "llm", "verdict": "success"},
        {"stage_index": 23, "stage_name": "implementation", "time_unix": 69_100.0, "kind": "checker", "verdict": "success"},
    ]

    stage = build_stage_metrics(actions)[0]

    assert stage["duration_seconds"] == 1800
    assert stage["elapsed_span_seconds"] == 69000


def test_baseline_token_rows_distinguish_main_and_summary_models() -> None:
    rows = [
        {
            "timestamp": "2026-09-04T01:00:00+00:00",
            "status": "success",
            "model_instance": 1,
            "prompt_tokens": 1767,
            "completion_tokens": 20,
            "latency_ms": 4000,
        },
        {
            "timestamp": "2026-09-04T01:01:00+00:00",
            "status": "success",
            "model_instance": 2,
            "prompt_tokens": 19232,
            "completion_tokens": 425,
            "latency_ms": 98475,
        },
    ]

    actions = parse_baseline_token_actions(rows, [])
    metrics = build_metrics(actions, [])

    assert [action["detail"]["role"] for action in actions] == ["main", "summary"]
    assert metrics["prompt_tokens"] == 20999
    assert metrics["latest_prompt_tokens"] == 19232
    assert metrics["latest_completion_tokens"] == 425
    assert metrics["latest_llm_role"] == "summary"
    assert metrics["latest_llm_latency_ms"] == 98475


def test_live_cache_discovers_runs_created_after_server_start(tmp_path: Path) -> None:
    experiment_root = tmp_path / "future-runs"
    cache = PayloadCache(
        [],
        title="live",
        include_tool_history=False,
        max_detail_chars=1000,
        source_paths=[f"current={experiment_root}"],
        max_traces=8,
    )

    assert cache.get()["trace_count"] == 0

    _make_run(experiment_root, "candidate__seed_1")
    payload = cache.get()

    assert payload["trace_count"] == 1
    assert payload["traces"][0]["label"] == "current"
    assert payload["traces"][0]["current_stage_index"] == 21


def test_legacy_upstream_run_is_discovered_and_reconstructed(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_1" / "FSM__seed_7"
    run_dir.mkdir(parents=True)
    (run_dir / "ucagent-log.log").write_text("upstream log\n", encoding="utf-8")
    (run_dir / "ucagent-msg.log").write_text(
        """================================ Human Message =================================
mission: FSM芯片验证任务
current_stage:
  index: 0
  task:
    title: 1-requirement_analysis_and_planning-需求分析与验证规划
================================== AI Message ==================================
Tool Calls:
  ReadTextFile (call_read)
 Call ID: call_read
  Args:
    path: FSM/README.md
================================= Tool Message =================================
Name: ReadTextFile
[INFO] Read complete.
""",
        encoding="utf-8",
    )
    _write_jsonl(
        run_dir / "baseline_token_usage.jsonl",
        [{
            "event": "llm_request",
            "timestamp": "2026-09-04T02:53:40+00:00",
            "status": "success",
            "model": "test-model",
            "latency_ms": 1000,
            "ttft_ms": 500,
            "prompt_tokens": 900,
            "completion_tokens": 20,
            "token_source": "approx",
        }],
    )

    sources = discover_sources([f"baseline={tmp_path}"])
    trace = build_trace(sources[0])

    assert len(sources) == 1
    assert sources[0].event_path.name == "ucagent-log.log"
    assert trace["telemetry_mode"] == "legacy_log_reconstruction"
    assert trace["experiment_kind"] == "upstream_baseline"
    assert trace["dut"] == "FSM"
    assert trace["current_stage_index"] == 0
    assert trace["metrics"]["reads"] == 1
    assert trace["metrics"]["prompt_tokens"] == 900
    assert trace["legacy_reconstruction"]["tool_calls_recovered"] == 1


def test_legacy_upstream_stage_timeline_assigns_tokens_and_current_stage(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_1" / "FSM__seed_7"
    run_dir.mkdir(parents=True)
    (run_dir / "ucagent-log.log").write_text(
        """2026-09-04 13:48:18,096 - ucagent-log - INFO - Stages:
 0:   1-requirement_analysis_and_planning-需求分析与验证规划
 1:   2-dut_function_understanding-FSM功能理解
2026-09-04 13:48:18,097 - ucagent-log - INFO - Current stage index is 0.
2026-09-04 13:51:31,744 - ucagent-log - INFO - ToolComplete:
complete: true
message: 'Stage 0 completed successfully. Current stage index is now 1.'
""",
        encoding="utf-8",
    )
    (run_dir / "ucagent-msg.log").write_text(
        """================================ Human Message =================================
mission: FSM芯片验证任务
current_stage:
  index: 0
  task:
    title: 1-requirement_analysis_and_planning-需求分析与验证规划
================================== AI Message ==================================
working
""",
        encoding="utf-8",
    )
    _write_jsonl(
        run_dir / "baseline_token_usage.jsonl",
        [
            {
                "event": "llm_request",
                "timestamp": "2026-09-04T13:50:00+08:00",
                "status": "success",
                "prompt_tokens": 100,
                "completion_tokens": 10,
            },
            {
                "event": "llm_request",
                "timestamp": "2026-09-04T13:52:00+08:00",
                "status": "success",
                "prompt_tokens": 200,
                "completion_tokens": 20,
            },
        ],
    )

    trace = build_trace(discover_sources([f"baseline={tmp_path}"])[0])
    llm_actions = [action for action in trace["actions"] if action["kind"] == "llm"]

    assert [action["stage_index"] for action in llm_actions] == [0, 1]
    assert llm_actions[1]["stage_name"] == "dut_function_understanding"
    assert trace["current_stage_index"] == 1
    assert trace["current_stage_name"] == "dut_function_understanding"
    assert trace["legacy_reconstruction"]["stage_transitions_recovered"] == 2


def test_live_cache_can_throttle_repeated_full_discovery(tmp_path: Path, monkeypatch) -> None:
    experiment_root = tmp_path / "runs"
    _make_run(experiment_root, "candidate__seed_1")
    cache = PayloadCache(
        discover_sources([str(experiment_root)]),
        title="live",
        include_tool_history=False,
        max_detail_chars=1000,
        source_paths=[str(experiment_root)],
    )
    first = cache.get()

    def fail_discovery(*_args, **_kwargs):
        raise AssertionError("discovery should be throttled within one refresh interval")

    monkeypatch.setattr(trajectory_visualizer, "discover_sources", fail_discovery)
    second = cache.get(min_refresh_interval=20)

    assert second is first


def test_live_index_and_selected_trace_avoid_bulk_action_details(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    payload = build_payload(discover_sources([str(run_dir)]), title="live")

    index = payload_index(payload)
    selected = trace_transport(payload["traces"][0])

    assert index["trace_count"] == 1
    assert index["traces"][0]["actions"] == []
    assert index["traces"][0]["actions_loaded"] is False
    assert selected["actions_loaded"] is True
    assert selected["actions"]
    assert all("detail" not in action for action in selected["actions"])


def test_explicit_empty_live_root_does_not_fall_back_to_unrelated_runs(tmp_path: Path) -> None:
    empty_root = tmp_path / "empty"
    empty_root.mkdir()

    assert discover_sources([str(empty_root)]) == []


def test_duplicate_live_labels_include_run_id(tmp_path: Path) -> None:
    experiment_root = tmp_path / "runs"
    _make_run(experiment_root, "arm__seed_1")
    _make_run(experiment_root, "arm__seed_2")

    sources = discover_sources([f"current={experiment_root}"])

    assert {source.label for source in sources} == {
        "current/arm__seed_1",
        "current/arm__seed_2",
    }


def test_no_paths_indexes_entire_default_experiment_root(tmp_path: Path, monkeypatch) -> None:
    experiment_root = tmp_path / "benchmark" / "ucagent_experiments"
    _make_run(experiment_root, "arm-a__seed_1")
    _make_run(experiment_root, "arm-b__seed_2")
    monkeypatch.setattr(trajectory_visualizer, "REPO_ROOT", tmp_path)

    sources = trajectory_visualizer.discover_sources([])

    assert {source.run_dir.name for source in sources} == {
        "arm-a__seed_1",
        "arm-b__seed_2",
    }


def test_live_cache_only_rebuilds_new_or_changed_runs(tmp_path: Path, monkeypatch) -> None:
    experiment_root = tmp_path / "runs"
    run_a = _make_run(experiment_root, "arm-a__seed_1")
    _make_run(experiment_root, "arm-b__seed_2")
    ledger_path = experiment_root / "ledger.json"
    ledger_path.write_text(json.dumps({
        "runs": {
            "arm-a__seed_1": {"status": "running"},
            "arm-b__seed_2": {"status": "running"},
        }
    }), encoding="utf-8")
    original_build_trace = trajectory_visualizer.build_trace
    rebuilt: list[str] = []

    def recording_build_trace(source, **kwargs):
        rebuilt.append(source.run_dir.name)
        return original_build_trace(source, **kwargs)

    monkeypatch.setattr(trajectory_visualizer, "build_trace", recording_build_trace)
    cache = trajectory_visualizer.PayloadCache(
        [],
        title="incremental",
        include_tool_history=False,
        max_detail_chars=1000,
        source_paths=[str(experiment_root)],
        max_traces=0,
    )

    first = cache.get()
    assert first["incremental_update"] == {
        "rebuilt_trace_count": 2,
        "reused_trace_count": 0,
    }
    assert len(rebuilt) == 2

    second = cache.get()
    assert second["incremental_update"] == {
        "rebuilt_trace_count": 0,
        "reused_trace_count": 2,
    }
    assert len(rebuilt) == 2

    ledger_path.write_text(json.dumps({
        "runs": {
            "arm-a__seed_1": {"status": "running"},
            "arm-b__seed_2": {"status": "completed"},
        }
    }), encoding="utf-8")
    status_update = cache.get()
    assert status_update["incremental_update"] == {
        "rebuilt_trace_count": 1,
        "reused_trace_count": 1,
    }
    assert rebuilt[-1] == "arm-b__seed_2"

    event_path = run_a / "workspace" / "unity_test" / "structured_events.jsonl"
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_event("stage_transition", 1_787_300_060, success=True)) + "\n")

    third = cache.get()
    assert third["incremental_update"] == {
        "rebuilt_trace_count": 1,
        "reused_trace_count": 1,
    }
    assert rebuilt[-1] == "arm-a__seed_1"
