import csv
import importlib.util
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_adder_experiments.py"
SPEC = importlib.util.spec_from_file_location("run_adder_experiments", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def matrix():
    return {
        "schema_version": 1,
        "seeds": [11, 22, 33],
        "arms": [
            {"id": "important", "repeat": 3, "environment": {}, "overrides": []},
            {"id": "cheap", "repeat": 1, "environment": {}, "overrides": []},
        ],
    }


def test_select_runs_pairs_important_arm_across_three_fixed_seeds():
    runner.validate_matrix(matrix())
    runs = runner.select_runs(matrix(), set(), set())
    assert [item["run_id"] for item in runs] == [
        "important__seed_11",
        "important__seed_22",
        "important__seed_33",
        "cheap__seed_11",
    ]


def test_select_runs_filters_arm_and_seed():
    runs = runner.select_runs(matrix(), {"important"}, {22})
    assert [(item["arm"]["id"], item["seed"]) for item in runs] == [("important", 22)]


def test_select_runs_requires_explicit_opt_in_for_unlisted_seed():
    with pytest.raises(ValueError, match="--allow-unlisted-seed"):
        runner.select_runs(matrix(), {"important"}, {44})

    runs = runner.select_runs(
        matrix(),
        {"important"},
        {55, 44},
        allow_unlisted_seeds=True,
    )
    assert [item["run_id"] for item in runs] == [
        "important__seed_44",
        "important__seed_55",
    ]


def test_explicit_seed_selection_overrides_pilot_repeat():
    runs = runner.select_runs(matrix(), {"cheap"}, {22})
    assert [item["run_id"] for item in runs] == ["cheap__seed_22"]


def test_validate_matrix_rejects_summary_threshold_at_model_limit():
    value = matrix()
    value["fixed_configuration"] = {
        "num_ctx": 65536,
        "summary_max_tokens": 65536,
    }

    with pytest.raises(ValueError, match="below num_ctx"):
        runner.validate_matrix(value)


def test_validate_matrix_accepts_high_watermark_and_hard_cap():
    value = matrix()
    value["fixed_configuration"] = {
        "num_ctx": 65536,
        "summary_max_tokens": 57344,
        "summary_hard_max_tokens": 61440,
        "summary_max_output_tokens": 1024,
        "summary_max_keep_msgs": 100,
        "summary_tail_keep_msgs": 10,
    }

    runner.validate_matrix(value)


def test_validate_matrix_rejects_input_output_budget_above_model_context():
    value = matrix()
    value["fixed_configuration"] = {
        "num_ctx": 65536,
        "summary_max_tokens": 57344,
        "summary_hard_max_tokens": 65000,
        "summary_max_output_tokens": 1024,
    }

    with pytest.raises(ValueError, match="plus summary_max_output_tokens"):
        runner.validate_matrix(value)


def test_environment_for_python_prepends_selected_environment_bin():
    env = runner.environment_for_python(
        "/opt/conda/envs/uc/bin/python",
        {"PATH": "/usr/bin:/opt/conda/envs/uc/bin"},
    )

    assert env["PATH"].split(":") == ["/opt/conda/envs/uc/bin", "/usr/bin"]


def test_build_run_environment_applies_fixed_context_and_isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_TEMPERATURE", "0.7")
    monkeypatch.setenv("SUMMARY_TAIL_KEEP_MSG", "999")
    env = runner.build_run_environment(
        "/opt/conda/envs/uc/bin/python",
        {
            "unset_environment": ["OPENAI_TEMPERATURE"],
            "summary_max_tokens": 57344,
            "summary_hard_max_tokens": 61440,
            "summary_max_output_tokens": 1024,
            "summary_max_keep_msgs": 100,
            "summary_tail_keep_msgs": 10,
        },
        {"environment": {"UC_MODEL_TYPE": "openai"}},
        "model-64k",
        "http://server/v1",
        "key",
        65536,
        tmp_path / "home",
        seed=20260825,
    )

    assert "OPENAI_TEMPERATURE" not in env
    assert env["OPENAI_MODEL"] == "model-64k"
    assert env["OLLAMA_NUM_CTX"] == "65536"
    assert env["SUMMARY_MAX_CTX_TOKEN"] == "57344"
    assert env["SUMMARY_HARD_MAX_CTX_TOKEN"] == "61440"
    assert env["SUMMARY_MAX_SUM_TOKEN"] == "1024"
    assert env["SUMMARY_MAX_KEEP_MSG"] == "100"
    assert env["SUMMARY_TAIL_KEEP_MSG"] == "10"
    assert env["HOME"] == str(tmp_path / "home")
    assert env["PYTHONHASHSEED"] == "20260825"


def test_build_run_environment_overrides_stale_arm_hash_seed():
    env = runner.build_run_environment(
        "/opt/conda/envs/uc/bin/python",
        {},
        {"environment": {"PYTHONHASHSEED": "20260721"}},
        "model-64k",
        "http://server/v1",
        "key",
        65536,
        seed=20260826,
    )

    assert env["PYTHONHASHSEED"] == "20260826"


def test_enable_baseline_token_meter_scopes_sitecustomize_to_agent(tmp_path):
    env = runner.enable_baseline_token_meter(
        {"PYTHONPATH": "/existing"},
        tmp_path / "run",
        tmp_path / "upstream" / "ucagent.py",
    )

    assert env["PYTHONPATH"].split(":")[:2] == [
        str(runner.BASELINE_TOKEN_METER_BOOTSTRAP),
        str(runner.BASELINE_TOKEN_METER.parent),
    ]
    assert env["PYTHONPATH"].split(":")[-1] == "/existing"
    assert env["UCAGENT_BASELINE_TOKEN_LOG"].endswith(runner.BASELINE_TOKEN_LOG_NAME)
    assert env["UCAGENT_BASELINE_TOKEN_TARGET"].endswith("upstream/ucagent.py")


def test_baseline_token_meter_bootstrap_injects_callback_only_into_target(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from langchain_openai import ChatOpenAI\n"
        "model = ChatOpenAI(model='probe', api_key='x', base_url='http://127.0.0.1:1/v1')\n"
        "assert any(type(callback).__name__ == 'BaselineTokenCallback' for callback in model.callbacks)\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join(
        [
            str(runner.BASELINE_TOKEN_METER_BOOTSTRAP),
            str(runner.BASELINE_TOKEN_METER.parent),
        ]
    )
    env["UCAGENT_BASELINE_TOKEN_LOG"] = str(tmp_path / "tokens.jsonl")
    env["UCAGENT_BASELINE_TOKEN_TARGET"] = str(probe)

    result = subprocess.run(
        [sys.executable, str(probe)],
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr


def test_agent_output_guard_aborts_after_three_chunked_empty_ai_turns():
    guard = runner.AgentOutputGuard(empty_turn_limit=3)
    marker = b"================================== AI Message ==================================\n"

    assert guard.feed(marker[:25]) is None
    assert guard.feed(marker[25:] + b"None\n") is None
    assert guard.feed(b"\n" + marker + b"None\n") is None
    assert guard.feed(marker + b"No") is None
    assert guard.feed(b"ne\n") == "consecutive_empty_ai_turns"
    assert guard.consecutive_empty_turns == 3


def test_agent_output_guard_resets_after_nonempty_ai_turn():
    guard = runner.AgentOutputGuard(empty_turn_limit=3)
    marker = b"================================== AI Message ==================================\n"

    assert guard.feed(marker + b"None\n") is None
    assert guard.feed(marker + b"Tool Calls:\n") is None
    assert guard.feed(marker + b"None\n" + marker + b"None\n") is None
    assert guard.consecutive_empty_turns == 2


def _structured_line(event):
    return (
        "[data_collection][structured_event] "
        + json.dumps(event, ensure_ascii=False)
        + "\n"
    ).encode()


def _failed_check(stage, title, category, failed_case):
    return {
        "event_type": "check_result",
        "stage_index": stage,
        "stage_name": "test_case_implementation_in_batch",
        "stage_title": title,
        "success": False,
        "result": {
            "check_pass": False,
            "checker_categories": [category],
            "failed_cases_top": [failed_case],
            "failed_checkpoints_top": [],
            "error_top": [],
        },
    }


def test_stage_stall_watchdog_ignores_category_and_line_number_churn():
    watchdog = runner.StageStallWatchdog(
        checker_repeat_limit=3,
        llm_request_limit=2,
        prompt_token_limit=999_999,
        no_progress_seconds=999,
        stage_timeout_seconds=999,
        now_fn=lambda: 0.0,
    )
    title = "14.1-test_case_implementation_in_batch[0/14]"

    assert watchdog.feed(
        _structured_line(
            _failed_check(
                23,
                title,
                "generic_checker_failure",
                "tests/test_Adder.py:10-20::test_sum",
            )
        ),
        now=1.0,
    ) is None
    assert watchdog.feed(
        b"[data_collection][llm_request] status=success prompt_tokens=600000\n"
        b"[data_collection][llm_request] status=success prompt_tokens=600000\n",
        now=2.0,
    ) is None
    assert watchdog.feed(
        _structured_line(
            _failed_check(
                23,
                title,
                "bug_doc_checkpoint_not_found",
                "tests/test_Adder.py:30-45::test_sum",
            )
        ),
        now=3.0,
    ) is None
    reason = watchdog.feed(
        _structured_line(
            _failed_check(
                23,
                title,
                "failed_cases_need_classification",
                "tests/test_Adder.py:50-60::test_sum",
            )
        ),
        now=4.0,
    )

    assert reason == "stage_stall_watchdog:stage=23"
    assert watchdog.snapshot()["checker_repeats"] == 3
    assert watchdog.snapshot()["semantic_checker_key"] == [
        "case:tests/test_Adder.py::test_sum"
    ]


def test_stage_stall_watchdog_resets_on_internal_batch_progress():
    watchdog = runner.StageStallWatchdog(
        checker_repeat_limit=2,
        llm_request_limit=1,
        prompt_token_limit=1,
        no_progress_seconds=999,
        stage_timeout_seconds=999,
        now_fn=lambda: 0.0,
    )
    first = _failed_check(
        23,
        "14.1-test_case_implementation_in_batch[0/14]",
        "generic_checker_failure",
        "tests/test_Adder.py:10-20::test_sum",
    )
    advanced = _failed_check(
        23,
        "14.1-test_case_implementation_in_batch[10/14]",
        "generic_checker_failure",
        "tests/test_Adder.py:30-40::test_sum",
    )

    assert watchdog.feed(_structured_line(first), now=1.0) is None
    assert watchdog.feed(
        b"[data_collection][llm_request] status=success prompt_tokens=100\n",
        now=2.0,
    ) is None
    assert watchdog.feed(_structured_line(advanced), now=3.0) is None
    assert watchdog.checker_repeats == 1
    assert watchdog.llm_requests_since_progress == 0
    assert watchdog.current_stage_progress == (10, 14)


def test_stage_stall_watchdog_requires_checker_evidence_for_absolute_stage_cap():
    watchdog = runner.StageStallWatchdog(
        checker_repeat_limit=99,
        llm_request_limit=99,
        prompt_token_limit=999_999,
        no_progress_seconds=10,
        stage_timeout_seconds=20,
        now_fn=lambda: 0.0,
    )
    title = "14.1-test_case_implementation_in_batch[0/14]"
    event = _failed_check(
        23,
        title,
        "generic_checker_failure",
        "tests/test_Adder.py:10-20::test_sum",
    )

    assert watchdog.feed(_structured_line(event), now=1.0) is None
    assert watchdog.poll(now=30.0) is None
    assert watchdog.feed(_structured_line(event), now=31.0) == "stage_stall_watchdog:stage=23"


def test_stage_stall_watchdog_waits_for_failed_checker_boundary_after_budget_exhaustion():
    watchdog = runner.StageStallWatchdog(
        checker_repeat_limit=2,
        llm_request_limit=1,
        prompt_token_limit=1,
        no_progress_seconds=999,
        stage_timeout_seconds=999,
        now_fn=lambda: 0.0,
    )
    event = _failed_check(
        21,
        "12-basic_api_functional_test",
        "generic_checker_failure",
        "tests/test_Mux_api.py:10-20::test_sel3",
    )

    assert watchdog.feed(_structured_line(event), now=1.0) is None
    assert watchdog.feed(_structured_line(event), now=2.0) is None
    assert watchdog.feed(
        b"[data_collection][llm_request] status=success prompt_tokens=100\n",
        now=3.0,
    ) is None
    assert watchdog.feed(_structured_line({
        "event_type": "file_mutation",
        "stage_index": 21,
        "stage_name": "basic_api_functional_test",
        "success": True,
    }), now=4.0) is None

    assert watchdog.feed(_structured_line(event), now=5.0) == "stage_stall_watchdog:stage=21"


def test_stage_stall_watchdog_recognizes_decreasing_unmarked_function_count():
    watchdog = runner.StageStallWatchdog(
        checker_repeat_limit=2,
        llm_request_limit=1,
        prompt_token_limit=1,
        no_progress_seconds=999,
        stage_timeout_seconds=999,
        now_fn=lambda: 0.0,
    )

    def event(count):
        value = _failed_check(
            21,
            "12-basic_api_functional_test",
            "generic_checker_failure",
            "tests/test_ShiftRegister_api.py:10-20::test_reset_during_load",
        )
        value["result"]["error_top"] = [
            f"Find {count} functions do not have correct check point marks."
        ]
        return value

    assert watchdog.feed(_structured_line(event(24)), now=1.0) is None
    assert watchdog.feed(
        b"[data_collection][llm_request] status=success prompt_tokens=100\n",
        now=2.0,
    ) is None
    assert watchdog.feed(_structured_line(event(23)), now=3.0) is None
    assert watchdog.checker_repeats == 1
    assert "metric:unmarked_test_functions:23" in watchdog.semantic_checker_key


def test_validate_matrix_rejects_duplicate_arm_ids():
    value = matrix()
    value["arms"][1]["id"] = "important"
    with pytest.raises(ValueError, match="unique"):
        runner.validate_matrix(value)


def test_build_agent_command_records_seed_workspace_and_overrides(tmp_path):
    command = runner.build_agent_command(
        "/env/bin/python",
        tmp_path / "workspace",
        tmp_path / "run",
        123,
        {"overrides": ["context_upgrade.enable_context_reuse=false"]},
    )
    assert command[:2] == ["/env/bin/python", "ucagent.py"]
    assert command[command.index("--seed") + 1] == "123"
    assert command[command.index("--config") + 1] == "ucagent/setting.yaml"
    assert "--exit-on-completion" in command
    assert "-hm" not in command
    assert command[-2:] == ["--override", "context_upgrade.enable_context_reuse=false"]


def test_build_agent_command_accepts_frozen_workflow_config(tmp_path):
    config = tmp_path / "legacy.yaml"
    command = runner.build_agent_command(
        "/env/bin/python",
        tmp_path / "workspace",
        tmp_path / "run",
        123,
        {"overrides": []},
        config,
    )

    assert command[command.index("--config") + 1] == str(config)


def test_default_matrix_selects_upstream_64k_baseline():
    value = runner.load_json(runner.DEFAULT_MATRIX)
    fixed = value["fixed_configuration"]

    assert fixed["num_ctx"] == 65536
    assert fixed["summary_max_tokens"] == 51200
    assert fixed["summary_max_output_tokens"] == 1024
    assert fixed["workflow_profile"] == "upstream_0.9.1_20260716"
    assert fixed["interaction_mode"] == "standard"
    assert fixed["stream_output"] is True
    assert fixed["isolate_home"] is True
    assert fixed["required_python_packages"] == {
        "langchain": "1.2.15",
        "langchain-core": "1.2.10",
        "langgraph": "1.1.5",
        "langgraph-prebuilt": "1.0.9",
    }
    assert fixed["check_summarization_middleware"] is True
    assert fixed["requires_langchain_tool_call_preflight"] is True
    assert value["arms"][0]["id"] == "UPSTREAM_BASELINE"
    assert value["arms"][0]["overrides"] == ["openai.model_kwargs.stop=@delete"]
    assert "stop sequence" in fixed["note"]


def test_legacy_workflow_hash_and_stage_contract():
    import ucagent.checkers as checkers
    from ucagent.util.config import get_config

    legacy_matrix = (
        runner.REPO_ROOT
        / "benchmark/ucagent_experiments/adder_legacy_20260717_b0_b3_matrix.json"
    )
    value = runner.load_json(legacy_matrix)
    fixed = value["fixed_configuration"]
    workflow = runner.resolve_repo_file(fixed["agent_config"], label="agent configuration")
    workflow_source = runner.resolve_repo_file(
        fixed["workflow_source_config"],
        label="workflow source configuration",
    )
    assert runner.sha256_file(workflow_source) == fixed["workflow_source_sha256"]
    legacy_checker = runner.resolve_repo_file(
        "ucagent/checkers/legacy_20260717_unity_test.py",
        label="legacy checker source",
    )
    assert runner.sha256_file(legacy_checker) == fixed["legacy_checker_source_sha256"]

    config = get_config(config_file=str(workflow)).as_dict()

    checker_names = []

    def runnable_names(stage):
        names = []
        checker_names.extend(checker["clss"] for checker in stage.get("checker", []) or [])
        for child in stage.get("stage", []) or []:
            names.extend(runnable_names(child))
        is_group = (
            not stage.get("checker")
            and not stage.get("output_files")
            and not stage.get("reference_files")
            and bool(stage.get("stage"))
        )
        if not is_group:
            names.append(stage["name"])
        return names

    names = []
    for stage in config["stage"]:
        names.extend(runnable_names(stage))

    assert len(names) == 26
    assert names[19:23] == [
        "basic_api_functional_test",
        "create_test_case_templates",
        "test_case_implementation_in_batch",
        "comprehensive_verification_and_bug_analysis",
    ]
    assert not set(fixed["excluded_upstream_workflow_stages"]) & set(names)
    from ucagent.util.functions import import_class_from_str

    assert all(
        import_class_from_str(name) if "." in name else hasattr(checkers, name)
        for name in checker_names
    )


def test_initial_ledger_hashes_runtime_decision_code(tmp_path):
    value = runner.load_json(runner.DEFAULT_MATRIX)
    ledger = runner.initial_ledger(
        runner.DEFAULT_MATRIX,
        value,
        "/env/bin/python",
        "http://server/v1",
        value["fixed_configuration"]["model"],
        65536,
    )

    assert "scripts/run_adder_experiments.py" in ledger["artifacts"]
    agent_config = runner.resolve_repo_file(
        value["fixed_configuration"]["agent_config"],
        label="agent configuration",
    )
    assert runner.artifact_key(agent_config) in ledger["artifacts"]
    assert ledger["source_snapshot"]["root"] == str(
        runner.resolve_source_root(value["fixed_configuration"])
    )
    assert (
        ledger["source_snapshot"]["sha256"]
        == value["fixed_configuration"]["source_snapshot_sha256"]
    )
    runner.validate_frozen_artifacts(ledger)


def test_build_agent_command_can_execute_external_upstream_source(tmp_path):
    source_root = tmp_path / "upstream"
    source_root.mkdir()
    (source_root / "ucagent.py").write_text("", encoding="utf-8")

    command = runner.build_agent_command(
        "/env/bin/python",
        tmp_path / "workspace",
        tmp_path / "run",
        123,
        {"overrides": []},
        source_root / "config.yaml",
        source_root,
        "standard",
        True,
    )

    assert command[1] == str(source_root / "ucagent.py")
    assert command[command.index("-im") + 1] == "standard"
    assert "--stream-output" in command


def test_validate_frozen_artifacts_rejects_changed_hash():
    with pytest.raises(RuntimeError, match="use a new --run-root"):
        runner.validate_frozen_artifacts({
            "artifacts": {"ucagent/util/test_result.py": "not-the-current-hash"},
        })


def test_python_dependency_preflight_requires_exact_versions(monkeypatch):
    actual = {
        "langchain": "1.2.15",
        "langchain-core": "1.2.10",
    }

    def fake_run(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0, json.dumps(actual), "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.check_python_package_versions("/env/bin/python", actual) == actual

    with pytest.raises(RuntimeError, match="langchain: expected 1.2.3, found 1.2.15"):
        runner.check_python_package_versions(
            "/env/bin/python",
            {
                "langchain": "1.2.3",
                "langchain-core": "1.2.10",
            },
        )


def test_frozen_python_dependencies_reject_environment_drift():
    ledger = {
        "runtime_environment": {
            "python_packages": {
                "langchain": "1.2.15",
            }
        }
    }

    runner.validate_frozen_python_packages(ledger, {"langchain": "1.2.15"})
    with pytest.raises(RuntimeError, match="use a new --run-root"):
        runner.validate_frozen_python_packages(ledger, {"langchain": "1.2.16"})


def test_workspace_completed_rejects_missing_and_accepts_explicit_completion(tmp_path):
    assert runner.workspace_completed(tmp_path) is False
    (tmp_path / ".ucagent_info.json").write_text('{"all_completed": false}', encoding="utf-8")
    assert runner.workspace_completed(tmp_path) is False
    (tmp_path / ".ucagent_info.json").write_text('{"all_completed": true}', encoding="utf-8")
    assert runner.workspace_completed(tmp_path) is True


def test_workspace_completed_accepts_new_upstream_state_path(tmp_path):
    state_dir = tmp_path / ".ucagent"
    state_dir.mkdir()
    (state_dir / "ucagent_info.json").write_text(
        '{"all_completed": true}',
        encoding="utf-8",
    )

    assert runner.workspace_completed(tmp_path) is True


def test_workspace_completed_falls_back_after_invalid_new_state(tmp_path):
    state_dir = tmp_path / ".ucagent"
    state_dir.mkdir()
    (state_dir / "ucagent_info.json").write_text("{broken", encoding="utf-8")
    (tmp_path / ".ucagent_info.json").write_text(
        '{"all_completed": true}',
        encoding="utf-8",
    )

    assert runner.workspace_completed(tmp_path) is True


def test_check_model_available_requires_exact_model(monkeypatch):
    payload = json.dumps({"data": [{"id": "wanted:model"}, {"id": "other:model"}]}).encode()

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda request, timeout: Response(payload))
    assert runner.check_model_available("http://server/v1", "wanted:model", "key") == [
        "other:model",
        "wanted:model",
    ]
    with pytest.raises(RuntimeError, match="not available"):
        runner.check_model_available("http://server/v1", "missing:model", "key")


def test_warmup_ollama_model_sends_requested_64k_context(monkeypatch):
    payload = json.dumps(
        {
            "done": True,
            "load_duration": 2_000_000_000,
            "prompt_eval_count": 13,
            "prompt_eval_duration": 3_000_000_000,
        }
    ).encode()
    captured = {}

    class Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        return Response(payload)

    monkeypatch.setattr(runner.urllib.request, "urlopen", fake_urlopen)
    result = runner.warmup_ollama_model("http://server:11434/v1", "model", "key", 65536, 7)
    assert captured["url"] == "http://server:11434/api/chat"
    assert captured["body"]["options"]["num_ctx"] == 65536
    assert result["num_ctx"] == 65536
    assert result["load_seconds"] == 2.0


def test_warmup_ollama_model_reports_nonresident_loading_before_total_timeout(
    monkeypatch,
    capsys,
):
    import time

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/api/ps"):
            return Response(b'{"models": []}')
        time.sleep(0.2)
        return Response(b'{"done": true}')

    monkeypatch.setattr(runner.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="timed out after"):
        runner.warmup_ollama_model(
            "http://server:11434/v1",
            "model",
            "key",
            65536,
            7,
            timeout=0.04,
            status_interval=0.01,
        )

    output = capsys.readouterr().out
    assert "still waiting for Ollama" in output
    assert "resident=<none" in output


def test_ollama_running_models_reads_scheduler_state(monkeypatch):
    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    payload = json.dumps(
        {
            "models": [
                {"name": "model-a"},
                {"model": "model-b"},
            ]
        }
    ).encode()
    monkeypatch.setattr(
        runner.urllib.request,
        "urlopen",
        lambda request, timeout: Response(payload),
    )

    assert runner.ollama_running_models("http://server:11434/v1") == [
        "model-a",
        "model-b",
    ]


def test_check_ollama_model_context_requires_modelfile_match(monkeypatch):
    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    payload = json.dumps({"parameters": "top_p 0.95\nnum_ctx 65536\ntemperature 0.6"}).encode()
    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda request, timeout: Response(payload))
    assert runner.check_ollama_model_context("http://server/v1", "model", 65536) == 65536
    with pytest.raises(RuntimeError, match="expected 32768"):
        runner.check_ollama_model_context("http://server/v1", "model", 32768)


def test_parse_agent_log_summary_recovers_resumed_duration_and_tokens(tmp_path):
    log_path = tmp_path / "ucagent-log.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-08-12 12:00:00,000 - ucagent-log - INFO - Verify Agent started at: 2026-08-12 12:00:00",
                "2026-08-12 12:05:00,000 - ucagent-log - INFO - [data_collection][llm_request] role=main prompt_tokens=100 completion_tokens=10 ttft_ms=250.0",
                "2026-08-12 12:10:00,000 - ucagent-log - INFO - Verify Agent started at: 2026-08-12 12:10:00",
                "2026-08-12 12:30:00,000 - ucagent-log - INFO - [data_collection][llm_request] role=main prompt_tokens=200 completion_tokens=20 ttft_ms=NA",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.parse_agent_log_summary(log_path)

    assert result["agent_start_count"] == 2
    assert result["active_seconds"] == 1500
    assert result["prompt_tokens"] == 300
    assert result["completion_tokens"] == 30
    assert result["llm_request_count"] == 2
    assert result["ttft_p50_ms"] == 250.0
    assert result["token_measurement"] == "native_data_collection"


def test_parse_agent_log_summary_excludes_offline_gap_between_resumes(tmp_path):
    log_path = tmp_path / "ucagent-log.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-09-16 16:00:00,000 - INFO - Verify Agent started at: 2026-09-16 16:00:00",
                "2026-09-16 16:20:00,000 - INFO - [data_collection][llm_request] status=success prompt_tokens=10 completion_tokens=1 ttft_ms=10",
                "2026-09-17 10:57:00,000 - INFO - preparing resumed process",
                "2026-09-17 10:58:00,000 - INFO - Verify Agent started at: 2026-09-17 10:58:00",
                "2026-09-17 11:08:00,000 - INFO - Verify Agent is exited.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.parse_agent_log_summary(log_path)

    assert result["agent_start_count"] == 2
    assert result["active_seconds"] == 1800
    assert [segment["duration_seconds"] for segment in result["attempt_segments"]] == [1200, 600]


def test_parse_agent_log_summary_uses_external_meter_when_native_telemetry_is_absent(tmp_path):
    log_path = tmp_path / "ucagent-log.log"
    token_path = tmp_path / runner.BASELINE_TOKEN_LOG_NAME
    log_path.write_text(
        "2026-08-25 12:00:00,000 - ucagent-log - INFO - Verify Agent started at: 2026-08-25 12:00:00\n"
        "2026-08-25 12:30:00,000 - ucagent-log - INFO - Verify Agent finished at: 2026-08-25 12:30:00\n",
        encoding="utf-8",
    )
    token_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "llm_request",
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                        "ttft_ms": 300.0,
                        "token_source": "approx",
                    }
                ),
                "not-json",
                json.dumps(
                    {
                        "event": "llm_request",
                        "prompt_tokens": 200,
                        "completion_tokens": 20,
                        "ttft_ms": 500.0,
                        "token_source": "provider",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.parse_agent_log_summary(log_path, token_path)

    assert result["prompt_tokens"] == 300
    assert result["completion_tokens"] == 30
    assert result["llm_request_count"] == 2
    assert result["token_measurement"] == "external_langchain_callback"
    assert result["token_sources"] == {"approx": 1, "provider": 1}
    assert result["token_meter_invalid_records"] == 1


def test_finalize_completed_result_prefers_attempt_history_and_writes_once(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "ucagent-log.log").write_text(
        "2026-08-12 12:00:00,000 - ucagent-log - INFO - Verify Agent started at: 2026-08-12 12:00:00\n"
        "2026-08-12 12:30:00,000 - ucagent-log - INFO - [data_collection][llm_request] role=main prompt_tokens=1200 completion_tokens=34 ttft_ms=100.0\n",
        encoding="utf-8",
    )
    start = datetime(2026, 8, 12, tzinfo=timezone.utc)
    record = {
        "arm": "CO_TEST",
        "seed": 7,
        "attempts": 2,
        "attempt_history": [
            {
                "duration_seconds": 3600,
                "started_at": start.isoformat(),
                "ended_at": (start + timedelta(hours=1)).isoformat(),
            },
            {
                "duration_seconds": 1800,
                "started_at": (start + timedelta(hours=2)).isoformat(),
                "ended_at": (start + timedelta(hours=2, minutes=30)).isoformat(),
            },
        ],
    }
    res_path = tmp_path / "res.csv"

    first = runner.finalize_completed_result(record, "CO_TEST__seed_7", run_dir, res_path)
    second = runner.finalize_completed_result(record, "CO_TEST__seed_7", run_dir, res_path)

    assert first["active_seconds"] == 5400
    assert first["formatted_duration"] == "1h30min"
    assert first["duration_source"] == "attempt_history"
    assert second == first
    rows = list(csv.DictReader(res_path.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["Model"] == "CO_TEST(s7)"
    assert rows[0]["time"] == "1h30min"
    assert rows[0]["token_in"] == "1.2K"
