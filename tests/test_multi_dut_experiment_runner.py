from __future__ import annotations

import json

from scripts import run_multi_dut_experiments as runner
from scripts import run_upstream_baseline_multi_dut as baseline_runner


def _suite() -> dict:
    return json.loads(runner.DEFAULT_SUITE.read_text(encoding="utf-8"))


def test_default_suite_covers_ten_duts_with_adaptive_profiles() -> None:
    suite = _suite()

    runner.validate_suite(suite)
    runs = runner.selected_runs(suite, set(), [])

    assert len(runs) == 10
    assert {item["profile"] for item in runs} == {"light", "medium", "heavy"}
    assert {item["dut"] for item in runs} == {
        "ALU754", "Adder", "DualPort", "FSM", "HPerfCounter",
        "IntegerDivider", "Mux", "Sbuffer", "ShiftRegister", "uart_tx",
    }


def test_multi_seed_selection_is_resumable_by_stable_run_id() -> None:
    runs = runner.selected_runs(_suite(), {"Adder", "FSM"}, [11, 12])

    assert [item["run_id"] for item in runs] == [
        "Adder__seed_11", "Adder__seed_12", "FSM__seed_11", "FSM__seed_12",
    ]


def test_initialization_marker_skips_reinitializing_existing_workspace(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "Adder").mkdir(parents=True)
    runner.atomic_write_json(
        workspace / ".ucagent_batch_initialized.json",
        {"dut": "Adder", "initialized_at": "now"},
    )
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    runner.init_workspace("python", tmp_path, workspace, "Adder", {})

    assert calls == []


def test_summary_csv_records_stall_evidence(tmp_path) -> None:
    path = tmp_path / "summary.csv"
    ledger = {
        "runs": {
            "Adder__seed_1": {
                "dut": "Adder",
                "seed": 1,
                "profile": "light",
                "status": "stalled",
                "attempts": 1,
                "duration_seconds": 7200,
                "failure_reason": "stage_stall_watchdog:stage=23",
                "workspace": "/tmp/workspace",
                "stall_watchdog": {
                    "stage_index": 23,
                    "checker_repeats": 6,
                    "llm_requests_since_progress": 45,
                    "prompt_tokens_since_progress": 1_000_000,
                },
            }
        }
    }

    runner.write_suite_summary(path, ledger)
    text = path.read_text(encoding="utf-8")

    assert "Adder__seed_1" in text
    assert "stalled" in text
    assert "stage_stall_watchdog:stage=23" in text
    assert ",2.0,,,,,23,6,45,1000000," in text


def test_summary_csv_records_full_run_token_usage(tmp_path) -> None:
    path = tmp_path / "summary.csv"
    ledger = {
        "runs": {
            "FSM__seed_1": {
                "dut": "FSM",
                "seed": 1,
                "profile": "baseline",
                "status": "completed",
                "attempts": 1,
                "duration_seconds": 3600,
                "workspace": "/tmp/workspace",
                "stall_watchdog": {"enabled": False},
                "token_usage": {
                    "prompt_tokens": 123456,
                    "completion_tokens": 7890,
                    "llm_request_count": 42,
                    "token_measurement": "external_langchain_callback",
                },
            }
        }
    }

    runner.write_suite_summary(path, ledger)
    text = path.read_text(encoding="utf-8")

    assert "123456,7890,42,external_langchain_callback" in text


def test_agent_command_enables_skills_when_suite_requires_them(tmp_path) -> None:
    suite = _suite()
    suite["use_skill"] = True

    command = runner.build_agent_command(
        "python",
        tmp_path / "source",
        tmp_path / "workspace",
        tmp_path / "run",
        "FSM",
        7,
        suite,
    )

    assert command.count("--use-skill") == 1


def test_upstream_baseline_suite_includes_dualport_and_qwen_compatibility() -> None:
    suite = json.loads(baseline_runner.DEFAULT_SUITE.read_text(encoding="utf-8"))

    assert "DualPort" in {item["name"] for item in suite["duts"]}
    assert "examples/DualPort" in suite["source_snapshot_paths"]
    assert "openai.model_kwargs.stop=@delete" in suite["overrides"]


def test_remote_stream_disconnect_is_infrastructure_failure(tmp_path) -> None:
    attempt_root = tmp_path / "attempt_1"
    run_id = "FSM__seed_7"
    run_dir = attempt_root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "console.log").write_text(
        "httpx.RemoteProtocolError: peer closed connection without sending complete message body",
        encoding="utf-8",
    )

    assert baseline_runner.is_infrastructure_failure(attempt_root, run_id, {"status": "failed"})
