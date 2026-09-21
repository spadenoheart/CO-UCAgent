from __future__ import annotations

import json

from scripts.prepare_multi_dut_stage_resume import filter_stage_events, truncate_log_from_timestamp


def test_filter_stage_events_keeps_incoming_transition(tmp_path):
    path = tmp_path / "events.jsonl"
    events = [
        {"event_type": "check_result", "stage_index": 20, "stage_after": {"stage_index": 21}, "timestamp": "2026-08-28T01:00:00"},
        {"event_type": "stage_state_package_update", "stage_index": 21, "timestamp": "2026-08-28T01:00:01"},
        {"event_type": "file_mutation", "stage_index": 21, "timestamp": "2026-08-28T01:00:02"},
    ]
    path.write_text("".join(json.dumps(item) + "\n" for item in events), encoding="utf-8")

    removed, cutoff = filter_stage_events(path, 21)
    remaining = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert removed == 2
    assert cutoff == "2026-08-28T01:00:01"
    assert remaining == [events[0]]


def test_truncate_log_from_stage_timestamp(tmp_path):
    path = tmp_path / "ucagent-log.log"
    path.write_text(
        "2026-08-28 00:59:59,999 - keep\n"
        "2026-08-28 01:00:01,000 - remove\n"
        "continuation also removed\n",
        encoding="utf-8",
    )

    removed = truncate_log_from_timestamp(path, "2026-08-28T01:00:01")

    assert removed == 2
    assert path.read_text(encoding="utf-8") == "2026-08-28 00:59:59,999 - keep\n"


def test_truncate_console_accepts_dot_milliseconds_and_ansi_prefix(tmp_path):
    path = tmp_path / "console.log"
    path.write_text(
        "\x1b[32m[2026-08-28 01:00:00.999 INFO] keep\x1b[0m\n"
        "\x1b[32m[2026-08-28 01:00:01.000 INFO] remove\x1b[0m\n",
        encoding="utf-8",
    )

    assert truncate_log_from_timestamp(path, "2026-08-28T01:00:01") == 1
    assert "keep" in path.read_text(encoding="utf-8")
    assert "remove" not in path.read_text(encoding="utf-8")
