from ucagent.verify_agent import VerifyAgent


def parser() -> VerifyAgent:
    return VerifyAgent.__new__(VerifyAgent)


def test_resume_stats_use_carry_forward_marker(tmp_path):
    log_path = tmp_path / "ucagent-log.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-08-26 09:59:59,000 - ucagent-log - INFO - Load config completed.",
                "2026-08-26 10:00:00,000 - ucagent-log - INFO - Verify Agent started at: 2026-08-26 10:00:00",
                "2026-08-26 15:00:00,000 - ucagent-log - INFO - an old stage action",
                "2026-08-26 16:00:00,000 - ucagent-log - INFO - [data_collection][resume_carry_forward] reset_stage=23 time_seconds=17168.687 token_in=9915860 token_out=87152",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert parser()._parse_resume_stats_from_log(str(log_path)) == (
        17168.687,
        9915860,
        87152,
    )


def test_resume_stats_fall_back_to_interrupted_run_timestamps(tmp_path):
    log_path = tmp_path / "ucagent-log.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-08-26 10:00:00,000 - ucagent-log - INFO - Verify Agent started at: 2026-08-26 10:00:00",
                "2026-08-26 10:10:00,000 - ucagent-log - INFO - last observable action",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert parser()._parse_resume_stats_from_log(str(log_path)) == (600, 0, 0)
