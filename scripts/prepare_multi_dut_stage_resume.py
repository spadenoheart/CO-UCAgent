#!/usr/bin/env python3
"""Archive a falsely-stopped stage tail and prepare an auditable resume."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from scripts.run_adder_experiments import atomic_write_json, utc_now
    from scripts.run_multi_dut_experiments import write_suite_summary
except ModuleNotFoundError:
    from run_adder_experiments import atomic_write_json, utc_now
    from run_multi_dut_experiments import write_suite_summary


VALIDITY = "patched_resume_not_clean_end_to_end"
LOG_TIMESTAMP_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,.](\d{3})"
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def filter_stage_events(path: Path, stage_index: int) -> tuple[int, str | None]:
    """Remove events owned by a stage while retaining its incoming transition."""
    if not path.is_file():
        return 0, None
    retained: list[str] = []
    removed = 0
    cutoff: str | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            retained.append(raw_line)
            continue
        if event.get("stage_index") == stage_index:
            removed += 1
            timestamp = str(event.get("timestamp") or "")
            if timestamp and (cutoff is None or timestamp < cutoff):
                cutoff = timestamp
            continue
        retained.append(raw_line)
    suffix = "\n" if retained else ""
    path.write_text("\n".join(retained) + suffix, encoding="utf-8")
    return removed, cutoff


def truncate_log_from_timestamp(path: Path, cutoff_iso: str | None) -> int:
    if not path.is_file() or not cutoff_iso:
        return 0
    cutoff = datetime.fromisoformat(cutoff_iso).replace(tzinfo=None)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    split_at = len(lines)
    for index, line in enumerate(lines):
        match = LOG_TIMESTAMP_RE.search(line)
        if not match:
            continue
        timestamp = datetime.strptime(
            f"{match.group(1)},{match.group(2)}",
            "%Y-%m-%d %H:%M:%S,%f",
        )
        if timestamp >= cutoff:
            split_at = index
            break
    removed = len(lines) - split_at
    path.write_text("".join(lines[:split_at]), encoding="utf-8")
    return removed


def prepare_stage_resume(
    run_root: Path,
    run_id: str,
    stage_index: int,
    reason: str,
    archive_label: str,
) -> dict[str, Any]:
    ledger_path = run_root / "ledger.json"
    ledger = load_json(ledger_path)
    record = ledger.get("runs", {}).get(run_id)
    if not isinstance(record, dict):
        raise KeyError(f"run id not found: {run_id}")
    run_dir = run_root / run_id
    workspace = Path(str(record.get("workspace") or run_dir / "workspace"))
    status_path = workspace / ".ucagent" / "ucagent_info.json"
    status = load_json(status_path)
    actual_stage = int(status.get("stage_index", -1))
    if actual_stage != stage_index:
        raise RuntimeError(f"{run_id} is at Stage {actual_stage}, expected Stage {stage_index}")

    archive_dir = run_dir / "resume_archives" / archive_label
    archive_dir.mkdir(parents=True, exist_ok=False)
    archived: list[str] = []
    for source in (
        run_dir / "ucagent-log.log",
        run_dir / "console.log",
        run_dir / "ucagent-msg.log",
        run_dir / "stall_watchdog.json",
        workspace / "unity_test" / "structured_events.jsonl",
    ):
        if source.is_file():
            target = archive_dir / source.name
            shutil.copy2(source, target)
            archived.append(str(target))

    events_path = workspace / "unity_test" / "structured_events.jsonl"
    removed_events, cutoff = filter_stage_events(events_path, stage_index)
    removed_log_lines = truncate_log_from_timestamp(run_dir / "ucagent-log.log", cutoff)
    removed_console_lines = truncate_log_from_timestamp(run_dir / "console.log", cutoff)

    dumps_dir = workspace / "unity_test" / "llm_input_dumps"
    dump_archive = archive_dir / "llm_input_dumps"
    removed_dumps = 0
    for path in sorted(dumps_dir.glob(f"req_*_stage_{stage_index}.json")) if dumps_dir.is_dir() else []:
        dump_archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dump_archive / path.name)
        path.unlink()
        removed_dumps += 1

    watchdog_path = run_dir / "stall_watchdog.json"
    if watchdog_path.exists():
        watchdog_path.unlink()

    stage_info = status.get("stages_info", {}).get(str(stage_index))
    if isinstance(stage_info, dict):
        stage_info["time_cost"] = 0.0
    atomic_write_json(status_path, status)

    audit = {
        "prepared_at": utc_now(),
        "run_id": run_id,
        "stage_index": stage_index,
        "reason": reason,
        "archive_dir": str(archive_dir),
        "archived_files": archived,
        "removed_stage_events": removed_events,
        "removed_llm_dumps": removed_dumps,
        "removed_log_lines": removed_log_lines,
        "removed_console_lines": removed_console_lines,
        "cutoff_timestamp": cutoff,
        "experimental_validity": VALIDITY,
        "preserved": "workspace files and pre-stage telemetry",
        "not_cleaned": "ucagent-msg.log is archived but retained because it is an audit transcript, not active metric input",
    }
    atomic_write_json(archive_dir / "resume_audit.json", audit)

    record.setdefault("resume_patches", []).append(audit)
    record.update({
        "status": "interrupted",
        "failure_reason": "",
        "experimental_validity": VALIDITY,
        "stall_watchdog": {},
    })
    ledger["updated_at"] = utc_now()
    atomic_write_json(ledger_path, ledger)
    write_suite_summary(run_root / "summary.csv", ledger)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--archive-label", required=True)
    args = parser.parse_args()
    audit = prepare_stage_resume(
        args.run_root.resolve(),
        args.run_id,
        args.stage,
        args.reason,
        args.archive_label,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
