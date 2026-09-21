#!/usr/bin/env python3
"""Analyze UCAgent structured event logs.

The script accepts either:
- output/structured_events.jsonl
- ucagent-log.log files containing "[data_collection][structured_event] {...}"
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


EVENT_PREFIX = "[data_collection][structured_event] "


def iter_events(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8", errors="ignore") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if EVENT_PREFIX in line:
                line = line.split(EVENT_PREFIX, 1)[1]
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event_type"):
                yield event


def key_text(key: tuple) -> str:
    return " | ".join(str(item) for item in key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="structured_events.jsonl or ucagent-log.log path(s)")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON summary")
    args = parser.parse_args()

    events = []
    for raw_path in args.paths:
        path = Path(raw_path)
        if path.is_dir():
            candidates = list(path.rglob("structured_events.jsonl")) + list(path.rglob("ucagent-log.log"))
            for candidate in sorted(candidates):
                events.extend(iter_events(candidate))
        else:
            events.extend(iter_events(path))

    by_category = Counter()
    by_operation = Counter()
    by_stage = Counter()
    by_stage_category = Counter()
    by_event_type = Counter()
    by_check = Counter()
    by_test = Counter()
    by_transition = Counter()
    failed = Counter()
    paths_by_category: dict[str, Counter] = defaultdict(Counter)

    for event in events:
        event_type = event.get("event_type", "unknown")
        dut = event.get("dut", "")
        stage_index = event.get("stage_index", "")
        stage_name = event.get("stage_name", "")
        category = event.get("path_category", "unknown")
        operation = event.get("operation", "unknown")
        success = bool(event.get("success"))
        path = event.get("path", "")

        by_event_type[(event_type, success)] += 1
        by_stage[(event_type, dut, stage_index, stage_name, success)] += 1

        if event_type == "file_mutation":
            by_category[(category, success)] += 1
            by_operation[(operation, success)] += 1
            by_stage_category[(dut, stage_index, stage_name, category, success)] += 1
            paths_by_category[category][path] += 1
        elif event_type == "test_run":
            result = event.get("result") if isinstance(event.get("result"), dict) else {}
            by_test[(
                dut,
                stage_index,
                stage_name,
                success,
                result.get("tests_total"),
                result.get("tests_failed"),
                result.get("tests_passed_all"),
            )] += 1
        elif event_type == "check_result":
            result = event.get("result") if isinstance(event.get("result"), dict) else {}
            categories = result.get("checker_categories") or []
            category_key = ",".join(str(item) for item in categories[:3]) if categories else ""
            by_check[(event.get("tool", ""), dut, stage_index, stage_name, success, category_key)] += 1
        elif event_type == "stage_transition":
            from_stage = event.get("from_stage") if isinstance(event.get("from_stage"), dict) else {}
            to_stage = event.get("to_stage") if isinstance(event.get("to_stage"), dict) else {}
            by_transition[(
                dut,
                from_stage.get("stage_index", stage_index),
                to_stage.get("stage_index", stage_index),
                bool(event.get("advanced")),
                bool(event.get("all_completed")),
                success,
            )] += 1
        if not success:
            failed[(event_type, dut, stage_index, stage_name, category, operation, path)] += 1

    summary = {
        "event_count": len(events),
        "by_event_type": {key_text(k): v for k, v in by_event_type.most_common()},
        "by_category": {key_text(k): v for k, v in by_category.most_common()},
        "by_operation": {key_text(k): v for k, v in by_operation.most_common()},
        "by_stage": {key_text(k): v for k, v in by_stage.most_common()},
        "by_stage_category": {key_text(k): v for k, v in by_stage_category.most_common()},
        "by_test": {key_text(k): v for k, v in by_test.most_common()},
        "by_check": {key_text(k): v for k, v in by_check.most_common()},
        "by_transition": {key_text(k): v for k, v in by_transition.most_common()},
        "failed": {key_text(k): v for k, v in failed.most_common(50)},
        "top_paths_by_category": {
            category: dict(counter.most_common(20))
            for category, counter in sorted(paths_by_category.items())
        },
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print(f"events: {summary['event_count']}")
    print("\n[by_event_type]")
    for key, value in summary["by_event_type"].items():
        print(f"{value:5d}  {key}")
    print("\n[by_category]")
    for key, value in summary["by_category"].items():
        print(f"{value:5d}  {key}")
    print("\n[by_operation]")
    for key, value in summary["by_operation"].items():
        print(f"{value:5d}  {key}")
    print("\n[by_test]")
    for key, value in list(summary["by_test"].items())[:30]:
        print(f"{value:5d}  {key}")
    print("\n[by_check]")
    for key, value in list(summary["by_check"].items())[:30]:
        print(f"{value:5d}  {key}")
    print("\n[by_transition]")
    for key, value in list(summary["by_transition"].items())[:30]:
        print(f"{value:5d}  {key}")
    print("\n[by_stage_category]")
    for key, value in list(summary["by_stage_category"].items())[:40]:
        print(f"{value:5d}  {key}")
    if failed:
        print("\n[failed top]")
        for key, value in summary["failed"].items():
            print(f"{value:5d}  {key}")


if __name__ == "__main__":
    main()
