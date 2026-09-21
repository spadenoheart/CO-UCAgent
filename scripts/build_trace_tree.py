#!/usr/bin/env python3
"""Build a TRACE-style trajectory tree from UCAgent structured events."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ucagent.memory.context_reuse import failure_signature_from_summary
from ucagent.util.test_result import (
    TEST_OUTCOME_PASS,
    classify_test_outcome,
    stage_allows_expected_failures,
)


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


def load_events(paths: list[str]) -> tuple[list[dict], list[str]]:
    events: list[dict] = []
    sources: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        candidates: list[Path]
        if path.is_dir():
            candidates = sorted(path.rglob("structured_events.jsonl")) + sorted(path.rglob("ucagent-log.log"))
        else:
            candidates = [path]
        for candidate in candidates:
            loaded = list(iter_events(candidate))
            if loaded:
                sources.append(str(candidate))
                events.extend(loaded)
    events.sort(key=lambda item: (float(item.get("time_unix", 0) or 0), int(item.get("round_index", 0) or 0)))
    return events, sources


def filter_events_by_run_id(events: list[dict], run_ids: set[str]) -> list[dict]:
    if not run_ids:
        return events
    return [
        event
        for event in events
        if str(event.get("run_id") or event.get("thread_id") or "") in run_ids
    ]


def stage_from_event(event: dict) -> dict:
    for key in ("stage_before", "from_stage"):
        value = event.get(key)
        if isinstance(value, dict) and value.get("stage_index") is not None:
            return {
                "stage_index": value.get("stage_index"),
                "stage_name": value.get("stage_name", ""),
                "stage_title": value.get("stage_title", ""),
            }
    return {
        "stage_index": event.get("stage_index"),
        "stage_name": event.get("stage_name", ""),
        "stage_title": event.get("stage_title", ""),
    }


def event_role(event_type: str) -> str:
    if event_type == "file_mutation":
        return "action"
    if event_type in {"test_run", "check_result"}:
        return "observation"
    if event_type == "stage_transition":
        return "reward"
    if event_type == "context_reuse_decision":
        return "memory_action"
    return "event"


def compact_list(value: Any, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value[:limit]]


def derive_tests_passed_all(result: dict, stage_name: str = "") -> Any:
    outcome, _ = classify_test_outcome(
        result,
        allow_expected_failures=stage_allows_expected_failures(stage_name),
    )
    return outcome == TEST_OUTCOME_PASS


def failure_signature(event: dict) -> str:
    result = event.get("result") if isinstance(event.get("result"), dict) else {}
    return failure_signature_from_summary(
        event.get("event_type", ""),
        event.get("stage_name", ""),
        result,
    )


def node_from_event(event: dict, node_id: str, parent_id: str | None) -> dict:
    event_type = event.get("event_type", "unknown")
    stage = stage_from_event(event)
    result = event.get("result") if isinstance(event.get("result"), dict) else {}
    node = {
        "node_id": node_id,
        "parent_id": parent_id,
        "event_type": event_type,
        "role": event_role(event_type),
        "time_unix": event.get("time_unix"),
        "timestamp": event.get("timestamp"),
        "round_index": event.get("round_index"),
        "run_id": str(event.get("run_id") or event.get("thread_id") or ""),
        "thread_id": event.get("thread_id"),
        "success": event.get("success"),
        **stage,
    }
    if event_type == "file_mutation":
        node["action"] = {
            "tool": event.get("tool"),
            "operation": event.get("operation"),
            "path": event.get("path"),
            "path_category": event.get("path_category"),
            "detail": event.get("detail", {}),
        }
    elif event_type == "test_run":
        test_outcome, test_outcome_reason = classify_test_outcome(
            result,
            allow_expected_failures=stage_allows_expected_failures(stage.get("stage_name")),
        )
        node["observation"] = {
            "tool": event.get("tool"),
            "target": event.get("target"),
            "timeout": event.get("timeout"),
            "duration_ms": event.get("duration_ms"),
            "tests_total": result.get("tests_total"),
            "tests_failed": result.get("tests_failed"),
            "run_test_success": result.get("run_test_success"),
            "test_outcome": test_outcome,
            "test_outcome_reason": test_outcome_reason,
            "tests_passed_all": derive_tests_passed_all(result, stage.get("stage_name", "")),
            "test_status_counts": result.get("test_status_counts", {}),
            "failed_cases_top": compact_list(result.get("failed_cases_top")),
            "failed_checkpoints_top": compact_list(result.get("failed_checkpoints_top")),
            "error_top": compact_list(result.get("error_top"), 4),
            "failure_signature": failure_signature(event),
        }
    elif event_type == "check_result":
        node["observation"] = {
            "tool": event.get("tool"),
            "timeout": event.get("timeout"),
            "is_complete": event.get("is_complete"),
            "check_pass": result.get("check_pass", event.get("success")),
            "checker_categories": compact_list(result.get("checker_categories")),
            "failed_cases_top": compact_list(result.get("failed_cases_top")),
            "failed_checkpoints_top": compact_list(result.get("failed_checkpoints_top")),
            "error_top": compact_list(result.get("error_top"), 4),
            "failure_signature": failure_signature(event),
        }
    elif event_type == "stage_transition":
        node["reward"] = {
            "trigger": event.get("trigger"),
            "advanced": bool(event.get("advanced")),
            "all_completed": bool(event.get("all_completed")),
            "from_stage": event.get("from_stage"),
            "to_stage": event.get("to_stage"),
            "terminal_reward": 1 if event.get("advanced") or event.get("all_completed") else 0,
        }
    elif event_type == "context_reuse_decision":
        node["memory_decision"] = {
            "decision": event.get("decision"),
            "reason": event.get("reason"),
            "failure": event.get("failure", {}),
            "strategies": event.get("strategies", []),
            "search_summary": event.get("search_summary", {}),
            "hint_count": event.get("hint_count", 0),
            "prompt_item_count": event.get("prompt_item_count", 0),
        }
    return node


def build_trace(events: list[dict], sources: list[str]) -> dict:
    nodes = []
    stages: dict[Any, dict] = {}
    previous_node_by_run: dict[str, str] = {}
    counters = Counter()
    dut_values = [event.get("dut") for event in events if event.get("dut")]
    dut = Counter(dut_values).most_common(1)[0][0] if dut_values else ""

    for index, event in enumerate(events):
        stage = stage_from_event(event)
        stage_index = stage.get("stage_index")
        event_type = event.get("event_type", "unknown")
        node_id = f"n{index:05d}"
        run_id = str(event.get("run_id") or event.get("thread_id") or "legacy")
        node = node_from_event(event, node_id, previous_node_by_run.get(run_id))
        nodes.append(node)
        previous_node_by_run[run_id] = node_id
        counters[event_type] += 1

        stage_key = (run_id, stage_index)
        stage_bucket = stages.setdefault(stage_key, {
            **stage,
            "run_id": run_id,
            "node_ids": [],
            "event_type_counts": Counter(),
            "file_mutations": 0,
            "test_runs": 0,
            "check_failures": 0,
            "check_passes": 0,
            "stage_advanced": False,
            "all_completed": False,
            "failure_signatures": Counter(),
        })
        stage_bucket["node_ids"].append(node_id)
        stage_bucket["event_type_counts"][event_type] += 1
        if event_type == "file_mutation":
            stage_bucket["file_mutations"] += 1
        elif event_type == "test_run":
            stage_bucket["test_runs"] += 1
            obs = node.get("observation", {})
            if obs.get("tests_passed_all") is False and obs.get("failure_signature"):
                stage_bucket["failure_signatures"][obs["failure_signature"]] += 1
        elif event_type == "check_result":
            obs = node.get("observation", {})
            if obs.get("check_pass"):
                stage_bucket["check_passes"] += 1
            else:
                stage_bucket["check_failures"] += 1
            if not obs.get("check_pass") and obs.get("failure_signature"):
                stage_bucket["failure_signatures"][obs["failure_signature"]] += 1
        elif event_type == "stage_transition":
            reward = node.get("reward", {})
            stage_bucket["stage_advanced"] = stage_bucket["stage_advanced"] or bool(reward.get("advanced"))
            stage_bucket["all_completed"] = stage_bucket["all_completed"] or bool(reward.get("all_completed"))

    stage_list = []
    for (run_id, stage_index), stage in sorted(
        stages.items(),
        key=lambda item: (item[0][0], item[0][1] is None, item[0][1]),
    ):
        stage_list.append({
            "run_id": run_id,
            "stage_index": stage.get("stage_index"),
            "stage_name": stage.get("stage_name"),
            "stage_title": stage.get("stage_title"),
            "node_ids": stage["node_ids"],
            "event_type_counts": dict(stage["event_type_counts"]),
            "file_mutations": stage["file_mutations"],
            "test_runs": stage["test_runs"],
            "check_failures": stage["check_failures"],
            "check_passes": stage["check_passes"],
            "stage_advanced": stage["stage_advanced"],
            "all_completed": stage["all_completed"],
            "failure_signatures": dict(stage["failure_signatures"].most_common(10)),
        })

    return {
        "schema_version": 1,
        "trace_type": "ucagent_structured_trajectory",
        "sources": sources,
        "dut": dut,
        "event_count": len(events),
        "node_count": len(nodes),
        "run_count": len(previous_node_by_run),
        "stage_count": len(stage_list),
        "event_type_counts": dict(counters),
        "stages": stage_list,
        "nodes": nodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="structured_events.jsonl, ucagent-log.log, or directories")
    parser.add_argument("--out", help="Output JSON path. Defaults to stdout.")
    parser.add_argument("--nodes-jsonl", help="Optional output path for flat node JSONL.")
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="Keep only this run/thread id. Repeat to include multiple runs.",
    )
    args = parser.parse_args()

    events, sources = load_events(args.paths)
    events = filter_events_by_run_id(events, set(args.run_id))
    if args.run_id and not events:
        parser.error(f"no structured events matched --run-id {args.run_id}")
    trace = build_trace(events, sources)
    text = json.dumps(trace, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.nodes_jsonl:
        nodes_path = Path(args.nodes_jsonl)
        nodes_path.parent.mkdir(parents=True, exist_ok=True)
        with nodes_path.open("w", encoding="utf-8") as fout:
            for node in trace["nodes"]:
                fout.write(json.dumps(node, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
