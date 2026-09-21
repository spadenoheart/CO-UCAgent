#!/usr/bin/env python3
"""Render UCAgent runs as article-style vertical execution trajectories.

The visualizer is intentionally read-only and dependency-free. It combines:

* ``structured_events.jsonl`` for mutations, tests, Checker results, stage
  transitions, memory retrieval, and progress-control decisions;
* ``ucagent-log.log`` for measured LLM latency and token usage;
* ``llm_input_dumps`` for read/search tool calls that predate structured tool
  telemetry.

Static mode writes a self-contained HTML report. Server mode rereads active run
artifacts and refreshes the browser without modifying the running workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import statistics
import sys
import threading
import time
import webbrowser
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
EVENT_PREFIX = "[data_collection][structured_event] "
LOG_TIMESTAMP_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
)
LLM_REQUEST_MARKER = "[data_collection][llm_request] "
REQUEST_INDEX_RE = re.compile(r"req_(\d+)_stage_")

AREA_ORDER = [
    "需求与规范",
    "RTL/DUT",
    "Wrapper/Bundle",
    "覆盖率模型",
    "Env/API",
    "定向测试",
    "随机测试",
    "Bug分析",
    "验证/Checker",
    "记忆与控制",
    "其他/基础设施",
]

MUTATION_TOOLS = {
    "WriteTextFile",
    "EditTextFile",
    "EditFileDialogs",
    "ReplaceStringInFile",
    "CopyFile",
    "MoveFile",
    "DeleteFile",
    "CreateDirectory",
}
STRUCTURED_TOOL_NAMES = MUTATION_TOOLS | {"RunTestCases", "Check", "Complete"}
READ_TOOLS = {"ReadTextFile", "GetFileInfo", "PathList", "RoleInfo", "Detail", "Status", "CurrentTips"}
SEARCH_TOOLS = {"SearchText", "FindFiles"}

ERROR_HEAD_RE = re.compile(
    r"(?:^|\n)\s*(?:\[ERROR\]|ERROR:|Error:|Traceback|FAILED|"
    r"MUTATION_BUDGET_BLOCKED|No such file|not found:)",
    re.IGNORECASE,
)
EMPTY_RESULT_RE = re.compile(
    r"No matches found|No files found|Found 0|0 matching|empty result|没有找到",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RunSource:
    label: str
    event_path: Path
    run_dir: Path
    log_path: Path | None
    dump_dir: Path | None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_iso_time(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        return 0.0


def parse_log_time(value: str) -> float:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S,%f").timestamp()
    except ValueError:
        return 0.0


def compact(value: Any, *, max_chars: int = 5000, depth: int = 0) -> Any:
    """Bound event details before embedding them into a standalone report."""
    if depth >= 4:
        return str(value)[:max_chars]
    if isinstance(value, dict):
        result = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 40:
                result["..."] = f"{len(value) - 40} keys omitted"
                break
            result[str(key)] = compact(item, max_chars=max_chars, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        items = [compact(item, max_chars=max_chars, depth=depth + 1) for item in value[:40]]
        if len(value) > 40:
            items.append(f"... {len(value) - 40} items omitted")
        return items
    if isinstance(value, str):
        if len(value) <= max_chars:
            return value
        return value[:max_chars].rstrip() + f"\n... [{len(value) - max_chars} chars omitted]"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:max_chars]


def normalize_path(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").lstrip("./")


def classify_area(*, path: str = "", target: str = "", stage_name: str = "", tool: str = "") -> str:
    text = " ".join((normalize_path(path), normalize_path(target), str(stage_name), str(tool))).lower()
    base = Path(normalize_path(path) or normalize_path(target) or "_").name.lower()

    if any(token in text for token in ("bug_analysis", "static_bug", "bug-analysis")):
        return "Bug分析"
    if "random" in text:
        return "随机测试"
    if any(token in text for token in ("function_coverage", "coverage", "fc_cover", "covergroup", "check_point")):
        return "覆盖率模型"
    if any(token in text for token in ("_api.py", "conftest.py", "fixture", "env_", "environment")):
        return "Env/API"
    if any(token in text for token in ("bundle", "wrapper", "dut_creation", "__init__.py")):
        return "Wrapper/Bundle"
    if base.endswith((".v", ".sv", ".vh", ".svh")) or any(
        token in text for token in ("rtl/", "adder/adder", "dut source")
    ):
        return "RTL/DUT"
    if any(token in text for token in (
        "verification_needs", "functions_and_checks", "specification", "requirement",
        "guide_doc", "readme.md", "functional_specification",
    )):
        return "需求与规范"
    if any(token in text for token in ("test_", "/tests/", "runtestcases", "pytest")):
        return "定向测试"
    if tool in {"Check", "Complete", "RunTestCases", "StdCheck"} or any(
        token in text for token in ("checker", "test_run", "stage_transition")
    ):
        return "验证/Checker"
    if any(token in text for token in (
        "context_reuse", "progress_control", "stage_state", "observation_mask",
        "mutation_budget", "llm", "summary",
    )):
        return "记忆与控制"
    return "其他/基础设施"


def stable_failure_signature(event_type: str, stage_name: str, result: dict[str, Any]) -> str:
    parts = [event_type, stage_name, str(result.get("test_outcome") or "")]
    for key in ("checker_categories", "failed_cases_top", "failed_checkpoints_top", "error_top"):
        value = result.get(key)
        if isinstance(value, list):
            parts.append("|".join(str(item)[:300] for item in value[:4]))
    return hashlib.sha256("\n".join(parts).encode("utf-8", errors="replace")).hexdigest()[:16]


def event_time(event: dict[str, Any]) -> float:
    return safe_float(event.get("time_unix")) or parse_iso_time(event.get("timestamp"))


def iter_structured_events(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            text = line.strip()
            if EVENT_PREFIX in text:
                text = text.split(EVENT_PREFIX, 1)[1]
            if not text.startswith("{"):
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("event_type"):
                yield event


def find_run_dir(event_path: Path) -> Path:
    for parent in (event_path.parent, *event_path.parents):
        if (parent / "ucagent-log.log").is_file() or (parent / "console.log").is_file():
            return parent
    if event_path.parent.name == "unity_test" and event_path.parent.parent.name == "workspace":
        return event_path.parent.parent.parent
    return event_path.parent


def source_from_event_path(event_path: Path, label_hint: str | None = None) -> RunSource:
    event_path = event_path.resolve()
    if event_path.name.endswith(".log"):
        run_dir = event_path.parent
        dump_dir = run_dir / "workspace/unity_test/llm_input_dumps"
        if not dump_dir.is_dir():
            dump_dir = None
        return RunSource(
            label=label_hint or run_dir.name,
            event_path=event_path,
            run_dir=run_dir,
            log_path=event_path,
            dump_dir=dump_dir,
        )
    run_dir = find_run_dir(event_path)
    log_path = run_dir / "ucagent-log.log"
    if not log_path.is_file():
        candidates = [path for path in run_dir.glob("*.log") if "msg" not in path.name]
        log_path = candidates[0] if candidates else None
    dump_dir = event_path.parent / "llm_input_dumps"
    if not dump_dir.is_dir():
        dump_dir = None
    label = label_hint or run_dir.name or event_path.parent.name
    return RunSource(label=label, event_path=event_path, run_dir=run_dir, log_path=log_path, dump_dir=dump_dir)


def parse_source_argument(raw: str) -> tuple[str | None, Path]:
    """Parse ``LABEL=PATH`` without requiring PATH to exist yet.

    Live experiment roots are often created after the visualizer starts.  A
    syntactically path-like right hand side is therefore accepted even when it
    has not appeared on disk yet.
    """
    if "=" in raw:
        maybe_label, maybe_path = raw.split("=", 1)
        if maybe_label.strip() and maybe_path.strip():
            return maybe_label.strip(), Path(maybe_path.strip()).expanduser()
    return None, Path(raw).expanduser()


def discover_sources(paths: list[str], max_traces: int = 0) -> list[RunSource]:
    if not paths:
        paths = [str(REPO_ROOT / "benchmark/ucagent_experiments")]
    candidates: list[tuple[Path, str | None]] = []
    for raw in paths:
        label_hint, path = parse_source_argument(raw)
        if path.is_file():
            if path.name == "structured_events.jsonl":
                candidates.append((path, label_hint))
            elif path.name.endswith(".log"):
                event_candidates = [
                    item for item in path.parent.rglob("structured_events.jsonl")
                    if ".ucagent/history" not in item.as_posix()
                ]
                if event_candidates:
                    candidates.extend((item, label_hint) for item in event_candidates)
                elif path.name == "ucagent-log.log":
                    candidates.append((path, label_hint))
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)
        direct_candidates = [
            path / "workspace/unity_test/structured_events.jsonl",
            path / "unity_test/structured_events.jsonl",
            path / "structured_events.jsonl",
        ]
        found_direct = next((item for item in direct_candidates if item.is_file()), None)
        if found_direct:
            candidates.append((found_direct, label_hint))
            continue
        recursive_events = [
            item for item in path.rglob("structured_events.jsonl")
            if ".ucagent/history" not in item.as_posix()
        ]
        covered_run_dirs = {find_run_dir(item).resolve() for item in recursive_events}
        recursive_logs = [
            item for item in path.rglob("ucagent-log.log")
            if ".ucagent/history" not in item.as_posix()
            and item.parent.resolve() not in covered_run_dirs
        ]
        recursive = recursive_events + recursive_logs
        recursive.sort(key=lambda item: item.stat().st_mtime_ns, reverse=True)
        candidates.extend((item, label_hint) for item in recursive)

    seen = set()
    sources = []
    label_counts = Counter(label for _, label in candidates if label)
    for event_path, label_hint in candidates:
        resolved = event_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        source = source_from_event_path(event_path, label_hint)
        if label_hint and label_counts[label_hint] > 1:
            source = RunSource(
                label=f"{label_hint}/{source.run_dir.name}",
                event_path=source.event_path,
                run_dir=source.run_dir,
                log_path=source.log_path,
                dump_dir=source.dump_dir,
            )
        sources.append(source)
        if max_traces > 0 and len(sources) >= max_traces:
            break
    return sources


def read_run_status(run_dir: Path) -> str:
    """Return the experiment-runner status when a nearby ledger is present."""
    run_id = run_dir.name
    for parent in (run_dir.parent, *run_dir.parents[:3]):
        ledger_path = parent / "ledger.json"
        if not ledger_path.is_file():
            continue
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        record = ledger.get("runs", {}).get(run_id, {})
        status = str(record.get("status") or "").strip()
        if status:
            return status
    if (run_dir / "result_summary.json").is_file():
        return "completed"
    return "unknown"


def stage_info(event: dict[str, Any]) -> tuple[int | None, str, str]:
    for key in ("stage_before", "from_stage"):
        value = event.get(key)
        if isinstance(value, dict) and value.get("stage_index") is not None:
            return value.get("stage_index"), str(value.get("stage_name") or ""), str(value.get("stage_title") or "")
    index = event.get("stage_index")
    return index if isinstance(index, int) else None, str(event.get("stage_name") or ""), str(event.get("stage_title") or "")


def make_action(
    *,
    action_id: str,
    time_unix: float,
    event_type: str,
    kind: str,
    tool: str,
    label: str,
    stage_index: int | None,
    stage_name: str,
    stage_title: str,
    area: str,
    verdict: str = "success",
    path: str = "",
    target: str = "",
    duration_ms: float = 0.0,
    failure_signature: str = "",
    detail: Any = None,
    timing_quality: str = "exact",
) -> dict[str, Any]:
    return {
        "id": action_id,
        "time_unix": round(time_unix, 3),
        "timestamp": datetime.fromtimestamp(time_unix).isoformat(timespec="seconds") if time_unix else "",
        "event_type": event_type,
        "kind": kind,
        "tool": tool,
        "label": label,
        "stage_index": stage_index,
        "stage_name": stage_name,
        "stage_title": stage_title,
        "area": area if area in AREA_ORDER else "其他/基础设施",
        "verdict": verdict,
        "path": normalize_path(path),
        "target": str(target or "").strip(),
        "duration_ms": round(max(0.0, duration_ms), 1),
        "failure_signature": failure_signature,
        "turn_type": "",
        "turn_reason": "",
        "timing_quality": timing_quality,
        "detail": compact(detail or {}),
    }


def structured_actions(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, tuple[str, str]], dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    stage_map: dict[int, tuple[str, str]] = {}
    instrumentation = Counter()
    previous_policy: dict[int | None, str] = {}

    for sequence, event in enumerate(events):
        event_type = str(event.get("event_type") or "unknown")
        stage_index, stage_name, stage_title = stage_info(event)
        if stage_index is not None:
            stage_map[stage_index] = (stage_name, stage_title)
        timestamp = event_time(event)
        action_id = f"s{sequence:06d}"
        result = event.get("result") if isinstance(event.get("result"), dict) else {}
        success = bool(event.get("success"))

        if event_type == "file_mutation":
            path = str(event.get("path") or "")
            verdict = "success" if success else "failure"
            actions.append(make_action(
                action_id=action_id,
                time_unix=timestamp,
                event_type=event_type,
                kind="mutation",
                tool=str(event.get("tool") or "mutation"),
                label=f"{event.get('operation') or 'edit'} {Path(path).name or path}",
                stage_index=stage_index,
                stage_name=stage_name,
                stage_title=stage_title,
                area=classify_area(path=path, stage_name=stage_name, tool=str(event.get("tool") or "")),
                verdict=verdict,
                path=path,
                detail=event,
            ))
        elif event_type == "test_run":
            target = str(event.get("target") or "")
            outcome = str(result.get("test_outcome") or "")
            tests_total = result.get("tests_total")
            if outcome:
                valid_pass = outcome == "pass"
            else:
                # Legacy traces may not have test_outcome. Even there, a
                # zero-test invocation is never evidence of a valid pass.
                valid_pass = bool(success or result.get("tests_passed_all") is True)
                if tests_total is not None and safe_int(tests_total) <= 0:
                    valid_pass = False
            verdict = "success" if valid_pass else "failure"
            total = tests_total
            failed = result.get("tests_failed")
            suffix = ""
            if total is not None:
                suffix = f" ({safe_int(total) - safe_int(failed)}/{safe_int(total)} pass)"
            if outcome and outcome != "pass":
                suffix += f" [{outcome}]"
            signature = "" if valid_pass else stable_failure_signature(event_type, stage_name, result)
            actions.append(make_action(
                action_id=action_id,
                time_unix=timestamp,
                event_type=event_type,
                kind="test",
                tool=str(event.get("tool") or "RunTestCases"),
                label=f"Test {Path(target.split('::', 1)[0]).name or '<all>'}{suffix}",
                stage_index=stage_index,
                stage_name=stage_name,
                stage_title=stage_title,
                area=classify_area(target=target, stage_name=stage_name, tool="RunTestCases"),
                verdict=verdict,
                target=target,
                duration_ms=safe_float(event.get("duration_ms")),
                failure_signature=signature,
                detail=event,
            ))
        elif event_type == "check_result":
            check_pass = result.get("check_pass") is True or success
            verdict = "success" if check_pass else "failure"
            tool = str(event.get("tool") or "Check")
            categories = result.get("checker_categories") if isinstance(result.get("checker_categories"), list) else []
            label = tool + (" pass" if check_pass else " fail")
            if categories:
                label += f" [{categories[0]}]"
            signature = "" if check_pass else stable_failure_signature(event_type, stage_name, result)
            actions.append(make_action(
                action_id=action_id,
                time_unix=timestamp,
                event_type=event_type,
                kind="checker",
                tool=tool,
                label=label,
                stage_index=stage_index,
                stage_name=stage_name,
                stage_title=stage_title,
                area="验证/Checker",
                verdict=verdict,
                duration_ms=safe_float(event.get("duration_ms")),
                failure_signature=signature,
                detail=event,
            ))
        elif event_type == "stage_transition" and (event.get("advanced") or event.get("all_completed")):
            to_stage = event.get("to_stage") if isinstance(event.get("to_stage"), dict) else {}
            to_index = to_stage.get("stage_index")
            label = "Mission complete" if event.get("all_completed") else f"Stage {stage_index} -> {to_index}"
            action = make_action(
                action_id=action_id,
                time_unix=timestamp,
                event_type=event_type,
                kind="milestone",
                tool=str(event.get("trigger") or "Complete"),
                label=label,
                stage_index=stage_index,
                stage_name=stage_name,
                stage_title=stage_title,
                area="验证/Checker",
                verdict="milestone",
                detail=event,
            )
            action["turn_type"] = "stage"
            action["turn_reason"] = "阶段推进"
            actions.append(action)
        elif event_type == "context_reuse_decision":
            decision = str(event.get("decision") or "miss")
            strategies = event.get("strategies") if isinstance(event.get("strategies"), list) else []
            verdict = "success" if decision == "inject" else "empty" if decision in {"miss", "throttled"} else "info"
            actions.append(make_action(
                action_id=action_id,
                time_unix=timestamp,
                event_type=event_type,
                kind="memory",
                tool="ContextReuse",
                label=f"Memory {decision}" + (f" ({len(strategies)})" if strategies else ""),
                stage_index=stage_index,
                stage_name=stage_name,
                stage_title=stage_title,
                area="记忆与控制",
                verdict=verdict,
                failure_signature=str((event.get("failure") or {}).get("signature") or ""),
                detail=event,
            ))
        elif event_type == "progress_control_decision":
            policy = str(event.get("policy") or (event.get("control") or {}).get("state") or "unknown")
            credit = str(event.get("credit_role") or "")
            # Keep policy changes, pivots, regressions, and invalid observations. Repeated
            # identical controller emissions remain available in aggregate metrics only.
            keep = previous_policy.get(stage_index) != policy or policy in {"pivot", "contain", "repair_execution"}
            previous_policy[stage_index] = policy
            instrumentation[event_type] += 1
            if not keep:
                continue
            verdict = "pivot" if policy == "pivot" else "failure" if credit in {"regression", "invalid"} else "info"
            action = make_action(
                action_id=action_id,
                time_unix=timestamp,
                event_type=event_type,
                kind="control",
                tool="ProgressController",
                label=f"Policy {policy} ({credit or 'n/a'})",
                stage_index=stage_index,
                stage_name=stage_name,
                stage_title=stage_title,
                area="记忆与控制",
                verdict=verdict,
                failure_signature=str((event.get("observation_after") or {}).get("signature") or ""),
                detail=event,
            )
            if policy == "pivot":
                action["turn_type"] = "pivot"
                action["turn_reason"] = "进展控制器要求切换修复方向"
            actions.append(action)
        elif event_type == "mutation_budget_blocked":
            action = make_action(
                action_id=action_id,
                time_unix=timestamp,
                event_type=event_type,
                kind="control",
                tool=str(event.get("tool") or "MutationGuard"),
                label=f"Mutation blocked: {Path(str(event.get('path') or '')).name}",
                stage_index=stage_index,
                stage_name=stage_name,
                stage_title=stage_title,
                area=classify_area(path=str(event.get("path") or ""), stage_name=stage_name),
                verdict="blocked",
                path=str(event.get("path") or ""),
                detail=event,
            )
            action["turn_type"] = "dead_end"
            action["turn_reason"] = "修改预算耗尽，当前路线被阻断"
            actions.append(action)
        else:
            instrumentation[event_type] += 1

    return actions, stage_map, {"instrumentation_counts": dict(instrumentation)}


def dump_sort_key(path: Path) -> int:
    match = REQUEST_INDEX_RE.search(path.name)
    return int(match.group(1)) if match else 0


def tool_verdict(tool: str, status: Any, content: str) -> str:
    normalized_status = str(status or "").lower()
    if normalized_status in {"error", "failed", "failure"}:
        return "failure"
    if tool in SEARCH_TOOLS and EMPTY_RESULT_RE.search(content[:1200]):
        return "empty"
    # Successful reads and searches often return source text or logs that
    # contain words such as ERROR/FAILED. Those are observations, not failed
    # tool executions, so an explicit successful status takes precedence.
    if normalized_status in {"success", "completed", "ok"}:
        return "success"
    head_tail = content[:500] + "\n" + content[-1200:]
    if ERROR_HEAD_RE.search(head_tail):
        return "failure"
    return "success"


def summarize_tool_label(tool: str, args: dict[str, Any]) -> tuple[str, str, str]:
    path = str(args.get("path") or args.get("directory") or args.get("source_path") or "")
    target = str(args.get("target") or "")
    if tool == "ReadTextFile":
        return f"Read {Path(path).name or path}", path, target
    if tool == "SearchText":
        pattern = str(args.get("pattern") or "")
        return f"Search {pattern[:60]!r}", path, target
    if tool == "FindFiles":
        return f"Find {str(args.get('pattern') or '')[:60]!r}", path, target
    if tool == "PathList":
        return f"List {path or '.'}", path, target
    return tool, path, target


def recover_read_actions(
    dump_dir: Path | None,
    stage_map: dict[int, tuple[str, str]],
    *,
    max_detail_chars: int = 5000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if dump_dir is None or not dump_dir.is_dir():
        return [], {"available": False, "timing_quality": "unavailable", "files_scanned": 0}
    dump_paths = sorted(
        (path for path in dump_dir.glob("req_*_stage_*.json") if path.name != "latest.json"),
        key=dump_sort_key,
    )
    call_args: dict[str, tuple[str, dict[str, Any], str]] = {}
    emitted: set[str] = set()
    actions = []
    parse_errors = 0
    for dump_path in dump_paths:
        try:
            payload = json.loads(dump_path.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, json.JSONDecodeError):
            parse_errors += 1
            continue
        created_at = parse_iso_time(payload.get("created_at")) or dump_path.stat().st_mtime
        stage_index = safe_int(payload.get("stage"), -1)
        stage_index_value = stage_index if stage_index >= 0 else None
        stage_name, stage_title = stage_map.get(stage_index, ("", ""))
        messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
        for message in messages:
            if not isinstance(message, dict):
                continue
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                call_id = str(call.get("id") or "")
                tool = str(call.get("name") or "")
                args = call.get("args") if isinstance(call.get("args"), dict) else {}
                if call_id and call_id not in call_args:
                    call_args[call_id] = (tool, args, str(message.get("content") or "")[:max_detail_chars])
            if message.get("type") != "tool":
                continue
            call_id = str(message.get("tool_call_id") or "")
            if not call_id or call_id in emitted or call_id not in call_args:
                continue
            tool, args, reasoning = call_args[call_id]
            emitted.add(call_id)
            if tool in STRUCTURED_TOOL_NAMES:
                continue
            content = str(message.get("content") or "")
            verdict = tool_verdict(tool, message.get("status"), content)
            label, path, target = summarize_tool_label(tool, args)
            kind = "read" if tool in READ_TOOLS else "search" if tool in SEARCH_TOOLS else "tool"
            actions.append(make_action(
                action_id=f"t{len(actions):06d}",
                time_unix=created_at + (len(actions) % 1000) / 1000000.0,
                event_type="tool_observation_recovered",
                kind=kind,
                tool=tool,
                label=label,
                stage_index=stage_index_value,
                stage_name=stage_name,
                stage_title=stage_title,
                area=classify_area(path=path, target=target, stage_name=stage_name, tool=tool),
                verdict=verdict,
                path=path,
                target=target,
                detail={
                    "args": compact(args, max_chars=max_detail_chars),
                    "result": compact(content, max_chars=max_detail_chars),
                    "reasoning_summary": compact(reasoning, max_chars=max_detail_chars),
                    "recovered_from": str(dump_path),
                    "timing_note": "Approximate: placed at the next llm_input_dump creation time.",
                },
                timing_quality="next_request_approximation",
            ))
    return actions, {
        "available": bool(dump_paths),
        "timing_quality": "next_request_approximation",
        "files_scanned": len(dump_paths),
        "tool_calls_recovered": len(actions),
        "parse_errors": parse_errors,
    }


def parse_key_value(line: str, key: str) -> str | None:
    match = re.search(rf"(?:^|\s){re.escape(key)}=([^\s]+)", line)
    return match.group(1).strip("'\"") if match else None


def parse_llm_actions(log_path: Path | None, stage_map: dict[int, tuple[str, str]]) -> list[dict[str, Any]]:
    if log_path is None or not log_path.is_file():
        return []
    actions = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if LLM_REQUEST_MARKER not in line:
                continue
            timestamp_match = LOG_TIMESTAMP_RE.match(line)
            if not timestamp_match:
                continue
            end_time = parse_log_time(timestamp_match.group("timestamp"))
            body = line.split(LLM_REQUEST_MARKER, 1)[1]
            stage_index = safe_int(parse_key_value(body, "stage"), -1)
            stage_value = stage_index if stage_index >= 0 else None
            stage_name, stage_title = stage_map.get(stage_index, ("", ""))
            latency_ms = safe_float(parse_key_value(body, "latency_ms"))
            status = str(parse_key_value(body, "status") or "unknown")
            prompt_tokens = safe_int(parse_key_value(body, "prompt_tokens"))
            completion_tokens = safe_int(parse_key_value(body, "completion_tokens"))
            role = str(parse_key_value(body, "role") or "main")
            model = str(parse_key_value(body, "model") or "")
            ttft = parse_key_value(body, "ttft_ms")
            ttft_ms = None if ttft in (None, "NA") else safe_float(ttft)
            actions.append(make_action(
                action_id=f"l{len(actions):06d}",
                time_unix=end_time,
                event_type="llm_request",
                kind="llm",
                tool="LLM",
                label=f"LLM {role}: {prompt_tokens}->{completion_tokens} tok",
                stage_index=stage_value,
                stage_name=stage_name,
                stage_title=stage_title,
                area="记忆与控制",
                verdict="success" if status == "success" else "failure",
                duration_ms=latency_ms,
                detail={
                    "role": role,
                    "model": model,
                    "status": status,
                    "latency_ms": latency_ms,
                    "ttft_ms": ttft_ms,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "prefill_tps_est": parse_key_value(body, "prefill_tps_est"),
                    "decode_tps_est": parse_key_value(body, "decode_tps_est"),
                    "token_source": parse_key_value(body, "token_source"),
                },
            ))
    return actions


MESSAGE_SECTION_RE = re.compile(
    r"^=+\s+(Human Message|AI Message|Tool Message)\s+=+\s*$",
    re.MULTILINE,
)
LEGACY_TOOL_CALL_RE = re.compile(
    r"^  (?P<tool>[A-Za-z][A-Za-z0-9_]*) \((?P<call_id>[^)]+)\)\s*$",
    re.MULTILINE,
)


def load_baseline_token_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "baseline_token_usage.jsonl"
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("event") == "llm_request":
                rows.append(row)
    return rows


def _legacy_stage_name(title: str) -> str:
    match = re.match(r"[^-]+-([A-Za-z0-9_]+)(?:-|$)", title)
    return match.group(1) if match else ""


def parse_legacy_stage_history(
    log_path: Path | None,
) -> tuple[dict[int, tuple[str, str]], list[tuple[float, int]]]:
    """Recover the authoritative Stage map and transition timeline from upstream logs."""
    if log_path is None or not log_path.is_file():
        return {}, []

    stage_map: dict[int, tuple[str, str]] = {}
    timeline: list[tuple[float, int]] = []
    latest_time = 0.0
    reading_stage_list = False
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            timestamp_match = LOG_TIMESTAMP_RE.match(line)
            if timestamp_match:
                latest_time = parse_log_time(timestamp_match.group("timestamp"))
            if " - INFO - Stages:" in line:
                reading_stage_list = True
                continue
            if reading_stage_list:
                stage_match = re.match(r"^\s*(\d+):\s+(.+?)\s*$", line)
                if stage_match:
                    stage_index = int(stage_match.group(1))
                    stage_title = stage_match.group(2).strip()
                    stage_map[stage_index] = (_legacy_stage_name(stage_title), stage_title)
                    continue
                if timestamp_match:
                    reading_stage_list = False

            transition_match = re.search(
                r"Current stage index is (?:now )?(\d+)",
                line,
            )
            if not transition_match or latest_time <= 0:
                continue
            stage_index = int(transition_match.group(1))
            if timeline and stage_index <= timeline[-1][1]:
                continue
            timeline.append((latest_time, stage_index))

    return stage_map, timeline


def legacy_stage_context_at(
    timestamp: float,
    stage_timeline: list[tuple[float, int]],
    stage_map: dict[int, tuple[str, str]],
) -> dict[str, Any]:
    stage_index: int | None = None
    for stage_started_at, candidate in stage_timeline:
        if stage_started_at > timestamp:
            break
        stage_index = candidate
    stage_name, stage_title = stage_map.get(stage_index, ("", ""))
    return {
        "stage_index": stage_index,
        "stage_name": stage_name,
        "stage_title": stage_title,
    }


def legacy_stage_actions(
    stage_timeline: list[tuple[float, int]],
    stage_map: dict[int, tuple[str, str]],
) -> list[dict[str, Any]]:
    actions = []
    for index, (timestamp, stage_index) in enumerate(stage_timeline):
        stage_name, stage_title = stage_map.get(stage_index, ("", ""))
        actions.append(make_action(
            action_id=f"s{index:06d}",
            time_unix=timestamp,
            event_type="legacy_stage_started",
            kind="milestone",
            tool="StageManager",
            label=f"Stage {stage_index} started",
            stage_index=stage_index,
            stage_name=stage_name,
            stage_title=stage_title,
            area="验证/Checker",
            verdict="milestone",
            detail={"source": "ucagent-log.log"},
            timing_quality="measured",
        ))
    return actions


def legacy_tool_result_verdict(tool: str, content: str) -> str:
    lower = content.lower()
    if tool in {"Check", "Complete"}:
        if re.search(r"check_pass\s*[:=]\s*false|check pass(?:ed)?\s*[:=]\s*false", lower):
            return "failure"
        if re.search(r"check_pass\s*[:=]\s*true|check pass(?:ed)?\s*[:=]\s*true", lower):
            return "success"
    if tool == "RunTestCases" and re.search(r"(?:^|\s)failed(?:\s|$)|\b\d+\s+failed\b", lower):
        return "failure"
    status = "error" if ERROR_HEAD_RE.search(content[:600] + "\n" + content[-1600:]) else "success"
    return tool_verdict(tool, status, content)


def parse_legacy_message_actions(
    run_dir: Path,
    token_rows: list[dict[str, Any]],
    *,
    max_detail_chars: int,
) -> tuple[list[dict[str, Any]], dict[int, tuple[str, str]], list[dict[str, Any]], str, dict[str, Any]]:
    """Recover upstream actions from the human-readable message transcript.

    Upstream UCAgent does not emit structured events. Tool timestamps are
    approximated by the corresponding LLM completion timestamp and are marked
    accordingly in the payload.
    """
    message_path = run_dir / "ucagent-msg.log"
    if not message_path.is_file():
        return [], {}, [], "", {"available": False, "timing_quality": "unavailable", "files_scanned": 0}
    text = message_path.read_text(encoding="utf-8", errors="ignore")
    matches = list(MESSAGE_SECTION_RE.finditer(text))
    sections = [
        (match.group(1), text[match.end():(matches[index + 1].start() if index + 1 < len(matches) else len(text))])
        for index, match in enumerate(matches)
    ]
    actions: list[dict[str, Any]] = []
    request_contexts: list[dict[str, Any]] = []
    stage_map: dict[int, tuple[str, str]] = {}
    pending_by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    current_stage: int | None = None
    current_name = ""
    current_title = ""
    dut = ""

    for section_type, content in sections:
        if section_type == "Human Message":
            stage_match = re.search(r"current_stage:\s*\n\s*index:\s*['\"]?(\d+)", content)
            title_match = re.search(r"^\s*title:\s*['\"]?(.+?)['\"]?\s*$", content, re.MULTILINE)
            mission_match = re.search(r"^\s*mission:\s*([^\n]+?)芯片验证任务\s*$", content, re.MULTILINE)
            if stage_match:
                current_stage = int(stage_match.group(1))
            if title_match:
                current_title = title_match.group(1).strip(" '\"")
                name_match = re.match(r"[^-]+-([A-Za-z0-9_]+)(?:-|$)", current_title)
                current_name = name_match.group(1) if name_match else ""
            if current_stage is not None:
                stage_map[current_stage] = (current_name, current_title)
            if mission_match:
                dut = mission_match.group(1).strip(" |")
            continue

        if section_type == "AI Message":
            request_index = len(request_contexts)
            context = {
                "stage_index": current_stage,
                "stage_name": current_name,
                "stage_title": current_title,
            }
            request_contexts.append(context)
            row = token_rows[request_index] if request_index < len(token_rows) else {}
            timestamp = parse_iso_time(row.get("timestamp")) or message_path.stat().st_mtime + request_index
            call_matches = list(LEGACY_TOOL_CALL_RE.finditer(content))
            for call_offset, call_match in enumerate(call_matches):
                block_end = call_matches[call_offset + 1].start() if call_offset + 1 < len(call_matches) else len(content)
                block = content[call_match.end():block_end]
                args = {
                    match.group("key"): match.group("value").strip().strip("'\"")
                    for match in re.finditer(
                        r"^\s{4}(?P<key>[A-Za-z_][A-Za-z0-9_]*):\s*(?P<value>.*)$",
                        block,
                        re.MULTILINE,
                    )
                }
                tool = call_match.group("tool")
                label, path, target = summarize_tool_label(tool, args)
                if tool in MUTATION_TOOLS and path:
                    label = f"{tool} {Path(path).name}"
                kind = (
                    "mutation" if tool in MUTATION_TOOLS
                    else "test" if tool == "RunTestCases"
                    else "checker" if tool in {"Check", "Complete"}
                    else "read" if tool in READ_TOOLS
                    else "search" if tool in SEARCH_TOOLS
                    else "tool"
                )
                action = make_action(
                    action_id=f"u{len(actions):06d}",
                    time_unix=timestamp + call_offset / 1000.0,
                    event_type="legacy_tool_call_recovered",
                    kind=kind,
                    tool=tool,
                    label=label,
                    stage_index=current_stage,
                    stage_name=current_name,
                    stage_title=current_title,
                    area=classify_area(path=path, target=target, stage_name=current_name, tool=tool),
                    verdict="success",
                    path=path,
                    target=target,
                    detail={
                        "args": compact(args, max_chars=max_detail_chars),
                        "call_id": call_match.group("call_id"),
                        "recovered_from": str(message_path),
                        "timing_note": "Approximate: aligned to the corresponding LLM completion.",
                    },
                    timing_quality="llm_completion_approximation",
                )
                actions.append(action)
                pending_by_tool[tool].append(action)
            continue

        name_match = re.search(r"^Name:\s*([^\s]+)", content, re.MULTILINE)
        if not name_match:
            continue
        tool = name_match.group(1)
        pending = pending_by_tool.get(tool) or []
        if not pending:
            continue
        action = pending.pop(0)
        action["verdict"] = legacy_tool_result_verdict(tool, content)
        action["detail"]["result"] = compact(content, max_chars=max_detail_chars)

    return actions, stage_map, request_contexts, dut, {
        "available": True,
        "timing_quality": "llm_completion_approximation",
        "files_scanned": 1,
        "tool_calls_recovered": len(actions),
        "parse_errors": 0,
    }


def parse_baseline_token_actions(
    rows: list[dict[str, Any]],
    request_contexts: list[dict[str, Any]],
    *,
    stage_timeline: list[tuple[float, int]] | None = None,
    stage_map: dict[int, tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    actions = []
    for index, row in enumerate(rows):
        timestamp = parse_iso_time(row.get("timestamp"))
        if stage_timeline:
            context = legacy_stage_context_at(timestamp, stage_timeline, stage_map or {})
        else:
            context = request_contexts[index] if index < len(request_contexts) else {}
        status = str(row.get("status") or "unknown")
        prompt_tokens = safe_int(row.get("prompt_tokens"))
        completion_tokens = safe_int(row.get("completion_tokens"))
        role = str(row.get("role") or "").strip().lower()
        if not role:
            # The upstream baseline meter cannot see LangChain's semantic role,
            # but the second instrumented ChatOpenAI instance is the summary model.
            role = "summary" if safe_int(row.get("model_instance")) > 1 else "main"
        actions.append(make_action(
            action_id=f"b{index:06d}",
            time_unix=timestamp,
            event_type="baseline_llm_request",
            kind="llm",
            tool="LLM",
            label=f"LLM {role}: {prompt_tokens}->{completion_tokens} tok",
            stage_index=context.get("stage_index"),
            stage_name=str(context.get("stage_name") or ""),
            stage_title=str(context.get("stage_title") or ""),
            area="记忆与控制",
            verdict="success" if status == "success" else "failure",
            duration_ms=safe_float(row.get("latency_ms")),
            detail={
                "role": role,
                "model": str(row.get("model") or ""),
                "status": status,
                "error": str(row.get("error") or ""),
                "latency_ms": safe_float(row.get("latency_ms")),
                "ttft_ms": safe_float(row.get("ttft_ms")) if row.get("ttft_ms") is not None else None,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "token_source": str(row.get("token_source") or ""),
                "recovered_from": "baseline_token_usage.jsonl",
            },
            timing_quality="measured",
        ))
    return actions


def action_fingerprint(action: dict[str, Any]) -> str:
    locator = action.get("path") or action.get("target") or action.get("failure_signature") or action.get("label")
    normalized = re.sub(r"\d+", "#", str(locator).lower())
    return f"{action.get('tool')}|{normalized[:300]}"


def annotate_turns_and_retries(actions: list[dict[str, Any]]) -> None:
    relevant = [
        action for action in actions
        if action.get("kind") not in {"llm", "memory", "control", "milestone"}
    ]
    last_failure_signature: dict[tuple[Any, str], str] = {}
    previous: dict[str, Any] | None = None
    previous_fingerprint = ""
    previous_was_issue = False

    for action in relevant:
        signature = str(action.get("failure_signature") or "")
        if signature:
            key = (action.get("stage_index"), action.get("kind"))
            if last_failure_signature.get(key) == signature:
                action["verdict"] = "retry"
                action["turn_type"] = "retry"
                action["turn_reason"] = "相同 Stage、工具角色和失败签名再次出现"
            last_failure_signature[key] = signature

        fingerprint = action_fingerprint(action)
        is_issue = action.get("verdict") in {"failure", "blocked", "empty", "retry"}
        if fingerprint == previous_fingerprint and (is_issue or previous_was_issue):
            action["verdict"] = "retry"
            action["turn_type"] = action.get("turn_type") or "retry"
            action["turn_reason"] = action.get("turn_reason") or "相同工具和目标连续调用，且调用簇包含失败/扑空"
        if previous and previous_was_issue and action.get("area") != previous.get("area"):
            action["turn_type"] = action.get("turn_type") or "recovery"
            action["turn_reason"] = action.get("turn_reason") or "失败后切换工程区域"
        previous = action
        previous_fingerprint = fingerprint
        previous_was_issue = is_issue


def human_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def quantile_median(values: list[float]) -> float | None:
    return round(statistics.median(values), 2) if values else None


def active_duration_seconds(times: list[float], idle_gap_seconds: float = 30 * 60) -> float:
    """Sum observable active intervals while excluding resume/offline gaps."""
    ordered = sorted(set(times))
    return round(
        sum(
            right - left
            for left, right in zip(ordered, ordered[1:])
            if 0 <= right - left <= idle_gap_seconds
        ),
        3,
    )


def build_metrics(actions: list[dict[str, Any]], raw_events: list[dict[str, Any]]) -> dict[str, Any]:
    action_times = [safe_float(action.get("time_unix")) for action in actions if action.get("time_unix")]
    start_time = min(action_times) if action_times else 0.0
    end_time = max(action_times) if action_times else 0.0
    primary = [action for action in actions if action.get("kind") not in {"llm", "memory", "control", "milestone"}]
    mutations = [
        action for action in primary
        if action.get("kind") == "mutation"
        and action.get("verdict") == "success"
        and action.get("tool") != "CreateDirectory"
        and str((action.get("detail") or {}).get("operation") or "") != "create_directory"
    ]
    first_mutation_position = None
    investigation_before_mutation = 0
    if mutations and primary:
        first_index = primary.index(mutations[0])
        first_mutation_position = round(100.0 * first_index / max(1, len(primary) - 1), 1)
        investigation_before_mutation = sum(
            action.get("kind") in {"read", "search"} for action in primary[:first_index]
        )
    area_switches = sum(
        left.get("area") != right.get("area")
        for left, right in zip(primary, primary[1:])
    )
    llm = [action for action in actions if action.get("kind") == "llm"]
    prompt_tokens = sum(safe_int((action.get("detail") or {}).get("prompt_tokens")) for action in llm)
    completion_tokens = sum(safe_int((action.get("detail") or {}).get("completion_tokens")) for action in llm)
    latest_llm = max(llm, key=lambda action: safe_float(action.get("time_unix")), default={})
    latest_llm_detail = latest_llm.get("detail") or {}
    ttfts = [
        safe_float((action.get("detail") or {}).get("ttft_ms"))
        for action in llm
        if (action.get("detail") or {}).get("ttft_ms") is not None
    ]
    retries = sum(action.get("verdict") == "retry" for action in actions)
    failures = sum(action.get("verdict") in {"failure", "blocked", "retry"} for action in primary)
    all_failure_events = sum(
        action.get("verdict") in {"failure", "blocked", "retry"} for action in actions
    )
    empty = sum(action.get("verdict") == "empty" for action in actions)
    context_injections = sum(
        action.get("event_type") == "context_reuse_decision"
        and str((action.get("detail") or {}).get("decision")) == "inject"
        for action in actions
    )
    instrumentation = Counter(event.get("event_type") for event in raw_events)
    active_seconds = active_duration_seconds(action_times)
    return {
        "duration_seconds": active_seconds,
        "duration": human_duration(active_seconds),
        "elapsed_span_seconds": round(max(0.0, end_time - start_time), 3),
        "observable_actions": len(primary),
        "all_visual_events": len(actions),
        "reads": sum(action.get("kind") == "read" for action in primary),
        "searches": sum(action.get("kind") == "search" for action in primary),
        "mutations": len(mutations),
        "tests": sum(action.get("kind") == "test" for action in primary),
        "checker_calls": sum(action.get("kind") == "checker" for action in primary),
        "failures": failures,
        "all_failure_events": all_failure_events,
        "llm_errors": sum(
            action.get("kind") == "llm" and action.get("verdict") == "failure"
            for action in actions
        ),
        "control_failure_events": sum(
            action.get("kind") == "control"
            and action.get("verdict") in {"failure", "blocked", "retry"}
            for action in actions
        ),
        "empty_results": empty,
        "blind_retries": retries,
        "area_switches": area_switches,
        "first_mutation_percent": first_mutation_position,
        "investigation_actions_before_first_mutation": investigation_before_mutation,
        "llm_requests": len(llm),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latest_prompt_tokens": safe_int(latest_llm_detail.get("prompt_tokens")),
        "latest_completion_tokens": safe_int(latest_llm_detail.get("completion_tokens")),
        "latest_llm_latency_ms": safe_float(latest_llm_detail.get("latency_ms")),
        "latest_llm_role": str(latest_llm_detail.get("role") or ""),
        "median_ttft_ms": quantile_median(ttfts),
        "context_injections": context_injections,
        "progress_pivots": sum(action.get("verdict") == "pivot" for action in actions),
        "mutation_budget_blocks": instrumentation.get("mutation_budget_blocked", 0),
        "observation_masking_events": instrumentation.get("observation_masking", 0),
        "observation_size_guards": instrumentation.get("observation_size_guard", 0),
    }


def build_stage_metrics(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int | None, list[dict[str, Any]]] = defaultdict(list)
    for action in actions:
        grouped[action.get("stage_index")].append(action)
    rows = []
    for stage_index, stage_actions in sorted(grouped.items(), key=lambda item: (item[0] is None, item[0])):
        times = [safe_float(action.get("time_unix")) for action in stage_actions if action.get("time_unix")]
        primary = [action for action in stage_actions if action.get("kind") not in {"llm", "memory", "control", "milestone"}]
        llm = [action for action in stage_actions if action.get("kind") == "llm"]
        active_seconds = active_duration_seconds(times)
        rows.append({
            "stage_index": stage_index,
            "stage_name": next((str(action.get("stage_name") or "") for action in stage_actions if action.get("stage_name")), ""),
            "stage_title": next((str(action.get("stage_title") or "") for action in stage_actions if action.get("stage_title")), ""),
            "start_time": min(times) if times else 0,
            "end_time": max(times) if times else 0,
            "duration_seconds": active_seconds,
            "duration": human_duration(active_seconds),
            "elapsed_span_seconds": round(max(times) - min(times), 3) if len(times) >= 2 else 0,
            "actions": len(primary),
            "mutations": sum(action.get("kind") == "mutation" for action in primary),
            "tests": sum(action.get("kind") == "test" for action in primary),
            "check_failures": sum(action.get("kind") == "checker" and action.get("verdict") in {"failure", "retry"} for action in primary),
            "failures": sum(action.get("verdict") in {"failure", "blocked", "retry"} for action in stage_actions),
            "retries": sum(action.get("verdict") == "retry" for action in stage_actions),
            "prompt_tokens": sum(safe_int((action.get("detail") or {}).get("prompt_tokens")) for action in llm),
            "completion_tokens": sum(safe_int((action.get("detail") or {}).get("completion_tokens")) for action in llm),
        })
    return rows


def build_failure_loops(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, str, str], list[dict[str, Any]]] = defaultdict(list)
    for action in actions:
        signature = str(action.get("failure_signature") or "")
        if signature and action.get("verdict") in {"failure", "retry", "blocked"}:
            grouped[(action.get("stage_index"), signature, str(action.get("kind")))].append(action)
    loops = []
    for (stage_index, signature, kind), items in grouped.items():
        if len(items) < 2:
            continue
        loops.append({
            "stage_index": stage_index,
            "signature": signature,
            "kind": kind,
            "count": len(items),
            "label": items[-1].get("label"),
            "first_time": items[0].get("time_unix"),
            "last_time": items[-1].get("time_unix"),
            "duration_seconds": round(safe_float(items[-1].get("time_unix")) - safe_float(items[0].get("time_unix")), 3),
        })
    return sorted(loops, key=lambda item: (item["count"], item["duration_seconds"]), reverse=True)[:30]


def build_trace(source: RunSource, *, include_tool_history: bool = True, max_detail_chars: int = 5000) -> dict[str, Any]:
    trace_id = hashlib.sha256(str(source.event_path).encode("utf-8", errors="replace")).hexdigest()[:12]
    raw_events = list(iter_structured_events(source.event_path))
    actions, stage_map, extra = structured_actions(raw_events)
    telemetry_mode = "structured_events" if raw_events else "legacy_log_reconstruction"
    legacy_dut = ""
    token_rows = load_baseline_token_rows(source.run_dir) if not raw_events else []
    if not raw_events:
        log_stage_map, stage_timeline = parse_legacy_stage_history(source.log_path)
        legacy_actions, legacy_stage_map, request_contexts, legacy_dut, legacy_meta = parse_legacy_message_actions(
            source.run_dir,
            token_rows,
            max_detail_chars=max_detail_chars,
        )
        actions.extend(legacy_actions)
        stage_map.update(legacy_stage_map)
        stage_map.update(log_stage_map)
        actions.extend(legacy_stage_actions(stage_timeline, stage_map))
        actions.extend(parse_baseline_token_actions(
            token_rows,
            request_contexts,
            stage_timeline=stage_timeline,
            stage_map=stage_map,
        ))
        legacy_meta["stage_transitions_recovered"] = len(stage_timeline)
        extra["legacy_reconstruction"] = legacy_meta
    recovery_meta = {"available": False, "timing_quality": "disabled", "files_scanned": 0}
    if include_tool_history:
        recovered, recovery_meta = recover_read_actions(
            source.dump_dir,
            stage_map,
            max_detail_chars=max_detail_chars,
        )
        actions.extend(recovered)
    if not token_rows:
        actions.extend(parse_llm_actions(source.log_path, stage_map))
    actions.sort(key=lambda action: (safe_float(action.get("time_unix")), str(action.get("id"))))
    for index, action in enumerate(actions):
        action["id"] = f"{trace_id}-{action['id']}"
        action["index"] = index
        action["progress"] = round(index / max(1, len(actions) - 1), 6)
    annotate_turns_and_retries(actions)

    dut_counts = Counter(str(event.get("dut") or "") for event in raw_events if event.get("dut"))
    dut = dut_counts.most_common(1)[0][0] if dut_counts else legacy_dut
    if not dut and "__seed_" in source.run_dir.name:
        dut = source.run_dir.name.split("__seed_", 1)[0]
    metrics = build_metrics(actions, raw_events)
    latest_time = max((safe_float(action.get("time_unix")) for action in actions), default=0.0)
    latest_action = actions[-1] if actions else {}
    is_upstream_baseline = (
        (source.run_dir / "baseline_token_usage.jsonl").is_file()
        or "multi_dut_upstream_baseline" in source.run_dir.as_posix().lower()
    )
    return {
        "schema_version": 1,
        "trace_id": trace_id,
        "label": source.label,
        "dut": dut,
        "run_dir": str(source.run_dir),
        "event_path": str(source.event_path),
        "log_path": str(source.log_path) if source.log_path else None,
        "dump_dir": str(source.dump_dir) if source.dump_dir else None,
        "raw_event_count": len(raw_events),
        "telemetry_mode": telemetry_mode,
        "experiment_kind": "upstream_baseline" if is_upstream_baseline else "ucagent_candidate",
        "run_status": read_run_status(source.run_dir),
        "current_stage_index": latest_action.get("stage_index"),
        "current_stage_name": latest_action.get("stage_name") or "",
        "last_event_at": (
            datetime.fromtimestamp(latest_time).isoformat(timespec="seconds") if latest_time else ""
        ),
        "idle_seconds": round(max(0.0, time.time() - latest_time), 1) if latest_time else None,
        "actions": actions,
        "metrics": metrics,
        "stages": build_stage_metrics(actions),
        "failure_loops": build_failure_loops(actions),
        "tool_history_recovery": recovery_meta,
        **extra,
    }


def build_payload(
    sources: list[RunSource],
    *,
    title: str,
    include_tool_history: bool = True,
    max_detail_chars: int = 5000,
) -> dict[str, Any]:
    traces = [
        build_trace(
            source,
            include_tool_history=include_tool_history,
            max_detail_chars=max_detail_chars,
        )
        for source in sources
    ]
    return payload_from_traces(traces, title=title)


def payload_from_traces(traces: list[dict[str, Any]], *, title: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "title": title,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "area_order": AREA_ORDER,
        "trace_count": len(traces),
        "traces": traces,
        "methodology": {
            "vertical_axis": "Observable action order, bottom to top; not wall-clock distance.",
            "horizontal_axis": "UCAgent engineering area inferred from tool, path, target, and stage.",
            "read_search_timing": "Recovered read/search calls use the next llm_input_dump creation time and are approximate.",
            "structured_timing": "Structured mutation/test/check/stage timestamps and LLM latency/token fields are measured values.",
            "retry_rule": "Repeated failure signature, or consecutive same tool+target calls where the cluster contains failure/empty result.",
            "failure_metric": "Default failure count covers observable engineering actions; LLM and controller failures are reported separately.",
        },
    }


def payload_index(payload: dict[str, Any]) -> dict[str, Any]:
    """Return all run metadata without transferring every action."""
    indexed = dict(payload)
    indexed["traces"] = []
    for trace in payload.get("traces", []):
        summary = {key: value for key, value in trace.items() if key != "actions"}
        summary["action_count"] = len(trace.get("actions", []))
        summary["actions"] = []
        summary["actions_loaded"] = False
        indexed["traces"].append(summary)
    return indexed


def trace_transport(trace: dict[str, Any]) -> dict[str, Any]:
    """Return one selected trace; action details are fetched on demand."""
    transported = dict(trace)
    transported["actions"] = [
        {key: value for key, value in action.items() if key != "detail"}
        for action in trace.get("actions", [])
    ]
    transported["action_count"] = len(transported["actions"])
    transported["actions_loaded"] = True
    return transported


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>UCAgent Trajectory Atlas</title>
<style>
:root{--bg:#f5f7fb;--panel:#fff;--ink:#182230;--muted:#667085;--line:#d9e0ea;--accent:#3157d5;--danger:#dc2626;--shadow:0 10px 30px rgba(30,45,75,.08)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:22px 28px 18px;background:linear-gradient(135deg,#172554,#243b78 58%,#3157d5);color:#fff}
header h1{margin:0 0 7px;font-size:25px;letter-spacing:.2px}header p{margin:0;max-width:1050px;color:#dbe5ff}
.toolbar{position:sticky;top:0;z-index:20;display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:12px 20px;background:rgba(255,255,255,.96);border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}
.toolbar label{display:flex;gap:6px;align-items:center;color:#475467}.toolbar input[type=search],.toolbar select{height:34px;border:1px solid #cbd5e1;border-radius:8px;padding:0 10px;background:#fff}.toolbar input[type=search]{width:230px}
button{border:1px solid #cbd5e1;border-radius:8px;background:#fff;color:#344054;padding:7px 11px;cursor:pointer}button:hover{border-color:#3157d5;color:#3157d5}.primary{background:#3157d5;border-color:#3157d5;color:#fff}.primary:hover{color:#fff;background:#2848b3}
.main{padding:18px 20px 60px}.card{background:var(--panel);border:1px solid #e4e9f1;border-radius:13px;box-shadow:var(--shadow);margin-bottom:16px}.card h2{font-size:16px;margin:0;padding:15px 17px;border-bottom:1px solid #edf0f5}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px;padding:14px}.metric{border:1px solid #e6eaf1;border-radius:10px;padding:11px;background:#fbfcfe}.metric b{display:block;font-size:20px;margin-top:3px}.metric small{color:var(--muted)}
.legend{display:flex;flex-wrap:wrap;gap:12px;padding:11px 16px;border-top:1px solid #edf0f5;color:#475467}.dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:5px}.cross{color:#dc2626;font-weight:800;margin-right:4px}.turn{display:inline-block;width:10px;height:10px;border:2px solid #111827;border-radius:2px;margin-right:5px}
#charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));gap:16px}.trace-card{min-width:0}.trace-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;padding:14px 16px;border-bottom:1px solid #edf0f5}.trace-head h3{margin:0;font-size:16px}.trace-head .sub{font-size:12px;color:var(--muted);word-break:break-all;margin-top:3px}.trace-stats{font-size:12px;color:#475467;text-align:right;white-space:nowrap}
.svg-wrap{height:1000px;overflow:auto;background:linear-gradient(#fff,#fbfcff)}svg.trajectory{display:block;width:100%;min-width:540px;height:980px}.axis-label{font-size:10px;fill:#475467}.stage-label{font-size:9px;fill:#64748b}.node{cursor:pointer}.node:hover{filter:drop-shadow(0 0 4px rgba(49,87,213,.55))}.trajectory-line{fill:none;stroke:#b8c1cf;stroke-width:1.1;opacity:.72}.grid-line{stroke:#e8edf4;stroke-width:1}.stage-band{opacity:.055}.dimmed{opacity:.1}.hidden-node{display:none}
.table-wrap{overflow:auto;max-height:480px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:8px 10px;border-bottom:1px solid #edf0f5;text-align:left;white-space:nowrap}th{position:sticky;top:0;background:#f8fafc;color:#475467;z-index:1}td.wrap{white-space:normal;min-width:260px}
.two-col{display:grid;grid-template-columns:minmax(0,2fr) minmax(320px,1fr);gap:16px}.note{padding:13px 16px;color:#475467}.warn{color:#9a3412;background:#fff7ed;border:1px solid #fed7aa;border-radius:9px;padding:9px 11px;margin:10px 16px}
#drawer{position:fixed;top:0;right:-620px;width:min(620px,94vw);height:100vh;background:#fff;z-index:50;box-shadow:-12px 0 35px rgba(15,23,42,.22);transition:right .2s ease;display:flex;flex-direction:column}#drawer.open{right:0}.drawer-head{display:flex;justify-content:space-between;gap:12px;padding:16px 18px;border-bottom:1px solid #e5e7eb}.drawer-body{overflow:auto;padding:16px 18px}.drawer-grid{display:grid;grid-template-columns:120px 1fr;gap:7px 10px;margin-bottom:15px}.drawer-grid dt{color:var(--muted)}.drawer-grid dd{margin:0;word-break:break-word}pre{white-space:pre-wrap;word-break:break-word;background:#0f172a;color:#dbeafe;border-radius:9px;padding:13px;max-height:55vh;overflow:auto;font-size:11px}
.playback{min-width:190px}.playback input{width:125px}.live{display:none;color:#047857;font-weight:650}.live.on{display:inline}.live.stale{color:#b45309}.issue{color:#b42318;font-weight:650}.good{color:#027a48}.muted{color:var(--muted)}
@media(max-width:900px){.two-col{grid-template-columns:1fr}.toolbar{position:static}.svg-wrap{height:820px}svg.trajectory{height:800px}#charts{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><h1 id="page-title">UCAgent Trajectory Atlas</h1><p>从下向上回放 Agent 的调查、修改、验证、失败与换向。结果是否完成和执行路径是否高效需要分开判断。</p></header>
<div class="toolbar">
  <label>Run <select id="run-filter"><option value="all">全部</option></select></label>
  <label>Stage <select id="stage-filter"><option value="all">全部</option></select></label>
  <label><input id="issues-only" type="checkbox">只看失败/重试</label>
  <label><input id="show-memory" type="checkbox">Memory/Control</label>
  <label><input id="show-llm" type="checkbox">LLM请求</label>
  <label><input id="show-read" type="checkbox" checked>读取/搜索</label>
  <input id="search" type="search" placeholder="搜索工具、路径、失败签名">
  <span class="playback"><button id="play">播放</button> <input id="progress" type="range" min="1" max="100" value="100"><span id="progress-label">100%</span></span>
  <select id="speed"><option value="1">1x</option><option value="10">10x</option><option value="50">50x</option><option value="200">200x</option></select>
  <button id="reset">重置</button><span id="live" class="live">● LIVE</span>
</div>
<main class="main">
  <section class="card"><h2>运行对比摘要</h2><div id="summary-metrics" class="metrics"></div><div id="method-note" class="note"></div><div class="legend">
    <span><i class="dot" style="background:#2563eb"></i>读取</span><span><i class="dot" style="background:#7c3aed"></i>搜索</span><span><i class="dot" style="background:#f59e0b"></i>修改</span><span><i class="dot" style="background:#0891b2"></i>测试</span><span><i class="dot" style="background:#334155"></i>Checker</span><span><i class="dot" style="background:#8b5cf6"></i>Memory/Control</span><span><i class="cross">×</i>失败</span><span><i class="turn"></i>转折/重试/阻断</span>
  </div></section>
  <section id="charts"></section>
  <section class="two-col">
    <div class="card"><h2>Stage 成本与失败</h2><div class="table-wrap"><table><thead><tr><th>Run</th><th>Stage</th><th>耗时</th><th>动作</th><th>修改</th><th>测试</th><th>Checker失败</th><th>重试</th><th>累计输入Token</th></tr></thead><tbody id="stage-table"></tbody></table></div></div>
    <div class="card"><h2>重复失败签名</h2><div class="table-wrap"><table><thead><tr><th>Run</th><th>Stage</th><th>次数</th><th>跨度</th><th>签名/动作</th></tr></thead><tbody id="loop-table"></tbody></table></div></div>
  </section>
</main>
<aside id="drawer"><div class="drawer-head"><div><strong id="drawer-title"></strong><div id="drawer-sub" class="muted"></div></div><button id="drawer-close">关闭</button></div><div class="drawer-body"><dl id="drawer-grid" class="drawer-grid"></dl><pre id="drawer-json"></pre></div></aside>
<script>
const INITIAL_DATA=__TRACE_DATA__;
const LIVE_CONFIG=__LIVE_CONFIG__;
let DATA=INITIAL_DATA, playback=100, timer=null;
const COLORS={read:'#2563eb',search:'#7c3aed',mutation:'#f59e0b',test:'#0891b2',checker:'#334155',memory:'#8b5cf6',control:'#64748b',milestone:'#16a34a',llm:'#ec4899',tool:'#0f766e'};
const ISSUE=new Set(['failure','blocked','retry']);
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=n=>new Intl.NumberFormat('zh-CN').format(Number(n||0));
function metric(label,value,note=''){
  if(label==='总 Token')label='累计 Token';
  if(label==='输入 Token'){
    label='累计输入 Token';
    note=note||'已完成请求之和，非单次输入';
  }
  return `<div class="metric"><small>${esc(label)}</small><b>${esc(value)}</b><small>${esc(note)}</small></div>`;
}
function currentFilters(){return {run:$('run-filter').value,stage:$('stage-filter').value,issues:$('issues-only').checked,memory:$('show-memory').checked,llm:$('show-llm').checked,read:$('show-read').checked,q:$('search').value.trim().toLowerCase()}}
function actionVisible(a,f){if(playback<100&&a.progress*100>playback)return false;if(f.stage!=='all'&&String(a.stage_index)!==f.stage)return false;if(!f.memory&&(a.kind==='memory'||a.kind==='control'))return false;if(!f.llm&&a.kind==='llm')return false;if(!f.read&&(a.kind==='read'||a.kind==='search'))return false;if(f.issues&&!ISSUE.has(a.verdict)&&a.verdict!=='pivot')return false;if(f.q){const blob=[a.label,a.tool,a.path,a.target,a.failure_signature,a.stage_name,a.area,JSON.stringify(a.detail)].join(' ').toLowerCase();if(!blob.includes(f.q))return false}return true}
function summaryTraces(){const run=$('run-filter').value;return run==='all'?DATA.traces:DATA.traces.filter(t=>t.trace_id===run)}
function selectedTraces(){const run=$('run-filter').value;if(run==='all')return LIVE_CONFIG.enabled?[]:DATA.traces;return DATA.traces.filter(t=>t.trace_id===run&&t.actions_loaded!==false)}
function traceKind(t){if(t.experiment_kind==='upstream_baseline')return 'UPSTREAM BASELINE';if(t.telemetry_mode==='legacy_log_reconstruction')return 'LEGACY LOG';return 'STRUCTURED'}
function updateRunFilter(){const old=$('run-filter').value;$('run-filter').innerHTML='<option value="all">全部</option>'+DATA.traces.map(t=>`<option value="${esc(t.trace_id)}">[${esc(traceKind(t))}] ${esc(t.label)} · ${esc(t.dut||'-')}</option>`).join('');if(old==='all'||DATA.traces.some(t=>t.trace_id===old))$('run-filter').value=old;else $('run-filter').value='all'}
function updateStageFilter(){const old=$('stage-filter').value;const stages=new Map();selectedTraces().forEach(t=>t.stages.forEach(s=>{if(s.stage_index!==null)stages.set(s.stage_index,s.stage_name)}));$('stage-filter').innerHTML='<option value="all">全部</option>'+[...stages.entries()].sort((a,b)=>a[0]-b[0]).map(([i,n])=>`<option value="${i}">${i} · ${esc(n)}</option>`).join('');if(old==='all'||stages.has(Number(old)))$('stage-filter').value=old;else $('stage-filter').value='all'}
function renderSummary(){
  const traces=summaryTraces();
  const total=traces.reduce((o,t)=>(o.actions+=t.metrics.observable_actions,o.fail+=t.metrics.failures,o.retry+=t.metrics.blind_retries,o.tokens+=t.metrics.total_tokens,o),{actions:0,fail:0,retry:0,tokens:0});
  const active=traces.filter(t=>t.run_status==='running');
  const newest=traces.reduce((a,t)=>a===null||Number(t.idle_seconds)<a?Number(t.idle_seconds):a,null);
  const selected=traces.length===1?traces[0]:null;
  const latest=selected?metric(
    '最近完成 Prompt',
    fmt(selected.metrics.latest_prompt_tokens),
    `${selected.metrics.latest_llm_role||'llm'} · ${Math.round(Number(selected.metrics.latest_llm_latency_ms||0)/1000)}s`
  ):'';
  $('page-title').textContent=DATA.title;
  $('summary-metrics').innerHTML=metric('当前轨迹',traces.length,`已索引 ${DATA.trace_count}`)+metric('运行中',active.length)+metric('可观察动作',fmt(total.actions))+metric('工程动作失败',fmt(total.fail))+metric('重复失败',fmt(total.retry))+metric('总 Token',fmt(total.tokens))+latest+metric('最后刷新',DATA.generated_at);
  const live=$('live');
  live.classList.toggle('stale',newest!==null&&newest>Math.max(120,LIVE_CONFIG.refreshMs/1000*3));
  live.textContent=newest===null?'● WAITING':(live.classList.contains('stale')?'● STALE':'● LIVE');
  $('method-note').innerHTML=`<b>读图口径：</b>${esc(DATA.methodology.vertical_axis)} 横向区域由路径、工具和 Stage 推断。Token 总量与轨迹卡片中的“累计输入 Token”都是已完成请求之和，不代表当前单次 Prompt；“最近完成 Prompt”才是最近一条已返回请求。读取/搜索动作的时间来自下一次 llm_input dump，属于近似恢复；测试、Checker、修改和 LLM 延迟为采集值。默认失败数不混入 LLM API 错误和后台控制器阻断。`;
}
function nodeMarkup(a,x,y){const color=COLORS[a.kind]||'#64748b';const turn=a.turn_type?`<rect x="${x-7}" y="${y-7}" width="14" height="14" rx="2" fill="white" stroke="#111827" stroke-width="1.4"/>`:'';let mark;if(ISSUE.has(a.verdict)){mark=`<line x1="${x-4}" y1="${y-4}" x2="${x+4}" y2="${y+4}" stroke="#dc2626" stroke-width="2"/><line x1="${x+4}" y1="${y-4}" x2="${x-4}" y2="${y+4}" stroke="#dc2626" stroke-width="2"/>`}else{mark=`<circle cx="${x}" cy="${y}" r="${a.verdict==='milestone'?5:3.5}" fill="${color}" stroke="white" stroke-width="1"/>`}return `<g class="node" data-id="${esc(a.id)}">${turn}${mark}<title>${esc(a.label)} · ${esc(a.area)} · ${esc(a.verdict)}</title></g>`}
function renderTrace(trace){const f=currentFilters(),A=DATA.area_order,W=760,H=960,L=74,R=42,T=95,B=45,plotW=W-L-R,plotH=H-T-B;const visible=trace.actions.filter(a=>actionVisible(a,f));let minP=0,maxP=1;if(f.stage!=='all'){const selected=trace.actions.filter(a=>String(a.stage_index)===f.stage);if(selected.length){minP=Math.min(...selected.map(a=>a.progress));maxP=Math.max(...selected.map(a=>a.progress));if(maxP-minP<.001)maxP=minP+.001}}const py=a=>H-B-((a.progress-minP)/(maxP-minP))*plotH;const px=a=>L+A.indexOf(a.area)*plotW/(A.length-1);const pts=visible.map(a=>`${px(a).toFixed(1)},${py(a).toFixed(1)}`).join(' ');let grid='';A.forEach((name,i)=>{const x=L+i*plotW/(A.length-1);grid+=`<line class="grid-line" x1="${x}" y1="${T}" x2="${x}" y2="${H-B}"/><text class="axis-label" transform="translate(${x-2},${T-9}) rotate(-42)" text-anchor="start">${esc(name)}</text>`});for(let p=0;p<=100;p+=20){const y=H-B-p/100*plotH;grid+=`<line class="grid-line" x1="${L}" y1="${y}" x2="${W-R}" y2="${y}"/><text class="axis-label" x="${L-12}" y="${y+3}" text-anchor="end">${p}%</text>`}let bands='';trace.stages.forEach((s,i)=>{const stageActions=trace.actions.filter(a=>a.stage_index===s.stage_index);if(!stageActions.length)return;let p0=Math.min(...stageActions.map(a=>a.progress)),p1=Math.max(...stageActions.map(a=>a.progress));if(p1<minP||p0>maxP)return;p0=Math.max(minP,p0);p1=Math.min(maxP,p1);const y1=H-B-(p1-minP)/(maxP-minP)*plotH,y0=H-B-(p0-minP)/(maxP-minP)*plotH;bands+=`<rect class="stage-band" x="${L}" y="${y1}" width="${plotW}" height="${Math.max(2,y0-y1)}" fill="${i%2?'#3157d5':'#0f766e'}"/><text class="stage-label" x="${W-R-2}" y="${Math.max(T+9,y1+10)}" text-anchor="end">S${s.stage_index}</text>`});const nodes=visible.map(a=>nodeMarkup(a,px(a),py(a))).join('');const m=trace.metrics;const idle=trace.idle_seconds===null?'-':Math.round(trace.idle_seconds/60)+' min';return `<article class="card trace-card"><div class="trace-head"><div><h3>[${esc(traceKind(trace))}] ${esc(trace.label)}</h3><div class="sub">${esc(trace.dut)} · ${esc(trace.event_path)}</div></div><div class="trace-stats"><b>${esc(trace.run_status)}</b> · Stage ${esc(trace.current_stage_index??'-')}<br>${esc(trace.current_stage_name||'')}<br>最后事件 ${esc(trace.last_event_at||'-')} · idle ${esc(idle)}</div></div><div class="metrics">${metric('运行耗时',m.duration)}${metric('首次文件修改',m.first_mutation_percent===null?'无修改':m.first_mutation_percent+'%','不计建目录')}${metric('区域切换',m.area_switches)}${metric('工程动作失败',m.failures)}${metric('重复失败',m.blind_retries)}${metric('输入 Token',fmt(m.prompt_tokens))}</div><div class="svg-wrap"><svg class="trajectory" viewBox="0 0 ${W} ${H}" data-trace="${esc(trace.trace_id)}">${bands}${grid}<polyline class="trajectory-line" points="${pts}"/>${nodes}<text class="axis-label" x="16" y="${T+plotH/2}" transform="rotate(-90 16 ${T+plotH/2})" text-anchor="middle">动作进度：从下到上</text></svg></div><div class="note">显示 ${visible.length}/${trace.actions.length} 个事件。遥测来源：${esc(trace.telemetry_mode)}；读取轨迹恢复：${esc(trace.tool_history_recovery.timing_quality)}。后台另有 ${m.control_failure_events} 个控制失败/阻断、${m.llm_errors} 个 LLM 请求错误。 <button class="export-svg" data-trace="${esc(trace.trace_id)}">导出 SVG</button></div></article>`}
function renderTables(){let stages='',loops='';selectedTraces().forEach(t=>{t.stages.forEach(s=>{stages+=`<tr><td>${esc(t.label)}</td><td>${esc(String(s.stage_index))} · ${esc(s.stage_name)}</td><td>${esc(s.duration)}</td><td>${fmt(s.actions)}</td><td>${fmt(s.mutations)}</td><td>${fmt(s.tests)}</td><td class="${s.check_failures?'issue':''}">${fmt(s.check_failures)}</td><td>${fmt(s.retries)}</td><td>${fmt(s.prompt_tokens)}</td></tr>`});t.failure_loops.forEach(l=>{loops+=`<tr><td>${esc(t.label)}</td><td>${esc(l.stage_index)}</td><td class="issue">${fmt(l.count)}</td><td>${esc(Math.round(l.duration_seconds/60)+'m')}</td><td class="wrap"><code>${esc(l.signature)}</code><br>${esc(l.label)}</td></tr>`})});$('stage-table').innerHTML=stages||'<tr><td colspan="9">暂无 Stage 数据</td></tr>';$('loop-table').innerHTML=loops||'<tr><td colspan="5">未检测到重复失败签名</td></tr>'}
function render(){updateRunFilter();updateStageFilter();renderSummary();const traces=selectedTraces();$('charts').innerHTML=traces.length?traces.map(renderTrace).join(''):(LIVE_CONFIG.enabled?'<section class="card"><div class="note">已读取全部运行索引。请在 Run 中选择一条 candidate 或 UPSTREAM BASELINE 轨迹加载动作明细。</div></section>':'');renderTables();bindNodes()}
function findAction(id){for(const t of DATA.traces){const a=t.actions.find(x=>x.id===id);if(a)return [t,a]}return [null,null]}
function bindNodes(){document.querySelectorAll('.node').forEach(node=>node.addEventListener('click',()=>openDrawer(...findAction(node.dataset.id))));document.querySelectorAll('.export-svg').forEach(btn=>btn.addEventListener('click',()=>exportSvg(btn.dataset.trace)))}
async function openDrawer(trace,a){if(!a)return;$('drawer-title').textContent=a.label;$('drawer-sub').textContent=trace.label;const fields={Stage:`${a.stage_index} · ${a.stage_name}`,区域:a.area,类型:`${a.kind} / ${a.event_type}`,工具:a.tool,判定:a.verdict,时间:a.timestamp,耗时:a.duration_ms?`${a.duration_ms} ms`:'-',路径:a.path||a.target||'-',失败签名:a.failure_signature||'-',转折:a.turn_reason||'-',时间质量:a.timing_quality};$('drawer-grid').innerHTML=Object.entries(fields).map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');$('drawer-json').textContent=a.detail===undefined?'正在加载动作详情...':JSON.stringify(a.detail,null,2);$('drawer').classList.add('open');if(LIVE_CONFIG.enabled&&a.detail===undefined){try{const u=`${LIVE_CONFIG.actionUrl}?trace_id=${encodeURIComponent(trace.trace_id)}&action_id=${encodeURIComponent(a.id)}`;const response=await fetch(u,{cache:'no-store'});if(response.ok){const item=await response.json();a.detail=item.detail;$('drawer-json').textContent=JSON.stringify(item.detail,null,2)}}catch(e){$('drawer-json').textContent='动作详情加载失败：'+String(e)}}}
function exportSvg(label){const svg=[...document.querySelectorAll('svg.trajectory')].find(s=>s.dataset.trace===label);if(!svg)return;const clone=svg.cloneNode(true);clone.setAttribute('xmlns','http://www.w3.org/2000/svg');const style=document.createElementNS('http://www.w3.org/2000/svg','style');style.textContent='.axis-label{font:10px sans-serif;fill:#475467}.stage-label{font:9px sans-serif;fill:#64748b}.trajectory-line{fill:none;stroke:#b8c1cf;stroke-width:1.1}.grid-line{stroke:#e8edf4}.stage-band{opacity:.055}';clone.prepend(style);const blob=new Blob([new XMLSerializer().serializeToString(clone)],{type:'image/svg+xml'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=label.replace(/[^\w.-]+/g,'_')+'_trajectory.svg';a.click();setTimeout(()=>URL.revokeObjectURL(url),500)}
function reset(){playback=100;$('progress').value=100;$('progress-label').textContent='100%';$('run-filter').value='all';$('stage-filter').value='all';$('issues-only').checked=false;$('show-memory').checked=false;$('show-llm').checked=false;$('show-read').checked=true;$('search').value='';render()}
function togglePlay(){if(timer){clearInterval(timer);timer=null;$('play').textContent='播放';return}if(playback>=100){playback=1;$('progress').value=1}const speed=Number($('speed').value);$('play').textContent='暂停';timer=setInterval(()=>{playback=Math.min(100,playback+Math.max(.15,speed*.08));$('progress').value=playback;$('progress-label').textContent=Math.round(playback)+'%';render();if(playback>=100)togglePlay()},120)}
async function loadSelectedTrace(){const id=$('run-filter').value;if(!LIVE_CONFIG.enabled||id==='all')return;const current=DATA.traces.find(t=>t.trace_id===id);if(!current||current.actions_loaded!==false)return;try{const response=await fetch(`${LIVE_CONFIG.traceUrl}?id=${encodeURIComponent(id)}`,{cache:'no-store'});if(!response.ok)return;const full=await response.json();const index=DATA.traces.findIndex(t=>t.trace_id===id);if(index>=0)DATA.traces[index]=full}catch(e){console.warn('trace load failed',e)}}
async function refreshLive(){try{const response=await fetch(LIVE_CONFIG.dataUrl,{cache:'no-store'});if(!response.ok)return;const next=await response.json(),loaded=new Map(DATA.traces.filter(t=>t.actions_loaded).map(t=>[t.trace_id,t]));next.traces=next.traces.map(t=>{const old=loaded.get(t.trace_id);return old&&old.last_event_at===t.last_event_at&&old.raw_event_count===t.raw_event_count?{...t,actions:old.actions,actions_loaded:true}:t});DATA=next;await loadSelectedTrace();render()}catch(e){console.warn('live refresh failed',e)}}
$('run-filter').addEventListener('change',async()=>{render();await loadSelectedTrace();render()});['stage-filter','issues-only','show-memory','show-llm','show-read','speed'].forEach(id=>$(id).addEventListener('change',render));$('search').addEventListener('input',render);$('progress').addEventListener('input',e=>{playback=Number(e.target.value);$('progress-label').textContent=Math.round(playback)+'%';render()});$('play').addEventListener('click',togglePlay);$('reset').addEventListener('click',reset);$('drawer-close').addEventListener('click',()=>$('drawer').classList.remove('open'));document.addEventListener('keydown',e=>{if(e.key==='Escape')$('drawer').classList.remove('open')});
render();if(LIVE_CONFIG.enabled){$('live').classList.add('on');setInterval(refreshLive,LIVE_CONFIG.refreshMs)}
</script>
</body></html>'''


def render_html(payload: dict[str, Any], *, live: bool = False, refresh_seconds: float = 15.0) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    live_json = json.dumps({
        "enabled": live,
        "dataUrl": "/api/index",
        "traceUrl": "/api/trace",
        "actionUrl": "/api/action",
        "refreshMs": max(2000, int(refresh_seconds * 1000)),
    })
    return HTML_TEMPLATE.replace("__TRACE_DATA__", data_json).replace("__LIVE_CONFIG__", live_json)


class PayloadCache:
    def __init__(
        self,
        sources: list[RunSource],
        *,
        title: str,
        include_tool_history: bool,
        max_detail_chars: int,
        source_paths: list[str] | None = None,
        max_traces: int = 0,
    ):
        self.sources = sources
        self.source_paths = source_paths
        self.max_traces = max_traces
        self.title = title
        self.include_tool_history = include_tool_history
        self.max_detail_chars = max_detail_chars
        self._trace_cache: dict[str, dict[str, Any]] = {}
        self._trace_fingerprints: dict[str, tuple[Any, ...]] = {}
        self._payload = None
        self._lock = threading.Lock()
        self._last_refresh_monotonic = 0.0

    def _discover(self) -> None:
        if self.source_paths is not None:
            try:
                self.sources = discover_sources(self.source_paths, max_traces=self.max_traces)
            except FileNotFoundError:
                self.sources = []

    def source_fingerprint(self, source: RunSource) -> tuple[Any, ...]:
        values: list[Any] = []
        for path in (source.event_path, source.log_path):
            if path and path.exists():
                stat = path.stat()
                values.extend((str(path), stat.st_mtime_ns, stat.st_size))
        for name in ("ucagent-msg.log", "baseline_token_usage.jsonl"):
            path = source.run_dir / name
            if path.is_file():
                stat = path.stat()
                values.extend((str(path), stat.st_mtime_ns, stat.st_size))
        if self.include_tool_history and source.dump_dir and source.dump_dir.exists():
            latest = source.dump_dir / "latest.json"
            if latest.exists():
                stat = latest.stat()
                values.extend((str(latest), stat.st_mtime_ns, stat.st_size))
            else:
                # Compatibility path for old dumps. Current collectors update
                # latest.json atomically, avoiding a scan over every request.
                dumps = list(source.dump_dir.glob("req_*_stage_*.json"))
                values.append(len(dumps))
                if dumps:
                    newest = max(dumps, key=lambda path: path.stat().st_mtime_ns)
                    stat = newest.stat()
                    values.extend((str(newest), stat.st_mtime_ns, stat.st_size))
        # A shared ledger changes whenever any sibling run advances. Fingerprint
        # only this run's status so one active run does not rebuild every trace.
        values.extend(("run_status", read_run_status(source.run_dir)))
        return tuple(values)

    def get(self, *, min_refresh_interval: float = 0.0) -> dict[str, Any]:
        with self._lock:
            now_monotonic = time.monotonic()
            if (
                self._payload is not None
                and min_refresh_interval > 0
                and now_monotonic - self._last_refresh_monotonic < min_refresh_interval
            ):
                return self._payload
            self._discover()
            active_keys = {str(source.event_path.resolve()) for source in self.sources}
            rebuilt = []
            for source in self.sources:
                key = str(source.event_path.resolve())
                fingerprint = self.source_fingerprint(source)
                if self._trace_fingerprints.get(key) == fingerprint and key in self._trace_cache:
                    continue
                self._trace_cache[key] = build_trace(
                    source,
                    include_tool_history=self.include_tool_history,
                    max_detail_chars=self.max_detail_chars,
                )
                self._trace_fingerprints[key] = fingerprint
                rebuilt.append(key)
            for key in set(self._trace_cache) - active_keys:
                self._trace_cache.pop(key, None)
                self._trace_fingerprints.pop(key, None)
            traces = [
                self._trace_cache[str(source.event_path.resolve())]
                for source in self.sources
                if str(source.event_path.resolve()) in self._trace_cache
            ]
            self._payload = payload_from_traces(traces, title=self.title)
            self._payload["incremental_update"] = {
                "rebuilt_trace_count": len(rebuilt),
                "reused_trace_count": len(traces) - len(rebuilt),
            }
            # Liveness must continue changing even when the event files do not.
            now = time.time()
            self._payload["generated_at"] = datetime.now().isoformat(timespec="seconds")
            for trace in self._payload.get("traces", []):
                latest = parse_iso_time(trace.get("last_event_at"))
                trace["idle_seconds"] = round(max(0.0, now - latest), 1) if latest else None
            self._last_refresh_monotonic = now_monotonic
            return self._payload

    def get_index(self, *, min_refresh_interval: float = 0.0) -> dict[str, Any]:
        return payload_index(self.get(min_refresh_interval=min_refresh_interval))

    def get_trace(self, trace_id: str, *, min_refresh_interval: float = 0.0) -> dict[str, Any] | None:
        payload = self.get(min_refresh_interval=min_refresh_interval)
        for trace in payload.get("traces", []):
            if trace.get("trace_id") == trace_id:
                return trace
        return None


def serve(cache: PayloadCache, *, host: str, port: int, refresh_seconds: float, open_browser: bool) -> None:
    initial_html = render_html(
        cache.get_index(min_refresh_interval=refresh_seconds),
        live=True,
        refresh_seconds=refresh_seconds,
    ).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib API
            parsed = urlparse(self.path)
            route = parsed.path
            query = parse_qs(parsed.query)
            if route in {"/", "/index.html"}:
                body, content_type = initial_html, "text/html; charset=utf-8"
            elif route == "/api/index":
                body = json.dumps(
                    cache.get_index(min_refresh_interval=refresh_seconds),
                    ensure_ascii=False,
                ).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            elif route == "/api/trace":
                trace_id = str(query.get("id", [""])[0])
                trace = cache.get_trace(trace_id, min_refresh_interval=refresh_seconds)
                if trace is None:
                    self.send_error(404, "trace not found")
                    return
                body = json.dumps(trace_transport(trace), ensure_ascii=False).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            elif route == "/api/action":
                trace_id = str(query.get("trace_id", [""])[0])
                action_id = str(query.get("action_id", [""])[0])
                trace = cache.get_trace(trace_id, min_refresh_interval=refresh_seconds)
                action = next(
                    (item for item in (trace or {}).get("actions", []) if item.get("id") == action_id),
                    None,
                )
                if action is None:
                    self.send_error(404, "action not found")
                    return
                body = json.dumps({"detail": action.get("detail")}, ensure_ascii=False).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            elif route == "/api/traces":
                body = json.dumps(
                    cache.get(min_refresh_interval=refresh_seconds),
                    ensure_ascii=False,
                ).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            elif route == "/health":
                body, content_type = b'{"status":"ok"}', "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("[trajectory-viz] " + (fmt % args) + "\n")

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host if host not in {'0.0.0.0', '::'} else '127.0.0.1'}:{port}/"
    print(f"Serving UCAgent trajectory visualizer at {url}")
    print("Press Ctrl-C to stop. The server only reads run artifacts.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping trajectory visualizer.")
    finally:
        server.server_close()


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "benchmark/ucagent_trajectory_viz" / f"trajectory_{timestamp}.html"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize UCAgent execution paths by engineering area, action order, failure, and retry."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Run directories, experiment roots, or structured_events.jsonl files. Use LABEL=PATH to override a label.",
    )
    parser.add_argument("--out", help="Static self-contained HTML output path.")
    parser.add_argument("--title", default="UCAgent 轨迹图：调查、修改、失败与收敛")
    parser.add_argument("--serve", action="store_true", help="Serve a live-refreshing read-only report.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--refresh-seconds", type=float, default=15.0)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--no-tool-history", action="store_true", help="Skip read/search recovery from llm_input_dumps.")
    parser.add_argument("--max-detail-chars", type=int, default=5000)
    parser.add_argument(
        "--max-traces",
        type=int,
        default=0,
        help="Maximum traces to index; 0 indexes every run under the supplied roots (default).",
    )
    parser.add_argument("--summary-json", help="Optional machine-readable trajectory summary output.")
    args = parser.parse_args()

    try:
        sources = discover_sources(args.paths, max_traces=max(0, args.max_traces))
    except FileNotFoundError:
        sources = []
    if not sources and not args.serve:
        parser.error("no UCAgent structured event or legacy run logs found")
    include_tool_history = not args.no_tool_history
    cache = PayloadCache(
        sources,
        title=args.title,
        include_tool_history=include_tool_history,
        max_detail_chars=max(200, args.max_detail_chars),
        source_paths=args.paths if args.serve else None,
        max_traces=max(0, args.max_traces),
    )
    payload = cache.get()
    for trace in payload["traces"]:
        metrics = trace["metrics"]
        print(
            f"{trace['label']}: actions={metrics['observable_actions']} "
            f"failures={metrics['failures']} retries={metrics['blind_retries']} "
            f"first_mutation={metrics['first_mutation_percent']}% duration={metrics['duration']}"
        )

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote trajectory JSON: {summary_path}")

    if args.serve:
        serve(
            cache,
            host=args.host,
            port=args.port,
            refresh_seconds=args.refresh_seconds,
            open_browser=args.open_browser,
        )
        return

    out_path = Path(args.out) if args.out else default_output_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(payload), encoding="utf-8")
    print(f"Wrote trajectory HTML: {out_path.resolve()}")


if __name__ == "__main__":
    main()
