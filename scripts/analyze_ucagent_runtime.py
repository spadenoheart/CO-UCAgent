#!/usr/bin/env python3
"""Summarize UCAgent wall time, stage residency, LLM latency, and failure loops."""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


LINE_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3})")
LLM_REQUEST_RE = re.compile(
    r"\[data_collection\]\[llm_request\].*?"
    r"role=(?P<role>\S+).*?"
    r"stage=(?P<stage>\S+).*?"
    r"status=(?P<status>\S+).*?"
    r"latency_ms=(?P<latency>[0-9.]+|NA).*?"
    r"ttft_ms=(?P<ttft>[0-9.]+|NA).*?"
    r"prompt_tokens=(?P<prompt>\d+).*?"
    r"completion_tokens=(?P<completion>\d+)"
)
TEXT_TRANSITION_RE = re.compile(
    r"Stage (?P<source>\d+) completed successfully\. "
    r"Current stage index is now (?P<target>\d+)"
)
STAGE_DEFINITION_RE = re.compile(r"^\s*(?P<index>\d+):\s+(?P<title>.+?)\s*$")
STRUCTURED_EVENT_MARKER = "[data_collection][structured_event] "


def parse_line_timestamp(line: str) -> datetime | None:
    match = LINE_TIMESTAMP_RE.match(line)
    if not match:
        return None
    return datetime.strptime(match.group(1) + match.group(2), "%Y-%m-%d %H:%M:%S%f")


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def duration_text(seconds: float | None) -> str:
    if seconds is None:
        return "NA"
    hours, remainder = divmod(max(0, int(round(seconds))), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def run_label(raw_path: str) -> str:
    path = Path(raw_path)
    if len(path.parents) >= 2:
        return f"{path.parent.parent.name}/{path.parent.name}"
    return path.parent.name


def subtract_gaps(start: datetime, end: datetime, gaps: list[tuple[datetime, datetime]]) -> float:
    seconds = max(0.0, (end - start).total_seconds())
    for gap_start, gap_end in gaps:
        overlap_start = max(start, gap_start)
        overlap_end = min(end, gap_end)
        if overlap_end > overlap_start:
            seconds -= (overlap_end - overlap_start).total_seconds()
    return max(0.0, seconds)


def discover_logs(paths: Iterable[Path]) -> list[Path]:
    logs: list[Path] = []
    for path in paths:
        path = path.expanduser().resolve()
        if path.is_file():
            logs.append(path)
        elif path.is_dir():
            logs.extend(path.rglob("ucagent-log.log"))
        else:
            raise FileNotFoundError(path)
    return sorted(set(logs))


def analyze_log(path: Path, pause_gap_seconds: float) -> dict[str, Any]:
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    previous_timestamp: datetime | None = None
    raw_gaps: list[tuple[datetime, datetime]] = []
    llm_requests: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    text_transitions: list[dict[str, Any]] = []
    stage_titles: dict[int, str] = {}

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            timestamp = parse_line_timestamp(line)
            if timestamp is not None:
                if first_timestamp is None:
                    first_timestamp = timestamp
                last_timestamp = timestamp
                if (
                    previous_timestamp is not None
                    and (timestamp - previous_timestamp).total_seconds() >= pause_gap_seconds
                ):
                    raw_gaps.append((previous_timestamp, timestamp))
                previous_timestamp = timestamp

            stage_match = STAGE_DEFINITION_RE.match(line)
            if stage_match:
                stage_titles[int(stage_match.group("index"))] = stage_match.group("title")

            llm_match = LLM_REQUEST_RE.search(line)
            if llm_match and timestamp is not None:
                values = llm_match.groupdict()
                stage = int(values["stage"]) if values["stage"].isdigit() else -1
                llm_requests.append(
                    {
                        "timestamp": timestamp,
                        "role": values["role"],
                        "stage_index": stage,
                        "status": values["status"],
                        "latency_seconds": (
                            None
                            if values["latency"] == "NA"
                            else float(values["latency"]) / 1000.0
                        ),
                        "ttft_seconds": (
                            None if values["ttft"] == "NA" else float(values["ttft"]) / 1000.0
                        ),
                        "prompt_tokens": int(values["prompt"]),
                        "completion_tokens": int(values["completion"]),
                    }
                )

            if STRUCTURED_EVENT_MARKER in line:
                try:
                    event = json.loads(line.split(STRUCTURED_EVENT_MARKER, 1)[1])
                except json.JSONDecodeError:
                    continue
                event["_line_timestamp"] = timestamp
                events.append(event)
                stage_index = event.get("stage_index")
                stage_name = event.get("stage_name")
                if isinstance(stage_index, int) and stage_name:
                    stage_titles.setdefault(stage_index, str(stage_name))

            transition_match = TEXT_TRANSITION_RE.search(line)
            if transition_match and timestamp is not None:
                text_transitions.append(
                    {
                        "timestamp": timestamp,
                        "source": int(transition_match.group("source")),
                        "target": int(transition_match.group("target")),
                    }
                )

    wall_seconds = (
        (last_timestamp - first_timestamp).total_seconds()
        if first_timestamp is not None and last_timestamp is not None
        else 0.0
    )
    pause_seconds = sum((end - start).total_seconds() for start, end in raw_gaps)
    active_seconds = max(0.0, wall_seconds - pause_seconds)

    transitions: list[dict[str, Any]] = []
    for event in events:
        if event.get("event_type") != "stage_transition" or not event.get("success"):
            continue
        source = event.get("from_stage") or {}
        target = event.get("to_stage") or {}
        source_index = source.get("stage_index")
        target_index = target.get("stage_index")
        if source_index == target_index or event.get("_line_timestamp") is None:
            continue
        transitions.append(
            {
                "timestamp": event["_line_timestamp"],
                "source": source_index,
                "target": target_index,
            }
        )
    if not transitions:
        transitions = text_transitions
    transitions.sort(key=lambda item: item["timestamp"])

    stage_seconds: collections.Counter[int] = collections.Counter()
    if first_timestamp is not None and last_timestamp is not None and transitions:
        boundary = first_timestamp
        for transition in transitions:
            stage_seconds[transition["source"]] += subtract_gaps(
                boundary,
                transition["timestamp"],
                raw_gaps,
            )
            boundary = transition["timestamp"]
        stage_seconds[transitions[-1]["target"]] += subtract_gaps(
            boundary,
            last_timestamp,
            raw_gaps,
        )

    successful_llm = [
        request
        for request in llm_requests
        if request["status"] == "success" and request["latency_seconds"] is not None
    ]
    main_llm = [
        request
        for request in successful_llm
        if request["role"] == "main"
    ]
    ttft_values = [
        request["ttft_seconds"]
        for request in main_llm
        if request["ttft_seconds"] is not None
    ]
    llm_seconds_by_stage: collections.Counter[int] = collections.Counter()
    llm_requests_by_stage: collections.Counter[int] = collections.Counter()
    for request in successful_llm:
        llm_seconds_by_stage[request["stage_index"]] += request["latency_seconds"]
        llm_requests_by_stage[request["stage_index"]] += 1

    checks = [event for event in events if event.get("event_type") == "check_result"]
    failed_checks = [event for event in checks if not event.get("success")]
    failures_by_stage: collections.Counter[int] = collections.Counter(
        int(event["stage_index"])
        for event in failed_checks
        if isinstance(event.get("stage_index"), int)
    )
    failure_categories: collections.Counter[str] = collections.Counter(
        str(category)
        for event in failed_checks
        for category in ((event.get("result") or {}).get("checker_categories") or [])
    )
    test_runs = [event for event in events if event.get("event_type") == "test_run"]
    test_outcomes: collections.Counter[str] = collections.Counter(
        str((event.get("result") or {}).get("test_outcome", "unknown"))
        for event in test_runs
    )
    test_seconds = sum(float(event.get("duration_ms", 0) or 0) / 1000.0 for event in test_runs)
    reuse_decisions = [
        event for event in events if event.get("event_type") == "context_reuse_decision"
    ]
    reuse_outcomes: collections.Counter[str] = collections.Counter(
        str(event.get("decision", "unknown")) for event in reuse_decisions
    )

    top_stages = []
    for stage_index, seconds in stage_seconds.most_common():
        top_stages.append(
            {
                "stage_index": stage_index,
                "stage_name": stage_titles.get(stage_index, f"stage_{stage_index}"),
                "wall_seconds": seconds,
                "llm_seconds": llm_seconds_by_stage.get(stage_index, 0.0),
                "llm_requests": llm_requests_by_stage.get(stage_index, 0),
                "failed_checks": failures_by_stage.get(stage_index, 0),
            }
        )

    return {
        "path": str(path),
        "first_timestamp": first_timestamp.isoformat() if first_timestamp else None,
        "last_timestamp": last_timestamp.isoformat() if last_timestamp else None,
        "wall_seconds": wall_seconds,
        "inferred_pause_seconds": pause_seconds,
        "active_seconds": active_seconds,
        "inferred_pause_count": len(raw_gaps),
        "llm_request_count": len(llm_requests),
        "main_llm_success_count": len(main_llm),
        "llm_latency_seconds": sum(request["latency_seconds"] for request in successful_llm),
        "llm_active_time_ratio": (
            sum(request["latency_seconds"] for request in successful_llm) / active_seconds
            if active_seconds > 0
            else None
        ),
        "prompt_tokens": sum(request["prompt_tokens"] for request in llm_requests),
        "completion_tokens": sum(request["completion_tokens"] for request in llm_requests),
        "ttft_p50_seconds": percentile(ttft_values, 0.50),
        "ttft_p95_seconds": percentile(ttft_values, 0.95),
        "stage_transition_count": len(transitions),
        "last_stage_index": transitions[-1]["target"] if transitions else None,
        "failed_check_count": len(failed_checks),
        "failure_categories": dict(failure_categories),
        "test_run_count": len(test_runs),
        "test_seconds": test_seconds,
        "test_outcomes": dict(test_outcomes),
        "context_reuse_decisions": dict(reuse_outcomes),
        "top_stages": top_stages,
    }


def markdown_report(results: list[dict[str, Any]], top_n: int) -> str:
    lines = [
        "# UCAgent Runtime Analysis",
        "",
        "The active-time estimate removes log gaps above the configured pause threshold. "
        "It is diagnostic rather than a replacement for the runner ledger.",
        "",
        "| Run | Wall | Inferred pause | Active | Last stage | LLM requests | LLM/active | Failed checks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        ratio = result["llm_active_time_ratio"]
        lines.append(
            "| {run} | {wall} | {pause} | {active} | {stage} | {requests} | {ratio} | {fails} |".format(
                run=run_label(result["path"]),
                wall=duration_text(result["wall_seconds"]),
                pause=duration_text(result["inferred_pause_seconds"]),
                active=duration_text(result["active_seconds"]),
                stage=result["last_stage_index"] if result["last_stage_index"] is not None else "NA",
                requests=result["llm_request_count"],
                ratio=f"{ratio:.1%}" if ratio is not None else "NA",
                fails=result["failed_check_count"],
            )
        )

    for result in results:
        lines.extend(
            [
                "",
                f"## {run_label(result['path'])}",
                "",
                f"- Log: `{result['path']}`",
                f"- Prompt/completion tokens: {result['prompt_tokens']:,} / {result['completion_tokens']:,}",
                "- TTFT p50/p95: {}/{}".format(
                    duration_text(result["ttft_p50_seconds"]),
                    duration_text(result["ttft_p95_seconds"]),
                ),
                f"- Test execution: {result['test_run_count']} runs, {duration_text(result['test_seconds'])}; "
                f"outcomes={json.dumps(result['test_outcomes'], ensure_ascii=False, sort_keys=True)}",
                f"- Failure categories: {json.dumps(result['failure_categories'], ensure_ascii=False, sort_keys=True)}",
                f"- Context-reuse decisions: {json.dumps(result['context_reuse_decisions'], ensure_ascii=False, sort_keys=True)}",
                "",
                "| Stage | Active residency | LLM latency | Requests | Failed checks |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for stage in result["top_stages"][:top_n]:
            lines.append(
                "| {index} {name} | {wall} | {llm} | {requests} | {fails} |".format(
                    index=stage["stage_index"],
                    name=stage["stage_name"],
                    wall=duration_text(stage["wall_seconds"]),
                    llm=duration_text(stage["llm_seconds"]),
                    requests=stage["llm_requests"],
                    fails=stage["failed_checks"],
                )
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Log files or directories to scan")
    parser.add_argument(
        "--pause-gap-seconds",
        type=float,
        default=3600.0,
        help="Treat silent gaps at least this long as an interrupted/offline interval",
    )
    parser.add_argument("--top", type=int, default=8, help="Number of stages shown per run")
    parser.add_argument("--json", type=Path, dest="json_path")
    parser.add_argument("--markdown", type=Path, dest="markdown_path")
    args = parser.parse_args()

    logs = discover_logs(args.paths)
    if not logs:
        raise SystemExit("no ucagent-log.log files found")
    results = [analyze_log(path, args.pause_gap_seconds) for path in logs]
    report = markdown_report(results, args.top)
    print(report)

    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps({"schema_version": 1, "runs": results}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.markdown_path:
        args.markdown_path.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_path.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
