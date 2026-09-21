#!/usr/bin/env python3
"""Generate a cross-DUT failure pattern report from UCAgent trace trees."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ucagent.memory.context_reuse import classify_failure_pattern


def load_trace(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("trace_type") != "ucagent_structured_trajectory":
        raise ValueError(f"not a trace tree file: {path}")
    return data


def compact(value: Any, limit: int = 500) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text[:limit]


def classify_pattern(node: dict) -> str:
    obs = node.get("observation") if isinstance(node.get("observation"), dict) else {}
    action = node.get("action") if isinstance(node.get("action"), dict) else {}
    return classify_failure_pattern(
        obs,
        stage_name=str(node.get("stage_name", "") or ""),
        event_type=str(node.get("event_type", "") or ""),
        path_category=str(action.get("path_category", "") or ""),
        path=str(action.get("path", "") or ""),
    )


def is_failure_node(node: dict) -> bool:
    if node.get("event_type") == "check_result":
        obs = node.get("observation") or {}
        return obs.get("check_pass") is False or node.get("success") is False
    if node.get("event_type") == "test_run":
        obs = node.get("observation") or {}
        return obs.get("tests_passed_all") is False
    return False


def stage_score(stage: dict) -> int:
    return (
        int(stage.get("check_failures", 0) or 0) * 4
        + int(stage.get("test_runs", 0) or 0) * 2
        + int(stage.get("file_mutations", 0) or 0)
        + len(stage.get("failure_signatures", {}) or {}) * 3
    )


def analyze(traces: list[dict]) -> dict:
    pattern_counts = Counter()
    pattern_by_dut = defaultdict(Counter)
    pattern_by_stage = defaultdict(Counter)
    high_stages = []
    failure_examples = defaultdict(list)
    file_category_counts = Counter()
    stage_commonality = defaultdict(lambda: {"duts": set(), "score": 0, "failures": 0, "mutations": 0})

    for trace in traces:
        dut = trace.get("dut", "")
        for stage in trace.get("stages", []):
            score = stage_score(stage)
            if score > 0:
                high_stages.append({
                    "dut": dut,
                    "stage_index": stage.get("stage_index"),
                    "stage_name": stage.get("stage_name"),
                    "score": score,
                    "file_mutations": stage.get("file_mutations", 0),
                    "test_runs": stage.get("test_runs", 0),
                    "check_failures": stage.get("check_failures", 0),
                    "failure_signature_count": len(stage.get("failure_signatures", {}) or {}),
                    "stage_advanced": stage.get("stage_advanced", False),
                    "all_completed": stage.get("all_completed", False),
                })
                bucket = stage_commonality[stage.get("stage_name", "")]
                bucket["duts"].add(dut)
                bucket["score"] += score
                bucket["failures"] += int(stage.get("check_failures", 0) or 0)
                bucket["mutations"] += int(stage.get("file_mutations", 0) or 0)
        for node in trace.get("nodes", []):
            action = node.get("action") if isinstance(node.get("action"), dict) else {}
            if action.get("path_category"):
                file_category_counts[(dut, node.get("stage_name"), action.get("path_category"))] += 1
            if not is_failure_node(node):
                continue
            pattern = classify_pattern(node)
            pattern_counts[pattern] += 1
            pattern_by_dut[dut][pattern] += 1
            pattern_by_stage[(dut, node.get("stage_index"), node.get("stage_name"))][pattern] += 1
            if len(failure_examples[pattern]) < 8:
                obs = node.get("observation") or {}
                reward = node.get("reward") or {}
                failure_examples[pattern].append({
                    "dut": dut,
                    "stage_index": node.get("stage_index"),
                    "stage_name": node.get("stage_name"),
                    "event_type": node.get("event_type"),
                    "tool": obs.get("tool") or reward.get("trigger"),
                    "failed_cases_top": obs.get("failed_cases_top", [])[:3],
                    "failed_checkpoints_top": obs.get("failed_checkpoints_top", [])[:3],
                    "error_top": [compact(item, 180) for item in (obs.get("error_top", [])[:2])],
                    "failure_signature": obs.get("failure_signature"),
                })

    common_stages = []
    for stage_name, item in stage_commonality.items():
        common_stages.append({
            "stage_name": stage_name,
            "dut_count": len(item["duts"]),
            "duts": sorted(item["duts"]),
            "score": item["score"],
            "check_failures": item["failures"],
            "file_mutations": item["mutations"],
        })

    return {
        "trace_count": len(traces),
        "dut_list": sorted({trace.get("dut", "") for trace in traces if trace.get("dut")}),
        "pattern_counts": dict(pattern_counts.most_common()),
        "pattern_by_dut": {dut: dict(counter.most_common()) for dut, counter in sorted(pattern_by_dut.items())},
        "high_failure_stages": sorted(high_stages, key=lambda item: item["score"], reverse=True),
        "common_stages": sorted(common_stages, key=lambda item: (item["dut_count"], item["score"]), reverse=True),
        "pattern_by_stage": {
            " | ".join(str(part) for part in key): dict(counter.most_common())
            for key, counter in sorted(pattern_by_stage.items(), key=lambda item: sum(item[1].values()), reverse=True)
        },
        "file_category_hotspots": {
            " | ".join(str(part) for part in key): value
            for key, value in file_category_counts.most_common(40)
        },
        "failure_examples": dict(failure_examples),
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = []
    lines.append("# UCAgent Failure Pattern Report")
    lines.append("")
    lines.append(f"- Trace count: {report['trace_count']}")
    lines.append(f"- DUTs: {', '.join(report['dut_list'])}")
    lines.append("")
    lines.append("## Pattern Counts")
    for name, count in report["pattern_counts"].items():
        lines.append(f"- {name}: {count}")
    lines.append("")
    lines.append("## High-Failure Stages")
    for item in report["high_failure_stages"][:20]:
        lines.append(
            f"- {item['dut']} stage {item['stage_index']} `{item['stage_name']}`: "
            f"score={item['score']}, mutations={item['file_mutations']}, "
            f"test_runs={item['test_runs']}, check_failures={item['check_failures']}, "
            f"failure_signatures={item['failure_signature_count']}"
        )
    lines.append("")
    lines.append("## Cross-DUT Common Stages")
    for item in report["common_stages"]:
        if item["dut_count"] < 2:
            continue
        lines.append(
            f"- `{item['stage_name']}`: duts={','.join(item['duts'])}, "
            f"score={item['score']}, check_failures={item['check_failures']}, mutations={item['file_mutations']}"
        )
    lines.append("")
    lines.append("## Failure Examples")
    for pattern, examples in report["failure_examples"].items():
        lines.append(f"### {pattern}")
        for ex in examples[:5]:
            err = "; ".join(ex.get("error_top") or [])
            cases = ", ".join(ex.get("failed_cases_top") or [])
            lines.append(
                f"- {ex['dut']} stage {ex['stage_index']} `{ex['stage_name']}` "
                f"{ex['event_type']} sig={ex.get('failure_signature')}; cases={cases}; error={err}"
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_trees", nargs="+", help="Trace tree JSON files from scripts/build_trace_tree.py")
    parser.add_argument("--out-json", default="benchmark/ucagent_failure_patterns/failure_pattern_report.json")
    parser.add_argument("--out-md", default="benchmark/ucagent_failure_patterns/failure_pattern_report.md")
    args = parser.parse_args()

    traces = [load_trace(Path(path)) for path in args.trace_trees]
    report = analyze(traces)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, out_md)
    print(json.dumps({
        "out_json": str(out_json),
        "out_md": str(out_md),
        "pattern_counts": report["pattern_counts"],
        "top_stages": report["high_failure_stages"][:5],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
