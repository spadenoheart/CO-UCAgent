#!/usr/bin/env python3
"""Inspect CO-UCAgent long-term memory files without loading a model."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


MEMORY_NAMES = {"memory.jsonl", "memory.candidate.jsonl", "metrics.jsonl"}


def discover(paths: Iterable[str]) -> list[Path]:
    found: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            found.update(item for item in path.rglob("*.jsonl") if item.name in MEMORY_NAMES)
        elif path.is_file():
            found.add(path)
    return sorted(found)


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                yield item


def summarize(paths: list[Path]) -> dict:
    types: Counter[str] = Counter()
    stages: Counter[str] = Counter()
    duts: Counter[str] = Counter()
    files: dict[str, int] = {}
    metric_totals: Counter[str] = Counter()

    for path in paths:
        count = 0
        for item in read_jsonl(path):
            count += 1
            if path.name == "metrics.jsonl":
                for key, value in item.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        metric_totals[key] += value
                continue
            types[str(item.get("memory_type") or item.get("type") or "unknown")] += 1
            stages[str(item.get("stage_name") or item.get("stage") or "unknown")] += 1
            duts[str(item.get("dut") or item.get("source_dut") or "unknown")] += 1
        files[str(path)] = count

    return {
        "files": files,
        "memory_entries": sum(count for path, count in files.items() if not path.endswith("metrics.jsonl")),
        "by_type": dict(types.most_common()),
        "by_stage": dict(stages.most_common()),
        "by_dut": dict(duts.most_common()),
        "metric_totals": dict(metric_totals.most_common()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="memory JSONL file(s), workspace, or experiment root")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    paths = discover(args.paths)
    if not paths:
        parser.error("no memory.jsonl, memory.candidate.jsonl, or metrics.jsonl files found")
    summary = summarize(paths)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"files={len(summary['files'])} memory_entries={summary['memory_entries']}")
        for path, count in summary["files"].items():
            print(f"{count:6d}  {path}")
        for name in ("by_type", "by_stage", "by_dut", "metric_totals"):
            if summary[name]:
                print(f"\n{name}:")
                for key, value in summary[name].items():
                    print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
