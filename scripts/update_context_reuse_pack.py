#!/usr/bin/env python3
"""Incrementally merge new UCAgent runs into the context-reuse strategy pack.

Inputs can be existing trace-tree JSON files, ``structured_events.jsonl`` files,
``ucagent-log.log`` files, or run directories containing those files. The script
builds trace trees for raw logs/events, extracts compact repair episodes with
``build_context_reuse_pack.py``, and merges them into the cross-DUT pack with
stable deduplication keys.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_context_reuse_pack import build_pack, load_trace, write_markdown
from scripts.build_trace_tree import build_trace, load_events
from scripts.curate_context_reuse_pack import audit_pack, curate_pack, write_markdown as write_curated_markdown


DEFAULT_PACK_JSON = Path("benchmark/ucagent_context_reuse/context_reuse_source_v1.json")
DEFAULT_PACK_MD = Path("benchmark/ucagent_context_reuse/context_reuse_source_v1.md")
DEFAULT_CURATED_PACK_JSON = Path("benchmark/ucagent_context_reuse/context_reuse_v1.json")
DEFAULT_CURATED_PACK_MD = Path("benchmark/ucagent_context_reuse/context_reuse_v1.md")
DEFAULT_BASELINE_PACK_JSON = Path("benchmark/ucagent_context_reuse/context_reuse_v0.json")
DEFAULT_TRACE_DIR = Path("benchmark/ucagent_trace_tree")


def is_trace_tree(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != ".json":
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return data.get("trace_type") == "ucagent_structured_trajectory"


def safe_name(value: Any, fallback: str = "run") -> str:
    text = str(value or "").strip() or fallback
    out = []
    for char in text:
        if char.isalnum() or char in ("-", "_", "."):
            out.append(char)
        else:
            out.append("_")
    return "".join(out).strip("_") or fallback


def trace_name(trace: dict, source: Path, index: int) -> str:
    dut = safe_name(trace.get("dut"), "DUT")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_stem = safe_name(source.stem if source.name else source.parent.name, f"run{index}")
    return f"{dut}_{stamp}_{index:02d}_{source_stem}_trace_tree.json"


def load_or_build_trace(raw_path: Path) -> dict:
    if is_trace_tree(raw_path):
        return load_trace(raw_path)
    events, sources = load_events([str(raw_path)])
    if not events:
        raise ValueError(f"no structured UCAgent events found in {raw_path}")
    return build_trace(events, sources)


def write_trace_outputs(trace: dict, out_path: Path, nodes_jsonl: bool = True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if nodes_jsonl:
        nodes_path = out_path.with_name(out_path.name.replace("_trace_tree.json", "_trace_nodes.jsonl"))
        with nodes_path.open("w", encoding="utf-8") as fout:
            for node in trace.get("nodes", []):
                fout.write(json.dumps(node, ensure_ascii=False, sort_keys=True) + "\n")


def empty_pack() -> dict:
    return {
        "schema_version": 2,
        "pack_type": "ucagent_context_reuse_v1_source",
        "trace_count": 0,
        "dut_list": [],
        "effective_items": [],
        "compression_hints": [],
        "stage_summaries": [],
        "update_history": [],
    }


def load_pack(path: Path) -> dict:
    if not path.exists():
        return empty_pack()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"pack is not a JSON object: {path}")
    data.setdefault("schema_version", 2)
    data.setdefault("pack_type", "ucagent_context_reuse_v1_source")
    data.setdefault("trace_count", 0)
    data.setdefault("dut_list", [])
    data.setdefault("effective_items", [])
    data.setdefault("compression_hints", [])
    data.setdefault("stage_summaries", [])
    data.setdefault("update_history", [])
    return data


def action_key(actions: list[dict]) -> tuple[tuple[Any, Any, Any], ...]:
    return tuple(
        (action.get("operation"), action.get("path"), action.get("path_category"))
        for action in actions
        if isinstance(action, dict)
    )


def effective_key(item: dict) -> tuple[Any, ...]:
    failure = item.get("failure") if isinstance(item.get("failure"), dict) else {}
    return (
        item.get("dut"),
        item.get("stage_index"),
        item.get("stage_name"),
        item.get("failure_pattern") or failure.get("failure_pattern"),
        failure.get("failure_signature"),
        action_key(item.get("actions") if isinstance(item.get("actions"), list) else []),
    )


def hint_key(item: dict) -> tuple[Any, ...]:
    return (
        item.get("dut"),
        item.get("stage_index"),
        item.get("stage_name"),
        item.get("failure_pattern"),
        item.get("failure_signature"),
    )


def stage_summary_key(item: dict) -> tuple[Any, ...]:
    repeats = item.get("failure_signature_repeats") if isinstance(item.get("failure_signature_repeats"), dict) else {}
    return (
        item.get("dut"),
        item.get("stage_index"),
        item.get("stage_name"),
        tuple(sorted(repeats.items())),
    )


def merge_unique(existing: list[dict], incoming: list[dict], key_fn) -> tuple[list[dict], int]:
    merged = list(existing or [])
    seen = {key_fn(item) for item in merged}
    added = 0
    for item in incoming or []:
        key = key_fn(item)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        added += 1
    return merged, added


def merge_packs(base: dict, incoming: dict, source_labels: list[str]) -> tuple[dict, dict]:
    merged = dict(base)
    merged["effective_items"], added_effective = merge_unique(
        base.get("effective_items", []),
        incoming.get("effective_items", []),
        effective_key,
    )
    merged["compression_hints"], added_hints = merge_unique(
        base.get("compression_hints", []),
        incoming.get("compression_hints", []),
        hint_key,
    )
    merged["stage_summaries"], added_stage_summaries = merge_unique(
        base.get("stage_summaries", []),
        incoming.get("stage_summaries", []),
        stage_summary_key,
    )
    added_any = added_effective > 0 or added_hints > 0 or added_stage_summaries > 0
    merged["dut_list"] = sorted(set(base.get("dut_list", [])) | set(incoming.get("dut_list", [])))
    merged["trace_count"] = int(base.get("trace_count", 0) or 0)
    if added_any:
        merged["trace_count"] += int(incoming.get("trace_count", 0) or 0)
    update_entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sources": source_labels,
        "incoming_traces": incoming.get("trace_count", 0),
        "incoming_effective_items": len(incoming.get("effective_items", [])),
        "incoming_compression_hints": len(incoming.get("compression_hints", [])),
        "added_effective_items": added_effective,
        "added_compression_hints": added_hints,
        "added_stage_summaries": added_stage_summaries,
        "added_trace_count": int(incoming.get("trace_count", 0) or 0) if added_any else 0,
    }
    history = list(base.get("update_history", []) or [])
    history.append(update_entry)
    merged["update_history"] = history[-50:]
    stats = dict(update_entry)
    stats.update({
        "total_effective_items": len(merged["effective_items"]),
        "total_compression_hints": len(merged["compression_hints"]),
        "total_stage_summaries": len(merged["stage_summaries"]),
        "dut_list": merged["dut_list"],
        "trace_count": merged["trace_count"],
    })
    return merged, stats


def backup_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_suffix(path.suffix + "." + datetime.now().strftime("%Y%m%d_%H%M%S") + ".bak")
    shutil.copy2(path, backup)
    return backup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Trace tree JSON, structured_events.jsonl, ucagent-log.log, or run directory.")
    parser.add_argument("--pack-json", default=str(DEFAULT_PACK_JSON))
    parser.add_argument("--pack-md", default=str(DEFAULT_PACK_MD))
    parser.add_argument("--curated-pack-json", default=str(DEFAULT_CURATED_PACK_JSON))
    parser.add_argument("--curated-pack-md", default=str(DEFAULT_CURATED_PACK_MD))
    parser.add_argument("--min-quality", type=float, default=0.68)
    parser.add_argument("--baseline-pack-json", default=str(DEFAULT_BASELINE_PACK_JSON))
    parser.add_argument("--no-curate", action="store_true", help="Do not rebuild the validated v1 strategy pack.")
    parser.add_argument("--trace-dir", default=str(DEFAULT_TRACE_DIR))
    parser.add_argument("--no-save-trace", action="store_true", help="Do not persist trace trees built from raw logs/events.")
    parser.add_argument("--no-nodes-jsonl", action="store_true", help="Do not write flat trace_nodes JSONL files.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak copies before overwriting pack files.")
    parser.add_argument("--dry-run", action="store_true", help="Build and merge in memory, but do not write files.")
    args = parser.parse_args()

    traces = []
    trace_outputs = []
    source_labels = []
    trace_dir = Path(args.trace_dir)
    for index, raw in enumerate(args.inputs, start=1):
        raw_path = Path(raw)
        trace = load_or_build_trace(raw_path)
        traces.append(trace)
        source_labels.append(str(raw_path))
        if not args.no_save_trace and not is_trace_tree(raw_path):
            out_path = trace_dir / trace_name(trace, raw_path, index)
            trace_outputs.append(out_path)
            if not args.dry_run:
                write_trace_outputs(trace, out_path, nodes_jsonl=not args.no_nodes_jsonl)

    incoming_pack = build_pack(traces)
    pack_json = Path(args.pack_json)
    pack_md = Path(args.pack_md)
    curated_pack_json = Path(args.curated_pack_json)
    curated_pack_md = Path(args.curated_pack_md)
    base_pack = load_pack(pack_json)
    merged_pack, stats = merge_packs(base_pack, incoming_pack, source_labels)
    curated_pack = None if args.no_curate else curate_pack(merged_pack, min_quality=args.min_quality)
    baseline_pack_path = Path(args.baseline_pack_json)
    if curated_pack is not None and baseline_pack_path.is_file():
        baseline_pack = json.loads(baseline_pack_path.read_text(encoding="utf-8"))
        curated_pack["legacy_baseline_audit"] = audit_pack(baseline_pack)

    backups = []
    if not args.dry_run:
        pack_json.parent.mkdir(parents=True, exist_ok=True)
        pack_md.parent.mkdir(parents=True, exist_ok=True)
        if not args.no_backup:
            paths_to_backup = [pack_json, pack_md]
            if curated_pack is not None:
                paths_to_backup.extend([curated_pack_json, curated_pack_md])
            for path in paths_to_backup:
                backup = backup_file(path)
                if backup is not None:
                    backups.append(str(backup))
        pack_json.write_text(json.dumps(merged_pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_markdown(merged_pack, pack_md)
        if curated_pack is not None:
            curated_pack_json.parent.mkdir(parents=True, exist_ok=True)
            curated_pack_md.parent.mkdir(parents=True, exist_ok=True)
            curated_pack_json.write_text(
                json.dumps(curated_pack, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            write_curated_markdown(curated_pack, curated_pack_md)

    stats.update({
        "pack_json": str(pack_json),
        "pack_md": str(pack_md),
        "curated_pack_json": str(curated_pack_json) if curated_pack is not None else None,
        "curated_pack_md": str(curated_pack_md) if curated_pack is not None else None,
        "curation": curated_pack.get("curation") if curated_pack is not None else None,
        "trace_outputs": [str(path) for path in trace_outputs],
        "backups": backups,
        "dry_run": bool(args.dry_run),
    })
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
