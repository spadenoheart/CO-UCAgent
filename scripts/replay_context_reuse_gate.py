#!/usr/bin/env python3
"""Replay recorded context-reuse queries against the current retrieval gates.

This is a counterfactual retrieval audit, not an estimate of downstream agent
quality. It answers which historical injections would be kept or suppressed by
the current store implementation and configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ucagent.memory.context_reuse import ContextReuseStore


def normalize_failure(raw: dict) -> dict:
    failure = dict(raw) if isinstance(raw, dict) else {}
    failure["failure_pattern"] = failure.get("failure_pattern") or failure.get("pattern")
    failure["failure_group"] = failure.get("failure_group") or failure.get("group")
    failure["failure_signature"] = failure.get("failure_signature") or failure.get("signature")
    return failure


def replay(
    decisions: dict,
    *,
    workspace: Path,
    dut: str,
    pack_path: Path,
    min_score: float,
    min_quality_score: float,
    min_utility_score: float,
    prompt_limit: int,
    hint_limit: int,
) -> dict:
    store = ContextReuseStore(str(workspace), dut, {
        "pack_path": str(pack_path),
        "prompt_limit": prompt_limit,
        "compression_hint_limit": hint_limit,
        "min_score": min_score,
        "min_quality_score": min_quality_score,
        "include_cross_dut": True,
        "prefer_same_dut": True,
        "prefer_same_stage": True,
        "strict_stage_role_match": True,
        "enable_failure_pattern_match": True,
        "strict_failure_pattern_match": True,
        "prefer_action_category_match": True,
        "strict_action_category_match": True,
        "strict_exact_pattern_groups": True,
        "generic_requires_signature": True,
        "skip_non_reusable_failures": True,
        "strict_compression_hint_match": True,
        "destructive_requires_support": True,
        "min_destructive_support": 2,
        "enable_utility_gate": True,
        "min_utility_score": min_utility_score,
        "utility_unknown_policy": "reject",
        "utility_prompt_token_reference": 512,
        "utility_prompt_token_weight": 0.05,
    })

    gate_reasons: Counter[str] = Counter()
    records = []
    original_items = 0
    replay_items = 0
    suppressed_injections = 0
    changed_injections = 0
    for record in decisions.get("records", []):
        if record.get("decision") != "inject":
            continue
        failure = normalize_failure(record.get("failure", {}))
        stage_index = record.get("stage_index")
        stage_name = str(record.get("stage_name") or "")
        episodes = store.search(failure, stage_index, stage_name, limit=prompt_limit)
        summary = store.get_last_search_summary()
        hints = store.compression_hints_for(failure, stage_index, stage_name, limit=hint_limit)
        old_ids = [
            str(item.get("strategy_id"))
            for item in record.get("strategies", [])
            if isinstance(item, dict) and item.get("strategy_id")
        ]
        new_ids = [str(item.get("strategy_id")) for item in episodes if item.get("strategy_id")]
        original_hint_count = max(0, int(record.get("prompt_item_count", 0) or 0) - len(old_ids))
        original_items += len(old_ids) + original_hint_count
        replay_items += len(new_ids) + len(hints)
        for reason, count in (summary.get("hard_gate_reasons") or {}).items():
            gate_reasons[str(reason)] += int(count or 0)
        if not new_ids and not hints:
            suppressed_injections += 1
        if old_ids != new_ids or original_hint_count != len(hints):
            changed_injections += 1
        records.append({
            "stage_index": stage_index,
            "stage_name": stage_name,
            "failure_pattern": failure.get("failure_pattern"),
            "failure_signature": failure.get("failure_signature"),
            "original_strategy_ids": old_ids,
            "replay_strategy_ids": new_ids,
            "original_hint_count": original_hint_count,
            "replay_hint_count": len(hints),
            "hard_gate_reasons": summary.get("hard_gate_reasons", {}),
        })

    injection_count = len(records)
    return {
        "schema_version": 1,
        "analysis_type": "context_reuse_gate_counterfactual_replay",
        "causal_claim": False,
        "pack_path": str(pack_path),
        "injections_replayed": injection_count,
        "changed_injections": changed_injections,
        "suppressed_injections": suppressed_injections,
        "original_prompt_items": original_items,
        "replay_prompt_items": replay_items,
        "prompt_item_reduction": original_items - replay_items,
        "prompt_item_reduction_rate": round(
            (original_items - replay_items) / original_items, 4
        ) if original_items else None,
        "hard_gate_reasons": dict(gate_reasons),
        "records": records,
    }


def write_markdown(result: dict, path: Path) -> None:
    lines = [
        "# Context Reuse Gate Replay",
        "",
        "This is a counterfactual retrieval audit, not a causal agent-quality result.",
        "",
        f"- Injections replayed: {result['injections_replayed']}",
        f"- Changed injections: {result['changed_injections']}",
        f"- Fully suppressed injections: {result['suppressed_injections']}",
        f"- Prompt items: {result['original_prompt_items']} -> {result['replay_prompt_items']}",
        f"- Prompt-item reduction rate: {result['prompt_item_reduction_rate']}",
        f"- Hard-gate reasons: `{json.dumps(result['hard_gate_reasons'], sort_keys=True)}`",
        "",
        "## Changed Queries",
        "",
    ]
    for record in result["records"]:
        if (
            record["original_strategy_ids"] == record["replay_strategy_ids"]
            and record["original_hint_count"] == record["replay_hint_count"]
        ):
            continue
        lines.append(
            f"- stage {record['stage_index']} `{record['stage_name']}` / "
            f"`{record['failure_pattern']}`: strategies "
            f"{record['original_strategy_ids']} -> {record['replay_strategy_ids']}; "
            f"hints {record['original_hint_count']} -> {record['replay_hint_count']}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("decisions", type=Path)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--dut", required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--min-score", type=float, default=0.95)
    parser.add_argument("--min-quality", type=float, default=0.68)
    parser.add_argument("--min-utility", type=float, default=0.05)
    parser.add_argument("--prompt-limit", type=int, default=2)
    parser.add_argument("--hint-limit", type=int, default=1)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    result = replay(
        decisions,
        workspace=args.workspace,
        dut=args.dut,
        pack_path=args.pack,
        min_score=args.min_score,
        min_quality_score=args.min_quality,
        min_utility_score=args.min_utility,
        prompt_limit=args.prompt_limit,
        hint_limit=args.hint_limit,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.out_md:
        write_markdown(result, args.out_md)
    print(json.dumps({key: result[key] for key in (
        "injections_replayed",
        "changed_injections",
        "suppressed_injections",
        "original_prompt_items",
        "replay_prompt_items",
        "prompt_item_reduction_rate",
        "hard_gate_reasons",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
