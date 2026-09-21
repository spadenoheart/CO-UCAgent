#!/usr/bin/env python3
"""Validate, deduplicate, and aggregate a context-reuse pack.

The legacy v0 pack remains untouched as an experimental baseline. This script
normally consumes the strictly rebuilt v1 source pack and emits representative
runtime strategies backed by validated repair episodes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ucagent.memory.context_reuse import preferred_action_categories
from ucagent.util.test_result import (
    TEST_OUTCOME_CRASH,
    TEST_OUTCOME_FAILURE,
    TEST_OUTCOME_INFRASTRUCTURE_ERROR,
    TEST_OUTCOME_NO_TESTS,
    TEST_OUTCOME_PASS,
    TEST_OUTCOME_TIMEOUT,
    classify_test_outcome,
)


UTILITY_POLICY_VERSION = "verifier_grounded_utility_v0"
UTILITY_WEIGHTS = {
    "failure_reduction": 0.45,
    "stage_advance": 0.35,
    "information_gain": 0.20,
    "regression": -0.60,
    "action_cost": -0.15,
    "historical_token_cost": -0.10,
}


def _stable_id(prefix: str, value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _failure_pattern(item: dict) -> str:
    failure = item.get("failure_before") if isinstance(item.get("failure_before"), dict) else item.get("failure")
    failure = failure if isinstance(failure, dict) else {}
    return str(item.get("failure_pattern") or failure.get("failure_pattern") or "generic_checker_failure")


def _action_plan(item: dict) -> tuple[tuple[str, str], ...]:
    plan: list[tuple[str, str]] = []
    actions = item.get("actions")
    if not isinstance(actions, list):
        actions = item.get("action_sequence", [])
    for action in actions:
        if not isinstance(action, dict):
            continue
        if action.get("action_type") == "validation":
            continue
        step = (str(action.get("operation") or ""), str(action.get("path_category") or ""))
        if plan and plan[-1] == step:
            continue
        plan.append(step)
    return tuple(plan)


def _valid_success(success: dict) -> tuple[bool, str, float]:
    event_type = success.get("event_type")
    if event_type == "check_result":
        if success.get("check_pass") is True:
            return True, "verified_check_pass", 1.0
        return False, "check_did_not_pass", 0.0
    if event_type == "test_run":
        outcome, reason = classify_test_outcome(success)
        if outcome == TEST_OUTCOME_PASS:
            return True, reason, 0.95
        return False, f"test_success_is_{outcome}", 0.0
    return False, "unsupported_success_event", 0.0


def _valid_role_typed_progress(item: dict, observation_after: dict) -> tuple[bool, str, float]:
    """Validate v2 progress without requiring the whole test set to pass."""
    if item.get("stage_advanced") is True:
        return True, "verified_stage_advance", 1.0
    valid_success, success_reason, success_strength = _valid_success(observation_after)
    if valid_success:
        return valid_success, success_reason, success_strength
    delta = item.get("failure_set_delta") if isinstance(item.get("failure_set_delta"), dict) else {}
    net_reduction = int(delta.get("net_reduction", 0) or 0)
    regression_count = int(item.get("regression_count", 0) or 0)
    scope_match = item.get("validation_scope_match") is True
    if net_reduction > 0 and regression_count == 0 and scope_match:
        strength = min(0.95, 0.70 + 0.05 * net_reduction)
        return True, "verified_partial_failure_reduction", strength
    return False, f"unverified_progress_after_{success_reason}", 0.0


def _valid_failure(failure: dict) -> tuple[bool, str, float]:
    event_type = failure.get("event_type")
    if event_type == "check_result":
        if failure.get("check_pass") is False:
            return True, "verified_check_failure", 1.0
        return False, "check_is_not_failure", 0.0
    if event_type == "test_run":
        outcome, reason = classify_test_outcome(failure)
        strength = {
            TEST_OUTCOME_FAILURE: 1.0,
            TEST_OUTCOME_INFRASTRUCTURE_ERROR: 0.65,
            TEST_OUTCOME_NO_TESTS: 0.55,
            TEST_OUTCOME_TIMEOUT: 0.3,
            TEST_OUTCOME_CRASH: 0.3,
        }.get(outcome, 0.0)
        return outcome != TEST_OUTCOME_PASS, reason, strength
    return False, "unsupported_failure_event", 0.0


def evaluate_episode(item: dict) -> tuple[dict, list[str]]:
    failure = item.get("failure_before") if isinstance(item.get("failure_before"), dict) else item.get("failure")
    success = item.get("observation_after") if isinstance(item.get("observation_after"), dict) else item.get("success")
    failure = failure if isinstance(failure, dict) else {}
    success = success if isinstance(success, dict) else {}
    raw_actions = item.get("actions") if isinstance(item.get("actions"), list) else item.get("action_sequence", [])
    actions = [
        action for action in raw_actions
        if isinstance(action, dict) and action.get("action_type") != "validation"
    ]
    pattern = _failure_pattern(item)
    if item.get("credit_role"):
        valid_success, success_reason, success_strength = _valid_role_typed_progress(item, success)
    else:
        valid_success, success_reason, success_strength = _valid_success(success)
    valid_failure, failure_reason, failure_strength = _valid_failure(failure)
    drop_reasons = []
    if item.get("credit_role") and item.get("credit_role") != "progress":
        drop_reasons.append(f"credit_role_{item['credit_role']}")
    if item.get("reuse_eligible") is False:
        drop_reasons.append("not_reuse_eligible")
    if not valid_success:
        drop_reasons.append(success_reason)
    if not valid_failure:
        drop_reasons.append(failure_reason)
    if not actions:
        drop_reasons.append("missing_actions")

    complete_actions = [
        action for action in actions
        if action.get("operation") and action.get("path") and action.get("path_category")
    ]
    action_completeness = len(complete_actions) / max(1, len(actions))
    if action_completeness < 1.0:
        drop_reasons.append("incomplete_action_metadata")

    failure_evidence_fields = (
        failure.get("failure_signature"),
        failure.get("checker_categories"),
        failure.get("failed_cases_top"),
        failure.get("failed_checkpoints_top"),
        failure.get("error_top"),
    )
    failure_evidence = min(1.0, sum(bool(value) for value in failure_evidence_fields) / 2.0)
    preferred = item.get("preferred_action_categories") or preferred_action_categories(pattern)
    categories = item.get("action_categories") or [action.get("path_category") for action in actions]
    categories = [str(category) for category in categories if category]
    if preferred:
        action_alignment = 1.0 if set(categories) & set(preferred) else 0.0
    else:
        action_alignment = 0.5
    specificity = 0.25 if pattern == "generic_checker_failure" else 1.0
    causal_locality = {1: 1.0, 2: 0.75}.get(len(actions), 0.55)

    score = round(
        0.25 * success_strength
        + 0.20 * (failure_strength * failure_evidence)
        + 0.15 * action_completeness
        + 0.15 * action_alignment
        + 0.10 * specificity
        + 0.15 * causal_locality,
        4,
    )
    quality = {
        "score": score,
        "success_evidence": round(success_strength, 4),
        "failure_evidence": round(failure_strength * failure_evidence, 4),
        "action_completeness": round(action_completeness, 4),
        "action_alignment": round(action_alignment, 4),
        "failure_specificity": round(specificity, 4),
        "causal_locality": round(causal_locality, 4),
        "success_reason": success_reason,
        "failure_reason": failure_reason,
    }
    return quality, drop_reasons


def _episode_key(item: dict) -> tuple[Any, ...]:
    failure = item.get("failure_before") if isinstance(item.get("failure_before"), dict) else item.get("failure")
    failure = failure if isinstance(failure, dict) else {}
    return (
        item.get("dut"),
        item.get("stage_name"),
        _failure_pattern(item),
        failure.get("failure_signature"),
        _action_plan(item),
    )


def _strategy_key(item: dict) -> tuple[Any, ...]:
    return item.get("stage_name"), _failure_pattern(item), _action_plan(item)


def _quality_tier(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.68:
        return "medium"
    return "low"


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, number))


def _verified_after_pass(item: dict) -> bool:
    after = item.get("observation_after") if isinstance(item.get("observation_after"), dict) else item.get("success")
    after = after if isinstance(after, dict) else {}
    if after.get("event_type") == "check_result":
        return after.get("check_pass") is True
    if after.get("event_type") == "test_run":
        outcome, _ = classify_test_outcome(after)
        return outcome == TEST_OUTCOME_PASS
    return False


def episode_verifier_utility(item: dict) -> dict:
    """Compute an interpretable utility prior from verifier-grounded evidence."""
    delta = item.get("failure_set_delta") if isinstance(item.get("failure_set_delta"), dict) else {}
    net_reduction = max(0.0, float(delta.get("net_reduction", 0) or 0))
    failure_reduction = min(1.0, net_reduction / 4.0)
    if _verified_after_pass(item):
        # A verified pass with only generic failure evidence is useful, but less
        # informative than resolving a concrete multi-item failure set.
        failure_reduction = max(failure_reduction, 0.5)

    stage_advance = 1.0 if item.get("stage_advanced") is True else 0.0
    information = item.get("information_gain") if isinstance(item.get("information_gain"), dict) else {}
    information_gain = _clamp(information.get("score", 0.0))
    regression = min(1.0, max(0.0, float(item.get("regression_count", 0) or 0)) / 2.0)

    action = item.get("action_cost") if isinstance(item.get("action_cost"), dict) else {}
    raw_actions = item.get("actions") if isinstance(item.get("actions"), list) else []
    file_mutations = float(action.get("file_mutations", len(raw_actions)) or 0)
    mutation_cost = min(1.0, file_mutations / 4.0)
    line_cost = min(1.0, max(0.0, float(action.get("changed_lines_proxy", 0) or 0)) / 200.0)
    elapsed_cost = min(1.0, max(0.0, float(action.get("elapsed_ms", 0) or 0)) / 600000.0)
    action_cost = 0.50 * mutation_cost + 0.25 * line_cost + 0.25 * elapsed_cost

    llm_tokens = action.get("llm_tokens")
    token_cost_known = isinstance(llm_tokens, (int, float)) and llm_tokens >= 0
    historical_token_cost = min(1.0, float(llm_tokens) / 8192.0) if token_cost_known else 0.0
    components = {
        "failure_reduction": round(failure_reduction, 4),
        "stage_advance": round(stage_advance, 4),
        "information_gain": round(information_gain, 4),
        "regression": round(regression, 4),
        "action_cost": round(action_cost, 4),
        "historical_token_cost": round(historical_token_cost, 4),
    }
    weighted_terms = {
        name: round(components[name] * weight, 4)
        for name, weight in UTILITY_WEIGHTS.items()
    }
    score = round(sum(weighted_terms.values()), 4)
    return {
        "policy_version": UTILITY_POLICY_VERSION,
        "score": score,
        "components": components,
        "weighted_terms": weighted_terms,
        "token_cost_known": token_cost_known,
    }


def aggregate_strategy_utility(group: list[dict]) -> dict:
    episode_utilities = [episode_verifier_utility(item) for item in group]
    scores = [float(item["score"]) for item in episode_utilities]
    positive_rate = sum(score > 0 for score in scores) / max(1, len(scores))
    mean_score = sum(scores) / max(1, len(scores))
    component_names = tuple(UTILITY_WEIGHTS)
    mean_components = {
        name: round(
            sum(float(item["components"][name]) for item in episode_utilities) / len(episode_utilities),
            4,
        )
        for name in component_names
    }
    sample_penalty = 0.08 / math.sqrt(max(1, len(scores)))
    inconsistency_penalty = 0.10 * (1.0 - positive_rate)
    uncertainty_penalty = sample_penalty + inconsistency_penalty
    return {
        "policy_version": UTILITY_POLICY_VERSION,
        "gate_score": round(mean_score - uncertainty_penalty, 4),
        "mean_episode_score": round(mean_score, 4),
        "min_episode_score": round(min(scores), 4),
        "max_episode_score": round(max(scores), 4),
        "positive_episode_rate": round(positive_rate, 4),
        "support_count": len(scores),
        "uncertainty_penalty": round(uncertainty_penalty, 4),
        "mean_components": mean_components,
        "historical_token_coverage": round(
            sum(bool(item["token_cost_known"]) for item in episode_utilities) / len(episode_utilities),
            4,
        ),
    }


def audit_pack(source: dict) -> dict:
    reason_counts: Counter[str] = Counter()
    valid = 0
    for item in source.get("effective_items", []):
        if not isinstance(item, dict):
            reason_counts["malformed_episode"] += 1
            continue
        _, reasons = evaluate_episode(item)
        if reasons:
            reason_counts.update(reasons)
        else:
            valid += 1
    total = len(source.get("effective_items", []))
    return {
        "pack_type": source.get("pack_type"),
        "episode_count": total,
        "structurally_valid_episode_count": valid,
        "invalid_episode_count": total - valid,
        "invalid_reason_counts": dict(sorted(reason_counts.items())),
    }


def curate_pack(source: dict, min_quality: float = 0.68) -> dict:
    drop_counts: Counter[str] = Counter()
    candidates: list[dict] = []
    for raw_item in source.get("effective_items", []):
        if not isinstance(raw_item, dict):
            drop_counts["malformed_episode"] += 1
            continue
        item = copy.deepcopy(raw_item)
        quality, drop_reasons = evaluate_episode(item)
        if drop_reasons:
            drop_counts.update(drop_reasons)
            continue
        if quality["action_alignment"] == 0.0:
            drop_counts["action_category_mismatch"] += 1
            continue
        if quality["score"] < min_quality:
            drop_counts["below_episode_quality_threshold"] += 1
            continue
        item["quality"] = quality
        item["quality_score"] = quality["score"]
        item["episode_id"] = _stable_id("episode", _episode_key(item))
        candidates.append(item)

    deduped_by_key: dict[tuple[Any, ...], dict] = {}
    for item in candidates:
        key = _episode_key(item)
        previous = deduped_by_key.get(key)
        if previous is None or item["quality_score"] > previous["quality_score"]:
            if previous is not None:
                drop_counts["near_duplicate_episode"] += 1
            deduped_by_key[key] = item
        else:
            drop_counts["near_duplicate_episode"] += 1
    episodes = list(deduped_by_key.values())

    strategy_groups: dict[tuple[Any, ...], list[dict]] = defaultdict(list)
    for item in episodes:
        strategy_groups[_strategy_key(item)].append(item)

    strategies: list[dict] = []
    for key, group in strategy_groups.items():
        group.sort(key=lambda item: (item.get("quality_score", 0.0), item.get("dut", "")), reverse=True)
        representative = copy.deepcopy(group[0])
        source_duts = sorted({str(item.get("dut")) for item in group if item.get("dut")})
        mean_quality = sum(float(item.get("quality_score", 0.0)) for item in group) / len(group)
        support_confidence = min(1.0, max(0, len(group) - 1) / 2.0)
        diversity_confidence = min(1.0, max(0, len(source_duts) - 1) / 2.0)
        strategy_quality = min(
            1.0,
            0.75 * mean_quality
            + 0.15 * support_confidence
            + 0.10 * diversity_confidence,
        )
        pattern = _failure_pattern(representative)
        if pattern == "generic_checker_failure" and len(source_duts) < 2:
            drop_counts["generic_single_dut_strategy"] += 1
            continue
        if strategy_quality < min_quality:
            drop_counts["below_strategy_quality_threshold"] += 1
            continue
        strategy_id = _stable_id("strategy", key)
        verifier_utility = aggregate_strategy_utility(group)
        representative.update({
            "strategy_id": strategy_id,
            "strategy_key": {
                "stage_name": key[0],
                "failure_pattern": key[1],
                "action_plan": [
                    {"operation": operation, "path_category": category}
                    for operation, category in key[2]
                ],
            },
            "source_duts": source_duts,
            "support_count": len(group),
            "source_episode_ids": [item["episode_id"] for item in group],
            "mean_episode_quality": round(mean_quality, 4),
            "support_confidence": round(support_confidence, 4),
            "cross_dut_confidence": round(diversity_confidence, 4),
            "quality_score": round(strategy_quality, 4),
            "quality_tier": _quality_tier(strategy_quality),
            "verifier_utility": verifier_utility,
        })
        strategies.append(representative)

    strategies.sort(
        key=lambda item: (
            float(item.get("quality_score", 0.0)),
            int(item.get("support_count", 0)),
            str(item.get("stage_name", "")),
        ),
        reverse=True,
    )

    hint_groups: dict[tuple[Any, ...], list[dict]] = defaultdict(list)
    for hint in source.get("compression_hints", []):
        if not isinstance(hint, dict):
            continue
        key = (hint.get("stage_name"), hint.get("failure_pattern"), hint.get("failure_signature"))
        hint_groups[key].append(hint)
    curated_hints = []
    for group in hint_groups.values():
        representative = copy.deepcopy(max(group, key=lambda item: int(item.get("repeat_count", 0) or 0)))
        representative["source_duts"] = sorted({str(item.get("dut")) for item in group if item.get("dut")})
        representative["support_count"] = len(group)
        representative["repeat_count"] = max(int(item.get("repeat_count", 0) or 0) for item in group)
        curated_hints.append(representative)

    utility_scores = [float(item["verifier_utility"]["gate_score"]) for item in strategies]
    return {
        "schema_version": 2,
        "pack_type": (
            "ucagent_context_reuse_credit_v2_curated_preview"
            if source.get("credit_schema")
            else "ucagent_context_reuse_v1_curated"
        ),
        "source_pack_type": source.get("pack_type"),
        "trace_count": source.get("trace_count", 0),
        "dut_list": source.get("dut_list", []),
        "effective_items": strategies,
        "strategies": strategies,
        "compression_hints": curated_hints,
        "stage_summaries": source.get("stage_summaries", []),
        "utility_policy": {
            "policy_version": UTILITY_POLICY_VERSION,
            "weights": UTILITY_WEIGHTS,
            "failure_reduction_scale": 4,
            "action_cost_scales": {
                "file_mutations": 4,
                "changed_lines_proxy": 200,
                "elapsed_ms": 600000,
                "llm_tokens": 8192,
            },
            "uncertainty": "0.08/sqrt(support) + 0.10*(1-positive_episode_rate)",
        },
        "curation": {
            "min_quality": min_quality,
            "source_episode_count": len(source.get("effective_items", [])),
            "validated_episode_count": len(candidates),
            "deduplicated_episode_count": len(episodes),
            "strategy_count": len(strategies),
            "source_hint_count": len(source.get("compression_hints", [])),
            "curated_hint_count": len(curated_hints),
            "drop_reason_counts": dict(sorted(drop_counts.items())),
            "quality_tier_counts": dict(Counter(item["quality_tier"] for item in strategies)),
            "utility_gate_score": {
                "min": round(min(utility_scores), 4) if utility_scores else None,
                "mean": round(sum(utility_scores) / len(utility_scores), 4) if utility_scores else None,
                "max": round(max(utility_scores), 4) if utility_scores else None,
                "positive_count": sum(score > 0 for score in utility_scores),
                "non_positive_count": sum(score <= 0 for score in utility_scores),
            },
        },
    }


def write_markdown(pack: dict, path: Path) -> None:
    curation = pack.get("curation", {})
    baseline_audit = pack.get("legacy_baseline_audit", {})
    lines = [
        "# UCAgent Context Reuse Pack v1 (Curated)",
        "",
        f"- Source episodes: {curation.get('source_episode_count', 0)}",
        f"- Validated episodes: {curation.get('validated_episode_count', 0)}",
        f"- Deduplicated episodes: {curation.get('deduplicated_episode_count', 0)}",
        f"- Runtime strategies: {curation.get('strategy_count', 0)}",
        f"- Quality threshold: {curation.get('min_quality')}",
        f"- Drop reasons: `{json.dumps(curation.get('drop_reason_counts', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- Legacy v0 audit: `{json.dumps(baseline_audit, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Strategies",
    ]
    for item in pack.get("effective_items", []):
        plan = item.get("strategy_key", {}).get("action_plan", [])
        plan_text = ", ".join(f"{step.get('operation')}:{step.get('path_category')}" for step in plan)
        lines.append(
            f"- `{item.get('strategy_id')}` stage=`{item.get('stage_name')}` "
            f"pattern=`{item.get('failure_pattern')}` quality={item.get('quality_score')} "
            f"utility={item.get('verifier_utility', {}).get('gate_score')} "
            f"support={item.get('support_count')} DUTs={item.get('source_duts')} plan=[{plan_text}]"
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="benchmark/ucagent_context_reuse/context_reuse_source_v1.json",
        help="Source context-reuse pack; it is never modified.",
    )
    parser.add_argument(
        "--out-json",
        default="benchmark/ucagent_context_reuse/context_reuse_v1.json",
    )
    parser.add_argument(
        "--out-md",
        default="benchmark/ucagent_context_reuse/context_reuse_v1.md",
    )
    parser.add_argument("--min-quality", type=float, default=0.68)
    parser.add_argument(
        "--baseline",
        default="benchmark/ucagent_context_reuse/context_reuse_v0.json",
        help="Optional immutable baseline pack included in the audit metadata.",
    )
    args = parser.parse_args()

    source_path = Path(args.input)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    pack = curate_pack(source, min_quality=args.min_quality)
    baseline_path = Path(args.baseline) if args.baseline else None
    if baseline_path is not None and baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        pack["legacy_baseline_audit"] = audit_pack(baseline)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(pack, out_md)
    print(json.dumps({
        "input": str(source_path),
        "out_json": str(out_json),
        "out_md": str(out_md),
        **pack["curation"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
