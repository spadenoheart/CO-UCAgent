#!/usr/bin/env python3
"""Build verifier-grounded, role-typed repair episodes from UCAgent traces.

The generated v2 pack is an offline candidate. It must be manually audited and
curated before any item is used by runtime context-reuse injection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ucagent.memory.context_reuse import (
    classify_failure_pattern,
    failure_pattern_group,
    preferred_action_categories,
)
from ucagent.memory.trajectory_contract import build_transition_contract
from ucagent.util.test_result import (
    TEST_OUTCOME_CRASH,
    TEST_OUTCOME_INFRASTRUCTURE_ERROR,
    TEST_OUTCOME_NO_TESTS,
    TEST_OUTCOME_PASS,
    TEST_OUTCOME_TIMEOUT,
    classify_test_outcome,
    stage_allows_expected_failures,
)


CREDIT_ROLES = ("progress", "diagnostic", "no_progress", "regression", "invalid")
CREDIT_RULE_VERSION = "verifier_grounded_role_typed_v2.1"
INVALID_TEST_OUTCOMES = {
    TEST_OUTCOME_INFRASTRUCTURE_ERROR,
    TEST_OUTCOME_TIMEOUT,
    TEST_OUTCOME_CRASH,
    TEST_OUTCOME_NO_TESTS,
}


def load_trace(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("trace_type") != "ucagent_structured_trajectory":
        raise ValueError(f"not a trace tree file: {path}")
    return data


def short(value: Any, limit: int = 300) -> str:
    return str(value or "").replace("\n", " ").strip()[:limit]


def stable_id(prefix: str, value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def test_outcome_for_node(node: dict) -> tuple[str, str]:
    observation = node.get("observation") if isinstance(node.get("observation"), dict) else {}
    return classify_test_outcome(
        observation,
        allow_expected_failures=stage_allows_expected_failures(node.get("stage_name")),
    )


def is_failure_observation(node: dict) -> bool:
    if node.get("event_type") == "test_run":
        outcome, _ = test_outcome_for_node(node)
        return outcome != TEST_OUTCOME_PASS
    if node.get("event_type") == "check_result":
        obs = node.get("observation") or {}
        return obs.get("check_pass") is False
    return False


def is_success_observation(node: dict) -> bool:
    if node.get("event_type") == "test_run":
        outcome, _ = test_outcome_for_node(node)
        return outcome == TEST_OUTCOME_PASS
    if node.get("event_type") == "check_result":
        obs = node.get("observation") or {}
        return obs.get("check_pass") is True
    return False


def compact_action(node: dict) -> dict:
    action = node.get("action") or {}
    detail = action.get("detail") if isinstance(action.get("detail"), dict) else {}
    text_meta = {}
    for key in ("text", "old_text", "new_text"):
        if isinstance(detail.get(key), dict):
            text_meta[key] = {
                "chars": detail[key].get("chars"),
                "lines": detail[key].get("lines"),
                "sha256": detail[key].get("sha256"),
            }
    return {
        "node_id": node.get("node_id"),
        "action_type": "file_mutation",
        "tool": action.get("tool"),
        "operation": action.get("operation"),
        "path": action.get("path"),
        "path_category": action.get("path_category"),
        "time_unix": node.get("time_unix"),
        "text_meta": text_meta,
    }


def compact_validation_action(node: dict) -> dict:
    observation = node.get("observation") if isinstance(node.get("observation"), dict) else {}
    return {
        "node_id": node.get("node_id"),
        "action_type": "validation",
        "tool": observation.get("tool") or node.get("event_type"),
        "target": observation.get("target"),
        "timeout": observation.get("timeout"),
        "duration_ms": observation.get("duration_ms"),
        "time_unix": node.get("time_unix"),
    }


def normalized_action_key(nodes: list[dict]) -> tuple[tuple[Any, Any], ...]:
    """Collapse repeated edits to the same target for compact reuse matching."""
    normalized: list[tuple[Any, Any]] = []
    for node in nodes:
        action = node.get("action") or {}
        item = (action.get("operation"), action.get("path"))
        if normalized and normalized[-1] == item:
            continue
        normalized.append(item)
    return tuple(normalized)


def normalize_action_nodes(nodes: list[dict]) -> list[dict]:
    """Keep the last event for each consecutive run of the same edit target."""
    normalized: list[dict] = []
    for node in nodes:
        action = node.get("action") or {}
        item = (action.get("operation"), action.get("path"))
        if normalized:
            prev_action = normalized[-1].get("action") or {}
            prev_item = (prev_action.get("operation"), prev_action.get("path"))
            if prev_item == item:
                normalized[-1] = node
                continue
        normalized.append(node)
    return normalized


def compact_observation(node: dict) -> dict:
    obs = node.get("observation") or {}
    test_outcome = None
    test_outcome_reason = None
    if node.get("event_type") == "test_run":
        test_outcome, test_outcome_reason = test_outcome_for_node(node)
    pattern = classify_failure_pattern(
        obs,
        stage_name=str(node.get("stage_name", "") or ""),
        event_type=str(node.get("event_type", "") or ""),
    )
    return {
        "node_id": node.get("node_id"),
        "time_unix": node.get("time_unix"),
        "timestamp": node.get("timestamp"),
        "event_type": node.get("event_type"),
        "tool": obs.get("tool"),
        "target": obs.get("target"),
        "duration_ms": obs.get("duration_ms"),
        "failure_signature": obs.get("failure_signature"),
        "failure_pattern": pattern,
        "failure_group": failure_pattern_group(pattern),
        "check_pass": obs.get("check_pass"),
        "test_outcome": test_outcome,
        "test_outcome_reason": test_outcome_reason,
        "tests_passed_all": test_outcome == TEST_OUTCOME_PASS if test_outcome else obs.get("tests_passed_all"),
        "tests_total": obs.get("tests_total"),
        "tests_failed": obs.get("tests_failed"),
        "run_test_success": obs.get("run_test_success"),
        "test_status_counts": obs.get("test_status_counts", {}),
        "checker_categories": obs.get("checker_categories", [])[:5],
        "failed_cases_top": obs.get("failed_cases_top", [])[:5],
        "failed_checkpoints_top": obs.get("failed_checkpoints_top", [])[:5],
        "error_top": [short(item, 220) for item in obs.get("error_top", [])[:3]],
    }


def stage_key(node: dict, fallback_run_id: str = "legacy") -> tuple[str, Any, str]:
    run_id = str(node.get("run_id") or node.get("thread_id") or fallback_run_id)
    return run_id, node.get("stage_index"), node.get("stage_name", "")


def action_categories(nodes: list[dict]) -> list[str]:
    categories = []
    for node in nodes:
        action = node.get("action") if isinstance(node.get("action"), dict) else {}
        category = action.get("path_category")
        if category and category not in categories:
            categories.append(category)
    return categories


def _observation_is_invalid(observation: dict) -> bool:
    if observation.get("event_type") == "test_run":
        return observation.get("test_outcome") in INVALID_TEST_OUTCOMES
    if observation.get("event_type") == "check_result":
        return observation.get("check_pass") not in (True, False)
    return True


def _observation_is_success(observation: dict) -> bool:
    if observation.get("event_type") == "test_run":
        return observation.get("test_outcome") == TEST_OUTCOME_PASS
    if observation.get("event_type") == "check_result":
        return observation.get("check_pass") is True
    return False


def _normalize_test_case(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    text = re.sub(r":\d+(?:-\d+)?(?=::)", "", text)
    return text.removeprefix("unity_test/tests/")


def _normalize_target(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    text = re.sub(r"\s+-{1,2}[A-Za-z].*$", "", text).strip()
    return text.removeprefix("unity_test/tests/")


def _failure_items(observation: dict) -> set[str]:
    if _observation_is_success(observation):
        return set()
    items = {
        f"case:{_normalize_test_case(item)}"
        for item in observation.get("failed_cases_top", [])
        if _normalize_test_case(item)
    }
    items.update(
        f"checkpoint:{str(item).strip()}"
        for item in observation.get("failed_checkpoints_top", [])
        if str(item).strip()
    )
    items.update(
        f"category:{str(item).strip()}"
        for item in observation.get("checker_categories", [])
        if str(item).strip() and str(item).strip() != "generic_checker_failure"
    )
    if not items:
        pattern = str(observation.get("failure_pattern") or "")
        if pattern and pattern != "generic_checker_failure":
            items.add(f"pattern:{pattern}")
        elif observation.get("failure_signature"):
            items.add(f"signature:{observation['failure_signature']}")
    return items


def _specificity_score(observation: dict) -> float:
    score = 0.0
    if observation.get("failed_cases_top"):
        score += 0.30
    if observation.get("failed_checkpoints_top"):
        score += 0.25
    pattern = str(observation.get("failure_pattern") or "")
    if pattern and pattern != "generic_checker_failure":
        score += 0.20
    if observation.get("error_top"):
        score += 0.15
    if _normalize_target(observation.get("target")):
        score += 0.10
    return min(1.0, score)


def _actionable_error_evidence(observation: dict) -> set[str]:
    text = "\n".join(str(item) for item in observation.get("error_top", []))
    patterns = (
        ("signal_not_found", r"signal name:\s*([A-Za-z_][A-Za-z0-9_]*)\s+not found"),
        ("missing_attribute", r"has no attribute\s+['\"]([^'\"]+)['\"]"),
        ("undefined_name", r"nameerror:\s*name\s+['\"]([^'\"]+)['\"]\s+is not defined"),
        ("missing_module", r"modulenotfounderror:\s*no module named\s+['\"]([^'\"]+)['\"]"),
        ("import_error", r"importerror:\s*([^\n]{1,120})"),
    )
    evidence = set()
    for label, pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            detail = re.sub(r"\s+", " ", match.group(1)).strip().lower()
            if detail:
                evidence.add(f"{label}:{detail}")
    return evidence


def _information_gain(before: dict, after: dict, scope_relation: str = "same") -> dict:
    before_items = _failure_items(before)
    after_items = _failure_items(after)
    before_target = _normalize_target(before.get("target"))
    after_target = _normalize_target(after.get("target"))
    target_narrowed = scope_relation == "narrowed"
    target_expanded = scope_relation == "expanded"
    specificity_delta = _specificity_score(after) - _specificity_score(before)
    new_evidence = sorted(after_items - before_items)
    new_actionable_evidence = sorted(
        _actionable_error_evidence(after) - _actionable_error_evidence(before)
    )
    score = max(0.0, specificity_delta)
    score += min(0.30, 0.10 * len(new_evidence))
    score += min(0.30, 0.20 * len(new_actionable_evidence))
    return {
        # This is an operational evidence-gain proxy, not Shannon information.
        "score": round(min(1.0, score), 4),
        "specificity_delta": round(specificity_delta, 4),
        "target_narrowed": target_narrowed,
        "target_expanded": target_expanded,
        "new_evidence": new_evidence,
        "new_evidence_count": len(new_evidence),
        "new_actionable_evidence": new_actionable_evidence,
        "new_actionable_evidence_count": len(new_actionable_evidence),
    }


def _target_parts(value: Any) -> tuple[str, str]:
    target = _normalize_target(value)
    if not target:
        return "", ""
    file_name, separator, selector = target.partition("::")
    return file_name, selector if separator else ""


def _items_for_target(items: set[str], target: str) -> set[str]:
    file_name, selector = _target_parts(target)
    if not file_name:
        return set(items)
    selected = set()
    normalized_target = f"{file_name}::{selector}" if selector else file_name
    for item in items:
        if not item.startswith("case:"):
            continue
        case_name = item.removeprefix("case:")
        if selector and case_name == normalized_target:
            selected.add(item)
        elif not selector and (case_name == file_name or case_name.startswith(file_name + "::")):
            selected.add(item)
    return selected


def _specific_error_classes(observation: dict) -> set[str]:
    classes = {
        str(item).strip()
        for item in observation.get("checker_categories", [])
        if str(item).strip() and str(item).strip() != "generic_checker_failure"
    }
    pattern = str(observation.get("failure_pattern") or "").strip()
    if pattern and pattern != "generic_checker_failure":
        classes.add(pattern)
    return classes


def _failure_set_delta(before: dict, after: dict, scope_relation: str = "same") -> dict:
    before_items = _failure_items(before)
    after_items = _failure_items(after)
    unobserved_before: set[str] = set()
    scope_expansion_items: set[str] = set()
    if scope_relation == "narrowed":
        comparable_before = _items_for_target(before_items, _normalize_target(after.get("target")))
        comparable_after = after_items
        unobserved_before = before_items - comparable_before
    elif scope_relation == "expanded":
        comparable_before = before_items
        comparable_after = _items_for_target(after_items, _normalize_target(before.get("target")))
        scope_expansion_items = after_items - comparable_after
    else:
        comparable_before = before_items
        comparable_after = after_items

    observed_resolved = comparable_before - comparable_after
    observed_introduced = comparable_after - comparable_before
    persisted = comparable_before & comparable_after
    concrete_prefixes = ("case:", "checkpoint:", "category:")
    concrete_before = {item for item in comparable_before if item.startswith(concrete_prefixes)}
    concrete_after = {item for item in comparable_after if item.startswith(concrete_prefixes)}
    before_is_fallback = bool(before_items) and all(
        item.startswith(("signature:", "pattern:")) for item in before_items
    )
    after_has_concrete = bool(concrete_after)
    evidence_refinement = before_is_fallback and after_has_concrete
    resolved = {item for item in observed_resolved if item.startswith(concrete_prefixes)}
    if (
        not _observation_is_success(after)
        and not concrete_after
        and scope_relation != "expanded"
    ):
        resolved = set()
    introduced = {item for item in observed_introduced if item.startswith(concrete_prefixes)}
    regression_items = set() if evidence_refinement else set(introduced)

    if scope_relation not in {"same", "narrowed", "expanded"}:
        resolved = set()
        introduced = set()
        regression_items = set()

    semantic_regressions: set[str] = set()
    if (
        scope_relation == "same"
        and before.get("event_type") == "check_result"
        and after.get("event_type") == "check_result"
        and not _observation_is_success(after)
    ):
        new_error_classes = _specific_error_classes(after) - _specific_error_classes(before)
        semantic_regressions = {f"error_class:{item}" for item in new_error_classes}
        regression_items.update(semantic_regressions)
    return {
        "before": sorted(before_items),
        "after": sorted(after_items),
        "resolved": sorted(resolved),
        "introduced": sorted(introduced),
        "persisted": sorted(persisted),
        "observed_resolved": sorted(observed_resolved),
        "observed_introduced": sorted(observed_introduced),
        "unobserved_before": sorted(unobserved_before),
        "scope_expansion_items": sorted(scope_expansion_items),
        "semantic_regressions": sorted(semantic_regressions),
        "before_count": len(before_items),
        "after_count": len(after_items),
        "net_reduction": len(resolved) - len(regression_items),
        "evidence_refinement": evidence_refinement,
        "regression_items": sorted(regression_items),
    }


def _validation_scope_relation(before: dict, after: dict) -> str:
    before_type = before.get("event_type")
    after_type = after.get("event_type")
    if before_type == "check_result" and after_type == "check_result":
        return "same"
    if before_type != "test_run" or after_type != "test_run":
        return "incompatible"
    before_target = _normalize_target(before.get("target"))
    after_target = _normalize_target(after.get("target"))
    if before_target == after_target:
        return "same"
    before_file, before_selector = _target_parts(before_target)
    after_file, after_selector = _target_parts(after_target)
    if before_file and after_file and before_file != after_file:
        return "disjoint"
    if before_selector and not after_selector:
        return "expanded"
    if not before_selector and after_selector:
        return "narrowed"
    return "disjoint"


def _validation_scope_matches(before: dict, after: dict) -> bool:
    return _validation_scope_relation(before, after) in {"same", "narrowed", "expanded"}


def _stage_advanced_after(stage_nodes: list[dict], observation_index: int) -> bool:
    for node in stage_nodes[observation_index + 1:]:
        event_type = node.get("event_type")
        if event_type == "stage_transition":
            reward = node.get("reward") if isinstance(node.get("reward"), dict) else {}
            return bool(reward.get("advanced") or reward.get("all_completed"))
        if event_type in {"file_mutation", "test_run", "check_result"}:
            break
    return False


def _action_cost(before: dict, after: dict, file_actions: list[dict]) -> dict:
    changed_lines = 0
    changed_chars = 0
    for action in file_actions:
        text_meta = action.get("text_meta") if isinstance(action.get("text_meta"), dict) else {}
        line_values = [
            int(meta.get("lines") or 0)
            for meta in text_meta.values()
            if isinstance(meta, dict)
        ]
        char_values = [
            int(meta.get("chars") or 0)
            for meta in text_meta.values()
            if isinstance(meta, dict)
        ]
        changed_lines += max(line_values, default=0)
        changed_chars += max(char_values, default=0)
    start = before.get("time_unix")
    end = after.get("time_unix")
    elapsed_ms = None
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end >= start:
        elapsed_ms = int(round((end - start) * 1000))
    return {
        "file_mutations": len(file_actions),
        "validation_calls": 1,
        "unique_files": len({action.get("path") for action in file_actions if action.get("path")}),
        "changed_lines_proxy": changed_lines,
        "changed_chars_proxy": changed_chars,
        "validation_duration_ms": after.get("duration_ms"),
        "elapsed_ms": elapsed_ms,
        "llm_tokens": None,
    }


def _credit_confidence(
    before: dict,
    after: dict,
    file_actions: list[dict],
    *,
    stage_advanced: bool,
    scope_match: bool,
    scope_relation: str,
    credit_role: str,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    if _observation_is_invalid(after):
        return 0.95, ["explicit_invalid_test_outcome"]
    score += 0.25
    reasons.append("valid_after_observation")
    if scope_match:
        score += 0.20
        reasons.append(f"validation_scope_{scope_relation}")
    if _failure_items(before):
        score += 0.15
        reasons.append("structured_failure_before")
    if _failure_items(after) or _observation_is_success(after):
        score += 0.15
        reasons.append("structured_observation_after")
    if 0 < len(file_actions) <= 5:
        score += 0.15
        reasons.append("local_action_window")
    elif len(file_actions) > 8:
        score -= 0.15
        reasons.append("long_action_window_penalty")
    if stage_advanced:
        score += 0.10
        reasons.append("stage_transition_confirmation")
    if before.get("failure_pattern") == "generic_checker_failure":
        score -= 0.10
        reasons.append("generic_failure_penalty")
    if credit_role == "progress" and not scope_match and not stage_advanced:
        score -= 0.20
        reasons.append("unmatched_success_scope_penalty")
    if scope_relation == "narrowed" and not stage_advanced:
        score -= 0.10
        reasons.append("narrowed_scope_penalty")
    return round(max(0.0, min(1.0, score)), 4), reasons


def build_credit_episode(
    *,
    dut: str,
    run_id: str,
    stage_index: Any,
    stage_name: str,
    failure_node: dict,
    after_node: dict,
    after_index: int,
    stage_nodes: list[dict],
    action_nodes: list[dict],
) -> dict:
    before = compact_observation(failure_node)
    after = compact_observation(after_node)
    normalized_nodes = normalize_action_nodes(action_nodes)
    file_actions = [compact_action(node) for node in normalized_nodes]
    action_sequence = [*file_actions, compact_validation_action(after_node)]
    stage_advanced = _stage_advanced_after(stage_nodes, after_index)
    scope_relation = _validation_scope_relation(before, after)
    scope_match = scope_relation in {"same", "narrowed", "expanded"}
    delta = _failure_set_delta(before, after, scope_relation=scope_relation)
    information_gain = _information_gain(before, after, scope_relation=scope_relation)
    delta["comparable"] = scope_match
    delta["scope_relation"] = scope_relation
    regression_count = len(delta["regression_items"])

    if _observation_is_invalid(after):
        credit_role = "invalid"
    elif scope_match and regression_count > 0 and delta["net_reduction"] <= 0 and not stage_advanced:
        credit_role = "regression"
    elif stage_advanced or (
        scope_match
        and (
            _observation_is_success(after)
            or delta["net_reduction"] > 0
        )
    ):
        credit_role = "progress"
    elif information_gain["score"] >= 0.25:
        credit_role = "diagnostic"
    else:
        credit_role = "no_progress"

    confidence, confidence_reasons = _credit_confidence(
        before,
        after,
        file_actions,
        stage_advanced=stage_advanced,
        scope_match=scope_match,
        scope_relation=scope_relation,
        credit_role=credit_role,
    )
    categories = [
        category
        for category in (action.get("path_category") for action in file_actions)
        if category
    ]
    categories = list(dict.fromkeys(categories))
    episode_key = {
        "dut": dut,
        "run_id": run_id,
        "stage_index": stage_index,
        "failure_node": before.get("node_id"),
        "after_node": after.get("node_id"),
        "actions": [(item.get("operation"), item.get("path")) for item in file_actions],
    }
    reuse_eligible = bool(
        credit_role == "progress"
        and confidence >= 0.55
        and file_actions
        and scope_relation in {"same", "expanded"}
        and regression_count == 0
    )
    pattern = str(before.get("failure_pattern") or "generic_checker_failure")
    episode = {
        "episode_id": stable_id("credit", episode_key),
        "credit_rule_version": CREDIT_RULE_VERSION,
        "dut": dut,
        "run_id": run_id,
        "stage_index": stage_index,
        "stage_name": stage_name,
        "failure_pattern": pattern,
        "failure_group": failure_pattern_group(pattern),
        "action_categories": categories,
        "preferred_action_categories": preferred_action_categories(pattern),
        "failure_before": before,
        "action_sequence": action_sequence,
        "observation_after": after,
        "stage_advanced": stage_advanced,
        "failure_set_delta": delta,
        "regression_count": regression_count,
        "information_gain": information_gain,
        "action_cost": _action_cost(before, after, file_actions),
        "credit_role": credit_role,
        "confidence": confidence,
        "confidence_reasons": confidence_reasons,
        "validation_scope_match": scope_match,
        "validation_scope_relation": scope_relation,
        "reuse_eligible": reuse_eligible,
        # Compatibility aliases for the existing curator/runtime pack schema.
        "failure": before,
        "actions": file_actions,
        "success": after,
    }
    episode["transition_contract"] = build_transition_contract(episode)
    return episode


def build_pack(traces: list[dict], pack_type: str = "ucagent_context_reuse_credit_v2") -> dict:
    episodes: list[dict] = []
    effective_by_key: dict[tuple[Any, ...], dict] = {}
    compression_hints = []
    stage_summaries = []
    run_ids: set[str] = set()

    for trace_index, trace in enumerate(traces):
        dut = trace.get("dut", "")
        nodes = trace.get("nodes", [])
        terminal_seen: set[str] = set()
        filtered_nodes = []
        for node in sorted(
            nodes,
            key=lambda item: (
                float(item.get("time_unix", 0) or 0),
                int(item.get("round_index", 0) or 0),
                str(item.get("node_id") or ""),
            ),
        ):
            run_id = stage_key(node, f"trace-{trace_index:03d}")[0]
            if run_id in terminal_seen:
                continue
            filtered_nodes.append(node)
            reward = node.get("reward") if isinstance(node.get("reward"), dict) else {}
            if node.get("event_type") == "stage_transition" and reward.get("all_completed"):
                terminal_seen.add(run_id)
        by_stage = defaultdict(list)
        for node in filtered_nodes:
            fallback_run_id = f"trace-{trace_index:03d}"
            by_stage[stage_key(node, fallback_run_id)].append(node)

        for (run_id, stage_index, stage_name), stage_nodes in sorted(
            by_stage.items(),
            key=lambda item: (item[0][0], item[0][1] is None, item[0][1], item[0][2]),
        ):
            run_ids.add(run_id)
            stage_nodes.sort(
                key=lambda node: (
                    float(node.get("time_unix", 0) or 0),
                    int(node.get("round_index", 0) or 0),
                    str(node.get("node_id") or ""),
                )
            )
            failure_counts = Counter()
            action_counts = Counter()
            for node in stage_nodes:
                if is_failure_observation(node):
                    sig = (node.get("observation") or {}).get("failure_signature")
                    if sig:
                        failure_counts[sig] += 1
                if node.get("event_type") == "file_mutation":
                    action = node.get("action") or {}
                    action_counts[(action.get("path_category"), action.get("path"))] += 1

            repeated = failure_counts.most_common()
            if repeated:
                for sig, count in repeated:
                    if count <= 1:
                        continue
                    first_failure = None
                    for node in stage_nodes:
                        obs = node.get("observation") if isinstance(node.get("observation"), dict) else {}
                        if is_failure_observation(node) and obs.get("failure_signature") == sig:
                            first_failure = node
                            break
                    failure_obs = first_failure.get("observation") if isinstance(first_failure, dict) else {}
                    pattern = classify_failure_pattern(
                        failure_obs if isinstance(failure_obs, dict) else {},
                        stage_name=str(stage_name or ""),
                        event_type=str(first_failure.get("event_type") if isinstance(first_failure, dict) else ""),
                    )
                    compression_hints.append({
                        "dut": dut,
                        "run_id": run_id,
                        "stage_index": stage_index,
                        "stage_name": stage_name,
                        "failure_signature": sig,
                        "failure_pattern": pattern,
                        "failure_group": failure_pattern_group(pattern),
                        "preferred_action_categories": preferred_action_categories(pattern),
                        "repeat_count": count,
                        "policy": "keep_latest_failure_summary_and_the_adjacent_action_observation_transition; drop older raw outputs with the same signature",
                    })

            stage_episodes: list[dict] = []
            for idx, node in enumerate(stage_nodes):
                if not is_failure_observation(node):
                    continue
                after_index = None
                for candidate_index in range(idx + 1, len(stage_nodes)):
                    if stage_nodes[candidate_index].get("event_type") in {"test_run", "check_result"}:
                        after_index = candidate_index
                        break
                if after_index is None:
                    continue
                action_nodes = [
                    candidate
                    for candidate in stage_nodes[idx + 1:after_index]
                    if candidate.get("event_type") == "file_mutation" and candidate.get("success") is True
                ]
                episode = build_credit_episode(
                    dut=dut,
                    run_id=run_id,
                    stage_index=stage_index,
                    stage_name=stage_name,
                    failure_node=node,
                    after_node=stage_nodes[after_index],
                    after_index=after_index,
                    stage_nodes=stage_nodes,
                    action_nodes=action_nodes,
                )
                episodes.append(episode)
                stage_episodes.append(episode)
                if episode["reuse_eligible"]:
                    failure = episode["failure_before"]
                    action_key = tuple(
                        (action.get("operation"), action.get("path"))
                        for action in episode["actions"]
                    )
                    dedup_key = (
                        dut,
                        stage_index,
                        stage_name,
                        failure.get("failure_signature"),
                        action_key,
                    )
                    previous = effective_by_key.get(dedup_key)
                    if previous is None or episode["confidence"] > previous["confidence"]:
                        effective_by_key[dedup_key] = episode

            stage_summaries.append({
                "dut": dut,
                "run_id": run_id,
                "stage_index": stage_index,
                "stage_name": stage_name,
                "nodes": len(stage_nodes),
                "failure_signature_repeats": dict(failure_counts.most_common(10)),
                "credit_role_counts": dict(Counter(item["credit_role"] for item in stage_episodes)),
                "hot_action_paths": [
                    {"path_category": key[0], "path": key[1], "count": value}
                    for key, value in action_counts.most_common(10)
                ],
            })

    return {
        "schema_version": 3,
        "pack_type": pack_type,
        "credit_schema": "verifier_grounded_role_typed_v2",
        "credit_rule_version": CREDIT_RULE_VERSION,
        "trace_count": len(traces),
        "run_count": len(run_ids),
        "dut_list": sorted({trace.get("dut", "") for trace in traces if trace.get("dut")}),
        "credit_roles": list(CREDIT_ROLES),
        "credit_role_counts": dict(Counter(item["credit_role"] for item in episodes)),
        "episodes": episodes,
        "effective_items": list(effective_by_key.values()),
        "compression_hints": compression_hints,
        "stage_summaries": stage_summaries,
    }


def write_markdown(pack: dict, path: Path) -> None:
    lines = []
    lines.append(f"# UCAgent Context Reuse Pack ({pack.get('pack_type', 'unknown')})")
    lines.append("")
    lines.append(f"- Trace count: {pack['trace_count']}")
    lines.append(f"- Run count: {pack.get('run_count', 0)}")
    lines.append(f"- DUTs: {', '.join(pack['dut_list'])}")
    lines.append(f"- Candidate episodes: {len(pack.get('episodes', []))}")
    lines.append(f"- Effective repair items: {len(pack['effective_items'])}")
    lines.append(f"- Compression hints: {len(pack['compression_hints'])}")
    lines.append(
        "- Credit roles: `"
        + json.dumps(pack.get("credit_role_counts", {}), ensure_ascii=False, sort_keys=True)
        + "`"
    )
    lines.append("")
    lines.append("## Effective Repair Items")
    for item in pack["effective_items"][:30]:
        failure = item["failure"]
        actions = ", ".join(f"{a['operation']}:{a['path']}" for a in item["actions"])
        lines.append(
            f"- {item['dut']} stage {item['stage_index']} `{item['stage_name']}` "
            f"pattern={item.get('failure_pattern')} sig={failure.get('failure_signature')} "
            f"action_categories={item.get('action_categories', [])} actions=[{actions}]"
        )
    lines.append("")
    lines.append("## Compression Hints")
    for item in pack["compression_hints"][:40]:
        lines.append(
            f"- {item['dut']} stage {item['stage_index']} `{item['stage_name']}` "
            f"pattern={item.get('failure_pattern')} sig={item['failure_signature']} "
            f"repeats={item['repeat_count']}: {item['policy']}"
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def select_annotation_sample(episodes: list[dict], size: int = 40, seed: int = 20260715) -> list[dict]:
    """Select a deterministic role-stratified sample for manual attribution audit."""
    size = max(0, min(size, len(episodes)))
    rng = random.Random(seed)
    by_role: dict[str, list[dict]] = defaultdict(list)
    for episode in episodes:
        by_role[str(episode.get("credit_role") or "unknown")].append(episode)
    for items in by_role.values():
        items.sort(key=lambda item: str(item.get("episode_id") or ""))
        rng.shuffle(items)

    selected: list[dict] = []
    active_roles = [role for role in CREDIT_ROLES if by_role.get(role)]
    while len(selected) < size and active_roles:
        next_roles = []
        for role in active_roles:
            if by_role[role] and len(selected) < size:
                selected.append(by_role[role].pop())
            if by_role[role]:
                next_roles.append(role)
        active_roles = next_roles
    return selected


def annotation_record(episode: dict, *, blind: bool = False) -> dict:
    """Build one annotation record, optionally withholding rule predictions."""
    derived_evidence_fields = (
        "failure_before",
        "action_sequence",
        "observation_after",
        "stage_advanced",
        "failure_set_delta",
        "regression_count",
        "information_gain",
        "action_cost",
    )
    if blind:
        raw_observation_fields = (
            "node_id",
            "time_unix",
            "timestamp",
            "event_type",
            "tool",
            "target",
            "duration_ms",
            "check_pass",
            "tests_total",
            "tests_failed",
            "run_test_success",
            "test_status_counts",
            "checker_categories",
            "failed_cases_top",
            "failed_checkpoints_top",
            "error_top",
        )

        def raw_observation(name: str) -> dict:
            observation = episode.get(name)
            if not isinstance(observation, dict):
                return {}
            return {
                field: observation.get(field)
                for field in raw_observation_fields
                if field in observation
            }

        evidence = {
            "failure_before": raw_observation("failure_before"),
            "action_sequence": episode.get("action_sequence"),
            "observation_after": raw_observation("observation_after"),
            "stage_advanced": episode.get("stage_advanced"),
        }
    else:
        evidence = {key: episode.get(key) for key in derived_evidence_fields}
    record = {
        "episode_id": episode.get("episode_id"),
        "dut": episode.get("dut"),
        "run_id": episode.get("run_id"),
        "stage_index": episode.get("stage_index"),
        "stage_name": episode.get("stage_name"),
        "evidence": evidence,
        "annotations": [
            {"annotator": "", "credit_role": "", "notes": ""},
            {"annotator": "", "credit_role": "", "notes": ""},
        ],
    }
    if not blind:
        record["model_credit_role"] = episode.get("credit_role")
        record["model_confidence"] = episode.get("confidence")
        record["evidence"]["credit_role"] = episode.get("credit_role")
        record["evidence"]["confidence"] = episode.get("confidence")
    return record


def write_annotation_jsonl(
    pack: dict,
    path: Path,
    size: int = 40,
    seed: int = 20260715,
    *,
    blind: bool = False,
) -> int:
    sample = select_annotation_sample(pack.get("episodes", []), size=size, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for episode in sample:
            record = annotation_record(episode, blind=blind)
            fout.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return len(sample)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_trees", nargs="+", help="Trace tree JSON files from scripts/build_trace_tree.py")
    parser.add_argument(
        "--out-json",
        default="benchmark/ucagent_context_reuse/context_reuse_credit_v2_candidates.json",
    )
    parser.add_argument(
        "--out-md",
        default="benchmark/ucagent_context_reuse/context_reuse_credit_v2_candidates.md",
    )
    parser.add_argument("--pack-type", default="ucagent_context_reuse_credit_v2")
    parser.add_argument(
        "--annotation-out",
        default="benchmark/ucagent_context_reuse/episode_credit_v2_annotation_40.jsonl",
    )
    parser.add_argument(
        "--annotation-blind-out",
        default="",
        help="Optional blind JSONL with model role/confidence withheld.",
    )
    parser.add_argument("--annotation-size", type=int, default=40)
    parser.add_argument("--annotation-seed", type=int, default=20260715)
    args = parser.parse_args()

    traces = [load_trace(Path(path)) for path in args.trace_trees]
    pack = build_pack(traces, pack_type=args.pack_type)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(pack, out_md)
    annotation_count = 0
    if args.annotation_out:
        annotation_count = write_annotation_jsonl(
            pack,
            Path(args.annotation_out),
            size=args.annotation_size,
            seed=args.annotation_seed,
        )
    blind_annotation_count = 0
    if args.annotation_blind_out:
        blind_annotation_count = write_annotation_jsonl(
            pack,
            Path(args.annotation_blind_out),
            size=args.annotation_size,
            seed=args.annotation_seed,
            blind=True,
        )
    print(json.dumps({
        "out_json": str(out_json),
        "out_md": str(out_md),
        "effective_items": len(pack["effective_items"]),
        "candidate_episodes": len(pack.get("episodes", [])),
        "credit_role_counts": pack.get("credit_role_counts", {}),
        "compression_hints": len(pack["compression_hints"]),
        "annotation_out": args.annotation_out,
        "annotation_count": annotation_count,
        "annotation_blind_out": args.annotation_blind_out,
        "blind_annotation_count": blind_annotation_count,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
