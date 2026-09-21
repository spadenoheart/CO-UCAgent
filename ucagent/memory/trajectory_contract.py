# -*- coding: utf-8 -*-
"""Verifier-grounded contracts for reusable repair trajectories.

The contract is deliberately observational.  It records which verifier-state
transition followed a historical action sequence; it does not claim that the
memory caused that transition.  Runtime code can therefore audit contracts in
shadow mode before using them as an enforcement gate.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set


CONTRACT_VERSION = "verifier_transition_contract_v0"
INVALID_TEST_OUTCOMES = {
    "infrastructure_error",
    "timeout",
    "crash",
    "no_tests_collected",
}


def _as_list(value: Any, limit: int = 12) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value[:limit] if str(item).strip()]


def _normalize_test_case(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    text = re.sub(r":\d+(?:-\d+)?(?=::)", "", text)
    return text.removeprefix("unity_test/tests/")


def observation_failure_items(observation: Dict[str, Any]) -> Set[str]:
    """Return structured failure atoms suitable for transition comparison."""
    observation = observation if isinstance(observation, dict) else {}
    items = {
        f"case:{_normalize_test_case(item)}"
        for item in _as_list(observation.get("failed_cases_top"))
        if _normalize_test_case(item)
    }
    items.update(
        f"checkpoint:{str(item).strip()}"
        for item in _as_list(observation.get("failed_checkpoints_top"))
        if str(item).strip()
    )
    items.update(
        f"category:{str(item).strip()}"
        for item in _as_list(observation.get("checker_categories"))
        if str(item).strip() and str(item).strip() != "generic_checker_failure"
    )
    if not items:
        pattern = str(observation.get("failure_pattern") or "").strip()
        if pattern and pattern != "generic_checker_failure":
            items.add(f"pattern:{pattern}")
        elif observation.get("failure_signature"):
            items.add(f"signature:{observation['failure_signature']}")
    return items


def _item_types(items: Iterable[str]) -> List[str]:
    return sorted({str(item).partition(":")[0] for item in items if str(item)})


def _is_success(observation: Dict[str, Any]) -> bool:
    event_type = observation.get("event_type")
    if event_type == "test_run":
        if observation.get("test_outcome"):
            return observation.get("test_outcome") == "pass"
        try:
            total = int(observation.get("tests_total", 0) or 0)
            failed = int(observation.get("tests_failed", 0) or 0)
        except (TypeError, ValueError):
            return False
        return total > 0 and failed == 0 and observation.get("tests_passed_all") is True
    if event_type == "check_result":
        return observation.get("check_pass") is True
    return False


def _is_invalid(observation: Dict[str, Any]) -> bool:
    event_type = observation.get("event_type")
    if event_type == "test_run":
        outcome = str(observation.get("test_outcome") or "")
        if outcome in INVALID_TEST_OUTCOMES:
            return True
        try:
            return int(observation.get("tests_total", 0) or 0) <= 0
        except (TypeError, ValueError):
            return True
    if event_type == "check_result":
        return observation.get("check_pass") not in (True, False)
    return True


def build_transition_contract(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Derive a falsifiable transition contract from one historical episode."""
    episode = episode if isinstance(episode, dict) else {}
    before = episode.get("failure_before")
    if not isinstance(before, dict):
        before = episode.get("failure") if isinstance(episode.get("failure"), dict) else {}
    after = episode.get("observation_after")
    if not isinstance(after, dict):
        after = episode.get("success") if isinstance(episode.get("success"), dict) else {}
    delta = episode.get("failure_set_delta")
    delta = delta if isinstance(delta, dict) else {}
    information_gain = episode.get("information_gain")
    information_gain = information_gain if isinstance(information_gain, dict) else {}
    actions = episode.get("actions") if isinstance(episode.get("actions"), list) else []
    categories = _as_list(episode.get("action_categories"))
    if not categories:
        categories = list(dict.fromkeys(
            str(action.get("path_category"))
            for action in actions
            if isinstance(action, dict) and action.get("path_category")
        ))
    before_items = observation_failure_items(before)
    historical_net_reduction = max(0, int(delta.get("net_reduction", 0) or 0))
    expected_role = str(episode.get("credit_role") or "progress")
    return {
        "contract_version": CONTRACT_VERSION,
        "evidence_type": "observational_verifier_transition",
        "preconditions": {
            "stage_name": str(episode.get("stage_name") or ""),
            "failure_pattern": str(
                episode.get("failure_pattern") or before.get("failure_pattern") or ""
            ),
            "failure_group": str(
                episode.get("failure_group") or before.get("failure_group") or ""
            ),
            "failure_signature": before.get("failure_signature"),
            "failure_item_types": _item_types(before_items),
            "preferred_action_categories": _as_list(
                episode.get("preferred_action_categories")
            ),
        },
        "action_contract": {
            "action_categories": categories,
            "operations": list(dict.fromkeys(
                str(action.get("operation"))
                for action in actions
                if isinstance(action, dict) and action.get("operation")
            )),
            "max_file_mutations": max(1, len(actions)),
        },
        "expected_transition": {
            "credit_role": expected_role,
            "min_failure_net_reduction": 1 if historical_net_reduction > 0 else 0,
            "historical_failure_net_reduction": historical_net_reduction,
            "stage_advance_is_fulfillment": True,
            "min_information_gain": (
                min(1.0, max(0.25, float(information_gain.get("score", 0.0) or 0.0)))
                if expected_role == "diagnostic"
                else 0.0
            ),
        },
        "safety_constraints": {
            "max_regression_count": 0,
            "reject_invalid_validation": True,
        },
        "verification": {
            "event_type": str(after.get("event_type") or ""),
            "require_executed_tests": after.get("event_type") == "test_run",
            "accepted_scope_relations": ["same", "expanded"],
        },
        "historical_evidence": {
            "stage_advanced": bool(episode.get("stage_advanced")),
            "regression_count": int(episode.get("regression_count", 0) or 0),
            "confidence": episode.get("confidence"),
            "support_count": int(episode.get("support_count", 1) or 1),
        },
    }


def contract_applicability(
    contract: Dict[str, Any],
    failure: Dict[str, Any],
    *,
    stage_name: str = "",
    min_score: float = 0.55,
) -> Dict[str, Any]:
    """Score whether a historical contract is applicable to current failure state."""
    contract = contract if isinstance(contract, dict) else {}
    failure = failure if isinstance(failure, dict) else {}
    pre = contract.get("preconditions") if isinstance(contract.get("preconditions"), dict) else {}
    action = contract.get("action_contract") if isinstance(contract.get("action_contract"), dict) else {}
    score = 0.0
    reasons: List[str] = []
    hard_mismatches: List[str] = []

    expected_pattern = str(pre.get("failure_pattern") or "")
    current_pattern = str(failure.get("failure_pattern") or "")
    expected_group = str(pre.get("failure_group") or "")
    current_group = str(failure.get("failure_group") or "")
    if expected_pattern and current_pattern and expected_pattern == current_pattern:
        score += 0.40
        reasons.append("same_failure_pattern")
    elif expected_group and current_group and expected_group == current_group:
        score += 0.20
        reasons.append("same_failure_group")
    elif expected_pattern and current_pattern:
        hard_mismatches.append("failure_pattern_mismatch")

    expected_stage = str(pre.get("stage_name") or "")
    if expected_stage and stage_name and expected_stage == stage_name:
        score += 0.20
        reasons.append("same_stage")

    required_types = set(_as_list(pre.get("failure_item_types")))
    current_types = set(_item_types(observation_failure_items(failure)))
    if required_types and current_types:
        overlap = len(required_types & current_types) / max(1, len(required_types | current_types))
        score += 0.20 * overlap
        if overlap:
            reasons.append(f"failure_item_type_overlap={overlap:.2f}")

    expected_signature = pre.get("failure_signature")
    current_signature = failure.get("failure_signature")
    if current_pattern == "generic_checker_failure" and expected_signature:
        if current_signature == expected_signature:
            score += 0.20
            reasons.append("same_generic_signature")
        else:
            hard_mismatches.append("generic_signature_mismatch")

    preferred = set(_as_list(failure.get("preferred_action_categories")))
    action_categories = set(_as_list(action.get("action_categories")))
    if preferred and action_categories:
        overlap = preferred & action_categories
        if overlap:
            score += 0.20
            reasons.append("action_category_match=" + ",".join(sorted(overlap)))
        else:
            hard_mismatches.append("action_category_mismatch")

    score = round(min(1.0, max(0.0, score)), 4)
    return {
        "applicable": not hard_mismatches and score >= float(min_score),
        "score": score,
        "threshold": float(min_score),
        "reasons": reasons,
        "hard_mismatches": hard_mismatches,
        "contract_version": contract.get("contract_version"),
    }


def evaluate_transition_contract(
    contract: Dict[str, Any],
    before: Dict[str, Any],
    after: Dict[str, Any],
    *,
    stage_advanced: bool = False,
    scope_relation: str = "same",
) -> Dict[str, Any]:
    """Evaluate one injected contract against the next verifier observation."""
    contract = contract if isinstance(contract, dict) else {}
    before = before if isinstance(before, dict) else {}
    after = after if isinstance(after, dict) else {}
    expected = contract.get("expected_transition")
    expected = expected if isinstance(expected, dict) else {}
    safety = contract.get("safety_constraints")
    safety = safety if isinstance(safety, dict) else {}
    verification = contract.get("verification")
    verification = verification if isinstance(verification, dict) else {}

    if _is_invalid(after):
        return {
            "status": "invalid",
            "fulfilled": False,
            "utility": -1.0,
            "reason": "validation_not_executed_or_invalid",
            "contract_version": contract.get("contract_version"),
        }

    accepted_scope_relations = set(
        _as_list(verification.get("accepted_scope_relations"))
    )
    if accepted_scope_relations and scope_relation not in accepted_scope_relations:
        return {
            "status": "inconclusive",
            "fulfilled": False,
            "utility": 0.0,
            "reason": "incompatible_validation_scope",
            "expected_scope_relations": sorted(accepted_scope_relations),
            "observed_scope_relation": scope_relation,
            "contract_version": contract.get("contract_version"),
        }

    expected_event = str(verification.get("event_type") or "")
    if expected_event and after.get("event_type") != expected_event:
        return {
            "status": "inconclusive",
            "fulfilled": False,
            "utility": 0.0,
            "reason": "incompatible_verifier_event",
            "expected_event_type": expected_event,
            "observed_event_type": after.get("event_type"),
            "contract_version": contract.get("contract_version"),
        }

    before_items = observation_failure_items(before)
    after_items = set() if _is_success(after) else observation_failure_items(after)
    resolved = before_items - after_items
    introduced = after_items - before_items
    # An expanded test target can reveal pre-existing failures that the narrow
    # target could not observe.  They are scope-expansion evidence, not a
    # regression caused by the intervening action.
    regression_items = set() if scope_relation == "expanded" else {
        item
        for item in introduced
        if item.startswith(("case:", "checkpoint:", "category:"))
    }
    regression_count = len(regression_items)
    net_reduction = len(resolved) - regression_count
    max_regressions = int(safety.get("max_regression_count", 0) or 0)
    expected_role = str(expected.get("credit_role") or "progress")
    min_reduction = int(expected.get("min_failure_net_reduction", 0) or 0)
    information_gain = max(0.0, min(1.0, 0.15 * len(introduced)))

    if regression_count > max_regressions:
        status = "regression"
        fulfilled = False
        utility = -1.0 - 0.25 * regression_count
    elif expected_role == "diagnostic":
        min_information_gain = float(expected.get("min_information_gain", 0.25) or 0.25)
        fulfilled = information_gain >= min_information_gain
        status = "fulfilled" if fulfilled else "no_progress"
        utility = information_gain if fulfilled else -0.1
    else:
        fulfilled = bool(
            stage_advanced
            or _is_success(after)
            or (net_reduction >= max(1, min_reduction))
        )
        status = "fulfilled" if fulfilled else "no_progress"
        utility = (
            1.0
            + (1.0 if stage_advanced else 0.0)
            + 0.25 * max(0, net_reduction)
            if fulfilled
            else -0.2
        )

    return {
        "status": status,
        "fulfilled": fulfilled,
        "utility": round(utility, 4),
        "stage_advanced": bool(stage_advanced),
        "failure_set_delta": {
            "before": sorted(before_items),
            "after": sorted(after_items),
            "resolved": sorted(resolved),
            "introduced": sorted(introduced),
            "regression_items": sorted(regression_items),
            "net_reduction": net_reduction,
            "regression_count": regression_count,
            "scope_relation": scope_relation,
        },
        "information_gain_proxy": round(information_gain, 4),
        "contract_version": contract.get("contract_version"),
        "evidence_type": "observational_verifier_transition",
    }
