# -*- coding: utf-8 -*-
"""Failure-aware repair strategy retrieval for UCAgent.

This module consumes the offline ``context_reuse_v0.json`` pack and exposes a
small runtime retrieval API. It is intentionally separate from long-term memory:
long-term memory is a general store, while this store is keyed around recent
verification failures and compact repair episodes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from ucagent.memory.trajectory_contract import (
    build_transition_contract,
    contract_applicability,
)
from ucagent.util.log import info, warning


def _as_list(value: Any, limit: int = 8) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value[:limit] if str(item).strip()]


def _short(value: Any, limit: int = 220) -> str:
    return str(value or "").replace("\n", " ").strip()[:limit]


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


def _tokenize(text: str) -> List[str]:
    return [
        token
        for token in re.split(r"[^a-zA-Z0-9_./:-]+", str(text or "").lower())
        if len(token) >= 3
    ]


def _token_overlap_score(left: List[str], right: List[str]) -> float:
    left_tokens = set()
    right_tokens = set()
    for item in left:
        left_tokens.update(_tokenize(item))
    for item in right:
        right_tokens.update(_tokenize(item))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


PATTERN_GROUPS = {
    "api_or_env_interface_mismatch": "interface",
    "test_import_error": "interface",
    "reference_files_unread": "interface",
    "coverage_or_test_marks_missing": "coverage",
    "test_marking_contract_incomplete": "coverage",
    "dynamic_label_used_in_coverage": "coverage",
    "test_template_not_implemented": "test_logic",
    "template_source_contract_incomplete": "test_logic",
    "template_validation_or_runner_contract": "test_logic",
    "test_structure_or_discovery": "test_logic",
    "test_logic_or_case_implementation": "test_logic",
    "test_assertion_or_expectation_mismatch": "test_logic",
    "missing_assert": "test_logic",
    "bug_doc_schema_or_marking": "bug_document",
    "bug_doc_incomplete_tc_label": "bug_document",
    "bug_doc_passed_tc_marked_as_bug": "bug_document",
    "bug_doc_checkpoint_not_marked": "bug_document",
    "label_or_report_mismatch": "bug_document",
    "duplicate_label_definition": "bug_document",
    "failed_cases_need_classification": "bug_document",
    "syntax_or_format_error": "format",
    "generic_checker_failure": "generic",
}


PATTERN_ACTION_CATEGORIES = {
    "api_or_env_interface_mismatch": ("test_code", "code"),
    "test_import_error": ("test_code", "code"),
    "reference_files_unread": ("document", "spec_check_document", "test_code"),
    "coverage_or_test_marks_missing": ("coverage_code", "test_code", "spec_check_document"),
    "test_marking_contract_incomplete": ("test_code", "coverage_code"),
    "dynamic_label_used_in_coverage": ("coverage_code", "test_code"),
    "test_template_not_implemented": ("test_code",),
    "template_source_contract_incomplete": ("test_code", "coverage_code"),
    "template_validation_or_runner_contract": ("test_code", "coverage_code"),
    "test_structure_or_discovery": ("test_code",),
    "test_logic_or_case_implementation": ("test_code",),
    "test_assertion_or_expectation_mismatch": ("test_code",),
    "missing_assert": ("test_code",),
    "bug_doc_schema_or_marking": ("bug_document",),
    "bug_doc_incomplete_tc_label": ("bug_document",),
    "bug_doc_passed_tc_marked_as_bug": ("bug_document", "test_code"),
    "bug_doc_checkpoint_not_marked": ("bug_document", "test_code"),
    "label_or_report_mismatch": ("bug_document", "coverage_code"),
    "duplicate_label_definition": ("bug_document", "spec_check_document", "coverage_code", "test_code"),
    "failed_cases_need_classification": ("bug_document", "test_code"),
    "api_test_collection_regression": ("test_code",),
    "random_test_source_contract_violation": ("test_code",),
    "syntax_or_format_error": (
        "test_code",
        "coverage_code",
        "bug_document",
        "spec_check_document",
        "code",
        "document",
    ),
}


GROUP_ACTION_CATEGORIES = {
    "interface": ("test_code", "code"),
    "coverage": ("coverage_code", "test_code", "spec_check_document"),
    "test_logic": ("test_code",),
    "bug_document": ("bug_document", "coverage_code", "test_code"),
    "format": (
        "test_code",
        "coverage_code",
        "bug_document",
        "spec_check_document",
        "code",
        "document",
    ),
    "generic": (),
}


# These failures describe missing workflow evidence rather than a repair action.
# Reusing a file-edit episode for them encourages the model to modify code before
# satisfying the checker contract (for example, before reading a required file).
NON_REUSABLE_FAILURE_PATTERNS = {
    "reference_files_unread",
}


# Subtypes in these groups have different repair contracts. A strategy for an
# incomplete TC label is not interchangeable with one for a passed TC recorded
# as a bug, even though both touch the same document.
EXACT_PATTERN_GROUPS = {
    "bug_document",
}


CHECKER_CATEGORY_PRIORITY = [
    "api_test_collection_regression",
    "random_test_source_contract_violation",
    "test_marking_contract_incomplete",
    "template_source_contract_incomplete",
    "bug_doc_checkpoint_not_marked",
    "label_or_report_mismatch",
    "failed_cases_need_classification",
    "bug_doc_passed_tc_marked_as_bug",
    "bug_doc_incomplete_tc_label",
    "duplicate_label_definition",
    "coverage_or_test_marks_missing",
    "dynamic_label_used_in_coverage",
    "missing_assert",
    "test_template_not_implemented",
    "template_validation_or_runner_contract",
    "test_structure_or_discovery",
    "test_import_error",
    "api_or_env_interface_mismatch",
    "reference_files_unread",
    "generic_checker_failure",
]


_STAGE_ROLES = {
    "create_test_case_templates": "template_generation",
    "test_case_implementation_in_batch": "test_implementation",
    "generate_random_test_cases": "test_implementation",
}


def stage_role(stage_name: str) -> str:
    """Return the retrieval role for stages with incompatible action semantics."""
    return _STAGE_ROLES.get(str(stage_name or "").strip().lower(), "")


def stage_roles_compatible(current_stage: str, source_stage: str) -> bool:
    """Prevent template-generation and implementation episodes from mixing."""
    current_role = stage_role(current_stage)
    source_role = stage_role(source_stage)
    if "template_generation" in (current_role, source_role):
        return current_role == source_role
    return True


def failure_pattern_group(pattern: str) -> str:
    return PATTERN_GROUPS.get(str(pattern or ""), "generic")


def preferred_action_categories(pattern: str) -> List[str]:
    pattern = str(pattern or "")
    categories = PATTERN_ACTION_CATEGORIES.get(pattern)
    if categories is None:
        categories = GROUP_ACTION_CATEGORIES.get(failure_pattern_group(pattern), ())
    return list(categories)


def preferred_action_categories_for_failure(
    failure: Dict[str, Any],
    stage_name: str = "",
) -> List[str]:
    """Return action categories grounded in the current verifier evidence."""
    failure = failure if isinstance(failure, dict) else {}
    pattern = str(failure.get("failure_pattern") or "")
    if pattern in NON_REUSABLE_FAILURE_PATTERNS or pattern == "generic_checker_failure":
        return []

    if pattern == "duplicate_label_definition":
        text = _text_blob_from_failure(failure, stage_name=stage_name)
        if any(token in text for token in ("bug_analysis", "bug analysis", "bug document", "<bg-", "<tc-")):
            return ["bug_document"]
        if any(token in text for token in ("functions_and_checks", "function and check", "<fg-", "<fc-", "<ck-")):
            return ["spec_check_document"]
        if "coverage" in text:
            return ["coverage_code"]

    return preferred_action_categories(pattern)


def _text_blob_from_failure(failure: Dict[str, Any], stage_name: str = "", event_type: str = "") -> str:
    parts = [
        stage_name,
        event_type,
        ",".join(_as_list(failure.get("checker_categories"), 10)),
        ",".join(_as_list(failure.get("failed_cases_top"), 10)),
        ",".join(_as_list(failure.get("failed_checkpoints_top"), 10)),
        " ".join(_as_list(failure.get("error_top"), 6)),
        str(failure.get("message", "")),
    ]
    return " ".join(str(part).lower() for part in parts if part)


def classify_failure_pattern(
    failure: Dict[str, Any],
    stage_name: str = "",
    event_type: str = "",
    path_category: str = "",
    path: str = "",
) -> str:
    """Classify a compact failure summary into a reusable repair pattern.

    This intentionally mirrors the prompt-v1 checker categories where present,
    then falls back to text and stage heuristics. It is shared by runtime
    retrieval and offline pack building so the same failure taxonomy is used at
    both ends of context reuse.
    """
    failure = failure if isinstance(failure, dict) else {}
    raw_categories = [str(item) for item in _as_list(failure.get("checker_categories"), 12)]
    categories = {item.lower() for item in raw_categories}
    for category in CHECKER_CATEGORY_PRIORITY:
        if category.lower() in categories:
            return category

    text = _text_blob_from_failure(failure, stage_name=stage_name, event_type=event_type)
    stage = str(stage_name or "").lower()
    path_text = f"{path_category} {path}".lower()
    combined = f"{text} {path_text}"

    if any(token in combined for token in ("not found in the test report", "not found in the function coverage")):
        return "label_or_report_mismatch"
    if any(marker in combined for marker in (
        "all failed test cases must indicate bugs",
        "all failed test cases must be documented",
        "[undocumented failed cases]",
    )):
        return "failed_cases_need_classification"
    if "[api test collection regression]" in combined:
        return "api_test_collection_regression"
    if "random-test source contract violations" in combined:
        return "random_test_source_contract_violation"
    if "[template source contract]" in combined:
        return "template_source_contract_incomplete"
    if (
        "[checkpoint not marked]" in combined
        or "have not called mark_function for their associated checkpoints" in combined
    ):
        return "bug_doc_checkpoint_not_marked"
    if (
        "do not have correct check point marks" in combined
        or "not marked with 'mark_function'" in combined
        or 'not marked with "mark_function"' in combined
    ):
        return "test_marking_contract_incomplete"
    if "should be ''failed'' but found to be ''passed''" in combined or "should be 'failed' but found to be 'passed'" in combined:
        return "bug_doc_passed_tc_marked_as_bug"
    if "incomplete label" in combined and "<tc-*>" in combined:
        return "bug_doc_incomplete_tc_label"
    if "defined multiple times" in combined:
        return "duplicate_label_definition"
    if any(token in combined for token in ("bug analysis", "<bg-", "<tc-", "marked 0 bugs", "bug document")):
        return "bug_doc_schema_or_marking"
    if any(token in combined for token in ("importerror", "modulenotfounderror", "cannot import")):
        return "test_import_error"
    if any(token in combined for token in (
        "attributeerror",
        "has no attribute",
        "nameerror",
        "not defined",
        "undefined name",
        "hasattr(",
        "应该有",
        "missing method",
        "missing attribute",
    )):
        return "api_or_env_interface_mismatch"
    if any(token in combined for token in (
        "insufficient env fixture test coverage",
        "env test functions found",
        "test function was not collected",
        "test functions found, minimum required",
    )):
        return "test_structure_or_discovery"
    if "do not contain assert" in combined or "must contain at least one assert" in combined:
        return "missing_assert"
    if any(token in combined for token in ("unmarked_check_points", "unmarked", "mark_function", "check point", "checkpoint", "coverage")):
        return "coverage_or_test_marks_missing"
    if any(token in stage for token in ("coverage_group", "coverage_point", "function_checks")):
        return "coverage_or_test_marks_missing"
    if "create_test_case_templates" in stage:
        return "template_validation_or_runner_contract"
    if any(token in combined for token in (
        "assertionerror: not implemented",
        'assert false, "not implemented"',
        "assert false, 'not implemented'",
        "not implemented",
        "todo placeholder",
    )):
        return "test_template_not_implemented"
    if "test_case_implementation" in stage or "generate_random" in stage:
        return "test_logic_or_case_implementation"
    if "bug_analysis" in stage or "verification_review" in stage or "comprehensive_verification" in stage:
        return "bug_doc_schema_or_marking"
    if any(token in stage for token in ("env_fixture", "basic_api", "dut_creation")):
        return "api_or_env_interface_mismatch"
    if any(token in combined for token in ("assert", "mismatch", "expected", "actual")):
        return "test_assertion_or_expectation_mismatch"
    if any(token in combined for token in ("syntaxerror", "indentationerror", "parse", "invalid syntax")):
        return "syntax_or_format_error"
    if any(token in combined for token in ("interface mismatch", "api mismatch", "env interface")):
        return "api_or_env_interface_mismatch"
    return "generic_checker_failure"


def failure_signature_from_summary(event_type: str, stage_name: str, summary: Dict[str, Any]) -> str:
    """Return the same compact signature shape used by trace tree generation."""
    summary = summary if isinstance(summary, dict) else {}
    parts = [
        str(event_type or ""),
        str(stage_name or ""),
        str(summary.get("test_outcome") or ""),
        ",".join(_as_list(summary.get("checker_categories"), 4)),
        ",".join(_as_list(summary.get("failed_cases_top"), 4)),
        ",".join(_as_list(summary.get("failed_checkpoints_top"), 4)),
        "|".join(_as_list(summary.get("error_top"), 2)),
    ]
    raw = "\n".join(parts)
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:16]


class ContextReuseStore:
    """Read-only store for compact repair episodes and compression hints."""

    def __init__(self, workspace: str, dut_name: str, options: Optional[Dict[str, Any]] = None):
        self.workspace = os.path.abspath(workspace)
        self.dut_name = dut_name
        self.options = options or {}
        if hasattr(self.options, "as_dict"):
            self.options = self.options.as_dict()
        elif not isinstance(self.options, dict):
            self.options = {}

        self.pack_path = self._resolve_pack_path(
            self.options.get("pack_path", "benchmark/ucagent_context_reuse/context_reuse_v0.json")
        )
        self.prompt_limit = int(self.options.get("prompt_limit", 2) or 2)
        self.hint_limit = int(self.options.get("compression_hint_limit", 1) or 1)
        self.min_score = float(self.options.get("min_score", 0.95) or 0.95)
        self.min_quality_score = float(self.options.get("min_quality_score", 0.0) or 0.0)
        self.include_cross_dut = bool(self.options.get("include_cross_dut", True))
        self.prefer_same_dut = bool(self.options.get("prefer_same_dut", True))
        self.prefer_same_stage = bool(self.options.get("prefer_same_stage", True))
        self.strict_stage_role_match = bool(self.options.get("strict_stage_role_match", True))
        self.enable_failure_pattern_match = bool(self.options.get("enable_failure_pattern_match", True))
        self.strict_failure_pattern_match = bool(self.options.get("strict_failure_pattern_match", True))
        self.prefer_action_category_match = bool(self.options.get("prefer_action_category_match", True))
        self.penalize_action_mismatch = float(self.options.get("penalize_action_mismatch", 0.7) or 0.7)
        self.strict_action_category_match = _as_bool(
            self.options.get("strict_action_category_match", True),
            default=True,
        )
        self.strict_exact_pattern_groups = _as_bool(
            self.options.get("strict_exact_pattern_groups", True),
            default=True,
        )
        self.generic_requires_signature = _as_bool(
            self.options.get("generic_requires_signature", True),
            default=True,
        )
        self.skip_non_reusable_failures = _as_bool(
            self.options.get("skip_non_reusable_failures", True),
            default=True,
        )
        self.strict_compression_hint_match = _as_bool(
            self.options.get("strict_compression_hint_match", True),
            default=True,
        )
        self.destructive_requires_support = _as_bool(
            self.options.get("destructive_requires_support", True),
            default=True,
        )
        self.min_destructive_support = max(
            1,
            int(self.options.get("min_destructive_support", 2) or 2),
        )
        self.enable_utility_gate = _as_bool(self.options.get("enable_utility_gate", False))
        self.min_utility_score = float(self.options.get("min_utility_score", 0.0) or 0.0)
        self.utility_unknown_policy = str(
            self.options.get("utility_unknown_policy", "reject") or "reject"
        ).strip().lower()
        self.utility_prompt_token_reference = max(
            1.0,
            float(self.options.get("utility_prompt_token_reference", 512) or 512),
        )
        self.utility_prompt_token_weight = max(
            0.0,
            float(self.options.get("utility_prompt_token_weight", 0.05) or 0.05),
        )
        self.contract_policy_mode = str(
            self.options.get("contract_policy_mode", "off") or "off"
        ).strip().lower()
        if self.contract_policy_mode not in {"off", "shadow", "enforce"}:
            warning(
                f"[context_reuse] unknown contract_policy_mode={self.contract_policy_mode}; "
                "falling back to off"
            )
            self.contract_policy_mode = "off"
        self.min_contract_applicability = float(
            self.options.get("min_contract_applicability", 0.55) or 0.55
        )

        self.effective_items: List[Dict[str, Any]] = []
        self.compression_hints: List[Dict[str, Any]] = []
        self._runtime_metrics = {
            "queries": 0,
            "hits": 0,
            "misses": 0,
            "injected_items": 0,
            "hint_hits": 0,
            "utility_candidates": 0,
            "utility_passed": 0,
            "utility_rejected": 0,
            "utility_unknown": 0,
            "utility_prompt_tokens_estimated": 0,
            "hard_gate_rejected": 0,
            "hard_gate_queries": 0,
            "contract_candidates": 0,
            "contract_applicable": 0,
            "contract_inapplicable": 0,
            "contract_enforced_rejected": 0,
        }
        self._last_search_summary: Dict[str, Any] = {}
        self._load()

    def _resolve_pack_path(self, raw_path: str) -> str:
        path = os.path.expanduser(str(raw_path or ""))
        if os.path.isabs(path):
            return os.path.abspath(path)

        candidates = [
            os.path.join(self.workspace, path),
            os.path.join(os.getcwd(), path),
            os.path.join(os.path.dirname(self.workspace), path),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
        return os.path.abspath(candidates[0])

    def _load(self) -> None:
        if not os.path.exists(self.pack_path):
            warning(f"[context_reuse] pack not found: {self.pack_path}")
            return
        try:
            with open(self.pack_path, "r", encoding="utf-8") as fin:
                data = json.load(fin)
        except Exception as exc:
            warning(f"[context_reuse] failed to load {self.pack_path}: {exc}")
            return
        self.effective_items = data.get("effective_items", []) if isinstance(data.get("effective_items"), list) else []
        self.compression_hints = data.get("compression_hints", []) if isinstance(data.get("compression_hints"), list) else []
        for item in self.effective_items:
            self._normalize_item_metadata(item)
        for hint in self.compression_hints:
            self._normalize_hint_metadata(hint)
        info(
            f"[context_reuse] loaded pack={self.pack_path} "
            f"episodes={len(self.effective_items)} hints={len(self.compression_hints)} "
            f"utility_gate={self.enable_utility_gate} min_utility={self.min_utility_score} "
            f"contract_policy={self.contract_policy_mode}"
        )

    def _normalize_item_metadata(self, item: Dict[str, Any]) -> None:
        if not isinstance(item, dict):
            return
        failure = item.get("failure") if isinstance(item.get("failure"), dict) else {}
        pattern = item.get("failure_pattern") or failure.get("failure_pattern")
        if not pattern:
            pattern = classify_failure_pattern(
                failure,
                stage_name=str(item.get("stage_name") or ""),
                event_type=str(failure.get("event_type") or ""),
            )
        item["failure_pattern"] = pattern
        item["failure_group"] = item.get("failure_group") or failure_pattern_group(pattern)
        failure.setdefault("failure_pattern", pattern)
        failure.setdefault("failure_group", item["failure_group"])
        item["failure"] = failure
        actions = item.get("actions") if isinstance(item.get("actions"), list) else []
        action_categories = item.get("action_categories")
        if not isinstance(action_categories, list) or not action_categories:
            action_categories = []
            for action in actions:
                if isinstance(action, dict) and action.get("path_category"):
                    category = str(action.get("path_category"))
                    if category not in action_categories:
                        action_categories.append(category)
        item["action_categories"] = action_categories
        preferred = item.get("preferred_action_categories")
        if not isinstance(preferred, list) or not preferred:
            item["preferred_action_categories"] = preferred_action_categories(pattern)
        contract = item.get("transition_contract")
        if not isinstance(contract, dict) or not contract:
            item["transition_contract"] = build_transition_contract(item)

    def _normalize_hint_metadata(self, hint: Dict[str, Any]) -> None:
        if not isinstance(hint, dict):
            return
        pattern = hint.get("failure_pattern")
        if not pattern:
            pattern = classify_failure_pattern(
                hint.get("failure", {}) if isinstance(hint.get("failure"), dict) else hint,
                stage_name=str(hint.get("stage_name") or ""),
            )
        hint["failure_pattern"] = pattern
        hint["failure_group"] = hint.get("failure_group") or failure_pattern_group(pattern)

    def get_runtime_metrics(self) -> Dict[str, Any]:
        metrics = dict(self._runtime_metrics)
        queries = max(1, int(metrics.get("queries", 0) or 0))
        metrics["hit_rate"] = round(int(metrics.get("hits", 0) or 0) / queries, 4)
        utility_candidates = max(1, int(metrics.get("utility_candidates", 0) or 0))
        metrics["utility_pass_rate"] = round(
            int(metrics.get("utility_passed", 0) or 0) / utility_candidates,
            4,
        )
        return metrics

    def get_last_search_summary(self) -> Dict[str, Any]:
        return dict(self._last_search_summary)

    def _estimate_prompt_tokens(self, item: Dict[str, Any]) -> int:
        failure = item.get("failure") if isinstance(item.get("failure"), dict) else {}
        actions = item.get("actions") if isinstance(item.get("actions"), list) else []
        compact = {
            "stage": item.get("stage_name"),
            "failure_pattern": item.get("failure_pattern"),
            "failed_cases_top": _as_list(failure.get("failed_cases_top"), 3),
            "failed_checkpoints_top": _as_list(failure.get("failed_checkpoints_top"), 3),
            "error_top": [_short(value, 180) for value in _as_list(failure.get("error_top"), 2)],
            "actions": [
                {
                    "operation": action.get("operation"),
                    "path": action.get("path"),
                    "path_category": action.get("path_category"),
                }
                for action in actions[:3]
                if isinstance(action, dict)
            ],
        }
        chars = len(json.dumps(compact, ensure_ascii=False, sort_keys=True))
        return max(1, (chars + 3) // 4)

    def _utility_gate(self, item: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        if not self.enable_utility_gate:
            return True, {
                "enabled": False,
                "passed": True,
                "reason": "utility_gate_disabled",
            }

        self._runtime_metrics["utility_candidates"] += 1
        utility = item.get("verifier_utility") if isinstance(item.get("verifier_utility"), dict) else {}
        try:
            historical_score = float(utility.get("gate_score"))
        except (TypeError, ValueError):
            historical_score = None
        estimated_tokens = self._estimate_prompt_tokens(item)
        self._runtime_metrics["utility_prompt_tokens_estimated"] += estimated_tokens
        if historical_score is None:
            self._runtime_metrics["utility_unknown"] += 1
            passed = self.utility_unknown_policy == "allow"
            self._runtime_metrics["utility_passed" if passed else "utility_rejected"] += 1
            return passed, {
                "enabled": True,
                "passed": passed,
                "reason": f"utility_unknown_{self.utility_unknown_policy}",
                "historical_score": None,
                "prompt_tokens_estimated": estimated_tokens,
                "prompt_cost": None,
                "net_score": None,
                "threshold": self.min_utility_score,
                "policy_version": utility.get("policy_version"),
            }

        prompt_cost = self.utility_prompt_token_weight * min(
            1.0,
            estimated_tokens / self.utility_prompt_token_reference,
        )
        net_score = historical_score - prompt_cost
        passed = net_score > self.min_utility_score
        self._runtime_metrics["utility_passed" if passed else "utility_rejected"] += 1
        return passed, {
            "enabled": True,
            "passed": passed,
            "reason": "positive_net_utility" if passed else "non_positive_net_utility",
            "historical_score": round(historical_score, 4),
            "prompt_tokens_estimated": estimated_tokens,
            "prompt_cost": round(prompt_cost, 4),
            "net_score": round(net_score, 4),
            "threshold": self.min_utility_score,
            "policy_version": utility.get("policy_version"),
            "mean_components": utility.get("mean_components", {}),
        }

    @staticmethod
    def _item_action_categories(item: Dict[str, Any]) -> List[str]:
        categories = item.get("action_categories")
        if isinstance(categories, list) and categories:
            return [str(category) for category in categories if str(category)]
        actions = item.get("actions") if isinstance(item.get("actions"), list) else []
        return [
            str(action.get("path_category"))
            for action in actions
            if isinstance(action, dict) and action.get("path_category")
        ]

    def _hard_gate_reason(
        self,
        item: Dict[str, Any],
        failure: Dict[str, Any],
        stage_name: str,
    ) -> str:
        current_pattern = str(failure.get("failure_pattern") or "")
        current_group = str(failure.get("failure_group") or failure_pattern_group(current_pattern))
        item_failure = item.get("failure") if isinstance(item.get("failure"), dict) else {}
        item_pattern = str(item.get("failure_pattern") or item_failure.get("failure_pattern") or "")

        if self.skip_non_reusable_failures and current_pattern in NON_REUSABLE_FAILURE_PATTERNS:
            return "non_reusable_failure"

        if self.generic_requires_signature and current_pattern == "generic_checker_failure":
            current_signature = failure.get("failure_signature")
            item_signature = item_failure.get("failure_signature")
            if not current_signature or current_signature != item_signature:
                return "generic_signature_mismatch"

        if (
            self.strict_exact_pattern_groups
            and current_group in EXACT_PATTERN_GROUPS
            and current_pattern != item_pattern
        ):
            return "exact_failure_pattern_mismatch"

        if self.strict_action_category_match:
            preferred = preferred_action_categories_for_failure(failure, stage_name=stage_name)
            item_actions = self._item_action_categories(item)
            if preferred and not set(preferred).intersection(item_actions):
                return "action_category_mismatch"

        if self.destructive_requires_support:
            actions = item.get("actions") if isinstance(item.get("actions"), list) else []
            destructive = any(
                isinstance(action, dict)
                and str(action.get("operation") or "") in {"delete_file", "write", "move"}
                for action in actions
            )
            support_count = int(item.get("support_count", 0) or 0)
            same_signature = bool(
                failure.get("failure_signature")
                and failure.get("failure_signature") == item_failure.get("failure_signature")
            )
            if destructive and support_count < self.min_destructive_support and not same_signature:
                return "low_support_destructive_action"

        return ""

    def search(
        self,
        failure: Dict[str, Any],
        stage_index: Optional[int] = None,
        stage_name: str = "",
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        failure = failure if isinstance(failure, dict) else {}
        self._ensure_failure_metadata(failure, stage_name=stage_name)
        limit = int(limit or self.prompt_limit)
        self._runtime_metrics["queries"] += 1
        search_summary = {
            "utility_gate_enabled": self.enable_utility_gate,
            "quality_candidates": 0,
            "match_candidates": 0,
            "utility_passed": 0,
            "utility_rejected": 0,
            "utility_unknown": 0,
            "hard_gate_rejected": 0,
            "hard_gate_reasons": {},
            "returned": 0,
        }
        if self.contract_policy_mode != "off":
            search_summary.update({
                "contract_policy_mode": self.contract_policy_mode,
                "contract_candidates": 0,
                "contract_applicable": 0,
                "contract_inapplicable": 0,
                "contract_enforced_rejected": 0,
            })
        current_pattern = str(failure.get("failure_pattern") or "")
        if self.skip_non_reusable_failures and current_pattern in NON_REUSABLE_FAILURE_PATTERNS:
            search_summary["hard_gate_rejected"] = len(self.effective_items)
            search_summary["hard_gate_reasons"] = {"non_reusable_failure": len(self.effective_items)}
            self._runtime_metrics["hard_gate_rejected"] += len(self.effective_items)
            self._runtime_metrics["hard_gate_queries"] += 1
            self._runtime_metrics["misses"] += 1
            self._last_search_summary = search_summary
            return []
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for item in self.effective_items:
            if self.strict_stage_role_match and not stage_roles_compatible(
                stage_name, str(item.get("stage_name") or "")
            ):
                continue
            if not self.include_cross_dut and item.get("dut") != self.dut_name:
                source_duts = item.get("source_duts") if isinstance(item.get("source_duts"), list) else []
                if self.dut_name not in source_duts:
                    continue
            try:
                quality_score = float(item.get("quality_score", 0.0) or 0.0)
            except (TypeError, ValueError):
                quality_score = 0.0
            if quality_score < self.min_quality_score:
                continue
            search_summary["quality_candidates"] += 1
            hard_gate_reason = self._hard_gate_reason(item, failure, stage_name)
            if hard_gate_reason:
                search_summary["hard_gate_rejected"] += 1
                reasons = search_summary["hard_gate_reasons"]
                reasons[hard_gate_reason] = int(reasons.get(hard_gate_reason, 0) or 0) + 1
                self._runtime_metrics["hard_gate_rejected"] += 1
                continue
            contract_result = {
                "applicable": None,
                "score": None,
                "threshold": self.min_contract_applicability,
                "reasons": [],
                "hard_mismatches": [],
                "mode": self.contract_policy_mode,
            }
            if self.contract_policy_mode != "off":
                search_summary["contract_candidates"] += 1
                self._runtime_metrics["contract_candidates"] += 1
                contract_result = contract_applicability(
                    item.get("transition_contract", {}),
                    failure,
                    stage_name=stage_name,
                    min_score=self.min_contract_applicability,
                )
                contract_result["mode"] = self.contract_policy_mode
                result_key = "contract_applicable" if contract_result["applicable"] else "contract_inapplicable"
                search_summary[result_key] += 1
                self._runtime_metrics[result_key] += 1
                if self.contract_policy_mode == "enforce" and not contract_result["applicable"]:
                    search_summary["contract_enforced_rejected"] += 1
                    self._runtime_metrics["contract_enforced_rejected"] += 1
                    continue
            score, reasons = self._score_item(item, failure, stage_index, stage_name)
            if score < self.min_score:
                continue
            search_summary["match_candidates"] += 1
            utility_passed, utility = self._utility_gate(item)
            if utility.get("historical_score") is None and utility.get("enabled"):
                search_summary["utility_unknown"] += 1
            if not utility_passed:
                search_summary["utility_rejected"] += 1
                continue
            if utility.get("enabled"):
                search_summary["utility_passed"] += 1
            scored.append((score, {
                **item,
                "__context_reuse__": {
                    "score": round(score, 4),
                    "reasons": reasons,
                    "utility": utility,
                    "contract_policy": contract_result,
                },
            }))

        if int(search_summary.get("hard_gate_rejected", 0) or 0) > 0:
            self._runtime_metrics["hard_gate_queries"] += 1

        scored.sort(
            key=lambda pair: (
                pair[0],
                float(((pair[1].get("__context_reuse__") or {}).get("utility") or {}).get("net_score") or 0.0),
                1 if pair[1].get("dut") == self.dut_name else 0,
                int(pair[1].get("stage_index") == stage_index),
            ),
            reverse=True,
        )
        results = []
        seen_keys = set()
        for _, item in scored:
            actions = item.get("actions") if isinstance(item.get("actions"), list) else []
            item_failure = item.get("failure") if isinstance(item.get("failure"), dict) else {}
            dedup_key = (
                item.get("strategy_id") or (
                    item.get("dut"),
                    item.get("stage_name"),
                    item_failure.get("failure_signature"),
                    tuple((action.get("operation"), action.get("path")) for action in actions[:3] if isinstance(action, dict)),
                )
            )
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            results.append(item)
            if len(results) >= limit:
                break
        if results:
            self._runtime_metrics["hits"] += 1
            self._runtime_metrics["injected_items"] += len(results)
        else:
            self._runtime_metrics["misses"] += 1
        search_summary["returned"] = len(results)
        self._last_search_summary = search_summary
        return results

    def compression_hints_for(
        self,
        failure: Dict[str, Any],
        stage_index: Optional[int] = None,
        stage_name: str = "",
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        failure = failure if isinstance(failure, dict) else {}
        self._ensure_failure_metadata(failure, stage_name=stage_name)
        limit = int(limit or self.hint_limit)
        sig = failure.get("failure_signature")
        pattern = failure.get("failure_pattern")
        group = failure.get("failure_group")
        if self.skip_non_reusable_failures and pattern in NON_REUSABLE_FAILURE_PATTERNS:
            return []
        scored = []
        for hint in self.compression_hints:
            if self.strict_stage_role_match and not stage_roles_compatible(
                stage_name, str(hint.get("stage_name") or "")
            ):
                continue
            if not self.include_cross_dut and hint.get("dut") != self.dut_name:
                continue
            if self.strict_compression_hint_match:
                same_signature = bool(sig and hint.get("failure_signature") == sig)
                same_pattern_and_stage = bool(
                    pattern
                    and pattern != "generic_checker_failure"
                    and hint.get("failure_pattern") == pattern
                    and stage_name
                    and hint.get("stage_name") == stage_name
                )
                if not (same_signature or same_pattern_and_stage):
                    continue
            score = 0.0
            reasons = []
            if sig and hint.get("failure_signature") == sig:
                score += 2.0
                reasons.append("same_failure_signature")
            if self.enable_failure_pattern_match and pattern:
                hint_pattern = hint.get("failure_pattern")
                hint_group = hint.get("failure_group")
                if hint_pattern == pattern:
                    score += 0.65
                    reasons.append("same_failure_pattern")
                elif group and hint_group == group:
                    score += 0.25
                    reasons.append("same_failure_group")
                elif (
                    self.strict_failure_pattern_match
                    and not (sig and hint.get("failure_signature") == sig)
                    and group not in ("generic", None, "")
                    and hint_group not in ("generic", None, "")
                ):
                    continue
            if stage_name and hint.get("stage_name") == stage_name:
                score += 0.45
                reasons.append("same_stage_name")
            if stage_index is not None and hint.get("stage_index") == stage_index:
                score += 0.15
                reasons.append("same_stage_index")
            if hint.get("dut") == self.dut_name:
                score += 0.25
                reasons.append("same_dut")
            if score < 0.85:
                continue
            scored.append((score, {**hint, "__context_reuse__": {"score": round(score, 4), "reasons": reasons}}))
        scored.sort(key=lambda pair: (pair[0], int(pair[1].get("repeat_count", 0) or 0)), reverse=True)
        hints = [item for _, item in scored[:limit]]
        if hints:
            self._runtime_metrics["hint_hits"] += 1
        return hints

    def _ensure_failure_metadata(self, failure: Dict[str, Any], stage_name: str = "") -> None:
        if not isinstance(failure, dict):
            return
        pattern = failure.get("failure_pattern")
        if not pattern:
            pattern = classify_failure_pattern(
                failure,
                stage_name=stage_name or str(failure.get("stage_name") or ""),
                event_type=str(failure.get("event_type") or ""),
            )
            failure["failure_pattern"] = pattern
        failure["failure_group"] = failure.get("failure_group") or failure_pattern_group(str(pattern))
        preferred = failure.get("preferred_action_categories")
        if not isinstance(preferred, list) or not preferred:
            failure["preferred_action_categories"] = preferred_action_categories_for_failure(
                failure,
                stage_name=stage_name,
            )

    def _score_item(
        self,
        item: Dict[str, Any],
        failure: Dict[str, Any],
        stage_index: Optional[int],
        stage_name: str,
    ) -> Tuple[float, List[str]]:
        self._ensure_failure_metadata(failure, stage_name=stage_name)
        item_failure = item.get("failure") if isinstance(item.get("failure"), dict) else {}
        score = 0.0
        reasons: List[str] = []
        source_duts = item.get("source_duts") if isinstance(item.get("source_duts"), list) else []
        if self.prefer_same_dut and (item.get("dut") == self.dut_name or self.dut_name in source_duts):
            score += 0.25
            reasons.append("same_dut")
        if self.prefer_same_stage and stage_name and item.get("stage_name") == stage_name:
            score += 0.55
            reasons.append("same_stage_name")
        if stage_index is not None and item.get("stage_index") == stage_index:
            score += 0.15
            reasons.append("same_stage_index")
        if failure.get("failure_signature") and item_failure.get("failure_signature") == failure.get("failure_signature"):
            score += 2.0
            reasons.append("same_failure_signature")

        if self.enable_failure_pattern_match:
            current_pattern = failure.get("failure_pattern")
            item_pattern = item.get("failure_pattern") or item_failure.get("failure_pattern")
            current_group = failure.get("failure_group")
            item_group = item.get("failure_group") or item_failure.get("failure_group")
            if current_pattern and item_pattern and current_pattern == item_pattern:
                score += 0.75
                reasons.append("same_failure_pattern")
            elif current_group and item_group and current_group == item_group:
                score += 0.35
                reasons.append("same_failure_group")
            elif self.strict_failure_pattern_match and current_group not in ("generic", None, "") and item_group not in ("generic", None, ""):
                score -= 0.65
                reasons.append(f"failure_group_mismatch={current_group}!={item_group}")

        if self.prefer_action_category_match:
            preferred = failure.get("preferred_action_categories") or preferred_action_categories_for_failure(
                failure,
                stage_name=stage_name,
            )
            item_actions = item.get("action_categories")
            if not isinstance(item_actions, list) or not item_actions:
                actions = item.get("actions") if isinstance(item.get("actions"), list) else []
                item_actions = [
                    str(action.get("path_category"))
                    for action in actions
                    if isinstance(action, dict) and action.get("path_category")
                ]
            overlap = [category for category in item_actions if category in preferred]
            if preferred and overlap:
                first_rank = min(preferred.index(category) for category in overlap)
                bonus = max(0.25, 0.55 - 0.15 * first_rank)
                score += bonus
                reasons.append("action_category_match=" + ",".join(overlap[:3]))
            elif preferred and item_actions:
                score -= self.penalize_action_mismatch
                reasons.append("action_category_mismatch")

        checkpoint_overlap = _token_overlap_score(
            _as_list(failure.get("failed_checkpoints_top"), 8),
            _as_list(item_failure.get("failed_checkpoints_top"), 8),
        )
        if checkpoint_overlap > 0:
            score += checkpoint_overlap * 0.6
            reasons.append(f"checkpoint_overlap={checkpoint_overlap:.2f}")

        case_overlap = _token_overlap_score(
            _as_list(failure.get("failed_cases_top"), 8),
            _as_list(item_failure.get("failed_cases_top"), 8),
        )
        if case_overlap > 0:
            score += case_overlap * 0.35
            reasons.append(f"case_overlap={case_overlap:.2f}")

        error_overlap = _token_overlap_score(
            _as_list(failure.get("error_top"), 3),
            _as_list(item_failure.get("error_top"), 3),
        )
        if error_overlap > 0:
            score += error_overlap * 0.35
            reasons.append(f"error_overlap={error_overlap:.2f}")

        current_categories = set(_as_list(failure.get("checker_categories"), 5))
        item_categories = set(_as_list(item_failure.get("checker_categories"), 5))
        if current_categories and item_categories:
            overlap = len(current_categories & item_categories) / max(1, len(current_categories | item_categories))
            if overlap > 0:
                score += overlap * 0.3
                reasons.append(f"category_overlap={overlap:.2f}")

        try:
            quality_score = float(item.get("quality_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            quality_score = 0.0
        if quality_score > 0:
            quality_bonus = max(0.0, min(0.3, (quality_score - 0.5) * 0.6))
            score += quality_bonus
            reasons.append(f"quality={quality_score:.2f}")
        support_count = int(item.get("support_count", 0) or 0)
        if support_count > 1:
            support_bonus = min(0.15, 0.03 * (support_count - 1))
            score += support_bonus
            reasons.append(f"support={support_count}")

        return score, reasons

    def render_episode(self, item: Dict[str, Any], max_actions: int = 3) -> Dict[str, Any]:
        failure = item.get("failure") if isinstance(item.get("failure"), dict) else {}
        success = item.get("success") if isinstance(item.get("success"), dict) else {}
        actions = item.get("actions") if isinstance(item.get("actions"), list) else []
        rendered = {
            "strategy_id": item.get("strategy_id"),
            "source_dut": item.get("dut"),
            "source_duts": _as_list(item.get("source_duts"), 8),
            "stage": item.get("stage_name"),
            "score": (item.get("__context_reuse__") or {}).get("score"),
            "quality_score": item.get("quality_score"),
            "support_count": item.get("support_count"),
            "match_reasons": (item.get("__context_reuse__") or {}).get("reasons", [])[:4],
            "failure_pattern": item.get("failure_pattern") or failure.get("failure_pattern"),
            "failure_group": item.get("failure_group") or failure.get("failure_group"),
            "action_categories": _as_list(item.get("action_categories"), 5),
            "failure": {
                "signature": failure.get("failure_signature"),
                "pattern": failure.get("failure_pattern"),
                "failed_cases_top": _as_list(failure.get("failed_cases_top"), 3),
                "failed_checkpoints_top": _as_list(failure.get("failed_checkpoints_top"), 3),
                "error_top": [_short(item, 180) for item in _as_list(failure.get("error_top"), 2)],
            },
            "effective_actions": [
                {
                    "operation": action.get("operation"),
                    "path": action.get("path"),
                    "path_category": action.get("path_category"),
                }
                for action in actions[:max_actions]
                if isinstance(action, dict)
            ],
            "success": {
                "event_type": success.get("event_type"),
                "check_pass": success.get("check_pass"),
                "tests_passed_all": success.get("tests_passed_all"),
            },
        }
        # Shadow mode must not alter the LLM input.  Contract details become
        # prompt-visible only in an explicit enforcement experiment.
        if self.contract_policy_mode == "enforce":
            rendered["transition_contract"] = item.get("transition_contract", {})
            rendered["contract_policy"] = (
                (item.get("__context_reuse__") or {}).get("contract_policy") or {}
            )
        return rendered

    def render_hint(self, hint: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "source_dut": hint.get("dut"),
            "stage": hint.get("stage_name"),
            "failure_signature": hint.get("failure_signature"),
            "failure_pattern": hint.get("failure_pattern"),
            "failure_group": hint.get("failure_group"),
            "repeat_count": hint.get("repeat_count"),
            "policy": hint.get("policy"),
            "match_reasons": (hint.get("__context_reuse__") or {}).get("reasons", [])[:4],
        }
