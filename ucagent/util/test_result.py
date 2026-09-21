# -*- coding: utf-8 -*-
"""Conservative classification for pytest/UnityChip test execution results."""

from __future__ import annotations

import re
from typing import Any, Mapping


TEST_OUTCOME_PASS = "pass"
TEST_OUTCOME_FAILURE = "test_failure"
TEST_OUTCOME_INFRASTRUCTURE_ERROR = "infrastructure_error"
TEST_OUTCOME_TIMEOUT = "timeout"
TEST_OUTCOME_CRASH = "crash"
TEST_OUTCOME_NO_TESTS = "no_tests_collected"

TEST_OUTCOMES = {
    TEST_OUTCOME_PASS,
    TEST_OUTCOME_FAILURE,
    TEST_OUTCOME_INFRASTRUCTURE_ERROR,
    TEST_OUTCOME_TIMEOUT,
    TEST_OUTCOME_CRASH,
    TEST_OUTCOME_NO_TESTS,
}


_TIMEOUT_RE = re.compile(
    r"(?:\btimed?\s+out\b|\btimeoutexpired\b|\btimeout\s+expired\b|"
    r"\btime\s+limit\s+exceeded\b|\btimeout\s*\(>\s*\d)",
    re.IGNORECASE,
)
_CRASH_RE = re.compile(
    r"(?:segmentation\s+fault|fatal\s+python\s+error|core\s+dumped|"
    r"terminated\s+by\s+signal|signal\s+11|exited\s+with\s+code\s+-\d+|"
    r"process[^\n]*(?:aborted|killed)|out\s+of\s+memory|cuda\s+out\s+of\s+memory)",
    re.IGNORECASE,
)
_INFRASTRUCTURE_RE = re.compile(
    r"(?:error\s+collecting|errors?\s+during\s+collection|"
    r"importerror|modulenotfounderror|no\s+collectors|"
    r"file\s+or\s+directory\s+not\s+found|unrecognized\s+arguments|"
    r"internalerror|permission\s+denied|command\s+not\s+found|"
    r"failed\s+to\s+(?:generate|load)\s+(?:the\s+)?report|"
    r"error\s+at\s+(?:setup|teardown)|traceback\s+\(most\s+recent\s+call\s+last\))",
    re.IGNORECASE,
)
_NO_TESTS_RE = re.compile(
    r"(?:no\s+tests\s+ran|no\s+tests\s+collected|collected\s+0\s+items|0\s+tests\s+collected)",
    re.IGNORECASE,
)
_PYTEST_FAILURE_RE = re.compile(
    r"(?:\b\d+\s+failed\b|={2,}\s*failures?\s*={2,})",
    re.IGNORECASE,
)
_PYTEST_XFAIL_RE = re.compile(r"\b(?P<count>\d+)\s+xfailed\b", re.IGNORECASE)

_PASS_STATUSES = {"PASS", "PASSED"}
_EXPECTED_FAILURE_STATUSES = {"XFAIL", "XFAILED"}
_INFRASTRUCTURE_STATUSES = {"ERROR"}

_DOCUMENTABLE_FAILURE_STAGES = {
    "basic_api_functional_test",
    "test_case_implementation_in_batch",
    "comprehensive_verification_and_bug_analysis",
    "refine_test_cases_based_on_functional_points",
    "static_bug_validation",
    "line_coverage_analysis_and_improvement",
    "generate_random_test_cases",
    "verification_review_and_summary",
}

_CHECKER_ONLY_STAGES = {
    "static_bug_analysis",
}


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_nonempty_list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    return [item for item in value if str(item).strip()]


def _diagnostic_text(summary: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("message", "stderr", "stdout", "error", "exit_detail"):
        value = summary.get(key)
        if value not in (None, ""):
            parts.append(str(value))
    for key in ("error_top",):
        value = summary.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value if str(item).strip())
    return "\n".join(parts)


def stage_allows_expected_failures(stage_name: str | None) -> bool:
    """Return whether expected pytest failures are required by the stage."""
    return str(stage_name or "").strip().lower() == "create_test_case_templates"


def stage_allows_documentable_failures(stage_name: str | None) -> bool:
    """Return whether ordinary assertion failures can be valid DUT evidence.

    This does not make a failed test a successful stage result. The stage
    Checker remains authoritative and must verify the test, source analysis,
    coverage labels, and bug-document evidence chain. XFAIL/SKIP are not
    documentable failures because they hide the explicit failing assertion.
    """
    return str(stage_name or "").strip().lower() in _DOCUMENTABLE_FAILURE_STAGES


def stage_uses_checker_only_validation(stage_name: str | None) -> bool:
    """Return whether the stage contract explicitly forbids test execution."""
    return str(stage_name or "").strip().lower() in _CHECKER_ONLY_STAGES


def _status_counts(summary: Mapping[str, Any]) -> dict[str, int]:
    raw = summary.get("test_status_counts")
    if not isinstance(raw, Mapping):
        return {}
    counts: dict[str, int] = {}
    for status, value in raw.items():
        count = _as_int(value)
        if count is not None and count > 0:
            counts[str(status).strip().upper()] = count
    return counts


def classify_test_outcome(
    summary: Mapping[str, Any] | None,
    *,
    allow_expected_failures: bool = False,
) -> tuple[str, str]:
    """Classify a compact test summary into one of six explicit outcomes.

    Passing is deliberately strict: at least one test must have run, no failed
    case may be present, and either the runner or a legacy summary must provide
    an explicit success signal. All ambiguous states are infrastructure errors.
    """
    data = summary if isinstance(summary, Mapping) else {}
    text = _diagnostic_text(data)
    total = _as_int(data.get("tests_total"))
    failed = _as_int(data.get("tests_failed"))
    failed_cases = _as_nonempty_list(data.get("failed_cases_top"))
    run_success = data.get("run_test_success")
    legacy_pass = data.get("tests_passed_all") is True
    status_counts = _status_counts(data)

    explicit_outcome = str(data.get("test_outcome") or "")
    if explicit_outcome == TEST_OUTCOME_PASS and not allow_expected_failures and not (
        total is not None and total > 0 and failed == 0 and not failed_cases
    ):
        return TEST_OUTCOME_INFRASTRUCTURE_ERROR, "inconsistent_explicit_pass"

    if _TIMEOUT_RE.search(text):
        return TEST_OUTCOME_TIMEOUT, "timeout_marker"
    if _CRASH_RE.search(text):
        return TEST_OUTCOME_CRASH, "crash_marker"
    if total in (None, 0) and _INFRASTRUCTURE_RE.search(text):
        return TEST_OUTCOME_INFRASTRUCTURE_ERROR, "infrastructure_error_before_collection"
    if total == 0 or _NO_TESTS_RE.search(text):
        return TEST_OUTCOME_NO_TESTS, "zero_tests_collected"

    # Template creation deliberately produces XFAIL. This is a valid result
    # only for that stage; in implementation stages the same status remains a
    # failure so placeholders cannot silently survive.
    if allow_expected_failures and total is not None and total > 0 and failed == 0 and run_success is True:
        xfail_from_status = sum(status_counts.get(status, 0) for status in _EXPECTED_FAILURE_STATUSES)
        xfail_from_text = sum(int(match.group("count")) for match in _PYTEST_XFAIL_RE.finditer(text))
        unexpected_statuses = {
            status: count
            for status, count in status_counts.items()
            if status not in _PASS_STATUSES | _EXPECTED_FAILURE_STATUSES
        }
        if max(xfail_from_status, xfail_from_text) > 0 and not unexpected_statuses and not _PYTEST_FAILURE_RE.search(text):
            return TEST_OUTCOME_PASS, "expected_template_xfail"

    non_pass_statuses = {
        status: count
        for status, count in status_counts.items()
        if status not in _PASS_STATUSES
    }
    infrastructure_statuses = {
        status: count
        for status, count in non_pass_statuses.items()
        if status in _INFRASTRUCTURE_STATUSES
    }
    if infrastructure_statuses:
        return TEST_OUTCOME_INFRASTRUCTURE_ERROR, "reported_infrastructure_status"
    if non_pass_statuses:
        return TEST_OUTCOME_FAILURE, "reported_non_pass_status"
    if (failed is not None and failed > 0) or failed_cases or _PYTEST_FAILURE_RE.search(text):
        return TEST_OUTCOME_FAILURE, "reported_failed_tests"
    if _INFRASTRUCTURE_RE.search(text):
        return TEST_OUTCOME_INFRASTRUCTURE_ERROR, "infrastructure_error_marker"
    if explicit_outcome in TEST_OUTCOMES:
        return explicit_outcome, str(data.get("test_outcome_reason") or "explicit_outcome")

    has_verified_counts = total is not None and total > 0 and failed == 0 and not failed_cases
    if has_verified_counts and (run_success is True or (run_success is None and legacy_pass)):
        return TEST_OUTCOME_PASS, "verified_nonempty_pass"
    if run_success is False:
        return TEST_OUTCOME_INFRASTRUCTURE_ERROR, "runner_failed_without_test_failures"
    return TEST_OUTCOME_INFRASTRUCTURE_ERROR, "insufficient_success_evidence"


def apply_test_outcome(
    summary: dict[str, Any],
    *,
    allow_expected_failures: bool = False,
) -> dict[str, Any]:
    """Attach a normalized outcome and make pass/fail semantics unambiguous."""
    outcome, reason = classify_test_outcome(
        summary,
        allow_expected_failures=allow_expected_failures,
    )
    summary["test_outcome"] = outcome
    summary["test_outcome_reason"] = reason
    summary["tests_passed_all"] = outcome == TEST_OUTCOME_PASS
    return summary
