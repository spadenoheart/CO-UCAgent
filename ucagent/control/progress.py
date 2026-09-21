# -*- coding: utf-8 -*-
"""Verifier-grounded progress control for long-running repair loops."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any, Mapping

from ucagent.memory.context_reuse import failure_signature_from_summary
from ucagent.util.test_result import (
    TEST_OUTCOME_CRASH,
    TEST_OUTCOME_INFRASTRUCTURE_ERROR,
    TEST_OUTCOME_NO_TESTS,
    TEST_OUTCOME_PASS,
    TEST_OUTCOME_TIMEOUT,
    stage_allows_documentable_failures,
    stage_allows_expected_failures,
    stage_uses_checker_only_validation,
)


INVALID_TEST_OUTCOMES = {
    TEST_OUTCOME_INFRASTRUCTURE_ERROR,
    TEST_OUTCOME_TIMEOUT,
    TEST_OUTCOME_CRASH,
    TEST_OUTCOME_NO_TESTS,
}


PROGRESS_METRIC_PATTERNS = (
    (
        "unmarked_test_functions",
        re.compile(
            r"\bFind\s+(\d+)\s+functions?\s+do not have correct check point marks\b",
            re.IGNORECASE,
        ),
    ),
    (
        "unmarked_check_points",
        re.compile(r"\bunmarked_check_points\b[^0-9]{0,20}(\d+)", re.IGNORECASE),
    ),
)


def _stable_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _short_hash(value: Any) -> str:
    return hashlib.sha256(_stable_text(value).encode("utf-8")).hexdigest()[:16]


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


def _string_items(value: Any, limit: int = 20) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item)[:500] for item in value[:limit] if str(item).strip()]


def _progress_metrics(result: Mapping[str, Any]) -> dict[str, int]:
    """Extract lower-is-better Checker counts from human-readable evidence."""
    metrics: dict[str, int] = {}
    for text in _string_items(result.get("error_top")):
        for name, pattern in PROGRESS_METRIC_PATTERNS:
            match = pattern.search(text)
            if match:
                metrics[name] = int(match.group(1))
    return metrics


def _failure_set(
    result: Mapping[str, Any],
    signature: str,
    *,
    use_fallback: bool = True,
) -> set[str]:
    items: set[str] = set()
    for key, prefix in (
        ("failed_cases_top", "case"),
        ("failed_checkpoints_top", "checkpoint"),
        ("checker_categories", "checker"),
    ):
        for item in _string_items(result.get(key)):
            # Source edits routinely move a pytest function without changing
            # the failing node. Treating only that line-range churn as a
            # resolved+introduced failure incorrectly rewards no-op rewrites.
            if prefix == "case":
                item = _PYTEST_LINE_RANGE_RE.sub(r"\g<file>::", item)
            items.add(f"{prefix}:{item}")
    items.update(
        f"metric:{name}:{value}"
        for name, value in _progress_metrics(result).items()
    )
    if not items and signature and use_fallback:
        items.add(f"signature:{signature}")
    return items


_PYTEST_LINE_RANGE_RE = re.compile(r"(?P<file>\.py):\d+(?:-\d+)?::")


def _semantic_checker_state_key(failures: set[str], stage_name: str) -> str:
    """Track verifier progress without treating category churn as progress.

    In documentable-failure stages, ordinary FAILED cases are expected to
    remain while their evidence is recorded. Line numbers and checker category
    names can change after edits even when the underlying failed tests do not.
    """
    normalized = {
        _PYTEST_LINE_RANGE_RE.sub(r"\g<file>::", item)
        for item in failures
    }
    if stage_allows_documentable_failures(stage_name):
        evidence = {
            item for item in normalized
            if item.startswith(("case:", "checkpoint:", "contract:"))
        }
        if evidence:
            return _short_hash(sorted(evidence))
    return _short_hash(sorted(normalized))


class ProgressAwareLoopController:
    """Convert verifier deltas into repair policy and enforceable budgets.

    Checker state is authoritative. Targeted tests provide local evidence but
    cannot clear a repeated Checker failure or stage-level stagnation.
    """

    def __init__(self, options: Mapping[str, Any] | None = None):
        cfg = dict(options or {})
        self.enabled = _as_bool(cfg.get("enabled", True), True)
        self.hard_enforce_mutation_budget = _as_bool(
            cfg.get("hard_enforce_mutation_budget", False), False
        )
        self.no_progress_patience = max(1, int(cfg.get("no_progress_patience", 2) or 2))
        self.max_actions = {
            "exploit": max(1, int(cfg.get("max_mutations_progress", 3) or 3)),
            "diagnose": max(1, int(cfg.get("max_mutations_diagnostic", 2) or 2)),
            "pivot": max(1, int(cfg.get("max_mutations_pivot", 1) or 1)),
            "contain": max(1, int(cfg.get("max_mutations_regression", 1) or 1)),
            "repair_execution": max(0, int(cfg.get("max_mutations_invalid", 3) or 0)),
            "restore_template_contract": max(
                1, int(cfg.get("max_mutations_template_contract", 8) or 8)
            ),
            "repair_marking_contract": max(
                1, int(cfg.get("max_mutations_marking_contract", 16) or 16)
            ),
            "reconcile_checkpoint_evidence": max(
                1, int(cfg.get("max_mutations_checkpoint_evidence", 12) or 12)
            ),
            "classify_failure": max(
                1, int(cfg.get("max_mutations_documentable_failure", 3) or 3)
            ),
            "verify_stage": 0,
        }
        self._pending_actions: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._last_observation: dict[tuple[int, str], dict[str, Any]] = {}
        self._signature_streak: dict[tuple[int, str], int] = defaultdict(int)
        self._channel_no_progress_streak: dict[tuple[int, str], int] = defaultdict(int)
        self._checker_state_key: dict[int, str] = {}
        self._checker_state_repeats: dict[int, int] = defaultdict(int)
        self._regression_count: dict[int, int] = defaultdict(int)
        self._latest_decision: dict[int, dict[str, Any]] = {}
        self._mutation_budget: dict[int, dict[str, Any]] = {}
        self._budget_epoch: dict[int, int] = defaultdict(int)

    def observe(self, event: Mapping[str, Any]) -> dict[str, Any] | None:
        if not self.enabled or not isinstance(event, Mapping):
            return None
        event_type = str(event.get("event_type") or "")
        stage_index = event.get("stage_index")
        if not isinstance(stage_index, int):
            return None
        if event_type == "file_mutation":
            self._observe_action(stage_index, event)
            return None
        if event_type == "stage_transition":
            if not event.get("advanced"):
                return None
            self._reset_stage(stage_index)
            self._set_mutation_budget(stage_index, "exploit")
            decision = {
                "schema_version": 1,
                "policy_version": "verifier_progress_controller_v2_stage_contract",
                "credit_role": "progress",
                "stage_advanced": True,
                "policy": "exploit",
                "mutation_budget": self.mutation_budget_status(stage_index),
                "control": {
                    "state": "exploit",
                    "credit_role": "progress",
                    "stage_advanced": True,
                    "max_next_file_mutations": self.max_actions["exploit"],
                    "hard_budget_enforced": self.hard_enforce_mutation_budget,
                    "required_validation": (
                        "Read CurrentTips and validate the new stage contract before editing."
                    ),
                    "constraints": [
                        "The previous stage advanced; do not carry its failure hypothesis into the new stage."
                    ],
                },
            }
            self._latest_decision[stage_index] = decision
            return decision
        if event_type not in {"test_run", "check_result"}:
            return None
        decision = self._observe_outcome(stage_index, event_type, event)
        self._latest_decision[stage_index] = decision
        return decision

    def latest(self, stage_index: int) -> dict[str, Any] | None:
        value = self._latest_decision.get(stage_index)
        return dict(value) if isinstance(value, dict) else None

    def mutation_budget_status(self, stage_index: int) -> dict[str, Any]:
        value = self._mutation_budget.get(stage_index)
        if isinstance(value, dict):
            return dict(value)
        return {
            "enforced": self.hard_enforce_mutation_budget,
            "active": False,
            "limit": None,
            "remaining": None,
            "consumed": 0,
            "policy": None,
            "epoch": int(self._budget_epoch.get(stage_index, 0)),
        }

    def authorize_mutation(
        self,
        stage_index: int,
        *,
        tool: str = "",
        path: str = "",
    ) -> tuple[bool, dict[str, Any]]:
        """Consume one mutation slot or reject the tool call before execution."""
        status = self.mutation_budget_status(stage_index)
        detail = {**status, "tool": str(tool), "path": str(path)[:500]}
        if not self.enabled or not self.hard_enforce_mutation_budget or not status["active"]:
            detail["reason"] = "budget_not_active"
            return True, detail
        remaining = int(status.get("remaining", 0) or 0)
        if remaining <= 0:
            detail["reason"] = "mutation_budget_exhausted"
            return False, detail
        budget = self._mutation_budget[stage_index]
        budget["remaining"] = remaining - 1
        budget["consumed"] = int(budget.get("consumed", 0) or 0) + 1
        detail.update(budget)
        detail["reason"] = "mutation_budget_consumed"
        return True, detail

    def refund_mutation(self, stage_index: int) -> dict[str, Any]:
        """Refund a pre-authorized slot when the mutation tool itself fails."""
        status = self._mutation_budget.get(stage_index)
        if not isinstance(status, dict) or not status.get("active"):
            return self.mutation_budget_status(stage_index)
        consumed = int(status.get("consumed", 0) or 0)
        if consumed <= 0:
            return self.mutation_budget_status(stage_index)
        limit = int(status.get("limit", 0) or 0)
        remaining = int(status.get("remaining", 0) or 0)
        status["consumed"] = consumed - 1
        status["remaining"] = min(limit, remaining + 1)
        return self.mutation_budget_status(stage_index)

    def _mutation_limit(self, policy: str, stage_name: str = "") -> int:
        limit = int(self.max_actions[policy])
        if (
            stage_allows_expected_failures(stage_name)
            and policy in {"restore_template_contract", "diagnose", "pivot"}
        ):
            # Template batches legitimately touch several test skeletons. Keep
            # the controller's pivot/diagnostic policy, but do not collapse a
            # multi-file batch to the generic one-mutation pivot budget.
            limit = max(limit, int(self.max_actions["restore_template_contract"]))
        return limit

    def _set_mutation_budget(
        self,
        stage_index: int,
        policy: str,
        stage_name: str = "",
    ) -> None:
        self._budget_epoch[stage_index] += 1
        limit = self._mutation_limit(policy, stage_name)
        self._mutation_budget[stage_index] = {
            "enforced": self.hard_enforce_mutation_budget,
            "active": True,
            "limit": limit,
            "remaining": limit,
            "consumed": 0,
            "policy": policy,
            "epoch": self._budget_epoch[stage_index],
        }

    def _observe_action(self, stage_index: int, event: Mapping[str, Any]) -> None:
        self._pending_actions[stage_index].append({
            "tool": str(event.get("tool") or ""),
            "operation": str(event.get("operation") or ""),
            "path": str(event.get("path") or "")[:500],
            "path_category": str(event.get("path_category") or "unknown"),
            "success": bool(event.get("success")),
        })
        self._pending_actions[stage_index] = self._pending_actions[stage_index][-20:]

    def _observe_outcome(
        self,
        stage_index: int,
        event_type: str,
        event: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = event.get("result") if isinstance(event.get("result"), Mapping) else {}
        stage_name = str(event.get("stage_name") or "")
        channel = "test" if event_type == "test_run" else "checker"
        observation_scope = channel
        if channel == "test":
            target = " ".join(str(event.get("target") or "").split()) or "<all>"
            observation_scope = f"test:{_short_hash(target)}"
        signature = failure_signature_from_summary(event_type, stage_name, dict(result))
        outcome = str(result.get("test_outcome") or "")
        outcome_reason = str(result.get("test_outcome_reason") or "")
        success = (
            outcome == TEST_OUTCOME_PASS
            if event_type == "test_run"
            else result.get("check_pass") is True
        )
        template_contract_violation = (
            event_type == "test_run"
            and stage_allows_expected_failures(stage_name)
            and outcome == TEST_OUTCOME_PASS
            and outcome_reason != "expected_template_xfail"
        )
        if template_contract_violation:
            success = False
        invalid = event_type == "test_run" and outcome in INVALID_TEST_OUTCOMES
        current_failures = _failure_set(result, signature, use_fallback=not success)
        current_metrics = _progress_metrics(result)
        if template_contract_violation:
            current_failures.add("contract:template_tests_must_expected_fail")
        previous = self._last_observation.get((stage_index, observation_scope))
        previous_failures = set(previous.get("failure_set", [])) if previous else set()
        previous_metrics = dict(previous.get("progress_metrics", {})) if previous else {}
        removed = sorted(previous_failures - current_failures)
        added = sorted(current_failures - previous_failures)
        actions = list(self._pending_actions.pop(stage_index, []))
        comparable_metrics = current_metrics.keys() & previous_metrics.keys()
        metric_improved = any(
            current_metrics[name] < previous_metrics[name]
            for name in comparable_metrics
        )
        metric_regressed = any(
            current_metrics[name] > previous_metrics[name]
            for name in comparable_metrics
        )

        if success:
            credit_role = "progress"
        elif invalid:
            credit_role = "invalid"
        elif previous is None:
            credit_role = "diagnostic"
        elif metric_improved and not metric_regressed:
            credit_role = "progress"
        elif metric_regressed and not metric_improved:
            credit_role = "regression"
        elif len(removed) > len(added):
            credit_role = "progress"
        elif len(added) > len(removed):
            credit_role = "regression"
        elif current_failures == previous_failures:
            credit_role = "no_progress"
        else:
            credit_role = "diagnostic"

        channel_key = (stage_index, observation_scope)
        if credit_role == "no_progress":
            self._channel_no_progress_streak[channel_key] += 1
        else:
            self._channel_no_progress_streak[channel_key] = 0
        if credit_role == "regression":
            self._regression_count[stage_index] += 1

        if channel == "checker":
            if success:
                self._checker_state_key.pop(stage_index, None)
                self._checker_state_repeats[stage_index] = 0
            else:
                checker_key = _semantic_checker_state_key(current_failures, stage_name)
                if checker_key and checker_key == self._checker_state_key.get(stage_index):
                    self._checker_state_repeats[stage_index] += 1
                else:
                    self._checker_state_key[stage_index] = checker_key
                    self._checker_state_repeats[stage_index] = 1
        checker_no_progress = max(0, self._checker_state_repeats[stage_index] - 1)

        streak_key = (stage_index, signature)
        self._signature_streak[streak_key] = (
            self._signature_streak[streak_key] + 1
            if previous and signature == previous.get("signature")
            else 1
        )
        union_size = len(previous_failures | current_failures)
        information_gain = (
            round(len(previous_failures ^ current_failures) / max(1, union_size), 4)
            if previous is not None
            else 1.0
        )
        explicit_evidence = bool(
            _string_items(result.get("failed_cases_top"))
            or _string_items(result.get("failed_checkpoints_top"))
            or _string_items(result.get("checker_categories"))
            or current_metrics
            or success
            or invalid
        )
        confidence = 0.95 if invalid or success else (0.85 if explicit_evidence else 0.65)
        policy = self._policy_for(
            stage_index,
            observation_scope,
            credit_role,
            event_type=event_type,
            stage_name=stage_name,
            outcome=outcome,
            result=result,
            template_contract_violation=template_contract_violation,
            success=success,
        )
        self._set_mutation_budget(stage_index, policy, stage_name)
        action_categories = sorted({item["path_category"] for item in actions})
        control = self._render_control(
            credit_role=credit_role,
            policy=policy,
            no_progress_streak=self._channel_no_progress_streak[channel_key],
            checker_no_progress_streak=checker_no_progress,
            signature=signature,
            action_categories=action_categories,
            removed_count=len(removed),
            added_count=len(added),
            outcome=outcome,
            outcome_reason=outcome_reason,
            event_type=event_type,
            stage_name=stage_name,
            result=result,
        )
        decision = {
            "schema_version": 1,
            "policy_version": "verifier_progress_controller_v2_stage_contract",
            "credit_role": credit_role,
            "failure_before": {
                "signature": previous.get("signature") if previous else None,
                "failure_set": sorted(previous_failures),
            },
            "action_sequence": actions,
            "action_fingerprint": _short_hash(actions) if actions else "",
            "observation_after": {
                "event_type": event_type,
                "tool": event.get("tool"),
                "test_outcome": outcome or None,
                "signature": signature,
                "failure_set": sorted(current_failures),
                "progress_metrics": current_metrics,
            },
            "stage_advanced": False,
            "failure_set_delta": {
                "removed": removed,
                "added": added,
                "removed_count": len(removed),
                "added_count": len(added),
            },
            "regression_count": self._regression_count[stage_index],
            "information_gain": information_gain,
            "action_cost": {
                "mutations": len(actions),
                "successful_mutations": sum(bool(item.get("success")) for item in actions),
            },
            "confidence": confidence,
            "observation_channel": channel,
            "observation_scope": observation_scope,
            "same_signature_streak": self._signature_streak[streak_key],
            "no_progress_streak": self._channel_no_progress_streak[channel_key],
            "checker_no_progress_streak": checker_no_progress,
            "policy": policy,
            "control": control,
            "mutation_budget": self.mutation_budget_status(stage_index),
        }
        self._last_observation[(stage_index, observation_scope)] = {
            "signature": signature,
            "failure_set": sorted(current_failures),
            "progress_metrics": current_metrics,
            "credit_role": credit_role,
        }
        return decision

    def _policy_for(
        self,
        stage_index: int,
        observation_scope: str,
        credit_role: str,
        *,
        event_type: str,
        stage_name: str,
        outcome: str,
        result: Mapping[str, Any],
        template_contract_violation: bool,
        success: bool,
    ) -> str:
        if credit_role == "invalid":
            return "repair_execution"
        checker_no_progress = max(0, self._checker_state_repeats[stage_index] - 1)
        channel_no_progress = self._channel_no_progress_streak[(stage_index, observation_scope)]
        checker_categories = set(_string_items(result.get("checker_categories")))
        if credit_role != "progress" and checker_no_progress < self.no_progress_patience:
            if "template_source_contract_incomplete" in checker_categories:
                return "restore_template_contract"
            if "test_marking_contract_incomplete" in checker_categories:
                return "repair_marking_contract"
            if "bug_doc_checkpoint_not_marked" in checker_categories:
                return "reconcile_checkpoint_evidence"
        if checker_no_progress >= self.no_progress_patience:
            return "pivot"
        if credit_role == "no_progress" and channel_no_progress >= self.no_progress_patience:
            return "pivot"
        if event_type == "test_run" and stage_allows_expected_failures(stage_name):
            if template_contract_violation or outcome != TEST_OUTCOME_PASS:
                return "restore_template_contract"
        if credit_role == "regression":
            return "contain"
        if (
            stage_allows_documentable_failures(stage_name)
            and (
                (event_type == "test_run" and outcome and outcome != TEST_OUTCOME_PASS)
                or (
                    event_type == "check_result"
                    and not success
                    and any(
                        category in {
                            "failed_cases_need_classification",
                            "unsupported_test_status",
                            "test_execution_contract",
                        }
                        for category in _string_items(result.get("checker_categories"))
                    )
                )
            )
        ):
            return "classify_failure"
        if success:
            return "verify_stage"
        if credit_role in {"no_progress", "diagnostic"}:
            return "diagnose"
        return "exploit"

    def _render_control(
        self,
        *,
        credit_role: str,
        policy: str,
        no_progress_streak: int,
        checker_no_progress_streak: int,
        signature: str,
        action_categories: list[str],
        removed_count: int,
        added_count: int,
        outcome: str,
        outcome_reason: str,
        event_type: str,
        stage_name: str,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        checker_only = stage_uses_checker_only_validation(stage_name)
        required_validation = (
            "Repair the smallest document/schema issue, then call Check/Complete; do not run pytest in this stage."
            if checker_only
            else "Run the smallest test/check target that can validate the current hypothesis."
        )
        if policy == "repair_execution":
            constraints = [
                "The latest test did not execute valid tests; repair invocation, collection, timeout, or infrastructure first.",
                "Do not interpret this observation as DUT behavior and do not add it to the bug document.",
            ]
            required_validation = "Obtain a non-empty, valid test execution before semantic repair."
        elif policy == "restore_template_contract":
            constraints = [
                "This stage creates test templates, not executable DUT tests.",
                "Each template must keep mark_function, a concrete TODO, and assert False, 'Not implemented'; do not add functional stimulus or make the suite pass.",
                "Only zero collection, import/setup errors, timeout, or crash are infrastructure failures here.",
            ]
            required_validation = (
                "Run the template Check/Complete contract; a non-empty expected-placeholder failure is the valid result."
            )
        elif policy == "repair_marking_contract":
            constraints = [
                "Treat the complete Checker list as one repair batch; do not fix one mark per round.",
                "Every mark_function second argument must be the current callable test function: module functions use test_case and class methods use self.test_case, never a string or .__name__.",
                "Preserve all pytest node ids and already passing tests; do not create renamed test copies.",
            ]
            required_validation = (
                "Repair all listed mark_function sites, then call Check/Complete once and require the unmarked-function count to decrease."
            )
        elif policy == "reconcile_checkpoint_evidence":
            constraints = [
                "Use the Checker's exact TC-to-FG/FC/CK mismatch list as the complete repair batch.",
                "If a failing TC genuinely proves the checkpoint, add that exact mark_function mapping; otherwise remove only the incorrect TC reference from that bug-document branch.",
                "Preserve ordinary FAILED evidence and pytest node ids; do not rewrite the test collection or duplicate BG trees.",
            ]
            required_validation = (
                "After reconciling the complete mismatch list, call Check/Complete and require the checkpoint-not-marked count to decrease."
            )
        elif policy == "classify_failure":
            statuses = {
                str(key).upper(): int(value or 0)
                for key, value in (result.get("test_status_counts") or {}).items()
            } if isinstance(result.get("test_status_counts"), Mapping) else {}
            constraints = [
                "Classify each non-passing test as test/API error, invalid execution, or reproducible DUT defect before editing.",
                "A confirmed DUT defect must remain an ordinary FAILED assertion and must be linked to source evidence and a complete FG/FC/CK/BG/TC bug-document path.",
                "Do not convert a confirmed defect to PASS, XFAIL, or SKIP; XFAIL/SKIP are rejected by the test-execution contract.",
                "Preserve already passing tests and avoid whole-file or whole-suite rewrites.",
            ]
            if statuses.get("XFAIL", 0) or statuses.get("XFAILED", 0):
                constraints.insert(
                    0,
                    "The report contains XFAIL/XFAILED: remove the expected-failure marker, then keep a valid defect reproduction as ordinary FAILED or repair the test if it is not a DUT bug.",
                )
            required_validation = (
                "Run the smallest reproduction, then call Check/Complete so the Checker can validate the evidence chain."
            )
        elif policy == "verify_stage":
            constraints = [
                "The latest verifier result is locally successful, but the stage has not advanced.",
                "Do not make another file mutation before calling Check/Complete for stage-level evidence.",
            ]
            required_validation = "Call Check or Complete now; only a stage transition clears this gate."
        elif policy == "contain":
            constraints = [
                "The observable failure set regressed; inspect the latest diff and isolate newly introduced failures.",
                "Do not expand the edit scope until the regression is removed or explained.",
            ]
        elif policy == "pivot":
            constraints = [
                "The authoritative Checker state has repeated; do not repeat the previous edit strategy.",
                "Gather one new diagnostic fact, state one falsifiable root-cause hypothesis, then make at most one focused mutation.",
                "A local test pass does not clear this state; rerun Check/Complete after targeted validation.",
            ]
            if stage_allows_expected_failures(stage_name):
                constraints[1] = (
                    "Treat the current template batch as one focused repair unit; edit only the skeletons named by CurrentTips/Checker and stay within the reported mutation budget."
                )
        elif policy == "diagnose":
            constraints = [
                "Use the failure-set delta to narrow the root cause before broad edits.",
                "Prefer one targeted diagnostic or minimal reproduction over rewriting the test collection.",
            ]
        else:
            constraints = [
                "The latest action made measurable local progress; continue only along the supported root cause.",
                "Validate the next focused change and then confirm it with Check/Complete.",
            ]
        if stage_allows_expected_failures(stage_name) and policy != "restore_template_contract":
            constraints.append(
                "Template-stage contract still applies: keep TODO plus assert False, 'Not implemented'; do not implement DUT behavior to escape this state."
            )
        if stage_allows_documentable_failures(stage_name) and policy != "classify_failure":
            constraints.append(
                "This stage permits documented DUT assertion failures: preserve confirmed defects as ordinary FAILED, never PASS/XFAIL/SKIP, while changing strategy for the repeated Checker blocker."
            )
        if checker_only:
            constraints = [
                "This is a document/schema stage: do not run pytest or create ad-hoc tests as validation.",
                "Preserve the required output file and repair it incrementally; do not delete or recreate the whole document.",
                *[
                    item for item in constraints
                    if "local test" not in item.lower() and "targeted diagnostic" not in item.lower()
                ],
            ]
        return {
            "state": policy,
            "credit_role": credit_role,
            "failure_signature": signature,
            "failure_set_delta": {"removed": removed_count, "added": added_count},
            "same_failure_no_progress": no_progress_streak,
            "checker_no_progress": checker_no_progress_streak,
            "previous_action_categories": action_categories,
            "max_next_file_mutations": self._mutation_limit(policy, stage_name),
            "hard_budget_enforced": self.hard_enforce_mutation_budget,
            "required_validation": required_validation,
            "constraints": constraints,
            "test_outcome": outcome or None,
            "test_outcome_reason": outcome_reason or None,
            "stage_contract": (
                "template_expected_failure"
                if stage_allows_expected_failures(stage_name)
                else "documentable_dut_failure"
                if stage_allows_documentable_failures(stage_name)
                else "checker_only_document"
                if checker_only
                else "tests_must_pass"
            ),
            "observation_channel": "test" if event_type == "test_run" else "checker",
        }

    def _reset_stage(self, stage_index: int) -> None:
        self._pending_actions.pop(stage_index, None)
        self._checker_state_key.pop(stage_index, None)
        self._checker_state_repeats.pop(stage_index, None)
        self._mutation_budget.pop(stage_index, None)
        for key in list(self._last_observation):
            if key[0] == stage_index:
                self._last_observation.pop(key, None)
        for key in list(self._signature_streak):
            if key[0] == stage_index:
                self._signature_streak.pop(key, None)
        for key in list(self._channel_no_progress_streak):
            if key[0] == stage_index:
                self._channel_no_progress_streak.pop(key, None)
