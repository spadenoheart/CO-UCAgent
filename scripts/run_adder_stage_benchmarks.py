#!/usr/bin/env python3
"""Prepare and run resumable single-stage UCAgent microbenchmarks.

Checkpoints are reconstructed from a completed run's UCAgent history commits,
saved stage state, and time-bounded long-term memory. Each benchmark copies a
read-only checkpoint, runs exactly one target stage, and records wall time and
LLM usage without continuing into the next stage.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_adder_experiments as full_runner


REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_REFERENCE_RUN = (
    REPO_ROOT
    / "benchmark/ucagent_experiments/adder_co_contract_fix_20260812_runs"
    / "CO_SOTA_V22_SCHEMA__seed_20260721"
)
DEFAULT_MATRIX = REPO_ROOT / "benchmark/ucagent_experiments/adder_co_sota_20260724_matrix.json"
DEFAULT_CHECKPOINT_ROOT = (
    REPO_ROOT / "benchmark/ucagent_experiments/adder_stage_checkpoints_s20260721"
)
DEFAULT_RUN_ROOT = REPO_ROOT / "benchmark/ucagent_experiments/adder_stage_benchmark_runs"
DEFAULT_STAGES = (21, 23, 28)
COMMON_STATIC_WORKSPACE_ITEMS = ("Guide_Doc", "AGENTS.md")
TELEMETRY_PREFIXES = (
    "unity_test/llm_input_dumps/",
    "unity_test/.pytest_cache/",
    "unity_test/__pycache__/",
)
TELEMETRY_FILES = {"unity_test/structured_events.jsonl"}
STAGE_DATA_AVAILABLE_AFTER = {
    "COVER_GROUP_DOC_CK_LIST": 5,
    "_CK_REFINE_RESULT": 6,
    "_BASIC_API_PASSED_TEST_NODEIDS": 21,
    "TEST_TEMPLATE_IMP_REPORT": 22,
    "_TC_REFINE_RESULT": 25,
    "_RANDOM_TEST_CASES_RESULT": 28,
    "RANDOM_TEST_CASES_REPORT": 28,
}
RESULT_FIELDS = (
    "run_id",
    "label",
    "stage_index",
    "stage_title",
    "seed",
    "status",
    "duration_seconds",
    "duration",
    "summary_soft_limit",
    "hard_input_limit",
    "prompt_tokens",
    "completion_tokens",
    "llm_requests",
    "failed_checks",
    "passed_checks",
    "test_runs",
    "test_outcomes",
    "context_injections",
    "contract_candidates",
    "contract_applicable",
    "contract_inapplicable",
    "contract_enforced_rejected",
    "progress_decisions",
    "progress_roles",
    "progress_policies",
    "pivot_decisions",
    "verify_stage_decisions",
    "mutation_budget_blocks",
    "stage_state_updates",
    "observation_masking_events",
    "max_masked_observations",
    "max_masked_chars_saved",
    "file_mutations",
    "successful_mutations",
    "attempts",
    "checkpoint_sha256",
    "source_sha256",
    "finished_at",
)
PAIR_RESULT_FIELDS = (
    "pair_id",
    "pair_label",
    "stage_index",
    "seed",
    "control_arm",
    "treatment_arm",
    "control_status",
    "treatment_status",
    "comparison_valid",
    "comparison_invalid_reason",
    "quality_noninferior",
    "control_duration_seconds",
    "treatment_duration_seconds",
    "duration_delta_seconds",
    "time_reduction_ratio",
    "control_tokens",
    "treatment_tokens",
    "token_delta",
    "token_reduction_ratio",
    "control_regressions",
    "treatment_regressions",
    "regression_delta",
    "control_invalid_tests",
    "treatment_invalid_tests",
    "invalid_test_delta",
    "utility_score",
    "winner",
    "checkpoint_sha256",
    "control_run_id",
    "treatment_run_id",
    "finished_at",
)
LOG_TIME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}")
FIELD_PATTERNS = {
    "stage": re.compile(r"\bstage=(\d+)"),
    "prompt_tokens": re.compile(r"\bprompt_tokens=(\d+)"),
    "completion_tokens": re.compile(r"\bcompletion_tokens=(\d+)"),
}


def load_json(path: Path) -> dict[str, Any]:
    return full_runner.load_json(path)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    full_runner.atomic_write_json(path, value)


def git_output(history: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(history), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )
    if result.returncode != 0:
        stderr = result.stderr if isinstance(result.stderr, str) else result.stderr.decode(errors="replace")
        raise RuntimeError(f"git {' '.join(args)} failed in {history}: {stderr.strip()}")
    return result.stdout


def load_stage_transitions(events_path: Path) -> dict[int, dict[str, Any]]:
    transitions: dict[int, dict[str, Any]] = {}
    with events_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                event.get("event_type") != "stage_transition"
                or not event.get("success")
                or not event.get("advanced")
            ):
                continue
            to_stage = event.get("to_stage") or {}
            target = to_stage.get("stage_index")
            if isinstance(target, int) and target not in transitions:
                transitions[target] = event
    return transitions


def load_history_commits(history: Path) -> list[dict[str, Any]]:
    output = str(
        git_output(history, "log", "--all", "--reverse", "--format=%H%x09%ct%x09%s")
    )
    commits = []
    for line in output.splitlines():
        commit, timestamp, subject = line.split("\t", 2)
        commits.append({"commit": commit, "time_unix": float(timestamp), "subject": subject})
    if not commits:
        raise RuntimeError(f"no commits found in UCAgent history: {history}")
    return commits


def select_boundary(
    stage_index: int,
    transitions: dict[int, dict[str, Any]],
    commits: list[dict[str, Any]],
) -> dict[str, Any]:
    if stage_index == 0:
        raise ValueError(
            "stage 0 has no pre-stage history commit in the reference run; "
            "using its first commit would leak stage-0 outputs"
        )
    event = transitions.get(stage_index)
    if event is None:
        raise ValueError(
            f"stage {stage_index} has no successful entry transition in the reference run; "
            "it may have been skipped"
        )
    event_time = float(event.get("time_unix", 0.0))
    commit = min(commits, key=lambda item: abs(float(item["time_unix"]) - event_time))
    delta = abs(float(commit["time_unix"]) - event_time)
    if delta > 5.0:
        raise RuntimeError(
            f"cannot align stage {stage_index} transition with history commit (nearest delta={delta:.1f}s)"
        )
    to_stage = event.get("to_stage") or {}
    return {
        "commit": commit["commit"],
        "time_unix": event_time,
        "timestamp": event.get("timestamp"),
        "stage_title": str(to_stage.get("stage_title", "")),
        "stage_name": str(to_stage.get("stage_name", "")),
        "from_stage": event.get("from_stage"),
        "history_subject": commit["subject"],
        "alignment_delta_seconds": delta,
    }


def extract_workspace_snapshot(history: Path, commit: str, destination: Path) -> list[str]:
    names = str(git_output(history, "ls-tree", "-r", "--name-only", commit)).splitlines()
    extracted = []
    for name in names:
        if not name.startswith("unity_test/"):
            continue
        if name in TELEMETRY_FILES or any(name.startswith(prefix) for prefix in TELEMETRY_PREFIXES):
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes(git_output(history, "show", f"{commit}:{name}", binary=True)))
        extracted.append(name)
    if not extracted:
        raise RuntimeError(f"history commit {commit} contains no usable unity_test files")
    return extracted


def sanitize_stage_state(
    final_state: dict[str, Any],
    stage_index: int,
    snapshot_workspace: Path,
) -> dict[str, Any]:
    state = json.loads(json.dumps(final_state))
    state["stage_index"] = stage_index
    state["all_completed"] = False
    state["time_begin"] = 0.0
    state["time_end"] = None
    state["is_agent_exit"] = False
    state["is_wait_human_check"] = False
    stages_info = state.get("stages_info", {})
    if isinstance(stages_info, dict):
        state["stages_info"] = {
            str(index): value
            for index, value in stages_info.items()
            if int(index) < stage_index
        }

    stage_data = state.get("stage_data", {})
    if not isinstance(stage_data, dict):
        stage_data = {}
    stage_data = {
        key: value
        for key, value in stage_data.items()
        if STAGE_DATA_AVAILABLE_AFTER.get(key, stage_index - 1) < stage_index
    }
    report_path = snapshot_workspace / "unity_test" / ".TEST_TEMPLATE_IMP_REPORT.json"
    if "TEST_TEMPLATE_IMP_REPORT" in stage_data and report_path.is_file():
        try:
            stage_data["TEST_TEMPLATE_IMP_REPORT"] = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid checkpoint report: {report_path}") from exc
    state["stage_data"] = stage_data
    return state


def _filter_timed_jsonl(source: Path, destination: Path, cutoff: float) -> set[str]:
    hashes: set[str] = set()
    if not source.is_file():
        return hashes
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8", errors="replace") as src, destination.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            timestamp = (row.get("meta") or {}).get("timestamp", row.get("ts"))
            if not isinstance(timestamp, (int, float)) or float(timestamp) > cutoff:
                continue
            dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            if row.get("hash"):
                hashes.add(str(row["hash"]))
    return hashes


def copy_memory_at_boundary(
    reference_workspace: Path,
    destination: Path,
    cutoff: float,
    dut_name: str = "Adder",
) -> int:
    source_dir = reference_workspace / ".ucagent_memory" / dut_name
    target_dir = destination / ".ucagent_memory" / dut_name
    kept_hashes = set()
    for name in ("memory.candidate.jsonl", "memory.jsonl"):
        kept_hashes.update(_filter_timed_jsonl(source_dir / name, target_dir / name, cutoff))
    _filter_timed_jsonl(source_dir / "metrics.jsonl", target_dir / "metrics.jsonl", cutoff)
    embedding_source = source_dir / "memory.emb.jsonl"
    if embedding_source.is_file():
        target_dir.mkdir(parents=True, exist_ok=True)
        with embedding_source.open("r", encoding="utf-8", errors="replace") as src, (
            target_dir / "memory.emb.jsonl"
        ).open("w", encoding="utf-8") as dst:
            for line in src:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(row.get("hash", "")) in kept_hashes:
                    dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(kept_hashes)


def prepare_checkpoint(reference_run: Path, checkpoint_root: Path, stage_index: int, force: bool) -> Path:
    reference_workspace = reference_run / "workspace"
    history = reference_workspace / ".ucagent" / "history"
    final_state_path = reference_workspace / ".ucagent" / "ucagent_info.json"
    events_path = reference_workspace / "unity_test" / "structured_events.jsonl"
    for required in (history / ".git", final_state_path, events_path):
        if not required.exists():
            raise FileNotFoundError(f"reference run lacks checkpoint input: {required}")

    transitions = load_stage_transitions(events_path)
    commits = load_history_commits(history)
    boundary = select_boundary(stage_index, transitions, commits)
    final_state = load_json(final_state_path)
    dut_name = str(final_state.get("dut_name") or final_state.get("DUT") or "").strip()
    if not dut_name:
        raise RuntimeError(f"cannot determine DUT name from {final_state_path}")
    static_workspace_items = (*COMMON_STATIC_WORKSPACE_ITEMS, dut_name, f"{dut_name}_RTL")
    checkpoint_dir = checkpoint_root / f"stage_{stage_index:02d}"
    if checkpoint_dir.exists():
        if not force:
            raise FileExistsError(f"checkpoint already exists: {checkpoint_dir}; use --force")
        shutil.rmtree(checkpoint_dir)
    workspace = checkpoint_dir / "workspace"
    workspace.mkdir(parents=True)

    for name in static_workspace_items:
        source = reference_workspace / name
        target = workspace / name
        if source.is_dir():
            shutil.copytree(source, target, symlinks=True)
        elif source.is_file():
            shutil.copy2(source, target)
    extracted = extract_workspace_snapshot(history, str(boundary["commit"]), workspace)
    state = sanitize_stage_state(final_state, stage_index, workspace)
    state_path = workspace / ".ucagent" / "ucagent_info.json"
    atomic_write_json(state_path, state)
    memory_entries = copy_memory_at_boundary(
        reference_workspace,
        workspace,
        float(boundary["time_unix"]),
        dut_name,
    )

    workspace_hash = full_runner.sha256_tree(
        workspace,
        [name for name in static_workspace_items if (workspace / name).exists()]
        + ["unity_test", ".ucagent/ucagent_info.json"]
        + ([".ucagent_memory"] if (workspace / ".ucagent_memory").exists() else []),
    )
    manifest = {
        "schema_version": 1,
        "created_at": full_runner.utc_now(),
        "reference_run": str(reference_run),
        "reference_workspace": str(reference_workspace),
        "reference_seed": final_state.get("seed"),
        "dut_name": dut_name,
        "stage_index": stage_index,
        "stage_name": boundary.get("stage_name", ""),
        "stage_title": boundary.get("stage_title", ""),
        "boundary": boundary,
        "history_commit": boundary["commit"],
        "workspace_sha256": workspace_hash,
        "extracted_file_count": len(extracted),
        "memory_entry_count": memory_entries,
        "stage_data_keys": sorted(state.get("stage_data", {})),
        "total_stages": len(final_state.get("stages_info", {})),
        "fidelity": {
            "workspace_files": "exact history commit at stage entry, excluding telemetry",
            "stage_state": "completed prior stages retained; target and later stages removed",
            "long_term_memory": "entries truncated at stage-entry timestamp",
            "short_term_messages": "not persisted by UCAgent; benchmark starts a fresh conversation",
        },
    }
    atomic_write_json(checkpoint_dir / "manifest.json", manifest)
    return checkpoint_dir


def reset_run_state(workspace: Path) -> None:
    state_path = workspace / ".ucagent" / "ucagent_info.json"
    state = load_json(state_path)
    state["time_begin"] = time.time()
    state["time_end"] = None
    state["is_agent_exit"] = False
    state["all_completed"] = False
    atomic_write_json(state_path, state)


def target_stage_completed(workspace: Path, stage_index: int) -> bool:
    state_path = workspace / ".ucagent" / "ucagent_info.json"
    if not state_path.is_file():
        return False
    try:
        state = load_json(state_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    stage_info = (state.get("stages_info") or {}).get(str(stage_index), {})
    return bool(stage_info.get("is_completed")) and int(state.get("stage_index", 0)) > stage_index


def parse_stage_metrics(run_dir: Path, stage_index: int) -> dict[str, Any]:
    prompt_tokens = completion_tokens = llm_requests = 0
    log_path = run_dir / "ucagent-log.log"
    if log_path.is_file():
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "[data_collection][llm_request]" not in line:
                    continue
                stage_match = FIELD_PATTERNS["stage"].search(line)
                if not stage_match or int(stage_match.group(1)) != stage_index:
                    continue
                llm_requests += 1
                prompt_match = FIELD_PATTERNS["prompt_tokens"].search(line)
                completion_match = FIELD_PATTERNS["completion_tokens"].search(line)
                prompt_tokens += int(prompt_match.group(1)) if prompt_match else 0
                completion_tokens += int(completion_match.group(1)) if completion_match else 0

    failed_checks = passed_checks = test_runs = context_injections = 0
    contract_candidates = contract_applicable = contract_inapplicable = 0
    contract_enforced_rejected = 0
    file_mutations = successful_mutations = 0
    progress_decisions = pivot_decisions = verify_stage_decisions = mutation_budget_blocks = 0
    stage_state_updates = observation_masking_events = 0
    max_masked_observations = max_masked_chars_saved = 0
    progress_roles: Counter[str] = Counter()
    progress_policies: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    events_path = run_dir / "workspace" / "unity_test" / "structured_events.jsonl"
    if events_path.is_file():
        with events_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("stage_index") != stage_index:
                    continue
                event_type = event.get("event_type")
                if event_type == "check_result":
                    if event.get("success"):
                        passed_checks += 1
                    else:
                        failed_checks += 1
                elif event_type == "test_run":
                    test_runs += 1
                    outcomes[str((event.get("result") or {}).get("test_outcome", "unknown"))] += 1
                elif event_type == "context_reuse_decision":
                    if event.get("decision") == "inject":
                        context_injections += 1
                    summary = event.get("search_summary")
                    summary = summary if isinstance(summary, dict) else {}
                    contract_candidates += int(summary.get("contract_candidates", 0) or 0)
                    contract_applicable += int(summary.get("contract_applicable", 0) or 0)
                    contract_inapplicable += int(summary.get("contract_inapplicable", 0) or 0)
                    contract_enforced_rejected += int(
                        summary.get("contract_enforced_rejected", 0) or 0
                    )
                elif event_type == "file_mutation":
                    file_mutations += 1
                    if event.get("success") is True:
                        successful_mutations += 1
                elif event_type == "progress_control_decision":
                    progress_decisions += 1
                    progress_roles[str(event.get("credit_role", "unknown"))] += 1
                    policy = str(event.get("policy", "unknown"))
                    progress_policies[policy] += 1
                    if policy == "pivot":
                        pivot_decisions += 1
                    elif policy == "verify_stage":
                        verify_stage_decisions += 1
                elif event_type == "mutation_budget_blocked":
                    mutation_budget_blocks += 1
                elif event_type == "stage_state_package_update":
                    stage_state_updates += 1
                elif event_type == "observation_masking":
                    observation_masking_events += 1
                    max_masked_observations = max(
                        max_masked_observations,
                        int(event.get("masked_observations", 0) or 0),
                    )
                    max_masked_chars_saved = max(
                        max_masked_chars_saved,
                        int(event.get("saved_chars", 0) or 0),
                    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "llm_requests": llm_requests,
        "failed_checks": failed_checks,
        "passed_checks": passed_checks,
        "test_runs": test_runs,
        "test_outcomes": dict(outcomes),
        "context_injections": context_injections,
        "contract_candidates": contract_candidates,
        "contract_applicable": contract_applicable,
        "contract_inapplicable": contract_inapplicable,
        "contract_enforced_rejected": contract_enforced_rejected,
        "progress_decisions": progress_decisions,
        "progress_roles": dict(progress_roles),
        "progress_policies": dict(progress_policies),
        "pivot_decisions": pivot_decisions,
        "verify_stage_decisions": verify_stage_decisions,
        "mutation_budget_blocks": mutation_budget_blocks,
        "stage_state_updates": stage_state_updates,
        "observation_masking_events": observation_masking_events,
        "max_masked_observations": max_masked_observations,
        "max_masked_chars_saved": max_masked_chars_saved,
        "file_mutations": file_mutations,
        "successful_mutations": successful_mutations,
    }


def append_result_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    serialized = {}
    for field in RESULT_FIELDS:
        value = row.get(field, "")
        if isinstance(value, (dict, list)):
            serialized[field] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            serialized[field] = str(value)
    run_id = serialized["run_id"]
    for index, existing in enumerate(rows):
        if existing.get("run_id") == run_id:
            rows[index] = serialized
            break
    else:
        rows.append(serialized)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_pair_result_csv(path: Path, row: dict[str, Any]) -> None:
    """Upsert a paired result without changing the single-stage result schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    serialized = {}
    for field in PAIR_RESULT_FIELDS:
        value = row.get(field, "")
        serialized[field] = (
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (dict, list))
            else str(value)
        )
    pair_id = serialized["pair_id"]
    for index, existing in enumerate(rows):
        if existing.get("pair_id") == pair_id:
            rows[index] = serialized
            break
    else:
        rows.append(serialized)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _counter_value(result: dict[str, Any], field: str, key: str) -> int:
    values = result.get(field)
    if not isinstance(values, dict):
        return 0
    return int(values.get(key, 0) or 0)


def compare_stage_pair(
    control: dict[str, Any],
    treatment: dict[str, Any],
    *,
    pair_label: str,
    control_arm: str,
    treatment_arm: str,
) -> dict[str, Any]:
    """Compare matched stage runs with completion and safety as hard gates."""
    if control.get("stage_index") != treatment.get("stage_index"):
        raise ValueError("paired stage results must use the same stage_index")
    if control.get("seed") != treatment.get("seed"):
        raise ValueError("paired stage results must use the same seed")
    if control.get("checkpoint_sha256") != treatment.get("checkpoint_sha256"):
        raise ValueError("paired stage results must use the same checkpoint")

    control_seconds = float(control.get("duration_seconds", 0.0) or 0.0)
    treatment_seconds = float(treatment.get("duration_seconds", 0.0) or 0.0)
    control_tokens = int(control.get("prompt_tokens", 0) or 0) + int(
        control.get("completion_tokens", 0) or 0
    )
    treatment_tokens = int(treatment.get("prompt_tokens", 0) or 0) + int(
        treatment.get("completion_tokens", 0) or 0
    )
    control_regressions = _counter_value(control, "progress_roles", "regression")
    treatment_regressions = _counter_value(treatment, "progress_roles", "regression")
    invalid_outcomes = ("infrastructure_error", "timeout", "crash", "no_tests_collected")
    control_invalid = sum(_counter_value(control, "test_outcomes", key) for key in invalid_outcomes)
    treatment_invalid = sum(
        _counter_value(treatment, "test_outcomes", key) for key in invalid_outcomes
    )
    control_completed = control.get("status") == "completed"
    treatment_completed = treatment.get("status") == "completed"
    comparison_valid = control_completed and treatment_completed
    if comparison_valid:
        comparison_invalid_reason = ""
        quality_noninferior = (
            treatment_regressions <= control_regressions
            and treatment_invalid <= control_invalid
        )
        time_ratio = (
            (control_seconds - treatment_seconds) / control_seconds
            if control_seconds > 0
            else 0.0
        )
        token_ratio = (
            (control_tokens - treatment_tokens) / control_tokens
            if control_tokens > 0
            else 0.0
        )
        utility = (time_ratio + token_ratio) / 2 if quality_noninferior else -1.0
        if not quality_noninferior:
            winner = "control_quality"
        elif utility > 0:
            winner = "treatment"
        elif utility < 0:
            winner = "control"
        else:
            winner = "tie"
    elif treatment_completed and not control_completed:
        comparison_invalid_reason = "control_not_completed"
        quality_noninferior = True
        time_ratio = token_ratio = utility = None
        winner = "treatment_completion"
    elif control_completed and not treatment_completed:
        comparison_invalid_reason = "treatment_not_completed"
        quality_noninferior = False
        time_ratio = token_ratio = utility = None
        winner = "control_completion"
    else:
        comparison_invalid_reason = "both_not_completed"
        quality_noninferior = None
        time_ratio = token_ratio = utility = None
        winner = "inconclusive"
    stage_index = int(control["stage_index"])
    seed = int(control["seed"])
    safe_pair_label = safe_label(pair_label)
    return {
        "pair_id": f"{safe_pair_label}__stage_{stage_index:02d}__seed_{seed}",
        "pair_label": safe_pair_label,
        "stage_index": stage_index,
        "seed": seed,
        "control_arm": control_arm,
        "treatment_arm": treatment_arm,
        "control_status": control.get("status"),
        "treatment_status": treatment.get("status"),
        "comparison_valid": comparison_valid,
        "comparison_invalid_reason": comparison_invalid_reason,
        "quality_noninferior": quality_noninferior,
        "control_duration_seconds": round(control_seconds, 3),
        "treatment_duration_seconds": round(treatment_seconds, 3),
        "duration_delta_seconds": round(treatment_seconds - control_seconds, 3),
        "time_reduction_ratio": round(time_ratio, 6) if time_ratio is not None else None,
        "control_tokens": control_tokens,
        "treatment_tokens": treatment_tokens,
        "token_delta": treatment_tokens - control_tokens,
        "token_reduction_ratio": round(token_ratio, 6) if token_ratio is not None else None,
        "control_regressions": control_regressions,
        "treatment_regressions": treatment_regressions,
        "regression_delta": treatment_regressions - control_regressions,
        "control_invalid_tests": control_invalid,
        "treatment_invalid_tests": treatment_invalid,
        "invalid_test_delta": treatment_invalid - control_invalid,
        "utility_score": round(utility, 6) if utility is not None else None,
        "winner": winner,
        "checkpoint_sha256": control.get("checkpoint_sha256"),
        "control_run_id": control.get("run_id"),
        "treatment_run_id": treatment.get("run_id"),
        "finished_at": full_runner.utc_now(),
    }


def record_pair_result(run_root: Path, pair: dict[str, Any]) -> None:
    report_path = run_root / "paired_results.json"
    report = load_json(report_path) if report_path.is_file() else {
        "schema_version": 1,
        "analysis_unit": "matched_stage_checkpoint_seed",
        "causal_claim": "paired_replay_only; model sampling remains stochastic",
        "pairs": {},
    }
    report.setdefault("pairs", {})[pair["pair_id"]] = pair
    report["updated_at"] = full_runner.utc_now()
    atomic_write_json(report_path, report)
    append_pair_result_csv(run_root / "paired_results.csv", pair)


def safe_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not cleaned:
        raise ValueError("benchmark label must contain at least one safe character")
    return cleaned


def resolve_python(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "python", "") or "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Python executable does not exist: {path}")
        return str(path)
    return full_runner.conda_python(args.env)


def source_hash(source_root: Path, agent_config: Path) -> str:
    paths = ["ucagent", "ucagent.py"]
    if agent_config.is_relative_to(source_root):
        paths.append(str(agent_config.relative_to(source_root)))
    return full_runner.sha256_tree(source_root, paths)


def resolve_runtime(
    matrix_path: Path,
    arm_id: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, str, str, int]:
    matrix = load_json(matrix_path)
    full_runner.validate_matrix(matrix)
    arm = next((item for item in matrix["arms"] if item["id"] == arm_id), None)
    if arm is None:
        raise ValueError(f"arm {arm_id!r} not found in {matrix_path}")
    fixed = matrix.get("fixed_configuration", {})
    source_root = Path(args.source_root).resolve() if args.source_root else full_runner.resolve_source_root(fixed)
    agent_config = (
        Path(args.agent_config).resolve()
        if args.agent_config
        else full_runner.resolve_repo_file(fixed.get("agent_config", "ucagent/setting.yaml"), label="agent config")
    )
    api_base = str(args.api_base or os.environ.get("OPENAI_API_BASE") or fixed.get("api_base", "")).rstrip("/")
    model = str(args.model or os.environ.get("OPENAI_MODEL") or fixed.get("model", ""))
    num_ctx = int(args.num_ctx or os.environ.get("OLLAMA_NUM_CTX", 0) or fixed.get("num_ctx", 0))
    if not api_base or not model or num_ctx <= 0:
        raise RuntimeError("matrix/CLI must provide api_base, model, and positive num_ctx")
    return matrix, arm, source_root, agent_config, api_base, model, num_ctx


def run_one_stage(
    args: argparse.Namespace,
    checkpoint_dir: Path,
    matrix: dict[str, Any],
    arm: dict[str, Any],
    source_root: Path,
    agent_config: Path,
    api_base: str,
    model: str,
    num_ctx: int,
    python: str,
) -> dict[str, Any]:
    manifest = load_json(checkpoint_dir / "manifest.json")
    stage_index = int(manifest["stage_index"])
    reference_seed = int(manifest["reference_seed"])
    seed = int(args.seed if args.seed is not None else reference_seed)
    if seed != reference_seed and not args.allow_seed_override:
        raise ValueError(
            f"checkpoint stage {stage_index} was created with seed {reference_seed}; "
            "use the same seed or pass --allow-seed-override"
        )
    label = safe_label(args.label)
    run_id = f"{label}__stage_{stage_index:02d}__seed_{seed}"
    run_dir = args.run_root.resolve() / run_id
    workspace = run_dir / "workspace"
    ledger_path = args.run_root.resolve() / "ledger.json"
    ledger = load_json(ledger_path) if ledger_path.is_file() else {"schema_version": 1, "runs": {}}
    record = ledger.setdefault("runs", {}).setdefault(run_id, {})
    if record.get("status") == "completed" and not args.restart:
        print(f"SKIP completed {run_id}: {record['result']['duration']}")
        return record["result"]
    if args.restart and run_dir.exists():
        shutil.rmtree(run_dir)
        record.clear()
    attempts = int(record.get("attempts", 0))
    if attempts >= args.max_attempts:
        raise RuntimeError(f"{run_id} reached --max-attempts={args.max_attempts}; use --restart")

    current_source_hash = source_hash(source_root, agent_config)
    if workspace.exists() and record.get("source_sha256") not in (None, current_source_hash):
        raise RuntimeError(
            f"source changed during an incomplete run {run_id}; use --restart to preserve comparability"
        )
    if not workspace.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(checkpoint_dir / "workspace", workspace)
    reset_run_state(workspace)

    fixed = matrix.get("fixed_configuration", {})
    context_budget = {
        "num_ctx": num_ctx,
        "summary_soft_limit": int(fixed.get("summary_max_tokens", 0) or 0),
        "hard_input_limit": int(fixed.get("summary_hard_max_tokens", 0) or 0),
        "summary_output_limit": int(fixed.get("summary_max_output_tokens", 0) or 0),
        "max_keep_msgs": int(fixed.get("summary_max_keep_msgs", 0) or 0),
        "tail_keep_msgs": int(fixed.get("summary_tail_keep_msgs", 10) or 10),
    }
    api_key = os.environ.get("OPENAI_API_KEY", "ollama")
    env = full_runner.build_run_environment(
        python,
        fixed,
        arm,
        model,
        api_base,
        api_key,
        num_ctx,
        run_dir / "home" if fixed.get("isolate_home", False) else None,
    )
    command = full_runner.build_agent_command(
        python,
        workspace,
        run_dir,
        seed,
        arm,
        agent_config,
        source_root,
        str(fixed.get("interaction_mode", "standard")),
        bool(fixed.get("stream_output", True)),
        dut=str(manifest.get("dut_name") or "Adder"),
    )
    # The checkpoint already pins ``stage_index``. Do not pass --force-stage-index
    # or skip later parent stages: skipping a parent can recursively mark the
    # target substage as skipped (Stage 23 is a child of Stage 24). The runner's
    # success condition terminates the process immediately after target advance.

    attempt = {
        "attempt": attempts + 1,
        "started_at": full_runner.utc_now(),
        "status": "running",
    }
    record.update(
        {
            "run_id": run_id,
            "label": label,
            "stage_index": stage_index,
            "stage_title": manifest.get("stage_title", ""),
            "seed": seed,
            "status": "running",
            "attempts": attempts + 1,
            "source_root": str(source_root),
            "source_sha256": current_source_hash,
            "checkpoint_sha256": manifest["workspace_sha256"],
            "matrix": str(args.matrix.resolve()),
            "matrix_sha256": full_runner.sha256_file(args.matrix.resolve()),
            "arm": arm.get("id"),
            "context_budget": context_budget,
            "workspace": str(workspace),
            "command": command,
        }
    )
    record.setdefault("attempt_history", []).append(attempt)
    ledger["updated_at"] = full_runner.utc_now()
    atomic_write_json(ledger_path, ledger)

    started = time.monotonic()
    try:
        return_code, timed_out, reason = full_runner.run_command(
            command,
            env,
            run_dir / "console.log",
            int(args.timeout_hours * 3600),
            source_root,
            success_condition=lambda: target_stage_completed(workspace, stage_index),
        )
    except KeyboardInterrupt:
        elapsed = time.monotonic() - started
        attempt.update(
            {
                "status": "interrupted",
                "ended_at": full_runner.utc_now(),
                "duration_seconds": round(elapsed, 3),
            }
        )
        record["status"] = "interrupted"
        ledger["updated_at"] = full_runner.utc_now()
        atomic_write_json(ledger_path, ledger)
        raise

    elapsed = time.monotonic() - started
    completed = target_stage_completed(workspace, stage_index)
    status = "completed" if completed else ("timeout" if timed_out else "failed")
    attempt.update(
        {
            "status": status,
            "ended_at": full_runner.utc_now(),
            "duration_seconds": round(elapsed, 3),
            "return_code": return_code,
            "reason": reason,
        }
    )
    total_seconds = sum(float(item.get("duration_seconds", 0.0)) for item in record["attempt_history"])
    metrics = parse_stage_metrics(run_dir, stage_index)
    result = {
        "run_id": run_id,
        "label": label,
        "stage_index": stage_index,
        "stage_title": manifest.get("stage_title", ""),
        "seed": seed,
        "status": status,
        "duration_seconds": round(total_seconds, 3),
        "duration": full_runner.format_duration(total_seconds),
        "summary_soft_limit": context_budget["summary_soft_limit"],
        "hard_input_limit": context_budget["hard_input_limit"],
        **metrics,
        "attempts": attempts + 1,
        "checkpoint_sha256": manifest["workspace_sha256"],
        "source_sha256": current_source_hash,
        "finished_at": full_runner.utc_now(),
    }
    record.update({"status": status, "result": result, "ended_at": full_runner.utc_now()})
    ledger["updated_at"] = full_runner.utc_now()
    atomic_write_json(run_dir / "result.json", result)
    atomic_write_json(ledger_path, ledger)
    append_result_csv(args.run_root.resolve() / "stage_results.csv", result)
    print(
        f"FINISH {run_id}: {status}, {result['duration']}, "
        f"requests={result['llm_requests']}, tokens={result['prompt_tokens']}/{result['completion_tokens']}"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="reconstruct immutable stage-entry checkpoints")
    prepare.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE_RUN)
    prepare.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    prepare.add_argument("--stage", action="append", type=int, default=[])
    prepare.add_argument("--force", action="store_true")

    run = subparsers.add_parser("run", help="run one or more stages from prepared checkpoints")
    run.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    run.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    run.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    run.add_argument("--arm", default="CO_SOTA_V22_SCHEMA")
    run.add_argument("--label", required=True, help="configuration label, e.g. baseline or candidate_mask_v1")
    run.add_argument("--stage", action="append", type=int, default=[])
    run.add_argument("--seed", type=int)
    run.add_argument("--allow-seed-override", action="store_true")
    run.add_argument("--env", default="uc")
    run.add_argument("--python", help="Python executable; bypasses --env when provided")
    run.add_argument("--source-root", type=Path)
    run.add_argument("--agent-config", type=Path)
    run.add_argument("--api-base")
    run.add_argument("--model")
    run.add_argument("--num-ctx", type=int)
    run.add_argument("--timeout-hours", type=float, default=6.0)
    run.add_argument("--max-attempts", type=int, default=2)
    run.add_argument("--restart", action="store_true")
    run.add_argument("--skip-preflight", action="store_true")
    run.add_argument("--dry-run", action="store_true")

    pair = subparsers.add_parser(
        "run-pair",
        help="run matched control/treatment arms from identical stage checkpoints",
    )
    pair.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    pair.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    pair.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    pair.add_argument("--control-arm", required=True)
    pair.add_argument("--treatment-arm", required=True)
    pair.add_argument("--pair-label", required=True)
    pair.add_argument("--stage", action="append", type=int, default=[])
    pair.add_argument("--seed", action="append", type=int, default=[])
    pair.add_argument("--allow-seed-override", action="store_true")
    pair.add_argument("--order", choices=("alternating", "control-first", "treatment-first"), default="alternating")
    pair.add_argument("--env", default="uc")
    pair.add_argument("--python", help="Python executable; bypasses --env when provided")
    pair.add_argument("--source-root", type=Path)
    pair.add_argument("--agent-config", type=Path)
    pair.add_argument("--api-base")
    pair.add_argument("--model")
    pair.add_argument("--num-ctx", type=int)
    pair.add_argument("--timeout-hours", type=float, default=6.0)
    pair.add_argument("--max-attempts", type=int, default=2)
    pair.add_argument("--restart", action="store_true")
    pair.add_argument("--skip-preflight", action="store_true")
    pair.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    stages = args.stage or list(DEFAULT_STAGES)
    if args.command == "prepare":
        reference_run = args.reference_run.resolve()
        checkpoint_root = args.checkpoint_root.resolve()
        for stage_index in stages:
            path = prepare_checkpoint(reference_run, checkpoint_root, stage_index, args.force)
            manifest = load_json(path / "manifest.json")
            print(
                f"PREPARED stage {stage_index}: {path} "
                f"commit={manifest['history_commit'][:8]} files={manifest['extracted_file_count']} "
                f"memory={manifest['memory_entry_count']}"
            )
        return 0

    matrix_path = args.matrix.resolve()
    if args.command == "run-pair":
        control_runtime = resolve_runtime(matrix_path, args.control_arm, args)
        treatment_runtime = resolve_runtime(matrix_path, args.treatment_arm, args)
        matrix, control_arm, source_root, agent_config, api_base, model, num_ctx = control_runtime
        (
            treatment_matrix,
            treatment_arm,
            treatment_source,
            treatment_config,
            treatment_api_base,
            treatment_model,
            treatment_num_ctx,
        ) = treatment_runtime
        if (
            source_root != treatment_source
            or agent_config != treatment_config
            or api_base != treatment_api_base
            or model != treatment_model
            or num_ctx != treatment_num_ctx
        ):
            raise ValueError("paired arms must share source, config, API, model, and num_ctx")
        if matrix.get("fixed_configuration", {}) != treatment_matrix.get("fixed_configuration", {}):
            raise ValueError("paired arms must share the same fixed_configuration")

        python = resolve_python(args)
        checkpoint_dirs = [args.checkpoint_root.resolve() / f"stage_{stage:02d}" for stage in stages]
        for path in checkpoint_dirs:
            if not (path / "manifest.json").is_file():
                raise FileNotFoundError(f"checkpoint not prepared: {path}")
        seeds = list(args.seed) if args.seed else [None]
        print(
            f"Pair: {safe_label(args.pair_label)}; control={args.control_arm}; "
            f"treatment={args.treatment_arm}"
        )
        print(f"Stages: {', '.join(str(stage) for stage in stages)}; seeds={seeds}")
        print(f"Source: {source_root}")
        print(f"Model: {model}, num_ctx={num_ctx}")
        if args.dry_run:
            return 0
        if not args.skip_preflight:
            api_key = os.environ.get("OPENAI_API_KEY", "ollama")
            full_runner.check_model_available(api_base, model, api_key)
            full_runner.check_ollama_model_context(api_base, model, num_ctx)
            print("Model preflight: OK")
        args.run_root.mkdir(parents=True, exist_ok=True)
        pair_index = 0
        for checkpoint_dir in checkpoint_dirs:
            manifest = load_json(checkpoint_dir / "manifest.json")
            for requested_seed in seeds:
                seed = int(
                    requested_seed
                    if requested_seed is not None
                    else manifest["reference_seed"]
                )
                base_args = copy.copy(args)
                base_args.seed = seed
                control_args = copy.copy(base_args)
                treatment_args = copy.copy(base_args)
                control_args.label = f"{safe_label(args.pair_label)}__control"
                treatment_args.label = f"{safe_label(args.pair_label)}__treatment"
                ordered = [
                    ("control", control_args, control_arm),
                    ("treatment", treatment_args, treatment_arm),
                ]
                if args.order == "treatment-first" or (
                    args.order == "alternating" and pair_index % 2 == 1
                ):
                    ordered.reverse()
                results = {}
                for role, role_args, role_arm in ordered:
                    results[role] = run_one_stage(
                        role_args,
                        checkpoint_dir,
                        matrix,
                        role_arm,
                        source_root,
                        agent_config,
                        api_base,
                        model,
                        num_ctx,
                        python,
                    )
                comparison = compare_stage_pair(
                    results["control"],
                    results["treatment"],
                    pair_label=args.pair_label,
                    control_arm=args.control_arm,
                    treatment_arm=args.treatment_arm,
                )
                record_pair_result(args.run_root.resolve(), comparison)
                time_reduction = comparison.get("time_reduction_ratio")
                token_reduction = comparison.get("token_reduction_ratio")
                time_text = f"{time_reduction:.3f}" if isinstance(time_reduction, (int, float)) else "NA"
                token_text = f"{token_reduction:.3f}" if isinstance(token_reduction, (int, float)) else "NA"
                print(
                    f"PAIR {comparison['pair_id']}: winner={comparison['winner']} "
                    f"valid={comparison['comparison_valid']} "
                    f"quality_noninferior={comparison['quality_noninferior']} "
                    f"time_reduction={time_text} "
                    f"token_reduction={token_text}"
                )
                pair_index += 1
        return 0

    matrix, arm, source_root, agent_config, api_base, model, num_ctx = resolve_runtime(
        matrix_path, args.arm, args
    )
    python = resolve_python(args)
    checkpoint_dirs = [args.checkpoint_root.resolve() / f"stage_{stage:02d}" for stage in stages]
    for path in checkpoint_dirs:
        if not (path / "manifest.json").is_file():
            raise FileNotFoundError(f"checkpoint not prepared: {path}")
    print(f"Label: {safe_label(args.label)}")
    print(f"Stages: {', '.join(str(stage) for stage in stages)}")
    print(f"Source: {source_root}")
    print(f"Model: {model}, num_ctx={num_ctx}, seed={args.seed or 'checkpoint'}")
    fixed = matrix.get("fixed_configuration", {})
    print(
        "Context budget: "
        f"soft={int(fixed.get('summary_max_tokens', 0) or 0)}, "
        f"hard={int(fixed.get('summary_hard_max_tokens', 0) or 0)}, "
        f"summary_output={int(fixed.get('summary_max_output_tokens', 0) or 0)}, "
        f"max_messages={int(fixed.get('summary_max_keep_msgs', 0) or 0)}, "
        f"tail={int(fixed.get('summary_tail_keep_msgs', 10) or 10)}"
    )
    if args.dry_run:
        return 0
    if not args.skip_preflight:
        api_key = os.environ.get("OPENAI_API_KEY", "ollama")
        full_runner.check_model_available(api_base, model, api_key)
        full_runner.check_ollama_model_context(api_base, model, num_ctx)
        print("Model preflight: OK")
    args.run_root.mkdir(parents=True, exist_ok=True)
    for checkpoint_dir in checkpoint_dirs:
        run_one_stage(
            args,
            checkpoint_dir,
            matrix,
            arm,
            source_root,
            agent_config,
            api_base,
            model,
            num_ctx,
            python,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
