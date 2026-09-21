#!/usr/bin/env python3
"""Run a resumable, serial multi-DUT UCAgent experiment suite."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from scripts.run_adder_experiments import (
        BASELINE_TOKEN_LOG_NAME,
        StageStallWatchdog,
        atomic_write_json,
        build_run_environment,
        check_embedding_service,
        check_langchain_tool_call_path,
        check_model_available,
        check_ollama_model_context,
        check_python_package_versions,
        check_summarization_middleware_compatibility,
        conda_python,
        enable_baseline_token_meter,
        environment_for_python,
        load_json,
        parse_agent_log_summary,
        run_command,
        sha256_tree,
        utc_now,
        warmup_ollama_model,
        workspace_completed,
    )
except ModuleNotFoundError:  # Direct execution puts scripts/ rather than the repository on sys.path.
    from run_adder_experiments import (
        BASELINE_TOKEN_LOG_NAME,
        StageStallWatchdog,
        atomic_write_json,
        build_run_environment,
        check_embedding_service,
        check_langchain_tool_call_path,
        check_model_available,
        check_ollama_model_context,
        check_python_package_versions,
        check_summarization_middleware_compatibility,
        conda_python,
        enable_baseline_token_meter,
        environment_for_python,
        load_json,
        parse_agent_log_summary,
        run_command,
        sha256_tree,
        utc_now,
        warmup_ollama_model,
        workspace_completed,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = REPO_ROOT / "benchmark/ucagent_experiments/multi_dut_contract_v1_suite.json"
DEFAULT_RUN_ROOT = REPO_ROOT / "benchmark/ucagent_experiments/multi_dut_contract_v1_runs"
DEFAULT_NOTIFY_HELPER = REPO_ROOT / "scripts/codebridge_weixin_notify.ts"
DEFAULT_CODEBRIDGE_STATE = Path.home() / ".codexbridge"
DEFAULT_NODE = Path.home() / ".local/bin/node"
DEFAULT_CODEBRIDGE_ROOT = REPO_ROOT.parent / "CodexBridge"


class CodeBridgeNotifier:
    def __init__(
        self,
        *,
        enabled: bool,
        helper: Path = DEFAULT_NOTIFY_HELPER,
        state_dir: Path = DEFAULT_CODEBRIDGE_STATE,
        scope: str = "",
    ) -> None:
        self.enabled = enabled
        self.helper = helper
        self.state_dir = state_dir
        self.scope = scope

    def command(self, message: str, *, dry_run: bool = False) -> list[str]:
        command = [str(DEFAULT_NODE), "--import", "tsx", str(self.helper), "--state-dir", str(self.state_dir)]
        if self.scope:
            command.extend(["--scope", self.scope])
        if dry_run:
            command.append("--dry-run")
        else:
            command.extend(["--message", message])
        return command

    def preflight(self) -> None:
        if not self.enabled:
            return
        if not DEFAULT_NODE.is_file() or not self.helper.is_file():
            raise FileNotFoundError("CodeBridge Node/notification helper is missing")
        result = subprocess.run(
            self.command("", dry_run=True),
            cwd=DEFAULT_CODEBRIDGE_ROOT,
            text=True,
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"CodeBridge notification preflight failed: {result.stderr.strip()}")

    def send(self, message: str) -> bool:
        if not self.enabled:
            return False
        try:
            result = subprocess.run(
                self.command(message),
                cwd=DEFAULT_CODEBRIDGE_ROOT,
                text=True,
                capture_output=True,
                timeout=60,
            )
        except Exception as exc:
            print(f"[notify] CodeBridge delivery failed: {exc}", file=sys.stderr)
            return False
        if result.returncode != 0:
            print(f"[notify] CodeBridge delivery failed: {result.stderr.strip()}", file=sys.stderr)
            return False
        return True


def validate_suite(suite: dict[str, Any]) -> None:
    if suite.get("schema_version") != 1:
        raise ValueError("suite schema_version must be 1")
    profiles = suite.get("profiles")
    duts = suite.get("duts")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("suite profiles must be a non-empty object")
    if not isinstance(duts, list) or not duts:
        raise ValueError("suite duts must be a non-empty list")
    if "use_skill" in suite and not isinstance(suite["use_skill"], bool):
        raise ValueError("suite use_skill must be a boolean")
    if "baseline_token_meter" in suite and not isinstance(suite["baseline_token_meter"], bool):
        raise ValueError("suite baseline_token_meter must be a boolean")
    names = [str(item.get("name") or "") for item in duts if isinstance(item, dict)]
    if len(names) != len(duts) or not all(names) or len(names) != len(set(names)):
        raise ValueError("every DUT must have a unique non-empty name")
    for item in duts:
        profile_name = str(item.get("profile") or "")
        if profile_name not in profiles:
            raise ValueError(f"DUT {item['name']} refers to unknown profile {profile_name!r}")


def selected_runs(suite: dict[str, Any], duts: set[str], seeds: list[int]) -> list[dict[str, Any]]:
    default_seed = int(suite["seed"])
    run_seeds = seeds or [default_seed]
    runs = []
    known = {str(item["name"]) for item in suite["duts"]}
    unknown = duts - known
    if unknown:
        raise ValueError("unknown DUT(s): " + ", ".join(sorted(unknown)))
    for item in suite["duts"]:
        dut = str(item["name"])
        if duts and dut not in duts:
            continue
        for seed in run_seeds:
            runs.append({
                "dut": dut,
                "profile": str(item["profile"]),
                "seed": seed,
                "run_id": f"{dut}__seed_{seed}",
            })
    return runs


def init_workspace(python: str, source_root: Path, workspace: Path, dut: str, env: dict[str, str]) -> None:
    marker = workspace / ".ucagent_batch_initialized.json"
    if marker.is_file():
        try:
            initialized = load_json(marker)
        except (OSError, json.JSONDecodeError):
            initialized = {}
        if initialized.get("dut") == dut and (workspace / dut).is_dir():
            return
    workspace.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["make", f"init_{dut}", f"CWD={workspace}", f"PYTHON={python}"],
        cwd=source_root,
        env=environment_for_python(python, env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot initialize {dut}:\n" + "\n".join(result.stdout.splitlines()[-80:]))
    atomic_write_json(marker, {"dut": dut, "initialized_at": utc_now(), "source_root": str(source_root)})


def write_suite_summary(path: Path, ledger: dict[str, Any]) -> None:
    """Write a compact CSV after every ledger update for live experiment review."""
    fields = [
        "run_id", "dut", "seed", "profile", "status", "attempts", "duration_hours",
        "prompt_tokens", "completion_tokens", "llm_request_count", "token_measurement",
        "stage_index", "checker_repeats", "llm_requests_since_progress",
        "prompt_tokens_since_progress", "failure_reason", "workspace",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for run_id, record in sorted(ledger.get("runs", {}).items()):
        watchdog = record.get("stall_watchdog") if isinstance(record.get("stall_watchdog"), dict) else {}
        token_usage = record.get("token_usage") if isinstance(record.get("token_usage"), dict) else {}
        writer.writerow({
            "run_id": run_id,
            "dut": record.get("dut", ""),
            "seed": record.get("seed", ""),
            "profile": record.get("profile", ""),
            "status": record.get("status", ""),
            "attempts": record.get("attempts", ""),
            "duration_hours": round(float(record.get("duration_seconds", 0) or 0) / 3600, 4),
            "prompt_tokens": token_usage.get("prompt_tokens", ""),
            "completion_tokens": token_usage.get("completion_tokens", ""),
            "llm_request_count": token_usage.get("llm_request_count", ""),
            "token_measurement": token_usage.get("token_measurement", ""),
            "stage_index": watchdog.get("stage_index", ""),
            "checker_repeats": watchdog.get("checker_repeats", ""),
            "llm_requests_since_progress": watchdog.get("llm_requests_since_progress", ""),
            "prompt_tokens_since_progress": watchdog.get("prompt_tokens_since_progress", ""),
            "failure_reason": record.get("failure_reason", ""),
            "workspace": record.get("workspace", ""),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(output.getvalue(), encoding="utf-8")
    os.replace(temporary, path)


def build_agent_command(
    python: str,
    source_root: Path,
    workspace: Path,
    run_dir: Path,
    dut: str,
    seed: int,
    suite: dict[str, Any],
) -> list[str]:
    command = [
        python,
        str(source_root / "ucagent.py"),
        str(workspace),
        dut,
        "--config",
        str(Path(suite["agent_config"]).resolve()),
        "-im",
        str(suite.get("interaction_mode", "standard")),
        "-l",
        "--log",
        "--log-file",
        str(run_dir / "ucagent-log.log"),
        "--msg-file",
        str(run_dir / "ucagent-msg.log"),
        "--seed",
        str(seed),
        "--exit-on-completion",
        "--stream-output",
    ]
    for override in suite.get("overrides", []):
        command.extend(["--override", str(override)])
    if bool(suite.get("use_skill", False)):
        command.append("--use-skill")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--dut", action="append", default=[])
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--env", default="uc")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--restart-run", action="append", default=[])
    parser.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--notify",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Send CodeBridge notifications; disabled by default for portable public runs.",
    )
    parser.add_argument("--notify-scope", default="")
    parser.add_argument("--skip-model-preflight", action="store_true")
    parser.add_argument("--skip-embedding-preflight", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    suite_path = args.suite.resolve()
    suite = load_json(suite_path)
    validate_suite(suite)
    runs = selected_runs(suite, set(args.dut), args.seed)
    source_root = Path(suite["source_root"]).resolve()
    snapshot_paths = suite.get("source_snapshot_paths")
    expected_snapshot = str(suite.get("source_snapshot_sha256") or "")
    if snapshot_paths and expected_snapshot:
        actual_snapshot = sha256_tree(source_root, [str(path) for path in snapshot_paths])
        if actual_snapshot != expected_snapshot:
            raise RuntimeError(
                f"source snapshot mismatch for {source_root}: "
                f"expected {expected_snapshot}, found {actual_snapshot}"
            )
    for item in runs:
        if not (source_root / "examples" / item["dut"]).is_dir():
            raise FileNotFoundError(f"missing DUT example: {source_root / 'examples' / item['dut']}")

    print(f"Suite: {suite_path}")
    print(f"Run root: {args.run_root.resolve()}")
    print("Runs:")
    for item in runs:
        profile = suite["profiles"][item["profile"]]
        print(
            f"  {item['run_id']}: profile={item['profile']} "
            f"timeout={profile['timeout_hours']}h stage_cap={profile['stage_timeout_hours']}h"
        )
    if args.dry_run:
        return 0

    notifier = CodeBridgeNotifier(enabled=args.notify, scope=args.notify_scope)
    notifier.preflight()
    python = conda_python(args.env)
    required_packages = suite.get("required_python_packages", {})
    if required_packages:
        check_python_package_versions(python, required_packages)
    if bool(suite.get("check_summarization_middleware", False)):
        check_summarization_middleware_compatibility(python, source_root)
    model = str(suite["model"])
    api_base = str(suite["api_base"]).rstrip("/")
    api_key = os.environ.get("OPENAI_API_KEY", "ollama")
    num_ctx = int(suite["num_ctx"])
    if not args.skip_model_preflight:
        check_model_available(api_base, model, api_key)
        check_ollama_model_context(api_base, model, num_ctx)
    if not args.skip_embedding_preflight:
        check_embedding_service("http://127.0.0.1:5000/v1")
    if not args.skip_warmup:
        warmup_ollama_model(api_base, model, api_key, num_ctx, runs[0]["seed"], timeout=900, status_interval=30)
    if bool(suite.get("requires_langchain_tool_call_preflight", False)):
        preflight_env = build_run_environment(
            python,
            {},
            {"environment": suite.get("environment", {})},
            model,
            api_base,
            api_key,
            num_ctx,
            None,
            seed=int(runs[0]["seed"]),
        )
        for key in suite.get("unset_environment", []):
            preflight_env.pop(str(key), None)
        check_langchain_tool_call_path(
            python,
            source_root,
            Path(suite["agent_config"]).resolve(),
            preflight_env,
            [str(value) for value in suite.get("overrides", [])],
            int(runs[0]["seed"]),
        )

    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    ledger_path = run_root / "ledger.json"
    summary_path = run_root / "summary.csv"
    ledger = load_json(ledger_path) if ledger_path.exists() else {
        "schema_version": 1,
        "experiment": suite["experiment"],
        "suite_path": str(suite_path),
        "created_at": utc_now(),
        "runs": {},
    }
    restart = set(args.restart_run)

    for index, item in enumerate(runs, start=1):
        dut, seed, run_id = item["dut"], item["seed"], item["run_id"]
        profile = suite["profiles"][item["profile"]]
        run_dir = run_root / run_id
        workspace = run_dir / "workspace"
        if run_id in restart and run_dir.exists():
            import shutil
            shutil.rmtree(run_dir)
            ledger["runs"].pop(run_id, None)
        record = ledger["runs"].setdefault(run_id, {})
        if workspace_completed(workspace):
            record.update({"status": "completed", "completed_from_workspace": True})
            write_suite_summary(summary_path, ledger)
            print(f"SKIP completed {run_id}")
            continue
        if record.get("status") in {"failed", "timeout", "stalled"} and not args.retry_failed:
            print(f"SKIP {record['status']} {run_id}; pass --retry-failed to resume")
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        env = build_run_environment(
            python,
            {},
            {"environment": suite.get("environment", {})},
            model,
            api_base,
            api_key,
            num_ctx,
            run_dir / "home",
            seed=seed,
        )
        for key in suite.get("unset_environment", []):
            env.pop(str(key), None)
        baseline_token_meter = bool(suite.get("baseline_token_meter", False))
        if baseline_token_meter:
            env = enable_baseline_token_meter(env, run_dir, source_root / "ucagent.py")
        init_workspace(python, source_root, workspace, dut, env)
        command = build_agent_command(python, source_root, workspace, run_dir, dut, seed, suite)
        started = time.monotonic()
        previous_attempts = int(record.get("attempts", 0))
        token_log_path = run_dir / BASELINE_TOKEN_LOG_NAME if baseline_token_meter else None
        prior_runtime = parse_agent_log_summary(run_dir / "ucagent-log.log", token_log_path)
        attempt_entry = {
            "attempt": previous_attempts + 1,
            "started_at": utc_now(),
            "status": "running",
            "active_seconds_before_attempt": float(prior_runtime.get("active_seconds", 0) or 0),
        }
        record.setdefault("attempt_history", []).append(attempt_entry)
        record.update({
            "dut": dut,
            "seed": seed,
            "profile": item["profile"],
            "status": "running",
            "started_at": utc_now(),
            "attempts": previous_attempts + 1,
            "workspace": str(workspace),
            "command": command,
            "limits": profile,
            "failure_reason": "",
            "stall_watchdog": {},
        })
        if int(record.get("attempts", 0)) > 1:
            resume_entry = {
                "resumed_at": utc_now(),
                "patch_label": str(suite.get("resume_patch_label") or "multi_dut_resume"),
                "experimental_validity": "patched_resume_not_clean_end_to_end",
            }
            record.setdefault("resume_attempts", []).append(resume_entry)
            record["experimental_validity"] = resume_entry["experimental_validity"]
        ledger["updated_at"] = utc_now()
        atomic_write_json(ledger_path, ledger)
        write_suite_summary(summary_path, ledger)
        notifier.send(f"[UCAgent批处理] 开始 {run_id} ({index}/{len(runs)})，档位={item['profile']}")

        watchdog_enabled = bool(profile.get("watchdog_enabled", True))
        watchdog = (
            StageStallWatchdog(
                checker_repeat_limit=int(profile["checker_repeat_limit"]),
                llm_request_limit=int(profile["llm_request_limit"]),
                prompt_token_limit=int(profile["prompt_token_limit"]),
                no_progress_seconds=float(profile["no_progress_minutes"]) * 60,
                stage_timeout_seconds=float(profile["stage_timeout_hours"]) * 3600,
            )
            if watchdog_enabled
            else None
        )
        try:
            rc, timed_out, reason = run_command(
                command,
                env,
                run_dir / "console.log",
                int(float(profile["timeout_hours"]) * 3600),
                source_root,
                stall_watchdog=watchdog,
            )
        except KeyboardInterrupt:
            record.update({"status": "interrupted", "ended_at": utc_now()})
            ledger["updated_at"] = utc_now()
            atomic_write_json(ledger_path, ledger)
            write_suite_summary(summary_path, ledger)
            notifier.send(f"[UCAgent批处理] 人工中断 {run_id}，可用相同命令续跑。")
            return 130

        completed = workspace_completed(workspace)
        status = "completed" if completed else ("timeout" if timed_out else ("stalled" if reason and reason.startswith("stage_stall_watchdog") else "failed"))
        attempt_duration = round(time.monotonic() - started, 3)
        token_usage = parse_agent_log_summary(run_dir / "ucagent-log.log", token_log_path)
        total_active_seconds = round(
            float(token_usage.get("active_seconds", 0) or 0)
            or (float(prior_runtime.get("active_seconds", 0) or 0) + attempt_duration),
            3,
        )
        attempt_entry.update({
            "status": status,
            "ended_at": utc_now(),
            "duration_seconds": attempt_duration,
            "return_code": rc,
            "failure_reason": reason,
        })
        record.update({
            "status": status,
            "return_code": rc,
            "failure_reason": reason,
            "ended_at": utc_now(),
            "duration_seconds": total_active_seconds,
            "latest_attempt_duration_seconds": attempt_duration,
            "stall_watchdog": watchdog.snapshot() if watchdog is not None else {"enabled": False},
            "token_usage": token_usage,
        })
        ledger["updated_at"] = utc_now()
        atomic_write_json(ledger_path, ledger)
        write_suite_summary(summary_path, ledger)
        detail = watchdog.snapshot() if watchdog is not None else {}
        suffix = f"，Stage={detail.get('stage_index')}，Checker重复={detail.get('checker_repeats')}" if status == "stalled" else ""
        notifier.send(
            f"[UCAgent批处理] {run_id} 结束：{status}，"
            f"本次={attempt_duration / 3600:.2f}h，累计active={total_active_seconds / 3600:.2f}h{suffix}"
        )
        print(
            f"FINISH {run_id}: {status} rc={rc} "
            f"attempt={attempt_duration / 3600:.2f}h total_active={total_active_seconds / 3600:.2f}h"
        )
        if status != "completed" and not args.keep_going:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
