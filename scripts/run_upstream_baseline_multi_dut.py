#!/usr/bin/env python3
"""Run untouched upstream baselines with at most two clean attempts per DUT."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = REPO_ROOT / "benchmark/ucagent_experiments/multi_dut_upstream_baseline_v1_suite.json"
DEFAULT_RUN_ROOT = REPO_ROOT / "benchmark/ucagent_experiments/multi_dut_upstream_baseline_20260902_runs"
RUNNER = REPO_ROOT / "scripts/run_multi_dut_experiments.py"
INFRASTRUCTURE_ERROR_SIGNATURES = (
    "connection error.",
    "peer closed connection without sending complete message body",
    "incomplete chunked read",
    "httpcore.remoteprotocolerror",
    "httpx.remoteprotocolerror",
    "remotedisconnected",
    "apiconnectionerror",
    "connection refused",
    "server disconnected",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def workspace_completed(workspace: Path) -> bool:
    for path in (workspace / ".ucagent/ucagent_info.json", workspace / ".ucagent_info.json"):
        if not path.is_file():
            continue
        try:
            if bool(load_json(path).get("all_completed", False)):
                return True
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return False


def attempt_record(attempt_root: Path, run_id: str) -> dict[str, Any]:
    ledger_path = attempt_root / "ledger.json"
    if not ledger_path.is_file():
        return {}
    return dict(load_json(ledger_path).get("runs", {}).get(run_id, {}))


def is_infrastructure_failure(attempt_root: Path, run_id: str, record: dict[str, Any]) -> bool:
    """Separate transport failures from DUT/agent completion failures."""
    reason = str(record.get("failure_reason") or "").lower()
    if any(signature in reason for signature in INFRASTRUCTURE_ERROR_SIGNATURES):
        return True
    console_path = attempt_root / run_id / "console.log"
    if not console_path.is_file():
        return False
    try:
        tail = console_path.read_bytes()[-256_000:].decode("utf-8", errors="ignore").lower()
    except OSError:
        return False
    return any(signature in tail for signature in INFRASTRUCTURE_ERROR_SIGNATURES)


def archive_infrastructure_failure(
    root: Path,
    attempt_root: Path,
    run_id: str,
    clean_attempt: int,
    record: dict[str, Any],
) -> Path | None:
    """Preserve diagnostics while resetting the clean attempt workspace."""
    run_dir = attempt_root / run_id
    archive_root = root / "infrastructure_failures"
    archive_root.mkdir(parents=True, exist_ok=True)
    sequence = 1
    destination = archive_root / f"{run_id}__clean_{clean_attempt}__infra_{sequence}"
    while destination.exists():
        sequence += 1
        destination = archive_root / f"{run_id}__clean_{clean_attempt}__infra_{sequence}"
    if run_dir.exists():
        shutil.move(str(run_dir), str(destination))
        (destination / "infrastructure_record.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        destination = None

    ledger_path = attempt_root / "ledger.json"
    if ledger_path.is_file():
        ledger = load_json(ledger_path)
        ledger.get("runs", {}).pop(run_id, None)
        ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def write_outcomes(path: Path, outcomes: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "outcomes.json").write_text(
        json.dumps(outcomes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (path / "outcomes.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "dut", "seed", "status", "successful_attempt", "attempts_run",
            "duration_hours", "prompt_tokens", "completion_tokens", "llm_request_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dut, result in outcomes["duts"].items():
            writer.writerow({key: result.get(key, "") for key in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--dut", action="append", default=[])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--env", default="uc")
    parser.add_argument("--no-notify", action="store_true")
    parser.add_argument("--infrastructure-retries", type=int, default=3)
    parser.add_argument("--infrastructure-retry-delay", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.attempts != 2:
        parser.error("the frozen baseline policy requires exactly two clean attempts")
    if args.infrastructure_retries < 0 or args.infrastructure_retry_delay < 0:
        parser.error("infrastructure retry values must be non-negative")

    suite = load_json(args.suite.resolve())
    known_duts = [str(item["name"]) for item in suite["duts"]]
    duts = args.dut or known_duts
    unknown = sorted(set(duts) - set(known_duts))
    if unknown:
        parser.error("unknown DUT(s): " + ", ".join(unknown))
    seed = int(args.seed if args.seed is not None else suite["seed"])
    root = args.run_root.resolve()
    print(f"Baseline suite: {args.suite.resolve()}")
    print(f"Run root: {root}")
    print(f"DUTs: {', '.join(duts)}")
    print("Policy: watchdog disabled; 16h total timeout; at most two clean attempts")
    if args.dry_run:
        return 0

    outcomes: dict[str, Any] = {
        "schema_version": 1,
        "suite": str(args.suite.resolve()),
        "seed": seed,
        "policy": {
            "watchdog": "disabled",
            "attempts": 2,
            "retry_workspace": "fresh",
            "infrastructure_failures_consume_attempt": False,
            "infrastructure_retries_per_clean_attempt": args.infrastructure_retries,
            "terminal_failure": "cannot_complete",
            "token_measurement": "external_langchain_callback",
        },
        "duts": {},
    }
    root.mkdir(parents=True, exist_ok=True)

    for dut in duts:
        run_id = f"{dut}__seed_{seed}"
        attempts: list[dict[str, Any]] = []
        infrastructure_failures: list[dict[str, Any]] = []
        successful_attempt = 0
        for attempt in range(1, args.attempts + 1):
            attempt_root = root / f"attempt_{attempt}"
            workspace = attempt_root / run_id / "workspace"
            infrastructure_retry = 0
            while True:
                record = attempt_record(attempt_root, run_id)
                if workspace_completed(workspace):
                    successful_attempt = attempt
                    attempts.append(record)
                    break
                if record and is_infrastructure_failure(attempt_root, run_id, record):
                    archived = archive_infrastructure_failure(root, attempt_root, run_id, attempt, record)
                    infrastructure_failures.append({
                        "clean_attempt": attempt,
                        "retry": infrastructure_retry,
                        "archive": str(archived) if archived else "",
                        "record": record,
                    })
                    record = {}
                if record.get("status") in {"failed", "timeout", "stalled"}:
                    attempts.append(record)
                    break
                command = [
                    sys.executable,
                    str(RUNNER),
                    "--suite", str(args.suite.resolve()),
                    "--run-root", str(attempt_root),
                    "--dut", dut,
                    "--seed", str(seed),
                    "--env", args.env,
                    "--skip-embedding-preflight",
                ]
                if args.no_notify:
                    command.append("--no-notify")
                print(f"START baseline {dut} clean attempt {attempt}/2", flush=True)
                process = subprocess.run(command, cwd=REPO_ROOT, check=False)
                record = attempt_record(attempt_root, run_id)
                infrastructure_error = not record or is_infrastructure_failure(attempt_root, run_id, record)
                if process.returncode != 0 and infrastructure_error:
                    archived = archive_infrastructure_failure(root, attempt_root, run_id, attempt, record)
                    infrastructure_retry += 1
                    infrastructure_failures.append({
                        "clean_attempt": attempt,
                        "retry": infrastructure_retry,
                        "archive": str(archived) if archived else "",
                        "record": record,
                    })
                    print(
                        f"INFRASTRUCTURE baseline {dut} clean attempt {attempt}: "
                        f"retry {infrastructure_retry}/{args.infrastructure_retries}",
                        flush=True,
                    )
                    if infrastructure_retry > args.infrastructure_retries:
                        break
                    if args.infrastructure_retry_delay:
                        time.sleep(args.infrastructure_retry_delay)
                    continue
                attempts.append(record)
                break
            if workspace_completed(workspace):
                successful_attempt = attempt
                break
            if infrastructure_retry > args.infrastructure_retries and not attempts:
                break

        selected = attempts[successful_attempt - 1] if successful_attempt else (attempts[-1] if attempts else {})
        token = selected.get("token_usage") if isinstance(selected.get("token_usage"), dict) else {}
        result = {
            "dut": dut,
            "seed": seed,
            "status": (
                "completed" if successful_attempt
                else ("infrastructure_error" if infrastructure_failures and not attempts else "cannot_complete")
            ),
            "successful_attempt": successful_attempt or "",
            "attempts_run": len(attempts),
            "duration_hours": round(float(selected.get("duration_seconds", 0) or 0) / 3600, 4),
            "prompt_tokens": token.get("prompt_tokens", ""),
            "completion_tokens": token.get("completion_tokens", ""),
            "llm_request_count": token.get("llm_request_count", ""),
            "attempt_records": attempts,
            "infrastructure_failures": infrastructure_failures,
        }
        outcomes["duts"][dut] = result
        write_outcomes(root, outcomes)
        print(f"FINISH baseline {dut}: {result['status']}", flush=True)

    return 0 if all(item["status"] == "completed" for item in outcomes["duts"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
