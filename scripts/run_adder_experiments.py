#!/usr/bin/env python3
"""Run a frozen Adder experiment matrix with resumable state.

Each arm/seed pair owns an isolated workspace and log directory.  The runner
never edits ``ucagent/setting.yaml``: arm differences are expressed through
environment variables and CLI overrides recorded in the experiment ledger.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import re
import select
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = (
    REPO_ROOT
    / "benchmark/ucagent_experiments/adder_upstream_20260723_qwen_compatible_baseline_matrix.json"
)
DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "benchmark/ucagent_experiments/adder_upstream_20260723_qwen_compatible_baseline_runs"
)
TERMINAL_STATES = {"completed"}
EMPTY_AI_TURN_LIMIT = 3
ACTIVE_LOG_IDLE_GAP_SECONDS = 30 * 60
ANSI_ESCAPE_RE = re.compile(rb"\x1b\[[0-?]*[ -/]*[@-~]")
LOG_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}")
LLM_REQUEST_RE = re.compile(r"\[data_collection\]\[llm_request\]")
PROMPT_TOKENS_RE = re.compile(r"\bprompt_tokens=(\d+)")
COMPLETION_TOKENS_RE = re.compile(r"\bcompletion_tokens=(\d+)")
TTFT_RE = re.compile(r"\bttft_ms=([\d.]+|NA)")
STRUCTURED_EVENT_MARKER = "[data_collection][structured_event]"
PYTEST_NODE_LINE_RE = re.compile(r"(?P<path>[^\s,:]+\.py):\d+(?:-\d+)?(?P<node>::[^\s,\]]+)")
STAGE_PROGRESS_RE = re.compile(r"\[(\d+)/(\d+)\]\s*$")
CHECKER_PROGRESS_PATTERNS = (
    (
        "unmarked_test_functions",
        re.compile(r"\bFind\s+(\d+)\s+functions?\s+do not have correct check point marks\b", re.IGNORECASE),
    ),
    (
        "unmarked_check_points",
        re.compile(r"\bunmarked_check_points\b[^0-9]{0,20}(\d+)", re.IGNORECASE),
    ),
)
BASELINE_TOKEN_METER = REPO_ROOT / "scripts/baseline_token_meter.py"
BASELINE_TOKEN_METER_BOOTSTRAP = REPO_ROOT / "scripts/baseline_token_meter_bootstrap"
BASELINE_TOKEN_LOG_NAME = "baseline_token_usage.jsonl"
RES_CSV_FIELDS = (
    "Model",
    "DUT",
    "time",
    "token_in",
    "token_out",
    "stage_cache_hit_rate",
    "prefetch_hit_rate",
    "stage_first_turn_hit_rate",
    "retrieval_hit_rate",
    "recent_fallback_rate",
    "useful_hit_rate",
    "retrieval_useful_hit_rate",
    "prefetch_useful_hit_rate",
    "fallback_useful_hit_rate",
    "stale_hit_rate",
    "memory_pollution_rate",
    "prefetch_pollution_rate",
    "memory_token_increase",
    "token_roi_memory",
)
SOURCE_TREE_EXCLUDES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}


class AgentOutputGuard:
    """Detect repeated empty final AI turns in streamed console output."""

    def __init__(self, empty_turn_limit: int = EMPTY_AI_TURN_LIMIT) -> None:
        self.empty_turn_limit = empty_turn_limit
        self._line_buffer = b""
        self._awaiting_ai_payload = False
        self.consecutive_empty_turns = 0

    def feed(self, chunk: bytes) -> str | None:
        self._line_buffer += chunk
        while b"\n" in self._line_buffer:
            raw_line, self._line_buffer = self._line_buffer.split(b"\n", 1)
            line = ANSI_ESCAPE_RE.sub(b"", raw_line).decode("utf-8", errors="replace").strip()
            if "AI Message" in line:
                self._awaiting_ai_payload = True
                continue
            if not self._awaiting_ai_payload or not line:
                continue
            self._awaiting_ai_payload = False
            if line == "None":
                self.consecutive_empty_turns += 1
                if self.consecutive_empty_turns >= self.empty_turn_limit:
                    return "consecutive_empty_ai_turns"
            else:
                self.consecutive_empty_turns = 0
        return None


class StageStallWatchdog:
    """Abort expensive runs only when structured evidence shows no progress."""

    def __init__(
        self,
        *,
        checker_repeat_limit: int = 6,
        llm_request_limit: int = 40,
        prompt_token_limit: int = 1_200_000,
        no_progress_seconds: float = 3600.0,
        stage_timeout_seconds: float = 4 * 3600.0,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.checker_repeat_limit = checker_repeat_limit
        self.llm_request_limit = llm_request_limit
        self.prompt_token_limit = prompt_token_limit
        self.no_progress_seconds = no_progress_seconds
        self.stage_timeout_seconds = stage_timeout_seconds
        self._now_fn = now_fn
        self._line_buffer = b""
        self.current_stage: int | None = None
        self.current_stage_name = ""
        self.current_stage_progress: tuple[int, int] | None = None
        self.stage_started_at = self._now_fn()
        self.last_progress_at = self.stage_started_at
        self.semantic_checker_key: tuple[str, ...] = ()
        self.checker_repeats = 0
        self.llm_requests_since_progress = 0
        self.prompt_tokens_since_progress = 0
        self.trigger: dict[str, Any] | None = None

    @staticmethod
    def _normalize_evidence(value: Any) -> str:
        text = " ".join(str(value).split())
        text = PYTEST_NODE_LINE_RE.sub(r"\g<path>\g<node>", text)
        return text[:1000]

    @classmethod
    def _checker_key(cls, event: dict[str, Any]) -> tuple[str, ...]:
        result = event.get("result") if isinstance(event.get("result"), dict) else {}
        evidence = {
            "case:" + cls._normalize_evidence(value)
            for value in result.get("failed_cases_top", []) or []
            if str(value).strip()
        }
        evidence.update(
            "checkpoint:" + cls._normalize_evidence(value)
            for value in result.get("failed_checkpoints_top", []) or []
            if str(value).strip()
        )
        # Exact identifiers in Checker errors remain stable when a broad error
        # is classified differently on consecutive calls.
        for value in result.get("error_top", []) or []:
            text = cls._normalize_evidence(value)
            for metric_name, pattern in CHECKER_PROGRESS_PATTERNS:
                match = pattern.search(text)
                if match:
                    evidence.add(f"metric:{metric_name}:{match.group(1)}")
            for match in PYTEST_NODE_LINE_RE.finditer(text):
                evidence.add("case:" + match.group("path") + match.group("node"))
            evidence.update(
                "contract:" + token
                for token in re.findall(r"(?:FG|FC|CK)-[A-Za-z0-9_.-]+", text)
            )
            evidence.update(
                "tc:" + token
                for token in re.findall(r"TC-[A-Za-z0-9_./:-]+", text)
            )
        if evidence:
            return tuple(sorted(evidence))
        categories = result.get("checker_categories", []) or []
        return tuple(sorted("category:" + cls._normalize_evidence(value) for value in categories))

    @staticmethod
    def _stage_progress(event: dict[str, Any]) -> tuple[int, int] | None:
        title = str(event.get("stage_title", ""))
        match = STAGE_PROGRESS_RE.search(title)
        return (int(match.group(1)), int(match.group(2))) if match else None

    def _reset_progress(self, now: float, *, clear_checker: bool = True) -> None:
        self.last_progress_at = now
        self.llm_requests_since_progress = 0
        self.prompt_tokens_since_progress = 0
        if clear_checker:
            self.semantic_checker_key = ()
            self.checker_repeats = 0

    def _set_stage(self, event: dict[str, Any], now: float) -> None:
        stage = event.get("stage_index")
        if not isinstance(stage, int):
            return
        if stage != self.current_stage:
            self.current_stage = stage
            self.current_stage_name = str(event.get("stage_name", ""))
            self.current_stage_progress = self._stage_progress(event)
            self.stage_started_at = now
            self._reset_progress(now)
            return
        progress = self._stage_progress(event)
        if (
            progress is not None
            and self.current_stage_progress is not None
            and progress[1] == self.current_stage_progress[1]
            and progress[0] > self.current_stage_progress[0]
        ):
            self.current_stage_progress = progress
            self._reset_progress(now)
        elif progress is not None and self.current_stage_progress is None:
            self.current_stage_progress = progress

    def _record_llm_request(self, line: str) -> None:
        if "[data_collection][llm_request]" not in line or "status=success" not in line:
            return
        self.llm_requests_since_progress += 1
        match = PROMPT_TOKENS_RE.search(line)
        if match:
            self.prompt_tokens_since_progress += int(match.group(1))

    def _record_event(self, event: dict[str, Any], now: float) -> bool:
        """Record one event and report whether it is a safe abort boundary."""
        self._set_stage(event, now)
        event_type = event.get("event_type")
        if event_type == "stage_transition" and event.get("advanced") is True:
            to_stage = event.get("to_stage")
            if isinstance(to_stage, dict):
                self._set_stage(to_stage, now)
            self._reset_progress(now)
            return False
        if event_type != "check_result":
            return False
        result = event.get("result") if isinstance(event.get("result"), dict) else {}
        if result.get("check_pass") is True or event.get("success") is True:
            self._reset_progress(now)
            return False
        key = self._checker_key(event)
        if not key:
            return False
        if key == self.semantic_checker_key:
            self.checker_repeats += 1
        else:
            self.semantic_checker_key = key
            self.checker_repeats = 1
            self._reset_progress(now, clear_checker=False)
        # Check/Complete has returned and UCAgent has persisted the verifier
        # observation. Stopping here cannot discard a generated tool action.
        return True

    def _evaluate(self, now: float) -> str | None:
        if self.trigger is not None or self.current_stage is None:
            return None if self.trigger is None else str(self.trigger["reason"])
        no_progress = max(0.0, now - self.last_progress_at)
        stage_elapsed = max(0.0, now - self.stage_started_at)
        repeated_checker = self.checker_repeats >= self.checker_repeat_limit
        cost_exhausted = (
            self.llm_requests_since_progress >= self.llm_request_limit
            or self.prompt_tokens_since_progress >= self.prompt_token_limit
            or no_progress >= self.no_progress_seconds
        )
        absolute_stall = (
            self.checker_repeats >= 2
            and stage_elapsed >= self.stage_timeout_seconds
            and no_progress >= self.no_progress_seconds
        )
        if not ((repeated_checker and cost_exhausted) or absolute_stall):
            return None
        reason = f"stage_stall_watchdog:stage={self.current_stage}"
        self.trigger = {
            "reason": reason,
            "triggered_at": utc_now(),
            "stage_index": self.current_stage,
            "stage_name": self.current_stage_name,
            "stage_progress": self.current_stage_progress,
            "checker_repeats": self.checker_repeats,
            "semantic_checker_key": list(self.semantic_checker_key),
            "llm_requests_since_progress": self.llm_requests_since_progress,
            "prompt_tokens_since_progress": self.prompt_tokens_since_progress,
            "no_progress_seconds": round(no_progress, 3),
            "stage_elapsed_seconds": round(stage_elapsed, 3),
            "limits": {
                "checker_repeats": self.checker_repeat_limit,
                "llm_requests": self.llm_request_limit,
                "prompt_tokens": self.prompt_token_limit,
                "no_progress_seconds": self.no_progress_seconds,
                "stage_timeout_seconds": self.stage_timeout_seconds,
            },
        }
        return reason

    def feed(self, chunk: bytes, *, now: float | None = None) -> str | None:
        current = self._now_fn() if now is None else now
        self._line_buffer += chunk
        while b"\n" in self._line_buffer:
            raw_line, self._line_buffer = self._line_buffer.split(b"\n", 1)
            line = ANSI_ESCAPE_RE.sub(b"", raw_line).decode("utf-8", errors="replace")
            self._record_llm_request(line)
            if STRUCTURED_EVENT_MARKER not in line:
                continue
            payload = line.split(STRUCTURED_EVENT_MARKER, 1)[1].strip()
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                safe_boundary = self._record_event(event, current)
                if safe_boundary:
                    reason = self._evaluate(current)
                    if reason:
                        return reason
        return None

    def poll(self, *, now: float | None = None) -> str | None:
        # Semantic no-progress termination is deferred to a failed verifier
        # boundary. The outer wall-clock timeout remains responsible for a
        # genuinely silent or hung process.
        return str(self.trigger["reason"]) if self.trigger is not None else None

    def snapshot(self) -> dict[str, Any]:
        return dict(self.trigger or {})
FROZEN_CODE_ARTIFACTS = (
    "ucagent/setting.yaml",
    "ucagent/util/test_result.py",
    "ucagent/abackend/langchain/agent.py",
    "ucagent/abackend/langchain/message/conversation.py",
    "ucagent/checkers/base.py",
    "ucagent/checkers/legacy_20260717_unity_test.py",
    "ucagent/checkers/legacy_20260717_unity_test_random.py",
    "ucagent/checkers/toffee_report.py",
    "ucagent/checkers/unity_test.py",
    "ucagent/stage/vmanager.py",
    "ucagent/tools/testops.py",
    "ucagent/util/functions.py",
    "ucagent/verify_agent.py",
    "ucagent/memory/context_reuse.py",
    "scripts/run_adder_experiments.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path, relative_paths: list[str]) -> str:
    """Hash selected source-tree paths, including relative filenames."""
    digest = hashlib.sha256()
    files: list[Path] = []
    for raw_path in relative_paths:
        path = (root / raw_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"source snapshot path does not exist: {path}")
        if path.is_file():
            files.append(path)
            continue
        files.extend(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file()
            and not any(part in SOURCE_TREE_EXCLUDES for part in candidate.relative_to(root).parts)
            and candidate.suffix != ".pyc"
        )
    for path in sorted(set(files), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def resolve_repo_file(raw_path: str | Path, *, label: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def resolve_source_root(fixed: dict[str, Any]) -> Path:
    raw_path = fixed.get("source_root", REPO_ROOT)
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    for required in ("ucagent.py", "Makefile", "ucagent/setting.yaml"):
        if not (path / required).is_file():
            raise FileNotFoundError(f"source root lacks {required}: {path}")
    return path


def artifact_key(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def artifact_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else REPO_ROOT / path


def conda_python(env_name: str) -> str:
    command = ["conda", "run", "-n", env_name, "python", "-c", "import sys; print(sys.executable)"]
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"cannot resolve Python from conda env {env_name!r}: {result.stderr.strip()}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines or not Path(lines[-1]).is_file():
        raise RuntimeError(f"conda env {env_name!r} did not return a Python executable")
    return lines[-1]


def environment_for_python(python: str, base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base if base is not None else os.environ)
    python_bin = str(Path(python).resolve().parent)
    current_path = env.get("PATH", "")
    path_entries = [entry for entry in current_path.split(os.pathsep) if entry]
    env["PATH"] = os.pathsep.join([python_bin, *[entry for entry in path_entries if entry != python_bin]])
    return env


def build_run_environment(
    python: str,
    fixed: dict[str, Any],
    arm: dict[str, Any],
    model: str,
    api_base: str,
    api_key: str,
    num_ctx: int,
    home: Path | None = None,
    seed: int | None = None,
) -> dict[str, str]:
    """Build the exact environment shared by preflight and the real run."""
    env = environment_for_python(python)
    for key in fixed.get("unset_environment", []):
        env.pop(str(key), None)
    summary_environment = {
        "summary_max_tokens": "SUMMARY_MAX_CTX_TOKEN",
        "summary_hard_max_tokens": "SUMMARY_HARD_MAX_CTX_TOKEN",
        "summary_max_output_tokens": "SUMMARY_MAX_SUM_TOKEN",
        "summary_max_keep_msgs": "SUMMARY_MAX_KEEP_MSG",
        "summary_tail_keep_msgs": "SUMMARY_TAIL_KEEP_MSG",
    }
    # Experiment matrices are authoritative. Do not inherit stale Makefile or
    # interactive-shell summary settings when a matrix omits a field.
    for env_name in summary_environment.values():
        env.pop(env_name, None)
    env.update({str(key): str(value) for key, value in arm.get("environment", {}).items()})
    if seed is not None:
        # The run seed, rather than a stale arm-level constant, controls Python
        # hash randomization for reproducible per-seed behavior.
        env["PYTHONHASHSEED"] = str(seed)
    env.update(
        {
            "OPENAI_MODEL": model,
            "OPENAI_API_BASE": api_base,
            "OPENAI_API_KEY": api_key,
            "OLLAMA_NUM_CTX": str(num_ctx),
        }
    )
    for field, env_name in summary_environment.items():
        if field in fixed:
            env[env_name] = str(int(fixed[field]))
    if fixed.get("summary_model"):
        env.update(
            {
                "SUMMARY_MODEL": str(fixed["summary_model"]),
                "SUMMARY_OPENAI_API_BASE": str(fixed.get("summary_api_base", api_base)),
                "SUMMARY_OPENAI_API_KEY": str(fixed.get("summary_api_key", api_key)),
            }
        )
    if fixed.get("embedding_model"):
        env.update(
            {
                "EMBED_MODEL": str(fixed["embedding_model"]),
                "EMBED_OPENAI_API_BASE": str(fixed.get("embedding_api_base", "")),
                "EMBED_OPENAI_API_KEY": str(fixed.get("embedding_api_key", "EMPTY")),
            }
        )
    if home is not None:
        home.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(home)
    return env


def enable_baseline_token_meter(
    env: dict[str, str],
    run_dir: Path,
    agent_entrypoint: Path,
) -> dict[str, str]:
    """Inject token telemetry without editing the upstream baseline source."""
    if not BASELINE_TOKEN_METER.is_file():
        raise FileNotFoundError(f"baseline token meter does not exist: {BASELINE_TOKEN_METER}")
    sitecustomize = BASELINE_TOKEN_METER_BOOTSTRAP / "sitecustomize.py"
    if not sitecustomize.is_file():
        raise FileNotFoundError(f"baseline token meter bootstrap does not exist: {sitecustomize}")

    result = dict(env)
    existing = [value for value in result.get("PYTHONPATH", "").split(os.pathsep) if value]
    injected = [str(BASELINE_TOKEN_METER_BOOTSTRAP), str(BASELINE_TOKEN_METER.parent)]
    result["PYTHONPATH"] = os.pathsep.join(
        injected + [value for value in existing if value not in injected]
    )
    result["UCAGENT_BASELINE_TOKEN_LOG"] = str(run_dir / BASELINE_TOKEN_LOG_NAME)
    result["UCAGENT_BASELINE_TOKEN_TARGET"] = str(agent_entrypoint.resolve())
    return result


def python_package_versions(python: str, package_names: list[str]) -> dict[str, str | None]:
    """Read package versions from the exact interpreter used by the experiment."""
    if not package_names:
        return {}
    script = """
import importlib.metadata
import json
import sys

versions = {}
for name in sys.argv[1:]:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps(versions, sort_keys=True))
"""
    result = subprocess.run(
        [python, "-c", script, *package_names],
        cwd=REPO_ROOT,
        env=environment_for_python(python),
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cannot inspect packages with {python}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    try:
        versions = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid package-version output from {python}: {result.stdout!r}") from exc
    if not isinstance(versions, dict):
        raise RuntimeError(f"invalid package-version output from {python}: {versions!r}")
    return {str(name): None if value is None else str(value) for name, value in versions.items()}


def check_python_package_versions(
    python: str,
    required_versions: dict[str, Any],
) -> dict[str, str]:
    """Require exact package versions so a resumed run cannot cross API versions."""
    expected = {str(name): str(version) for name, version in required_versions.items()}
    actual = python_package_versions(python, sorted(expected))
    mismatches = [
        f"{name}: expected {expected[name]}, found {actual.get(name) or '<missing>'}"
        for name in sorted(expected)
        if actual.get(name) != expected[name]
    ]
    if mismatches:
        pins = " ".join(f"{name}=={version}" for name, version in sorted(expected.items()))
        raise RuntimeError(
            "Python dependency preflight failed:\n"
            + "\n".join(mismatches)
            + f"\nInstall the frozen baseline dependencies with:\n{python} -m pip install {pins}"
        )
    return {name: str(actual[name]) for name in sorted(expected)}


def check_summarization_middleware_compatibility(python: str, source_root: Path) -> dict[str, str]:
    """Exercise the cutoff call that is reached only after a long conversation."""
    script = """
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from ucagent.abackend.langchain.middleware.messages import (
    MessageStatistic,
    TrimAndSummaryMiddleware,
)

middleware = TrimAndSummaryMiddleware(
    MessageStatistic(),
    max_summary_tokens=1024,
    max_keep_msgs=100,
    max_tokens=51200,
    tail_keep_msgs=10,
    model=object(),
).reset_chat()
result = middleware.before_model(
    {
        "messages": [
            SystemMessage(content="role"),
            HumanMessage(content="request"),
            AIMessage(content="answer"),
        ]
    }
)
assert result and result.get("messages"), result
print("ok")
"""
    result = subprocess.run(
        [python, "-c", script],
        cwd=source_root,
        env=environment_for_python(python),
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "summarization middleware compatibility preflight failed:\n"
            + (result.stderr.strip() or result.stdout.strip())
        )
    return {"summarization_cutoff_path": "ok"}


def check_langchain_tool_call_path(
    python: str,
    source_root: Path,
    agent_config: Path,
    runtime_env: dict[str, str],
    overrides: list[str],
    seed: int,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Exercise the real stage-0 prompt and complete tool set before a long run."""
    script = """
import json
import sys

from ucagent.cli import get_override_dict
from ucagent.verify_agent import VerifyAgent

workspace, config_file, seed, overrides_json = sys.argv[1:]
cfg_override = []
for raw_override in json.loads(overrides_json):
    parsed = get_override_dict(raw_override)
    if isinstance(parsed, list):
        cfg_override.extend(parsed)
    else:
        cfg_override.append(parsed)

agent = VerifyAgent(
    workspace=workspace,
    dut_name="Adder",
    output="unity_test",
    config_file=config_file,
    cfg_override=cfg_override,
    stream_output=True,
    seed=int(seed),
    no_embed_tools=True,
    interaction_mode="standard",
    exit_on_completion=True,
)
tips = agent.get_current_tips()
response = agent.backend.model.bind_tools(agent.test_tools).invoke(tips["messages"])
calls = response.tool_calls
valid_tool_names = {tool.name for tool in agent.test_tools}
if not calls or any(call.get("name") not in valid_tool_names for call in calls):
    raise RuntimeError(
        f"expected a parsed UCAgent tool call, got content={response.content!r}, "
        f"tool_calls={calls!r}, invalid_tool_calls={response.invalid_tool_calls!r}"
    )
print(
    json.dumps(
        {
            "status": "ok",
            "tool_names": [call.get("name") for call in calls],
            "available_tool_count": len(agent.test_tools),
            "seed": int(seed),
            "finish_reason": response.response_metadata.get("finish_reason"),
            "input_tokens": (response.usage_metadata or {}).get("input_tokens"),
            "output_tokens": (response.usage_metadata or {}).get("output_tokens"),
        },
        sort_keys=True,
    )
)
"""
    with tempfile.TemporaryDirectory(prefix="ucagent-agent-preflight-") as temp_dir:
        temp_root = Path(temp_dir)
        workspace = temp_root / "workspace"
        home = temp_root / "home"
        home.mkdir()
        env = environment_for_python(python, runtime_env)
        env["HOME"] = str(home)
        init_workspace(python, workspace, source_root, env, quiet=True)
        attempts = []
        for _ in range(3):
            result = subprocess.run(
                [
                    python,
                    "-c",
                    script,
                    str(workspace),
                    str(agent_config),
                    str(seed),
                    json.dumps(overrides),
                ],
                cwd=source_root,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout + 30,
            )
            attempts.append(result)
            if result.returncode == 0:
                break
    if result.returncode != 0:
        detail_lines = []
        for index, attempt in enumerate(attempts, start=1):
            detail_lines.append(f"--- preflight attempt {index}/3 ---")
            detail_lines.extend(attempt.stderr.splitlines()[-30:])
            detail_lines.extend(attempt.stdout.splitlines()[-30:])
        raise RuntimeError(
            "full-agent LangChain tool-call preflight failed:\n"
            + "\n".join(detail_lines)
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    try:
        value = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "invalid full-agent tool-call preflight output:\n" + "\n".join(lines[-40:])
        ) from exc
    if not isinstance(value, dict) or value.get("status") != "ok":
        raise RuntimeError(f"invalid full-agent tool-call preflight result: {value!r}")
    return value


def check_model_available(api_base: str, model: str, api_key: str, timeout: float = 10.0) -> list[str]:
    """Fail before a run when the selected OpenAI-compatible endpoint lacks the model."""
    url = api_base.rstrip("/") + "/models"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"model preflight failed for {url}: {exc}") from exc
    models = sorted(
        str(item.get("id"))
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    )
    if model not in models:
        raise RuntimeError(
            f"model {model!r} is not available at {api_base}; available models: "
            + (", ".join(models) if models else "<none>")
        )
    return models


def check_ollama_model_context(
    api_base: str,
    model: str,
    expected_num_ctx: int,
    timeout: float = 30.0,
) -> int:
    parsed = urllib.parse.urlsplit(api_base)
    base_path = parsed.path.rstrip("/")
    if base_path.endswith("/v1"):
        base_path = base_path[:-3]
    url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, base_path + "/api/show", "", ""))
    request = urllib.request.Request(
        url,
        data=json.dumps({"model": model}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot inspect Ollama model context at {url}: {exc}") from exc
    match = re.search(r"^num_ctx\s+(\d+)\s*$", str(result.get("parameters", "")), re.MULTILINE)
    actual = int(match.group(1)) if match else 0
    if actual != expected_num_ctx:
        raise RuntimeError(
            f"model {model!r} has Modelfile num_ctx={actual or '<missing>'}, "
            f"expected {expected_num_ctx}; OpenAI-compatible requests cannot override it reliably"
        )
    return actual


def ollama_running_models(api_base: str, timeout: float = 3.0) -> list[str]:
    """Return model names currently reported by Ollama's scheduler."""
    parsed = urllib.parse.urlsplit(api_base)
    base_path = parsed.path.rstrip("/")
    if base_path.endswith("/v1"):
        base_path = base_path[:-3]
    url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, base_path + "/api/ps", "", ""))
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    names = {
        str(item.get("name") or item.get("model"))
        for item in value.get("models", [])
        if isinstance(item, dict) and (item.get("name") or item.get("model"))
    }
    return sorted(names)


def warmup_ollama_model(
    api_base: str,
    model: str,
    api_key: str,
    num_ctx: int,
    seed: int,
    timeout: float = 1200.0,
    status_interval: float = 15.0,
) -> dict[str, Any]:
    """Require a real one-token completion and report slow model loading."""
    parsed = urllib.parse.urlsplit(api_base)
    base_path = parsed.path.rstrip("/")
    if base_path.endswith("/v1"):
        base_path = base_path[:-3]
    url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, base_path + "/api/chat", "", ""))
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Only answer OK"}],
        "stream": False,
        "keep_alive": -1,
        "options": {"num_ctx": num_ctx, "num_predict": 1, "seed": seed},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    outcomes: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
            outcomes.put(("result", value))
        except BaseException as exc:  # Hand the original error back to the main thread.
            outcomes.put(("error", exc))

    started = time.monotonic()
    worker = threading.Thread(target=invoke, name="ollama-inference-preflight", daemon=True)
    worker.start()
    print(
        "Inference preflight: requesting a real 1-token completion "
        f"(model={model}, num_ctx={num_ctx}, timeout={int(timeout)}s).",
        flush=True,
    )
    result: dict[str, Any] | None = None
    while result is None:
        elapsed = time.monotonic() - started
        remaining = timeout - elapsed
        if remaining <= 0:
            raise RuntimeError(
                f"64K inference preflight timed out after {timeout:.0f}s for {url}. "
                "Ollama accepted the connection but did not finish model loading/inference; "
                "inspect /api/ps, container logs, GPU memory, and concurrent requests."
            )
        try:
            outcome, value = outcomes.get(timeout=min(max(0.1, status_interval), remaining))
        except queue.Empty:
            try:
                running_models = ollama_running_models(api_base)
                residency = (
                    "resident=" + ",".join(running_models)
                    if running_models
                    else "resident=<none; loading is not yet visible in /api/ps>"
                )
            except Exception as status_exc:
                residency = f"api_ps_status=unavailable({status_exc})"
            print(
                "Inference preflight: still waiting for Ollama "
                f"(elapsed={time.monotonic() - started:.0f}s). "
                f"{residency}. Ollama may expose the model in /api/ps before runner "
                "initialization finishes; a cold 125B Q4 load can take several minutes. "
                "Keep waiting unless the runner log reports an error, and do not probe with a "
                "different num_ctx because that forces Ollama to unload and reload the model.",
                flush=True,
            )
            continue
        if outcome == "result":
            result = value
            break
        exc = value
        if isinstance(exc, urllib.error.HTTPError):
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except OSError:
                detail = ""
            raise RuntimeError(
                f"64K inference preflight failed for {url}: HTTP {exc.code} {detail}"
            ) from exc
        if isinstance(exc, (OSError, urllib.error.URLError, json.JSONDecodeError)):
            raise RuntimeError(f"64K inference preflight failed for {url}: {exc}") from exc
        raise exc
    if not result.get("done"):
        raise RuntimeError(f"64K inference preflight returned no completed response: {result}")
    return {
        "url": url,
        "model": model,
        "num_ctx": num_ctx,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "load_seconds": round(float(result.get("load_duration", 0) or 0) / 1e9, 3),
        "prompt_eval_count": int(result.get("prompt_eval_count", 0) or 0),
        "prompt_eval_seconds": round(float(result.get("prompt_eval_duration", 0) or 0) / 1e9, 3),
    }


def check_embedding_service(url: str, timeout: float = 5.0) -> None:
    health_url = url.rstrip("/")
    if health_url.endswith("/v1"):
        health_url = health_url[:-3]
    health_url += "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
    except (OSError, urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise RuntimeError(f"embedding service preflight failed for {health_url}: {exc}") from exc


def git_revision() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=REPO_ROOT, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def workspace_completed(workspace: Path) -> bool:
    status_paths = (
        workspace / ".ucagent" / "ucagent_info.json",
        workspace / ".ucagent_info.json",
    )
    for status_path in status_paths:
        if not status_path.is_file():
            continue
        try:
            if bool(load_json(status_path).get("all_completed", False)):
                return True
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return False


def validate_matrix(matrix: dict[str, Any]) -> None:
    if matrix.get("schema_version") != 1:
        raise ValueError("matrix schema_version must be 1")
    seeds = matrix.get("seeds")
    arms = matrix.get("arms")
    if not isinstance(seeds, list) or not seeds or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("matrix seeds must be a non-empty integer list")
    if not isinstance(arms, list) or not arms:
        raise ValueError("matrix arms must be a non-empty list")
    ids = [arm.get("id") for arm in arms if isinstance(arm, dict)]
    if len(ids) != len(arms) or len(ids) != len(set(ids)) or not all(ids):
        raise ValueError("every arm must have a unique non-empty id")
    for arm in arms:
        if not isinstance(arm.get("environment", {}), dict):
            raise ValueError(f"arm {arm['id']} environment must be an object")
        if not isinstance(arm.get("overrides", []), list):
            raise ValueError(f"arm {arm['id']} overrides must be a list")
    fixed = matrix.get("fixed_configuration", {})
    if not isinstance(fixed, dict):
        raise ValueError("fixed_configuration must be an object")
    num_ctx = int(fixed.get("num_ctx", 0) or 0)
    soft_limit = int(fixed.get("summary_max_tokens", 0) or 0)
    hard_limit = int(fixed.get("summary_hard_max_tokens", 0) or 0)
    summary_output_limit = int(fixed.get("summary_max_output_tokens", 0) or 0)
    max_keep_msgs = int(fixed.get("summary_max_keep_msgs", 0) or 0)
    tail_keep_msgs = int(fixed.get("summary_tail_keep_msgs", 0) or 0)
    if num_ctx > 0 and soft_limit >= num_ctx:
        raise ValueError(
            "summary_max_tokens must be below num_ctx so summarization starts before the model limit"
        )
    if hard_limit > 0:
        if soft_limit <= 0 or hard_limit <= soft_limit:
            raise ValueError("summary_hard_max_tokens must be greater than summary_max_tokens")
        if num_ctx > 0 and hard_limit >= num_ctx:
            raise ValueError("summary_hard_max_tokens must be below num_ctx")
        if num_ctx > 0 and summary_output_limit > 0 and hard_limit + summary_output_limit > num_ctx:
            raise ValueError(
                "summary_hard_max_tokens plus summary_max_output_tokens must not exceed num_ctx"
            )
    if max_keep_msgs > 0 and tail_keep_msgs > max_keep_msgs:
        raise ValueError("summary_tail_keep_msgs must not exceed summary_max_keep_msgs")


def select_runs(
    matrix: dict[str, Any],
    selected_arms: set[str],
    selected_seeds: set[int],
    allow_unlisted_seeds: bool = False,
) -> list[dict[str, Any]]:
    matrix_seeds = matrix["seeds"]
    unlisted_seeds = selected_seeds - set(matrix_seeds)
    if unlisted_seeds and not allow_unlisted_seeds:
        values = ", ".join(str(seed) for seed in sorted(unlisted_seeds))
        raise ValueError(
            f"selected seed(s) are not declared by the frozen matrix: {values}; "
            "pass --allow-unlisted-seed and use a new --run-root for an explicit seed extension"
        )
    runs: list[dict[str, Any]] = []
    for arm in matrix["arms"]:
        arm_id = str(arm["id"])
        if selected_arms and arm_id not in selected_arms:
            continue
        repeat = int(arm.get("repeat", len(matrix_seeds)))
        if repeat < 1 or repeat > len(matrix_seeds):
            raise ValueError(f"arm {arm_id} repeat must be between 1 and {len(matrix_seeds)}")
        # An explicit seed list selects complete arm/seed runs. This also makes
        # a repeat=1 pilot arm usable for a separately declared multiseed run.
        run_seeds = sorted(selected_seeds) if selected_seeds else matrix_seeds[:repeat]
        for seed in run_seeds:
            runs.append({"arm": arm, "seed": seed, "run_id": f"{arm_id}__seed_{seed}"})
    if not runs:
        raise ValueError("selection produced no runs")
    return runs


def init_workspace(
    python: str,
    workspace: Path,
    source_root: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    quiet: bool = False,
) -> None:
    workspace.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "make",
        "init_Adder",
        f"CWD={workspace}",
        f"PYTHON={python}",
    ]
    result = subprocess.run(
        command,
        cwd=source_root,
        env=environment_for_python(python, env),
        text=quiet,
        stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.STDOUT if quiet else None,
    )
    if result.returncode != 0:
        output = (result.stdout or "") if quiet else ""
        raise RuntimeError(
            f"cannot initialize Adder workspace {workspace}:\n"
            + "\n".join(output.splitlines()[-80:])
        )


def build_agent_command(
    python: str,
    workspace: Path,
    run_dir: Path,
    seed: int,
    arm: dict[str, Any],
    agent_config: str | Path = "ucagent/setting.yaml",
    source_root: Path = REPO_ROOT,
    interaction_mode: str = "advanced",
    stream_output: bool = False,
    dut: str = "Adder",
) -> list[str]:
    agent_entrypoint = source_root / "ucagent.py"
    try:
        entrypoint_arg = str(agent_entrypoint.relative_to(REPO_ROOT))
    except ValueError:
        entrypoint_arg = str(agent_entrypoint)
    command = [
        python,
        entrypoint_arg,
        str(workspace),
        dut,
        "--config",
        str(agent_config),
        "-im",
        interaction_mode,
        "-l",
        "--log",
        "--log-file",
        str(run_dir / "ucagent-log.log"),
        "--msg-file",
        str(run_dir / "ucagent-msg.log"),
        "--seed",
        str(seed),
        "--exit-on-completion",
    ]
    if stream_output:
        command.append("--stream-output")
    for override in arm.get("overrides", []):
        command.extend(["--override", str(override)])
    command.extend(str(value) for value in arm.get("extra_args", []))
    return command


def stop_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        # UCAgent handles SIGINT as a PDB break, so SIGTERM is required here.
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
        process.wait(timeout=10)


def run_command(
    command: list[str],
    env: dict[str, str],
    console_path: Path,
    timeout_seconds: int,
    cwd: Path = REPO_ROOT,
    success_condition: Callable[[], bool] | None = None,
    stall_watchdog: StageStallWatchdog | None = None,
) -> tuple[int, bool, str | None]:
    console_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    fatal_model_error = False
    condition_met = False
    failure_reason = None
    output_tail = b""
    output_guard = AgentOutputGuard()
    with console_path.open("ab") as console:
        console.write(f"\n[{utc_now()}] COMMAND {shlex.join(command)}\n".encode())
        console.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        started = time.monotonic()
        assert process.stdout is not None
        try:
            while True:
                readable, _, _ = select.select(
                    [process.stdout], [], [], 0.1 if success_condition is not None else 0.5
                )
                if readable:
                    chunk = os.read(process.stdout.fileno(), 4096)
                else:
                    chunk = b""
                if chunk:
                    console.write(chunk)
                    console.flush()
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                    output_tail = (output_tail + chunk)[-4096:].lower()
                    if b"model '" in output_tail and b"not found" in output_tail:
                        fatal_model_error = True
                        failure_reason = "model_not_found"
                        break
                    guard_failure = output_guard.feed(chunk)
                    if guard_failure:
                        failure_reason = guard_failure
                        message = (
                            f"\n[runner] aborting after "
                            f"{output_guard.consecutive_empty_turns} consecutive empty AI turns; "
                            "check model stop sequences and tool-call compatibility.\n"
                        ).encode()
                        console.write(message)
                        console.flush()
                        sys.stdout.buffer.write(message)
                        sys.stdout.buffer.flush()
                        break
                    if stall_watchdog is not None:
                        watchdog_failure = stall_watchdog.feed(chunk)
                        if watchdog_failure:
                            failure_reason = watchdog_failure
                            atomic_write_json(
                                console_path.parent / "stall_watchdog.json",
                                stall_watchdog.snapshot(),
                            )
                            snapshot = stall_watchdog.snapshot()
                            message = (
                                "\n[runner] aborting ineffective loop: "
                                f"stage={snapshot.get('stage_index')} "
                                f"checker_repeats={snapshot.get('checker_repeats')} "
                                f"llm_requests={snapshot.get('llm_requests_since_progress')} "
                                f"prompt_tokens={snapshot.get('prompt_tokens_since_progress')} "
                                f"no_progress_seconds={snapshot.get('no_progress_seconds')}.\n"
                            ).encode()
                            console.write(message)
                            console.flush()
                            sys.stdout.buffer.write(message)
                            sys.stdout.buffer.flush()
                            break
                elif process.poll() is not None:
                    break
                if stall_watchdog is not None and not failure_reason:
                    watchdog_failure = stall_watchdog.poll()
                    if watchdog_failure:
                        failure_reason = watchdog_failure
                        atomic_write_json(
                            console_path.parent / "stall_watchdog.json",
                            stall_watchdog.snapshot(),
                        )
                        break
                if success_condition is not None and success_condition():
                    condition_met = True
                    break
                if timeout_seconds > 0 and time.monotonic() - started >= timeout_seconds:
                    timed_out = True
                    break
        except KeyboardInterrupt:
            stop_process_group(process)
            raise
        if timed_out or fatal_model_error or failure_reason or condition_met:
            stop_process_group(process)
        else:
            remainder = process.stdout.read()
            if remainder:
                console.write(remainder)
                sys.stdout.buffer.write(remainder)
                sys.stdout.buffer.flush()
        if timed_out:
            return 124, True, "timeout"
        if condition_met:
            return 0, False, "success_condition"
        if fatal_model_error:
            return 126, False, failure_reason
        if failure_reason:
            return 125, False, failure_reason
        return int(process.wait()), False, None


def parse_token_meter_summary(token_log_path: Path) -> dict[str, Any]:
    """Aggregate external baseline token-meter records."""
    prompt_tokens = 0
    completion_tokens = 0
    request_count = 0
    ttft_values: list[float] = []
    token_sources: dict[str, int] = {}
    invalid_records = 0
    if not token_log_path.is_file():
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "llm_request_count": 0,
            "ttft_values": [],
            "token_sources": {},
            "invalid_records": 0,
        }
    with token_log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_records += 1
                continue
            if not isinstance(record, dict) or record.get("event") != "llm_request":
                continue
            prompt_tokens += max(0, int(record.get("prompt_tokens", 0) or 0))
            completion_tokens += max(0, int(record.get("completion_tokens", 0) or 0))
            request_count += 1
            ttft_ms = record.get("ttft_ms")
            if isinstance(ttft_ms, (int, float)):
                ttft_values.append(float(ttft_ms))
            source = str(record.get("token_source", "unknown"))
            token_sources[source] = token_sources.get(source, 0) + 1
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "llm_request_count": request_count,
        "ttft_values": ttft_values,
        "token_sources": token_sources,
        "invalid_records": invalid_records,
    }


def parse_agent_log_summary(
    log_path: Path,
    token_log_path: Path | None = None,
) -> dict[str, Any]:
    """Recover active runtime and token totals when Agent finalization was skipped."""
    starts: list[datetime] = []
    segments: list[dict[str, Any]] = []
    current_start: datetime | None = None
    current_last_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    prompt_tokens = 0
    completion_tokens = 0
    request_count = 0
    ttft_values: list[float] = []

    if not log_path.is_file():
        result = {
            "active_seconds": 0.0,
            "attempt_segments": [],
            "agent_start_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "llm_request_count": 0,
            "ttft_p50_ms": None,
            "ttft_p95_ms": None,
            "token_measurement": "unavailable",
            "token_sources": {},
            "token_meter_invalid_records": 0,
        }
        if token_log_path is not None:
            external = parse_token_meter_summary(token_log_path)
            if external["llm_request_count"]:
                values = sorted(float(value) for value in external["ttft_values"])

                def meter_percentile(fraction: float) -> float | None:
                    if not values:
                        return None
                    index = min(
                        len(values) - 1,
                        max(0, int(round((len(values) - 1) * fraction))),
                    )
                    return round(values[index], 1)

                result.update(
                    {
                        "prompt_tokens": int(external["prompt_tokens"]),
                        "completion_tokens": int(external["completion_tokens"]),
                        "llm_request_count": int(external["llm_request_count"]),
                        "ttft_p50_ms": meter_percentile(0.50),
                        "ttft_p95_ms": meter_percentile(0.95),
                        "token_measurement": "external_langchain_callback",
                        "token_sources": dict(external["token_sources"]),
                        "token_meter_invalid_records": int(external["invalid_records"]),
                    }
                )
        return result

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            timestamp_match = LOG_TIMESTAMP_RE.match(line)
            timestamp = (
                datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S")
                if timestamp_match
                else None
            )
            if (
                current_start is not None
                and current_last_timestamp is not None
                and timestamp is not None
                and (timestamp - current_last_timestamp).total_seconds() > ACTIVE_LOG_IDLE_GAP_SECONDS
            ):
                duration = max(0.0, (current_last_timestamp - current_start).total_seconds())
                segments.append(
                    {
                        "started_at": current_start.isoformat(),
                        "ended_at": current_last_timestamp.isoformat(),
                        "duration_seconds": duration,
                    }
                )
                current_start = None
                current_last_timestamp = None

            if "Verify Agent started at:" in line and timestamp is not None:
                if current_start is not None and current_last_timestamp is not None:
                    duration = max(0.0, (current_last_timestamp - current_start).total_seconds())
                    segments.append(
                        {
                            "started_at": current_start.isoformat(),
                            "ended_at": current_last_timestamp.isoformat(),
                            "duration_seconds": duration,
                        }
                    )
                current_start = timestamp
                current_last_timestamp = timestamp
                starts.append(timestamp)

            if LLM_REQUEST_RE.search(line):
                prompt_match = PROMPT_TOKENS_RE.search(line)
                completion_match = COMPLETION_TOKENS_RE.search(line)
                ttft_match = TTFT_RE.search(line)
                if prompt_match:
                    prompt_tokens += int(prompt_match.group(1))
                if completion_match:
                    completion_tokens += int(completion_match.group(1))
                request_count += 1
                if ttft_match and ttft_match.group(1) != "NA":
                    ttft_values.append(float(ttft_match.group(1)))

            if timestamp is not None:
                if current_start is not None:
                    current_last_timestamp = timestamp
                last_timestamp = timestamp

            if current_start is not None and any(
                marker in line
                for marker in (
                    "Verify Agent finished at:",
                    "Verify Agent is exited.",
                    "UCAgent encountered an error:",
                )
            ):
                segment_end = timestamp or current_last_timestamp
                if segment_end is not None:
                    segments.append(
                        {
                            "started_at": current_start.isoformat(),
                            "ended_at": segment_end.isoformat(),
                            "duration_seconds": max(0.0, (segment_end - current_start).total_seconds()),
                        }
                    )
                current_start = None
                current_last_timestamp = None

    if current_start is not None and current_last_timestamp is not None:
        duration = max(0.0, (current_last_timestamp - current_start).total_seconds())
        segments.append(
            {
                "started_at": current_start.isoformat(),
                "ended_at": current_last_timestamp.isoformat(),
                "duration_seconds": duration,
            }
        )

    token_measurement = "native_data_collection" if request_count else "unavailable"
    token_sources: dict[str, int] = {}
    token_meter_invalid_records = 0
    if request_count == 0 and token_log_path is not None:
        external = parse_token_meter_summary(token_log_path)
        if external["llm_request_count"]:
            prompt_tokens = int(external["prompt_tokens"])
            completion_tokens = int(external["completion_tokens"])
            request_count = int(external["llm_request_count"])
            ttft_values = list(external["ttft_values"])
            token_sources = dict(external["token_sources"])
            token_meter_invalid_records = int(external["invalid_records"])
            token_measurement = "external_langchain_callback"

    def percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
        return round(ordered[index], 1)

    return {
        "active_seconds": sum(float(segment["duration_seconds"]) for segment in segments),
        "attempt_segments": segments,
        "agent_start_count": len(starts),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "llm_request_count": request_count,
        "ttft_p50_ms": percentile(ttft_values, 0.50),
        "ttft_p95_ms": percentile(ttft_values, 0.95),
        "token_measurement": token_measurement,
        "token_sources": token_sources,
        "token_meter_invalid_records": token_meter_invalid_records,
    }


def attempt_history_seconds(record: dict[str, Any]) -> float | None:
    history = record.get("attempt_history")
    if not isinstance(history, list) or len(history) != int(record.get("attempts", 0)):
        return None
    durations = [item.get("duration_seconds") for item in history if isinstance(item, dict)]
    if len(durations) != len(history) or any(value is None for value in durations):
        return None
    return sum(max(0.0, float(value)) for value in durations)


def format_duration(seconds: float) -> str:
    total_minutes = max(0, int(round(seconds / 60.0)))
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}h{minutes:02d}min" if hours else f"{minutes}min"


def format_tokens_short(tokens: int) -> str:
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.2f}M"
    if tokens >= 1_000:
        return f"{tokens / 1_000:.1f}K"
    return str(tokens)


def append_res_csv(res_path: Path, row: dict[str, str]) -> None:
    """Append one row without changing the repository's historical CSV schema."""
    res_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not res_path.exists() or res_path.stat().st_size == 0
    identity_fields = ("Model", "DUT", "time", "token_in", "token_out")
    if not needs_header:
        with res_path.open("r", encoding="utf-8", newline="") as handle:
            if any(
                all(existing.get(field, "") == row.get(field, "") for field in identity_fields)
                for existing in csv.DictReader(handle)
            ):
                return
    with res_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RES_CSV_FIELDS, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RES_CSV_FIELDS})


def finalize_completed_result(
    record: dict[str, Any],
    run_id: str,
    run_dir: Path,
    res_path: Path,
) -> dict[str, Any]:
    """Persist a reproducible result summary and append it to res.csv once."""
    log_summary = parse_agent_log_summary(
        run_dir / "ucagent-log.log",
        run_dir / BASELINE_TOKEN_LOG_NAME,
    )
    exact_seconds = attempt_history_seconds(record)
    active_seconds = exact_seconds if exact_seconds is not None else float(log_summary["active_seconds"])
    duration_source = "attempt_history" if exact_seconds is not None else "agent_log_recovery"
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "arm": str(record.get("arm", run_id.split("__seed_", 1)[0])),
        "dut": "Adder",
        "seed": record.get("seed"),
        "status": "completed",
        "attempts": int(record.get("attempts", 0)),
        "active_seconds": round(active_seconds, 3),
        "formatted_duration": format_duration(active_seconds),
        "duration_source": duration_source,
        **log_summary,
    }
    # Keep the selected duration after merging the fallback log statistics.
    summary["active_seconds"] = round(active_seconds, 3)
    summary["duration_source"] = duration_source
    atomic_write_json(run_dir / "result_summary.json", summary)

    if not record.get("res_csv_written"):
        append_res_csv(
            res_path,
            {
                "Model": f"{summary['arm']}(s{summary['seed']})",
                "DUT": "Adder",
                "time": str(summary["formatted_duration"]),
                "token_in": format_tokens_short(int(summary["prompt_tokens"])),
                "token_out": format_tokens_short(int(summary["completion_tokens"])),
            },
        )
        record["res_csv_written"] = True
        record["res_csv_path"] = str(res_path)
        record["res_csv_written_at"] = utc_now()
    record["result_summary"] = summary
    return summary


def initial_ledger(
    matrix_path: Path,
    matrix: dict[str, Any],
    python: str,
    api_base: str,
    model: str,
    num_ctx: int,
    python_packages: dict[str, str] | None = None,
    runtime_compatibility: dict[str, str] | None = None,
    baseline_token_meter: bool = False,
) -> dict[str, Any]:
    fixed = matrix.get("fixed_configuration", {})
    source_root = resolve_source_root(fixed)
    artifact_paths = {REPO_ROOT / "scripts/run_adder_experiments.py"}
    if baseline_token_meter:
        artifact_paths.update(
            {
                BASELINE_TOKEN_METER,
                BASELINE_TOKEN_METER_BOOTSTRAP / "sitecustomize.py",
            }
        )
    if source_root == REPO_ROOT:
        artifact_paths.update(REPO_ROOT / path for path in FROZEN_CODE_ARTIFACTS)
    agent_config = resolve_repo_file(
        fixed.get("agent_config", "ucagent/setting.yaml"),
        label="agent configuration",
    )
    artifact_paths.add(agent_config)
    workflow_source = fixed.get("workflow_source_config")
    if workflow_source:
        artifact_paths.add(resolve_repo_file(workflow_source, label="workflow source configuration"))
    for arm in matrix.get("arms", []):
        pack = arm.get("environment", {}).get("UCAGENT_CONTEXT_REUSE_PACK")
        if pack:
            artifact_paths.add(REPO_ROOT / str(pack))
    missing_artifacts = [str(path) for path in artifact_paths if not path.is_file()]
    if missing_artifacts:
        raise FileNotFoundError("missing frozen experiment artifacts: " + ", ".join(missing_artifacts))
    source_snapshot_paths = list(
        fixed.get(
            "source_snapshot_paths",
            ["ucagent", "ucagent.py", "Makefile", "config.yaml", "examples/Adder"],
        )
    )
    source_snapshot_sha256 = sha256_tree(source_root, source_snapshot_paths)
    expected_source_hash = fixed.get("source_snapshot_sha256")
    if expected_source_hash and source_snapshot_sha256 != expected_source_hash:
        raise RuntimeError(
            f"source snapshot mismatch for {source_root}: "
            f"expected {expected_source_hash}, found {source_snapshot_sha256}"
        )
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "matrix_path": str(matrix_path.relative_to(REPO_ROOT)),
        "matrix_sha256": sha256_file(matrix_path),
        "python": python,
        "git": git_revision(),
        "artifacts": {
            artifact_key(path): sha256_file(path)
            for path in sorted(artifact_paths)
        },
        "source_snapshot": {
            "root": str(source_root),
            "paths": source_snapshot_paths,
            "sha256": source_snapshot_sha256,
        },
        "runtime_environment": {
            "OPENAI_MODEL": model,
            "OPENAI_API_BASE": api_base,
            "OLLAMA_NUM_CTX": num_ctx,
            "python_packages": python_packages or {},
            "runtime_compatibility": runtime_compatibility or {},
            "baseline_token_meter": baseline_token_meter,
        },
        "fixed_configuration": matrix.get("fixed_configuration", {}),
        "runs": {},
    }


def validate_frozen_artifacts(ledger: dict[str, Any]) -> None:
    expected = ledger.get("artifacts")
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError("experiment ledger does not contain frozen artifact hashes")
    changed = []
    for raw_path, expected_hash in expected.items():
        path = artifact_path(str(raw_path))
        actual_hash = sha256_file(path) if path.is_file() else "<missing>"
        if actual_hash != expected_hash:
            changed.append(f"{raw_path}: expected {expected_hash}, found {actual_hash}")
    if changed:
        raise RuntimeError(
            "frozen experiment artifacts changed; use a new --run-root:\n" + "\n".join(changed)
        )
    source_snapshot = ledger.get("source_snapshot")
    if isinstance(source_snapshot, dict):
        source_root = Path(str(source_snapshot.get("root", ""))).resolve()
        source_paths = source_snapshot.get("paths")
        expected_hash = source_snapshot.get("sha256")
        if not isinstance(source_paths, list) or not expected_hash:
            raise RuntimeError("experiment ledger contains an invalid source snapshot")
        actual_hash = sha256_tree(source_root, [str(path) for path in source_paths])
        if actual_hash != expected_hash:
            raise RuntimeError(
                "frozen source tree changed; use a new --run-root:\n"
                f"{source_root}: expected {expected_hash}, found {actual_hash}"
            )


def validate_frozen_python_packages(
    ledger: dict[str, Any],
    current_versions: dict[str, str],
) -> None:
    runtime_environment = ledger.get("runtime_environment", {})
    expected = runtime_environment.get("python_packages", {})
    if expected != current_versions:
        raise RuntimeError(
            "frozen Python dependencies changed; use a new --run-root:\n"
            f"expected {expected}, found {current_versions}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--env", default="uc", help="Conda environment (default: uc)")
    parser.add_argument("--api-base", help="OpenAI-compatible /v1 endpoint; overrides matrix and environment")
    parser.add_argument("--model", help="Model id; overrides matrix and environment")
    parser.add_argument("--num-ctx", type=int, help="Ollama context length; overrides matrix")
    parser.add_argument("--skip-model-preflight", action="store_true")
    parser.add_argument("--skip-inference-preflight", action="store_true")
    parser.add_argument(
        "--inference-preflight-timeout-seconds",
        type=float,
        default=1200.0,
        help="Total wait for model loading plus a real one-token inference (default: 1200)",
    )
    parser.add_argument(
        "--inference-preflight-status-seconds",
        type=float,
        default=15.0,
        help="Progress-report interval while waiting for inference preflight (default: 15)",
    )
    parser.add_argument("--skip-agent-preflight", action="store_true")
    parser.add_argument("--skip-embedding-preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true", help="Check services and warm the model, then exit")
    parser.add_argument("--arm", action="append", default=[], help="Run only this arm; repeatable")
    parser.add_argument("--seed", action="append", type=int, default=[], help="Run only this seed; repeatable")
    parser.add_argument(
        "--allow-unlisted-seed",
        action="store_true",
        help="Allow explicit --seed values outside the frozen matrix; requires a new run root",
    )
    parser.add_argument(
        "--baseline-token-meter",
        action="store_true",
        help="Externally meter baseline LLM tokens without modifying the selected source tree",
    )
    parser.add_argument("--timeout-hours", type=float, default=8.0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--stall-watchdog",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Abort only when repeated semantic Checker failures also exceed a cost/time gate",
    )
    parser.add_argument("--stall-checker-repeats", type=int, default=6)
    parser.add_argument("--stall-llm-requests", type=int, default=40)
    parser.add_argument("--stall-prompt-tokens", type=int, default=1_200_000)
    parser.add_argument("--stall-no-progress-minutes", type=float, default=60.0)
    parser.add_argument("--stall-stage-timeout-hours", type=float, default=4.0)
    parser.add_argument("--retry-failed", action="store_true", help="Retry runs already at max attempts")
    parser.add_argument("--restart-run", action="append", default=[], help="Delete and restart a run id")
    parser.add_argument("--results-csv", type=Path, default=REPO_ROOT / "res.csv")
    parser.add_argument(
        "--backfill-results",
        action="store_true",
        help="Recover completed run durations from logs and update res.csv without running Agent",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.inference_preflight_timeout_seconds <= 0:
        parser.error("--inference-preflight-timeout-seconds must be positive")
    if args.inference_preflight_status_seconds <= 0:
        parser.error("--inference-preflight-status-seconds must be positive")
    if args.stall_checker_repeats < 2:
        parser.error("--stall-checker-repeats must be at least 2")
    if args.stall_llm_requests <= 0 or args.stall_prompt_tokens <= 0:
        parser.error("stall cost limits must be positive")
    if args.stall_no_progress_minutes <= 0 or args.stall_stage_timeout_hours <= 0:
        parser.error("stall time limits must be positive")

    matrix_path = args.matrix.resolve()
    run_root = args.run_root.resolve()
    matrix = load_json(matrix_path)
    validate_matrix(matrix)
    runs = select_runs(
        matrix,
        set(args.arm),
        set(args.seed),
        allow_unlisted_seeds=args.allow_unlisted_seed,
    )
    results_csv = args.results_csv.resolve()

    if args.backfill_results:
        ledger_path = run_root / "ledger.json"
        if not ledger_path.is_file():
            raise FileNotFoundError(f"cannot backfill without ledger: {ledger_path}")
        ledger = load_json(ledger_path)
        updated = 0
        for item in runs:
            run_id = item["run_id"]
            record = ledger.get("runs", {}).get(run_id)
            workspace = run_root / run_id / "workspace"
            if not isinstance(record, dict) or not workspace_completed(workspace):
                print(f"SKIP incomplete {run_id}")
                continue
            record["status"] = "completed"
            summary = finalize_completed_result(
                record,
                run_id,
                run_root / run_id,
                results_csv,
            )
            updated += 1
            print(
                f"BACKFILL {run_id}: {summary['formatted_duration']} "
                f"({summary['duration_source']}, attempts={summary['attempts']})"
            )
        ledger["updated_at"] = utc_now()
        atomic_write_json(ledger_path, ledger)
        print(f"Backfilled {updated} completed run(s) into {results_csv}")
        return 0

    python = conda_python(args.env)

    fixed = matrix.get("fixed_configuration", {})
    source_root = resolve_source_root(fixed)
    package_versions = check_python_package_versions(
        python,
        fixed.get("required_python_packages", {}),
    )
    runtime_compatibility = (
        check_summarization_middleware_compatibility(python, source_root)
        if fixed.get("check_summarization_middleware", False)
        else {}
    )
    agent_config = resolve_repo_file(
        fixed.get("agent_config", "ucagent/setting.yaml"),
        label="agent configuration",
    )
    api_base = (
        args.api_base
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_API_BASE_URL")
        or fixed.get("api_base")
    )
    model = args.model or os.environ.get("OPENAI_MODEL") or fixed.get("model")
    num_ctx = args.num_ctx or int(os.environ.get("OLLAMA_NUM_CTX", 0) or 0) or int(fixed.get("num_ctx", 0) or 0)
    if not api_base or not model or num_ctx <= 0:
        raise RuntimeError("API base, model, and positive num_ctx must be provided by CLI, environment, or matrix")
    api_base = str(api_base).rstrip("/")
    model = str(model)
    api_key = os.environ.get("OPENAI_API_KEY", "ollama")
    if not args.skip_model_preflight:
        check_model_available(api_base, model, api_key)
        check_ollama_model_context(api_base, model, num_ctx)

    run_root.mkdir(parents=True, exist_ok=True)
    ledger_path = run_root / "ledger.json"
    ledger = (
        load_json(ledger_path)
        if ledger_path.exists()
        else initial_ledger(
            matrix_path,
            matrix,
            python,
            api_base,
            model,
            num_ctx,
            package_versions,
            runtime_compatibility,
            args.baseline_token_meter,
        )
    )
    if ledger.get("matrix_sha256") != sha256_file(matrix_path):
        raise RuntimeError("matrix changed after this run root was created; use a new --run-root")
    validate_frozen_artifacts(ledger)
    validate_frozen_python_packages(ledger, package_versions)
    ledger_token_meter = bool(
        ledger.get("runtime_environment", {}).get("baseline_token_meter", False)
    )
    if ledger_token_meter != args.baseline_token_meter:
        raise RuntimeError(
            "baseline token-meter mode changed for this run root; use a new --run-root"
        )

    print(f"Matrix: {matrix_path}")
    print(f"Run root: {run_root}")
    print(f"Python: {python}")
    print(f"API base: {api_base}")
    print(f"Model: {model}")
    print(f"Ollama num_ctx: {num_ctx}")
    print(f"Source root: {source_root}")
    print(f"Agent config: {agent_config}")
    print(f"Baseline token meter: {'enabled' if args.baseline_token_meter else 'disabled'}")
    if package_versions:
        print(
            "Python packages: "
            + ", ".join(f"{name}={version}" for name, version in package_versions.items())
        )
    if runtime_compatibility:
        print("Summarization middleware preflight: OK")
    print("Runs: " + ", ".join(item["run_id"] for item in runs))
    if args.dry_run:
        return 0

    requires_embedding = bool(fixed.get("requires_embedding", True))
    if not args.skip_embedding_preflight and requires_embedding:
        check_embedding_service(str(fixed.get("embedding_api_base", "http://127.0.0.1:5000/v1")))
        print("Embedding preflight: OK")
    if not args.skip_inference_preflight:
        preflight = warmup_ollama_model(
            api_base,
            model,
            api_key,
            num_ctx,
            runs[0]["seed"],
            timeout=args.inference_preflight_timeout_seconds,
            status_interval=args.inference_preflight_status_seconds,
        )
        ledger["inference_preflight"] = preflight
        ledger["updated_at"] = utc_now()
        atomic_write_json(ledger_path, ledger)
        print(
            "Inference preflight: OK "
            f"(num_ctx={num_ctx}, elapsed={preflight['elapsed_seconds']}s, "
            f"load={preflight['load_seconds']}s)"
        )
    if not args.skip_agent_preflight and fixed.get("requires_langchain_tool_call_preflight", False):
        agent_preflights = {
            item["run_id"]: check_langchain_tool_call_path(
                python,
                source_root,
                agent_config,
                build_run_environment(
                    python,
                    fixed,
                    item["arm"],
                    model,
                    api_base,
                    api_key,
                    num_ctx,
                    seed=int(item["seed"]),
                ),
                [str(value) for value in item["arm"].get("overrides", [])],
                int(item["seed"]),
            )
            for item in runs
        }
        ledger["langchain_tool_call_preflight"] = agent_preflights
        ledger["updated_at"] = utc_now()
        atomic_write_json(ledger_path, ledger)
        print(
            "Full-agent LangChain tool-call preflight: OK "
            f"(runs={','.join(agent_preflights)}, "
            "all returned parsed UCAgent tool calls)"
        )
    if args.preflight_only:
        return 0

    timeout_seconds = int(args.timeout_hours * 3600)
    for item in runs:
        arm = item["arm"]
        seed = item["seed"]
        run_id = item["run_id"]
        run_dir = run_root / run_id
        workspace = run_dir / "workspace"
        if run_id in set(args.restart_run) and run_dir.exists():
            import shutil

            shutil.rmtree(run_dir)
            ledger.get("runs", {}).pop(run_id, None)

        record = ledger.setdefault("runs", {}).setdefault(run_id, {})
        if record.get("status") in TERMINAL_STATES or workspace_completed(workspace):
            record["status"] = "completed"
            record["completed_from_workspace"] = True
            summary = finalize_completed_result(record, run_id, run_dir, results_csv)
            ledger["updated_at"] = utc_now()
            atomic_write_json(ledger_path, ledger)
            print(f"SKIP completed {run_id} ({summary['formatted_duration']})")
            continue

        attempts = int(record.get("attempts", 0))
        if attempts >= args.max_attempts and not args.retry_failed:
            print(f"SKIP max attempts {run_id}; use --retry-failed or --restart-run {run_id}")
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        init_workspace(python, workspace, source_root)
        isolated_home = run_dir / "home" if fixed.get("isolate_home", False) else None
        env = build_run_environment(
            python,
            fixed,
            arm,
            model,
            api_base,
            api_key,
            num_ctx,
            isolated_home,
            seed=seed,
        )
        if args.baseline_token_meter:
            env = enable_baseline_token_meter(
                env,
                run_dir,
                source_root / "ucagent.py",
            )
        command = build_agent_command(
            python,
            workspace,
            run_dir,
            seed,
            arm,
            agent_config,
            source_root,
            str(fixed.get("interaction_mode", "advanced")),
            bool(fixed.get("stream_output", False)),
        )
        attempt_started_at = utc_now()
        attempt_started_monotonic = time.monotonic()
        attempt_record = {
            "attempt": attempts + 1,
            "started_at": attempt_started_at,
            "status": "running",
        }
        record.setdefault("attempt_history", []).append(attempt_record)
        record.update(
            {
                "arm": arm["id"],
                "description": arm.get("description", ""),
                "seed": seed,
                "status": "running",
                "attempts": attempts + 1,
                "started_at": attempt_started_at,
                "workspace": str(workspace),
                "log_dir": str(run_dir),
                "environment": {
                    **arm.get("environment", {}),
                    "PYTHONHASHSEED": str(seed),
                },
                "baseline_token_meter": args.baseline_token_meter,
                "baseline_token_log": (
                    str(run_dir / BASELINE_TOKEN_LOG_NAME)
                    if args.baseline_token_meter
                    else None
                ),
                "num_ctx": num_ctx,
                "source_root": str(source_root),
                "agent_config": str(agent_config),
                "interaction_mode": str(fixed.get("interaction_mode", "advanced")),
                "stream_output": bool(fixed.get("stream_output", False)),
                "overrides": arm.get("overrides", []),
                "command": command,
                "stall_watchdog": (
                    {
                        "enabled": True,
                        "checker_repeat_limit": args.stall_checker_repeats,
                        "llm_request_limit": args.stall_llm_requests,
                        "prompt_token_limit": args.stall_prompt_tokens,
                        "no_progress_seconds": args.stall_no_progress_minutes * 60.0,
                        "stage_timeout_seconds": args.stall_stage_timeout_hours * 3600.0,
                    }
                    if args.stall_watchdog
                    else {"enabled": False}
                ),
            }
        )
        ledger["updated_at"] = utc_now()
        atomic_write_json(ledger_path, ledger)

        try:
            stall_watchdog = (
                StageStallWatchdog(
                    checker_repeat_limit=args.stall_checker_repeats,
                    llm_request_limit=args.stall_llm_requests,
                    prompt_token_limit=args.stall_prompt_tokens,
                    no_progress_seconds=args.stall_no_progress_minutes * 60.0,
                    stage_timeout_seconds=args.stall_stage_timeout_hours * 3600.0,
                )
                if args.stall_watchdog
                else None
            )
            return_code, timed_out, failure_reason = run_command(
                command,
                env,
                run_dir / "console.log",
                timeout_seconds,
                source_root,
                stall_watchdog=stall_watchdog,
            )
        except KeyboardInterrupt:
            attempt_ended_at = utc_now()
            attempt_record.update(
                {
                    "status": "interrupted",
                    "ended_at": attempt_ended_at,
                    "duration_seconds": round(time.monotonic() - attempt_started_monotonic, 3),
                }
            )
            record.update({"status": "interrupted", "ended_at": attempt_ended_at})
            ledger["updated_at"] = utc_now()
            atomic_write_json(ledger_path, ledger)
            print(f"\nInterrupted {run_id}; rerun the same command to resume it.")
            return 130

        completed = workspace_completed(workspace)
        status = "completed" if completed else ("timeout" if timed_out else "failed")
        attempt_ended_at = utc_now()
        attempt_record.update(
            {
                "status": status,
                "return_code": return_code,
                "failure_reason": failure_reason,
                "ended_at": attempt_ended_at,
                "duration_seconds": round(time.monotonic() - attempt_started_monotonic, 3),
            }
        )
        record.update(
            {
                "status": status,
                "return_code": return_code,
                "workspace_completed": completed,
                "failure_reason": failure_reason,
                "ended_at": attempt_ended_at,
            }
        )
        summary = None
        if completed:
            summary = finalize_completed_result(record, run_id, run_dir, results_csv)
        ledger["updated_at"] = utc_now()
        atomic_write_json(ledger_path, ledger)
        duration = f", duration={summary['formatted_duration']}" if summary else ""
        print(f"FINISH {run_id}: {status} (rc={return_code}{duration})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
