#!/usr/bin/env python3
"""Freeze a reproducible, secret-redacted UCAgent experiment snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = (
    "README.md",
    "Makefile",
    "config.yaml",
    "ucagent/setting.yaml",
    "ucagent/abackend/langchain/agent.py",
    "ucagent/abackend/langchain/message/conversation.py",
    "ucagent/checkers/base.py",
    "ucagent/util/test_result.py",
    "ucagent/util/config.py",
    "ucagent/stage/vmanager.py",
    "ucagent/verify_agent.py",
    "ucagent/tools/fileops.py",
    "ucagent/tools/testops.py",
    "ucagent/checkers/unity_test.py",
    "ucagent/memory/context_reuse.py",
    "scripts/build_trace_tree.py",
    "scripts/build_context_reuse_pack.py",
    "scripts/curate_context_reuse_pack.py",
    "scripts/evaluate_episode_credit_annotations.py",
    "scripts/update_context_reuse_pack.py",
    "scripts/analyze_context_reuse_decisions.py",
    "scripts/replay_context_reuse_gate.py",
    "scripts/freeze_experiment_baseline.py",
    "tests/test_test_result_classification.py",
    "tests/test_episode_credit.py",
    "tests/test_context_reuse_curation.py",
    "tests/test_experiment_baseline.py",
    "tests/test_message_lifecycle.py",
    "benchmark/ucagent_context_reuse/context_reuse_v0.json",
    "benchmark/ucagent_context_reuse/context_reuse_v1.json",
    "benchmark/ucagent_context_reuse/context_reuse_credit_v2_rule_v2_curated_preview.json",
    "benchmark/ucagent_context_reuse/context_reuse_credit_v2_rule_v2_curated_preview.md",
    "benchmark/ucagent_context_reuse/episode_credit_v2_research_protocol.md",
    "benchmark/ucagent_context_reuse/episode_credit_v2_annotation_metrics.json",
    "benchmark/ucagent_context_reuse/episode_credit_v2_holdout_adder_20260716_metrics.json",
    "benchmark/ucagent_context_reuse/Adder_20260717_v21_gate_replay.json",
    "benchmark/ucagent_context_reuse/Adder_20260717_v21_gate_replay.md",
)
YAML_KEY_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_.-]+)\s*:\s*.*$")
SENSITIVE_KEYS = {
    "api_key",
    "openai_api_key",
    "secret",
    "secret_key",
    "token",
}
CREDENTIAL_PATTERNS = (
    re.compile(r"\bhf_[A-Za-z0-9]+\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]+\b"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fin:
        for block in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    return result.stdout.strip()


def redact_setting(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        comment_prefix = ""
        candidate = line
        if stripped.startswith("#"):
            marker_index = line.index("#")
            comment_prefix = line[:marker_index + 1]
            candidate = line[marker_index + 1:]
        match = YAML_KEY_RE.match(candidate)
        if match:
            key = match.group("key").lower()
            sensitive = (
                key in SENSITIVE_KEYS
                or key.endswith("_api_key")
                or key.endswith("_secret_key")
                or key.endswith("_access_token")
            )
            if sensitive:
                line = comment_prefix + match.group("indent") + match.group("key") + ': "***REDACTED***"'
        for pattern in CREDENTIAL_PATTERNS:
            line = pattern.sub("***REDACTED***", line)
        lines.append(line)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def freeze(output_dir: Path, files: tuple[str, ...]) -> dict:
    snapshot_dir = output_dir / "snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for raw_path in files:
        source = REPO_ROOT / raw_path
        if not source.is_file():
            raise FileNotFoundError(f"baseline source does not exist: {source}")
        target = snapshot_dir / raw_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if raw_path == "ucagent/setting.yaml":
            target.write_text(redact_setting(source.read_text(encoding="utf-8")), encoding="utf-8")
            redacted = True
        else:
            shutil.copy2(source, target)
            redacted = False
        records.append({
            "path": raw_path,
            "source_sha256": sha256(source),
            "snapshot_sha256": sha256(target),
            "bytes": source.stat().st_size,
            "redacted_snapshot": redacted,
        })

    manifest = {
        "schema_version": 1,
        "experiment_baseline": output_dir.name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "git_head": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "worktree_dirty": bool(git_output("status", "--short")),
        "python_version": sys.version,
        "files": records,
        "comparison": {
            "shared_code_snapshot": "snapshot/",
            "B0_no_context_reuse": {
                "args": "--override context_upgrade.enable_context_reuse=false",
            },
            "B1_v1_retrieval": {
                "pack": "benchmark/ucagent_context_reuse/context_reuse_v1.json",
                "min_quality_score": 0.68,
                "environment": {
                    "UCAGENT_CONTEXT_REUSE_PACK": "benchmark/ucagent_context_reuse/context_reuse_v1.json",
                    "UCAGENT_CONTEXT_REUSE_MIN_QUALITY": "0.68",
                },
            },
            "B2_v21_utility_gate": {
                "pack": "benchmark/ucagent_context_reuse/context_reuse_credit_v2_rule_v2_curated_preview.json",
                "environment": {
                    "UCAGENT_CONTEXT_REUSE_PACK": "benchmark/ucagent_context_reuse/context_reuse_credit_v2_rule_v2_curated_preview.json",
                    "UCAGENT_CONTEXT_REUSE_MIN_QUALITY": "0.68",
                    "UCAGENT_CONTEXT_REUSE_UTILITY_GATE": "true",
                    "UCAGENT_CONTEXT_REUSE_MIN_UTILITY": "0.05",
                    "UCAGENT_CONTEXT_REUSE_STRICT_ACTION_MATCH": "false",
                    "UCAGENT_CONTEXT_REUSE_EXACT_PATTERN_GROUPS": "false",
                    "UCAGENT_CONTEXT_REUSE_GENERIC_SIGNATURE": "false",
                    "UCAGENT_CONTEXT_REUSE_SKIP_NON_REUSABLE": "false",
                    "UCAGENT_CONTEXT_REUSE_STRICT_HINT_MATCH": "false",
                    "UCAGENT_CONTEXT_REUSE_ALLOW_HINTS_ONLY": "true",
                    "UCAGENT_CONTEXT_REUSE_DESTRUCTIVE_SUPPORT_GATE": "false",
                },
            },
            "B3_v22_semantic_risk_gate": {
                "pack": "benchmark/ucagent_context_reuse/context_reuse_credit_v2_rule_v2_curated_preview.json",
                "environment": {
                    "UCAGENT_CONTEXT_REUSE_PACK": "benchmark/ucagent_context_reuse/context_reuse_credit_v2_rule_v2_curated_preview.json",
                    "UCAGENT_CONTEXT_REUSE_MIN_QUALITY": "0.68",
                    "UCAGENT_CONTEXT_REUSE_UTILITY_GATE": "true",
                    "UCAGENT_CONTEXT_REUSE_MIN_UTILITY": "0.05",
                },
            },
        },
        "notes": [
            "The repository worktree was dirty; this snapshot, not git HEAD alone, defines the baseline.",
            "The setting snapshot is redacted. Its source_sha256 identifies the exact local configuration.",
            "B0/B1/B2/B3 must use the same model, hardware, prompt variant, DUT input, seed, and timeout.",
            "B2 reproduces the observed v2.1 switches; B3 enables the v2.2 semantic/risk defaults.",
            "Historical gate replay is retrieval evidence, not a causal end-to-end performance result.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    readme = f"""# Experiment Baseline: {output_dir.name}

This directory freezes the post-upstream-sync code and configuration used for
the B0-B3 context-reuse comparison. The worktree was dirty, so the files under
`snapshot/` rather than Git HEAD alone are the source of truth.

## B0: no context reuse

```bash
make test_Adder ARGS='--override context_upgrade.enable_context_reuse=false'
```

## B1: v1 retrieval

```bash
UCAGENT_CONTEXT_REUSE_PACK=benchmark/ucagent_context_reuse/context_reuse_v1.json \\
UCAGENT_CONTEXT_REUSE_MIN_QUALITY=0.68 \\
make test_Adder
```

## B2: v2.1 utility gate

Use the `B2_v21_utility_gate.environment` mapping in `manifest.json`. This arm
uses the role-typed pack and utility gate, while disabling the v2.2 semantic
and destructive-action hard gates to reproduce the observed v2.1 behavior.

## B3: v2.2 semantic/risk gate

```bash
UCAGENT_CONTEXT_REUSE_PACK=benchmark/ucagent_context_reuse/context_reuse_credit_v2_rule_v2_curated_preview.json \\
UCAGENT_CONTEXT_REUSE_MIN_QUALITY=0.68 \\
UCAGENT_CONTEXT_REUSE_UTILITY_GATE=true \\
UCAGENT_CONTEXT_REUSE_MIN_UTILITY=0.05 \\
make test_Adder
```

Keep the model, hardware, prompt variant, DUT input, seed, timeout, and all
non-context-reuse settings identical between paired runs. Record unsuccessful
and interrupted runs rather than dropping them. The historical gate replay is
retrieval evidence only; it is not a causal end-to-end performance result.

Before a full-DUT run, execute the parser, curation, and retrieval unit tests
from this snapshot. The role attribution still requires holdout validation;
the frozen v2.2 arm is a research candidate, not a proven improvement.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="benchmark/ucagent_experiments/20260721_upstream_sync_v22_candidate",
    )
    parser.add_argument("--file", action="append", dest="files")
    args = parser.parse_args()
    files = tuple(args.files) if args.files else DEFAULT_FILES
    output_dir = REPO_ROOT / args.output_dir
    manifest = freeze(output_dir, files)
    print(json.dumps({
        "output_dir": str(output_dir),
        "files": len(manifest["files"]),
        "git_head": manifest["git_head"],
        "worktree_dirty": manifest["worktree_dirty"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
