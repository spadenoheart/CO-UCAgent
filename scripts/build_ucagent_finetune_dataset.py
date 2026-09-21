#!/usr/bin/env python3
"""Build UCAgent fine-tuning samples from high-quality local trajectories.

The preferred source is exact llm_input_dump JSON files when they still exist.
Older runs usually only keep ucagent-msg.log / ucagent-log.log, so this script
falls back to reconstructing visible chat context from ucagent-msg.log.

Output format is chat-SFT JSONL:

    {"id": "...", "messages": [{"role": "user", ...}, ..., {"role": "assistant", ...}], "metadata": {...}}

The last message is the assistant target to learn from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


SECTION_RE = re.compile(
    r"^=+ (?P<kind>Human|Ai|Tool) Message =+\n"
    r"(?P<body>.*?)(?=^=+ (?:Human|Ai|Tool) Message =+\n|\Z)",
    re.M | re.S,
)
LLM_REQUEST_RE = re.compile(r"\[data_collection\]\[llm_request\]\s+(?P<body>.*)$")
LLM_DUMP_RE = re.compile(
    r"\[llm_input_dump\]\s+request=(?P<request>\d+)\s+stage=(?P<stage>\S+).*?path=(?P<path>\S+)"
)
FIELD_RE = re.compile(r"(\w+)=('(?:[^']|\\')*'|\S+)")
STAGE_INDEX_RE = re.compile(r"(?:current_stage:\s*\n(?:.*\n){0,8}?\s*index:\s*|Current stage index is now\s+)(?P<stage>\d+)", re.I)
STAGE_TITLE_RE = re.compile(r"title:\s*(?P<title>[^\n]+)")
STAGE_COMPLETE_RE = re.compile(r"Stage\s+(?P<stage>\d+)\s+completed successfully", re.I)

NUMERIC_FIELDS = {
    "latency_ms",
    "ttft_ms",
    "prompt_tokens",
    "completion_tokens",
    "prefill_tps_est",
    "decode_tps_est",
}

RUN_SUCCESS_PATTERNS = [
    "All stages completed successfully",
    "All stages completed. Exiting the mission",
    "All stages completed",
]
SEVERE_FAILURE_PATTERNS = [
    "Connection error",
    "model requires more system memory",
    "Empty reply from server",
    "Error executing systematic strategy",
    "Break at loop",
    "Traceback",
]
POSITIVE_PATTERNS = [
    "check_pass: true",
    "checks passed successfully",
    "completed successfully",
    "Test case validation successful",
    "Your test implementation successfully validates",
    "All test cases have been implemented",
    "run_test_success: true",
    '"run_test_success": true',
    "Write ",
    "complete.",
]
STRONG_POSITIVE_PATTERNS = [
    "check_pass: true",
    "checks passed successfully",
    "completed successfully",
    "Test case validation successful",
    "Your test implementation successfully validates",
    "All test cases have been implemented",
    "run_test_success: true",
    '"run_test_success": true',
]
NEGATIVE_PATTERNS = [
    "check_pass: false",
    "run_test_success: false",
    '"run_test_success": false',
    "does not exist in workspace",
    "[ERROR]",
    "Traceback",
    "Connection error",
    "model requires more system memory",
]
VALUABLE_TOOL_NAMES = {
    "EditFileDialogs",
    "ReplaceStringInFile",
    "RunTestCases",
    "Check",
    "Complete",
    "ReadTextFile",
    "PathList",
    "SearchText",
    "FindFiles",
}
LOW_VALUE_ONLY_TOOLS = {
    "CurrentTips",
    "Status",
    "Detail",
    "GetToDoSummary",
}


PUBLIC_DATASET_CATALOG: list[dict[str, str]] = [
    {
        "bucket": "20% RTL/debug/bug-fix",
        "name": "architect-ubc-capstone/rtl-augmented-v3",
        "url": "https://huggingface.co/datasets/architect-ubc-capstone/rtl-augmented-v3",
        "license_note": "MIT shown on Hugging Face dataset card.",
        "recommended_use": "RTL bug-fix / debug SFT and preference data; sample manually before mixing.",
    },
    {
        "bucket": "20% RTL/debug/bug-fix",
        "name": "Zeeh-Lin/RtlDebugBenchDataset",
        "url": "https://huggingface.co/datasets/Zeeh-Lin/RtlDebugBenchDataset",
        "license_note": "Check current dataset card before redistribution.",
        "recommended_use": "RTL debugging prompts and bug-localization style examples.",
    },
    {
        "bucket": "20% RTL/debug/bug-fix",
        "name": "ahmedallam/RTL-Repo",
        "url": "https://huggingface.co/datasets/ahmedallam/RTL-Repo",
        "license_note": "Check current dataset card before redistribution.",
        "recommended_use": "Large-codebase RTL completion/context-understanding examples.",
    },
    {
        "bucket": "10% RTLLM/VerilogEval/spec-to-RTL",
        "name": "NVlabs/verilog-eval",
        "url": "https://github.com/NVlabs/verilog-eval",
        "license_note": "Check repository license and any mirrored dataset license.",
        "recommended_use": "Spec-to-RTL and code-completion evaluation; small, clean benchmark-like data.",
    },
    {
        "bucket": "10% RTLLM/VerilogEval/spec-to-RTL",
        "name": "RTLLM / RTLLM 2.0",
        "url": "https://github.com/hkust-zhiyao/RTLLM",
        "license_note": "Check repository license before training redistribution.",
        "recommended_use": "Natural-language spec + testbench + correct RTL examples.",
    },
    {
        "bucket": "10% RTLLM/VerilogEval/spec-to-RTL",
        "name": "OpenLLM-RTL paper data references",
        "url": "https://huggingface.co/papers/2503.15112",
        "license_note": "Use linked datasets according to their own licenses.",
        "recommended_use": "Survey anchor for RTLLM 2.0, AssertEval, and RTLCoder-Data style mixtures.",
    },
    {
        "bucket": "10% CircuitNet/Verilog raw domain data",
        "name": "SKLP-EDA-LAB/CircuitNet3.0",
        "url": "https://huggingface.co/datasets/SKLP-EDA-LAB/CircuitNet3.0",
        "license_note": "Apache-2.0 shown on Hugging Face dataset card.",
        "recommended_use": "Domain continued pretraining or retrieval over RTL/netlist/timing/power metadata; not primary Agent SFT.",
    },
    {
        "bucket": "10% CircuitNet/Verilog raw domain data",
        "name": "shailja/Verilog_GitHub",
        "url": "https://huggingface.co/datasets/shailja/Verilog_GitHub",
        "license_note": "MIT shown on Hugging Face dataset card.",
        "recommended_use": "Raw Verilog style and syntax continued pretraining; filter aggressively.",
    },
    {
        "bucket": "10% CircuitNet/Verilog raw domain data",
        "name": "bnadimi/PyraNet-Verilog",
        "url": "https://huggingface.co/datasets/bnadimi/PyraNet-Verilog",
        "license_note": "CC-BY-NC-SA-4.0 shown on Hugging Face dataset card; non-commercial constraints matter.",
        "recommended_use": "Hierarchical Verilog corpus for non-commercial domain adaptation only.",
    },
]


@dataclass
class Section:
    kind: str
    body: str
    tool_name: str = ""


@dataclass
class RequestRecord:
    ordinal: int
    line_number: int
    raw: str
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunFiles:
    run_name: str
    run_dir: Path
    dut: str
    log_files: list[Path]
    msg_files: list[Path]


@dataclass
class LightCandidate:
    run_name: str
    dut: str
    section_index: int
    assistant_ordinal: int
    target: str
    score: int
    stage_key: str
    stage: str
    stage_title: str
    reasons: list[str]
    reject_reasons: list[str]
    tool_names: list[str]
    request_fields: dict[str, Any]
    request_ordinal: int | None
    run_completed: bool
    run_has_severe_failure: bool


@dataclass
class Candidate:
    sample: dict[str, Any]
    score: int
    stage_key: str
    run_name: str


def parse_value(key: str, value: str) -> Any:
    value = value.strip("'")
    if key not in NUMERIC_FIELDS:
        return value
    if value == "NA":
        return None
    try:
        return float(value)
    except ValueError:
        return value


def rough_token_count(text: str) -> int:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    non_cjk = len(text) - cjk
    return int(cjk + non_cjk / 4)


def messages_rough_tokens(messages: list[dict[str, str]]) -> int:
    return sum(rough_token_count(str(msg.get("content", ""))) + 4 for msg in messages)


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def read_many(paths: Iterable[Path]) -> str:
    chunks = []
    for path in paths:
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(chunks)


def parse_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    for match in SECTION_RE.finditer(normalize_text(text)):
        kind = match.group("kind")
        body = match.group("body").strip("\n")
        tool_name = ""
        if kind == "Tool":
            name_match = re.match(r"Name:\s*(?P<name>[^\n]+)\n\n(?P<body>.*)", body, re.S)
            if name_match:
                tool_name = name_match.group("name").strip()
                body = name_match.group("body").strip("\n")
        sections.append(Section(kind=kind, body=body, tool_name=tool_name))
    return sections


def parse_requests(log_text: str) -> list[RequestRecord]:
    records: list[RequestRecord] = []
    ordinal = 0
    for line_number, line in enumerate(log_text.splitlines(), 1):
        match = LLM_REQUEST_RE.search(line)
        if not match:
            continue
        fields = {}
        for key, raw_value in FIELD_RE.findall(match.group("body")):
            fields[key] = parse_value(key, raw_value)
        if fields.get("role") != "main":
            continue
        ordinal += 1
        records.append(RequestRecord(ordinal=ordinal, line_number=line_number, raw=line, fields=fields))
    return records


def parse_dump_paths(log_text: str) -> dict[int, Path]:
    dump_paths: dict[int, Path] = {}
    for line in log_text.splitlines():
        match = LLM_DUMP_RE.search(line)
        if not match:
            continue
        request = int(match.group("request"))
        path = Path(match.group("path"))
        if path.exists():
            dump_paths[request] = path
    return dump_paths


def discover_runs(log_roots: list[Path]) -> list[RunFiles]:
    runs: list[RunFiles] = []
    for root in log_roots:
        if not root.exists():
            continue
        for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            log_files = sorted(run_dir.glob("ucagent-log*.log*"))
            msg_files = sorted(run_dir.glob("ucagent-msg*.log*"))
            if not log_files or not msg_files:
                continue
            dut = run_dir.name.split("+", 1)[0]
            runs.append(
                RunFiles(
                    run_name=run_dir.name,
                    run_dir=run_dir,
                    dut=dut,
                    log_files=log_files,
                    msg_files=msg_files,
                )
            )
    return runs


def section_to_message(section: Section) -> dict[str, str] | None:
    if section.kind == "Ai":
        role = "assistant"
        content = section.body
    elif section.kind == "Tool":
        role = "user"
        content = "TOOL MESSAGE"
        if section.tool_name:
            content += f" ({section.tool_name})"
        content += ":\n" + section.body
    else:
        role = "user"
        content = section.body
    if not content.strip():
        return None
    return {"role": role, "content": content.strip()}


def sections_to_messages(sections: list[Section]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for section in sections:
        msg = section_to_message(section)
        if not msg:
            continue
        if messages and messages[-1]["role"] == msg["role"]:
            messages[-1]["content"] += "\n\n" + msg["content"]
        else:
            messages.append(msg)
    return messages


def trim_prompt_messages(messages: list[dict[str, str]], max_context_tokens: int) -> tuple[list[dict[str, str]], bool]:
    if max_context_tokens <= 0:
        return messages, False
    kept: list[dict[str, str]] = []
    total = 0
    for msg in reversed(messages):
        cost = rough_token_count(str(msg.get("content", ""))) + 4
        if not kept and cost > max_context_tokens:
            trimmed = dict(msg)
            trimmed["content"] = trim_text_to_rough_tokens(
                str(msg.get("content", "")),
                max(128, max_context_tokens - 4),
            )
            return [trimmed], True
        if kept and total + cost > max_context_tokens:
            break
        kept.append(msg)
        total += cost
    kept.reverse()
    return kept, len(kept) < len(messages)


def trim_text_to_rough_tokens(text: str, max_tokens: int) -> str:
    if rough_token_count(text) <= max_tokens:
        return text
    low, high = 0, len(text)
    while low < high:
        mid = (low + high) // 2
        if rough_token_count(text[mid:]) <= max_tokens:
            high = mid
        else:
            low = mid + 1
    trimmed = text[low:]
    newline = trimmed.find("\n")
    if 0 <= newline < 200:
        trimmed = trimmed[newline + 1 :]
    return "[TRUNCATED_PREFIX: context tail kept by dataset builder]\n" + trimmed


def extract_tool_names(ai_body: str) -> list[str]:
    if "Tool Calls:" not in ai_body:
        return []
    names = []
    in_tool_block = False
    for line in ai_body.splitlines():
        if line.strip() == "Tool Calls:":
            in_tool_block = True
            continue
        if not in_tool_block:
            continue
        match = re.match(r"\s{2}([A-Za-z_][A-Za-z0-9_]*)\s+\(call[_-][^)]+\)\s*$", line)
        if match:
            names.append(match.group(1))
    return names


def infer_stage_from_text(text: str) -> tuple[str, str]:
    stage = ""
    for match in STAGE_INDEX_RE.finditer(text):
        stage = match.group("stage")
    title = ""
    tail = text[-8000:]
    for match in STAGE_TITLE_RE.finditer(tail):
        title = match.group("title").strip().strip("'\"")
    return stage, title


def has_any(text: str, patterns: Iterable[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def score_candidate(
    *,
    run_completed: bool,
    run_has_severe_failure: bool,
    target: str,
    next_text: str,
    context_text: str,
    tool_names: list[str],
) -> tuple[int, list[str], list[str]]:
    score = 0
    reasons: list[str] = []
    reject: list[str] = []

    if run_completed:
        score += 2
        reasons.append("run_completed")
    elif run_has_severe_failure:
        score -= 2
        reasons.append("run_has_severe_failure")

    for pattern in STRONG_POSITIVE_PATTERNS:
        if pattern in next_text:
            score += 3
            reasons.append(f"positive:{pattern}")
            break
    else:
        for pattern in POSITIVE_PATTERNS:
            if pattern in next_text:
                score += 1
                reasons.append(f"weak_positive:{pattern}")
                break

    if "run_test_success: true" in next_text or '"run_test_success": true' in next_text:
        score += 2
        reasons.append("test_run_success")
    if "check_pass: true" in next_text:
        score += 2
        reasons.append("checker_pass")
    if "completed successfully" in next_text:
        score += 2
        reasons.append("stage_progress")

    if any(tool in VALUABLE_TOOL_NAMES for tool in tool_names):
        score += 1
        reasons.append("valuable_tool_call")
    if tool_names and all(tool in LOW_VALUE_ONLY_TOOLS for tool in tool_names):
        score -= 2
        reasons.append("low_value_status_only")
    if any(tool in {"EditFileDialogs", "ReplaceStringInFile"} for tool in tool_names):
        score += 2
        reasons.append("file_edit_action")
    if "RunTestCases" in tool_names:
        score += 1
        reasons.append("test_execution_action")

    if has_any(context_text[-20000:], ["check_pass: false", "failed_cases", "failed_checkpoints", "[ERROR]"]):
        score += 1
        reasons.append("uses_failure_context")

    for pattern in NEGATIVE_PATTERNS:
        if pattern in next_text:
            score -= 4
            reject.append(f"negative_after:{pattern}")
            break
    for pattern in SEVERE_FAILURE_PATTERNS:
        if pattern in target or pattern in next_text:
            score -= 6
            reject.append(f"severe_failure:{pattern}")
            break

    if len(target.strip()) < 40:
        score -= 3
        reject.append("target_too_short")
    if "Tool Calls:" not in target and len(target.strip()) < 120:
        score -= 1
        reject.append("short_no_tool_call")

    return score, reasons, reject


def load_exact_prompt_messages(path: Path) -> list[dict[str, str]] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        return None
    messages: list[dict[str, str]] = []
    for raw in raw_messages:
        if not isinstance(raw, dict):
            continue
        msg_type = str(raw.get("type") or raw.get("class") or "").lower()
        if "ai" in msg_type:
            role = "assistant"
        elif "tool" in msg_type:
            role = "user"
        else:
            role = "user"
        content = str(raw.get("content") or "")
        if not content and raw.get("tool_calls"):
            content = "Tool Calls:\n" + json.dumps(raw.get("tool_calls"), ensure_ascii=False, indent=2)
        if role == "user" and "tool" in msg_type:
            name = raw.get("name") or raw.get("tool_call_id") or "tool"
            content = f"TOOL MESSAGE ({name}):\n{content}"
        if content.strip():
            messages.append({"role": role, "content": content.strip()})
    return messages or None


def stable_split(sample_id: str, val_ratio: float, test_ratio: float) -> str:
    bucket = int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + val_ratio:
        return "val"
    return "train"


def scan_candidates_for_run(
    run: RunFiles,
    *,
    min_quality_score: int,
    include_incomplete_runs: bool,
) -> tuple[list[LightCandidate], dict[str, Any]]:
    log_text = read_many(run.log_files)
    msg_text = read_many(run.msg_files)
    sections = parse_sections(msg_text)
    requests = parse_requests(log_text)
    dump_paths = parse_dump_paths(log_text)

    run_completed = has_any(log_text, RUN_SUCCESS_PATTERNS)
    run_has_severe_failure = has_any(log_text, SEVERE_FAILURE_PATTERNS)
    completed_stages = sorted({int(m.group("stage")) for m in STAGE_COMPLETE_RE.finditer(log_text)})

    info = {
        "run": run.run_name,
        "dut": run.dut,
        "run_dir": str(run.run_dir),
        "sections": len(sections),
        "assistant_sections": sum(1 for s in sections if s.kind == "Ai"),
        "llm_requests": len(requests),
        "completed": run_completed,
        "has_severe_failure": run_has_severe_failure,
        "completed_stages": completed_stages,
        "exact_dump_paths_available": len(dump_paths),
        "candidates": 0,
        "selected": 0,
        "rejected": 0,
    }

    if not sections:
        return [], info
    if not run_completed and not include_incomplete_runs:
        return [], info

    candidates: list[LightCandidate] = []
    assistant_ordinal = 0
    recent_bodies: deque[str] = deque(maxlen=24)

    def append_section_to_history(item: Section) -> None:
        recent_bodies.append(item.body)

    for idx, section in enumerate(sections):
        if section.kind != "Ai":
            append_section_to_history(section)
            continue
        assistant_ordinal += 1
        target = section.body.strip()
        tool_names = extract_tool_names(target)
        next_sections = sections[idx + 1 : idx + 6]
        prompt_text_tail = "\n\n".join(list(recent_bodies)[-12:])
        next_text = "\n\n".join(s.body for s in next_sections)
        context_text = "\n\n".join(recent_bodies)
        request = requests[assistant_ordinal - 1] if assistant_ordinal <= len(requests) else None

        score, reasons, reject_reasons = score_candidate(
            run_completed=run_completed,
            run_has_severe_failure=run_has_severe_failure,
            target=target,
            next_text=next_text,
            context_text=context_text,
            tool_names=tool_names,
        )
        if score < min_quality_score:
            info["rejected"] += 1
            append_section_to_history(section)
            continue

        inferred_stage, inferred_title = infer_stage_from_text(prompt_text_tail)
        stage = ""
        if request is not None:
            stage = str(request.fields.get("stage") or "")
        stage = stage or inferred_stage
        stage_title = inferred_title

        stage_key = stage or "NA"
        candidates.append(
            LightCandidate(
                run_name=run.run_name,
                dut=run.dut,
                section_index=idx,
                assistant_ordinal=assistant_ordinal,
                target=target,
                score=score,
                stage_key=stage_key,
                stage=stage,
                stage_title=stage_title,
                reasons=reasons,
                reject_reasons=reject_reasons,
                tool_names=tool_names,
                request_fields=request.fields if request else {},
                request_ordinal=request.ordinal if request else None,
                run_completed=run_completed,
                run_has_severe_failure=run_has_severe_failure,
            )
        )
        info["candidates"] += 1

        append_section_to_history(section)

    return candidates, info


def select_light_candidates(
    candidates: list[LightCandidate],
    *,
    max_samples: int,
    max_samples_per_run: int,
    max_samples_per_stage_per_run: int,
) -> list[LightCandidate]:
    by_run: dict[str, list[LightCandidate]] = defaultdict(list)
    for cand in candidates:
        by_run[cand.run_name].append(cand)

    selected: list[LightCandidate] = []
    for run_name, run_candidates in by_run.items():
        run_candidates.sort(
            key=lambda cand: (
                -cand.score,
                cand.stage == "",
                cand.assistant_ordinal,
            )
        )
        per_stage: Counter[str] = Counter()
        run_selected = 0
        for cand in run_candidates:
            if run_selected >= max_samples_per_run:
                break
            if per_stage[cand.stage_key] >= max_samples_per_stage_per_run:
                continue
            selected.append(cand)
            per_stage[cand.stage_key] += 1
            run_selected += 1

    selected.sort(
        key=lambda cand: (
            cand.dut,
            cand.run_name,
            cand.assistant_ordinal,
        )
    )
    if max_samples > 0:
        selected = sorted(selected, key=lambda cand: -cand.score)[:max_samples]
        selected.sort(
            key=lambda cand: (
                cand.dut,
                cand.run_name,
                cand.assistant_ordinal,
            )
        )
    return selected


def materialize_candidates_for_run(
    run: RunFiles,
    selected: list[LightCandidate],
    *,
    max_context_tokens: int,
) -> list[Candidate]:
    if not selected:
        return []
    log_text = read_many(run.log_files)
    msg_text = read_many(run.msg_files)
    sections = parse_sections(msg_text)
    dump_paths = parse_dump_paths(log_text)
    by_section = {cand.section_index: cand for cand in selected}
    messages_so_far: list[dict[str, str]] = []
    materialized: list[Candidate] = []

    def append_section_to_history(item: Section) -> None:
        msg = section_to_message(item)
        if not msg:
            return
        if messages_so_far and messages_so_far[-1]["role"] == msg["role"]:
            messages_so_far[-1]["content"] += "\n\n" + msg["content"]
        else:
            messages_so_far.append(msg)

    for idx, section in enumerate(sections):
        if section.kind == "Ai" and idx in by_section:
            cand = by_section[idx]
            exact_input_available = False
            exact_dump_path = ""
            exact_prompt = None
            if cand.request_ordinal is not None and cand.request_ordinal in dump_paths:
                exact_prompt = load_exact_prompt_messages(dump_paths[cand.request_ordinal])
                if exact_prompt:
                    exact_input_available = True
                    exact_dump_path = str(dump_paths[cand.request_ordinal])

            prompt_messages = exact_prompt if exact_prompt is not None else list(messages_so_far)
            prompt_messages, truncated = trim_prompt_messages(prompt_messages, max_context_tokens)
            messages = list(prompt_messages)
            messages.append({"role": "assistant", "content": cand.target})

            sample_id_base = f"{run.run_name}:ai:{cand.assistant_ordinal}:stage:{cand.stage}:score:{cand.score}"
            sample_id = hashlib.sha1(sample_id_base.encode("utf-8")).hexdigest()[:12]
            sample_id = f"{run.dut}_{run.run_name.replace('+', '_')}_ai{cand.assistant_ordinal:06d}_{sample_id}"
            metadata = {
                "source": "ucagent_local_trajectory",
                "source_run": run.run_name,
                "source_run_dir": str(run.run_dir),
                "source_msg_files": [str(path) for path in run.msg_files],
                "source_log_files": [str(path) for path in run.log_files],
                "dut": run.dut,
                "assistant_ordinal": cand.assistant_ordinal,
                "stage": cand.stage,
                "stage_title": cand.stage_title,
                "quality_score": cand.score,
                "selection_reasons": cand.reasons,
                "tool_names": cand.tool_names,
                "run_completed": cand.run_completed,
                "run_has_severe_failure": cand.run_has_severe_failure,
                "exact_input_available": exact_input_available,
                "exact_dump_path": exact_dump_path,
                "context_truncated": truncated,
                "context_tokens_rough": messages_rough_tokens(prompt_messages),
                "target_tokens_rough": rough_token_count(cand.target),
                "request_record": cand.request_fields,
                "rejected_risks_seen_but_overridden_by_score": cand.reject_reasons,
            }
            sample = {"id": sample_id, "messages": messages, "metadata": metadata}
            materialized.append(
                Candidate(sample=sample, score=cand.score, stage_key=cand.stage_key, run_name=run.run_name)
            )
        append_section_to_history(section)
    return materialized


def materialize_selected_candidates(
    runs: list[RunFiles],
    selected: list[LightCandidate],
    *,
    max_context_tokens: int,
) -> list[dict[str, Any]]:
    run_by_name = {run.run_name: run for run in runs}
    by_run: dict[str, list[LightCandidate]] = defaultdict(list)
    for cand in selected:
        by_run[cand.run_name].append(cand)

    materialized: list[Candidate] = []
    for run_name, run_candidates in by_run.items():
        run = run_by_name.get(run_name)
        if run is None:
            continue
        run_candidates.sort(key=lambda cand: cand.section_index)
        materialized.extend(
            materialize_candidates_for_run(run, run_candidates, max_context_tokens=max_context_tokens)
        )

    materialized.sort(
        key=lambda cand: (
            cand.sample["metadata"].get("dut", ""),
            cand.run_name,
            cand.sample["metadata"].get("assistant_ordinal", 0),
        )
    )
    return [cand.sample for cand in materialized]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_public_dataset_mix(out_dir: Path) -> None:
    write_json(out_dir / "public_dataset_catalog.json", PUBLIC_DATASET_CATALOG)
    lines = [
        "# Public Dataset Mix for UCAgent Fine-Tuning",
        "",
        "Recommended first-pass mixture:",
        "",
        "- 60% local UCAgent high-quality trajectories from `trajectory_sft_train.jsonl`.",
        "- 20% RTL/debug/bug-fix public data.",
        "- 10% RTLLM/VerilogEval/spec-to-RTL data.",
        "- 10% CircuitNet/Verilog raw domain data. This bucket is optional for the first adapter run.",
        "",
        "Use public data as a supplement. The local trajectory set should dominate because it teaches the UCAgent workflow, checker interpretation, tool usage, bug-document schema, and verification-loop behavior.",
        "",
        "## Catalog",
        "",
    ]
    for item in PUBLIC_DATASET_CATALOG:
        lines.extend(
            [
                f"### {item['name']}",
                "",
                f"- bucket: {item['bucket']}",
                f"- url: {item['url']}",
                f"- license_note: {item['license_note']}",
                f"- recommended_use: {item['recommended_use']}",
                "",
            ]
        )
    (out_dir / "public_dataset_mix.md").write_text("\n".join(lines), encoding="utf-8")


def summarize(samples: list[dict[str, Any]], run_infos: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_dut = Counter(sample["metadata"].get("dut", "NA") for sample in samples)
    by_stage = Counter(str(sample["metadata"].get("stage") or "NA") for sample in samples)
    by_split = Counter(sample["metadata"].get("split", "NA") for sample in samples)
    quality_scores = [int(sample["metadata"].get("quality_score", 0)) for sample in samples]
    return {
        "schema_version": 1,
        "description": "High-quality UCAgent local trajectory SFT dataset.",
        "generation_args": vars(args),
        "selection_policy": {
            "primary_source": "completed UCAgent runs from log/ and log_pre/",
            "positive_signals": STRONG_POSITIVE_PATTERNS,
            "negative_signals": NEGATIVE_PATTERNS,
            "min_quality_score": args.min_quality_score,
            "max_context_tokens_rough": args.max_context_tokens,
            "exact_input_policy": "Use llm_input_dump JSON when path still exists; otherwise reconstruct from ucagent-msg.log and mark exact_input_available=false.",
        },
        "counts": {
            "samples": len(samples),
            "by_split": dict(sorted(by_split.items())),
            "by_dut": dict(sorted(by_dut.items())),
            "by_stage": dict(sorted(by_stage.items(), key=lambda kv: (kv[0] == "NA", kv[0]))),
            "runs_discovered": len(run_infos),
            "runs_completed": sum(1 for info in run_infos if info.get("completed")),
            "runs_selected": sum(1 for info in run_infos if info.get("selected", 0) > 0),
            "exact_input_available_samples": sum(1 for s in samples if s["metadata"].get("exact_input_available")),
            "context_truncated_samples": sum(1 for s in samples if s["metadata"].get("context_truncated")),
            "quality_score_min": min(quality_scores) if quality_scores else None,
            "quality_score_max": max(quality_scores) if quality_scores else None,
        },
        "run_infos": run_infos,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-roots", nargs="+", default=["log", "log_pre"], help="Log roots to scan.")
    parser.add_argument("--out-dir", default="benchmark/ucagent_finetune_dataset", help="Output directory.")
    parser.add_argument("--max-context-tokens", type=int, default=64000, help="Rough token budget for prompt context.")
    parser.add_argument("--min-quality-score", type=int, default=4, help="Minimum quality score for a sample.")
    parser.add_argument("--max-samples", type=int, default=3000, help="Global sample cap; <=0 means no cap.")
    parser.add_argument("--max-samples-per-run", type=int, default=80, help="Per-run sample cap.")
    parser.add_argument("--max-samples-per-stage-per-run", type=int, default=10, help="Per-stage per-run sample cap.")
    parser.add_argument("--include-incomplete-runs", action="store_true", help="Also sample high-scoring steps from incomplete runs.")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument(
        "--holdout-dut",
        action="append",
        default=[],
        help="Put all samples from this DUT into test split. Can be repeated.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs([Path(root) for root in args.log_roots])

    all_candidates: list[LightCandidate] = []
    run_infos: list[dict[str, Any]] = []
    for run in runs:
        candidates, info = scan_candidates_for_run(
            run,
            min_quality_score=args.min_quality_score,
            include_incomplete_runs=args.include_incomplete_runs,
        )
        all_candidates.extend(candidates)
        run_infos.append(info)

    selected_light = select_light_candidates(
        all_candidates,
        max_samples=args.max_samples,
        max_samples_per_run=args.max_samples_per_run,
        max_samples_per_stage_per_run=args.max_samples_per_stage_per_run,
    )
    selected_by_run = Counter(cand.run_name for cand in selected_light)
    for info in run_infos:
        info["selected"] = selected_by_run.get(str(info.get("run")), 0)

    samples = materialize_selected_candidates(
        runs,
        selected_light,
        max_context_tokens=args.max_context_tokens,
    )

    holdout_duts = set(args.holdout_dut or [])
    for sample in samples:
        dut = str(sample["metadata"].get("dut") or "")
        if dut in holdout_duts:
            split = "test"
        else:
            split = stable_split(sample["id"], args.val_ratio, args.test_ratio)
        sample["metadata"]["split"] = split

    train = [sample for sample in samples if sample["metadata"]["split"] == "train"]
    val = [sample for sample in samples if sample["metadata"]["split"] == "val"]
    test = [sample for sample in samples if sample["metadata"]["split"] == "test"]

    write_jsonl(out_dir / "trajectory_sft.jsonl", samples)
    write_jsonl(out_dir / "trajectory_sft_train.jsonl", train)
    write_jsonl(out_dir / "trajectory_sft_val.jsonl", val)
    write_jsonl(out_dir / "trajectory_sft_test.jsonl", test)
    write_json(out_dir / "manifest.json", summarize(samples, run_infos, args))
    write_public_dataset_mix(out_dir)

    print(f"discovered_runs={len(runs)}")
    print(f"candidate_samples={len(all_candidates)}")
    print(f"selected_light_candidates={len(selected_light)}")
    print(f"selected_samples={len(samples)}")
    print(f"train={len(train)} val={len(val)} test={len(test)}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
