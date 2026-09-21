#!/usr/bin/env python3
"""Extract prompt-like benchmark cases from UCAgent logs.

The current UCAgent logs do not persist the exact LangGraph pre-model
`llm_input_messages` object. This script reconstructs the conversation
transcript visible in `ucagent-msg.log` before selected LLM responses, then
also builds a token-budget-matched tail prompt using the per-request
`prompt_tokens` value from `ucagent-log.log`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SECTION_RE = re.compile(
    r"^=+ (?P<kind>Human|Ai|Tool) Message =+\n"
    r"(?P<body>.*?)(?=^=+ (?:Human|Ai|Tool) Message =+\n|\Z)",
    re.M | re.S,
)
REQUEST_RE = re.compile(r"\[data_collection\]\[llm_request\]\s+(?P<body>.*)$")
FIELD_RE = re.compile(r"(\w+)=('(?:[^']|\\')*'|\S+)")
NUMERIC_FIELDS = {
    "latency_ms",
    "ttft_ms",
    "prompt_tokens",
    "completion_tokens",
    "prefill_tps_est",
    "decode_tps_est",
}


DEFAULT_CASES = [
    {
        "case_id": "success_20260611_req001_stage0_843tok",
        "run": "Adder+20260611_130606",
        "request_index": 1,
        "note": "Successful baseline, first stage, tiny initial context.",
    },
    {
        "case_id": "success_20260611_req016_stage1_9857tok",
        "run": "Adder+20260611_130606",
        "request_index": 16,
        "note": "Successful baseline, about 10K prompt tokens.",
    },
    {
        "case_id": "success_20260611_req133_stage19_30137tok",
        "run": "Adder+20260611_130606",
        "request_index": 133,
        "note": "Successful baseline, about 30K prompt tokens.",
    },
    {
        "case_id": "success_20260611_req074_stage11_50075tok",
        "run": "Adder+20260611_130606",
        "request_index": 74,
        "note": "Successful baseline, about 50K prompt tokens.",
    },
    {
        "case_id": "success_20260611_req304_stage26_49183tok",
        "run": "Adder+20260611_130606",
        "request_index": 304,
        "note": "Successful baseline, late final context.",
    },
    {
        "case_id": "slow_20260612_req374_stage0_843tok",
        "run": "Adder+20260612_103636",
        "request_index": 374,
        "note": "Slow run after parameter changes, tiny first-stage context.",
    },
    {
        "case_id": "slow_20260612_req381_stage0_11175tok",
        "run": "Adder+20260612_103636",
        "request_index": 381,
        "note": "Slow run after parameter changes, about 11K prompt tokens.",
    },
    {
        "case_id": "slow_20260612_req412_stage2_30004tok",
        "run": "Adder+20260612_103636",
        "request_index": 412,
        "note": "Slow run with low decode TPS, about 30K prompt tokens.",
    },
    {
        "case_id": "slow_20260612_req369_stage22_102810tok",
        "run": "Adder+20260612_103636",
        "request_index": 369,
        "note": "Slow run, very large stage-22 prompt near 100K tokens.",
    },
    {
        "case_id": "slow_20260612_req949_stage26_32062tok",
        "run": "Adder+20260612_103636",
        "request_index": 949,
        "note": "Slow run final request with very low observed decode TPS.",
    },
]


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


def parse_requests(log_path: Path) -> list[dict[str, Any]]:
    records = []
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            match = REQUEST_RE.search(line)
            if not match:
                continue
            record = {
                "line_number": line_number,
                "timestamp": line[:23],
                "raw": line.rstrip("\n"),
            }
            for key, raw_value in FIELD_RE.findall(match.group("body")):
                record[key] = parse_value(key, raw_value)
            records.append(record)
    return records


def parse_sections(msg_path: Path) -> list[dict[str, str]]:
    text = msg_path.read_text(encoding="utf-8", errors="replace")
    sections = []
    for match in SECTION_RE.finditer(text):
        kind = match.group("kind")
        body = match.group("body").strip("\n")
        tool_name = ""
        if kind == "Tool":
            name_match = re.match(r"Name:\s*(?P<name>[^\n]+)\n\n(?P<body>.*)", body, re.S)
            if name_match:
                tool_name = name_match.group("name").strip()
                body = name_match.group("body").strip("\n")
        sections.append({"kind": kind, "tool_name": tool_name, "body": body})
    return sections


def rough_token_count(text: str) -> int:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    non_cjk = len(text) - cjk
    return int(cjk + non_cjk / 4)


def trim_to_token_budget(text: str, target_tokens: int) -> str:
    if target_tokens <= 0:
        return text
    # Keep a little headroom because this heuristic under/over counts depending
    # on code density and Chinese text.
    budget = max(512, int(target_tokens * 1.08))
    if rough_token_count(text) <= budget:
        return text
    low, high = 0, len(text)
    while low < high:
        mid = (low + high) // 2
        if rough_token_count(text[mid:]) <= budget:
            high = mid
        else:
            low = mid + 1
    trimmed = text[low:]
    newline = trimmed.find("\n")
    if 0 <= newline < 200:
        trimmed = trimmed[newline + 1 :]
    return (
        "[TRUNCATED_PREFIX: token-matched tail extracted from ucagent-msg.log]\n"
        + trimmed
    )


def format_section(section: dict[str, str]) -> str:
    kind = section["kind"]
    if kind == "Human":
        heading = "================================ Human Message ================================="
    elif kind == "Ai":
        heading = "================================== Ai Message =================================="
    else:
        heading = "================================= Tool Message ================================="
    body = section["body"]
    if kind == "Tool" and section.get("tool_name"):
        body = f"Name: {section['tool_name']}\n\n{body}"
    return f"{heading}\n\n{body}".rstrip()


def sections_to_transcript(sections: list[dict[str, str]]) -> str:
    return "\n\n".join(format_section(section) for section in sections).strip() + "\n"


def sections_to_chat_messages(sections: list[dict[str, str]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for section in sections:
        if section["kind"] == "Ai":
            role = "assistant"
            content = section["body"]
        elif section["kind"] == "Tool":
            role = "user"
            content = "TOOL MESSAGE"
            if section.get("tool_name"):
                content += f" ({section['tool_name']})"
            content += ":\n" + section["body"]
        else:
            role = "user"
            content = section["body"]
        if not content.strip():
            continue
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n\n" + content
        else:
            messages.append({"role": role, "content": content})
    return messages


def selected_context_sections(
    sections: list[dict[str, str]],
    success_request_index: int,
) -> list[dict[str, str]]:
    """Return visible transcript sections before the Nth successful main response."""
    seen_ai = 0
    for index, section in enumerate(sections):
        if section["kind"] == "Ai":
            seen_ai += 1
            if seen_ai == success_request_index:
                return sections[:index]
    return sections


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", default="log")
    parser.add_argument("--out-dir", default="benchmark/llm_latency_cases")
    parser.add_argument("--model", default="mdq100/qwen3.5-coder:122b")
    args = parser.parse_args()

    log_root = Path(args.log_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_run: dict[str, dict[str, Any]] = {}
    for case in DEFAULT_CASES:
        run = case["run"]
        if run not in by_run:
            run_dir = log_root / run
            requests = [
                record
                for record in parse_requests(run_dir / "ucagent-log.log")
                if record.get("role") == "main" and record.get("status") == "success"
            ]
            sections = parse_sections(run_dir / "ucagent-msg.log")
            by_run[run] = {"requests": requests, "sections": sections}

    manifest = {
        "description": (
            "Prompt-like latency benchmark cases extracted from UCAgent logs. "
            "exact_llm_input_available=false because existing logs do not persist "
            "LangGraph pre_model_hook llm_input_messages."
        ),
        "model": args.model,
        "cases": [],
    }

    for case in DEFAULT_CASES:
        run_data = by_run[case["run"]]
        request_index = int(case["request_index"])
        requests = run_data["requests"]
        if request_index < 1 or request_index > len(requests):
            raise IndexError(f"{case['case_id']}: request_index out of range")
        request = requests[request_index - 1]
        context_sections = selected_context_sections(run_data["sections"], request_index)
        raw_transcript = sections_to_transcript(context_sections)
        target_tokens = int(request.get("prompt_tokens") or 0)
        token_matched = trim_to_token_budget(raw_transcript, target_tokens)

        case_dir = out_dir / case["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "raw_transcript_before_request.txt").write_text(
            raw_transcript, encoding="utf-8"
        )
        (case_dir / "token_matched_prompt.txt").write_text(
            token_matched, encoding="utf-8"
        )
        write_json(case_dir / "chat_messages_approx.json", sections_to_chat_messages(context_sections))
        write_json(
            case_dir / "ollama_chat_request.json",
            {
                "model": args.model,
                "messages": [{"role": "user", "content": token_matched}],
                "stream": False,
                "keep_alive": -1,
                "options": {"num_predict": 128},
            },
        )
        metadata = {
            **case,
            "request": request,
            "raw_transcript_chars": len(raw_transcript),
            "raw_transcript_rough_tokens": rough_token_count(raw_transcript),
            "token_matched_chars": len(token_matched),
            "token_matched_rough_tokens": rough_token_count(token_matched),
            "exact_llm_input_available": False,
            "reconstruction_note": (
                "raw_transcript_before_request is reconstructed from ucagent-msg.log "
                "visible messages before the corresponding successful main response. "
                "token_matched_prompt is a tail-trimmed transcript sized near the "
                "recorded prompt_tokens."
            ),
        }
        write_json(case_dir / "metadata.json", metadata)
        manifest["cases"].append(metadata)

    write_json(out_dir / "manifest.json", manifest)
    print(f"Wrote {len(DEFAULT_CASES)} cases to {out_dir}")


if __name__ == "__main__":
    main()
