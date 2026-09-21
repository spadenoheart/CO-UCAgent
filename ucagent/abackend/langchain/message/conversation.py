# --- coding: utf-8 ---
"""Message and state utilities for UCAgent."""

from .statistic import MessageStatistic
from ucagent.util.functions import fill_dlist_none
from ucagent.util.log import warning, info

from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.messages import AIMessage, RemoveMessage, BaseMessage, ToolMessage, HumanMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler
from langmem.short_term import SummarizationNode
from langgraph.prebuilt.chat_agent_executor import AgentState
from typing import Any, Dict, Union, Optional, List, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
import time
import yaml
import json
import os
import hashlib


def _coerce_count(value: Any) -> int:
    """Convert structured-summary count fields without trusting model types."""
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


class TokenSpeedCallbackHandler(BaseCallbackHandler):
    """Callback handler to monitor token generation speed."""

    def __init__(self):
        super().__init__()
        self.total_tokens_size = 0
        self.last_tokens_size = 0
        self.last_access_time = 0.0
        self.last_token_speed = 0.0

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.total_tokens_size += len(token)
        message = kwargs["chunk"].message
        if message and hasattr(message, "tool_call_chunks"):
            for tool_call in message.tool_call_chunks:
                tool_name = tool_call["name"]
                args = tool_call["args"]
                if tool_name:
                    self.total_tokens_size += len(tool_name)
                if args:
                    self.total_tokens_size += len(args)

    def get_speed(self) -> float:
        if self.last_access_time == 0.0:
            self.last_access_time = time.time()
            self.last_tokens_size = self.total_tokens_size
            return 0.0
        now_time = time.time()
        delta_time = now_time - self.last_access_time
        if delta_time < 1.0:
            return self.last_token_speed
        delta_tokens = self.total_tokens_size - self.last_tokens_size
        self.last_access_time = now_time
        self.last_token_speed = delta_tokens / delta_time
        self.last_tokens_size = self.total_tokens_size
        return self.last_token_speed

    def total(self) -> int:
        return self.total_tokens_size


def fix_tool_call_args(input: Union[Dict[str, Any], BaseModel]) -> Dict[str, Any]:
        for msg in input["messages"][-4:]:
            if not isinstance(msg, AIMessage):
                continue
            if hasattr(msg, "additional_kwargs"):
                msg.additional_kwargs = fill_dlist_none(msg.additional_kwargs, '{}', "arguments", ["arguments"])
            if hasattr(msg, "invalid_tool_calls"):
                msg.invalid_tool_calls = fill_dlist_none(msg.invalid_tool_calls, '{}', "args", ["args"])


def _summary_model_name(model: Any) -> str:
    return str(getattr(model, "model_name", None) or getattr(model, "model", None) or model.__class__.__name__)


def _summary_messages_input_tokens(messages, instruction: str) -> int:
    return count_tokens_approximately([HumanMessage(content=instruction)] + list(messages or []))


def _record_summary_stats(stats_hook, payload: dict) -> None:
    if callable(stats_hook):
        try:
            stats_hook(payload)
        except Exception:
            pass


def _summary_runtime_defaults(summary_runtime: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    runtime = dict(summary_runtime or {})
    runtime.setdefault("fallback_to_main", True)
    runtime.setdefault("max_context_tokens", 0)
    runtime.setdefault("stats_hook", None)
    runtime.setdefault("kind", "plain")
    runtime.setdefault("repair_used", False)
    return runtime


def _invoke_summary_model_with_runtime(prompt_messages, model, summary_runtime: Optional[Dict[str, Any]] = None, structured_schema=None):
    runtime = _summary_runtime_defaults(summary_runtime)
    primary_model = model
    fallback_model = runtime.get("fallback_model")
    fallback_enabled = bool(runtime.get("fallback_to_main", True))
    max_context_tokens = int(runtime.get("max_context_tokens", 0) or 0)
    input_tokens = count_tokens_approximately(prompt_messages)
    selected_model = primary_model
    fallback_used = False
    fallback_reason = ""
    if (
        max_context_tokens > 0
        and input_tokens > max_context_tokens
        and fallback_enabled
        and fallback_model is not None
        and fallback_model is not primary_model
    ):
        selected_model = fallback_model
        fallback_used = True
        fallback_reason = f"context_overflow:{input_tokens}>{max_context_tokens}"
        warning(
            "Summary model context overflow; switch to main model for this summary "
            f"(input_tokens={input_tokens}, max_context_tokens={max_context_tokens})."
        )

    start = time.time()
    selected_name = _summary_model_name(selected_model)
    try:
        if structured_schema is not None and hasattr(selected_model, "with_structured_output"):
            structured_model = selected_model.with_structured_output(structured_schema)
            response = structured_model.invoke(prompt_messages)
        else:
            response = selected_model.invoke(prompt_messages)
    except Exception as exc:
        if (
            fallback_enabled
            and fallback_model is not None
            and selected_model is not fallback_model
        ):
            fallback_used = True
            fallback_reason = f"invoke_error:{type(exc).__name__}"
            warning(
                f"Summary model invoke failed on {selected_name}: {exc}. "
                "Falling back to main model for this summary."
            )
            selected_model = fallback_model
            selected_name = _summary_model_name(selected_model)
            response = selected_model.invoke(prompt_messages)
        else:
            raise
    latency_ms = int((time.time() - start) * 1000)
    return response, {
        "kind": runtime.get("kind", "plain"),
        "summary_model_name": selected_name,
        "summary_latency_ms": latency_ms,
        "summary_fallback": fallback_used,
        "summary_fallback_reason": fallback_reason,
        "summary_input_tokens": input_tokens,
    }


def summarize_messages(messages, summarization_size, model, fallback_model=None, summary_runtime: Optional[Dict[str, Any]] = None):
    """Summarize messages to reduce their token count."""
    from langchain_core.messages import HumanMessage
    instruction = (f"Summarize the conversation in less than {summarization_size} tokens, keeping the important information and context. Be concise and clear. "
                   "You must follow the rules below:\n"
                   "1. The system message should be preserved as much as possible.\n"
                   "2. The tool call results should be concise and clear, need removal of unnecessary details (e.g. file content, irrelevant context, code snippets).\n"
                   "3. Record current task status if any.\n"
                   "4. Record the verification experience you have learned.\n"
                   "5. Record the tools behavior you have learned.\n"
                   "6. Record the tools error handle suggestions you have learned.\n"
                   "7. Record the important actions you have taken and their outcomes.\n"
                   "8. Record any other important information and context.\n"
                   "You need to define the format of the summary which should be friendly to any LLMs.\n"
                   "Note: the first followed message may be the previous summary you provided before, you need to incorporate it into the new summary.\n"
                   "The result you provide should be only the summary, no other explanations or additional information."
                   )
    prompt_messages = [HumanMessage(content=instruction)] + messages
    warning(f"Summarizing messages({count_tokens_approximately(messages)} tokens, {len(messages)} messages) to reduce context size ...")
    try:
        summary_response, runtime_info = _invoke_summary_model_with_runtime(
            prompt_messages,
            model,
            summary_runtime={
                **_summary_runtime_defaults(summary_runtime),
                "fallback_model": fallback_model,
                "kind": "plain",
            },
        )
    except Exception as exc:
        warning(
            "All summary model attempts failed; using deterministic local "
            f"summary instead ({type(exc).__name__}: {exc})."
        )
        summary_response = AIMessage(content=_build_fallback_plain_summary(messages))
        runtime_info = {
            "kind": "plain",
            "summary_model_name": _summary_model_name(model),
            "summary_latency_ms": 0,
            "summary_fallback": True,
            "summary_fallback_reason": f"invoke_error_exhausted:{type(exc).__name__}",
            "summary_input_tokens": _summary_messages_input_tokens(messages, instruction),
        }
    summary_text = _normalize_message_content_text(getattr(summary_response, "content", ""))
    quality_guard_fail = False
    if _summary_response_has_tool_calls(summary_response) or not summary_text.strip():
        quality_guard_fail = True
        warning("Summary response was empty or contained tool calls; replacing it with fallback summary content.")
        summary_response = AIMessage(content=_build_fallback_plain_summary(messages))
    elif not isinstance(summary_response, AIMessage) or not isinstance(summary_response.content, str):
        summary_response = AIMessage(content=summary_text)
    output_tokens = count_tokens_approximately(summary_response.content)
    _record_summary_stats(
        _summary_runtime_defaults(summary_runtime).get("stats_hook"),
        {
            **runtime_info,
            "summary_output_tokens": output_tokens,
            "summary_repair_used": False,
            "summary_quality_guard_fail": quality_guard_fail,
        },
    )
    warning(
        "Summarization done, summary length: {} tokens. model={}, latency_ms={}, fallback={}.".format(
            output_tokens,
            runtime_info["summary_model_name"],
            runtime_info["summary_latency_ms"],
            runtime_info["summary_fallback"],
        )
    )
    return summary_response


def summarize_messages_structured(messages, summarization_size, model, fallback_model=None, summary_runtime: Optional[Dict[str, Any]] = None):
    """Summarize messages with a structured schema to aid downstream retrieval."""
    instruction = (
        "Summarize the following verification conversation in less than {size} tokens. "
        "Output MUST be valid JSON with EXACTLY these top-level keys:\n"
        "{{"
        "\"stage_info\": {{\"stage_index\": int, \"stage_title\": str, \"section_index\": str, \"progress\": str, \"constraints\": str}},"
        "\"test_report\": {{\"pytest_cmd\": str, \"report_paths\": [str], \"total\": int, \"passed\": int, \"failed\": int, "
        "\"failed_cases_top\": [{{\"case\": str, \"symptom\": str}}], \"failed_checkpoints_top\": [str]}},"
        "\"coverage_status\": {{\"function_points_total\": int, \"check_points_total\": int, \"failed_check_points\": int, "
        "\"unmarked_check_points\": int, \"failed_check_point_list_top\": [str]}},"
        "\"bug_tracking\": {{\"bug_doc_path\": str, \"bugs_marked_count\": int, \"bug_ids_added\": [str], \"zero_occurrence_bug_ids\": [str], "
        "\"root_cause_hypotheses_top\": [{{\"hypothesis\": str, \"ref\": str}}]}},"
        "\"next\": {{\"todo\": str, \"blockers\": str, \"notes\": str}}"
        "}}\n"
        "Rules:\n"
        "- Never include raw code blocks or long pytest outputs; use short references like 'unity_test/tests/x.py:123' or 'uc_test_report/index.html'.\n"
        "- If a field is unknown, use empty string / 0 / empty list, but NEVER omit keys.\n"
        "- Incorporate the previous summary if present as the first message."
    ).format(size=summarization_size)
    warning(f"Structured summarizing messages({count_tokens_approximately(messages)} tokens, {len(messages)} messages) ...")
    prompt_messages = [HumanMessage(content=instruction)] + messages
    summary_response = None
    runtime_info = None
    if hasattr(model, "with_structured_output"):
        try:
            structured_response, runtime_info = _invoke_summary_model_with_runtime(
                prompt_messages,
                model,
                summary_runtime={
                    **_summary_runtime_defaults(summary_runtime),
                    "fallback_model": fallback_model,
                    "kind": "structured",
                },
                structured_schema=StructuredSummarySchema,
            )
            if isinstance(structured_response, BaseModel):
                summary_response = AIMessage(content=json.dumps(structured_response.model_dump(), ensure_ascii=False))
            elif isinstance(structured_response, dict):
                summary_response = AIMessage(content=json.dumps(structured_response, ensure_ascii=False))
            else:
                summary_response = structured_response
        except Exception:
            summary_response = None
            runtime_info = None
    if summary_response is None:
        try:
            summary_response, runtime_info = _invoke_summary_model_with_runtime(
                prompt_messages,
                model,
                summary_runtime={
                    **_summary_runtime_defaults(summary_runtime),
                    "fallback_model": fallback_model,
                    "kind": "structured",
                },
            )
        except Exception as exc:
            warning(
                "All structured-summary model attempts failed; using "
                f"deterministic local JSON instead ({type(exc).__name__}: {exc})."
            )
            summary_response = AIMessage(
                content=json.dumps(_build_fallback_structured_summary(messages), ensure_ascii=False)
            )
            runtime_info = {
                "kind": "structured",
                "summary_model_name": _summary_model_name(model),
                "summary_latency_ms": 0,
                "summary_fallback": True,
                "summary_fallback_reason": f"invoke_error_exhausted:{type(exc).__name__}",
                "summary_input_tokens": _summary_messages_input_tokens(messages, instruction),
            }
    raw_summary_text = _normalize_message_content_text(getattr(summary_response, "content", ""))
    repaired, repair_used = _coerce_structured_summary(
        raw_summary_text,
        messages,
        model,
        fallback_model=fallback_model,
        summary_runtime=summary_runtime,
    )
    if repaired is not None:
        if _summary_response_has_tool_calls(summary_response) or not raw_summary_text.strip():
            warning("Structured summary response was empty or contained tool calls; replaced with sanitized JSON summary.")
        else:
            warning("Structured summary repaired/normalized to valid JSON.")
        result = AIMessage(content=repaired)
        output_tokens = count_tokens_approximately(result.content)
        _record_summary_stats(
            _summary_runtime_defaults(summary_runtime).get("stats_hook"),
            {
                **(runtime_info or {}),
                "summary_model_name": (runtime_info or {}).get("summary_model_name", _summary_model_name(model)),
                "summary_latency_ms": (runtime_info or {}).get("summary_latency_ms", 0),
                "summary_fallback": (runtime_info or {}).get("summary_fallback", False),
                "summary_fallback_reason": (runtime_info or {}).get("summary_fallback_reason", ""),
                "summary_input_tokens": (runtime_info or {}).get("summary_input_tokens", _summary_messages_input_tokens(messages, instruction)),
                "summary_output_tokens": output_tokens,
                "summary_repair_used": repair_used,
                "summary_quality_guard_fail": _summary_response_has_tool_calls(summary_response) or not raw_summary_text.strip(),
            },
        )
        warning(
            "Structured summarization done, summary length: {} tokens. model={}, latency_ms={}, fallback={}, repair_used={}.".format(
                output_tokens,
                (runtime_info or {}).get("summary_model_name", _summary_model_name(model)),
                (runtime_info or {}).get("summary_latency_ms", 0),
                (runtime_info or {}).get("summary_fallback", False),
                repair_used,
            )
        )
        return result
    warning(f"Structured summarization done, summary length: {count_tokens_approximately(raw_summary_text)} tokens.")
    return AIMessage(content=raw_summary_text)


class StructuredSummarySchema(BaseModel):
    stage_info: Dict[str, object] = Field(default_factory=dict)
    test_report: Dict[str, object] = Field(default_factory=dict)
    coverage_status: Dict[str, object] = Field(default_factory=dict)
    bug_tracking: Dict[str, object] = Field(default_factory=dict)
    next: Dict[str, object] = Field(default_factory=dict)


def _empty_structured_summary(stage_info: dict) -> dict:
    return {
        "stage_info": {
            "stage_index": stage_info.get("stage_index", 0) if isinstance(stage_info.get("stage_index"), int) else 0,
            "stage_title": stage_info.get("stage_title", "") or "",
            "section_index": stage_info.get("section_index", "") or "",
            "progress": stage_info.get("progress", "") or "",
            "constraints": "",
        },
        "test_report": {
            "pytest_cmd": "",
            "report_paths": [],
            "total": 0,
            "passed": 0,
            "failed": 0,
            "failed_cases_top": [],
            "failed_checkpoints_top": [],
        },
        "coverage_status": {
            "function_points_total": 0,
            "check_points_total": 0,
            "failed_check_points": 0,
            "unmarked_check_points": 0,
            "failed_check_point_list_top": [],
        },
        "bug_tracking": {
            "bug_doc_path": "",
            "bugs_marked_count": 0,
            "bug_ids_added": [],
            "zero_occurrence_bug_ids": [],
            "root_cause_hypotheses_top": [],
        },
        "next": {
            "todo": "",
            "blockers": "",
            "notes": "",
        },
    }


def _try_parse_structured_json(text: str) -> Optional[dict]:
    obj = _load_json_maybe(text)
    if obj is None:
        return None
    if not isinstance(obj, dict):
        return None
    required = {"stage_info", "test_report", "coverage_status", "bug_tracking", "next"}
    if not required.issubset(obj.keys()):
        return None
    return _normalize_structured_summary(obj)


def _coerce_structured_summary(text: str, messages, model, fallback_model=None, summary_runtime: Optional[Dict[str, Any]] = None) -> Tuple[Optional[str], bool]:
    if not isinstance(text, str) or not text.strip():
        fallback = _build_fallback_structured_summary(messages)
        return json.dumps(fallback, ensure_ascii=False), True
    parsed = _try_parse_structured_json(text)
    if parsed is not None:
        return json.dumps(_fill_from_context(parsed, messages), ensure_ascii=False), False
    repair_prompt = (
        "You MUST output a valid JSON object with EXACTLY these top-level keys:\n"
        "[stage_info, test_report, coverage_status, bug_tracking, next].\n"
        "Do not include any extra text. If a field is unknown, use empty values.\n"
        "Convert the following content into the required JSON:\n\n"
        f"{text}"
    )
    try:
        repaired, _ = _invoke_summary_model_with_runtime(
            [HumanMessage(content=repair_prompt)] + messages[-8:],
            model,
            summary_runtime={
                **_summary_runtime_defaults(summary_runtime),
                "fallback_model": fallback_model,
                "kind": "structured_repair",
                "repair_used": True,
            },
        )
        parsed = _try_parse_structured_json(getattr(repaired, "content", ""))
        if parsed is not None:
            return json.dumps(_fill_from_context(parsed, messages), ensure_ascii=False), True
    except Exception:
        pass
    fallback = _build_fallback_structured_summary(messages)
    return json.dumps(fallback, ensure_ascii=False), True


def _normalize_message_content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _summary_response_has_tool_calls(summary_response: Any) -> bool:
    if not isinstance(summary_response, AIMessage):
        return False
    return bool(
        getattr(summary_response, "tool_calls", None)
        or getattr(summary_response, "invalid_tool_calls", None)
    )


def _build_fallback_plain_summary(messages: List) -> str:
    fallback = _build_fallback_structured_summary(messages)
    return "SUMMARY_FALLBACK:\n" + yaml.safe_dump(fallback, allow_unicode=True, sort_keys=False)


def _load_json_maybe(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        pass
    # try to extract the largest JSON object in the text
    if not isinstance(text, str):
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start:end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None


def _extract_stage_info_from_stage_summary(messages: List) -> Tuple[Optional[int], str, str, str]:
    for msg in reversed(messages[-20:]):
        if not isinstance(msg, AIMessage):
            continue
        content = msg.content or ""
        if not isinstance(content, str) or "STAGE_SUMMARY" not in content:
            continue
        try:
            _, payload = content.split("STAGE_SUMMARY:", 1)
            data = _safe_load_yaml(payload.strip())
            if not isinstance(data, dict):
                continue
            si = data.get("stage_info") or {}
            if isinstance(si, dict):
                idx = si.get("stage_index")
                title = si.get("stage_title") or ""
                sec = si.get("section_index") or ""
                prog = si.get("progress") or ""
                if isinstance(idx, int):
                    return idx, title, sec, prog
        except Exception:
            continue
    return None, "", "", ""


def _extract_latest_batch_summary(messages: List) -> Optional[dict]:
    for msg in reversed(messages[-40:]):
        if isinstance(msg, ToolMessage):
            content = msg.content or ""
            if not isinstance(content, str):
                continue
            if "BATCH_SUMMARY" in content:
                try:
                    _, payload = content.split("BATCH_SUMMARY", 1)
                    data = _safe_load_yaml(payload.strip())
                    if isinstance(data, dict):
                        return data
                except Exception:
                    continue
            # fall back to parse raw YAML tool output if any
            data = _safe_load_yaml(content)
            if isinstance(data, dict):
                return data
    return None


def _extract_latest_check_summary(messages: List) -> Optional[dict]:
    for msg in reversed(messages[-40:]):
        if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "Check":
            content = msg.content or ""
            if not isinstance(content, str):
                continue
            data = _safe_load_yaml(content)
            if isinstance(data, dict):
                return data
    return None


def _build_fallback_structured_summary(messages: List) -> dict:
    stage_index, stage_title, section_index, progress = _extract_stage_info_from_tips(messages)
    if stage_index is None:
        stage_index, stage_title, section_index, progress = _extract_stage_info_from_stage_summary(messages)
    base = _empty_structured_summary(
        {
            "stage_index": stage_index,
            "stage_title": stage_title,
            "section_index": section_index,
            "progress": progress,
        }
    )
    batch = _extract_latest_batch_summary(messages)
    if isinstance(batch, dict):
        report_root = _unwrap_report(batch)
        tests = _extract_tests_block(report_root) or {}
        if isinstance(tests, dict):
            base["test_report"]["total"] = int(tests.get("total", 0) or 0)
            base["test_report"]["passed"] = int(tests.get("passed", 0) or 0)
            base["test_report"]["failed"] = int(tests.get("failed", 0) or 0)
        failed_cases = _extract_failed_cases(report_root, tests)
        if failed_cases:
            base["test_report"]["failed_cases_top"] = [
                {"case": str(c), "symptom": ""} for c in failed_cases[:10]
            ]
        failed_cp = _extract_failed_checkpoints(report_root, tests)
        if failed_cp:
            base["test_report"]["failed_checkpoints_top"] = [str(x) for x in failed_cp[:12]]
            base["coverage_status"]["failed_check_point_list_top"] = [str(x) for x in failed_cp[:12]]
        report_paths = (
            report_root.get("report_paths")
            or report_root.get("report_path")
            or report_root.get("report_dir")
        )
        if isinstance(report_paths, list):
            base["test_report"]["report_paths"] = report_paths[:5]
        elif isinstance(report_paths, str) and report_paths:
            base["test_report"]["report_paths"] = [report_paths]
        pytest_cmd = report_root.get("pytest_cmd") or report_root.get("cmd")
        if isinstance(pytest_cmd, str):
            base["test_report"]["pytest_cmd"] = pytest_cmd
    elif _extract_latest_check_summary(messages):
        check_summary = _extract_latest_check_summary(messages) or {}
        check_info = check_summary.get("check_info") or {}
        if isinstance(check_info, dict):
            failed_cp = _extract_failed_checkpoints(check_info, {})
            if failed_cp:
                base["coverage_status"]["failed_check_point_list_top"] = [str(x) for x in failed_cp[:12]]
    return base


def _normalize_structured_summary(obj: dict) -> dict:
    base = _empty_structured_summary({"stage_index": 0, "stage_title": "", "section_index": "", "progress": ""})
    # stage_info mapping
    stage_info = obj.get("stage_info") or {}
    if isinstance(stage_info, dict):
        if "stage_index" not in stage_info and "index" in stage_info:
            stage_info["stage_index"] = stage_info.get("index")
        if "stage_title" not in stage_info and "title" in stage_info:
            stage_info["stage_title"] = stage_info.get("title")
        if "progress" not in stage_info and "description" in stage_info:
            stage_info["progress"] = stage_info.get("description")
    base["stage_info"].update(stage_info if isinstance(stage_info, dict) else {})
    # test_report
    test_report = obj.get("test_report") or {}
    if isinstance(test_report, dict):
        base["test_report"].update(test_report)
    # coverage_status
    coverage_status = obj.get("coverage_status") or {}
    if isinstance(coverage_status, dict):
        base["coverage_status"].update(coverage_status)
    # bug_tracking
    bug_tracking = obj.get("bug_tracking") or {}
    if isinstance(bug_tracking, dict):
        base["bug_tracking"].update(bug_tracking)
    # next
    next_info = obj.get("next") or {}
    if isinstance(next_info, dict):
        base["next"].update(next_info)
    return base


def _fill_from_context(obj: dict, messages: List) -> dict:
    base = _build_fallback_structured_summary(messages)
    # overlay obj on base
    normalized = _normalize_structured_summary(obj)
    # stage_info: never clobber fallback with empty defaults
    base["stage_info"] = _merge_stage_info(base.get("stage_info", {}), normalized.get("stage_info", {}))
    for k in base:
        if k == "stage_info":
            continue
        if isinstance(base[k], dict):
            base[k] = _merge_section(base[k], normalized.get(k, {}))
        else:
            base[k] = normalized.get(k, base[k])
    return base


def _merge_stage_info(base: dict, incoming: dict) -> dict:
    if not isinstance(base, dict):
        base = {}
    if not isinstance(incoming, dict):
        return base
    out = dict(base)
    idx = incoming.get("stage_index")
    if isinstance(idx, int) and idx != 0:
        out["stage_index"] = idx
    for key in ("stage_title", "section_index", "progress", "constraints"):
        val = incoming.get(key)
        if isinstance(val, str) and val.strip():
            out[key] = val
    return out


def _merge_section(base: dict, incoming: dict) -> dict:
    if not isinstance(base, dict):
        base = {}
    if not isinstance(incoming, dict):
        return base
    out = dict(base)
    for key, val in incoming.items():
        if isinstance(val, str):
            if val.strip():
                out[key] = val
        elif isinstance(val, list):
            if val:
                out[key] = val
        elif isinstance(val, (int, float)):
            if val != 0 or not out.get(key):
                out[key] = val
        else:
            if val is not None:
                out[key] = val
    return out


def _parse_summary_payload(content: str) -> Optional[dict]:
    if not isinstance(content, str) or not content.strip():
        return None
    if "BATCH_SUMMARY" in content:
        try:
            _, payload = content.split("BATCH_SUMMARY", 1)
            data = _safe_load_yaml(payload.strip())
            return data if isinstance(data, dict) else None
        except Exception:
            return None
    if "STAGE_SUMMARY" in content:
        try:
            _, payload = content.split("STAGE_SUMMARY:", 1)
            data = _safe_load_yaml(payload.strip())
            return data if isinstance(data, dict) else None
        except Exception:
            return None
    return _safe_load_yaml(content)


def _is_failure_summary(obj: dict) -> bool:
    if not isinstance(obj, dict):
        return False
    if obj.get("run_test_success") is False:
        return True
    tests = obj.get("tests")
    if isinstance(tests, dict):
        try:
            failed = int(tests.get("failed", 0) or 0)
        except Exception:
            failed = 0
        if failed > 0:
            return True
    for key in ("failed_test_cases_top",):
        block = obj.get(key)
        if isinstance(block, list) and block:
            return True
    test_report = obj.get("test_report")
    if isinstance(test_report, dict):
        try:
            failed = int(test_report.get("failed", 0) or 0)
        except Exception:
            failed = 0
        if failed > 0:
            return True
        for key in ("failed_cases_top", "failed_checkpoints_top"):
            block = test_report.get(key)
            if isinstance(block, list) and block:
                return True
    return False


def _extract_tests_block(obj: dict) -> Optional[dict]:
    # direct tests
    tests = obj.get("tests")
    if isinstance(tests, dict):
        return tests
    # nested reports
    for key in ("TEST_REPORT", "test_report", "report", "REPORT"):
        rep = obj.get(key)
        if isinstance(rep, dict):
            tests = rep.get("tests")
            if isinstance(tests, dict):
                return tests
    # check_info list
    check_info = obj.get("check_info")
    if isinstance(check_info, list):
        for item in check_info:
            if not isinstance(item, dict):
                continue
            rep = item.get("TEST_REPORT") or item.get("test_report")
            if isinstance(rep, dict):
                tests = rep.get("tests")
                if isinstance(tests, dict):
                    return tests
    return None


def _extract_failed_cases(obj: dict, tests: dict) -> List[str]:
    cases = []
    test_cases = tests.get("test_cases") if isinstance(tests, dict) else None
    if isinstance(test_cases, dict):
        cases = [k for k, v in test_cases.items() if str(v).upper() not in ("PASS", "PASSED")]
    if cases:
        return cases
    # fallback to failed_test_case_with_check_point_list keys
    for key in ("failed_test_case_with_check_point_list", "failed_cases"):
        block = obj.get(key)
        if isinstance(block, dict):
            return list(block.keys())
        if isinstance(block, list):
            return [str(x) for x in block]
    return []


def _extract_failed_checkpoints(obj: dict, tests: dict) -> List[str]:
    for key in (
        "failed_check_points_top",
        "failed_check_point_list",
        "failed_checkpoints",
        "marked_check_point_list",
    ):
        block = obj.get(key)
        if isinstance(block, list):
            return [str(x) for x in block]
    # from TEST_REPORT
    for key in ("TEST_REPORT", "test_report", "REPORT"):
        rep = obj.get(key)
        if isinstance(rep, dict):
            block = rep.get("failed_check_point_list")
            if isinstance(block, list):
                return [str(x) for x in block]
    return []


def _safe_load_yaml(text: str) -> Optional[dict]:
    if not text or not isinstance(text, str):
        return None
    try:
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _plain_config(value: Any) -> Dict[str, Any]:
    if hasattr(value, "as_dict"):
        value = value.as_dict()
    return value if isinstance(value, dict) else {}


def _bool_cfg(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except Exception:
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(v) for v in value]
        return str(value)


class LLMInputMessagesDumper:
    """Persist the exact post-hook messages that are sent to the chat model."""

    def __init__(self, workspace: str, config: Any, stage_getter=None):
        cfg = _plain_config(config)
        self.enabled = _bool_cfg(cfg.get("enabled"), False)
        self.workspace = workspace
        self.stage_getter = stage_getter
        self.include_content = _bool_cfg(cfg.get("include_content"), True)
        self.max_content_chars = int(cfg.get("max_content_chars", 0) or 0)
        self.max_messages = int(cfg.get("max_messages", 0) or 0)
        output_dir = str(cfg.get("output_dir") or "llm_input_dumps")
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(workspace, output_dir)
        self.output_dir = output_dir
        self._seq = 0

    def dump(self, messages: List, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self._seq += 1
            stage = self._stage()
            payload = {
                "schema_version": 1,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "request_index": self._seq,
                "stage": stage,
                "message_count": len(messages or []),
                "prompt_tokens_approx": int(count_tokens_approximately(messages or [])),
                "metadata": _jsonable(metadata or {}),
                "messages": self._serialize_messages(messages or []),
            }
            path = os.path.join(
                self.output_dir,
                f"req_{self._seq:06d}_stage_{self._safe_stage(stage)}.json",
            )
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            latest = os.path.join(self.output_dir, "latest.json")
            with open(latest, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            info(
                f"[llm_input_dump] request={self._seq} stage={stage} "
                f"messages={payload['message_count']} tokens≈{payload['prompt_tokens_approx']} path={path}"
            )
        except Exception as exc:
            warning(f"[llm_input_dump] failed: {exc}")

    def _stage(self) -> str:
        if not callable(self.stage_getter):
            return "NA"
        try:
            stage = self.stage_getter()
            return "NA" if stage is None else str(stage)
        except Exception:
            return "NA"

    def _safe_stage(self, stage: Any) -> str:
        text = str(stage)
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)[:64] or "NA"

    def _serialize_messages(self, messages: List) -> List[Dict[str, Any]]:
        selected = list(messages or [])
        if self.max_messages > 0:
            selected = selected[-self.max_messages:]
        return [self._serialize_message(i, msg) for i, msg in enumerate(selected)]

    def _serialize_message(self, index: int, msg: Any) -> Dict[str, Any]:
        content = _normalize_message_content_text(getattr(msg, "content", ""))
        if not self.include_content:
            content = ""
        elif self.max_content_chars > 0 and len(content) > self.max_content_chars:
            content = content[:self.max_content_chars] + "\n...[truncated]"
        data = {
            "index": index,
            "class": msg.__class__.__name__,
            "type": getattr(msg, "type", ""),
            "content": content,
        }
        for attr in ("id", "name", "tool_call_id", "status"):
            value = getattr(msg, attr, None)
            if value is not None:
                data[attr] = _jsonable(value)
        for attr in ("additional_kwargs", "response_metadata", "tool_calls", "invalid_tool_calls", "usage_metadata"):
            value = getattr(msg, attr, None)
            if value:
                data[attr] = _jsonable(value)
        return data


def _extract_stage_info_from_tips(messages: List) -> Tuple[Optional[int], str, str, str]:
    """
    Extract (stage_index, stage_title, section_index, progress) from the most recent tips-like HumanMessage.
    Tips are YAML generated by StageManager and include 'current_stage'.
    """
    from langchain_core.messages import HumanMessage as _HumanMessage
    for msg in reversed(messages[-12:]):
        if not isinstance(msg, _HumanMessage):
            continue
        data = _safe_load_yaml(msg.content)
        if not data:
            continue
        current_stage = data.get("current_stage") or {}
        if not isinstance(current_stage, dict):
            continue
        stage_index = current_stage.get("index")
        stage_title = ""
        section_index = str(current_stage.get("section_index") or "")
        progress = ""
        task = current_stage.get("task") or {}
        if isinstance(task, dict):
            stage_title = str(task.get("title") or "")
            # try to capture a progress line if present in description
            desc = task.get("description") or []
            if isinstance(desc, list):
                for line in reversed(desc[-8:]):
                    if isinstance(line, str) and ("完成进度" in line or "progress" in line.lower()):
                        progress = line.strip()
                        break
        if isinstance(stage_index, int):
            return stage_index, stage_title, section_index, progress
    return None, "", "", ""


def _summarize_run_testcases_yaml(tool_yaml: dict) -> dict:
    """
    Deterministic Batch Summary extractor for RunTestCases tool output (YAML dict).
    Keeps small, stable fields and drops huge STDOUT/STDERR.
    """
    ret: dict = {
        "run_test_success": bool(tool_yaml.get("run_test_success", False)),
        "tests": {},
        "failed_test_cases_top": [],
        "failed_check_points_top": [],
        "report_paths": [],
    }
    report_root = _unwrap_report(tool_yaml)
    tests = _extract_tests_block(report_root) or {}
    if isinstance(tests, dict):
        total = int(tests.get("total", 0) or 0)
        fails = int(tests.get("fails", 0) or 0)
        test_cases = tests.get("test_cases") or {}
        if isinstance(test_cases, dict):
            status_counts: dict[str, int] = {}
            for status in test_cases.values():
                normalized = str(status).strip().upper() or "UNKNOWN"
                status_counts[normalized] = status_counts.get(normalized, 0) + 1
            explicit_total = len(test_cases)
            total = max(total, explicit_total)
            passed = sum(status_counts.get(status, 0) for status in ("PASS", "PASSED"))
            failed = sum(status_counts.get(status, 0) for status in ("FAIL", "FAILED"))
            other = max(0, total - passed - failed)
            ret["tests"] = {
                "total": total,
                "passed": passed,
                "failed": failed,
                "invalid_or_other": other,
                "status_counts": status_counts,
            }
            failed_cases = [
                k for k, v in test_cases.items()
                if str(v).strip().upper() not in ("PASS", "PASSED")
            ]
            ret["failed_test_cases_top"] = failed_cases[:10]
        else:
            # Without per-case statuses, only infer passes from a verified successful run.
            passed = max(0, total - fails) if ret["run_test_success"] and total > 0 else 0
            ret["tests"] = {
                "total": total,
                "passed": passed,
                "failed": fails,
                "invalid_or_other": max(0, total - passed - fails),
                "status_counts": {},
            }
    # checkpoint fields vary by checker; handle common names
    failed_cp = report_root.get("failed_check_point_list") or report_root.get("marked_check_point_list") or []
    if isinstance(failed_cp, list):
        ret["failed_check_points_top"] = [str(x) for x in failed_cp[:12]]
    for k in ("report_dir", "report_path", "result_json_path"):
        v = report_root.get(k)
        if isinstance(v, str) and v:
            ret["report_paths"].append(v)
    pytest_cmd = report_root.get("pytest_cmd") or report_root.get("cmd")
    if isinstance(pytest_cmd, str) and pytest_cmd:
        ret["pytest_cmd"] = pytest_cmd
    return ret


def _unwrap_report(obj: dict) -> dict:
    if not isinstance(obj, dict):
        return {}
    report = obj.get("REPORT")
    if isinstance(report, dict):
        return report
    return obj


def remove_messages(messages, max_keep_msgs):
    """Remove older messages to keep the most recent max_keep_msgs messages."""
    if len(messages) <= max_keep_msgs:
        return messages, []
    index = (-max_keep_msgs) % len(messages)
    # system messages are not removed
    return messages[index:], [RemoveMessage(id=msg.id) for msg in messages[:index] if msg.type != "system"]


class SummarizationAndFixToolCall(SummarizationNode):
    """Custom summarization node that fixes tool call arguments."""

    def set_llm_input_dumper(self, dumper):
        self.llm_input_dumper = dumper
        return self

    def set_max_keep_msgs(self, msg_stat: MessageStatistic, max_keep_msgs: int):
        self.max_keep_msgs = max_keep_msgs
        self.msg_stat = msg_stat
        return self

    def _func(self, input: Union[Dict[str, Any], BaseModel]) -> Dict[str, Any]:
        fix_tool_call_args(input)
        deleted_msg = []
        if hasattr(self, "max_keep_msgs"):
            messages, deleted_msg = remove_messages(input["messages"], self.max_keep_msgs)
            input["messages"] = messages
        ret = super()._func(input)
        if deleted_msg:
            ret["messages"] = deleted_msg
        if "llm_input_messages" in ret:
            llm_messages = ret["llm_input_messages"]
        else:
            llm_messages = ret["messages"]
        self.msg_stat.update_message(llm_messages)
        dumper = getattr(self, "llm_input_dumper", None)
        if dumper is not None:
            dumper.dump(llm_messages, metadata={"node": self.__class__.__name__})
        return ret

    def set_max_token(self, max_token: int):
        self.max_token = max_token
        return self

    def get_max_token(self) -> int:
        return self.max_token

    def get_max_keep_msgs(self) -> int:
        return self.max_keep_msgs


class UCMessagesNode:
    """
    Node to trim and summarize messages.
    Messages layout:
      local memory: role_info(system) + history_msgs
      llm input: summary_msgs(summarized by max_summary_tokens) + role_info + history_msg
    """

    def __init__(
        self,
        msg_stat: MessageStatistic,
        max_summary_tokens: int,
        max_keep_msgs: int,
        tail_keep_msgs: int,
        model,
        max_tokens: int = 0,
        hard_max_tokens: int = 0,
        structured_summary: bool = False,
        hierarchical_summary: bool = False,
        enable_long_term_memory: bool = False,
        long_term_memory=None,
        long_term_memory_cfg: Optional[Dict] = None,
        enable_failure_aware_context: bool = False,
        failure_aware_context_cfg: Optional[Dict] = None,
        enable_stage_state_package: bool = False,
        stage_state_package_cfg: Optional[Dict] = None,
        enable_observation_masking: bool = False,
        observation_masking_cfg: Optional[Dict] = None,
        vagent=None,
        dut_name: str = "",
        main_model=None,
        summary_model_cfg: Optional[Dict] = None,
        summary_model_source: str = "main",
        llm_input_dumper=None,
    ):
        self.msg_stat = msg_stat
        self.max_summary_tokens = max_summary_tokens
        self.max_keep_msgs = max_keep_msgs
        self.tail_keep_msgs = tail_keep_msgs
        self.max_token = max(0, int(max_tokens or 0))
        configured_hard_limit = int(hard_max_tokens or 0)
        self.hard_max_tokens = (
            max(self.max_token, configured_hard_limit)
            if configured_hard_limit > 0
            else 0
        )
        self.summary_data = []
        self.stage_summary_data = []
        self.batch_summary_data = []
        self._last_stage_index = None
        self._last_stage_info = {
            "stage_index": None,
            "stage_title": "",
            "section_index": "",
            "progress": "",
        }
        self._current_stage_index = None
        self._current_stage_title = ""
        self._current_section_index = ""
        self._current_progress = ""
        self._stage_batch_acc = []  # list of dict summaries for current stage
        self._toolmsg_summarized_ids = set()
        self._is_reset_chat = False
        self._is_reset_force = False
        self.system_message = None
        self.model = model
        self.main_model = main_model or model
        self.structured_summary = structured_summary
        self.hierarchical_summary = hierarchical_summary
        self.enable_long_term_memory = enable_long_term_memory
        self.long_term_memory = long_term_memory
        self.long_term_memory_cfg = long_term_memory_cfg or {}
        if hasattr(self.long_term_memory_cfg, "as_dict"):
            self.long_term_memory_cfg = self.long_term_memory_cfg.as_dict()
        elif not isinstance(self.long_term_memory_cfg, dict):
            self.long_term_memory_cfg = {}
        self.enable_failure_aware_context = enable_failure_aware_context
        self.failure_aware_context_cfg = failure_aware_context_cfg or {}
        if hasattr(self.failure_aware_context_cfg, "as_dict"):
            self.failure_aware_context_cfg = self.failure_aware_context_cfg.as_dict()
        elif not isinstance(self.failure_aware_context_cfg, dict):
            self.failure_aware_context_cfg = {}
        self.enable_stage_state_package = bool(enable_stage_state_package)
        self.stage_state_package_cfg = stage_state_package_cfg or {}
        if hasattr(self.stage_state_package_cfg, "as_dict"):
            self.stage_state_package_cfg = self.stage_state_package_cfg.as_dict()
        elif not isinstance(self.stage_state_package_cfg, dict):
            self.stage_state_package_cfg = {}
        self.enable_observation_masking = bool(enable_observation_masking)
        self.observation_masking_cfg = observation_masking_cfg or {}
        if hasattr(self.observation_masking_cfg, "as_dict"):
            self.observation_masking_cfg = self.observation_masking_cfg.as_dict()
        elif not isinstance(self.observation_masking_cfg, dict):
            self.observation_masking_cfg = {}
        self.stage_state_package: Dict[str, Any] = {}
        self._last_observation_mask_fingerprint = ""
        self.observation_mask_statistics = {
            "calls": 0,
            "masked_observations": 0,
            "original_chars": 0,
            "retained_chars": 0,
            "saved_chars": 0,
        }
        self.vagent = vagent
        self.dut_name = dut_name
        self.summary_model_source = summary_model_source
        self.llm_input_dumper = llm_input_dumper
        self.summary_model_cfg = summary_model_cfg or {}
        if hasattr(self.summary_model_cfg, "as_dict"):
            self.summary_model_cfg = self.summary_model_cfg.as_dict()
        elif not isinstance(self.summary_model_cfg, dict):
            self.summary_model_cfg = {}
        self.summary_runtime_cfg = {
            "fallback_to_main": bool(self.summary_model_cfg.get("fallback_to_main", True)),
            "max_context_tokens": int(self.summary_model_cfg.get("max_context_tokens", 0) or 0),
            "stats_hook": self._record_summary_runtime_stats,
        }
        self.summary_statistics = {
            "calls": 0,
            "fallbacks": 0,
            "repair_used": 0,
            "quality_guard_fail": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms_total": 0,
            "latency_ms_avg": 0.0,
            "last": {},
            "model_source": self.summary_model_source,
            "configured_enabled": bool(self.summary_model_cfg.get("enabled", False)),
        }
        self.arbit_summary_data = None
        self.failure_context_data = []
        self._toolmsg_runtime_seen_ids = set()
        self.memory_hit_statistics = {
            "injections": 0,
            "injected_items": 0,
            "useful_hits": 0,
            "retrieval_useful_hits": 0,
            "prefetch_useful_hits": 0,
            "fallback_useful_hits": 0,
            "stale_hits": 0,
            "memory_pollution": 0,
            "retrieval_pollution": 0,
            "prefetch_pollution": 0,
            "fallback_pollution": 0,
            "injections_by_source": {},
            "last": {},
        }
        self._memory_eval_pending = []
        self._memory_eval_window = int(self.long_term_memory_cfg.get("useful_hit_window", 3) or 3)
        self._last_runtime_failure_count = None

    @staticmethod
    def _trim_to_token_budget(messages: List, max_tokens: int) -> List:
        if max_tokens <= 0 or count_tokens_approximately(messages) <= max_tokens:
            return list(messages)
        source = list(messages)
        system_prefix = [source[0]] if source and isinstance(source[0], SystemMessage) else []
        body = source[1:] if system_prefix else source
        marker = HumanMessage(
            content=(
                "[CONTEXT_WINDOW_TRIMMED]\n"
                "Older messages were omitted only to satisfy the model input limit. "
                "Use the retained recent tool observations and re-read a file if required."
            )
        )
        fixed_prefix = system_prefix + [marker]
        body_budget = max(256, max_tokens - count_tokens_approximately(fixed_prefix))
        tail = list(trim_messages(
            body,
            max_tokens=body_budget,
            token_counter=count_tokens_approximately,
            strategy="last",
            include_system=False,
            start_on=("human", "ai"),
            allow_partial=False,
        ))
        if not tail and body:
            last_content = str(getattr(body[-1], "content", "") or "")
            char_budget = max(256, body_budget * 3)
            if len(last_content) > char_budget:
                half = max(128, (char_budget - 80) // 2)
                last_content = last_content[:half] + "\n...[TRUNCATED]...\n" + last_content[-half:]
            tail = [HumanMessage(content="[LATEST_OBSERVATION]\n" + last_content)]
        bounded = fixed_prefix + tail
        if count_tokens_approximately(bounded) > max_tokens:
            bounded = fixed_prefix
        return bounded

    def _bound_summary_inputs(self, messages: List) -> List:
        if self.hard_max_tokens <= 0:
            return list(messages)
        budget = max(1024, self.hard_max_tokens - self.max_summary_tokens - 512)
        trimmed = self._trim_to_token_budget(messages, budget)
        if len(trimmed) != len(messages):
            warning(
                f"Summary input hard-trimmed from {len(messages)} to {len(trimmed)} messages "
                f"to stay within approximately {budget} tokens."
            )
        return trimmed

    def _enforce_hard_input_budget(self, messages: List) -> tuple[List, bool]:
        if self.hard_max_tokens <= 0:
            return list(messages), False
        current = count_tokens_approximately(messages)
        if current <= self.hard_max_tokens:
            return list(messages), False
        trimmed = self._trim_to_token_budget(messages, self.hard_max_tokens)
        final_size = count_tokens_approximately(trimmed)
        warning(
            f"LLM input hard cap applied: approximately {current} -> {final_size} tokens "
            f"(limit={self.hard_max_tokens})."
        )
        return trimmed, True

    def _record_summary_runtime_stats(self, payload: dict) -> None:
        if not isinstance(payload, dict):
            return
        if payload.get("kind") == "structured_repair":
            return
        self.summary_statistics["calls"] += 1
        self.summary_statistics["fallbacks"] += int(bool(payload.get("summary_fallback", False)))
        self.summary_statistics["repair_used"] += int(bool(payload.get("summary_repair_used", False)))
        self.summary_statistics["quality_guard_fail"] += int(bool(payload.get("summary_quality_guard_fail", False)))
        self.summary_statistics["input_tokens"] += int(payload.get("summary_input_tokens", 0) or 0)
        self.summary_statistics["output_tokens"] += int(payload.get("summary_output_tokens", 0) or 0)
        self.summary_statistics["latency_ms_total"] += int(payload.get("summary_latency_ms", 0) or 0)
        calls = max(1, int(self.summary_statistics["calls"]))
        self.summary_statistics["latency_ms_avg"] = round(self.summary_statistics["latency_ms_total"] / calls, 2)
        self.summary_statistics["last"] = {
            "kind": payload.get("kind", ""),
            "summary_model_name": payload.get("summary_model_name", ""),
            "summary_latency_ms": int(payload.get("summary_latency_ms", 0) or 0),
            "summary_fallback": bool(payload.get("summary_fallback", False)),
            "summary_fallback_reason": payload.get("summary_fallback_reason", ""),
            "summary_input_tokens": int(payload.get("summary_input_tokens", 0) or 0),
            "summary_output_tokens": int(payload.get("summary_output_tokens", 0) or 0),
            "summary_repair_used": bool(payload.get("summary_repair_used", False)),
            "summary_quality_guard_fail": bool(payload.get("summary_quality_guard_fail", False)),
        }

    def get_summary_statistics(self) -> dict:
        return dict(self.summary_statistics)

    def get_memory_hit_statistics(self) -> dict:
        stats = dict(self.memory_hit_statistics)
        injections = max(1, int(stats.get("injections", 0) or 0))
        source_injections = stats.get("injections_by_source", {}) or {}
        total_memory_hits = int(source_injections.get("retrieval", 0) or 0) + int(source_injections.get("recent_fallback", 0) or 0)
        total_memory_hits += int(source_injections.get("prefetch", 0) or 0)
        useful = int(stats.get("useful_hits", 0) or 0)
        retrieval_useful = int(stats.get("retrieval_useful_hits", 0) or 0)
        prefetch_useful = int(stats.get("prefetch_useful_hits", 0) or 0)
        fallback_useful = int(stats.get("fallback_useful_hits", 0) or 0)
        stale = int(stats.get("stale_hits", 0) or 0)
        pollution = int(stats.get("memory_pollution", 0) or 0)
        retrieval_pollution = int(stats.get("retrieval_pollution", 0) or 0)
        prefetch_pollution = int(stats.get("prefetch_pollution", 0) or 0)
        fallback_pollution = int(stats.get("fallback_pollution", 0) or 0)
        stats["useful_hit_rate"] = round(useful / max(1, total_memory_hits), 4)
        stats["retrieval_useful_hit_rate"] = round(retrieval_useful / max(1, int(source_injections.get("retrieval", 0) or 0)), 4)
        stats["prefetch_useful_hit_rate"] = round(prefetch_useful / max(1, int(source_injections.get("prefetch", 0) or 0)), 4)
        stats["fallback_useful_hit_rate"] = round(fallback_useful / max(1, int(source_injections.get("recent_fallback", 0) or 0)), 4)
        stats["stale_hit_rate"] = round(stale / max(1, int(stats.get("injected_items", 0) or 0)), 4)
        stats["memory_pollution_rate"] = round(pollution / injections, 4)
        stats["retrieval_pollution_rate"] = round(retrieval_pollution / max(1, int(source_injections.get("retrieval", 0) or 0)), 4)
        stats["prefetch_pollution_rate"] = round(prefetch_pollution / max(1, int(source_injections.get("prefetch", 0) or 0)), 4)
        stats["fallback_pollution_rate"] = round(fallback_pollution / max(1, int(source_injections.get("recent_fallback", 0) or 0)), 4)
        return stats

    def note_memory_injection(self, memories: List[Dict], query: str = "", source: str = "retrieval", stage_index: Optional[int] = None):
        if not isinstance(memories, list) or not memories:
            return
        stage_idx = stage_index if isinstance(stage_index, int) else self._current_stage_index
        failure_hashes = []
        failure_signatures = set()
        stale_hits = 0
        for item in memories:
            if not isinstance(item, dict):
                continue
            if bool(item.get("resolved", False)) or float(item.get("stale_score", 0.0) or 0.0) >= 0.5:
                stale_hits += 1
            content = item.get("content") if isinstance(item.get("content"), dict) else {}
            if str(content.get("status", "")).lower() != "failure":
                continue
            if item.get("hash"):
                failure_hashes.append(item.get("hash"))
            for cp in content.get("failed_checkpoints_top", []) or []:
                failure_signatures.add(f"cp:{cp}")
            for case in content.get("failed_cases_top", []) or []:
                failure_signatures.add(f"case:{case}")
        self.memory_hit_statistics["injections"] += 1
        self.memory_hit_statistics["injected_items"] += len(memories)
        self.memory_hit_statistics["stale_hits"] += stale_hits
        source_counts = self.memory_hit_statistics.get("injections_by_source", {})
        source_counts[source] = int(source_counts.get(source, 0) or 0) + 1
        self.memory_hit_statistics["injections_by_source"] = source_counts
        self.memory_hit_statistics["last"] = {
            "query": query,
            "source": source,
            "stage_index": stage_idx,
            "memory_count": len(memories),
            "failure_memory_count": len(failure_hashes),
        }
        if failure_hashes:
            self._memory_eval_pending.append(
                {
                    "hashes": failure_hashes,
                    "signatures": sorted(failure_signatures),
                    "stage_index": stage_idx,
                    "source": source,
                    "query": query,
                    "remaining": self._memory_eval_window,
                    "baseline_failed": self._last_runtime_failure_count,
                }
            )

    def _evaluate_memory_injections(self, event: Dict) -> None:
        if not self._memory_eval_pending or not isinstance(event, dict):
            return
        stage_index = (event.get("stage_info") or {}).get("stage_index")
        current_failed = int((event.get("tests") or {}).get("failed", 0) or 0)
        current_signatures = set()
        for cp in event.get("failed_checkpoints_top", []) or []:
            current_signatures.add(f"cp:{cp}")
        for case in event.get("failed_cases_top", []) or []:
            current_signatures.add(f"case:{case}")

        still_pending = []
        for pending in self._memory_eval_pending:
            if pending.get("stage_index") != stage_index:
                still_pending.append(pending)
                continue
            useful = False
            reason = ""
            injected_signatures = set(pending.get("signatures") or [])
            baseline_failed = pending.get("baseline_failed")
            if str(event.get("status", "")).lower() == "success":
                useful = True
                reason = "next_batch_success"
            elif baseline_failed is not None and current_failed < int(baseline_failed):
                useful = True
                reason = "failed_count_reduced"
            elif injected_signatures and current_signatures and not (injected_signatures & current_signatures):
                useful = True
                reason = "failure_signature_not_repeated"

            if useful:
                self.memory_hit_statistics["useful_hits"] += 1
                if pending.get("source") == "retrieval":
                    self.memory_hit_statistics["retrieval_useful_hits"] += 1
                elif pending.get("source") == "prefetch":
                    self.memory_hit_statistics["prefetch_useful_hits"] += 1
                elif pending.get("source") == "recent_fallback":
                    self.memory_hit_statistics["fallback_useful_hits"] += 1
                if self.enable_long_term_memory and self.long_term_memory:
                    self.long_term_memory.mark_useful_hit(
                        pending.get("hashes", []),
                        useful=True,
                        reason=reason,
                        hit_type=pending.get("source", "retrieval"),
                    )
                continue

            pending["remaining"] = int(pending.get("remaining", 1) or 1) - 1
            if pending["remaining"] <= 0:
                self.memory_hit_statistics["memory_pollution"] += 1
                if pending.get("source") == "retrieval":
                    self.memory_hit_statistics["retrieval_pollution"] += 1
                elif pending.get("source") == "prefetch":
                    self.memory_hit_statistics["prefetch_pollution"] += 1
                elif pending.get("source") == "recent_fallback":
                    self.memory_hit_statistics["fallback_pollution"] += 1
                if self.enable_long_term_memory and self.long_term_memory:
                    self.long_term_memory.mark_useful_hit(
                        pending.get("hashes", []),
                        useful=False,
                        reason="window_expired",
                        hit_type=pending.get("source", "retrieval"),
                    )
                continue
            still_pending.append(pending)
        self._memory_eval_pending = still_pending
        self._last_runtime_failure_count = current_failed

    def set_stage_context(self, stage_index: Optional[int], stage_title: str, section_index: str, progress: str):
        """Update current stage context for memory tagging and stage summary."""
        if isinstance(stage_index, int) and isinstance(self._current_stage_index, int) and stage_index != self._current_stage_index:
            self._finalize_stage_rollup(self._current_stage_snapshot())
        if isinstance(stage_index, int):
            self._current_stage_index = stage_index
        self._current_stage_title = stage_title or self._current_stage_title
        self._current_section_index = section_index or self._current_section_index
        self._current_progress = progress or self._current_progress
        if self._last_stage_index is None and isinstance(stage_index, int):
            self._last_stage_index = stage_index
            self._last_stage_info = {
                "stage_index": stage_index,
                "stage_title": self._current_stage_title,
                "section_index": self._current_section_index,
                "progress": self._current_progress,
            }

    def update_stage_state_package(self, package: Dict[str, Any]) -> None:
        if not self.enable_stage_state_package or not isinstance(package, dict):
            return
        self.stage_state_package = dict(package)

    def _build_stage_state_prefix(self) -> List[AIMessage]:
        if not self.enable_stage_state_package or not self.stage_state_package:
            return []
        content = "STAGE_STATE_PACKAGE:\n" + yaml.safe_dump(
            self.stage_state_package,
            allow_unicode=True,
            sort_keys=False,
        )
        max_chars = max(1000, int(self.stage_state_package_cfg.get("max_chars", 6000) or 6000))
        if len(content) > max_chars:
            content = content[:max_chars] + "\npackage_truncated: true\n"
        return [AIMessage(content=content)]

    def _mask_observations(self, messages: List) -> List:
        if not self.enable_observation_masking:
            return list(messages)
        configured_names = self.observation_masking_cfg.get(
            "tool_names", ["RunTestCases", "Check", "Complete"]
        ) or []
        tool_names = {str(name) for name in configured_names}
        keep_recent = max(0, int(
            self.observation_masking_cfg.get("keep_recent_verifier_observations", 2) or 0
        ))
        min_chars = max(0, int(self.observation_masking_cfg.get("min_chars", 400) or 0))
        candidate_indices = [
            index
            for index, message in enumerate(messages)
            if isinstance(message, ToolMessage)
            and str(getattr(message, "name", "") or "") in tool_names
            and not str(getattr(message, "content", "") or "").startswith("OBSERVATION_MASKED:")
        ]
        keep = set(candidate_indices[-keep_recent:]) if keep_recent else set()
        masked = []
        masked_meta = []
        for index, message in enumerate(messages):
            if index not in candidate_indices or index in keep:
                masked.append(message)
                continue
            content = str(message.content or "")
            if len(content) < min_chars:
                masked.append(message)
                continue
            digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]
            replacement = (
                "OBSERVATION_MASKED:\n"
                "reason: superseded_by_stage_state_package\n"
                f"tool: {getattr(message, 'name', '')}\n"
                f"original_chars: {len(content)}\n"
                f"sha256: {digest}\n"
                "recovery: rerun the minimal target or read the persisted report if exact details are needed\n"
            )
            masked.append(ToolMessage(
                content=replacement,
                tool_call_id=message.tool_call_id,
                name=getattr(message, "name", None),
                status=getattr(message, "status", None),
                id=getattr(message, "id", None),
            ))
            masked_meta.append({
                "id": str(getattr(message, "id", "") or ""),
                "tool": str(getattr(message, "name", "") or ""),
                "original_chars": len(content),
                "retained_chars": len(replacement),
            })
        if masked_meta:
            original_chars = sum(item["original_chars"] for item in masked_meta)
            retained_chars = sum(item["retained_chars"] for item in masked_meta)
            fingerprint = hashlib.sha256(
                json.dumps(masked_meta, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16]
            self.observation_mask_statistics = {
                "calls": int(self.observation_mask_statistics.get("calls", 0)) + 1,
                "masked_observations": len(masked_meta),
                "original_chars": original_chars,
                "retained_chars": retained_chars,
                "saved_chars": max(0, original_chars - retained_chars),
            }
            if fingerprint != self._last_observation_mask_fingerprint:
                self._last_observation_mask_fingerprint = fingerprint
                info(
                    "[context_upgrade][observation_masking] "
                    f"masked={len(masked_meta)} chars={original_chars}->{retained_chars}"
                )
                vagent = getattr(self, "vagent", None)
                if vagent is not None and hasattr(vagent, "record_structured_event"):
                    vagent.record_structured_event("observation_masking", {
                        "masked_observations": len(masked_meta),
                        "original_chars": original_chars,
                        "retained_chars": retained_chars,
                        "saved_chars": max(0, original_chars - retained_chars),
                        "tools": sorted({item["tool"] for item in masked_meta}),
                    })
        return masked

    def _bound_oversized_tool_observations(self, messages: List) -> List:
        """Deterministically bound any single tool observation before summary.

        Observation masking handles repeated verifier outputs, but a latest
        PathList/ReadTextFile/custom-tool response can still exceed the whole
        model context. Keep both ends so command metadata and terminal
        diagnostics remain visible; exact details can be requested narrowly.
        """
        if not self.enable_observation_masking:
            return list(messages)
        max_chars = max(2000, int(
            self.observation_masking_cfg.get("max_tool_observation_chars", 24000) or 24000
        ))
        bounded = []
        bounded_meta = []
        for message in messages:
            if not isinstance(message, ToolMessage):
                bounded.append(message)
                continue
            content = str(getattr(message, "content", "") or "")
            if len(content) <= max_chars:
                bounded.append(message)
                continue
            digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]
            marker = (
                "\n\nOVERSIZED_TOOL_OBSERVATION_TRUNCATED:\n"
                f"tool: {getattr(message, 'name', '')}\n"
                f"original_chars: {len(content)}\n"
                f"sha256: {digest}\n"
                "recovery: rerun the tool with a narrower path, target, depth, or line range\n\n"
            )
            available = max(0, max_chars - len(marker))
            head_chars = int(available * 0.6)
            tail_chars = available - head_chars
            replacement = content[:head_chars] + marker
            if tail_chars:
                replacement += content[-tail_chars:]
            bounded.append(ToolMessage(
                content=replacement,
                tool_call_id=message.tool_call_id,
                name=getattr(message, "name", None),
                status=getattr(message, "status", None),
                id=getattr(message, "id", None),
            ))
            bounded_meta.append({
                "tool": str(getattr(message, "name", "") or ""),
                "original_chars": len(content),
                "retained_chars": len(replacement),
                "sha256": digest,
            })
        if bounded_meta:
            info(
                "[context_upgrade][observation_size_guard] "
                f"bounded={len(bounded_meta)} "
                f"chars={sum(item['original_chars'] for item in bounded_meta)}"
                f"->{sum(item['retained_chars'] for item in bounded_meta)}"
            )
            vagent = getattr(self, "vagent", None)
            if vagent is not None and hasattr(vagent, "record_structured_event"):
                vagent.record_structured_event("observation_size_guard", {
                    "bounded_observations": len(bounded_meta),
                    "max_tool_observation_chars": max_chars,
                    "items": bounded_meta[:10],
                })
        return bounded

    def reset_chat(self, force=False):
        self._is_reset_chat = True
        self._is_reset_force = bool(force)
        return self

    def set_system_message(self, system_message: SystemMessage):
        self.system_message = system_message
        return self

    def _make_stage_context_message(self) -> Optional[AIMessage]:
        idx = self._current_stage_index
        if not isinstance(idx, int):
            idx = self._last_stage_info.get("stage_index")
        if not isinstance(idx, int):
            return None
        stage_info = {
            "stage_index": idx,
            "stage_title": self._current_stage_title or self._last_stage_info.get("stage_title", ""),
            "section_index": self._current_section_index or self._last_stage_info.get("section_index", ""),
            "progress": self._current_progress or self._last_stage_info.get("progress", ""),
            "constraints": "",
        }
        payload = {"stage_info": stage_info}
        return AIMessage(content="STAGE_SUMMARY:\n" + yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))

    def _failure_buffer_limit(self) -> int:
        return int(self.failure_aware_context_cfg.get("buffer_limit", 12) or 12)

    def _failure_inject_limit(self) -> int:
        return int(self.failure_aware_context_cfg.get("max_failure_items", 2) or 2)

    def _success_inject_limit(self) -> int:
        return int(self.failure_aware_context_cfg.get("max_success_items", 1) or 1)

    def _prefer_same_stage(self) -> bool:
        return bool(self.failure_aware_context_cfg.get("prefer_same_stage", True))

    def _current_stage_snapshot(self) -> Dict[str, object]:
        return {
            "stage_index": self._current_stage_index,
            "stage_title": self._current_stage_title or self._last_stage_info.get("stage_title", ""),
            "section_index": self._current_section_index or self._last_stage_info.get("section_index", ""),
            "progress": self._current_progress or self._last_stage_info.get("progress", ""),
        }

    def _record_failure_context(self, item: Dict) -> None:
        if not isinstance(item, dict):
            return
        self.failure_context_data.append(item)
        self.failure_context_data = self.failure_context_data[-self._failure_buffer_limit():]

    def _build_batch_event(self, batch: Dict) -> Optional[Dict]:
        if not isinstance(batch, dict):
            return None
        tests = batch.get("tests") or {}
        total = int(tests.get("total", 0) or 0)
        passed = int(tests.get("passed", 0) or 0)
        failed = int(tests.get("failed", 0) or 0)
        failed_cases = [str(x) for x in (batch.get("failed_test_cases_top") or [])[:10]]
        failed_checkpoints = [str(x) for x in (batch.get("failed_check_points_top") or [])[:12]]
        if total <= 0 and not failed_cases and not failed_checkpoints:
            return None
        status = "failure" if failed > 0 or failed_cases or failed_checkpoints else "success"
        stage_info = self._current_stage_snapshot()
        summary = (
            f"batch {failed}/{total} failed; "
            f"failed_cp={', '.join(failed_checkpoints[:4]) if failed_checkpoints else 'none'}; "
            f"failed_cases={', '.join(failed_cases[:3]) if failed_cases else 'none'}"
        )
        action = ""
        if status == "failure":
            if failed_checkpoints:
                action = "优先检查失败检查点对应测试与 bug 分析文档是否一致"
            else:
                action = "优先检查失败测试对应实现与断言是否一致"
        return {
            "kind": "batch",
            "status": status,
            "stage_info": stage_info,
            "tests": {"total": total, "passed": passed, "failed": failed},
            "failed_cases_top": failed_cases,
            "failed_checkpoints_top": failed_checkpoints,
            "summary": summary,
            "root_cause": "",
            "action": action,
        }

    def _build_stage_rollup(self, stage_info: Dict[str, object], batch_items: List[Dict]) -> Optional[Dict]:
        if not batch_items:
            return None
        total_runs = 0
        total_tests = 0
        total_failed = 0
        failed_cases = []
        failed_checkpoints = []
        for item in batch_items:
            tests = item.get("tests") or {}
            total_runs += 1
            total_tests += int(tests.get("total", 0) or 0)
            total_failed += int(tests.get("failed", 0) or 0)
            failed_cases.extend(item.get("failed_cases_top") or [])
            failed_checkpoints.extend(item.get("failed_checkpoints_top") or [])
        failed_cases = list(dict.fromkeys([str(x) for x in failed_cases if x]))[:10]
        failed_checkpoints = list(dict.fromkeys([str(x) for x in failed_checkpoints if x]))[:12]
        status = "failure" if total_failed > 0 or failed_checkpoints else "success"
        summary = (
            f"stage runs={total_runs}; tests={total_tests}; failed={total_failed}; "
            f"failed_cp={', '.join(failed_checkpoints[:4]) if failed_checkpoints else 'none'}"
        )
        action = ""
        if status == "failure":
            action = "复核 bug_analysis 与失败检查点映射，避免在后续阶段重复全量回归"
        return {
            "kind": "stage",
            "status": status,
            "stage_info": dict(stage_info or {}),
            "tests": {"total": total_tests, "passed": max(0, total_tests - total_failed), "failed": total_failed},
            "failed_cases_top": failed_cases,
            "failed_checkpoints_top": failed_checkpoints,
            "summary": summary,
            "root_cause": "",
            "action": action,
        }

    def _finalize_stage_rollup(self, stage_info: Dict[str, object]) -> None:
        stage_event = self._build_stage_rollup(stage_info, self._stage_batch_acc)
        self._stage_batch_acc = []
        if stage_event is None:
            return
        self._record_failure_context(stage_event)
        if self.enable_long_term_memory and self.long_term_memory:
            self.long_term_memory.save(
                meta={
                    "type": "stage",
                    "dut": self.dut_name,
                    "stage_index": stage_event.get("stage_info", {}).get("stage_index"),
                    "stage_title": stage_event.get("stage_info", {}).get("stage_title", ""),
                    "section_index": stage_event.get("stage_info", {}).get("section_index", ""),
                    "timestamp": time.time(),
                },
                content=stage_event,
            )

    def _collect_runtime_events(self, llm_msgs: List) -> None:
        for msg in llm_msgs:
            if not isinstance(msg, ToolMessage):
                continue
            if getattr(msg, "name", "") != "RunTestCases":
                continue
            if msg.id in self._toolmsg_runtime_seen_ids:
                continue
            self._toolmsg_runtime_seen_ids.add(msg.id)
            parsed = _parse_summary_payload(msg.content) or _safe_load_yaml(msg.content)
            if not isinstance(parsed, dict):
                continue
            batch = _summarize_run_testcases_yaml(parsed)
            event = self._build_batch_event(batch)
            if event is None:
                continue
            self._evaluate_memory_injections(event)
            self._last_runtime_failure_count = _coerce_count(
                (event.get("tests") or {}).get("failed", 0)
            )
            self._stage_batch_acc.append(event)
            if self.enable_failure_aware_context:
                self._record_failure_context(event)
            if self.enable_long_term_memory and self.long_term_memory:
                self.long_term_memory.save(
                    meta={
                        "type": "batch",
                        "dut": self.dut_name,
                        "stage_index": event.get("stage_info", {}).get("stage_index"),
                        "stage_title": event.get("stage_info", {}).get("stage_title", ""),
                        "section_index": event.get("stage_info", {}).get("section_index", ""),
                        "timestamp": time.time(),
                    },
                    content=event,
                )

    def _build_turn_memory_entry(self, summary_text: str) -> Optional[Dict]:
        if not isinstance(summary_text, str) or not summary_text.strip():
            return None
        parsed = _try_parse_structured_json(summary_text) or _load_json_maybe(summary_text)
        if not isinstance(parsed, dict):
            return None
        stage_info = parsed.get("stage_info") if isinstance(parsed.get("stage_info"), dict) else {}
        test_report = parsed.get("test_report") if isinstance(parsed.get("test_report"), dict) else {}
        bug_tracking = parsed.get("bug_tracking") if isinstance(parsed.get("bug_tracking"), dict) else {}
        next_info = parsed.get("next") if isinstance(parsed.get("next"), dict) else {}
        failed = _coerce_count(test_report.get("failed", 0))
        failed_cases_top = []
        for item in test_report.get("failed_cases_top") or []:
            if isinstance(item, dict) and item.get("case"):
                failed_cases_top.append(str(item.get("case")))
            elif item:
                failed_cases_top.append(str(item))
        failed_checkpoints_top = [str(x) for x in (test_report.get("failed_checkpoints_top") or [])[:12]]
        root_causes = []
        for item in bug_tracking.get("root_cause_hypotheses_top") or []:
            if isinstance(item, dict) and item.get("hypothesis"):
                root_causes.append(str(item.get("hypothesis")))
            elif item:
                root_causes.append(str(item))
        action = ""
        for key in ("todo", "blockers", "notes"):
            value = next_info.get(key)
            if isinstance(value, str) and value.strip():
                action = value.strip()
                break
        if failed <= 0 and not failed_cases_top and not failed_checkpoints_top and not root_causes and not action:
            return None
        return {
            "kind": "turn",
            "status": "failure" if failed > 0 or failed_cases_top or failed_checkpoints_top else "success",
            "stage_info": {
                "stage_index": stage_info.get("stage_index", self._current_stage_index),
                "stage_title": stage_info.get("stage_title", self._current_stage_title),
                "section_index": stage_info.get("section_index", self._current_section_index),
                "progress": stage_info.get("progress", self._current_progress),
            },
            "tests": {
                "total": int(test_report.get("total", 0) or 0),
                "passed": int(test_report.get("passed", 0) or 0),
                "failed": failed,
            },
            "failed_cases_top": failed_cases_top[:10],
            "failed_checkpoints_top": failed_checkpoints_top[:12],
            "summary": f"turn summary; failed={failed}; checkpoints={', '.join(failed_checkpoints_top[:4]) if failed_checkpoints_top else 'none'}",
            "root_cause": " | ".join(root_causes[:3]),
            "action": action,
        }

    def _compact_failure_prompt_item(self, item: Dict) -> Dict:
        stage_info = item.get("stage_info") if isinstance(item.get("stage_info"), dict) else {}
        return {
            "kind": item.get("kind", ""),
            "stage_index": stage_info.get("stage_index"),
            "stage_title": stage_info.get("stage_title", ""),
            "status": item.get("status", ""),
            "summary": item.get("summary", ""),
            "failed_checkpoints_top": item.get("failed_checkpoints_top", [])[:6],
            "failed_cases_top": item.get("failed_cases_top", [])[:4],
            "root_cause": item.get("root_cause", ""),
            "action": item.get("action", ""),
        }

    def _build_failure_context_prefix(self) -> List[AIMessage]:
        failure_items = []
        non_failure_items = []
        items = list(self.failure_context_data)
        if self._prefer_same_stage():
            same_stage = []
            other_stage = []
            for item in items:
                stage_info = item.get("stage_info") if isinstance(item.get("stage_info"), dict) else {}
                if stage_info.get("stage_index") == self._current_stage_index:
                    same_stage.append(item)
                else:
                    other_stage.append(item)
            items = same_stage + other_stage
        for item in reversed(items):
            status = str(item.get("status", "")).lower()
            if status == "failure" and len(failure_items) < self._failure_inject_limit():
                failure_items.append(self._compact_failure_prompt_item(item))
            elif status != "failure" and len(non_failure_items) < self._success_inject_limit():
                non_failure_items.append(self._compact_failure_prompt_item(item))
            if len(failure_items) >= self._failure_inject_limit() and len(non_failure_items) >= self._success_inject_limit():
                break

        prefix = []
        if failure_items:
            prefix.append(
                AIMessage(
                    content="FAILURE_CONTEXT:\n"
                    + yaml.safe_dump(list(reversed(failure_items)), allow_unicode=True, sort_keys=False)
                )
            )
        if non_failure_items:
            prefix.append(
                AIMessage(
                    content="SUCCESS_CONTEXT:\n"
                    + yaml.safe_dump(list(reversed(non_failure_items)), allow_unicode=True, sort_keys=False)
                )
            )
        return prefix

    def _maybe_update_hierarchy(self, llm_msgs: List) -> List:
        """
        Optionally compress large tool outputs (Batch Summary) and maintain Stage Summary cache.
        This DOES NOT mutate agent state, only affects llm_input_messages.
        """
        if not self.hierarchical_summary:
            return llm_msgs

        stage_index, stage_title, section_index, progress = _extract_stage_info_from_tips(llm_msgs)
        if stage_index is None:
            stage_index = self._current_stage_index
            stage_title = self._current_stage_title
            section_index = self._current_section_index
            progress = self._current_progress
        # detect stage transition
        if stage_index is not None and self._last_stage_index is not None and stage_index != self._last_stage_index:
            if self._stage_batch_acc:
                # build a compact stage summary (deterministic, no extra LLM call)
                failed_cp = []
                total_runs = 0
                for b in self._stage_batch_acc:
                    total_runs += 1
                    failed_cp.extend(b.get("failed_check_points_top") or [])
                # de-dup but keep order
                seen = set()
                failed_cp_uniq = []
                for x in failed_cp:
                    if x in seen:
                        continue
                    seen.add(x)
                    failed_cp_uniq.append(x)
                stage_summary = {
                    "stage_info": {
                        "stage_index": self._last_stage_info.get("stage_index"),
                        "stage_title": self._last_stage_info.get("stage_title"),
                        "section_index": self._last_stage_info.get("section_index"),
                        "progress": self._last_stage_info.get("progress"),
                        "constraints": "",
                    },
                    "batch_runs": total_runs,
                    "failed_check_points_top": failed_cp_uniq[:20],
                }
                info(
                    f"[context_upgrade][hierarchical][stage_summary] stage_index={self._last_stage_index} "
                    f"batch_runs={total_runs} failed_check_points_top={failed_cp_uniq[:6]}"
                )
                self.stage_summary_data.append(AIMessage(content="STAGE_SUMMARY:\n" + yaml.safe_dump(stage_summary, allow_unicode=True, sort_keys=False)))
                self.stage_summary_data = self.stage_summary_data[-3:]
            self._stage_batch_acc = []

        if stage_index is not None:
            if self._last_stage_index is None or stage_index != self._last_stage_index:
                self._last_stage_index = stage_index
            self._last_stage_info = {
                "stage_index": stage_index,
                "stage_title": stage_title or "",
                "section_index": section_index or "",
                "progress": progress or "",
            }

        # compress RunTestCases tool outputs
        new_msgs: List = []
        for m in llm_msgs:
            if isinstance(m, ToolMessage) and getattr(m, "name", "") == "RunTestCases":
                if m.id not in self._toolmsg_summarized_ids and isinstance(m.content, str) and len(m.content) > 4000:
                    tool_dict = _safe_load_yaml(m.content) or {}
                    batch = _summarize_run_testcases_yaml(tool_dict) if isinstance(tool_dict, dict) else {"raw": "unparsable"}
                    self._stage_batch_acc.append(batch)
                    self._toolmsg_summarized_ids.add(m.id)
                    # keep the tool message but shrink the payload
                    new_content = "BATCH_SUMMARY(RunTestCases):\n" + yaml.safe_dump(batch, allow_unicode=True, sort_keys=False)
                    info(
                        f"[context_upgrade][hierarchical][batch_summary] stage_index={stage_index} "
                        f"tool_msg_id={m.id} content_len={len(m.content)} -> {len(new_content)} "
                        f"tests={batch.get('tests', {})}"
                    )
                    new_msgs.append(ToolMessage(content=new_content, tool_call_id=m.tool_call_id, name=m.name, status=getattr(m, "status", None)))
                    # also keep a tiny rolling cache as prefix (survives tail trimming)
                    self.batch_summary_data.append(AIMessage(content="BATCH_SUMMARY_CACHE:\n" + yaml.safe_dump(batch, allow_unicode=True, sort_keys=False)))
                    self.batch_summary_data = self.batch_summary_data[-3:]
                    continue
            new_msgs.append(m)
        return new_msgs

    def __call__(self, state):
        fix_tool_call_args(state)
        messages = state["messages"]
        role_info = messages[:1]
        llm_input_msgs = messages[1:]
        if self.system_message and (not role_info or role_info[0].type != "system"):
            warning("System message is missing, adding it back to the LLM input context.")
            role_info = [self.system_message]
            llm_input_msgs = messages
        reset_ret = {}
        if self._is_reset_chat:
            self.summary_data = []
            self.stage_summary_data = []
            self.batch_summary_data = []
            self._stage_batch_acc = []
            self._toolmsg_summarized_ids.clear()
            last_human = next(
                (index for index in range(len(llm_input_msgs) - 1, -1, -1)
                 if isinstance(llm_input_msgs[index], HumanMessage)),
                0,
            )
            original_llm_input = llm_input_msgs
            tail_after_reset = original_llm_input[last_human:]
            if self._is_reset_force and tail_after_reset:
                tail_after_reset = tail_after_reset[:1]
            removed_after = last_human + len(tail_after_reset)
            removed_messages = original_llm_input[:last_human] + original_llm_input[removed_after:]
            reset_ret["messages"] = [
                RemoveMessage(id=msg.id)
                for msg in removed_messages
            ]
            llm_input_msgs = tail_after_reset
            warning(
                f"Chat reset [force={self._is_reset_force}], kept "
                f"{len(tail_after_reset)} message(s) from the latest human turn."
            )
            self._is_reset_chat = False
            self._is_reset_force = False
        self._collect_runtime_events(llm_input_msgs)
        llm_input_msgs = self._mask_observations(llm_input_msgs)
        llm_input_msgs = self._maybe_update_hierarchy(llm_input_msgs)
        llm_input_msgs = self._bound_oversized_tool_observations(llm_input_msgs)
        stage_state_prefix = self._build_stage_state_prefix()
        tail_msgs = llm_input_msgs
        ret = reset_ret
        if self.arbit_summary_data is None:
            estimated_input_tokens = count_tokens_approximately(
                self.stage_summary_data
                + self.batch_summary_data
                + self.summary_data
                + stage_state_prefix
                + role_info
                + llm_input_msgs
            )
            exceeds_message_limit = len(llm_input_msgs) > self.max_keep_msgs
            exceeds_token_limit = self.max_token > 0 and estimated_input_tokens > self.max_token
            if exceeds_message_limit or exceeds_token_limit:
                if exceeds_token_limit:
                    warning(
                        f"LLM input is approximately {estimated_input_tokens} tokens, "
                        f"exceeding summary threshold {self.max_token}."
                    )
                # get init start index
                tail_msgs_start_index = (-self.tail_keep_msgs) % len(llm_input_msgs)
                start_msg = llm_input_msgs[tail_msgs_start_index]
                # search for the last not tool message
                while start_msg.type == "tool" and tail_msgs_start_index > 0:
                    tail_msgs_start_index -= 1
                    start_msg = llm_input_msgs[tail_msgs_start_index]
                tail_msgs = llm_input_msgs[tail_msgs_start_index:]
                if tail_msgs_start_index > 0:
                    if self.structured_summary:
                        summary_inputs = (
                            self.stage_summary_data
                            + self.batch_summary_data
                            + self.summary_data
                            + llm_input_msgs[:tail_msgs_start_index]
                        )
                        stage_ctx = self._make_stage_context_message()
                        if stage_ctx is not None:
                            summary_inputs = [stage_ctx] + summary_inputs
                        summary = summarize_messages_structured(
                            self._bound_summary_inputs(summary_inputs),
                            self.max_summary_tokens,
                            self.model,
                            fallback_model=self.main_model,
                            summary_runtime=self.summary_runtime_cfg,
                        )
                    else:
                        summary = summarize_messages(
                            self._bound_summary_inputs(
                                self.stage_summary_data
                                + self.batch_summary_data
                                + self.summary_data
                                + llm_input_msgs[:tail_msgs_start_index]
                            ),
                            self.max_summary_tokens,
                            self.model,
                            fallback_model=self.main_model,
                            summary_runtime=self.summary_runtime_cfg,
                        )
                    self.summary_data = [summary]
                    if self.enable_long_term_memory and self.long_term_memory and self.structured_summary:
                        turn_event = self._build_turn_memory_entry(getattr(summary, "content", str(summary)))
                        if turn_event is not None:
                            self.long_term_memory.save(
                                meta={
                                    "type": "turn",
                                    "dut": self.dut_name,
                                    "stage_index": self._current_stage_index,
                                    "stage_title": self._current_stage_title,
                                    "section_index": self._current_section_index,
                                    "timestamp": time.time(),
                                },
                                content=turn_event,
                            )
                    deleted_msgs = [RemoveMessage(id=msg.id) for msg in llm_input_msgs[:tail_msgs_start_index]]
                    warning(f"Trimmed {len(deleted_msgs)} messages, kept {len(tail_msgs)} tail messages and 1 summary message.")
                    ret["messages"] = deleted_msgs
                else:
                    tail_msgs = llm_input_msgs
        else:
            warning("Using arbitrary provided summary.")
            assert isinstance(self.arbit_summary_data, list), f"Need List, but find: {type(self.arbit_summary_data)}: {self.arbit_summary_data}"
            self.summary_data = self.arbit_summary_data
            self.arbit_summary_data = None
            ret["messages"] = [RemoveMessage(id=msg.id) for msg in tail_msgs]
            tail_msgs = []
        if self.enable_failure_aware_context:
            failure_prefix = self._build_failure_context_prefix()
            if failure_prefix:
                prefix = stage_state_prefix + failure_prefix + self.summary_data
            else:
                prefix = stage_state_prefix + self.stage_summary_data + self.batch_summary_data + self.summary_data
        else:
            prefix = stage_state_prefix + self.stage_summary_data + self.batch_summary_data + self.summary_data
        llm_input_messages, hard_trimmed = self._enforce_hard_input_budget(
            role_info + prefix + tail_msgs
        )
        ret["llm_input_messages"] = llm_input_messages
        self.msg_stat.update_message(ret["llm_input_messages"])
        if self.llm_input_dumper is not None:
            self.llm_input_dumper.dump(
                ret["llm_input_messages"],
                metadata={
                    "node": self.__class__.__name__,
                    "prefix_messages": len(prefix),
                    "tail_messages": len(tail_msgs),
                    "summary_messages": len(self.summary_data),
                    "stage_summary_messages": len(self.stage_summary_data),
                    "batch_summary_messages": len(self.batch_summary_data),
                    "stage_state_package_messages": len(stage_state_prefix),
                    "observation_masking": dict(self.observation_mask_statistics),
                    "soft_token_limit": self.max_token,
                    "hard_token_limit": self.hard_max_tokens,
                    "hard_budget_trimmed": hard_trimmed,
                },
            )
        return ret

    def set_arbit_summary(self, summary_text):
        """Set chat summary"""
        if isinstance(summary_text, str):
            info("Arbit Summary:\n" + summary_text)
            self.arbit_summary_data = [AIMessage(content=summary_text)]
        else:
            assert isinstance(summary_text, list)
            for m in summary_text:
                assert isinstance(m, BaseMessage), f"Need BaseMessage, but find: {type(m)}: {m}"
            info("Arbit Summary:\n" + "\n".join([x.content for x in summary_text]))
            self.arbit_summary_data = summary_text
        return self

    def force_summary(self, messages):
        """Generate chat summary from hist messages"""
        if self.structured_summary:
            stage_ctx = self._make_stage_context_message()
            summary_inputs = [stage_ctx] + messages if stage_ctx is not None else messages
            summary = summarize_messages_structured(
                summary_inputs,
                self.max_summary_tokens,
                self.model,
                fallback_model=self.main_model,
                summary_runtime=self.summary_runtime_cfg,
            )
        else:
            summary = summarize_messages(
                messages,
                self.max_summary_tokens,
                self.model,
                fallback_model=self.main_model,
                summary_runtime=self.summary_runtime_cfg,
            )
        return self.set_arbit_summary([summary])

    def set_max_keep_msgs(self, max_keep_msgs: int):
        self.max_keep_msgs = max_keep_msgs
        return self

    def set_max_token(self, max_token: int):
        self.max_token = max_token
        return self

    def get_max_token(self) -> int:
        return self.max_token

    def get_max_keep_msgs(self) -> int:
        return self.max_keep_msgs


class State(AgentState):
    """Agent state with additional context information."""
    # NOTE: we're adding this key to keep track of previous summary information
    # to make sure we're not summarizing on every LLM call
    context: Dict[str, Any]
