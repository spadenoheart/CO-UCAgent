# --- coding: utf-8 ---
"""Message and state utilities for UCAgent."""

from .statistic import MessageStatistic
from ucagent.util.functions import fill_dlist_none
from ucagent.util.log import warning, info

from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.messages import AIMessage, RemoveMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from langmem.short_term import SummarizationNode
from langgraph.prebuilt.chat_agent_executor import AgentState
from typing import Any, Dict, Union, Optional, List, Tuple
from pydantic import BaseModel, Field
import time
import yaml
import json


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


def summarize_messages(messages, summarization_size, model):
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
    warning(f"Summarizing messages({count_tokens_approximately(messages)} tokens, {len(messages)} messages) to reduce context size ...")
    summary_response = model.invoke([HumanMessage(content=instruction)] + messages)
    warning(f"Summarization done, summary length: {count_tokens_approximately(summary_response.content)} tokens.")
    return summary_response


def summarize_messages_structured(messages, summarization_size, model):
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
    # Prefer structured output when supported (tool/function-calling style)
    summary_response = None
    if hasattr(model, "with_structured_output"):
        try:
            structured_model = model.with_structured_output(StructuredSummarySchema)
            structured_obj = structured_model.invoke([HumanMessage(content=instruction)] + messages)
            if isinstance(structured_obj, BaseModel):
                summary_response = AIMessage(content=json.dumps(structured_obj.model_dump(), ensure_ascii=False))
            elif isinstance(structured_obj, dict):
                summary_response = AIMessage(content=json.dumps(structured_obj, ensure_ascii=False))
        except Exception:
            summary_response = None
    if summary_response is None:
        summary_response = model.invoke([HumanMessage(content=instruction)] + messages)
    repaired = _coerce_structured_summary(summary_response.content, messages, model)
    if repaired is not None:
        warning("Structured summary repaired/normalized to valid JSON.")
        return AIMessage(content=repaired)
    warning(f"Structured summarization done, summary length: {count_tokens_approximately(summary_response.content)} tokens.")
    return summary_response


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


def _coerce_structured_summary(text: str, messages, model) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return None
    parsed = _try_parse_structured_json(text)
    if parsed is not None:
        return json.dumps(_fill_from_context(parsed, messages), ensure_ascii=False)
    repair_prompt = (
        "You MUST output a valid JSON object with EXACTLY these top-level keys:\n"
        "[stage_info, test_report, coverage_status, bug_tracking, next].\n"
        "Do not include any extra text. If a field is unknown, use empty values.\n"
        "Convert the following content into the required JSON:\n\n"
        f"{text}"
    )
    try:
        repaired = model.invoke([HumanMessage(content=repair_prompt)] + messages[-8:])
        parsed = _try_parse_structured_json(getattr(repaired, "content", ""))
        if parsed is not None:
            return json.dumps(_fill_from_context(parsed, messages), ensure_ascii=False)
    except Exception:
        pass
    fallback = _build_fallback_structured_summary(messages)
    return json.dumps(fallback, ensure_ascii=False)


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
        passed = max(0, total - fails)
        ret["tests"] = {"total": total, "passed": passed, "failed": fails}
        test_cases = tests.get("test_cases") or {}
        if isinstance(test_cases, dict):
            failed_cases = [k for k, v in test_cases.items() if str(v).upper() not in ("PASS", "PASSED")]
            ret["failed_test_cases_top"] = failed_cases[:10]
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
            self.msg_stat.update_message(ret["llm_input_messages"])
        else:
            self.msg_stat.update_message(ret["messages"])
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
        structured_summary: bool = False,
        hierarchical_summary: bool = False,
        enable_long_term_memory: bool = False,
        long_term_memory=None,
        enable_failure_aware_context: bool = False,
        dut_name: str = "",
    ):
        self.msg_stat = msg_stat
        self.max_summary_tokens = max_summary_tokens
        self.max_keep_msgs = max_keep_msgs
        self.tail_keep_msgs = tail_keep_msgs
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
        self.model = model
        self.structured_summary = structured_summary
        self.hierarchical_summary = hierarchical_summary
        self.enable_long_term_memory = enable_long_term_memory
        self.long_term_memory = long_term_memory
        self.enable_failure_aware_context = enable_failure_aware_context
        self.dut_name = dut_name
        self.arbit_summary_data = None

    def set_stage_context(self, stage_index: Optional[int], stage_title: str, section_index: str, progress: str):
        """Update current stage context for memory tagging and stage summary."""
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

    def _build_failure_context_prefix(self) -> List[AIMessage]:
        stage_items = self.stage_summary_data[-3:] if self.stage_summary_data else []
        batch_items = self.batch_summary_data[-3:] if self.batch_summary_data else []
        failure_items = []
        non_failure_items = []

        for msg in stage_items + batch_items:
            if not isinstance(msg, AIMessage):
                continue
            parsed = _parse_summary_payload(msg.content)
            if not isinstance(parsed, dict):
                continue
            if _is_failure_summary(parsed):
                failure_items.append(parsed)
            else:
                non_failure_items.append(parsed)

        prefix = []
        if failure_items:
            prefix.append(
                AIMessage(
                    content="FAILURE_CONTEXT:\n"
                    + yaml.safe_dump(failure_items, allow_unicode=True, sort_keys=False)
                )
            )
        if non_failure_items:
            prefix.append(
                AIMessage(
                    content="SUCCESS_CONTEXT:\n"
                    + yaml.safe_dump(non_failure_items, allow_unicode=True, sort_keys=False)
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
                if self.enable_long_term_memory and self.long_term_memory:
                    self.long_term_memory.save(
                        meta={
                            "type": "stage",
                            "dut": self.dut_name,
                            "stage_index": self._last_stage_index,
                            "timestamp": time.time(),
                        },
                        content=stage_summary,
                    )
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
                    if self.enable_long_term_memory and self.long_term_memory:
                        self.long_term_memory.save(
                            meta={
                                "type": "batch",
                                "dut": self.dut_name,
                                "stage_index": stage_index if stage_index is not None else self._current_stage_index,
                                "timestamp": time.time(),
                            },
                            content=batch,
                        )
                    continue
            new_msgs.append(m)
        return new_msgs

    def __call__(self, state):
        fix_tool_call_args(state)
        messages = state["messages"]
        role_info = messages[:1]
        llm_input_msgs = messages[1:]
        llm_input_msgs = self._maybe_update_hierarchy(llm_input_msgs)
        tail_msgs = llm_input_msgs
        ret = {}
        if self.arbit_summary_data is None:
            if len(llm_input_msgs) > self.max_keep_msgs:
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
                        summary = summarize_messages_structured(summary_inputs, self.max_summary_tokens, self.model)
                    else:
                        summary = summarize_messages(
                            self.stage_summary_data + self.batch_summary_data + self.summary_data + llm_input_msgs[:tail_msgs_start_index],
                            self.max_summary_tokens,
                            self.model,
                        )
                    self.summary_data = [summary]
                    if self.enable_long_term_memory and self.long_term_memory and self.structured_summary:
                        summary_payload = {
                            "summary": getattr(summary, "content", str(summary)),
                        }
                        self.long_term_memory.save(
                            meta={
                                "type": "turn",
                                "dut": self.dut_name,
                                "stage_index": self._current_stage_index,
                                "timestamp": time.time(),
                            },
                            content=summary_payload,
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
                prefix = failure_prefix + self.summary_data
            else:
                prefix = self.stage_summary_data + self.batch_summary_data + self.summary_data
        else:
            prefix = self.stage_summary_data + self.batch_summary_data + self.summary_data
        ret["llm_input_messages"] = prefix + role_info + tail_msgs
        self.msg_stat.update_message(ret["llm_input_messages"])
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
            summary = summarize_messages_structured(summary_inputs, self.max_summary_tokens, self.model)
        else:
            summary = summarize_messages(messages, self.max_summary_tokens, self.model)
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
