#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for StageManager progress loading behavior."""

import os
import sys
import asyncio
from types import SimpleNamespace

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
repo_package_root = os.path.join(repo_root, "ucagent")
sys.path.insert(0, repo_root)

loaded_ucagent = sys.modules.get("ucagent")
loaded_ucagent_path = os.path.abspath(getattr(loaded_ucagent, "__file__", "") or "")
if loaded_ucagent is not None and not loaded_ucagent_path.startswith(repo_package_root + os.sep):
    for module_name in list(sys.modules):
        if module_name == "ucagent" or module_name.startswith("ucagent."):
            del sys.modules[module_name]

import ucagent.stage.vmanager as vmanager
from ucagent.stage.vstage import VerifyStage
from ucagent.stage.vmanager import StageManager, ToolDoCheck
from ucagent.tools.uctool import to_fastmcp
from ucagent.util import functions as fc


class _FakeStage:
    def __init__(self, index):
        self.index = index
        self.name = f"stage-{index}"
        self.fail_count = 0
        self.time_prev_cost = 0.0
        self.is_complete = False
        self.meta_data = {}
        self._is_reached = False
        self._is_skipped = False
        self.reference_file_status = {}
        self.init_count = 0
        self.hist_init_count = 0
        self.stage_manager = None

    def title(self):
        return self.name

    def is_skipped(self):
        return self._is_skipped

    def set_skip(self, value):
        self._is_skipped = value

    def is_reached(self):
        return self._is_reached

    def set_reached(self, value):
        self._is_reached = value

    def set_fail_count(self, value):
        self.fail_count = value

    def set_time_prev_cost(self, value):
        self.time_prev_cost = value

    def is_completed(self):
        return self.is_complete

    def set_reference_file_status(self, status):
        self.reference_file_status = status

    def set_stage_manager(self, manager):
        self.stage_manager = manager

    def on_init(self):
        self.init_count += 1

    def hist_init(self):
        self.hist_init_count += 1

    def get_time_cost(self):
        return self.time_prev_cost

    def is_wait_human_check(self):
        return False

    def detail(self):
        return {
            "task": {"reference_files": self.reference_file_status},
            "reached": self.is_reached(),
            "is_completed": self.is_completed(),
            "fail_count": self.fail_count,
            "is_skipped": self.is_skipped(),
        }


class _FakeRootStage:
    def __init__(self, stages):
        self._stages = stages

    def get_substages(self):
        return self._stages


class _FakeAgent:
    def __init__(self, cfg, is_exit=False):
        self.cfg = cfg
        self._is_exit = is_exit

    def is_break(self):
        return False

    def is_exit(self):
        return self._is_exit

    def get_stat_info(self):
        return {"version": "test"}


class _FakeCurrentStageManager:
    def __init__(self, current_stage):
        self._current_stage = current_stage

    def get_current_stage(self):
        return self._current_stage


class _RecordingCheckStage:
    name = "recording-stage"

    def __init__(self):
        self.calls = []
        self._is_reached = False
        self.init_count = 0

    def do_check(self, **kwargs):
        self.calls.append(kwargs)
        return True, {"seen": kwargs}

    def meta_get_journal(self):
        return "journal"

    def get_approved(self):
        return True

    def is_hmcheck_needed(self):
        return False

    def on_complete(self):
        pass

    def set_reached(self, value):
        self._is_reached = value

    def on_init(self):
        self.init_count += 1


class _PackageNode:
    def __init__(self):
        self.package = None

    def update_stage_state_package(self, package):
        self.package = package


class _PackageAgent:
    enable_stage_state_package = True
    enable_data_collection = False

    def __init__(self):
        self.message_manage_node = _PackageNode()


def _cfg():
    return SimpleNamespace(
        tools=SimpleNamespace(RunTestCases=SimpleNamespace(test_dir="tests")),
        mission=SimpleNamespace(name="mission"),
        vmanager=SimpleNamespace(
            llm_suggestion=SimpleNamespace(
                check_fail_refinement=None,
                check_pass_refinement=None,
            ),
        ),
        skill=SimpleNamespace(use_skill=False),
    )


def _saved_info():
    return {
        "stage_index": 3,
        "all_completed": True,
        "time_begin": 100.0,
        "time_end": 200.0,
        "is_agent_exit": True,
        "is_wait_human_check": True,
        "stages_info": {
            str(index): {
                "fail_count": index + 1,
                "time_cost": float((index + 1) * 10),
                "reached": True,
                "is_skipped": False,
                "is_completed": True,
                "task": {"reference_files": {"ref.md": "Readed"}},
                "meta_data": {"journal": f"journal-{index}"},
            }
            for index in range(4)
        },
    }


def test_stage23_empty_test_target_is_redirected_to_current_batch():
    manager = StageManager.__new__(StageManager)
    checker = SimpleNamespace(
        current_test_cases=["tests/test_DUT.py::test_a", "tests/test_DUT.py::test_b"],
        get_run_args=lambda _test_dir: (
            "test_DUT.py::test_a test_DUT.py::test_b",
            [],
        ),
    )
    manager.stage_index = 0
    manager.stages = [SimpleNamespace(
        name="test_case_implementation_in_batch",
        checker=[checker],
    )]
    manager.free_pytest_run = SimpleNamespace(test_dir="tests")
    events = []
    manager._record_structured_event = lambda event_type, detail: events.append((event_type, detail))
    manager.attach_todo_summary = lambda text: text

    result = manager.tool_run_test_cases("")

    assert "TEST_TARGET_CONTRACT_BLOCKED" in result
    assert "test_DUT.py::test_a test_DUT.py::test_b" in result
    assert events == [(
        "test_target_contract_blocked",
        {
            "tool": "RunTestCases",
            "target": "",
            "required_target": "test_DUT.py::test_a test_DUT.py::test_b",
            "reason": "stage23_empty_target_would_mix_future_templates",
        },
    )]


def test_force_stage_rewind_truncates_loaded_stage_progress(monkeypatch, tmp_path):
    stages = [_FakeStage(index) for index in range(4)]
    cfg = _cfg()
    workspace = str(tmp_path)

    monkeypatch.setattr(vmanager, "get_root_stage", lambda *_args: _FakeRootStage(stages))
    monkeypatch.setattr(vmanager, "get_llm_check_instance", lambda *_args: None)

    manager = StageManager(
        workspace,
        cfg,
        _FakeAgent(cfg),
        tool_read_text=None,
        ucagent_info=_saved_info(),
        force_stage_index=1,
        tool_inspect_file=[],
    )
    manager.force_stage_index_explicit = True

    manager.init_stage()

    assert manager.stage_index == 1
    assert manager.all_completed is False
    assert manager.time_end is None

    assert stages[0].is_completed() is True
    assert stages[0].fail_count == 1
    assert stages[0].time_prev_cost == 10.0
    assert stages[0].meta_data == {"journal": "journal-0"}

    for stage in stages[1:]:
        assert stage.is_completed() is False
        assert stage.fail_count == 0
        assert stage.time_prev_cost == 0.0
        assert stage.meta_data == {}

    saved = fc.load_ucagent_info(workspace)
    assert saved["stage_index"] == 1
    assert saved["all_completed"] is False
    assert saved["time_end"] is None
    assert saved["is_agent_exit"] is False
    assert saved["is_wait_human_check"] is False
    assert saved["stages_info"]["0"]["is_completed"] is True
    assert saved["stages_info"]["0"]["meta_data"] == {"journal": "journal-0"}
    assert saved["stages_info"]["1"]["is_completed"] is False
    assert saved["stages_info"]["1"]["fail_count"] == 0
    assert saved["stages_info"]["1"]["time_cost"] == 0.0
    assert saved["stages_info"]["1"]["meta_data"] == {}


def test_save_stage_info_persists_mission_name_and_exit_state(tmp_path):
    cfg = _cfg()
    cfg.mission.name = "Formal Coverage Mission"
    agent = _FakeAgent(cfg, is_exit=True)
    stage = _FakeStage(0)
    manager = StageManager(
        str(tmp_path),
        cfg,
        agent,
        tool_read_text=None,
        ucagent_info={},
        force_stage_index=0,
        tool_inspect_file=[],
    )
    manager.stages = [stage]
    manager.stage_index = 0
    manager.time_begin = 10.0
    manager.time_end = None

    manager.save_stage_info()

    saved = fc.load_ucagent_info(str(tmp_path))
    assert saved["mission_name"] == "Formal Coverage Mission"
    assert saved["is_agent_exit"] is True
    assert saved["all_completed"] is False


def _make_verify_stage(name, reference_files, parent=None):
    stage = VerifyStage.__new__(VerifyStage)
    stage.name = name
    stage.reference_files = dict(reference_files)
    stage.parent = parent
    stage.force_unactive = False
    stage.skill_list = {}
    stage.workspace = ""
    stage.vmanager = None
    stage.is_skill_path = lambda _file_path: False
    return stage


def test_on_file_read_marks_reference_files_in_parent_stage_chain():
    root = _make_verify_stage("root", {"shared.md": False, "root_only.md": False})
    parent = _make_verify_stage("parent", {"shared.md": False}, parent=root)
    child = _make_verify_stage("child", {"shared.md": False, "child_only.md": False}, parent=parent)
    manager = _FakeCurrentStageManager(child)
    root.vmanager = manager
    parent.vmanager = manager
    child.vmanager = manager

    child.on_file_read(True, "shared.md", "content")

    assert root.reference_files["shared.md"] is True
    assert parent.reference_files["shared.md"] is True
    assert child.reference_files["shared.md"] is True
    assert root.reference_files["root_only.md"] is False
    assert child.reference_files["child_only.md"] is False


def test_tool_do_check_passes_extra_arguments_to_function():
    calls = []

    def check_func(timeout, **kwargs):
        calls.append((timeout, kwargs))
        return "ok"

    tool = ToolDoCheck().set_function(check_func)

    result = tool.invoke({
        "timeout": 12,
        "refined": {"CK-1": "done"},
        "note": "extra",
        "detail": True,
    })

    assert result == "ok"
    assert calls == [(12, {"refined": {"CK-1": "done"}, "note": "extra", "detail": True})]


def test_stage_state_package_prefers_checker_over_local_test_result():
    manager = StageManager.__new__(StageManager)
    manager.agent = _PackageAgent()
    manager.stage_index = 0
    manager.stages = [_FakeStage(0)]
    manager.stages[0].reference_file_status = {
        "todo.md": "Not Read",
        "done.md": "Readed",
    }
    manager._stage_state_runtime = {
        "stage_index": None,
        "generation": 0,
        "last_test": None,
        "last_checker": None,
        "last_control": None,
    }

    manager._publish_stage_state_package("test_run", {
        "stage_index": 0,
        "result": {"test_outcome": "pass", "tests_total": 1, "tests_failed": 0},
    }, {"credit_role": "progress", "policy": "exploit"})
    manager._publish_stage_state_package("check_result", {
        "stage_index": 0,
        "result": {"check_pass": False, "checker_categories": ["bug_doc_schema"]},
    }, {"credit_role": "no_progress", "policy": "pivot", "checker_no_progress_streak": 2})

    package = manager.agent.message_manage_node.package
    assert package["last_test"]["test_outcome"] == "pass"
    assert package["task_contract"]["reference_files_unread"] == ["todo.md"]
    assert package["unresolved"]["source"] == "checker"
    assert package["unresolved"]["checker_categories"] == ["bug_doc_schema"]
    assert package["progress_control"]["policy"] == "pivot"


def test_check_summary_extracts_nested_test_report_and_failure_set():
    manager = StageManager.__new__(StageManager)
    manager.stage_index = 0
    stage = _FakeStage(0)
    stage.name = "comprehensive_verification_and_bug_analysis"
    manager.stages = [stage]
    result = {
        "last_check_result": {
            "check_pass": False,
            "checker_error_summary": {
                "categories": ["unsupported_test_status"],
                "raw_error_top": ["Unsupported test statuses: XFAIL"],
            },
            "check_info": [{
                "last_msg": {
                    "TEST_REPORT": {
                        "run_test_success": False,
                        "tests": {
                            "total": 3,
                            "fails": 2,
                            "test_cases": {
                                "test_ok": "PASSED",
                                "test_width": "FAILED",
                                "test_expected": "XFAIL",
                            },
                        },
                        "failed_check_point_list": ["FG-A/FC-B/CK-C"],
                    }
                }
            }],
        }
    }

    summary = manager._summarize_check_result(result)

    assert summary["checker_categories"] == ["unsupported_test_status"]
    assert summary["tests_total"] == 3
    assert summary["tests_failed"] == 2
    assert summary["test_status_counts"] == {"PASSED": 1, "FAILED": 1, "XFAIL": 1}
    assert summary["test_outcome"] == "test_failure"
    assert summary["failed_cases_top"] == ["test_width", "test_expected"]
    assert summary["failed_checkpoints_top"] == ["FG-A/FC-B/CK-C"]


def test_fastmcp_check_preserves_extra_arguments():
    calls = []

    def check_func(timeout, **kwargs):
        calls.append((timeout, kwargs))
        return "ok"

    tool = ToolDoCheck().set_function(check_func)
    mcp_tool = to_fastmcp(tool)

    extra_schema = mcp_tool.parameters["additionalProperties"]
    assert extra_schema
    assert "checker-specific" in extra_schema["description"]
    assert "top-level JSON fields" in extra_schema["description"]
    generated_schema = mcp_tool.parameters["properties"]["generated"]
    assert generated_schema["anyOf"][0]["type"] == "object"
    assert "generate_random_test_cases" in generated_schema["description"]
    result = asyncio.run(mcp_tool.run({
        "timeout": 13,
        "ctx": "internal context should not leak",
        "refined": {"CK-2": "done"},
        "generated": {"FG-A/FC-A/CK-A": "random test added"},
        "detail": False,
    }))

    assert result == "ok"
    assert calls == [(13, {
        "generated": {"FG-A/FC-A/CK-A": "random test added"},
        "refined": {"CK-2": "done"},
        "detail": False,
    })]


def test_stage_manager_check_and_complete_forward_extra_arguments():
    stage = _RecordingCheckStage()
    next_stage = _RecordingCheckStage()
    manager = StageManager.__new__(StageManager)
    manager.stage_index = 0
    manager.stages = [stage, next_stage]
    manager.last_check_info = None
    manager.llm_fail_suggestion = None
    manager.llm_pass_suggestion = None
    manager.all_completed = False
    manager.gen_fail_suggestion = lambda data: data
    manager.gen_pass_suggestion = lambda ck_info: ""
    manager._stage_complete = lambda _stage: None

    def next_stage_func():
        manager.stage_index += 1
        manager.all_completed = manager.stage_index >= len(manager.stages)
        return None if manager.all_completed else manager.stages[manager.stage_index]

    manager.next_stage = next_stage_func

    check_ret = manager.check(9, refined={"CK": "check"}, detail=True)
    complete_ret = manager.complete(10, refined={"CK": "complete"}, is_complete=False)

    assert check_ret["check_pass"] is True
    assert complete_ret["complete"] is True
    assert stage.calls == [
        {"refined": {"CK": "check"}, "detail": True, "timeout": 9},
        {"refined": {"CK": "complete"}, "timeout": 10, "is_complete": True},
    ]
