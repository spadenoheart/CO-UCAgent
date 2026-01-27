# -*- coding: utf-8 -*-
"""Verification manager for UCAgent stage execution."""

import copy
import json
import time
import traceback
from collections import OrderedDict
from typing import Optional, Callable

from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field

import ucagent.util.functions as fc
from ucagent.checkers import UnityChipCheckerTestFree
from ucagent.stage.vstage import get_root_stage
from ucagent.tools.uctool import UCTool, EmptyArgs
from ucagent.util.functions import make_llm_tool_ret
from ucagent.util.log import info, warning


class ManagerTool(UCTool):
    # custom vars
    function: Callable = None
    args_schema: Optional[ArgsSchema] = EmptyArgs

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.function()

    def set_function(self, func):
        self.function = func
        return self


class ToolStatus(ManagerTool):
    """List current missoin status."""
    name: str = "Status"
    description: str = (
        "Returns the current status of your mission."
    )


class ToolCurrentTips(ManagerTool):
    """Get tips for the current task."""
    name: str = "CurrentTips"
    description: str = (
        "Returns the tips for the current task."
    )


class ToolDetail(ManagerTool):
    """Get current missoin detials."""
    name: str = "Detail"
    description: str = (
        "Returns the detail info of your mission, including all stages and their details. \n"
    )


class ToolKillCheck(ManagerTool):
    """Kill the current check process."""
    name: str = "KillCheck"
    description: str = (
        "Kill the current check process. \n"
        "This tool is only used when the tool 'Check' is long time running or get stuck. \n"
    )


class ArgStdCheck(BaseModel):
    lines: int = Field(
        default=-1,
        description="lines to read, -1 means read all"
    )


class ToolStdCheck(ManagerTool):
    """get the standard output of the current check process."""
    name: str = "StdCheck"
    description: str = (
        "Get the standard output of the current check process. \n"
        "This tool is only used to get the output of the runnig tool 'Check'. \n"
        "You can specify the number of lines to read, -1 means read all lines. \n"
    )
    args_schema: Optional[ArgsSchema] = ArgStdCheck

    def _run(self, lines: int = -1, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.function(lines)


class ArgCheck(BaseModel):
    target: str = Field(
        default="",
        description=(
            "Target test cases to run, supports pytest-style arguments for precise test selection. "
            "Examples:\n"
            "• '' (empty): Run all test cases in the test directory\n"
            "• 'test_file.py': Run all tests in a specific file\n"
            "• 'test_file.py::test_function': Run a specific test function\n"
            "• 'test_file.py::TestClass::test_method': Run a specific test method in a class\n"
            "• '-k pattern': Run tests matching the given pattern\n"
            "• '-m marker': Run tests with specific markers\n"
        )
    )
    timeout: int = Field(
        default=0,
        description="Timeout for the test run in seconds. Zero means use default cfg.call_time_out."
    )
    return_line_coverage: bool = Field(
        default=False,
        description="Whether to return line coverage information in the test results."
    )


class ToolRunTestCases(ManagerTool):
    """Run test cases in current workspace."""
    name: str = "RunTestCases"
    description: str = (
        "This tool is used to execute the test cases in the workspace. "
        "Returns the result of the test execution. You should call this tool after you have implemented or modified the DUT or test cases. "
        "Current test directory is set to the '{TEST_DIR}',  the file path you passed should be relative to this directory."
    )
    args_schema: Optional[ArgsSchema] = ArgCheck

    def _run(self, target="", timeout=0, return_line_coverage=False,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            if timeout <= 0:
                timeout = self.get_call_time_out()
            return self.function(target, timeout, return_line_coverage)
        except Exception as e:
            traceback.print_exc()
            error_msg = f"Test execution failed: {str(e)}"
            info(error_msg)
            return error_msg


class ArgTimeout(BaseModel):
    timeout: int = Field(
        default=0,
        description="Timeout for the test run in seconds. Zero means use default cfg.call_time_out."
    )


class ToolDoCheck(ManagerTool):
    """Advanced validation tool for stage requirements and implementation quality."""
    name: str = "Check"
    description: str = (
        "Perform comprehensive validation of your current stage's implementation against requirements.\n"
        "The tool provides detailed feedback. Call this tool frequently to ensure continuous quality validation."
    )
    args_schema: Optional[ArgsSchema] = ArgTimeout

    def _run(self, timeout=0, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Execute stage validation with enhanced error handling and reporting.
        
        Args:
            target: Test target specification (pytest format)
            run_manager: Callback manager for tool execution
            
        Returns:
            str: Comprehensive validation report in JSON format
        """
        try:
            if timeout <= 0:
                timeout = self.get_call_time_out()
            return self.function(timeout)
        except Exception as e:
            traceback.print_exc()
            error_msg = f"Validation failed: {str(e)}"
            info(error_msg)
            return make_llm_tool_ret({
                "check_pass": False,
                "check_info": error_msg
            })


class ToolDoComplete(ManagerTool):
    """Tell the manager that you have completed the current stage."""
    name: str = "Complete"
    description: str = (
        "Tell the manager that you have completed the current stage. \n"
        "When you complete a stage, your should have passed all checks in the stage. \n"
        "You should double check your work before calling this tool. \n"
        "Returns the result of the completion."
    )
    args_schema: Optional[ArgsSchema] = ArgTimeout

    def _run(self, timeout=0, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            if timeout <= 0:
                timeout = self.get_call_time_out()
            return self.function(timeout)
        except Exception as e:
            traceback.print_exc()
            error_msg = f"Completion failed: {str(e)}"
            info(error_msg)
            return error_msg


class ArgToolGoToStage(BaseModel):
    index: int = Field(
        default=-1,
        description="Stage index to go to. "
    )


class ToolGoToStage(ManagerTool):
    """Go to a specific stage by index."""
    name: str = "GoToStage"
    description: str = (
        "Go to a specific stage by index. Only those stages that have been reached can be selected. \n"
        "Stage is reached means that all checks in the stage have been passed. \n"
        "This tool is used when you want refine your previous work, or want to go back to a previous stage. \n"
        "Returns the result of the operation."
    )
    args_schema: Optional[ArgsSchema] = ArgToolGoToStage

    def _run(self, index: int = -1, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.function(index)


class ToolDoExit(ManagerTool):
    """Exit the agent and end the mission after all stages are completed."""
    name: str = "Exit"
    description: str = (
        "Exit the agent and end the mission after all stages are completed. \n"
        "This tool is used when you have completed all stages and want to exit the agent. \n"
        "Returns a message indicating the exit status."
    )


class StageManager(object):
    def __init__(
            self, workspace, cfg, agent, tool_read_text, ucagent_info: dict,
            force_stage_index=0,
            force_todo=False,
            todo_panel=None,
            stage_skip_list=None,
            stage_unskip_list=None,
            reference_files=None,
    ):
        """
        Initialize the StageManager with an empty list of stages.
        """
        self.cfg = cfg
        self.data = {}
        self.workspace = workspace
        self.force_todo = force_todo
        self.todo_panel = todo_panel
        self.free_pytest_run = UnityChipCheckerTestFree("", cfg.tools.RunTestCases.test_dir, "").set_workspace(workspace)
        self.free_pytest_run.compact_test_output = cfg.get_value(
            "context_upgrade.enable_compact_test_output", False
        )
        self.agent = agent
        self.enable_data_collection = cfg.get_value("context_upgrade.enable_data_collection", False)
        self._stage_token_start = None
        self._stage_msg_in_start = None
        self._stage_msg_out_start = None
        self.tool_read_text = tool_read_text
        self.ucagent_info = ucagent_info
        self.force_stage_index = force_stage_index
        self.stage_skip_list = stage_skip_list
        self.stage_unskip_list = stage_unskip_list
        self.reference_files = reference_files

    def init_stage(self):
        from ucagent.stage import VerifyStage
        self.root_stage = get_root_stage(self.cfg, self.workspace, self.tool_read_text)
        self.stages = self.root_stage.get_substages()
        if self.reference_files:
            for si, flist in self.reference_files.items():
                if 0 <= si < len(self.stages):
                    info(f"Stage {si} try add reference files: {flist}")
                    self.stages[si].add_reference_files(flist)
                elif si == -1:
                    info(f"All stages try add reference files: {flist}")
                    for s in self.stages:
                        s.add_reference_files(flist)
                else:
                    warning(f"Invalid stage index {si} in reference_files, ignored.")
        self.mission = self.cfg.mission
        info(f"Initialized StageManager with {len(self.stages)} stages.")
        info("Stages:\n" + "\n".join([f"{i:2d}:   {stage.title()}{' (skipped)' if stage.is_skipped() else ''}" for i, stage in enumerate(self.stages)]))
        self.stage_index = min(max(0, self.force_stage_index), len(self.stages) - 1)
        for i in range(self.stage_index + 1):
            if self.stages[i].is_skipped():
                continue
            self.stages[i].set_reached(True)
        stages_info = self.ucagent_info.get("stages_info", {})

        for stage_idx_str, stage_info in stages_info.items():
            idx = int(stage_idx_str)
            if idx >= len(self.stages):
                continue
            stage: VerifyStage = self.stages[idx]
            stage.set_fail_count(stage_info.get("fail_count", 0))
            stage.set_time_prev_cost(stage_info.get("time_cost", 0.0))
            stage.set_reference_file_status(stage_info.get("task", {}).get("reference_files", {}))
        self._go_skip_stage()
        for s in self.stages:
            s.set_stage_manager(self)
        self.stages[self.stage_index].on_init()
        self._update_stage_context(self.stages[self.stage_index])
        self._reset_stage_token_start()
        self.last_check_info = {}
        self.all_completed = False
        if self.stage_skip_list:
            for si in self.stage_skip_list:
                self.skip_stage(si)
                info(f"Stage {si} is set to be skipped.")
        if self.stage_unskip_list:
            for sui in self.stage_unskip_list:
                self.unskip_stage(sui)
                info(f"Stage {sui} is set to be unskipped.")
        info("Current stage index is " + str(self.stage_index) + ".")
        self.time_begin = time.time()
        self.time_end = None

    def get_time_cost(self):
        if self.time_end is None:
            return time.time() - self.time_begin
        return self.time_end - self.time_begin

    def _update_stage_context(self, stage):
        msg_node = getattr(self.agent, "message_manage_node", None)
        if msg_node is None or not hasattr(msg_node, "set_stage_context"):
            return
        try:
            detail = stage.detail()
            section_index = detail.get("section_index", "")
            progress = f"{self.stage_index}/{len(self.stages)}"
            msg_node.set_stage_context(self.stage_index, stage.title(), section_index, progress)
        except Exception:
            return

    def attach_todo_summary(self, data):
        assert isinstance(data, str), "the target data type of attach_todo_summary must be str"
        if not self.force_todo:
            return data
        if not self.todo_panel:
            return data
        return data + self.todo_panel._summary()

    def set_data(self, key, value):
        self.data[key] = value

    def get_data(self, key, default=None):
        return self.data.get(key, default)

    def new_tools(self):
        """
        Create and return a list of tools for the current stage.
        """
        tools = [
            ToolCurrentTips().set_function(self.tool_current_tips),
            ToolDetail().set_function(self.tool_detail),
            ToolStatus().set_function(self.tool_status),
            ToolRunTestCases().set_function(self.tool_run_test_cases).render_desc({"TEST_DIR": self.free_pytest_run.test_dir}),
            ToolDoCheck().set_function(self.tool_check),
            ToolKillCheck().set_function(self.tool_kill_check),
            ToolStdCheck().set_function(self.tool_std_check),
            ToolDoComplete().set_function(self.tool_complete),
            ToolGoToStage().set_function(self.tool_go_to_stage),
            ToolDoExit().set_function(self.tool_exit),
        ]
        return tools

    def get_current_tips(self):
        if self.stage_index >= len(self.stages):
            return "Your mission is completed. No more stages available. You can use `Exit` tool to exit the mission or `GoToStage` tool to go to a specific stage to review."
        cstage = self.stages[self.stage_index]
        tips = OrderedDict()
        tips["mission"] = self.mission.name
        tips["current_stage"] = OrderedDict({
            "index": self.stage_index,
            **cstage.detail(),
        })
        ref_files = []
        for k, v in cstage.reference_files.items():
            if v:
                continue
            ref_files.append(k)
        if ref_files:
            tips["notes"] = f"You need use tool: {self.tool_read_text.name} to read the reference files."
        tips["process"] = f"{self.stage_index}/{len(self.stages)}"
        self._update_stage_context(cstage)
        tips = make_llm_tool_ret(tips)
        if getattr(self.agent, "enable_long_term_memory", False) and getattr(self.agent, "long_term_memory", None):
            ltm = self.agent.long_term_memory
            stage_title = cstage.title()
            query = f"{self.agent.dut_name} {stage_title}"
            filters = {"dut": self.agent.dut_name, "stage_index": self.stage_index}
            memories = ltm.search(query=query, limit=3, filters=filters)
            if not memories:
                memories = ltm.recent(limit=2, filters={"dut": self.agent.dut_name})
            if memories:
                if getattr(self.agent, "enable_failure_aware_context", False):
                    failure_mem, non_failure_mem = self._split_failure_context(memories)
                    parts = []
                    if failure_mem:
                        parts.append("FAILURE_CONTEXT:\n" + fc.yam_str(failure_mem))
                    if non_failure_mem:
                        parts.append("SUCCESS_CONTEXT:\n" + fc.yam_str(non_failure_mem))
                    memory_text = "LONG_TERM_MEMORY:\n" + "\n".join(parts)
                else:
                    memory_text = "LONG_TERM_MEMORY:\n" + fc.yam_str(memories)
                tips = memory_text + "\n\n" + tips
        return self.attach_todo_summary(tips)

    def _split_failure_context(self, memories):
        failure_mem = []
        non_failure_mem = []
        for m in memories:
            if self._is_failure_memory(m):
                failure_mem.append(m)
            else:
                non_failure_mem.append(m)
        return failure_mem, non_failure_mem

    def _is_failure_memory(self, entry) -> bool:
        if not isinstance(entry, dict):
            return False
        content = entry.get("content")
        return self._is_failure_content(content)

    def _is_failure_content(self, content) -> bool:
        if isinstance(content, dict):
            if content.get("run_test_success") is False:
                return True
            tests = content.get("tests")
            hard_fail = False
            if isinstance(tests, dict):
                try:
                    failed = int(tests.get("failed", 0) or 0)
                except Exception:
                    failed = 0
                if failed > 0:
                    hard_fail = True
            if isinstance(content.get("failed_test_cases_top"), list) and content.get("failed_test_cases_top"):
                hard_fail = True
            if hard_fail:
                return True
            if "test_report" in content and isinstance(content.get("test_report"), dict):
                tr = content.get("test_report") or {}
                try:
                    failed = int(tr.get("failed", 0) or 0)
                except Exception:
                    failed = 0
                if failed > 0:
                    return True
                for key in ("failed_cases_top", "failed_checkpoints_top"):
                    block = tr.get(key)
                    if isinstance(block, list) and block:
                        return True
            # Do not treat checkpoint-only metadata as hard failures by default.
            if "summary" in content and isinstance(content.get("summary"), str):
                parsed = self._try_parse_json(content.get("summary"))
                if isinstance(parsed, dict):
                    return self._is_failure_content(parsed)
            return False
        if isinstance(content, str):
            parsed = self._try_parse_json(content)
            if isinstance(parsed, dict):
                return self._is_failure_content(parsed)
        return False

    def _try_parse_json(self, text: str):
        if not isinstance(text, str) or not text.strip():
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        try:
            fixed = fc.fix_json_string(text)
            return json.loads(fixed)
        except Exception:
            return None

    def detail(self):
        """
        Get the details of the current mission, including all stages and their details.
        """
        ret = OrderedDict()
        ret["mission"] = self.mission.name
        ret["stage_list"] = []
        for i, stage in enumerate(self.stages):
            ret["stage_list"].append(stage.detail())
            ret["stage_list"][-1]["index"] = i
        ret["current_stage_index"] = self.stage_index
        ret["current_stage_name"] = self.stages[self.stage_index].name if self.stage_index < len(self.stages) else None
        return ret

    def status(self):
        ret = OrderedDict()
        ret["mission"] = self.mission.name
        ret["stage_list"] = []
        for i, stage in enumerate(self.stages):
            ret["stage_list"].append({
                "index": i,
                "title": stage.title(),
                "reached": stage.is_reached(),
                "fail_count": stage.fail_count,
                "is_skipped": stage.is_skipped(),
                "time_cost": stage.get_time_cost_str(),
                "needs_human_check": stage.is_hmcheck_needed(),
            })
        ret["process"] = f"{self.stage_index}/{len(self.stages)}"
        cstage = self.stages[self.stage_index] if self.stage_index < len(self.stages) else None
        ret["current_task"] = "No stages available (Maybe mission is completed, you can use the `GoToStage` tool to go back to a previous stage if needed)"
        if cstage:
            ret["current_stage_index"] = self.stage_index
            ret["current_stage_name"] = cstage.name
            ret["current_task"] = cstage.task_info()
        ret["last_check_result"] = self.last_check_info
        return ret

    def get_current_stage(self):
        return self.get_stage(self.stage_index)

    def get_stage(self, index):
        if 0 <= index < len(self.stages):
            return self.stages[index]
        return None

    def go_to_stage(self, index):
        """
        Go to a specific stage by index.
        """
        success = False
        if 0 <= index < len(self.stages):
            if index == self.stage_index:
                msg = f"Already at stage {index}: {self.stages[index].name}."
                success = True
            elif self.stages[index].is_skipped():
                msg = f"Can not goto the skipped stage"
            elif self.stages[index].is_reached():
                self.stage_index = index
                msg = f"Changed to stage {index}: {self.stages[index].name} success."
                success = True
            else:
                msg = f"Stage {index} is not reached yet. Can only go to stages that have been reached. You can use tool `ToolStaus` to find all reached stages."
        else:
            msg = f"Invalid stage index: {index}. No change made."
        return {"message": msg, "success": success}

    def force_go_to_stage(self, index):
        """
        Force go to a specific stage by index, ignoring whether it is reached or not.
        This is used when initializing the StageManager with a specific stage index.
        """
        if 0 <= index < len(self.stages):
            self.stage_index = index
            return True
        return False

    def check(self, timeout):
        if not self.stage_index < len(self.stages):
            return OrderedDict({
                "check_pass": False,
                "check_info": f"Stage index{self.stage_index} out of range. (Mission maybe completed, you can use the `GoToStage` tool to go back to a previous stage if needed)",
            })
        ck_pass, ck_info = self.stages[self.stage_index].do_check(**{"timeout": timeout})
        ret_data = OrderedDict({
            "check_info": ck_info,
            "check_pass": ck_pass,
        })
        if not ck_pass:
            ret_data["action"] = "Please fix the issues reported in 'check_info.last_msg.error' according to the suggestions, and then use the `Check` tool again to re-validate your work."
        self.last_check_info = copy.deepcopy(ret_data)
        if ck_pass:
            ret_data["message"] = f"Congratulations! Stage {self.stage_index} checks passed successfully, you can use tool 'Complete' to finish this stage."
        return ret_data

    def save_stage_info(self):
        info = self.agent.get_stat_info()
        info.update({
            "stage_index": self.stage_index,
            "all_completed": self.all_completed,
            "time_begin": self.time_begin,
            "time_end": self.time_end,
            "is_agent_exit": self.agent.is_exit(),
        })
        info["stages_info"] = {}
        for idx in range(self.stage_index + 1):
            if idx >= len(self.stages):
                break
            stage = self.stages[idx]
            stage_info = stage.detail()
            stage_info["time_cost"] = stage.get_time_cost()
            info["stages_info"][idx] = stage_info
        stage = self.get_current_stage()
        is_wait_human_check = False
        if stage:
            is_wait_human_check = stage.is_wait_human_check()
        info["is_wait_human_check"] = is_wait_human_check
        fc.save_ucagent_info(self.workspace, info)

    def next_stage(self):
        self.stage_index += 1
        self._go_skip_stage()
        self.save_stage_info()

    def _reset_stage_token_start(self):
        if not self.enable_data_collection:
            self._stage_token_start = None
            self._stage_msg_in_start = None
            self._stage_msg_out_start = None
            return
        try:
            total = self.agent.backend.token_total()
            stats = self.agent.backend.get_statistics()
        except Exception:
            total = None
            stats = {}
        self._stage_token_start = total if isinstance(total, int) and total >= 0 else None
        msg_in = stats.get("message_in") if isinstance(stats, dict) else None
        msg_out = stats.get("message_out") if isinstance(stats, dict) else None
        self._stage_msg_in_start = msg_in if isinstance(msg_in, (int, float)) and msg_in >= 0 else None
        self._stage_msg_out_start = msg_out if isinstance(msg_out, (int, float)) and msg_out >= 0 else None

    def _go_skip_stage(self):
        if self.stage_index >= len(self.stages):
            return
        sk = 0
        while self.stages[self.stage_index].is_skipped():
            self.stage_index += 1
            sk += 1
            if self.stage_index >= len(self.stages):
                break
        if sk > 0:
            info(f"skipped {sk} stages, current stage index is now {self.stage_index}.")

    def skip_stage(self, index):
        if 0 <= index < len(self.stages):
            self.stages[index].set_skip(True)
            info(f"Stage '{self.stages[index].name}' is set to be skipped.")
            if index == self.stage_index:
                self.next_stage()
        else:
            warning(f"Invalid stage index: {index}, can not set skip.")

    def unskip_stage(self, index):
        if 0 <= index < len(self.stages):
            self.stages[index].set_skip(False)
            info(f"Stage '{self.stages[index].name}' is set to be unskipped.")
        else:
            warning(f"Invalid stage index: {index}, can not set unskip.")

    def complete(self, timeout):
        if self.stage_index >= len(self.stages):
            return {
                "complete": False,
                "message": ("No more stages to complete. You can review your work and use the `GoToStage` tool to go back to a previous stage if needed. "
                            "Or you can use the `Exit` tool to exit the mission."),
                "last_check_result": self.last_check_info,
            }
        ck_pass, ck_info = self.stages[self.stage_index].do_check(**{"timeout": timeout, "is_complete": True})
        self.last_check_info = OrderedDict({
            "check_info": ck_info,
            "check_pass": ck_pass,
        })
        if ck_pass:
            message = f"Stage {self.stage_index} completed successfully. "
            if self.enable_data_collection:
                current_total = self.agent.backend.token_total()
                stats = self.agent.backend.get_statistics()
                msg_in = stats.get("message_in") if isinstance(stats, dict) else None
                msg_out = stats.get("message_out") if isinstance(stats, dict) else None
                if isinstance(current_total, int) and current_total >= 0 and self._stage_token_start is not None:
                    delta = current_total - self._stage_token_start
                    info(
                        f"[data_collection] Stage {self.stage_index} ({self.stages[self.stage_index].title()}) "
                        f"token_used_total={delta} total_tokens={current_total}"
                    )
                if (
                    isinstance(msg_in, (int, float))
                    and isinstance(msg_out, (int, float))
                    and self._stage_msg_in_start is not None
                    and self._stage_msg_out_start is not None
                ):
                    delta_in = msg_in - self._stage_msg_in_start
                    delta_out = msg_out - self._stage_msg_out_start
                    info(
                        f"[data_collection] Stage {self.stage_index} ({self.stages[self.stage_index].title()}) "
                        f"token_in={int(delta_in)} token_out={int(delta_out)} "
                        f"token_in_total={int(msg_in)} token_out_total={int(msg_out)}"
                    )
            self.stages[self.stage_index].on_complete()
            self.next_stage()
            if self.stage_index >= len(self.stages):
                message = ("All stages completed successfully. "
                           "Now you should review your work to check if everything is correct and all the users needs are matched. "
                           "When you are confident that everything is fine, you can use the `Exit` tool to exit the mission. "
                           )
                self.all_completed = True
            else:
                message += f"Current stage index is now {self.stage_index}. Use `CurrentTips` tool to get your new task. "
                self.stages[self.stage_index].set_reached(True)
                self.stages[self.stage_index].on_init()
                self._update_stage_context(self.stages[self.stage_index])
                self._reset_stage_token_start()
        else:
            message = f"Stage {self.stage_index} not completed. Please check the task requirements."
        ret = OrderedDict({
            "complete": ck_pass,
            "message": message,
            "last_check_result": self.last_check_info,
        })
        if not ck_pass:
            ret["action"] = "Please fix the issues reported in 'last_check_result.check_info.last_msg.error' according to the suggestions, and then use the `Complete` tool again to complete this stage."
        return ret

    def exit(self):
        """
        Exit the agent and end the mission after all stages are completed.
        """
        if self.all_completed:
            self.time_end = time.time()
            self.agent.exit()  # Exit the agent if all stages are completed
            self.save_stage_info()
            if getattr(self.agent, "enable_long_term_memory", False) and getattr(self.agent, "long_term_memory", None):
                self.agent.long_term_memory.archive_completed()
            self.agent.try_exit_on_completion()
            ex_msg = ""
            if self.agent._exit_on_completion:
                ex_msg = " UCAgent has quit. The MCP server is shutting down — all MCP tools will become unavailable. You need to stop Now!"
            return {
                "exit": True,
                "message": "All stages completed. Exiting the mission." + ex_msg
            }
        return {
            "exit": False,
            "message": "Not all stages are completed yet. Please complete all stages before exiting."
        }

    def tool_detail(self):
        """
        Get the details of the current mission, including all stages and their details.
        """
        detail = make_llm_tool_ret(self.detail())
        info("ToolDetail:\n" + detail)
        return self.attach_todo_summary(detail)

    def tool_status(self):
        stat = make_llm_tool_ret(self.status())
        info("ToolStatus:\n" + stat)
        return self.attach_todo_summary(stat)

    def tool_go_to_stage(self, index):
        ret = make_llm_tool_ret(self.go_to_stage(index))
        info("ToolGoToStage:\n" + ret)
        return self.attach_todo_summary(ret)

    def tool_check(self, timeout):
        ret = make_llm_tool_ret(self.check(timeout))
        info("ToolCheck:\n" + ret)
        return self.attach_todo_summary(ret)

    def tool_exit(self):
        ret = make_llm_tool_ret(self.exit())
        info("ToolExit:\n" + ret)
        return ret

    def tool_complete(self, timeout):
        ret = make_llm_tool_ret(self.complete(timeout))
        info("ToolComplete:\n" + ret)
        return self.attach_todo_summary(ret)

    def tool_kill_check(self):
        """
        Kill the current check process.
        This is used when the tool 'Check' is long time running or get stuck.
        """
        if not self.stage_index < len(self.stages):
            return f"Stage index({self.stage_index}) out of range. (Maybe mission is completed, you can use the `GoToStage` tool to go back to a previous stage if needed)"
        stage = self.stages[self.stage_index]
        ret = stage.do_kill()
        info("KillCheck:\n" + ret)
        return ret

    def tool_std_check(self, lines=-1):
        """
        Get the standard output of the current check process.
        This tool is only used to get the output of the running tool 'Check'.
        You can specify the number of lines to read, -1 means read all lines.
        """
        if not self.stage_index < len(self.stages):
            return f"Stage index({self.stage_index}) out of range. (Maybe mission is completed, you can use the `GoToStage` tool to go back to a previous stage if needed)"
        stage = self.stages[self.stage_index]
        ret = stage.do_std(lines)
        info("StdCheck:\n" + ret)
        return ret

    def tool_current_tips(self):
        """
        Get the tips for the current task.
        This is used to provide guidance to the user on what to do next.
        """
        tips = self.get_current_tips()
        info("Tips:\n" + tips)
        return tips

    def tool_run_test_cases(self, pytest_args="", timeout=0, return_line_coverage=False, raw_return=False, detail=False):
        """
        Run test cases.
        This tool is used to execute the test cases in the workspace.
        """
        ret = self.free_pytest_run.do_check(pytest_args, timeout=timeout, return_line_coverage=return_line_coverage, detail=detail)
        if raw_return:
            return ret
        ret = make_llm_tool_ret(ret[1])
        info("RunTestCases:\n" + ret)
        return self.attach_todo_summary(ret)
