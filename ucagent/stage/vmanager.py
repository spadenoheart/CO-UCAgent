# -*- coding: utf-8 -*-
"""Verification manager for UCAgent stage execution."""

import copy
import os
import time
import traceback
import random
import re
from collections import Counter, OrderedDict
from typing import Optional, Callable, Dict, Any

from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, ConfigDict, Field

import ucagent.util.functions as fc
from ucagent.checkers import UnityChipCheckerTestFree
from ucagent.memory.context_reuse import (
    classify_failure_pattern,
    failure_pattern_group,
    failure_signature_from_summary,
    preferred_action_categories_for_failure,
)
from ucagent.stage.vstage import get_root_stage
from ucagent.tools.uctool import UCTool, EmptyArgs
from ucagent.util.functions import make_llm_tool_ret
from ucagent.util.log import info, warning
from ucagent.util.test_result import (
    TEST_OUTCOME_PASS,
    apply_test_outcome,
    stage_allows_expected_failures,
)
from ucagent.stage.llm_suggestion.base_suggestion import get_llm_check_instance
from ucagent.tools.skill import _list_skills, list_skills_in_format


class ManagerTool(UCTool):
    # custom vars
    function: Callable = None
    args_schema: Optional[ArgsSchema] = EmptyArgs

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.function()

    def set_function(self, func):
        self.function = func
        return self


class ArgApproveStagePass(BaseModel):
    pass_or_not: bool = Field(
        default=True,
        description="Indicates whether to pass or not current stage task, True means pass and False means fail. Default is True."
    )


class ApproveStagePass(ManagerTool):
    """Approve the current stage as passed."""
    name: str = "ApproveStagePass"
    args_schema: Optional[ArgsSchema] = ArgApproveStagePass
    description: str = (
        "Approve the current stage as passed. \n"
        "This tool is used when you have verified that the current stage has been properly completed. \n"
    )
    def _run(self, pass_or_not: bool = True, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.function(pass_or_not)


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


class ToolStageJournal(ManagerTool):
    """Get the journal of the current stage."""
    name: str = "StageJournal"
    description: str = (
        "Get the journal of the current stage. "
    )


class ToolAllStageJournal(ManagerTool):
    """get the journal of the all stages."""
    name: str = "AllStageJournal"
    description: str = (
        "Get the journal of all stages. \n"
        "This tool is used to when continue a previous mission or the LLM context is compressed and other similar situations. "
    )


class ArgSetCurrentStageJournal(BaseModel):
    journal: str = Field(
        description="The journal content to set for the current stage. Cannot be empty."
    )


class ToolSetCurrentStageJournal(ManagerTool):
    """set the journal of the current stage."""
    name: str = "SetCurrentStageJournal"
    description: str = (
        "Set the journal of the current stage. \n"
        "This tool is used to record important information during the current stage. When completing the stage, the journal should be set. \n"
        "The journal content should be concise and clear and only the necessary information should be included.\n"
        "eg: - What you have done in this stage.\n"
        "    - What problems you have encountered and how you solved them.\n"
        "    - Experience or lessons learned during this stage.\n"
        "    - Files and its comments you have created or modified in this stage.\n"
        "    - Things to note when re-continuing this stage in the future.\n"
    )
    args_schema: Optional[ArgsSchema] = ArgSetCurrentStageJournal

    def _run(self, journal: str = "",
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not journal:
            return "Journal content cannot be empty."
        return self.function(journal)

class ArgSkillUsage(BaseModel):
    skill_usage: Dict[str, Any] = Field(
        description="The skill usage to set for the current stage. Cannot be empty."
    )

class ToolSetSkillUsage(ManagerTool):
    """Check and set the skill usage of the current stage."""
    name: str = "SetSkillUsage"
    description: str = (
        "Check the usage of the skills and set journal of the current stage. \n"
        "Analyze the conversation history and check usage of the skills specified in skill_list (if skills beyond the specified list were also used, analyze them as well).\n"
        "For each skill, analyze the following aspects:\n"
        "1. **list**: Whether the name and description of skill was listed in histoty context\n"
        "2. **read**: Whether the SKILL.md of skill was read by using tool `ReadTextFile`\n"
        "3. **use**: Whether completion of the current stage task followed the method steps in SKILL.md, or executed any specified code in that file\n"
        "**Returned dictionary format example**:\n"
        "{\n"
        "  'unitytest/ut-functions-and-checks': {'list': True, 'read': True, 'use': False},\n"
        "  'ext/custom/skill-name': {'list': True, 'read': False, 'use': False}\n"
        "}\n"
    )
    args_schema: Optional[ArgsSchema] = ArgSkillUsage

    def _run(self, skill_usage: Dict[str, Any] = None, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not skill_usage:
            return "Skill usage content cannot be empty, use tool `ToolSetSkillUsage` to check the skill usage and set the skill usage content."
        return self.function(skill_usage)


class ArgStageDiff(BaseModel):
    target_file: str = Field(".", description="The target file or path to get diff, default is current workspace directory.")
    show_detail: bool = Field(False, description="Whether to show detailed diff output, default is False.")
    start_line: int = Field(1, description="The starting line number for diff output, default is 1")
    line_count: int = Field(-1, description="The number of lines to show in the diff output, default is -1 (show all lines)")


class ToolStageDiff(ManagerTool):
    """Retrieve the differences between the current file and the file at the last `StageCommit`."""
    name: str = "StageDiff"
    description: str = (
        "Get the differences between the current file and the file at the last `StageCommit`. \n"
        "This tool helps you to identify changes made since the last commit in the current stage. \n"
        "Use this tool to review modifications before proceeding with further actions. \n"
    )
    args_schema: Optional[ArgsSchema] = ArgStageDiff
    def _run(self, target_file: str = ".",
             show_detail: bool = False,
             start_line: int = 1,
             line_count: int = -1,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.function(target_file, show_detail, start_line, line_count)


class ArgStageCommit(BaseModel):
    commit_message: str = Field(..., description="The commit message for the changes in the current stage.")


class ToolStageCommit(ManagerTool):
    """Commit current stage changes."""
    name: str = "StageCommit"
    description: str = (
        "Commit current stage changes. \n"
        "This tool records your progress and changes made during the current stage. \n"
        "Use this tool to ensure that all modifications are saved before proceeding with further actions. \n"
        "When called this tool, all changes in the current stage will be committed with the provided commit message and "
        "you cannot undo this action. \n"
    )
    args_schema: Optional[ArgsSchema] = ArgStageCommit
    def _run(self, commit_message: str = "",
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not commit_message:
            return "Commit message cannot be empty."
        return self.function(commit_message)


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


CHECK_EXTRA_ARGS_DESCRIPTION = (
    "Additional checker-specific arguments are allowed. Pass them as top-level "
    "JSON fields alongside timeout, not wrapped in args/check_args. Structured "
    "values such as objects or arrays must be passed as real JSON values, not as "
    "stringified JSON. The accepted extra argument names depend on the current "
    "stage checker and may be described by the task prompt or checker error message."
)


class ArgsDoCheck(BaseModel):
    """Arguments for Check/Complete; checker-specific top-level extras are allowed."""

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "description": (
                "Arguments for Check/Complete. Besides timeout, this schema accepts "
                "checker-specific extra top-level JSON fields."
            ),
            "additionalProperties": {
                "description": CHECK_EXTRA_ARGS_DESCRIPTION
            },
        },
    )

    timeout: int = Field(
        default=0,
        description=(
            "Timeout for Check/Complete tools. Zero means use default cfg.call_time_out. "
            "Checker-specific extra arguments may also be passed as sibling top-level "
            "JSON fields alongside timeout; pass object/array values as real JSON, "
            "not strings."
        )
    )
    generated: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Random-test CK processing records for the current batch. This field is "
            "required by the generate_random_test_cases stage when Check/Complete asks "
            "for generated records. Pass a real JSON object whose keys are exact "
            "FG-*/FC-*/CK-* paths from the current batch and whose values briefly "
            "describe the generated test or the reason that random testing is not useful. "
            "Do not put this argument on pytest test functions."
        ),
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


class ToolDoCheck(ManagerTool):
    """Advanced validation tool for stage requirements and implementation quality."""
    name: str = "Check"
    description: str = (
        "Perform comprehensive validation of your current stage's implementation against requirements.\n"
        "The tool provides detailed feedback.\n"
        "You may pass additional checker-specific arguments as extra JSON fields; "
        "they will be forwarded to the current stage checkers."
    )
    args_schema: Optional[ArgsSchema] = ArgsDoCheck

    def _run(self, timeout=0, run_manager: Optional[CallbackManagerForToolRun] = None, **check_args) -> str:
        """
        Execute stage validation with enhanced error handling and reporting.

        Args:
            timeout: Check timeout in seconds.
            **check_args: Additional keyword args passed to checkers.
            run_manager: Callback manager for tool execution

        Returns:
            str: Comprehensive validation report in JSON format
        """
        try:
            if check_args.get("generated") is None:
                check_args.pop("generated", None)
            if timeout <= 0:
                timeout = self.get_call_time_out()
            return self.function(timeout, **check_args)
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
        "Perform comprehensive validation of your current stage's implementation against requirements and mark the stage as complete if all checks pass.\n"
        "The tool provides detailed feedback (Different from tool 'Check': if all checks pass, the stage is marked as complete and the manager advances to the next stage).\n\n"
        "You may pass additional checker-specific arguments as extra JSON fields; "
        "they will be forwarded to the current stage checkers. "
        "The 'is_complete' flag is reserved and is always set to true by this tool."
    )
    args_schema: Optional[ArgsSchema] = ArgsDoCheck

    def _run(self, timeout=0, run_manager: Optional[CallbackManagerForToolRun] = None, **check_args) -> str:
        try:
            if check_args.get("generated") is None:
                check_args.pop("generated", None)
            if timeout <= 0:
                timeout = self.get_call_time_out()
            return self.function(timeout, **check_args)
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
            self, workspace, cfg, agent, tool_read_text, ucagent_info: Optional[dict] = None,
            force_stage_index=0,
            force_todo=False,
            todo_panel=None,
            stage_skip_list=None,
            stage_unskip_list=None,
            tool_inspect_file=None,
            reference_files=None,
            force_stage_index_explicit=False,
    ):
        """
        Initialize the StageManager with an empty list of stages.
        """
        self.cfg = cfg
        self.workspace = workspace
        self.force_todo = force_todo
        self.todo_panel = todo_panel
        self.free_pytest_run = UnityChipCheckerTestFree("", cfg.tools.RunTestCases.test_dir, "").set_workspace(workspace)
        self.agent = agent
        self.enable_data_collection = self._cfg_value(cfg, "context_upgrade.enable_data_collection", False)
        self.long_term_memory_cfg = self._cfg_value(cfg, "context_upgrade.long_term_memory", {}) or {}
        if hasattr(self.long_term_memory_cfg, "as_dict"):
            self.long_term_memory_cfg = self.long_term_memory_cfg.as_dict()
        elif not isinstance(self.long_term_memory_cfg, dict):
            self.long_term_memory_cfg = {}
        self.context_reuse_cfg = self._cfg_value(cfg, "context_upgrade.context_reuse", {}) or {}
        if hasattr(self.context_reuse_cfg, "as_dict"):
            self.context_reuse_cfg = self.context_reuse_cfg.as_dict()
        elif not isinstance(self.context_reuse_cfg, dict):
            self.context_reuse_cfg = {}
        self._stage_token_start = None
        self._stage_msg_in_start = None
        self._stage_msg_out_start = None
        self._ltm_prompt_cache = {}
        self._ltm_prefetch_cache = {}
        self._stage_prompt_seen = 0
        self._recent_failure_context = None
        self._recent_failure_seq = 0
        self._stage_state_runtime = {
            "stage_index": None,
            "generation": 0,
            "last_test": None,
            "last_checker": None,
            "last_control": None,
        }
        self._context_reuse_injected_keys = set()
        self._context_reuse_injected_key_counts = Counter()
        self._memory_metrics = {
            "stage_entries": 0,
            "prompt_queries": 0,
            "stage_cache_hits": 0,
            "prefetch_triggers": 0,
            "prefetch_fills": 0,
            "prefetch_hits": 0,
            "stage_first_turn_prefetch_hits": 0,
            "retrieval_hits": 0,
            "retrieval_misses": 0,
            "recent_fallback_hits": 0,
            "prompt_injections": 0,
            "injected_items": 0,
            "stale_hits": 0,
            "context_reuse_queries": 0,
            "context_reuse_hits": 0,
            "context_reuse_hint_hits": 0,
            "context_reuse_injections": 0,
            "context_reuse_items": 0,
            "context_reuse_throttled": 0,
        }
        self.tool_read_text = tool_read_text
        self.ucagent_info = ucagent_info or {}
        self.data = self.ucagent_info.get("stage_data", {})
        self.force_stage_index = force_stage_index
        self.force_stage_index_explicit = force_stage_index_explicit
        self._saved_info_truncated_by_force = False
        self.stage_skip_list = stage_skip_list
        self.stage_unskip_list = stage_unskip_list
        self.tool_inspect_file = tool_inspect_file
        self.reference_files = reference_files

    @staticmethod
    def _safe_int(value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _cfg_value(cfg, path: str, default=None):
        """Read a dotted config path from Config, dict, or test namespaces."""
        get_value = getattr(cfg, "get_value", None)
        if callable(get_value):
            return get_value(path, default)
        value = cfg
        for part in path.split("."):
            if isinstance(value, dict):
                if part not in value:
                    return default
                value = value[part]
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                return default
        return value

    def _truncate_loaded_info_for_forced_stage(self, force_stage_index):
        if not self.force_stage_index_explicit:
            return False
        saved_stage_index = self._safe_int(self.ucagent_info.get("stage_index", 0), 0)
        if force_stage_index >= saved_stage_index:
            return False

        warning(
            f"Force stage index {force_stage_index} is earlier than saved stage index "
            f"{saved_stage_index}; clearing saved progress from stage {force_stage_index} onward."
        )
        self.ucagent_info = copy.deepcopy(self.ucagent_info)
        stages_info = self.ucagent_info.get("stages_info", {})
        if isinstance(stages_info, dict):
            self.ucagent_info["stages_info"] = {
                stage_idx: stage_info
                for stage_idx, stage_info in stages_info.items()
                if self._safe_int(stage_idx, -1) < force_stage_index
            }
        self.ucagent_info["stage_index"] = force_stage_index
        self.ucagent_info["all_completed"] = False
        self.ucagent_info["time_end"] = None
        self.ucagent_info["is_agent_exit"] = False
        self.ucagent_info["is_wait_human_check"] = False
        return True

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
        self.stage_index = min(max(0, self.force_stage_index), len(self.stages))
        self._saved_info_truncated_by_force = self._truncate_loaded_info_for_forced_stage(self.stage_index)
        for i in range(min(self.stage_index + 1, len(self.stages))):
            self.stages[i].set_reached(True)
        stages_info = self.ucagent_info.get("stages_info", {})

        for stage_idx_str, stage_info in stages_info.items():
            idx = int(stage_idx_str)
            if idx >= len(self.stages):
                continue
            stage: VerifyStage = self.stages[idx]
            stage.set_fail_count(stage_info.get("fail_count", 0))
            stage.set_time_prev_cost(stage_info.get("time_cost", 0.0))
            stage.set_skip(stage_info.get("is_skipped", stage.is_skipped()))
            if idx < self.stage_index:
                stage.is_complete = stage_info.get("is_completed", stage.is_completed())
            stage.set_reference_file_status(stage_info.get("task", {}).get("reference_files", {}))
            if "meta_data" in stage_info:
                stage.meta_data = copy.deepcopy(stage_info["meta_data"])
        self._go_skip_stage()
        for s in self.stages:
            s.set_stage_manager(self)
        if self.stage_index < len(self.stages):
            self.stages[self.stage_index].on_init()
            self._on_stage_enter(self.stages[self.stage_index], trigger_prefetch=True)
        self.last_check_info = {}
        if self.stage_skip_list:
            for si in self.stage_skip_list:
                self.skip_stage(si)
                info(f"Stage {si} is set to be skipped.")
        if self.stage_unskip_list:
            for sui in self.stage_unskip_list:
                self.unskip_stage(sui)
                info(f"Stage {sui} is set to be unskipped.")
        self._refresh_all_completed()
        info("Current stage index is " + str(self.stage_index) + ".")
        self.time_begin = self.ucagent_info.get("time_begin", time.time())
        self.time_end = self.ucagent_info.get("time_end", None)
        self.llm_fail_suggestion = get_llm_check_instance(
            self.cfg.vmanager.llm_suggestion.check_fail_refinement,
            self,
            self.tool_inspect_file
        )
        self.llm_pass_suggestion = get_llm_check_instance(
            self.cfg.vmanager.llm_suggestion.check_pass_refinement,
            self,
            self.tool_inspect_file + [ApproveStagePass().set_function(self.tool_stage_approve),
                                      ToolStageDiff().set_function(self.tool_stage_diff),
                                      ToolStageCommit().set_function(self.tool_stage_commit)]
        )
        if self.stage_index < len(self.stages):
            self.stages[self.stage_index].hist_init()
        self.save_stage_info()

    def is_break(self):
        return self.agent.is_break()

    def tool_stage_approve(self, pass_or_not: bool = True) -> str:
        vstage = self.get_current_stage()
        if vstage is None:
            return "No current stage available."
        vstage.set_approved(pass_or_not)
        return f"Stage '{vstage.name}' approved status set to {pass_or_not}."

    def tool_stage_diff(self, target_file, show_detail, start_line, line_count) -> str:
        vstage = self.get_current_stage()
        if vstage is None:
            return "No current stage available."
        return vstage.hist_diff(target_file, show_detail, start_line, line_count)

    def tool_stage_commit(self, commit_message) -> str:
        vstage = self.get_current_stage()
        if vstage is None:
            return "No current stage available."
        vstage.hist_commit(commit_message)
        return f"Stage '{vstage.name}' changes committed."

    def gen_fail_suggestion(self, error_msg) -> str:
        stage = self.get_current_stage()
        if stage is None:
            return error_msg
        try:
            fail_suggestion = self.gen_llm_suggestion(
                error_msg,
                stage,
                self.llm_fail_suggestion,
                self.stage_need_llm_fail_suggestion,
            )
            stage.meta_set_llm_fail_suggestion(fail_suggestion)
            return fail_suggestion
        except Exception as e:
            traceback.print_exc()
            warning(f"Generate fail suggestion failed: {str(e)}")
            return error_msg

    def is_llm_suggestion_needed(self, stage_name, suggestion_ins, need_llm_suggestion) -> bool:
        if not suggestion_ins:
            return False
        if need_llm_suggestion is not None:
            return need_llm_suggestion
        cfg = suggestion_ins.get_cfg()
        bypass_stages = cfg.get("bypass_stages", [])
        if bypass_stages:
            if stage_name in bypass_stages:
                return False
        target_stages = cfg.get("target_stages", [])
        if target_stages:
            if stage_name not in target_stages:
                return False
        return cfg.get("default_apply_all_stages", True)

    def stage_need_llm_pass_suggestion(self, stage) -> bool:
        if stage is None:
            return False
        return self.is_llm_suggestion_needed(
            stage.name,
            self.llm_pass_suggestion,
            stage.need_pass_llm_suggestion,
        )

    def stage_need_llm_fail_suggestion(self, stage) -> bool:
        if stage is None:
            return False
        return self.is_llm_suggestion_needed(
            stage.name,
            self.llm_fail_suggestion,
            stage.need_fail_llm_suggestion,
        )

    def gen_pass_suggestion(self, raw_msg) -> str:
        stage = self.get_current_stage()
        if stage is None:
            return raw_msg
        try:
            pass_suggestion = self.gen_llm_suggestion(
                raw_msg,
                stage,
                self.llm_pass_suggestion,
                self.stage_need_llm_pass_suggestion,
                pre_llm_cb = lambda: stage.set_approved(False),
            )
            stage.meta_set_llm_pass_suggestion(pass_suggestion)
            return pass_suggestion
        except Exception as e:
            traceback.print_exc()
            warning(f"Generate pass suggestion failed: {str(e)}")
            warning("Set stage as approved due to exception in generating pass suggestion.")
            stage.set_approved(True)
            return raw_msg

    def gen_llm_suggestion(self, raw_msg, stage,
                           suggestion_instance,
                           stage_need_llm_suggestion_fc,
                           pre_llm_cb=None) -> str:
        assert stage is not None, "stage can not be None"
        if stage_need_llm_suggestion_fc(stage) is False:
            return raw_msg
        if callable(pre_llm_cb):
            pre_llm_cb()
        stage.set_force_unactive(True)
        try:
            return suggestion_instance.suggest([
                    stage.task_info(),
                    raw_msg],
                    stage)
        finally:
            stage.set_force_unactive(False)

    def get_time_cost(self):
        if self.time_end is None:
            return time.time() - self.time_begin
        return self.time_end - self.time_begin

    def _update_stage_context(self, stage):
        agent = getattr(self, "agent", None)
        msg_node = getattr(agent, "message_manage_node", None)
        if msg_node is None or not hasattr(msg_node, "set_stage_context"):
            return
        try:
            detail = stage.detail()
            section_index = detail.get("section_index", "")
            progress = f"{self.stage_index}/{len(self.stages)}"
            msg_node.set_stage_context(self.stage_index, stage.title(), section_index, progress)
            self._publish_stage_state_package("stage_context", {}, None)
        except Exception:
            return

    @staticmethod
    def _compact_stage_observation(summary):
        if not isinstance(summary, dict):
            return None
        compact = {}
        for key in (
            "test_outcome",
            "test_outcome_reason",
            "tests_total",
            "tests_failed",
            "tests_passed_all",
            "check_pass",
            "run_test_success",
        ):
            if key in summary:
                compact[key] = summary.get(key)
        for key in ("failed_cases_top", "failed_checkpoints_top", "checker_categories"):
            value = summary.get(key)
            if isinstance(value, list):
                compact[key] = [str(item)[:300] for item in value[:8]]
        errors = summary.get("error_top")
        if isinstance(errors, list):
            compact["error_top"] = [str(item)[:400] for item in errors[:3]]
        return compact

    def _publish_stage_state_package(self, event_type, payload, decision):
        agent = getattr(self, "agent", None)
        if not getattr(agent, "enable_stage_state_package", False):
            return
        msg_node = getattr(agent, "message_manage_node", None)
        if msg_node is None or not hasattr(msg_node, "update_stage_state_package"):
            return
        if not (0 <= self.stage_index < len(self.stages)):
            return
        event_stage = payload.get("stage_index") if isinstance(payload, dict) else None
        if isinstance(event_stage, int) and event_stage != self.stage_index:
            return
        runtime = self._stage_state_runtime
        if runtime.get("stage_index") != self.stage_index:
            runtime.update({
                "stage_index": self.stage_index,
                "generation": 0,
                "last_test": None,
                "last_checker": None,
                "last_control": None,
            })
        summary = payload.get("result") if isinstance(payload, dict) else None
        if event_type == "test_run":
            runtime["last_test"] = self._compact_stage_observation(summary)
        elif event_type == "check_result":
            runtime["last_checker"] = self._compact_stage_observation(summary)
        if isinstance(decision, dict):
            runtime["last_control"] = {
                "credit_role": decision.get("credit_role"),
                "policy": decision.get("policy"),
                "checker_no_progress_streak": decision.get("checker_no_progress_streak", 0),
                "failure_set_delta": decision.get("failure_set_delta", {}),
                "control": decision.get("control", {}),
                "mutation_budget": decision.get("mutation_budget", {}),
            }
        runtime["generation"] = int(runtime.get("generation", 0) or 0) + 1

        stage = self.stages[self.stage_index]
        detail = stage.detail()
        task = detail.get("task") if isinstance(detail.get("task"), dict) else {}
        reference_files = task.get("reference_files") if isinstance(task.get("reference_files"), dict) else {}
        def _is_unread(status):
            if isinstance(status, bool):
                return not status
            if status is None:
                return True
            text = str(status).strip().lower().replace("_", " ")
            return text in {"", "0", "false", "not read", "unread", "not readed"}

        unread = [str(path) for path, status in reference_files.items() if _is_unread(status)]
        checker_contract = detail.get("checker") or []
        if not isinstance(checker_contract, list):
            checker_contract = [checker_contract]
        last_checker = runtime.get("last_checker")
        last_test = runtime.get("last_test")
        unresolved = None
        if isinstance(last_checker, dict) and last_checker.get("check_pass") is False:
            unresolved = {"source": "checker", **last_checker}
        elif isinstance(last_test, dict) and last_test.get("test_outcome") != TEST_OUTCOME_PASS:
            unresolved = {"source": "test", **last_test}
        package = OrderedDict({
            "schema_version": 1,
            "stage": {
                "index": self.stage_index,
                "name": getattr(stage, "name", ""),
                "title": stage.title(),
                "progress": f"{self.stage_index}/{len(self.stages)}",
            },
            "task_contract": {
                "reference_files_unread": unread[:10],
                "output_files": [str(item) for item in (task.get("output_files") or [])[:10]],
                "checker": [str(item)[:400] for item in checker_contract[:8]],
            },
            "last_test": last_test,
            "last_checker": last_checker,
            "unresolved": unresolved,
            "progress_control": runtime.get("last_control"),
            "generation": runtime["generation"],
            "updated_by": event_type,
        })
        msg_node.update_stage_state_package(package)
        if getattr(agent, "enable_data_collection", False):
            agent.record_structured_event("stage_state_package_update", {
                "generation": runtime["generation"],
                "updated_by": event_type,
                "has_test": bool(last_test),
                "has_checker": bool(last_checker),
                "has_unresolved": bool(unresolved),
            })

    def _compute_all_completed(self):
        if not self.stages:
            return True
        if self.stage_index >= len(self.stages):
            return True
        return all(stage.is_skipped() or stage.is_completed() for stage in self.stages)

    def _refresh_all_completed(self):
        self.all_completed = self._compute_all_completed()
        return self.all_completed

    def _build_memory_query(self, stage_index: int, stage_title: str):
        query = f"{self.agent.dut_name} {stage_title}"
        filters = {"dut": self.agent.dut_name, "stage_index": stage_index}
        return query, filters

    def _prefetch_stage_memories(self, stage_index: int, stage=None):
        if not getattr(self.agent, "enable_long_term_memory", False):
            return []
        ltm = getattr(self.agent, "long_term_memory", None)
        if ltm is None:
            return []
        if not bool(self.long_term_memory_cfg.get("enable_stage_prefetch", True)):
            return []
        if not (0 <= stage_index < len(self.stages)):
            return []
        stage = stage or self.stages[stage_index]
        query, filters = self._build_memory_query(stage_index, stage.title())
        version = ltm.version()
        cache_key = (stage_index, query, filters.get("dut"), version)
        self._memory_metrics["prefetch_triggers"] += 1
        cached = self._ltm_prefetch_cache.get(stage_index)
        if cached and cached.get("cache_key") == cache_key:
            return copy.deepcopy(cached.get("memories", []))
        limit = int(self.long_term_memory_cfg.get("prefetch_limit", self.long_term_memory_cfg.get("prompt_limit", 3)) or 3)
        memories = []
        if hasattr(ltm, "prefetch_search"):
            memories = ltm.prefetch_search(
                query=query,
                stage_index=stage_index,
                stage_title=stage.title(),
                limit=limit,
                filters={"dut": self.agent.dut_name},
            )
        if not memories:
            memories = ltm.search(query=query, limit=limit, filters=filters)
        self._ltm_prefetch_cache[stage_index] = {
            "cache_key": cache_key,
            "query": query,
            "filters": filters,
            "memories": copy.deepcopy(memories),
        }
        if memories:
            self._memory_metrics["prefetch_fills"] += 1
            info(
                f"[long_term_memory][prefetch] stage={stage_index} "
                f"title='{stage.title()}' hits={len(memories)}"
            )
        return memories

    def _on_stage_enter(self, stage, trigger_prefetch=True):
        self._update_stage_context(stage)
        self._reset_stage_token_start()
        self._stage_prompt_seen = 0
        memory_metrics = getattr(self, "_memory_metrics", None)
        if isinstance(memory_metrics, dict):
            memory_metrics["stage_entries"] = memory_metrics.get("stage_entries", 0) + 1
        if trigger_prefetch:
            try:
                self._prefetch_stage_memories(self.stage_index, stage=stage)
            except Exception:
                pass

    def _reset_stage_token_start(self):
        if not getattr(self, "enable_data_collection", False):
            self._stage_token_start = None
            self._stage_msg_in_start = None
            self._stage_msg_out_start = None
            return
        try:
            agent = getattr(self, "agent", None)
            total = agent.backend.token_total()
            stats = agent.backend.get_statistics()
        except Exception:
            total = None
            stats = {}
        self._stage_token_start = total if isinstance(total, int) and total >= 0 else None
        msg_in = stats.get("message_in") if isinstance(stats, dict) else None
        msg_out = stats.get("message_out") if isinstance(stats, dict) else None
        self._stage_msg_in_start = msg_in if isinstance(msg_in, (int, float)) and msg_in >= 0 else None
        self._stage_msg_out_start = msg_out if isinstance(msg_out, (int, float)) and msg_out >= 0 else None

    def _select_prefetch_injection(self, prefetched):
        if not prefetched:
            return []
        inject_limit = int(self.long_term_memory_cfg.get("prefetch_inject_limit", 2) or 2)
        min_score = float(self.long_term_memory_cfg.get("prefetch_inject_min_score", 0.25) or 0.25)
        relative_score = float(self.long_term_memory_cfg.get("prefetch_relative_score", 0.82) or 0.82)
        max_per_source_stage = int(self.long_term_memory_cfg.get("prefetch_max_per_source_stage", 1) or 1)
        best_score = float(prefetched[0].get("__retrieval__", {}).get("score", 0.0) or 0.0)
        selected = []
        per_stage = {}
        for item in prefetched:
            score = float(item.get("__retrieval__", {}).get("score", 0.0) or 0.0)
            if score < min_score:
                continue
            if selected and best_score > 0 and score < best_score * relative_score:
                continue
            src_stage = item.get("meta", {}).get("stage_index")
            if src_stage is not None:
                if per_stage.get(src_stage, 0) >= max_per_source_stage:
                    continue
                per_stage[src_stage] = per_stage.get(src_stage, 0) + 1
            selected.append(item)
            if len(selected) >= inject_limit:
                break
        if not selected and prefetched:
            top = prefetched[0]
            if float(top.get("__retrieval__", {}).get("score", 0.0) or 0.0) >= min_score:
                return [top]
        return selected

    def attach_todo_summary(self, data):
        assert isinstance(data, str), "the target data type of attach_todo_summary must be str"
        if not self.force_todo:
            return data
        if not self.todo_panel:
            return data
        return data + self.todo_panel._summary()

    def _plain_cfg(self, value):
        if hasattr(value, "as_dict"):
            value = value.as_dict()
        return value if isinstance(value, dict) else {}

    def _bool_cfg(self, value, default=False):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(value)

    def _prompt_variant_cfg(self):
        return self._plain_cfg(self._cfg_value(self.cfg, "prompt_variant", {}))

    def _prompt_variant_name(self):
        return str(self._prompt_variant_cfg().get("name", "v0") or "v0").strip().lower()

    def _is_prompt_v1_enabled(self):
        return self._prompt_variant_name() in ("v1", "prompt_v1", "prompt-v1")

    def _prompt_guidance_for_stage(self, stage):
        if not self._is_prompt_v1_enabled() or stage is None:
            return None
        variant_cfg = self._prompt_variant_cfg()
        guidance_cfg = self._plain_cfg(variant_cfg.get("stage_guidance", {}))
        if not self._bool_cfg(guidance_cfg.get("enabled"), True):
            return None
        stage_indices = {
            int(x) for x in guidance_cfg.get("stage_indices", []) or []
            if isinstance(x, (int, float, str)) and str(x).strip().isdigit()
        }
        stage_names = {str(x) for x in guidance_cfg.get("stage_names", []) or []}
        has_selector = bool(stage_indices or stage_names)
        if has_selector and self.stage_index not in stage_indices and stage.name not in stage_names:
            return None
        stage_rules_by_name = self._plain_cfg(guidance_cfg.get("stages", {}))
        stage_rules = stage_rules_by_name.get(stage.name, []) or []
        if hasattr(stage_rules, "as_dict"):
            stage_rules = stage_rules.as_dict()
        return OrderedDict({
            "variant": "v1",
            "source": "prompt_variant.stage_guidance",
            "scope": {
                "stage_index": self.stage_index,
                "stage_name": stage.name,
                "stage_title": stage.title(),
            },
            "common_rules": guidance_cfg.get("common_rules", []) or [],
            "stage_rules": stage_rules,
            "checker_feedback_protocol": [
                "如果Check/Complete失败，先阅读checker_error_summary；若不存在，再从check_info.last_msg.error提取错误。",
                "修复前先写出内部fix_plan：错误类别、文件、测试、标签、修复动作、最小复验命令。",
                "修复后优先运行最小RunTestCases目标；通过后再调用Check或Complete。",
            ],
        })

    def _collect_text_items(self, value, path="", limit=80):
        items = []
        def visit(obj, cur_path):
            if len(items) >= limit:
                return
            if isinstance(obj, str):
                text = obj.strip()
                if text:
                    items.append((cur_path, text))
            elif isinstance(obj, dict):
                for key, val in obj.items():
                    next_path = f"{cur_path}.{key}" if cur_path else str(key)
                    visit(val, next_path)
            elif isinstance(obj, list):
                for idx, val in enumerate(obj):
                    visit(val, f"{cur_path}[{idx}]")
        visit(value, path)
        return items

    def _checker_error_summary(self, ret_data):
        if not self._is_prompt_v1_enabled():
            return None
        text_items = self._collect_text_items(ret_data)
        all_text = "\n".join(text for _, text in text_items).lower()
        categories = []
        actions = []

        def add(category, action):
            if category not in categories:
                categories.append(category)
                actions.append(action)

        stage = self.stages[self.stage_index] if getattr(self, "stages", None) else None
        stage_name = str(getattr(stage, "name", "") or "")
        has_template_placeholder = (
            "assertionerror: not implemented" in all_text
            or 'assert false, "not implemented"' in all_text
            or "assert false, 'not implemented'" in all_text
        )
        if "[template source contract]" in all_text:
            add(
                "template_source_contract_incomplete",
                "按Checker列出的源码位置一次性恢复所有模板骨架的活跃mark_function调用，第二个参数必须是当前测试函数对象；保留TODO和assert False/Not implemented。禁止创建conftest.py、删除tests/data或提前实现测试逻辑。",
            )
        if has_template_placeholder:
            if stage_name == "create_test_case_templates":
                add(
                    "template_validation_or_runner_contract",
                    "当前处于测试模板阶段，assert False/Not implemented是预期占位，禁止实现真实测试逻辑或将模板改成通过。若已收集到非零测试且失败均来自该占位，仅修正mark_function、TODO、模板结构或测试报告契约；若0 tests、收集错误、超时或崩溃，则先修复测试基础设施。",
                )
            else:
                add(
                    "test_template_not_implemented",
                    "当前测试仍是模板占位；先定位assert False/Not implemented并实现当前batch要求的测试逻辑，再运行最小RunTestCases目标。",
                )
        coverage_runtime_error = (
            ("attributeerror" in all_text or "has no attribute" in all_text or "nameerror" in all_text)
            and "function_coverage_def.py" in all_text
        )
        if coverage_runtime_error:
            add(
                "coverage_runtime_contract",
                "异常来自*_function_coverage_def.py的coverage callback。先读取RTL端口和coverage文件，一次性修正所有不存在的信号引用；最小测试确认采样回调不再抛异常前，不要改env/API或测试断言。",
            )
        elif "attributeerror" in all_text or "has no attribute" in all_text or "nameerror" in all_text:
            add(
                "api_or_env_interface_mismatch",
                "先读取当前API/env文件和RTL端口清单，确认真实属性/API名称；全局修正同类未定义属性或名称，不要把接口错误记录为DUT bug。",
            )
        if "importerror" in all_text or "modulenotfounderror" in all_text or "import errors in test files" in all_text:
            add(
                "test_import_error",
                "修正测试文件导入路径和from {DUT}_api import *等基础导入问题；这是测试环境错误，不要写入bug文档。",
            )
        if "do not contain assert" in all_text or "must contain at least one assert" in all_text:
            add(
                "missing_assert",
                "定位列出的测试函数，为每个可执行测试补充pytest assert或with pytest.raises；不要使用unittest断言。",
            )
        if "dict_keys(" in all_text or (
            "object at 0x" in all_text
            and "unmarked_check_point" in all_text
            and ("fg-" in all_text or "ck-" in all_text)
        ):
            add(
                "dynamic_label_used_in_coverage",
                "覆盖率FG/FC/CK必须使用功能文档中的字面量字符串；删除由对象repr、dict_keys或运行时拼接产生的动态标签。",
            )
        if "incomplete label" in all_text and "<tc-*>" in all_text:
            add(
                "bug_doc_incomplete_tc_label",
                "修复bug文档层级，确保每个已确认bug链路都有完整<FG>/<FC>/<CK>/<BG>/<TC>；没有真实失败TC时不要留下半截CK/BG标签。",
            )
        if (
            "bug analysis documentation parsing failed" in all_text
            and (
                "parent" in all_text
                or "proper nesting" in all_text
                or "must be preceded by" in all_text
            )
        ):
            add(
                "bug_doc_hierarchy_violation",
                "按Checker报告的行定位孤立标签，把它移动到已有的<FG>/<FC>/<CK>/<BG>合法父链下；不要追加第二份同名祖先。若多个FAILED测试来自同一RTL根因，只保留一个BG并把这些<TC-*>作为其子节点。",
            )
        if (
            "test case (<tc-" in all_text and "parse fail" in all_text
        ) or "test case format error" in all_text or (
            "<tc->" in all_text and "incorrect format" in all_text
        ):
            add(
                "bug_doc_tc_identifier_format",
                "先全文搜索并删除标题、说明文字或占位内容中的字面量<TC->；TC标签只能作为BG叶节点，格式为<TC-test_file.py::[ClassName::]test_name>，并与最新pytest报告中的真实FAILED node id一致。不要使用API名、描述文本或省略文件名。",
            )
        if (
            "should be ''failed'' but found to be ''passed''" in all_text
            or "should be 'failed' but found to be 'passed'" in all_text
            or (
                "[test case status mismatch]" in all_text
                and "must be failed" in all_text
            )
        ):
            add(
                "bug_doc_passed_tc_marked_as_bug",
                "若测试逻辑正确且最新报告为PASSED，使用RunSkillScript/prunebug.py删除该<TC-*>及空祖先；不要用recordbug.py追加‘已修复’或置信度0的BG，也不要把通过用例硬改成FAILED。",
            )
        checkpoint_not_found = (
            "[checkpoint not found]" in all_text
            or (
                "checkpoint(s)" in all_text
                and "do not exist in the test report" in all_text
            )
        )
        if checkpoint_not_found:
            add(
                "bug_doc_checkpoint_not_found",
                "只修正Checker点名的不存在FG/FC/CK路径：从最新TEST_REPORT.failed_test_case_with_check_point_list或功能覆盖定义复制完整且区分大小写的FG/FC/CK父链；不要改测试断言，不要新增近似标签。若同一TC已在正确父链下记录，删除错误或重复的旧链。",
            )
        checkpoint_not_marked = (
            "[checkpoint not marked]" in all_text
            or (
                "bug analysis document" in all_text
                and "have not called mark_function for their associated checkpoints" in all_text
            )
        )
        if checkpoint_not_marked:
            add(
                "bug_doc_checkpoint_not_marked",
                "Checker已给出TC与缺失FG/FC/CK的精确对应表。保持普通FAILED测试及node id不变：若TC确实证明该CK，就在原测试中一次性补齐对应mark_function；若无关，就只从bug文档的该CK/BG下移除错误TC引用。不要重写测试集合或复制新的BG链。",
            )
        if (
            not checkpoint_not_found
            and (
                "not found in the test report" in all_text
                or "not found in the function coverage" in all_text
            )
        ):
            add(
                "label_or_report_mismatch",
                "检查FG/FC/CK/TC是否同时存在于功能文档、coverage实现、测试mark_function和最新测试报告；删除或改正不存在/已过期的标签。",
            )
        if "defined multiple times" in all_text:
            add(
                "duplicate_label_definition",
                "同名FG/FC/CK/BG/TC只能定义一次；在原有节点下合并内容，删除重复创建的标签块。",
            )
        if "unmarked_check_points" in all_text:
            add(
                "coverage_or_test_marks_missing",
                "优先补齐测试mark_function与coverage字面量映射，使失败/未覆盖CK能对应到功能文档；不要通过重写规格文档绕过未标记问题。",
            )
        if (
            "do not have correct check point marks" in all_text
            or "not marked with 'mark_function'" in all_text
            or 'not marked with "mark_function"' in all_text
        ):
            add(
                "test_marking_contract_incomplete",
                "先读取Checker返回的完整未标记函数列表，按功能文档一次性为所有列出函数补齐字面量mark_function映射；不要每轮只修一个函数，也不要创建_fixed、_new或_backup测试副本。",
            )
        if "[api test collection regression]" in all_text:
            add(
                "api_test_collection_regression",
                "已通过API测试集合发生回退。禁止删除、改名或整文件重写test_{DUT}_api*.py；恢复Checker列出的缺失node id，只局部修复当前失败函数或共用API根因。",
            )
        if "random-test source contract violations" in all_text:
            add(
                "random_test_source_contract_violation",
                "按Checker列表一次性修复所有随机测试函数：第一个参数为env，并包含ucagent.repeat_count()和mark_function；完成整份列表后再重新Check/Complete。",
            )
        if any(marker in all_text for marker in (
            "all failed test cases must indicate bugs",
            "all failed test cases must be documented",
            "[undocumented failed cases]",
        )):
            add(
                "failed_cases_need_classification",
                "逐个分类FAILED测试：接口/API错误和测试错误先修正；只有源码分析支持的DUT bug才保留Fail并写入bug文档<TC-*>。",
            )
        if "unsupported test statuses" in all_text and (
            "xfail" in all_text or "xfailed" in all_text or "skipped" in all_text
        ):
            add(
                "unsupported_test_status",
                "Checker只接受PASSED和普通FAILED断言。删除XFAIL/SKIP标记：已确认DUT缺陷必须保持普通FAILED并补齐源码分析及FG/FC/CK/BG/TC证据链；非DUT缺陷则修正测试/API。",
            )
        elif "test execution contract" in all_text:
            add(
                "test_execution_contract",
                "先按测试执行契约修复零收集、导入/fixture、ERROR、timeout或crash；获得非零且状态仅为PASSED/FAILED的报告前，不要修改bug文档或推断DUT行为。",
            )
        if "readtextfile" in all_text and "reference" in all_text:
            add(
                "reference_files_unread",
                "先用ReadTextFile读取checker列出的参考文件，再继续Check/Complete。",
            )
        if "setskillusage" in all_text and "skill usage" in all_text:
            add(
                "stage_skill_usage_required",
                "按Checker要求调用SetSkillUsage检查并记录本阶段技能使用情况；完成后直接重新Check/Complete，不要修改测试或文档。",
            )
        if stage_name == "generate_random_test_cases" and (
            "no valid ck labels were recorded" in all_text
            or "need use args `generated: dict`" in all_text
            or "record generated={ck: note}" in all_text
        ):
            add(
                "random_test_generated_records_missing",
                "下一次Check或Complete必须在工具参数中显式传入顶层generated对象，key只能使用Checker当前批次列出的完整FG/FC/CK路径，value说明对应随机测试或跳过原因。SetCurrentStageJournal、测试函数参数和自然语言声明都不能替代generated工具参数；不要为此修改已通过的测试。",
            )
        if stage_name == "static_bug_analysis":
            if "tag hierarchy parse error" in all_text or (
                "bg-static" in all_text and "parent ck" in all_text
            ):
                add(
                    "static_bug_tag_hierarchy",
                    "保留现有文档并增量修正为FG→FC→CK→BG-STATIC→LINK-BUG→FILE层级；FG/FC/CK必须逐行使用functions_and_checks.md中的完整标签。不要删除或整份重写文档。",
                )
            if "not all 'rtl_file_to_analyze'" in all_text or "currentfiletips" in all_text:
                add(
                    "static_bug_batch_progress",
                    "在文档末尾的批次进度表中加入Checker点名的精确相对路径，格式必须为小写<file>path</file>；随后直接Check/Complete，不要运行pytest。",
                )
            if "expected '[bg-tbd]'" in all_text or "link-bug" in all_text and "static_bug_analysis" in all_text:
                add(
                    "static_bug_pending_link",
                    "本阶段每个真实BG-STATIC只能使用<LINK-BUG-[BG-TBD]>，实际动态BG关联留到static_bug_validation阶段处理。",
                )
            if "ck tags defined in the static bug doc" in all_text and "not found" in all_text:
                add(
                    "static_bug_ck_mismatch",
                    "从functions_and_checks.md复制完整FG/FC/CK父路径；不要拼接不同FG/FC下的标签，也不要通过删除文档规避错误。",
                )

        priority_errors = []
        secondary_errors = []
        for path, text in text_items:
            low = path.lower()
            if re.search(r"(?:^|\.)error(?:\[\d+\])?$", low):
                priority_errors.append(text[:1200])
            elif low == "check_info" and isinstance(text, str):
                priority_errors.append(text[:1200])
            elif "error" in low or "last_msg" in low or "action" in low:
                secondary_errors.append(text[:800])
        raw_errors = []
        for text in priority_errors + secondary_errors:
            if text not in raw_errors:
                raw_errors.append(text)
            if len(raw_errors) >= 6:
                break
        if not categories and raw_errors:
            add(
                "generic_checker_failure",
                "按raw_error_top逐条修复，修复后先运行最小验证目标再重新Check/Complete。",
            )
        if not categories:
            return None
        return OrderedDict({
            "purpose": "Prompt v1 compact blocker summary; fix these before the next Check/Complete.",
            "categories": categories,
            "primary_actions": actions,
            "raw_error_top": raw_errors[:6],
        })

    def set_data(self, key, value):
        self.data[key] = value

    def get_data(self, key, default=None):
        return self.data.get(key, default)

    def _stage_snapshot(self, index: Optional[int] = None) -> dict:
        if index is None:
            index = self.stage_index
        if isinstance(index, int) and 0 <= index < len(self.stages):
            stage = self.stages[index]
            return {
                "stage_index": index,
                "stage_name": getattr(stage, "name", "") or "",
                "stage_title": stage.title() if hasattr(stage, "title") else getattr(stage, "name", "") or "",
            }
        return {
            "stage_index": index,
            "stage_name": "",
            "stage_title": "",
        }

    def _record_structured_event(self, event_type: str, payload: Optional[dict] = None):
        agent = getattr(self, "agent", None)
        if not hasattr(agent, "record_structured_event"):
            return None
        payload = payload or {}
        decision = agent.record_structured_event(event_type, payload)
        if event_type in {"test_run", "check_result"}:
            self._publish_stage_state_package(event_type, payload, decision)
        return decision

    def _attach_progress_control(self, value, decision):
        if not isinstance(decision, dict):
            return value
        control = decision.get("control")
        if not isinstance(control, dict):
            return value
        if isinstance(value, dict):
            value["progress_control"] = control
            return value
        return "PROGRESS_CONTROL:\n" + fc.yam_str(control) + "\n\n" + str(value)

    def _summary_is_failure(self, event_type: str, summary: dict) -> bool:
        if not isinstance(summary, dict):
            return False
        if event_type == "test_run":
            if summary.get("test_outcome"):
                return summary.get("test_outcome") != TEST_OUTCOME_PASS
            if summary.get("tests_passed_all") is False:
                return True
            if summary.get("run_test_success") is False:
                return True
            try:
                return int(summary.get("tests_failed", 0) or 0) > 0
            except Exception:
                return False
        if event_type == "check_result":
            return summary.get("check_pass") is False
        return False

    def _remember_failure_context(self, event_type: str, tool: str, summary: dict, stage: Optional[dict] = None):
        agent = getattr(self, "agent", None)
        if not getattr(agent, "enable_context_reuse", False):
            return
        if not getattr(agent, "context_reuse", None):
            return
        stage = stage or self._stage_snapshot()
        if not self._summary_is_failure(event_type, summary):
            if self._recent_failure_context and self._recent_failure_context.get("stage_index") == stage.get("stage_index"):
                self._recent_failure_context = None
            return
        failure = copy.deepcopy(summary) if isinstance(summary, dict) else {}
        signature = failure_signature_from_summary(event_type, stage.get("stage_name", ""), failure)
        failure["failure_signature"] = signature
        pattern = classify_failure_pattern(
            failure,
            stage_name=str(stage.get("stage_name", "") or ""),
            event_type=event_type,
        )
        failure["failure_pattern"] = pattern
        failure["failure_group"] = failure_pattern_group(pattern)
        failure["preferred_action_categories"] = preferred_action_categories_for_failure(
            failure,
            stage_name=str(stage.get("stage_name", "") or ""),
        )
        self._recent_failure_seq += 1
        self._recent_failure_context = {
            "sequence": self._recent_failure_seq,
            "event_type": event_type,
            "tool": tool,
            "stage_index": stage.get("stage_index"),
            "stage_name": stage.get("stage_name", ""),
            "stage_title": stage.get("stage_title", ""),
            **failure,
        }
        info(
            f"[context_reuse] recent failure stage={stage.get('stage_index')} "
            f"tool={tool} sig={signature} pattern={pattern}"
        )

    def _find_first_value(self, data: Any, keys: set[str]) -> Any:
        if isinstance(data, dict):
            for key, value in data.items():
                if str(key) in keys:
                    return value
                found = self._find_first_value(value, keys)
                if found not in (None, "", [], {}):
                    return found
        elif isinstance(data, list):
            for item in data:
                found = self._find_first_value(item, keys)
                if found not in (None, "", [], {}):
                    return found
        return None

    def _collect_error_snippets(self, data: Any, limit: int = 4) -> list[str]:
        snippets = []

        def visit(value):
            if len(snippets) >= limit:
                return
            if isinstance(value, dict):
                for key, sub_value in value.items():
                    low_key = str(key).lower()
                    if any(token in low_key for token in ("error", "stderr", "last_msg", "fail")):
                        if isinstance(sub_value, str) and sub_value.strip():
                            snippets.append(sub_value.strip()[:500])
                        else:
                            visit(sub_value)
                    else:
                        visit(sub_value)
            elif isinstance(value, list):
                for item in value[:20]:
                    visit(item)
            elif isinstance(value, str):
                low = value.lower()
                if any(token in low for token in (
                    "error",
                    "failed",
                    "traceback",
                    "assert",
                    "no tests ran",
                    "no tests collected",
                    "collected 0 items",
                    "timed out",
                    "timeout",
                    "segmentation fault",
                    "fatal python",
                    "core dumped",
                    "exited with code",
                )):
                    snippets.append(value.strip()[:500])

        visit(data)
        deduped = []
        for item in snippets:
            if item and item not in deduped:
                deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped

    def _summarize_test_result(self, result: Any) -> dict:
        try:
            stage_name = str(self.stages[self.stage_index].name or "")
        except Exception:
            stage_name = ""
        allow_expected_failures = stage_allows_expected_failures(stage_name)
        summary = {
            "result_type": type(result).__name__,
            "tests_total": None,
            "tests_failed": None,
            "tests_passed_all": None,
            "failed_cases_top": [],
            "failed_checkpoints_top": [],
            "run_test_success": None,
            "test_outcome": None,
            "test_outcome_reason": None,
            "error_top": [],
        }
        if isinstance(result, dict):
            report = self._find_first_value(result, {"REPORT", "TEST_REPORT"})
            if not isinstance(report, dict):
                report = result
            tests = report.get("tests", {}) if isinstance(report, dict) else {}
            test_cases = tests.get("test_cases", {}) if isinstance(tests, dict) else {}
            if isinstance(tests, dict):
                summary["tests_total"] = tests.get("total")
                summary["tests_failed"] = tests.get("fails")
            if isinstance(report, dict):
                summary["run_test_success"] = report.get("run_test_success")
                failed_checkpoints = report.get("failed_check_point_list") or []
                if isinstance(failed_checkpoints, list):
                    summary["failed_checkpoints_top"] = [str(item) for item in failed_checkpoints[:10]]
            if isinstance(test_cases, dict):
                status_counts = Counter(str(status).strip().upper() for status in test_cases.values())
                summary["test_status_counts"] = dict(status_counts)
                accepted_statuses = {"PASSED", "PASS"}
                if allow_expected_failures:
                    accepted_statuses.update({"XFAIL", "XFAILED"})
                failed_cases = [
                    str(name) for name, status in test_cases.items()
                    if str(status).strip().upper() not in accepted_statuses
                ]
                summary["failed_cases_top"] = failed_cases[:10]
                if summary["tests_total"] is None:
                    summary["tests_total"] = len(test_cases)
                if summary["tests_failed"] is None:
                    summary["tests_failed"] = len(failed_cases)
            summary["error_top"] = self._collect_error_snippets(result)
        elif isinstance(result, str):
            summary["message"] = result[:500]
            summary["error_top"] = self._collect_error_snippets(result)
        else:
            summary["message"] = str(result)[:500]
        return apply_test_outcome(
            summary,
            allow_expected_failures=allow_expected_failures,
        )

    def _summarize_check_result(self, result: Any) -> dict:
        summary = {
            "result_type": type(result).__name__,
            "check_pass": None,
            "checker_categories": [],
            "failed_cases_top": [],
            "failed_checkpoints_top": [],
            "error_top": [],
        }
        if isinstance(result, dict):
            source = result.get("last_check_result") if isinstance(result.get("last_check_result"), dict) else result
            summary["check_pass"] = source.get("check_pass", result.get("complete"))
            checker_summary = source.get("checker_error_summary")
            if isinstance(checker_summary, dict):
                categories = checker_summary.get("categories") or []
                if isinstance(categories, list):
                    summary["checker_categories"] = [
                        str(item.get("category", item)) if isinstance(item, dict) else str(item)
                        for item in categories[:10]
                    ]
                raw_error = checker_summary.get("raw_error_top") or []
                if isinstance(raw_error, list):
                    summary["error_top"].extend(str(item)[:500] for item in raw_error[:4])
            check_info = source.get("check_info")
            test_report = self._find_first_value(check_info, {"REPORT", "TEST_REPORT"})
            if isinstance(test_report, dict):
                test_summary = self._summarize_test_result({"REPORT": test_report})
                for key in (
                    "tests_total",
                    "tests_failed",
                    "tests_passed_all",
                    "test_status_counts",
                    "test_outcome",
                    "test_outcome_reason",
                    "run_test_success",
                    "failed_cases_top",
                    "failed_checkpoints_top",
                ):
                    if key in test_summary:
                        summary[key] = test_summary.get(key)
            failed_cases = self._find_first_value(check_info, {"failed_cases", "failed_case_list", "failed_test_cases"})
            if isinstance(failed_cases, list) and not summary["failed_cases_top"]:
                summary["failed_cases_top"] = [str(item) for item in failed_cases[:10]]
            failed_checkpoints = self._find_first_value(check_info, {"failed_check_point_list", "failed_checkpoints"})
            if isinstance(failed_checkpoints, list) and not summary["failed_checkpoints_top"]:
                summary["failed_checkpoints_top"] = [str(item) for item in failed_checkpoints[:10]]
            for item in self._collect_error_snippets(result):
                if item not in summary["error_top"]:
                    summary["error_top"].append(item)
                if len(summary["error_top"]) >= 4:
                    break
        elif isinstance(result, str):
            summary["message"] = result[:500]
            summary["error_top"] = self._collect_error_snippets(result)
        else:
            summary["message"] = str(result)[:500]
        summary["error_top"] = summary["error_top"][:4]
        return summary

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
            ToolStageJournal().set_function(self.tool_get_current_journal),
            ToolAllStageJournal().set_function(self.tool_get_all_journal),
            ToolSetCurrentStageJournal().set_function(self.tool_set_journal),
        ]
        if self.agent.cfg.skill.use_skill:
            tools.append(ToolSetSkillUsage().set_function(self.tool_set_skill_usage))
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
            tips["notes"] = f"You need use tool: {self.tool_read_text.name} to read the reference files.\n"

        # list the skills needed to use in current stage
        skills_to_use = [skill_name for skill_name in cstage.skill_list]
        if skills_to_use:
            formatted_skill_list = list_skills_in_format(_list_skills(self.workspace), self.workspace, skills_to_use)
            tips["notes"] = tips.get("notes", "") + f"Firstly you must read the SKILL.md of the following skills to know how to complete current stage:\n{formatted_skill_list}\n"

        tips["process"] = f"{self.stage_index}/{len(self.stages)}"
        mission_tips = self.mission.get_value("prompt.tips")
        if mission_tips is not None:
            mission_tips = mission_tips.as_dict()
            current_tips = list(mission_tips.get("allways", []))
            random_tips = mission_tips.get("random", [])
            if random_tips:
                current_tips.append(random.choice(random_tips))
            if current_tips:
                random.shuffle(current_tips)
                tips["tips"] = current_tips

        prompt_guidance = self._prompt_guidance_for_stage(cstage)
        if prompt_guidance is not None:
            tips["prompt_variant_guidance"] = prompt_guidance
        self._update_stage_context(cstage)
        tips = make_llm_tool_ret(tips)
        if getattr(self.agent, "enable_long_term_memory", False) and getattr(self.agent, "long_term_memory", None):
            ltm = self.agent.long_term_memory
            memories, memory_meta = self._get_prompt_memories(ltm, cstage)
            if memories:
                if getattr(self.agent, "enable_failure_aware_context", False):
                    failure_mem, non_failure_mem = self._split_failure_context(memories)
                    parts = []
                    if failure_mem:
                        parts.append("FAILURE_CONTEXT:\n" + fc.yam_str([self._render_memory_prompt_item(x) for x in failure_mem]))
                    if non_failure_mem:
                        parts.append("SUCCESS_CONTEXT:\n" + fc.yam_str([self._render_memory_prompt_item(x) for x in non_failure_mem]))
                    memory_text = "LONG_TERM_MEMORY:\n" + "\n".join(parts)
                else:
                    memory_text = "LONG_TERM_MEMORY:\n" + fc.yam_str([self._render_memory_prompt_item(x) for x in memories])
                tips = memory_text + "\n\n" + tips
                msg_node = getattr(self.agent, "message_manage_node", None)
                if msg_node is not None and hasattr(msg_node, "note_memory_injection"):
                    try:
                        msg_node.note_memory_injection(
                            memories,
                            query=memory_meta.get("query", ""),
                            source=memory_meta.get("hit_type", "retrieval"),
                            stage_index=self.stage_index,
                        )
                    except Exception:
                        pass
        context_reuse_text = self._render_context_reuse_prompt(cstage, mark_injected=True)
        if context_reuse_text:
            tips = context_reuse_text + "\n\n" + tips
        return self.attach_todo_summary(tips)

    def _render_context_reuse_prompt(self, cstage, mark_injected: bool = True):
        store = getattr(self.agent, "context_reuse", None)
        if not getattr(self.agent, "enable_context_reuse", False) or store is None:
            return ""
        failure = self._recent_failure_context
        if not isinstance(failure, dict):
            return ""
        if failure.get("stage_index") != self.stage_index:
            return ""
        inject_once = bool(self.context_reuse_cfg.get("inject_once_per_failure", True))
        inject_scope = str(self.context_reuse_cfg.get("inject_once_scope", "signature") or "signature")
        failure_pattern = failure.get("failure_pattern") or classify_failure_pattern(
            failure,
            stage_name=str(failure.get("stage_name", "") or ""),
            event_type=str(failure.get("event_type", "") or ""),
        )
        failure_group = failure.get("failure_group") or failure_pattern_group(failure_pattern)
        signature = failure.get("failure_signature")
        if inject_scope == "sequence":
            inject_key = (
                "sequence",
                failure.get("sequence"),
                failure.get("stage_index"),
                signature,
            )
        else:
            inject_key = (
                "signature",
                failure.get("stage_index"),
                signature or failure.get("sequence"),
                failure_pattern,
            )
        pattern_key = (
            "pattern",
            failure.get("stage_index"),
            failure_pattern or failure_group,
        )
        same_signature_limit = int(self.context_reuse_cfg.get("same_signature_inject_limit", 1) or 1)
        same_pattern_limit = int(self.context_reuse_cfg.get("max_injections_per_stage_pattern", 2) or 2)
        if mark_injected and inject_once and self._context_reuse_injected_key_counts[inject_key] >= same_signature_limit:
            self._memory_metrics["context_reuse_throttled"] += 1
            self._record_context_reuse_decision(
                failure,
                cstage,
                decision="throttled",
                reason="same_signature",
            )
            info(
                f"[context_reuse] throttle stage={self.stage_index} sig={signature} "
                f"pattern={failure_pattern} reason=same_signature"
            )
            return ""
        if mark_injected and inject_once and same_pattern_limit > 0 and self._context_reuse_injected_key_counts[pattern_key] >= same_pattern_limit:
            self._memory_metrics["context_reuse_throttled"] += 1
            self._record_context_reuse_decision(
                failure,
                cstage,
                decision="throttled",
                reason="same_pattern",
            )
            info(
                f"[context_reuse] throttle stage={self.stage_index} sig={signature} "
                f"pattern={failure_pattern} reason=same_pattern"
            )
            return ""

        limit = int(self.context_reuse_cfg.get("prompt_limit", 2) or 2)
        hint_limit = int(self.context_reuse_cfg.get("compression_hint_limit", 1) or 1)
        stage_name = getattr(cstage, "name", "") or failure.get("stage_name", "")
        episodes = store.search(failure, self.stage_index, stage_name, limit=limit)
        search_summary = store.get_last_search_summary()
        hints = store.compression_hints_for(failure, self.stage_index, stage_name, limit=hint_limit)
        allow_hints_only_value = self.context_reuse_cfg.get("allow_hints_only", False)
        allow_hints_only = (
            allow_hints_only_value
            if isinstance(allow_hints_only_value, bool)
            else str(allow_hints_only_value).strip().lower() in {"1", "true", "yes", "on"}
        )
        hints_only_suppressed = bool(hints and not episodes and not allow_hints_only)
        if hints_only_suppressed:
            hints = []
        self._memory_metrics["context_reuse_queries"] += 1
        if episodes:
            self._memory_metrics["context_reuse_hits"] += 1
        if hints:
            self._memory_metrics["context_reuse_hint_hits"] += 1
        if not episodes and not hints:
            if mark_injected:
                miss_reason = (
                    "hints_only_suppressed"
                    if hints_only_suppressed
                    else "utility_gate_rejected_all"
                    if int(search_summary.get("utility_rejected", 0) or 0) > 0
                    else "no_candidate_above_gate"
                )
                self._record_context_reuse_decision(
                    failure,
                    cstage,
                    decision="miss",
                    reason=miss_reason,
                    search_summary=search_summary,
                )
            return ""

        if mark_injected:
            self._context_reuse_injected_keys.add(inject_key)
            self._context_reuse_injected_key_counts[inject_key] += 1
            self._context_reuse_injected_key_counts[pattern_key] += 1
        self._memory_metrics["context_reuse_injections"] += 1
        self._memory_metrics["context_reuse_items"] += len(episodes) + len(hints)
        if mark_injected:
            self._record_context_reuse_decision(
                failure,
                cstage,
                decision="inject",
                reason=(
                    "hints_only_after_utility_gate"
                    if not episodes and int(search_summary.get("utility_rejected", 0) or 0) > 0
                    else ""
                ),
                episodes=episodes,
                hints=hints,
                search_summary=search_summary,
            )
        current_failure = OrderedDict({
            "stage": failure.get("stage_name"),
            "tool": failure.get("tool"),
            "signature": failure.get("failure_signature"),
            "failure_pattern": failure_pattern,
            "failure_group": failure_group,
            "preferred_action_categories": preferred_action_categories_for_failure(
                failure,
                stage_name=stage_name,
            ),
            "failed_cases_top": (failure.get("failed_cases_top") or [])[:3],
            "failed_checkpoints_top": (failure.get("failed_checkpoints_top") or [])[:3],
            "error_top": (failure.get("error_top") or [])[:2],
        })
        matching_policy = (
            "Prefer episodes with the same failure_pattern and action_categories matching "
            "current preferred_action_categories. Treat mismatched action categories as weak or negative evidence."
        )
        if search_summary.get("utility_gate_enabled"):
            matching_policy += (
                " Injected repair episodes passed the configured historical verifier-utility gate, "
                "but this is not proof that they will work on the current DUT."
            )
        payload = OrderedDict({
            "purpose": (
                "Use these compact historical repair episodes as analogies for the current failure. "
                "Do not copy DUT-specific values blindly; first verify against current files."
            ),
            "matching_policy": matching_policy,
            "current_failure": current_failure,
            "repair_episodes": [store.render_episode(item) for item in episodes],
        })
        if hints:
            payload["compression_hints"] = [store.render_hint(item) for item in hints]
            payload["context_policy"] = (
                "If the same failure repeats, keep the latest failure summary and the last effective actions; "
                "avoid re-reading or restating older raw outputs with the same signature."
            )
        info(
            f"[context_reuse] inject stage={self.stage_index} "
            f"episodes={len(episodes)} hints={len(hints)} sig={failure.get('failure_signature')} "
            f"pattern={failure_pattern}"
        )
        return "REPAIR_EPISODE_CONTEXT:\n" + fc.yam_str(payload)

    def _record_context_reuse_decision(
        self,
        failure,
        cstage,
        *,
        decision: str,
        reason: str = "",
        episodes=None,
        hints=None,
        search_summary=None,
    ):
        """Persist an auditable retrieval decision for verifier-linked evaluation."""
        if not self.enable_data_collection:
            return
        episodes = episodes if isinstance(episodes, list) else []
        hints = hints if isinstance(hints, list) else []
        search_summary = search_summary if isinstance(search_summary, dict) else {}
        failure = failure if isinstance(failure, dict) else {}
        stage_name = getattr(cstage, "name", "") or failure.get("stage_name", "")
        strategies = []
        for item in episodes:
            if not isinstance(item, dict):
                continue
            retrieval = item.get("__context_reuse__") if isinstance(item.get("__context_reuse__"), dict) else {}
            utility = retrieval.get("utility") if isinstance(retrieval.get("utility"), dict) else {}
            contract_policy = (
                retrieval.get("contract_policy")
                if isinstance(retrieval.get("contract_policy"), dict)
                else {}
            )
            strategies.append({
                "strategy_id": item.get("strategy_id"),
                "episode_id": item.get("episode_id"),
                "source_dut": item.get("dut"),
                "source_duts": (item.get("source_duts") or [])[:8],
                "source_stage": item.get("stage_name"),
                "failure_pattern": item.get("failure_pattern"),
                "action_categories": (item.get("action_categories") or [])[:5],
                "score": retrieval.get("score"),
                "match_reasons": (retrieval.get("reasons") or [])[:6],
                "quality_score": item.get("quality_score"),
                "support_count": item.get("support_count"),
                "utility_gate": {
                    "enabled": utility.get("enabled"),
                    "passed": utility.get("passed"),
                    "reason": utility.get("reason"),
                    "historical_score": utility.get("historical_score"),
                    "prompt_tokens_estimated": utility.get("prompt_tokens_estimated"),
                    "prompt_cost": utility.get("prompt_cost"),
                    "net_score": utility.get("net_score"),
                    "threshold": utility.get("threshold"),
                    "policy_version": utility.get("policy_version"),
                },
                "transition_contract": item.get("transition_contract", {}),
                "contract_policy": {
                    "mode": contract_policy.get("mode"),
                    "applicable": contract_policy.get("applicable"),
                    "score": contract_policy.get("score"),
                    "threshold": contract_policy.get("threshold"),
                    "reasons": (contract_policy.get("reasons") or [])[:6],
                    "hard_mismatches": (contract_policy.get("hard_mismatches") or [])[:6],
                    "contract_version": contract_policy.get("contract_version"),
                },
            })
        self._record_structured_event("context_reuse_decision", {
            "decision": decision,
            "reason": reason,
            "failure": {
                "event_type": failure.get("event_type"),
                "tool": failure.get("tool"),
                "signature": failure.get("failure_signature"),
                "pattern": failure.get("failure_pattern"),
                "group": failure.get("failure_group"),
                "failed_cases_top": (failure.get("failed_cases_top") or [])[:5],
                "failed_checkpoints_top": (failure.get("failed_checkpoints_top") or [])[:5],
                "error_top": (failure.get("error_top") or [])[:3],
            },
            "stage_name": stage_name,
            "strategies": strategies,
            "search_summary": search_summary,
            "hint_count": len(hints),
            "prompt_item_count": len(episodes) + len(hints),
        })

    def _get_prompt_memories(self, ltm, cstage):
        query, filters = self._build_memory_query(self.stage_index, cstage.title())
        limit = int(self.long_term_memory_cfg.get("prompt_limit", 3) or 3)
        recent_limit = int(self.long_term_memory_cfg.get("recent_fallback_limit", 2) or 2)
        use_cache = bool(self.long_term_memory_cfg.get("cache_prompt_per_stage", True))
        cache_key = (self.stage_index, query, filters.get("dut"), ltm.version())
        is_first_stage_query = (self._stage_prompt_seen == 0)
        self._memory_metrics["prompt_queries"] += 1
        if use_cache and cache_key in self._ltm_prompt_cache:
            cached = copy.deepcopy(self._ltm_prompt_cache[cache_key])
            self._memory_metrics["stage_cache_hits"] += 1
            self._memory_metrics["prompt_injections"] += 1
            self._memory_metrics["injected_items"] += len(cached)
            self._memory_metrics["stale_hits"] += sum(
                1 for item in cached if bool(item.get("resolved", False)) or float(item.get("stale_score", 0.0) or 0.0) >= 0.5
            )
            ltm.mark_prompt_usage(query=query, entries=cached, hit_type="stage_cache", from_prefetch=False)
            self._stage_prompt_seen += 1
            return cached, {"query": query, "hit_type": "stage_cache"}
        prefetch_payload = self._ltm_prefetch_cache.get(self.stage_index)
        if prefetch_payload and prefetch_payload.get("cache_key") == cache_key:
            prefetched = self._select_prefetch_injection(copy.deepcopy(prefetch_payload.get("memories", []) or []))
            if prefetched:
                self._memory_metrics["prefetch_hits"] += 1
                self._memory_metrics["retrieval_hits"] += 1
                self._memory_metrics["prompt_injections"] += 1
                self._memory_metrics["injected_items"] += len(prefetched)
                self._memory_metrics["stale_hits"] += sum(
                    1 for item in prefetched if bool(item.get("resolved", False)) or float(item.get("stale_score", 0.0) or 0.0) >= 0.5
                )
                if is_first_stage_query:
                    self._memory_metrics["stage_first_turn_prefetch_hits"] += 1
                ltm.mark_prompt_usage(query=query, entries=prefetched, hit_type="prefetch", from_prefetch=True)
                if use_cache:
                    self._ltm_prompt_cache = {cache_key: copy.deepcopy(prefetched)}
                self._stage_prompt_seen += 1
                return prefetched, {"query": query, "hit_type": "prefetch", "from_prefetch": True}
        memories = ltm.search(query=query, limit=limit, filters=filters)
        hit_type = "retrieval"
        if not memories:
            memories = ltm.recent(limit=recent_limit, filters={"dut": self.agent.dut_name})
            hit_type = "recent_fallback" if memories else "miss"
        if memories:
            if hit_type == "retrieval":
                self._memory_metrics["retrieval_hits"] += 1
            elif hit_type == "recent_fallback":
                self._memory_metrics["recent_fallback_hits"] += 1
            self._memory_metrics["prompt_injections"] += 1
            self._memory_metrics["injected_items"] += len(memories)
            self._memory_metrics["stale_hits"] += sum(
                1 for item in memories if bool(item.get("resolved", False)) or float(item.get("stale_score", 0.0) or 0.0) >= 0.5
            )
            ltm.mark_prompt_usage(query=query, entries=memories, hit_type=hit_type, from_prefetch=False)
        else:
            self._memory_metrics["retrieval_misses"] += 1
        if use_cache:
            self._ltm_prompt_cache = {cache_key: copy.deepcopy(memories)}
        self._stage_prompt_seen += 1
        return memories, {"query": query, "hit_type": hit_type}

    def get_memory_metrics(self):
        metrics = dict(self._memory_metrics)
        queries = max(1, metrics["prompt_queries"])
        metrics["stage_cache_hit_rate"] = round(metrics["stage_cache_hits"] / queries, 4)
        metrics["prefetch_hit_rate"] = round(metrics["prefetch_hits"] / queries, 4)
        metrics["prefetch_fill_rate"] = round(metrics["prefetch_fills"] / max(1, metrics["prefetch_triggers"]), 4)
        metrics["stage_first_turn_hit_rate"] = round(
            metrics["stage_first_turn_prefetch_hits"] / max(1, metrics["stage_entries"]), 4
        )
        metrics["retrieval_hit_rate"] = round(metrics["retrieval_hits"] / queries, 4)
        metrics["recent_fallback_rate"] = round(metrics["recent_fallback_hits"] / queries, 4)
        metrics["stale_hit_rate"] = round(metrics["stale_hits"] / max(1, metrics["injected_items"]), 4)
        context_queries = max(1, metrics["context_reuse_queries"])
        metrics["context_reuse_hit_rate"] = round(metrics["context_reuse_hits"] / context_queries, 4)
        metrics["context_reuse_hint_rate"] = round(metrics["context_reuse_hint_hits"] / context_queries, 4)
        return metrics

    def _split_failure_context(self, memories):
        failure_mem = []
        non_failure_mem = []
        max_failure = int(self._cfg_value(self.cfg, "context_upgrade.failure_aware_context.max_failure_items", 2) or 2)
        max_success = int(self._cfg_value(self.cfg, "context_upgrade.failure_aware_context.max_success_items", 1) or 1)
        for m in memories:
            if self._is_failure_memory(m):
                if len(failure_mem) < max_failure:
                    failure_mem.append(m)
            else:
                if len(non_failure_mem) < max_success:
                    non_failure_mem.append(m)
        return failure_mem, non_failure_mem

    def _is_failure_memory(self, entry) -> bool:
        if not isinstance(entry, dict):
            return False
        content = entry.get("content")
        if isinstance(content, dict):
            status = str(content.get("status", "")).lower()
            if status == "failure":
                return True
            if status == "success":
                return False
        return self._is_failure_content(content)

    def _render_memory_prompt_item(self, entry):
        if not isinstance(entry, dict):
            return entry
        meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
        content = entry.get("content") if isinstance(entry.get("content"), dict) else {}
        stage_info = content.get("stage_info") if isinstance(content.get("stage_info"), dict) else {}
        return {
            "meta": {
                "type": meta.get("type"),
                "stage_index": meta.get("stage_index"),
                "stage_title": meta.get("stage_title") or stage_info.get("stage_title", ""),
                "level": entry.get("level", "candidate"),
                "support": entry.get("support", 1),
            },
            "content": {
                "status": content.get("status", ""),
                "summary": content.get("summary", ""),
                "failed_checkpoints_top": (content.get("failed_checkpoints_top") or [])[:6],
                "failed_cases_top": (content.get("failed_cases_top") or [])[:4],
                "root_cause": content.get("root_cause", ""),
                "action": content.get("action", ""),
            },
        }

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
        ret["all_completed"] = self._compute_all_completed()
        ret["stage_list"] = []
        for i, stage in enumerate(self.stages):
            ret["stage_list"].append({
                "index": i,
                "title": stage.title(),
                "reached": stage.is_reached(),
                "fail_count": stage.fail_count,
                "skill_list": list(stage.skill_list.keys()) if self.cfg.skill.use_skill else [],
                "is_skipped": stage.is_skipped(),
                "time_start": stage.get_time_start_str(),
                "time_end": stage.get_time_end_str(),
                "time_cost": stage.get_time_cost_str(),
                "is_completed": stage.is_completed(),
                "needs_human_check": stage.is_hmcheck_needed(),
                "need_fail_llm_suggestion": self.stage_need_llm_fail_suggestion(stage),
                "need_pass_llm_suggestion": self.stage_need_llm_pass_suggestion(stage),
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

    def set_current_stage_journal(self, journal):
        stage = self.get_current_stage()
        if stage:
            stage.meta_set_journal(journal)
            return "Set journal success."
        return "No current stage available."

    def get_current_stage_journal(self):
        stage = self.get_current_stage()
        if stage:
            return stage.meta_get_journal()
        return "No current stage available."

    def get_all_stage_journal(self):
        journals = OrderedDict()
        for stage in self.stages:
            journals[stage.title()] = stage.meta_get_journal()
        return journals

    def set_current_stage_skill_usage(self, skill_usage: Dict[str, Any]):
        """set the skill usage of curretn stage or return feedback based on skill_usage"""
        current_stage = self.get_current_stage()
        if current_stage.skill_list:
            for skill_name in current_stage.skill_list:
                skill_root = fc.get_workspace_skill_root(self.workspace)
                skill_root_abs = os.path.abspath(skill_root)
                skill_dir = os.path.abspath(os.path.join(skill_root_abs, skill_name))
                if os.path.commonpath([skill_root_abs, skill_dir]) != skill_root_abs or not os.path.isdir(skill_dir):
                    raise ValueError(f"Skill '{skill_name}' is not found in workspace. ")
                if skill_name not in skill_usage:
                    return f"You must use skill '{skill_name}' in current stage, using tool `ListSkill` to list and use it."
                else:
                    skill_info = skill_usage[skill_name]
                    current_stage.set_usage_skill_list(skill_name, listed=skill_info.get("list", False), read=skill_info.get("read", False), used=skill_info.get("use", False))
                    [u,v,w] = current_stage.skill_list[skill_name]
                    if u and v and w:
                        continue
                    if not u:
                        return f"You must re-complete the stage by using tool `ListSkill` to list and use the skill {skill_name}."
                    if not v:
                        return f"You must re-complete the stage by using tool `ReadTextFile` to read the SKILL.md of skill {skill_name} and use it"
                    if not w:
                        return f"You must re-complete the stage by using the skill {skill_name} according to the method steps mentioned in its SKILL.md."
            current_stage.meta_set_skill_usage(skill_usage)
            return "All skills in skill_list have been used."
        return "No skill need be used in current stage."

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
                self._on_stage_enter(self.stages[self.stage_index], trigger_prefetch=True)
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
            self._on_stage_enter(self.stages[self.stage_index], trigger_prefetch=True)
            return True
        return False

    def check(self, timeout, **check_args):
        if not self.stage_index < len(self.stages):
            ret_data = OrderedDict({
                "check_pass": False,
                "check_info": f"Stage index{self.stage_index} out of range. (Mission maybe completed, you can use the `GoToStage` tool to go back to a previous stage if needed)",
            })
            summary = self._summarize_check_result(ret_data)
            self._remember_failure_context("check_result", "Check", summary)
            decision = self._record_structured_event("check_result", {
                "tool": "Check",
                "success": False,
                "timeout": timeout,
                "is_complete": False,
                "result": summary,
            })
            return self._attach_progress_control(ret_data, decision)
        ck_pass, ck_info = self.stages[self.stage_index].do_check(
            **{**check_args, "timeout": timeout}
        )
        ret_data = OrderedDict({
            "check_info": ck_info,
            "check_pass": ck_pass,
        })
        if not ck_pass:
            ret_data["action"] = "Please fix the issues reported in 'check_info.last_msg.error' according to the suggestions, and then use the `Check` tool again to re-validate your work."
            checker_summary = self._checker_error_summary(ret_data)
            if checker_summary is not None:
                ret_data["checker_error_summary"] = checker_summary
        self.last_check_info = copy.deepcopy(ret_data)
        if ck_pass:
            ret_data["message"] = f"Congratulations! Stage {self.stage_index} checks passed successfully, you can use tool 'Complete' to finish this stage."
        summary = self._summarize_check_result(ret_data)
        self._remember_failure_context("check_result", "Check", summary)
        if self._summary_is_failure("check_result", summary):
            repair_text = self._render_context_reuse_prompt(self.stages[self.stage_index], mark_injected=True)
            if repair_text:
                ret_data["repair_episode_context"] = repair_text
        decision = self._record_structured_event("check_result", {
            "tool": "Check",
            "success": bool(ck_pass),
            "timeout": timeout,
            "is_complete": False,
            "result": summary,
        })
        self._attach_progress_control(ret_data, decision)
        if not ck_pass:
            suggestion = self.gen_fail_suggestion(ret_data)
            if suggestion is not ret_data:
                ret_data["llm_failure_suggestion"] = suggestion
        return ret_data

    def save_stage_info(self):
        all_completed = self._refresh_all_completed()
        info = self.agent.get_stat_info()
        info.update({
            "mission_name": self.agent.cfg.mission.name,
            "dut_name": getattr(self.agent, "dut_name", ""),
            "DUT": getattr(self.agent, "dut_name", ""),
            "stage_index": self.stage_index,
            "all_completed": all_completed,
            "time_begin": self.time_begin,
            "time_end": self.time_end,
            "is_agent_exit": self.agent.is_exit(),
            "stage_data": self.data,
        })
        info["stages_info"] = {}
        for idx, stage in enumerate(self.stages):
            stage_info = stage.detail()
            stage_info["time_cost"] = stage.get_time_cost()
            stage_info["meta_data"] = stage.meta_data
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
        self._refresh_all_completed()
        self.save_stage_info()
        return self.get_current_stage()

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

    def _stage_complete(self, stage):
        stage.on_complete()
        if self.llm_fail_suggestion:
            self.llm_fail_suggestion.on_stage_complete(stage)
        if self.llm_pass_suggestion:
            self.llm_pass_suggestion.on_stage_complete(stage)

    def complete(self, timeout, **check_args):
        before_index = self.stage_index
        before_stage = self._stage_snapshot(before_index)
        if self.stage_index >= len(self.stages):
            ret = {
                "complete": False,
                "message": ("No more stages to complete. You can review your work and use the `GoToStage` tool to go back to a previous stage if needed. "
                            "Or you can use the `Exit` tool to exit the mission."),
                "last_check_result": self.last_check_info,
            }
            summary = self._summarize_check_result(ret)
            self._remember_failure_context("check_result", "Complete", summary, stage=before_stage)
            decision = self._record_structured_event("check_result", {
                **before_stage,
                "tool": "Complete",
                "success": False,
                "timeout": timeout,
                "is_complete": True,
                "stage_before": before_stage,
                "result": summary,
            })
            self._attach_progress_control(ret, decision)
            self._record_structured_event("stage_transition", {
                **before_stage,
                "trigger": "Complete",
                "success": False,
                "advanced": False,
                "from_stage": before_stage,
                "to_stage": before_stage,
                "all_completed": bool(self.all_completed),
            })
            return ret
        ck_pass, ck_info = self.stages[self.stage_index].do_check(
            **{**check_args, "timeout": timeout, "is_complete": True}
        )
        stage = self.stages[self.stage_index]
        if ck_pass and stage.meta_get_journal() is None:
            ck_pass = False
            ck_info = {
                "error": "Please use tool 'SetCurrentStageJournal' to set the journal of this stage before completing it."
            }
        if ck_pass:
            llm_msg = self.gen_pass_suggestion(ck_info)
            ck_pass = stage.get_approved()
            if not ck_pass:
                ck_info = {"error": "Stage Complete Fail:\n\n" + str(llm_msg)}
        if ck_pass and stage.is_hmcheck_needed():
            hm_passed, ck_msg = stage.get_hmcheck_state()
            if hm_passed is None:
                self.agent._need_human = True
                ck_pass = False
                ck_info = {"error": (
                    "Now you have passed the self check of this stage, but human check is needed before completing it. "
                    "Please give a brief introduction of your work to help the human reviewer understand your implementation. "
                    "Then wait for human review and approval."
                )}
            elif hm_passed is False:
                self.agent._need_human = True
                ck_pass = False
                ck_info = {
                    "error": (
                        "Human check did not approve your work for this stage. "
                        "Please address the issues raised by the human reviewer and then use the `Complete` tool again to complete this stage."
                    ),
                    "human_review_msg": ck_msg,
                }
            else:
                assert hm_passed is True, "hm_passed should be True here"
                info("Human check approved for stage " + stage.name)

        self.last_check_info = OrderedDict({
            "check_info": ck_info,
            "check_pass": ck_pass,
        })
        if not ck_pass:
            checker_summary = self._checker_error_summary(self.last_check_info)
            if checker_summary is not None:
                self.last_check_info["checker_error_summary"] = checker_summary
        if ck_pass:
            message = f"Stage {self.stage_index} completed successfully. "
            self._stage_complete(self.stages[self.stage_index])
            self.next_stage()
            if self.all_completed:
                message = ("All stages completed successfully. "
                           "Now you should review your work to check if everything is correct and all the users needs are matched. "
                           "When you are confident that everything is fine, you can use the `Exit` tool to exit the mission. "
                           )
            else:
                message += f"Current stage index is now {self.stage_index}. Use `CurrentTips` tool to get your new task. "
                self.stages[self.stage_index].set_reached(True)
                self.stages[self.stage_index].on_init()
                self._on_stage_enter(self.stages[self.stage_index], trigger_prefetch=True)
        else:
            message = f"Stage {self.stage_index} not completed. Please check the task requirements."
        after_stage = self._stage_snapshot(self.stage_index)
        ret = OrderedDict({
            "complete": ck_pass,
            "message": message,
            "last_check_result": self.last_check_info,
        })
        if not ck_pass:
            ret["action"] = "Please fix the issues reported in 'last_check_result.check_info.last_msg.error' according to the suggestions, and then use the `Complete` tool again to complete this stage."
        summary = self._summarize_check_result(ret)
        self._remember_failure_context("check_result", "Complete", summary, stage=before_stage)
        if self._summary_is_failure("check_result", summary) and 0 <= before_index < len(self.stages):
            repair_text = self._render_context_reuse_prompt(self.stages[before_index], mark_injected=True)
            if repair_text:
                ret["repair_episode_context"] = repair_text
        decision = self._record_structured_event("check_result", {
            **before_stage,
            "tool": "Complete",
            "success": bool(ck_pass),
            "timeout": timeout,
            "is_complete": True,
            "stage_before": before_stage,
            "stage_after": self._stage_snapshot(self.stage_index),
            "result": summary,
        })
        self._attach_progress_control(ret, decision)
        self._record_structured_event("stage_transition", {
            **before_stage,
            "trigger": "Complete",
            "success": bool(ck_pass),
            "advanced": self.stage_index != before_index,
            "from_stage": before_stage,
            "to_stage": after_stage,
            "all_completed": bool(self.all_completed),
        })
        if not ck_pass:
            suggestion = self.gen_fail_suggestion(self.last_check_info)
            if suggestion is not self.last_check_info:
                ret["llm_failure_suggestion"] = suggestion
        return ret

    def exit(self):
        """
        Exit the agent and end the mission after all stages are completed.
        """
        if self._refresh_all_completed():
            self.time_end = time.time()
            self.agent.exit()  # Exit the agent if all stages are completed
            self.save_stage_info()
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

    def tool_set_journal(self, journal):
        """
        Set the journal of the current stage.
        This is used to when current stage is completed or the LLM context is compressed and other similar situations.
        The journal content should be concise and clear and only the necessary information should be included.
        """
        ret = make_llm_tool_ret(self.set_current_stage_journal(journal))
        info("ToolSetCurrentStageJournal:\n" + ret)
        return self.attach_todo_summary(ret)

    def tool_get_all_journal(self):
        ret = make_llm_tool_ret(self.get_all_stage_journal())
        info("ToolGetAllStageJournal:\n" + ret)
        return self.attach_todo_summary(ret)

    def tool_get_current_journal(self):
        ret = make_llm_tool_ret(self.get_current_stage_journal())
        info("ToolGetCurrentStageJournal:\n" + ret)
        return self.attach_todo_summary(ret)

    def tool_set_skill_usage(self, skill_usage: Dict[str, Any]):
        ret = make_llm_tool_ret(self.set_current_stage_skill_usage(skill_usage))
        info("ToolSetSkillUsage:\n" + ret)
        return self.attach_todo_summary(ret)

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

    def tool_check(self, timeout, **check_args):
        ret = make_llm_tool_ret(self.check(timeout, **check_args))
        info("ToolCheck:\n" + ret)
        return self.attach_todo_summary(ret)

    def tool_exit(self):
        ret = make_llm_tool_ret(self.exit())
        info("ToolExit:\n" + ret)
        return ret

    def tool_complete(self, timeout, **check_args):
        ret = make_llm_tool_ret(self.complete(timeout, **check_args))
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
        stage_name = ""
        stage = None
        if self.stage_index < len(self.stages):
            stage = self.stages[self.stage_index]
            stage_name = str(getattr(stage, "name", "") or "")
        if stage_name == "test_case_implementation_in_batch" and not str(pytest_args or "").strip():
            batch_target = ""
            for checker in getattr(stage, "checker", []) or []:
                get_run_args = getattr(checker, "get_run_args", None)
                current_cases = getattr(checker, "current_test_cases", None)
                if not callable(get_run_args) or not current_cases:
                    continue
                try:
                    batch_target = str(get_run_args(self.free_pytest_run.test_dir)[0] or "").strip()
                except (AttributeError, IndexError, TypeError, ValueError):
                    batch_target = ""
                if batch_target:
                    break
            if batch_target:
                ret = (
                    "[TEST_TARGET_CONTRACT_BLOCKED]\n"
                    "Stage 23 still has a current implementation batch. Running an empty target would execute "
                    "the entire suite and mix expected unimplemented templates into the failure signature. "
                    f"Run exactly the current batch target first: RunTestCases('{batch_target}')"
                )
                self._record_structured_event("test_target_contract_blocked", {
                    "tool": "RunTestCases",
                    "target": pytest_args,
                    "required_target": batch_target,
                    "reason": "stage23_empty_target_would_mix_future_templates",
                })
                info("RunTestCases:\n" + ret)
                return self.attach_todo_summary(ret)
        if (
            stage_name == "test_case_implementation_in_batch"
            and any(
                os.path.basename(token.strip("'\"")) == "recordbug.py"
                for token in str(pytest_args or "").split()
            )
        ):
            ret = (
                "[TEST_TARGET_CONTRACT_BLOCKED]\n"
                "recordbug.py is a Stage 23 schema-writing skill, not a pytest target. "
                "Invoke the stage-provided script with RunSkillScript, then run the current TEST_BATCH_RUN_ARGS."
            )
            self._record_structured_event("test_target_contract_blocked", {
                "tool": "RunTestCases",
                "target": pytest_args,
                "reason": "stage23_recordbug_is_not_pytest",
            })
            info("RunTestCases:\n" + ret)
            return self.attach_todo_summary(ret)
        started = time.time()
        ret = self.free_pytest_run.do_check(pytest_args, timeout=timeout, return_line_coverage=return_line_coverage, detail=detail)
        success = None
        result_payload = None
        if isinstance(ret, tuple) and len(ret) >= 2:
            success = bool(ret[0])
            result_payload = ret[1]
        summary = self._summarize_test_result(result_payload)
        self._remember_failure_context("test_run", "RunTestCases", summary)
        repair_text = ""
        if self._summary_is_failure("test_run", summary) and self.stage_index < len(self.stages):
            repair_text = self._render_context_reuse_prompt(self.stages[self.stage_index], mark_injected=True)
        decision = self._record_structured_event("test_run", {
            "tool": "RunTestCases",
            "target": pytest_args,
            "timeout": timeout,
            "return_line_coverage": bool(return_line_coverage),
            "detail": bool(detail),
            "success": success,
            "duration_ms": int((time.time() - started) * 1000),
            "result": summary,
        })
        if raw_return:
            return ret
        ret = make_llm_tool_ret(ret[1])
        if repair_text:
            ret = repair_text + "\n\n" + ret
        ret = self._attach_progress_control(ret, decision)
        info("RunTestCases:\n" + ret)
        return self.attach_todo_summary(ret)
