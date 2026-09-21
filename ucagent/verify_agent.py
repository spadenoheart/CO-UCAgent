# -*- coding: utf-8 -*-

from curses import echo
from .tools.context import ArbitContextSummary
from .util.config import get_config
from .util.log import echo_g, echo_r, info, message, warning, error, msg_msg, get_log_logger
from .util.functions import (
    fmt_time_deta,
    fmt_time_stamp,
    get_template_path,
    render_template_dir,
    import_and_instance_tools,
    copy_skill_files,
    yam_str,
    make_llm_tool_ret,
    rm_workspace_prefix,
    start_verify_mcps,
    create_verify_mcps,
    stop_verify_mcps,
)
from .memory.long_term import LongTermMemoryStore
from .memory.context_reuse import ContextReuseStore
from .control import ProgressAwareLoopController
import ucagent.util.functions as fc
from .util.test_tools import ucagent_lib_path

import ucagent.tools
from .tools import *
from .tools.skill import ListSkill, _list_skills, list_skills_in_format
from .tools.planning import *
from .stage import StageManager
from .verify_pdb import VerifyPDB
from .interaction import EnhancedInteractionLogic, AdvancedInteractionLogic
from .version import __version__, __email__

import time
import random
import signal
import copy
import json
import re
from datetime import datetime
import threading
import shutil
import os
import glob

from .abackend import get_backend
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from uuid import uuid4
from typing import Any, Dict, List, Optional, OrderedDict
import traceback


class VerifyAgent:
    """AI-powered hardware verification agent for chip design testing."""

    def __init__(
        self,
        workspace: str,
        dut_name: str,
        output: str,
        config_file: Optional[str] = None,
        cfg_override: Optional[Dict[str, Any]] = None,
        tmp_overwrite: bool = False,
        template_dir: Optional[str] = None,
        template_cfg: Optional[Dict[str, Any]] = None,
        guid_doc_path: List[str] = [],
        stream_output: bool = False,
        init_cmd: Optional[List[str]] = None,
        seed: Optional[int] = None,
        sys_tips: str = "",
        ex_tools: Optional[List[str]] = None,
        thread_id: Optional[int] = None,
        debug: bool = False,
        no_embed_tools: bool = False,
        force_stage_index: int = 0,
        force_todo: bool = False,
        no_write_targets: Optional[List[str]] = None,
        interaction_mode: str = "standard",
        gen_instruct_file: Optional[str] = None,
        stage_skip_list: Optional[List[int]] = None,
        stage_unskip_list: Optional[List[int]] = None,
        use_todo_tools: bool = False,
        reference_files: dict = None,
        no_history: bool = False,
        enable_context_manage_tools: bool = False,
        exit_on_completion: bool = False,
        meta: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Verify Agent with configuration and an optional agent.

        Args:
            workspace (str): The workspace directory where the agent will operate.
            dut_name (str): The name of the device under test (DUT).
            output (str): The output directory for the agent's results.
            config_file (str, optional): Path to the configuration file. Defaults to None.
            cfg_override (dict, optional): Dictionary to override configuration settings. Defaults to None.
            tmp_overwrite (bool, optional): Whether to overwrite existing templates in the workspace. Defaults to False.
            template_dir (str, optional): Path to the template directory. Defaults to None.
            stream_output (bool, optional): Whether to stream output to the console. Defaults to False.
            init_cmd (list, optional): Initial commands to run in the agent. Defaults to None.
            seed (int, optional): Seed for random number generation. Defaults to None.
            sys_tips (str, optional): Set of system tips to be used in the agent.
                                      Defaults to an empty string.
            model (ChatOpenAI, optional): An instance of ChatOpenAI to use as the agent model.
                                          If None, a default model will be created using the configuration.
                                          Defaults to None.
            ex_tools (list, optional): List of external tools class to be used by the agent, e.g., `--ex-tools SqThink`.
                                       Defaults to None.
            thread_id (int, optional): Thread ID for the agent. If None, a random ID will be generated.
                                       Defaults to None.
            debug (bool, optional): Whether to enable debug mode. Defaults to False.
            no_embed_tools (bool, optional): Whether to disable embedded tools. Defaults to False.
            force_stage_index (int, optional): Force starting from a specific stage index. Defaults to 0.
            no_write_targets (list, optional): List of files/directories that cannot be written to. Defaults to None.
            interaction_mode (str, optional): Interaction mode - 'standard', 'enhanced', or 'advanced'. Defaults to 'standard'.
        """
        saved_info = {}
        if not no_history:
            saved_info = fc.load_ucagent_info(workspace)
        force_stage_index_explicit = force_stage_index != 0
        if force_stage_index == 0:
            force_stage_index = saved_info.get("stage_index", force_stage_index)
            if force_stage_index > 0:
                warning(f"Resuming from saved stage index: {force_stage_index}")
        self.workspace = os.path.abspath(workspace)
        self.__version__ = __version__
        self.config_file = "" if config_file is None else str(config_file)
        saved_meta = saved_info.get("meta") if isinstance(saved_info.get("meta"), dict) else {}
        self.meta = copy.deepcopy(saved_meta)
        if meta:
            self.meta.update(copy.deepcopy(meta))
            updated_info = copy.deepcopy(saved_info)
            updated_info["meta"] = copy.deepcopy(self.meta)
            fc.save_ucagent_info(self.workspace, updated_info)
        self.cfg = get_config(config_file, cfg_override, self.workspace)
        temp_args = {
            "OUT": output,
            "DUT": dut_name,
            "Version": __version__,
            "WORKSPACE": self.workspace,
        }
        self.cfg.update_template(temp_args)
        template_overwrite = self.cfg.template_overwrite.as_dict()
        self.cfg.update_template(template_overwrite)
        self.cfg.un_freeze()
        self.cfg.seed = seed if seed is not None else random.randint(1, 999999)
        self.cfg._temp_cfg = temp_args
        self.cfg.freeze()
        self.enable_rerank = self.cfg.get_value("context_upgrade.enable_rerank", False)
        self.enable_structured_summary = self.cfg.get_value("context_upgrade.enable_structured_summary", False)
        self.enable_hierarchical_summary = self.cfg.get_value("context_upgrade.enable_hierarchical_summary", False)
        self.enable_long_term_memory = self.cfg.get_value("context_upgrade.enable_long_term_memory", False)
        self.enable_long_term_memory_embed = self.cfg.get_value("context_upgrade.enable_long_term_memory_embed", False)
        self.enable_compact_test_output = self.cfg.get_value("context_upgrade.enable_compact_test_output", False)
        self.enable_data_collection = self.cfg.get_value("context_upgrade.enable_data_collection", False)
        self.enable_failure_aware_context = self.cfg.get_value("context_upgrade.enable_failure_aware_context", False)
        self.enable_context_reuse = self.cfg.get_value("context_upgrade.enable_context_reuse", False)
        self.enable_progress_controller = self._cfg_bool(
            "context_upgrade.enable_progress_controller", False
        )
        self.enable_stage_state_package = self._cfg_bool(
            "context_upgrade.enable_stage_state_package", False
        )
        self.enable_observation_masking = self._cfg_bool(
            "context_upgrade.enable_observation_masking", False
        )
        self.long_term_memory_cfg = self.cfg.get_value("context_upgrade.long_term_memory", {}) or {}
        if hasattr(self.long_term_memory_cfg, "as_dict"):
            self.long_term_memory_cfg = self.long_term_memory_cfg.as_dict()
        elif not isinstance(self.long_term_memory_cfg, dict):
            self.long_term_memory_cfg = {}
        self.failure_aware_context_cfg = self.cfg.get_value("context_upgrade.failure_aware_context", {}) or {}
        if hasattr(self.failure_aware_context_cfg, "as_dict"):
            self.failure_aware_context_cfg = self.failure_aware_context_cfg.as_dict()
        elif not isinstance(self.failure_aware_context_cfg, dict):
            self.failure_aware_context_cfg = {}
        self.context_reuse_cfg = self.cfg.get_value("context_upgrade.context_reuse", {}) or {}
        if hasattr(self.context_reuse_cfg, "as_dict"):
            self.context_reuse_cfg = self.context_reuse_cfg.as_dict()
        elif not isinstance(self.context_reuse_cfg, dict):
            self.context_reuse_cfg = {}
        self.progress_controller_cfg = self.cfg.get_value(
            "context_upgrade.progress_controller", {}
        ) or {}
        if hasattr(self.progress_controller_cfg, "as_dict"):
            self.progress_controller_cfg = self.progress_controller_cfg.as_dict()
        elif not isinstance(self.progress_controller_cfg, dict):
            self.progress_controller_cfg = {}
        self.stage_state_package_cfg = self.cfg.get_value(
            "context_upgrade.stage_state_package", {}
        ) or {}
        if hasattr(self.stage_state_package_cfg, "as_dict"):
            self.stage_state_package_cfg = self.stage_state_package_cfg.as_dict()
        elif not isinstance(self.stage_state_package_cfg, dict):
            self.stage_state_package_cfg = {}
        self.observation_masking_cfg = self.cfg.get_value(
            "context_upgrade.observation_masking", {}
        ) or {}
        if hasattr(self.observation_masking_cfg, "as_dict"):
            self.observation_masking_cfg = self.observation_masking_cfg.as_dict()
        elif not isinstance(self.observation_masking_cfg, dict):
            self.observation_masking_cfg = {}
        self.progress_controller = (
            ProgressAwareLoopController(self.progress_controller_cfg)
            if self.enable_progress_controller
            else None
        )
        self._resume_time_seconds = 0.0
        self._resume_token_in = 0
        self._resume_token_out = 0
        self._resume_from_log = False
        info(
            "[context_upgrade] flags: "
            f"enable_rerank={self.enable_rerank}, "
            f"enable_structured_summary={self.enable_structured_summary}, "
            f"enable_hierarchical_summary={self.enable_hierarchical_summary}, "
            f"enable_long_term_memory={self.enable_long_term_memory}, "
            f"enable_long_term_memory_embed={self.enable_long_term_memory_embed}, "
            f"enable_compact_test_output={self.enable_compact_test_output}, "
            f"enable_data_collection={self.enable_data_collection}, "
            f"enable_failure_aware_context={self.enable_failure_aware_context}, "
            f"enable_context_reuse={self.enable_context_reuse}, "
            f"enable_progress_controller={self.enable_progress_controller}, "
            f"enable_stage_state_package={self.enable_stage_state_package}, "
            f"enable_observation_masking={self.enable_observation_masking}"
        )
        self.workspace = os.path.abspath(workspace)
        self.output_dir = os.path.join(self.workspace, output)
        self.long_term_memory = None
        if self.enable_long_term_memory:
            self.long_term_memory = LongTermMemoryStore(
                self.workspace,
                dut_name,
                max_entries=int(self.long_term_memory_cfg.get("max_entries", 256) or 256),
                enable_embed=self.enable_long_term_memory_embed,
                embed_config=self.cfg.embed.as_dict() if hasattr(self.cfg, "embed") else None,
                options=self.long_term_memory_cfg,
            )
            info(f"[long_term_memory] enabled at {self.long_term_memory.path}")
        self.context_reuse = None
        if self.enable_context_reuse:
            self.context_reuse = ContextReuseStore(
                self.workspace,
                dut_name,
                options=self.context_reuse_cfg,
            )
        # copy doc/Guide_Doc to workspace
        guide_doc_path = os.path.join(self.workspace, self.cfg.guide_doc.path)
        if not os.path.exists(guide_doc_path) and self.cfg.guide_doc.enable:
            doc_guide_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "lang",
                self.cfg.lang,
                "doc",
                "Guide_Doc",
            )
            doc_files_to_append = []
            if len(guid_doc_path) > 0:
                for gfile in guid_doc_path:
                    if os.path.exists(gfile) is False:
                        warning(
                            f"Specified guid_doc_path {gfile} does not exist, ignore it"
                        )
                        continue
                    if os.path.isfile(gfile):
                        doc_files_to_append.append(gfile)
                        continue
                    if os.path.isdir(gfile):
                        doc_guide_path = gfile
                        continue
                    assert False, (
                        f"Specified guid_doc_path {gfile} is not a valid file or directory"
                    )
                assert os.path.exists(doc_guide_path), (
                    f"Specified guid_doc_path {doc_guide_path} does not exist"
                )
            shutil.copytree(doc_guide_path, guide_doc_path)
            for f in doc_files_to_append:
                shutil.copy(f, guide_doc_path)

        # if use_skill is enabled, copy skills to workspace, and add skill tools
        self.tool_skill = []
        if self.cfg.skill.use_skill:
            copy_skill_files(self.cfg, self.workspace,root_dir=os.path.dirname(os.path.abspath(__file__)))
            self.tool_skill += [ListSkill(self.workspace).bind(self),RunSkillScript(self.workspace).bind(self)]

        self.thread_id = (
            thread_id if thread_id is not None else random.randint(100000, 999999)
        )
        self.dut_name = dut_name
        self.seed = seed if seed is not None else random.randint(1, 999999)
        self.template = get_template_path(
            self.cfg.template, self.cfg.lang, template_dir
        )
        self.render_template(template_cfg=template_cfg, tmp_overwrite=tmp_overwrite)
        self.tool_read_text = ReadTextFile(self.workspace)
        self.tool_list_dir = PathList(self.workspace)
        self.tool_search_text = SearchText(self.workspace)
        self.todo_panel = ToDoPanel()
        self.stage_manager = StageManager(
            self.workspace,
            self.cfg,
            self,
            self.tool_read_text,
            saved_info,
            force_stage_index,
            force_stage_index_explicit=force_stage_index_explicit,
            force_todo=force_todo,
            todo_panel=self.todo_panel,
            stage_skip_list=stage_skip_list,
            stage_unskip_list=stage_unskip_list,
            tool_inspect_file=[
                self.tool_read_text,
                self.tool_list_dir,
                self.tool_search_text,
            ],
            reference_files=reference_files,
        )
        self._default_system_prompt = (
            sys_tips if sys_tips else self.get_default_system_prompt()
        )
        self.tool_list_base = [
            self.tool_read_text,
            RoleInfo(self._default_system_prompt),
        ]
        if not no_embed_tools:
            self.tool_reference = SemanticSearchInGuidDoc(
                self.cfg.embed, workspace=self.workspace, doc_path="Guide_Doc"
            )
            self.tool_memory_put = MemoryPut().set_store(self.cfg.embed)
            self.tool_memory_get = MemoryGet().set_store(
                store=self.tool_memory_put.get_store()
            )
            self.tool_list_base += [
                self.tool_reference,
                self.tool_memory_put,
                self.tool_memory_get,
            ]
        if no_write_targets is not None:
            assert isinstance(no_write_targets, list), (
                "no_write_targets must be a list of directories or files"
            )
            for f in no_write_targets:
                abs_f = os.path.abspath(f)
                assert os.path.exists(abs_f), (
                    f"Specified no-write target {abs_f} does not exist"
                )
                assert abs_f.startswith(os.path.abspath(self.workspace)), (
                    f"Specified no-write target {abs_f} must be under the workspace {self.workspace}"
                )
                self.cfg.un_write_dirs.append(
                    rm_workspace_prefix(self.workspace, abs_f)
                )
        self.cwd_read_only_files = fc.chmode_ro_by_pattern(
            self.workspace, self.cfg.get_value("un_write_dirs", [])
        )
        self.tool_list_file = [
            # Directory and file listing tools
            self.tool_list_dir,
            GetFileInfo(self.workspace),
            # File reading tools
            # ReadBinFile(self.workspace), # ignore Binary file read
            # File searching tools
            self.tool_search_text,
            FindFiles(self.workspace),
            # File writing and editing tools (require permissions)
            DeleteFile(
                self.workspace,
                write_dirs=self.cfg.write_dirs,
                un_write_dirs=self.cfg.un_write_dirs,
            ),
            EditTextFile(
                self.workspace,
                write_dirs=self.cfg.write_dirs,
                un_write_dirs=self.cfg.un_write_dirs,
            ),
            EditFileDialogs(
                self.workspace,
                write_dirs=self.cfg.write_dirs,
                un_write_dirs=self.cfg.un_write_dirs,
            ),
            ReplaceStringInFile(
                self.workspace,
                write_dirs=self.cfg.write_dirs,
                un_write_dirs=self.cfg.un_write_dirs,
            ),
            # File management tools (require permissions)
            CopyFile(
                self.workspace,
                write_dirs=self.cfg.write_dirs,
                un_write_dirs=self.cfg.un_write_dirs,
            ),
            MoveFile(
                self.workspace,
                write_dirs=self.cfg.write_dirs,
                un_write_dirs=self.cfg.un_write_dirs,
            ),
            CreateDirectory(
                self.workspace,
                write_dirs=self.cfg.write_dirs,
                un_write_dirs=self.cfg.un_write_dirs,
            ),
            # Workspace git management tools
            WorkDiff(self.workspace),
            WorkCommit(self.workspace),
            # bash tool
            RunBashCommand(self.workspace),
        ]
        self.tool_list_task = self.stage_manager.new_tools()
        self.tool_list_ext = import_and_instance_tools(
            self.cfg.get_value("ex_tools", []), ucagent.tools
        ) + import_and_instance_tools(ex_tools, ucagent.tools) + self.tool_skill

        # Export workspace path via environment variable for ext tools
        os.environ["UCAGENT_WORKSPACE"] = self.workspace

        # Initialize planning tools
        self.planning_tools = []
        self.force_todo = force_todo
        if (interaction_mode == "standard" and force_todo) or use_todo_tools:
            self.planning_tools = [
                CreateToDo(self.todo_panel),
                CompleteToDoSteps(self.todo_panel),
                UndoToDoSteps(self.todo_panel),
                ResetToDo(self.todo_panel),
                GetToDoSummary(self.todo_panel),
                ToDoState(self.todo_panel),
            ]

        self.max_token = self.cfg.get_value(
            "conversation_summary.max_tokens", 20 * 1024
        )
        self.max_summary_tokens = self.cfg.get_value(
            "conversation_summary.max_summary_tokens", 1 * 1024
        )
        self.hard_max_token = self.cfg.get_value(
            "conversation_summary.hard_max_tokens", self.max_token
        )
        self.hard_max_token = max(int(self.max_token), int(self.hard_max_token))
        self.context_management_strategy = self.cfg.get_value(
            "conversation_summary.context_management_strategy",
            "TrimAndSummaryMiddleware",
        )
        self.max_keep_msgs = self.cfg.get_value(
            "conversation_summary.max_keep_msgs", 200
        )
        self.tail_keep_msgs = self.cfg.get_value(
            "conversation_summary.tail_keep_msgs", 20
        )
        self.message_echo_handler = None
        self.update_handler = None
        self._time_start = time.time()
        self._time_end = None
        # state
        self._msg_buffer = ""
        self._system_message = self._default_system_prompt
        # flags
        self.stream_output = stream_output
        self.invoke_round = 0
        self._tool__call_error = []
        self._is_exit = False
        self._sync_workspace_back_on_exit_done = False
        self._tip_index = 0
        self._need_break = False
        self._break_threads: set[int] = set()
        self._need_human = False
        self._force_trace = False
        self._continue_msg = None
        self._no_action_rounds = 0
        self._mcps = None               # set by PdbMcpServer for api_master heartbeat
        self._mcp_server_thread = None   # set by PdbMcpServer for api_master heartbeat
        self._mcps_logger = None
        self.structured_events_path = os.path.join(self.output_dir, "structured_events.jsonl")
        if self.enable_data_collection or self.enable_progress_controller:
            self._register_file_mutation_event_callbacks()
        self.original_sigint = signal.getsignal(signal.SIGINT)
        self._sigint_count = 0
        self._exit_on_completion = exit_on_completion
        self._exit_on_completion_pending = False
        self._exit_on_completion_queued = False
        self._is_work_busy = False
        self.handle_sigint()

        # Initialize interaction logic based on mode
        self.interaction_mode = interaction_mode
        self.enhanced_logic = None
        self.advanced_logic = None

        if interaction_mode == "enhanced":
            self.enhanced_logic = EnhancedInteractionLogic(self)
            info("Using enhanced interaction mode with planning and memory management")
        elif interaction_mode == "advanced":
            self.advanced_logic = AdvancedInteractionLogic(self)
            info("Using advanced interaction mode with adaptive strategies")
        else:
            info("Using standard interaction mode")
        self.generate_instruction_file(gen_instruct_file)
        cfg_icmds = self.cfg.get_value("init_cmds", [])
        if cfg_icmds:
            if init_cmd is None:
                init_cmd = []
            init_cmd = init_cmd + cfg_icmds
        # PDB and backend
        self.backend = get_backend(self, self.cfg)
        self.message_manage_node = self.backend.get_message_manage_node()
        self.context_tools = []
        if enable_context_manage_tools:
            if self.message_manage_node is not None:
                self.context_tools = [
                    ArbitContextSummary().bind(self.message_manage_node),
                ]
            else:
                warning(
                    "Context management tools are enabled but no message management node is available."
                )
        self.test_tools = fc.get_tools_from_cfg(
            self.tool_list_base
            + self.tool_list_file
            + self.tool_list_task
            + self.tool_list_ext
            + self.planning_tools
            + self.context_tools,
            self.cfg.tools.as_dict(),
        )
        self.pdb = VerifyPDB(
            self,
            init_cmd=init_cmd,
            max_loop_retry=self.cfg.loop_settings.max_loop_retry,
            retry_delay=(
                self.cfg.loop_settings.retry_delay_start,
                self.cfg.loop_settings.retry_delay_end,
            ),
            loop_alive_time=self.cfg.loop_settings.loop_alive_time,
        )
        self.backend.init()
        self.backend.set_debug(debug)
        self.set_tool_call_time_out(self.cfg.get_value("call_time_out", 300))
        self.stage_manager.init_stage()
        # Telemetry
        self.session_id = uuid4()
        langfuse_cfg = self.cfg.get_value("langfuse", {})
        self.langfuse_enable = langfuse_cfg.get_value("enable", False) is True
        if self.langfuse_enable:
            public_key = langfuse_cfg.get_value("public_key", "")
            secret_key = langfuse_cfg.get_value("secret_key", "")
            base_url = langfuse_cfg.get_value("base_url", "")
            self.langfuse = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                base_url=base_url,
            )
            assert self.langfuse.auth_check(), (
                "Can't connect to langfuse, please check your configuration"
            )
            self.langfuse_handler = CallbackHandler()

    def get_messages_cfg(self, keys: Optional[List[str]] = None) -> Dict[str, Any]:
        if self.message_manage_node is None:
            return {}
        ret = {"__manage_class__": self.message_manage_node.__class__.__name__}
        for k in keys:
            if hasattr(self.message_manage_node, k):
                ret[k] = getattr(self.message_manage_node, k)
        return ret

    def set_messages_cfg(self, cfg: Dict[str, Any]):
        success = {}
        if self.message_manage_node is None:
            return success
        for k, v in cfg.items():
            if hasattr(self.message_manage_node, k):
                setattr(self.message_manage_node, k, v)
                success[k] = v
        return success

    def summary_mode(self):
        if self.message_manage_node is None:
            return "None"
        name = self.message_manage_node.__class__.__name__
        if self.context_management_strategy == "TrimAndSummaryMiddleware":
            return f"{name}({self.max_keep_msgs})"
        return f"{name}({self.max_token})"

    def summary_max_tokens(self):
        return self.max_summary_tokens

    def generate_instruction_file(self, file_path):
        if not file_path:
            return
        if file_path.startswith(os.sep):
            file_path = file_path[1:]
        file_path = os.path.abspath(os.path.join(self.workspace, file_path))
        dut_readme = os.path.join(self.workspace, self.dut_name, "README.md")
        with open(file_path, "w", encoding="utf-8") as f:
            if os.path.exists(dut_readme):
                f.write("# Goal Description\n")
                with open(dut_readme, "r", encoding="utf-8") as df:
                    f.write(df.read() + "\n")
            f.write("# Verification Instruction\n")
            f.write(self._default_system_prompt + "\n")

    def render_template(self, template_cfg=None, tmp_overwrite=False):
        template_context = {
            "DUT": self.dut_name,
            "Version": __version__,
            "Email": __email__,
            "CWD": self.workspace,
            "UC_LIB_PATH": ucagent_lib_path(),
        }
        if template_cfg is not None:
            template_context.update(template_cfg)
        if self.template is not None:
            tmp_dir = os.path.join(self.workspace, os.path.basename(self.template))
            info(f"Rendering template from {self.template} to {tmp_dir}")
            if not os.path.exists(tmp_dir) or tmp_overwrite:
                try:
                    render_template_dir(self.workspace, self.template, template_context)
                except Exception as e:
                    debug(traceback.format_exc())
                    error(
                        f"Failed to render template from {self.template} to {tmp_dir}: {e}"
                    )
                    raise e

    def set_message_echo_handler(self, handler):
        """Set a custom message echo handler to process messages."""
        if not callable(handler):
            raise ValueError("Message echo handler must be callable")
        self.message_echo_handler = handler

    def unset_message_echo_handler(self):
        """Unset the custom message echo handler."""
        self.message_echo_handler = None

    def message_echo(self, msg, end="\n"):
        """Echo a message using the custom message echo handler if set."""
        if self.message_echo_handler is not None:
            self.message_echo_handler(msg, end)
            if msg:
                self._msg_buffer = self._msg_buffer + msg + end
            if end == "\n":
                msg_msg(self._msg_buffer)
                self._msg_buffer = ""
        else:
            message(msg, end=end)

    def handle_sigint(self):
        def _sigint_handler(s, f):
            self._sigint_count += 1
            if self._sigint_count > 4:
                return self.original_sigint(s, f)
            if self._sigint_count > 3:
                info("SIGINT received again, exiting...")
                self.exit()
                return
            if self._sigint_count > 1:
                # self.original_sigint(s, f)
                info("SIGINT received again, more times will exit directly")
                return
            info("SIGINT received")
            self.set_break(True)

        signal.signal(signal.SIGINT, _sigint_handler)

    def set_force_trace(self, value):
        self._force_trace = value

    def check_pdb_trace(self):
        if self._force_trace:
            self.pdb.set_trace()
        elif self.is_break():
            self.pdb.set_trace()

    def set_break(self, value=True):
        self._need_break = value

    def is_break(self):
        return self._need_break or threading.current_thread().ident in self._break_threads

    def set_break_thread(self, thread_id: int) -> None:
        self.set_break(True)
        self._break_threads.add(thread_id)

    def clear_break_thread(self, thread_id: int) -> None:
        self._break_threads.discard(thread_id)

    def clear_all_break_threads(self) -> None:
        self._break_threads.clear()

    def get_current_tips(self):
        if self._tool__call_error:
            return {"messages": copy.deepcopy(self._tool__call_error)}
        tips = self._continue_msg
        if self._continue_msg is None:
            tips = yam_str(self.stage_manager.get_current_tips())
        else:
            self._continue_msg = None
        self._tip_index += 1
        assert isinstance(tips, str), "StageManager should return a str type tips"
        msg = []
        if self._system_message:
            msg.append(self.backend.get_system_message(copy.copy(self._system_message)))
            self._system_message = None
        msg.append(self.backend.get_human_message(tips))
        return {"messages": msg}

    def set_system_message(self, msg: str):
        self._system_message = msg

    def get_system_message(self):
        """Get the current system message for the agent."""
        return self._system_message

    def get_default_system_prompt(self):
        """Get the default system prompt for the agent. And if skill is enabled, include skill prompt and skill list."""
        system = self.cfg.mission.prompt.get_value("system", "").strip()
        if self.cfg.skill.use_skill:
            formatted_skill_list = list_skills_in_format(_list_skills(self.workspace),self.workspace,self.cfg.skill.general_skill_list)
            skill_prompt = self.cfg.mission.prompt.get_value("skill_system", "").replace("{general_skill_list}", formatted_skill_list)
            system = system.replace("{skill_system}", skill_prompt)
        else:
            system = system.replace("{skill_system}", "")
        warning(f"System prompt: {system}")
        return system

    def set_continue_msg(self, msg: str):
        """Set the continue message for the agent."""
        if not isinstance(msg, str):
            raise ValueError("Continue message must be a string")
        try:
            msg.encode("utf-8").decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("Continue message must be a valid UTF-8 string")
        self._continue_msg = msg

    def get_stat_info(self):
        return {
            "version": self.__version__,
            "seed": self.seed,
            "dut_name": self.dut_name,
            "DUT": self.dut_name,
            "config_file": self.config_file,
            "config_arg": self.config_file,
            "mission_name": self.cfg.mission.name,
            "meta": copy.deepcopy(self.meta),
        }

    def is_exit(self):
        if self._is_exit:
            info("Verify Agent is exited.")
        return self._is_exit

    def exit(self):
        if self.is_exit():
            return
        try:
            self._is_exit = True
            try:
                if getattr(self, "stage_manager", None) is not None:
                    self.stage_manager.save_stage_info()
            except Exception as exc:
                warning(f"Failed to save stage information on exit: {exc}")
            self._sync_workspace_back_on_exit()
        finally:
            fc.chmode_rw(self.cwd_read_only_files)

    def exit_unset(self):
        if not self.is_exit():
            return False
        self._is_exit = False
        return True

    def _cfg_bool(self, key: str, default: bool = False) -> bool:
        value = self.cfg.get_value(key, default)
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        raw = str(value).strip().lower()
        if raw in {"1", "true", "yes", "y", "on"}:
            return True
        if raw in {"0", "false", "no", "n", "off"}:
            return False
        return default

    def _sync_workspace_back_on_exit(self):
        if self._sync_workspace_back_on_exit_done:
            return
        self._sync_workspace_back_on_exit_done = True
        if not self._cfg_bool("master_api.sync_workspace.on_exit", True):
            return
        pdb = getattr(self, "pdb", None)
        master_clients = getattr(pdb, "_master_clients", {}) or {}
        if not master_clients:
            return
        for url, client in list(master_clients.items()):
            if not getattr(client, "is_running", False):
                continue
            ok, msg = client.sync_workspace_back(reason="exit")
            if ok:
                info(msg)
            else:
                info(f"Workspace sync-back skipped for {url}: {msg}")

    def protect_files_on(self, new_files: List[str]):
        for f in new_files:
            fpath = os.path.abspath(self.workspace + os.path.sep + f)
            if not os.path.exists(fpath):
                warning(
                    f"File to protect does not exist: {f} in workspace {self.workspace}"
                )
                continue
            if fpath not in self.cwd_read_only_files:
                info(f"Set file to read-only: {fpath}")
                self.cwd_read_only_files.append(fpath)
        fc.chmode_ro(self.cwd_read_only_files)

    def protect_files_off(self, files: List[str]):
        off_files = []
        for f in files:
            fpath = os.path.abspath(self.workspace + os.path.sep + f)
            if fpath in self.cwd_read_only_files:
                info(f"Set file to read-write: {fpath}")
                off_files.append(fpath)
                self.cwd_read_only_files.remove(fpath)
            else:
                warning(
                    f"File to un-protect not in read-only list: {f} in workspace {self.workspace}"
                )
        if not files:
            info(
                "No files specified to un-protect, restoring all read-only files to read-write"
            )
            fc.chmode_rw(self.cwd_read_only_files)
        else:
            fc.chmode_rw(off_files)

    def try_exit_on_completion(self):
        if self._exit_on_completion:
            self.set_break(False)
            if self.is_work_busy():
                self._exit_on_completion_pending = True
                return
            self._queue_exit_on_completion()

    def _queue_exit_on_completion(self):
        if self._exit_on_completion_queued:
            return
        self._exit_on_completion_pending = False
        self._exit_on_completion_queued = True
        self.pdb.add_cmds(["sleep 5"] + ["quit"] * 3)

    def get_work_config(self):
        return self.backend.get_work_config()

    def run(self):
        self.pre_run()
        self.run_loop()

    def pre_run(self):
        time_start = self._time_start = time.time()
        info(
            "Verify Agent started at: "
            + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_start))
        )
        info("Seed: " + str(self.seed))
        self.check_pdb_trace()
        return self

    def run_loop(self, msg=None):
        if msg:
            self.set_continue_msg(msg)
        self._need_human = False
        # conversation loop
        while not self.is_exit():
            self.one_loop()
            if self.is_exit():
                break
            if self.is_break():
                info("Break at loop: " + str(self.invoke_round))
                return
            if self._need_human:
                info("Waiting for human input at loop: " + str(self.invoke_round))
                return
            self.check_pdb_trace()
        time_end = self._time_end = time.time()
        info("Verify Agent finished at: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_end)))
        total_elapsed = time_end - self._time_start
        if self._resume_from_log and self._resume_time_seconds > 0:
            total_elapsed += self._resume_time_seconds
        info(f"Total time taken: {fmt_time_deta(total_elapsed)}")
        if self.enable_data_collection:
            stats = self.backend.get_statistics()
            msg_in = stats.get("message_in") if isinstance(stats, dict) else None
            msg_out = stats.get("message_out") if isinstance(stats, dict) else None
            total_msg_in = int(msg_in) if isinstance(msg_in, (int, float)) and msg_in >= 0 else None
            total_msg_out = int(msg_out) if isinstance(msg_out, (int, float)) and msg_out >= 0 else None
            if self._resume_from_log:
                if total_msg_in is None:
                    total_msg_in = 0
                if total_msg_out is None:
                    total_msg_out = 0
                total_msg_in += self._resume_token_in
                total_msg_out += self._resume_token_out
            if isinstance(total_msg_in, int) and total_msg_in >= 0 and isinstance(total_msg_out, int) and total_msg_out >= 0:
                info(
                    f"Total tokens used: in={self._format_tokens_short(int(total_msg_in))} "
                    f"out={self._format_tokens_short(int(total_msg_out))}"
                )
                info(
                    f"[data_collection] summary: time={self._format_duration_hm(total_elapsed)} "
                    f"token_in={self._format_tokens_short(int(total_msg_in))} "
                    f"token_out={self._format_tokens_short(int(total_msg_out))}"
                )
            llm_performance = stats.get("llm_performance", {}) if isinstance(stats, dict) else {}
            for role, metrics in llm_performance.items():
                def perf_metric(name):
                    value = metrics.get(name)
                    return "NA" if value is None else f"{float(value):.2f}"

                info(
                    "[data_collection][llm_performance] "
                    f"role={role} requests={metrics.get('requests', 0)} "
                    f"success={metrics.get('success', 0)} errors={metrics.get('errors', 0)} "
                    f"latency_ms_p50={perf_metric('latency_ms_p50')} "
                    f"latency_ms_p95={perf_metric('latency_ms_p95')} "
                    f"ttft_ms_p50={perf_metric('ttft_ms_p50')} "
                    f"ttft_ms_p95={perf_metric('ttft_ms_p95')} "
                    f"prefill_tps_est_p50={perf_metric('prefill_tps_est_p50')} "
                    f"decode_tps_est_p50={perf_metric('decode_tps_est_p50')}"
                )
            memory_metrics = self._collect_memory_cache_metrics(stats if isinstance(stats, dict) else {})
            if memory_metrics:
                info(
                    "[data_collection][memory_cache] "
                    f"stage_cache_hit_rate={memory_metrics.get('stage_cache_hit_rate', 0.0):.4f} "
                    f"prefetch_hit_rate={memory_metrics.get('prefetch_hit_rate', 0.0):.4f} "
                    f"stage_first_turn_hit_rate={memory_metrics.get('stage_first_turn_hit_rate', 0.0):.4f} "
                    f"retrieval_hit_rate={memory_metrics.get('retrieval_hit_rate', 0.0):.4f} "
                    f"recent_fallback_rate={memory_metrics.get('recent_fallback_rate', 0.0):.4f} "
                    f"useful_hit_rate={memory_metrics.get('useful_hit_rate', 0.0):.4f} "
                    f"retrieval_useful_hit_rate={memory_metrics.get('retrieval_useful_hit_rate', 0.0):.4f} "
                    f"prefetch_useful_hit_rate={memory_metrics.get('prefetch_useful_hit_rate', 0.0):.4f} "
                    f"fallback_useful_hit_rate={memory_metrics.get('fallback_useful_hit_rate', 0.0):.4f} "
                    f"stale_hit_rate={memory_metrics.get('stale_hit_rate', 0.0):.4f} "
                    f"memory_pollution_rate={memory_metrics.get('memory_pollution_rate', 0.0):.4f} "
                    f"prefetch_pollution_rate={memory_metrics.get('prefetch_pollution_rate', 0.0):.4f} "
                    f"context_reuse_queries={int(memory_metrics.get('context_reuse_queries', 0) or 0)} "
                    f"context_reuse_hit_rate={memory_metrics.get('context_reuse_hit_rate', 0.0):.4f} "
                    f"context_reuse_injections={int(memory_metrics.get('context_reuse_injections', 0) or 0)} "
                    f"context_reuse_items={int(memory_metrics.get('context_reuse_items', 0) or 0)} "
                    f"context_reuse_throttled={int(memory_metrics.get('context_reuse_throttled', 0) or 0)} "
                    f"hard_gate_queries={int(memory_metrics.get('context_reuse_store_hard_gate_queries', 0) or 0)} "
                    f"hard_gate_rejected={int(memory_metrics.get('context_reuse_store_hard_gate_rejected', 0) or 0)} "
                    f"utility_candidates={int(memory_metrics.get('context_reuse_utility_candidates', 0) or 0)} "
                    f"utility_passed={int(memory_metrics.get('context_reuse_utility_passed', 0) or 0)} "
                    f"utility_rejected={int(memory_metrics.get('context_reuse_utility_rejected', 0) or 0)} "
                    f"utility_unknown={int(memory_metrics.get('context_reuse_utility_unknown', 0) or 0)} "
                    f"utility_pass_rate={memory_metrics.get('context_reuse_utility_pass_rate', 0.0):.4f} "
                    f"utility_prompt_tokens_estimated="
                    f"{int(memory_metrics.get('context_reuse_utility_prompt_tokens_estimated', 0) or 0)} "
                    f"contract_candidates="
                    f"{int(memory_metrics.get('context_reuse_store_contract_candidates', 0) or 0)} "
                    f"contract_applicable="
                    f"{int(memory_metrics.get('context_reuse_store_contract_applicable', 0) or 0)} "
                    f"contract_inapplicable="
                    f"{int(memory_metrics.get('context_reuse_store_contract_inapplicable', 0) or 0)} "
                    f"contract_enforced_rejected="
                    f"{int(memory_metrics.get('context_reuse_store_contract_enforced_rejected', 0) or 0)}"
                )
            self._append_res_csv(
                total_elapsed,
                total_msg_in,
                total_msg_out,
                memory_metrics=memory_metrics,
                replace_last=self._resume_from_log,
            )
        return self

    def _collect_memory_cache_metrics(self, backend_stats: dict) -> dict:
        metrics = {}
        stage_metrics = {}
        if hasattr(self, "stage_manager") and self.stage_manager is not None and hasattr(self.stage_manager, "get_memory_metrics"):
            stage_metrics = self.stage_manager.get_memory_metrics() or {}
        ltm_metrics = {}
        if getattr(self, "long_term_memory", None) is not None and hasattr(self.long_term_memory, "get_runtime_metrics"):
            ltm_metrics = self.long_term_memory.get_runtime_metrics() or {}
        context_reuse_metrics = {}
        if getattr(self, "context_reuse", None) is not None and hasattr(self.context_reuse, "get_runtime_metrics"):
            context_reuse_metrics = self.context_reuse.get_runtime_metrics() or {}
        message_metrics = {}
        if isinstance(backend_stats, dict):
            message_metrics = backend_stats.get("memory_hit", {}) or {}

        metrics.update(stage_metrics)
        for key, value in ltm_metrics.items():
            metrics.setdefault(key, value)
        for key, value in message_metrics.items():
            metrics[key] = value
        for key, value in context_reuse_metrics.items():
            metric_key = f"context_reuse_{key}" if key.startswith("utility_") else f"context_reuse_store_{key}"
            metrics[metric_key] = value

        prompt_queries = int(stage_metrics.get("prompt_queries", 0) or 0)
        injected_items = int(stage_metrics.get("injected_items", 0) or 0)
        useful_hits = int(message_metrics.get("useful_hits", 0) or 0)
        retrieval_useful_hits = int(message_metrics.get("retrieval_useful_hits", 0) or 0)
        prefetch_useful_hits = int(message_metrics.get("prefetch_useful_hits", 0) or 0)
        fallback_useful_hits = int(message_metrics.get("fallback_useful_hits", 0) or 0)
        memory_pollution = int(message_metrics.get("memory_pollution", 0) or 0)
        retrieval_pollution = int(message_metrics.get("retrieval_pollution", 0) or 0)
        prefetch_pollution = int(message_metrics.get("prefetch_pollution", 0) or 0)
        fallback_pollution = int(message_metrics.get("fallback_pollution", 0) or 0)
        total_memory_hits = (
            int(stage_metrics.get("retrieval_hits", 0) or 0)
            + int(stage_metrics.get("prefetch_hits", 0) or 0)
            + int(stage_metrics.get("recent_fallback_hits", 0) or 0)
        )
        stale_hits = max(
            int(stage_metrics.get("stale_hits", 0) or 0),
            int(message_metrics.get("stale_hits", 0) or 0),
            int(ltm_metrics.get("stale_hits", 0) or 0),
        )
        metrics["useful_hit_rate"] = round(useful_hits / max(1, total_memory_hits), 4)
        metrics["retrieval_useful_hit_rate"] = round(retrieval_useful_hits / max(1, int(stage_metrics.get("retrieval_hits", 0) or 0)), 4)
        metrics["prefetch_useful_hit_rate"] = round(prefetch_useful_hits / max(1, int(stage_metrics.get("prefetch_hits", 0) or 0)), 4)
        metrics["fallback_useful_hit_rate"] = round(fallback_useful_hits / max(1, int(stage_metrics.get("recent_fallback_hits", 0) or 0)), 4)
        metrics["memory_pollution_rate"] = round(memory_pollution / max(1, int(stage_metrics.get("prompt_injections", 0) or 0)), 4)
        metrics["retrieval_pollution_rate"] = round(retrieval_pollution / max(1, int(stage_metrics.get("retrieval_hits", 0) or 0)), 4)
        metrics["prefetch_pollution_rate"] = round(prefetch_pollution / max(1, int(stage_metrics.get("prefetch_hits", 0) or 0)), 4)
        metrics["fallback_pollution_rate"] = round(fallback_pollution / max(1, int(stage_metrics.get("recent_fallback_hits", 0) or 0)), 4)
        metrics["stale_hit_rate"] = round(stale_hits / max(1, injected_items), 4)
        token_increase = injected_items * 120
        metrics["memory_token_increase"] = token_increase
        metrics["token_roi_memory"] = round(useful_hits / max(1, token_increase), 6)
        metrics["prompt_queries"] = prompt_queries
        return metrics

    def _format_tokens_short(self, tokens: int) -> str:
        if tokens is None or tokens < 0:
            return "N/A"
        if tokens >= 1_000_000:
            return f"{tokens / 1_000_000:.2f}M"
        if tokens >= 1_000:
            return f"{tokens / 1_000:.1f}K"
        return str(tokens)

    def _format_duration_hm(self, seconds: float) -> str:
        if seconds is None:
            return "N/A"
        minutes = int(seconds // 60)
        hours = minutes // 60
        mins = minutes % 60
        if hours > 0:
            return f"{hours}h{mins}min"
        return f"{mins}min"

    def _get_log_file_path(self) -> Optional[str]:
        logger = get_log_logger()
        if not logger:
            return None
        for handler in logger.handlers:
            base_filename = getattr(handler, "baseFilename", None)
            if base_filename:
                return base_filename
        return None

    def _parse_tokens_short(self, value: str) -> Optional[int]:
        if not value:
            return None
        text = value.strip().rstrip(",")
        if text.upper() == "N/A":
            return None
        multiplier = 1
        if text.endswith("M"):
            multiplier = 1_000_000
            text = text[:-1]
        elif text.endswith("K"):
            multiplier = 1_000
            text = text[:-1]
        try:
            return int(float(text) * multiplier)
        except ValueError:
            return None

    def _parse_duration_hm(self, value: str) -> Optional[int]:
        if not value:
            return None
        match = re.match(r"^(?:(\d+)h)?(\d+)min$", value)
        if not match:
            return None
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2))
        return hours * 3600 + minutes * 60

    def _parse_duration_hms(self, value: str) -> Optional[int]:
        if not value:
            return None
        try:
            parts = value.split(":")
            if len(parts) != 3:
                return None
            hours, minutes, seconds = [int(p) for p in parts]
            return hours * 3600 + minutes * 60 + seconds
        except ValueError:
            return None

    def _parse_resume_stats_from_log(self, log_path: str):
        if not log_path or not os.path.isfile(log_path):
            return 0, 0, 0
        try:
            lines = open(log_path, "r", encoding="utf-8", errors="ignore").read().splitlines()
        except Exception:
            return 0, 0, 0

        ts_re = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} - .*? - .*? - (?P<msg>.*)$")
        summary_re = re.compile(r"\[data_collection\] summary: time=(\S+) token_in=(\S+) token_out=(\S+)")
        total_time_re = re.compile(r"Total time taken: (\d{2}:\d{2}:\d{2})")
        total_tokens_re = re.compile(r"Total tokens used: in=(\S+) out=(\S+)")
        stage_totals_re = re.compile(r"token_in_total=(\d+) token_out_total=(\d+)")
        carry_forward_re = re.compile(
            r"\[data_collection\]\[resume_carry_forward\] "
            r"reset_stage=(\d+) time_seconds=([0-9.]+) "
            r"token_in=(\d+) token_out=(\d+)"
        )

        runs = []
        current_run = None
        last_ts = None
        prev_ts = None

        def new_run(start_ts):
            return {
                "start_ts": start_ts,
                "end_ts": None,
                "summary_time": None,
                "total_time": None,
                "summary_tokens": None,
                "total_tokens": None,
                "stage_tokens": None,
                "carry_forward": None,
            }

        for line in lines:
            match = ts_re.match(line)
            if not match:
                continue
            ts = datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S")
            msg = match.group("msg")
            prev_ts = last_ts
            last_ts = ts

            if msg.startswith("Verify Agent started at:"):
                if current_run and current_run["end_ts"] is None:
                    current_run["end_ts"] = prev_ts
                    runs.append(current_run)
                current_run = new_run(ts)
                continue

            # Configuration and setup messages can precede the explicit agent
            # start marker. They are not part of a resumable execution segment.
            if current_run is None:
                continue

            if msg.startswith("Verify Agent finished at:"):
                current_run["end_ts"] = ts
                runs.append(current_run)
                current_run = None
                continue

            carry_forward_match = carry_forward_re.search(msg)
            if carry_forward_match:
                current_run["carry_forward"] = {
                    "reset_stage": int(carry_forward_match.group(1)),
                    "time_seconds": float(carry_forward_match.group(2)),
                    "tokens": (
                        int(carry_forward_match.group(3)),
                        int(carry_forward_match.group(4)),
                    ),
                }
                continue

            summary_match = summary_re.search(msg)
            if summary_match:
                summary_time = self._parse_duration_hm(summary_match.group(1))
                token_in = self._parse_tokens_short(summary_match.group(2))
                token_out = self._parse_tokens_short(summary_match.group(3))
                if summary_time is not None:
                    current_run["summary_time"] = summary_time
                if token_in is not None and token_out is not None:
                    current_run["summary_tokens"] = (token_in, token_out)
                continue

            total_time_match = total_time_re.search(msg)
            if total_time_match:
                total_time = self._parse_duration_hms(total_time_match.group(1))
                if total_time is not None:
                    current_run["total_time"] = total_time
                continue

            total_tokens_match = total_tokens_re.search(msg)
            if total_tokens_match:
                token_in = self._parse_tokens_short(total_tokens_match.group(1))
                token_out = self._parse_tokens_short(total_tokens_match.group(2))
                if token_in is not None and token_out is not None:
                    current_run["total_tokens"] = (token_in, token_out)
                continue

            stage_match = stage_totals_re.search(msg)
            if stage_match:
                current_run["stage_tokens"] = (int(stage_match.group(1)), int(stage_match.group(2)))

        if current_run:
            if current_run["end_ts"] is None:
                current_run["end_ts"] = last_ts
            runs.append(current_run)

        total_time_seconds = 0
        total_token_in = 0
        total_token_out = 0

        for run in runs:
            run_time = None
            if run["carry_forward"] is not None:
                run_time = run["carry_forward"]["time_seconds"]
            elif run["summary_time"] is not None:
                run_time = run["summary_time"]
            elif run["total_time"] is not None:
                run_time = run["total_time"]
            elif run["start_ts"] and run["end_ts"]:
                run_time = int((run["end_ts"] - run["start_ts"]).total_seconds())
            if run_time is not None and run_time >= 0:
                total_time_seconds += run_time

            tokens = None
            if run["carry_forward"] is not None:
                tokens = run["carry_forward"]["tokens"]
            elif run["summary_tokens"] is not None:
                tokens = run["summary_tokens"]
            elif run["total_tokens"] is not None:
                tokens = run["total_tokens"]
            elif run["stage_tokens"] is not None:
                tokens = run["stage_tokens"]
            if tokens:
                total_token_in += tokens[0]
                total_token_out += tokens[1]

        return total_time_seconds, total_token_in, total_token_out

    def _init_resume_data_collection(self):
        self._resume_time_seconds = 0.0
        self._resume_token_in = 0
        self._resume_token_out = 0
        self._resume_from_log = False

        log_path = self._get_log_file_path()
        if not log_path or not os.path.isfile(log_path):
            return
        if os.path.getsize(log_path) == 0:
            return
        resume_time, resume_in, resume_out = self._parse_resume_stats_from_log(log_path)
        if resume_time > 0 or resume_in > 0 or resume_out > 0:
            self._resume_time_seconds = resume_time
            self._resume_token_in = resume_in
            self._resume_token_out = resume_out
            self._resume_from_log = True

    def _context_upgrade_enabled(self) -> bool:
        return any([
            self.enable_rerank,
            self.enable_structured_summary,
            self.enable_hierarchical_summary,
            self.enable_compact_test_output,
            self.enable_long_term_memory,
            self.enable_long_term_memory_embed,
            self.enable_data_collection,
            self.enable_progress_controller,
            self.enable_stage_state_package,
            self.enable_observation_masking,
        ])

    def _register_file_mutation_event_callbacks(self):
        mutation_tool_names = {
            "EditTextFile",
            "EditFileDialogs",
            "ReplaceStringInFile",
            "CopyFile",
            "MoveFile",
            "DeleteFile",
            "CreateDirectory",
        }
        for tool in getattr(self, "tool_list_file", []):
            tool_name = getattr(tool, "name", tool.__class__.__name__)
            if tool_name not in mutation_tool_names or not hasattr(tool, "append_callback"):
                continue

            def _callback(success, path, detail, _tool=tool):
                self._record_file_mutation_event(_tool, success, path, detail)

            tool.append_callback(_callback)
            if (
                self.progress_controller is not None
                and self.progress_controller.hard_enforce_mutation_budget
                and hasattr(tool, "append_call_guard")
            ):
                def _guard(tool_input, _tool=tool):
                    return self._guard_file_mutation_call(_tool, tool_input)

                tool.append_call_guard(_guard)

    @staticmethod
    def _mutation_path_from_input(tool_input) -> str:
        value = tool_input
        if isinstance(value, dict) and isinstance(value.get("args"), dict):
            value = value["args"]
        if not isinstance(value, dict):
            return ""
        for key in ("path", "dest_path", "source_path", "directory"):
            if value.get(key):
                return str(value[key])
        return ""

    def _guard_file_mutation_call(self, tool, tool_input):
        controller = self.progress_controller
        if controller is None:
            return None
        stage_index = getattr(getattr(self, "stage_manager", None), "stage_index", None)
        if not isinstance(stage_index, int):
            return None
        tool_name = getattr(tool, "name", tool.__class__.__name__)
        path = self._mutation_path_from_input(tool_input)
        stage_contract = self._protected_stage_mutation_contract(
            stage_index,
            tool_name,
            path,
            tool_input,
        )
        if stage_contract:
            detail = {
                "tool": tool_name,
                "path": path,
                **stage_contract,
            }
            self.record_structured_event("mutation_contract_blocked", detail)
            return (
                "[MUTATION_CONTRACT_BLOCKED]\n"
                f"tool: {tool_name}\npath: {path}\n"
                f"reason: {stage_contract['reason']}\n"
                f"required_action: {stage_contract['required_action']}"
            )
        protected_output = self._protected_required_output(stage_index, tool_name, path)
        if protected_output:
            detail = {
                "tool": tool_name,
                "path": path,
                "required_output": protected_output,
                "reason": "required_stage_output_delete",
            }
            self.record_structured_event("mutation_contract_blocked", detail)
            return (
                "[MUTATION_CONTRACT_BLOCKED]\n"
                f"tool: {tool_name}\npath: {path}\n"
                f"reason: '{protected_output}' is a required output of the current stage.\n"
                "Do not delete and recreate this file. Read it, then use replace/edit mode for the smallest schema correction."
            )
        allowed, budget = controller.authorize_mutation(
            stage_index,
            tool=tool_name,
            path=path,
        )
        if allowed:
            return None
        self.record_structured_event("mutation_budget_blocked", {
            "tool": tool_name,
            "path": path,
            "budget": budget,
        })
        return (
            "[MUTATION_BUDGET_BLOCKED]\n"
            f"tool: {tool_name}\npath: {path}\n"
            f"policy: {budget.get('policy')}\n"
            f"limit: {budget.get('limit')}\nremaining: {budget.get('remaining')}\n"
            "Run the required targeted test or Check/Complete to obtain a new verifier observation before editing again."
        )

    def _protected_stage_mutation_contract(
        self,
        stage_index: int,
        tool_name: str,
        path: str,
        tool_input,
    ) -> dict[str, str] | None:
        """Reject destructive edits that violate a stage's monotonic contract."""
        try:
            stage_name = str(self.stage_manager.stages[stage_index].name or "")
        except (AttributeError, IndexError, TypeError):
            return None
        normalized = str(path or "").replace("\\", "/").lstrip("./")
        args = tool_input.get("args", tool_input) if isinstance(tool_input, dict) else {}
        args = args if isinstance(args, dict) else {}

        dut_name = str(getattr(self, "dut_name", "") or "").strip("/")
        dut_roots = tuple(
            root
            for root in (dut_name, f"{dut_name}_RTL" if dut_name else "")
            if root
        )
        mutates_dut = any(
            normalized == root or normalized.startswith(f"{root}/")
            for root in dut_roots
        )
        if mutates_dut:
            if stage_name == "test_case_implementation_in_batch":
                required_action = (
                    "The DUT defect must remain reproducible. Do not retry a DUT source edit and do not rerun "
                    "the identical test without a harness/test change. Keep the ordinary FAILED assertion, then "
                    "record one TC and one source location through RunSkillScript using exactly: "
                    '{"commands":[["unitytest/test-case-implementation-in-batch","recordbug.py",'
                    '"-BG \'BG-ROOT-CAUSE-95\' -TC \'TC-unity_test/tests/test_DUT_case.py::test_name\' '
                    '-BD \'observed failure\' -ROOT \'confirmed DUT root cause\' '
                    '-FILE \'DUT_RTL/DUT.v:10-14\' -FIX \'proposed corrected RTL source\'"]]}'
                )
            else:
                required_action = (
                    "Treat the DUT implementation as immutable. Fix only verification harness/test defects; "
                    "if the failure is a confirmed DUT defect, preserve the failing evidence for the designated "
                    "bug-analysis stage."
                )
            return {
                "reason": (
                    f"'{normalized}' is immutable DUT implementation evidence; verification must not repair "
                    "or rewrite the design under test."
                ),
                "required_action": required_action,
            }

        controller = getattr(self, "progress_controller", None)
        latest_decision = controller.latest(stage_index) if controller is not None else {}
        latest_decision = latest_decision or {}
        latest_policy = str(latest_decision.get("policy") or "")
        proposed_text = "\n".join(
            str(args.get(key) or "")
            for key in ("data", "new_string")
            if args.get(key) is not None
        )
        api_path = (
            dut_name
            and stage_name == "test_case_implementation_in_batch"
            and latest_policy == "classify_failure"
            and normalized.endswith(f"/tests/{dut_name}_api.py")
        )
        masks_dut_output = bool(
            re.search(r"&\s*0x[0-9a-fA-F]+", proposed_text)
            or re.search(
                r"(?:DUT\s*(?:Bug|defect)|DUT\s*(?:缺陷|错误)|由于\s*DUT)[\s\S]{0,240}?"
                r"(?:mask|掩码|截断|清除高位)",
                proposed_text,
                re.IGNORECASE,
            )
        )
        if api_path and masks_dut_output:
            return {
                "reason": (
                    "A valid ordinary FAILED assertion is being converted into an apparent pass by masking or "
                    "truncating DUT output in the verification API. This hides the defect and invalidates the run."
                ),
                "required_action": (
                    "Keep the API as a transparent DUT observation path, preserve the FAILED test, and record the "
                    "confirmed root cause with RunSkillScript/recordbug.py. Do not compensate for the DUT in tests, "
                    "fixtures, reference models, or API helpers."
                ),
            }

        if (
            stage_name == "test_case_implementation_in_batch"
            and os.path.basename(normalized) == "recordbug.py"
            and not normalized.startswith(".ucagent/skills/")
        ):
            return {
                "reason": "Stage 23 cannot create or edit a workspace-local recordbug.py helper.",
                "required_action": (
                    "Use the stage-provided recordbug.py through RunSkillScript. "
                    "Do not recreate the helper in unity_test or execute it as a pytest target."
                ),
            }

        if stage_name == "test_case_implementation_in_batch" and normalized.endswith("_bug_analysis.md"):
            return {
                "reason": "Stage 23 bug evidence must be written through the schema-aware recordbug.py skill.",
                "required_action": (
                    "Use RunSkillScript with recordbug.py to add FAILED evidence, or prunebug.py to remove a "
                    "Checker-identified PASSED TC and its empty ancestors. "
                    "Do not edit, replace, or delete the bug-analysis document directly."
                ),
            }

        if stage_name == "create_test_case_templates":
            basename = os.path.basename(normalized)
            if basename == "conftest.py":
                return {
                    "reason": "Stage 22 template failures must not be bypassed with a local conftest.py.",
                    "required_action": (
                        "Keep imports explicit with from {DUT}_api import * and repair every template source-contract "
                        "violation reported by Checker. Do not add or edit conftest.py."
                    ),
                }
            if "/tests/data" in f"/{normalized}" and tool_name == "DeleteFile":
                return {
                    "reason": "Stage 22 reporter data is generated evidence, not a repair target.",
                    "required_action": (
                        "Do not delete tests/data. Restore active mark_function calls in all listed placeholder "
                        "tests, then rerun Check/Complete so reporter data can be generated normally."
                    ),
                }

        api_test = (
            stage_name == "basic_api_functional_test"
            and "/tests/test_" in f"/{normalized}"
            and "_api" in os.path.basename(normalized)
            and normalized.endswith(".py")
        )
        if not api_test:
            return None
        invalid_mark_target = (
            re.search(r"\.mark_function\s*\([\s\S]{0,300}?\.__name__\b", proposed_text)
            or re.search(
                r"\.mark_function\s*\(\s*[^,\n]+,\s*['\"]test_[^'\"]+['\"]",
                proposed_text,
            )
        )
        if invalid_mark_target:
            return {
                "reason": "mark_function requires a callable test function, not a function-name string or .__name__.",
                "required_action": (
                    "Pass the test function object as mark_function's second argument, for example "
                    "mark_function(..., test_api_DUT_case, [...]); remove .__name__ and fix all listed functions in one batch."
                ),
            }
        if tool_name == "DeleteFile":
            return {
                "reason": "Stage 21 API tests are a monotonic collection and cannot be deleted.",
                "required_action": "Keep existing node ids and apply a function-local replacement.",
            }
        duplicate_suffix = re.search(
            r"(?:_fixed|_new|_copy|_backup|_temp|_v\d+)\.py$",
            os.path.basename(normalized),
            re.IGNORECASE,
        )
        creates_duplicate = (
            tool_name in {"CopyFile", "MoveFile"}
            or (
                tool_name in {"EditTextFile", "EditFileDialogs"}
                and str(args.get("mode") or "replace") == "write"
            )
        )
        if duplicate_suffix and creates_duplicate:
            return {
                "reason": "Stage 21 cannot create a renamed duplicate of the API test collection.",
                "required_action": (
                    "Edit the canonical test_{DUT}_api*.py file in place and preserve its pytest node ids; "
                    "do not create _fixed/_new/_backup variants."
                ),
            }
        if tool_name == "EditTextFile" and str(args.get("mode") or "replace") == "replace":
            start = int(args.get("start", 1) or 0)
            count = int(args.get("count", -1) or 0)
            if count == -1 and start <= 1:
                return {
                    "reason": "Stage 21 cannot replace an existing API test file from its first lines to EOF.",
                    "required_action": (
                        "Use ReplaceStringInFile or a bounded line replacement for only the failing function; "
                        "preserve every previously collected PASSED node id."
                    ),
                }
        return None

    def _protected_required_output(self, stage_index: int, tool_name: str, path: str) -> str:
        """Protect exact required outputs from destructive delete/recreate loops."""
        if tool_name != "DeleteFile" or not path:
            return ""
        try:
            stage = self.stage_manager.stages[stage_index]
            patterns = list(getattr(stage, "output_files", []) or [])
        except (AttributeError, IndexError, TypeError):
            return ""
        normalized_path = str(path).replace("\\", "/").lstrip("./")
        workspace = str(getattr(self, "workspace", "") or "")
        for raw_pattern in patterns:
            pattern = str(raw_pattern or "")
            if not pattern or glob.has_magic(pattern):
                continue
            if os.path.isabs(pattern) and workspace:
                try:
                    pattern = os.path.relpath(pattern, workspace)
                except ValueError:
                    continue
            normalized_pattern = pattern.replace("\\", "/").lstrip("./")
            if normalized_path == normalized_pattern:
                return str(raw_pattern)
        return ""

    def _current_stage_event_info(self) -> dict:
        stage_index = getattr(self.stage_manager, "stage_index", None)
        stage_name = ""
        stage_title = ""
        try:
            if isinstance(stage_index, int) and 0 <= stage_index < len(self.stage_manager.stages):
                stage = self.stage_manager.stages[stage_index]
                stage_name = getattr(stage, "name", "") or ""
                stage_title = stage.title() if hasattr(stage, "title") else stage_name
        except Exception:
            stage_name = ""
            stage_title = ""
        return {
            "stage_index": stage_index,
            "stage_name": stage_name,
            "stage_title": stage_title,
        }

    def _classify_mutation_path(self, path: str) -> str:
        rel = str(path or "").strip().lstrip("/").replace("\\", "/")
        low = rel.lower()
        base = os.path.basename(low)
        if not rel:
            return "unknown"
        if low.startswith("guide_doc/"):
            return "guide_doc"
        if low.endswith((".sv", ".svh", ".v", ".vh")):
            return "rtl"
        if low.startswith("tests/"):
            if low.endswith(".py"):
                return "test_code"
            if low.endswith(".ignore"):
                return "coverage_config"
            return "test_artifact"
        if base.endswith("_api.py") or (base.startswith("test_") and base.endswith(".py")):
            return "test_code"
        if base.endswith("_function_coverage_def.py"):
            return "coverage_code"
        if base.endswith("_bug_analysis.md"):
            return "bug_document"
        if base.endswith("_functions_and_checks.md"):
            return "spec_check_document"
        if base.endswith("_line_coverage_analysis.md"):
            return "line_coverage_document"
        if low.endswith((".yaml", ".yml", ".json", ".toml", ".ini", ".cfg")):
            return "config_or_metadata"
        if low.endswith(".md"):
            return "document"
        if low.endswith(".py"):
            return "code"
        return "other"

    def _sanitize_event_detail(self, detail):
        if isinstance(detail, dict):
            sanitized = {}
            for key, value in detail.items():
                if isinstance(value, dict):
                    sanitized[key] = self._sanitize_event_detail(value)
                elif isinstance(value, (list, tuple)):
                    sanitized[key] = [
                        self._sanitize_event_detail(item) if isinstance(item, dict) else str(item)[:500]
                        for item in value[:20]
                    ]
                elif isinstance(value, (str, int, float, bool)) or value is None:
                    sanitized[key] = value[:1000] if isinstance(value, str) else value
                else:
                    sanitized[key] = str(value)[:500]
            return sanitized
        if isinstance(detail, str):
            return {"message": detail[:1000]}
        return {"value": str(detail)[:1000]}

    def record_structured_event(self, event_type: str, payload: Optional[dict] = None, include_stage: bool = True):
        if not self.enable_data_collection and not self.enable_progress_controller:
            return None
        event = {
            "schema_version": 1,
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "time_unix": round(time.time(), 3),
            "thread_id": self.thread_id,
            "dut": self.dut_name,
            "round_index": self.invoke_round,
        }
        if include_stage:
            event.update(self._current_stage_event_info())
        if payload:
            event.update(self._sanitize_event_detail(payload))
        decision = self.progress_controller.observe(event) if self.progress_controller else None
        if self.enable_data_collection:
            self._write_structured_event(event)
            if decision:
                control_event = {
                    "schema_version": 1,
                    "event_type": "progress_control_decision",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "time_unix": round(time.time(), 3),
                    "thread_id": self.thread_id,
                    "dut": self.dut_name,
                    "round_index": self.invoke_round,
                    **self._current_stage_event_info(),
                    **decision,
                }
                self._write_structured_event(control_event)
        return decision

    def _write_structured_event(self, event: dict) -> None:
        encoded = json.dumps(event, ensure_ascii=False, sort_keys=True)
        info(f"[data_collection][structured_event] {encoded}")
        try:
            os.makedirs(os.path.dirname(self.structured_events_path), exist_ok=True)
            with open(self.structured_events_path, "a", encoding="utf-8") as f:
                f.write(encoded + "\n")
        except Exception as exc:
            warning(f"Failed to write structured event: {exc}")

    def _record_file_mutation_event(self, tool, success, path, detail):
        detail_data = self._sanitize_event_detail(detail)
        if not success and self.progress_controller is not None:
            stage_index = getattr(getattr(self, "stage_manager", None), "stage_index", None)
            if isinstance(stage_index, int):
                self.progress_controller.refund_mutation(stage_index)
        self.record_structured_event("file_mutation", {
            "tool": getattr(tool, "name", tool.__class__.__name__),
            "tool_call_count": getattr(tool, "call_count", None),
            "success": bool(success),
            "path": str(path or ""),
            "path_category": self._classify_mutation_path(str(path or "")),
            "operation": detail_data.get("op", "unknown") if isinstance(detail_data, dict) else "unknown",
            "detail": detail_data,
        })

    def _append_res_csv(self, elapsed_seconds: float, token_in: int, token_out: int, memory_metrics: dict = None, replace_last: bool = False):
        import csv
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        res_path = os.path.join(base_dir, "res.csv")
        header = [
            "Model", "DUT", "time", "token_in", "token_out",
            "stage_cache_hit_rate", "prefetch_hit_rate", "stage_first_turn_hit_rate",
            "retrieval_hit_rate", "recent_fallback_rate", "useful_hit_rate",
            "retrieval_useful_hit_rate", "prefetch_useful_hit_rate", "fallback_useful_hit_rate",
            "stale_hit_rate", "memory_pollution_rate", "prefetch_pollution_rate", "memory_token_increase",
            "token_roi_memory"
        ]
        model_label = "improved" if self._context_upgrade_enabled() else "origin"
        dut = self.dut_name
        time_str = self._format_duration_hm(elapsed_seconds)
        token_in_str = self._format_tokens_short(int(token_in)) if isinstance(token_in, (int, float)) else "N/A"
        token_out_str = self._format_tokens_short(int(token_out)) if isinstance(token_out, (int, float)) else "N/A"
        mm = memory_metrics or {}
        row = [
            model_label,
            dut,
            time_str,
            token_in_str,
            token_out_str,
            f"{float(mm.get('stage_cache_hit_rate', 0.0) or 0.0):.4f}",
            f"{float(mm.get('prefetch_hit_rate', 0.0) or 0.0):.4f}",
            f"{float(mm.get('stage_first_turn_hit_rate', 0.0) or 0.0):.4f}",
            f"{float(mm.get('retrieval_hit_rate', 0.0) or 0.0):.4f}",
            f"{float(mm.get('recent_fallback_rate', 0.0) or 0.0):.4f}",
            f"{float(mm.get('useful_hit_rate', 0.0) or 0.0):.4f}",
            f"{float(mm.get('retrieval_useful_hit_rate', 0.0) or 0.0):.4f}",
            f"{float(mm.get('prefetch_useful_hit_rate', 0.0) or 0.0):.4f}",
            f"{float(mm.get('fallback_useful_hit_rate', 0.0) or 0.0):.4f}",
            f"{float(mm.get('stale_hit_rate', 0.0) or 0.0):.4f}",
            f"{float(mm.get('memory_pollution_rate', 0.0) or 0.0):.4f}",
            f"{float(mm.get('prefetch_pollution_rate', 0.0) or 0.0):.4f}",
            self._format_tokens_short(int(mm.get('memory_token_increase', 0) or 0)),
            str(mm.get("token_roi_memory", 0.0)),
        ]
        if replace_last and os.path.exists(res_path):
            with open(res_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if lines:
                try:
                    existing_header = next(csv.reader([lines[0].strip()]))
                except Exception:
                    existing_header = []
                if existing_header != header:
                    lines[0] = ",".join(header) + "\n"
                last_idx = None
                for idx in range(len(lines) - 1, -1, -1):
                    if lines[idx].strip():
                        last_idx = idx
                        break
                if last_idx is not None and last_idx > 0:
                    last_row = next(csv.reader([lines[last_idx].strip()]))
                    if len(last_row) >= 2 and last_row[0] == model_label and last_row[1] == dut:
                        lines[last_idx] = ",".join(row) + "\n"
                        with open(res_path, "w", encoding="utf-8", newline="") as f:
                            f.writelines(lines)
                        return
        needs_header = not os.path.exists(res_path)
        with open(res_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if needs_header:
                writer.writerow(header)
            elif os.path.exists(res_path):
                with open(res_path, "r", encoding="utf-8", newline="") as rf:
                    reader = csv.reader(rf)
                    first_row = next(reader, [])
                if first_row != header:
                    with open(res_path, "r", encoding="utf-8") as rf:
                        old_lines = rf.readlines()
                    if old_lines:
                        old_lines[0] = ",".join(header) + "\n"
                        with open(res_path, "w", encoding="utf-8", newline="") as wf:
                            wf.writelines(old_lines)
                    with open(res_path, "a", encoding="utf-8", newline="") as af:
                        csv.writer(af).writerow(row)
                    return
            writer.writerow(row)

    def one_loop(self, msg=None):
        """Enhanced one loop with intelligent interaction logic based on configured mode"""
        # Use the configured interaction mode
        if self.interaction_mode == "advanced" and self.advanced_logic:
            try:
                return self.advanced_logic.advanced_one_loop(msg)
            except Exception as e:
                warning(
                    f"Advanced interaction logic failed, falling back to enhanced: {e}"
                )
                # Fall back to enhanced logic if available
                if self.enhanced_logic:
                    try:
                        return self.enhanced_logic.enhanced_one_loop(msg)
                    except Exception as e2:
                        warning(
                            f"Enhanced interaction logic also failed, using standard: {e2}"
                        )
                        # Fall back to standard logic
                        pass
        elif self.interaction_mode == "enhanced" and self.enhanced_logic:
            try:
                return self.enhanced_logic.enhanced_one_loop(msg)
            except Exception as e:
                warning(
                    f"Enhanced interaction logic failed, falling back to standard: {e}"
                )
                # Fall back to standard logic
                pass

        # Standard logic (fallback)
        if msg:
            self.set_continue_msg(msg)
        # one conversation round with retry on tool call error
        while True:
            tips = self.get_current_tips()
            if self.is_exit():
                return
            tool_calls_before = sum(int(getattr(tool, "call_count", 0) or 0) for tool in self.test_tools)
            stage_before = getattr(self.stage_manager, "stage_index", None)
            self.do_work(tips, self.get_work_config())
            tool_calls_after = sum(int(getattr(tool, "call_count", 0) or 0) for tool in self.test_tools)
            stage_after = getattr(self.stage_manager, "stage_index", None)
            if tool_calls_after == tool_calls_before and stage_after == stage_before:
                self._no_action_rounds += 1
                self.record_structured_event(
                    "no_action_round",
                    {
                        "streak": self._no_action_rounds,
                        "required_action": "execute_one_progress_relevant_tool",
                    },
                )
                self.set_continue_msg(
                    "上一轮没有执行任何工具，也没有推动当前阶段。不要重复分析或输出空回复。"
                    "请立即执行一个与当前 Checker 失败直接相关的工具动作：优先运行最小测试、"
                    "使用阶段要求的结构化 skill 记录证据，或调用 Check/Complete 获取最新失败集合。"
                    "一次只处理当前 batch；不得重读已经读取过的整份长文件。"
                )
            else:
                self._no_action_rounds = 0
            if not self._tool__call_error:
                break
            if self.is_break():
                return
        self.invoke_round += 1
        return self

    def custom_chat(self, msg):
        """Custom chat message to the agent."""
        self.do_work(
            {"messages": [self.backend.get_human_message(msg)]}, self.get_work_config()
        )

    def get_interaction_status(self):
        """Get the status of the interaction logic"""
        # Try advanced logic first
        if hasattr(self, "advanced_logic"):
            try:
                status = self.advanced_logic.get_interaction_status()
                status["logic_type"] = "advanced"
                return status
            except:
                pass

        # Fall back to enhanced logic
        if hasattr(self, "enhanced_logic"):
            try:
                status = self.enhanced_logic.get_interaction_status()
                status["logic_type"] = "enhanced"
                return status
            except:
                pass

        return {"status": "No enhanced logic available", "logic_type": "standard"}

    def set_interaction_phase(self, phase: str, sub_phase: str = "initial"):
        """Manually set the interaction phase"""
        # Try advanced logic first
        if hasattr(self, "advanced_logic"):
            try:
                self.advanced_logic.state.transition_to_phase(phase, sub_phase)
                info(f"Advanced interaction phase set to: {phase}.{sub_phase}")
                return
            except:
                pass

        # Fall back to enhanced logic
        if hasattr(self, "enhanced_logic"):
            try:
                self.enhanced_logic.state.transition_to_phase(phase)
                info(f"Enhanced interaction phase set to: {phase}")
                return
            except:
                pass

        warning("No enhanced logic available for phase setting")

    def force_reflection(self):
        """Force a reflection phase in the next loop"""
        # Try both logic systems
        success = False

        if hasattr(self, "advanced_logic"):
            try:
                self.advanced_logic.state.last_reflection_round = 0
                success = True
                info("Advanced logic: Reflection will be triggered in next loop")
            except:
                pass

        if hasattr(self, "enhanced_logic"):
            try:
                self.enhanced_logic.state.last_reflection_round = 0
                success = True
                info("Enhanced logic: Reflection will be triggered in next loop")
            except:
                pass

        if not success:
            warning("No enhanced logic available for reflection forcing")

    def use_advanced_logic(self, enable: bool = True):
        """Enable or disable advanced interaction logic for next loops"""
        self._use_advanced_logic = enable
        if enable:
            info("Advanced interaction logic will be used in subsequent loops")
        else:
            info("Advanced interaction logic disabled, will use enhanced logic")

    def get_performance_summary(self):
        """Get performance summary from advanced logic if available"""
        if hasattr(self, "advanced_logic"):
            try:
                return self.advanced_logic._get_performance_summary()
            except:
                pass
        return "Performance tracking not available"

    def do_work(self, instructions, config):
        """Perform the work using the agent."""
        self._is_work_busy = True
        self._tool__call_error = []
        try:
            if self.stream_output:
                self.do_work_stream(instructions, config)
            else:
                self.do_work_values(instructions, config)
        finally:
            self._is_work_busy = False
            if self._exit_on_completion_pending:
                self._queue_exit_on_completion()

    def is_work_busy(self):
        """Check if the agent is currently busy with work."""
        return self._is_work_busy

    def messages_get_raw(self):
        """Get the messages from the agent's state."""
        return self.backend.messages_get_raw()

    def messages_count(self):
        """Get the count of messages in the agent's state."""
        messages = self.messages_get_raw()
        return len(messages)

    def message_info(self):
        """Get information about the messages in the agent's state."""
        messages = self.messages_get_raw()
        return OrderedDict(
            {
                "count": len(messages),
                "size": sum([len(m.content) for m in messages]),
                "last_20type": ">".join([m.type for m in messages[-20:]]),
                "to_llm": self.backend.get_statistics(),
            }
        )

    def message_summary(self):
        """Summarize all the messages"""
        if self.message_manage_node is None:
            warning("No message management node available for summarization")
            return
        if not hasattr(self.message_manage_node, "force_summary"):
            warning(
                f"{self.message_manage_node.__class__.__name__} has not function 'force_summary'"
            )
            return
        self.message_manage_node.force_summary(self.messages_get_raw())

    def status_info(self):
        msg_info = self.message_info()
        msg_c, msg_s = msg_info.get("count", "-"), msg_info.get("size", "-")
        msg_stat = self.backend.get_statistics()
        stats = OrderedDict(
            {
                "UCAgent": self.__version__,
                "LLM": self.backend.model_name(),
                "Temperature": self.backend.temperature(),
                "IsBreak": self.is_break(),
                "Stream": self.stream_output,
                "Seed": self.seed,
                "SummaryMode": self.summary_mode(),
                "MessageCount": msg_c,
                "MessageSize": msg_s,
                "Interaction Mode": self.interaction_mode,
                "AI-Message": self.backend._stat_msg_count_ai,
                "Tool-Message": self.backend._stat_msg_count_tool,
                "Sys-Message": self.backend._stat_msg_count_system,
                "MsgIn(bytes)": msg_stat["message_in"],
                "MsgOut(bytes)": msg_stat["message_out"],
                "Start Time": fmt_time_stamp(self._time_start),
                "Run Time": fmt_time_deta(self.stage_manager.get_time_cost()),
                f"Token Reception({self.backend.token_total()})/TPS": self.backend.token_speed(),
            }
        )
        return stats

    def message_get_str(self, index, count):
        messages = self.messages_get_raw()
        if len(messages) == 0:
            warning(f"No messages found, cannot get message. Please try later.")
            return []
        index = index % len(messages)
        return [m.pretty_repr() for m in messages[index : index + count]]

    def do_work_values(self, instructions, config):
        return self.backend.do_work_values(instructions, config)

    def do_work_stream(self, instructions, config):
        return self.backend.do_work_stream(instructions, config)

    def get_tool_by_name(self, tool_name: str):
        """Get a tool by its name."""
        tool = next((tool for tool in self.test_tools if tool.name == tool_name), None)
        return tool

    def set_tool_call_time_out(self, time_out: int):
        """Set the tool call timeout in seconds."""
        if not isinstance(time_out, int) or time_out <= 0:
            raise ValueError("Tool call timeout must be a positive integer")
        for tool in self.test_tools:
            if hasattr(tool, "set_call_time_out"):
                tool.set_call_time_out(time_out)
            else:
                warning(f"Tool {tool.name} does not support setting call timeout")
        info(f"Tool call timeout set to {time_out} seconds")

    def set_one_tool_call_time_out(self, tool_name: str, time_out: int):
        """Set the tool call timeout for a specific tool in seconds."""
        if not isinstance(time_out, int) or time_out <= 0:
            raise ValueError("Tool call timeout must be a positive integer")
        tool = next((tool for tool in self.test_tools if tool.name == tool_name), None)
        if tool is None:
            raise ValueError(f"Tool {tool_name} not found")
        if hasattr(tool, "set_call_time_out"):
            tool.set_call_time_out(time_out)
            info(f"Tool {tool_name} call timeout set to {time_out} seconds")
        else:
            raise ValueError(f"Tool {tool_name} does not support setting call timeout")

    def list_tool_call_time_out(self):
        """List the tool call timeouts for all tools."""
        timeouts = OrderedDict()
        for tool in self.test_tools:
            if hasattr(tool, "get_call_time_out"):
                timeouts[tool.name] = tool.get_call_time_out()
            else:
                timeouts[tool.name] = None
        return timeouts

    def emulate_config(self):
        """Emulate the configuration process.
        Process:
        1. Echo the system prompt.
        2. Echo mission details.
        3. Echo current_tips
        4. Walk through all the stages.
            a. Echo the stage prompt.
            b. Call the 'Complete' tool.
        """
        echo_g("\nStart emulate config:")
        echo_g("="*80)
        echo_g("                First Tips (System Prompt)")
        echo_g(make_llm_tool_ret(self.get_current_tips()))
        echo_g("="*80)
        echo_g(f"               Mission Details (Total stages: {len(self.stage_manager.stages)})")
        # Force reset the stage index to 0
        self.stage_manager.stage_index = 0
        #echo_g(make_llm_tool_ret(self.stage_manager.detail()))
        echo_g("="*80)
        echo_g("                Config walkthrough")
        current_stage = self.stage_manager.get_current_stage()
        while current_stage is not None:
            echo_g(f"   Check Stage: {current_stage.title()}")
            echo_g("    - check stage task desc")
            self.get_current_tips()
            echo_g("    - check stage complete")
            self.stage_manager.complete(self.cfg.get_value("call_time_out", 300))
            current_stage = self.stage_manager.next_stage()
        echo_g("\n" + "="*80)
        echo_g("                Config walkthrough completed successfully!")
        echo_g("="*80)
