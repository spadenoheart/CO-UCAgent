#coding=utf-8

from ucagent.abackend.base import AgentBackendBase
from ucagent.util.log import info, warning, error
from .message import MessageStatistic, TokenSpeedCallbackHandler, LLMPerformanceCallbackHandler
from .message import LLMInputMessagesDumper, UCMessagesNode, SummarizationAndFixToolCall, State
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from ucagent.util.models import get_chat_model, get_chat_model_from_section
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from ucagent.util.functions import dump_as_json, get_ai_message_tool_call
from typing import List


class UCAgentLangChainBackend(AgentBackendBase):
    """
    LangChain-based agent backend implementation.
    """

    def __init__(self, vagent, config, **kwargs):
        super().__init__(vagent, config, **kwargs)
        self.message_statistic = MessageStatistic()
        self.cb_token_speed = TokenSpeedCallbackHandler()
        self.performance_callbacks = []
        self.performance_enabled = bool(getattr(vagent, "enable_data_collection", False))
        main_model_name = str(self.config.get_value("openai.model_name", "unknown"))
        stage_getter = lambda: getattr(getattr(vagent, "stage_manager", None), "stage_index", None)
        self.llm_input_dumper = LLMInputMessagesDumper(
            workspace=getattr(vagent, "workspace", "."),
            config=self.config.get_value("llm_input_dump", {}),
            stage_getter=stage_getter,
        )
        self.cb_llm_performance = None
        self.cb_llm_non_stream_performance = None
        self.cb_summary_performance = None
        main_callbacks = []
        if vagent.stream_output:
            main_callbacks.append(self.cb_token_speed)
        if self.performance_enabled:
            self.cb_llm_performance = LLMPerformanceCallbackHandler(
                role="main",
                model_name=main_model_name,
                streaming=True,
                stage_getter=stage_getter,
            )
            self.performance_callbacks.append(self.cb_llm_performance)
            main_callbacks.append(self.cb_llm_performance)
        self.model = get_chat_model(
            self.config,
            main_callbacks or None,
            streaming=True if (vagent.stream_output or self.performance_enabled) else False,
        )
        if vagent.stream_output or self.performance_enabled:
            non_stream_callbacks = None
            if self.performance_enabled:
                self.cb_llm_non_stream_performance = LLMPerformanceCallbackHandler(
                    role="main_fallback",
                    model_name=main_model_name,
                    streaming=False,
                    stage_getter=stage_getter,
                )
                self.performance_callbacks.append(self.cb_llm_non_stream_performance)
                non_stream_callbacks = [self.cb_llm_non_stream_performance]
            self.model_non_stream = get_chat_model(
                self.config,
                non_stream_callbacks,
                streaming=False,
            )
        else:
            self.model_non_stream = self.model
        self.summary_model_cfg = self._load_summary_model_cfg()
        self.summary_model_enabled = bool(self.summary_model_cfg.get("enabled", False))
        self.summary_model_fallback_to_main = bool(self.summary_model_cfg.get("fallback_to_main", True))
        self.summary_model_max_context_tokens = int(self.summary_model_cfg.get("max_context_tokens", 0) or 0)
        if self.performance_enabled:
            summary_model_name = (
                self.summary_model_cfg.get("openai", {}).get("model_name")
                if self.summary_model_enabled
                else main_model_name
            ) or "unknown"
            self.cb_summary_performance = LLMPerformanceCallbackHandler(
                role="summary",
                model_name=str(summary_model_name),
                streaming=False,
                stage_getter=stage_getter,
            )
            self.performance_callbacks.append(self.cb_summary_performance)
        self.sumary_model, self.summary_model_source = self._init_summary_model()
        self.checkpointer = MemorySaver()
        self.agent = None
        self.agent_non_stream = None
        self._stream_disabled_for_run = False
        self._stream_fallback_count = 0

        if vagent.context_management_strategy == "TrimAndSummaryMiddleware":
            message_manage_node = UCMessagesNode(
                msg_stat=self.message_statistic,
                max_summary_tokens=vagent.max_summary_tokens,
                max_keep_msgs=vagent.max_keep_msgs,
                tail_keep_msgs=vagent.cfg.get_value("conversation_summary.tail_keep_msgs", 20),
                model=self.sumary_model,
                max_tokens=vagent.max_token,
                hard_max_tokens=vagent.hard_max_token,
                structured_summary=getattr(vagent, "enable_structured_summary", False),
                hierarchical_summary=getattr(vagent, "enable_hierarchical_summary", False),
                enable_long_term_memory=getattr(vagent, "enable_long_term_memory", False),
                long_term_memory=getattr(vagent, "long_term_memory", None),
                long_term_memory_cfg=getattr(vagent, "long_term_memory_cfg", None),
                enable_failure_aware_context=getattr(vagent, "enable_failure_aware_context", False),
                failure_aware_context_cfg=getattr(vagent, "failure_aware_context_cfg", None),
                enable_stage_state_package=getattr(vagent, "enable_stage_state_package", False),
                stage_state_package_cfg=getattr(vagent, "stage_state_package_cfg", None),
                enable_observation_masking=getattr(vagent, "enable_observation_masking", False),
                observation_masking_cfg=getattr(vagent, "observation_masking_cfg", None),
                vagent=vagent,
                dut_name=getattr(vagent, "dut_name", ""),
                main_model=self.model_non_stream,
                summary_model_cfg=self.summary_model_cfg,
                summary_model_source=self.summary_model_source,
                llm_input_dumper=self.llm_input_dumper,
            )
        elif vagent.context_management_strategy == "SummarizationAndFixToolCall":
            info("Using SummarizationAndFixToolCall for conversation summarization (max_token={}, max_summary_tokens={})".format(vagent.max_token, vagent.max_summary_tokens))
            message_manage_node = SummarizationAndFixToolCall(
                token_counter=count_tokens_approximately,
                model=self.sumary_model,
                max_tokens=vagent.max_token,
                max_summary_tokens=vagent.max_summary_tokens,
                output_messages_key="llm_input_messages"
            ).set_max_keep_msgs(self.message_statistic, vagent.max_keep_msgs).set_llm_input_dumper(self.llm_input_dumper)
        else:
            raise ValueError(f"Unsupported context_management_strategy: {vagent.context_management_strategy}")
        message_manage_node.vagent = vagent
        self.message_manage_node = message_manage_node

    def _load_summary_model_cfg(self):
        cfg = self.config.get_value("summary_model", {})
        if hasattr(cfg, "as_dict"):
            return cfg.as_dict()
        if isinstance(cfg, dict):
            return dict(cfg)
        return {}

    def _init_summary_model(self):
        if not self.summary_model_enabled:
            if self.cb_summary_performance is not None:
                info("Summary model disabled; use a separately instrumented main model for summarization.")
                return get_chat_model(
                    self.config,
                    [self.cb_summary_performance],
                    streaming=False,
                ), "main"
            info("Summary model disabled; reuse main model for summarization.")
            return self.model_non_stream, "main"
        try:
            # Summary/compression requests should always be non-stream to keep the
            # local summary server protocol simple and avoid stream-only incompatibilities.
            summary_callbacks = [self.cb_summary_performance] if self.cb_summary_performance else None
            summary_model = get_chat_model_from_section(
                self.config,
                "summary_model",
                summary_callbacks,
                streaming=False,
            )
            try:
                summary_model_name = getattr(summary_model, "model_name", None) or getattr(summary_model, "model", None) or "unknown"
            except Exception:
                summary_model_name = "unknown"
            info(
                "Summary model enabled: source=summary_model, model={}, max_context_tokens={}, fallback_to_main={}, streaming=False".format(
                    summary_model_name,
                    self.summary_model_max_context_tokens or "unbounded",
                    self.summary_model_fallback_to_main,
                )
            )
            return summary_model, "summary_model"
        except Exception as exc:
            warning(f"Failed to initialize summary model from summary_model section: {exc}. Falling back to main model.")
            return self.model_non_stream, "main"

    def set_debug(self, debug):
        from langchain_core.globals import set_debug
        set_debug(debug)

    def init(self):
        self.agent = create_react_agent(
            model=self.model,
            tools=self.vagent.test_tools,
            checkpointer=self.checkpointer,
            pre_model_hook=self.message_manage_node,
            state_schema=State,
        )
        if self.vagent.stream_output:
            self.agent_non_stream = create_react_agent(
                model=self.model_non_stream,
                tools=self.vagent.test_tools,
                checkpointer=self.checkpointer,
                pre_model_hook=self.message_manage_node,
                state_schema=State,
            )

    def reset_chat(self, force=False):
        if hasattr(self.message_manage_node, "reset_chat"):
            return self.message_manage_node.reset_chat(force)
        return None

    def on_stage_complete(self, stage):
        self.reset_chat(force=False)

    def get_human_message(self, text):
        return HumanMessage(content=text)

    def get_system_message(self, text):
        msg = SystemMessage(content=text)
        if hasattr(self.message_manage_node, "set_system_message"):
            self.message_manage_node.set_system_message(msg)
        return msg

    def state_record_mesg(self, msg):
        if isinstance(msg, AIMessage):
            self._stat_msg_count_ai += 1
        elif isinstance(msg, ToolMessage):
            self._stat_msg_count_tool += 1
        elif isinstance(msg, SystemMessage):
            self._stat_msg_count_system += 1
        self.message_statistic.update_message(msg)

    def get_message_manage_node(self):
        return self.message_manage_node

    def _process_msg_content(self, msg):
        if isinstance(msg, str):
            return msg
        if isinstance(msg, dict):
            key = msg.get("type", "")
            if not key:
                return str(msg)
            v = msg.get(key)
            if not v:
                return str(msg)
            return self._process_msg_content(v)
        if isinstance(msg, list):
            str_text = ""
            for m in msg:
                str_text += self._process_msg_content(m)
            return str_text
        return str(msg)

    def do_work_stream(self, instructions, config):
        self._repair_unmatched_tool_calls(self.agent, config)
        if self._stream_disabled_for_run and self.agent_non_stream is not None:
            self._repair_unmatched_tool_calls(self.agent_non_stream, config)
            self._do_work_values_with_agent(self.agent_non_stream, instructions, config)
            return
        last_msg_index = None
        fist_ai_message = True
        try:
            for v, data in self.agent.stream(instructions, config, stream_mode=["values", "messages"]):
                if self.vagent.is_break():
                        break
                if v == "messages":
                    if fist_ai_message:
                        fist_ai_message = False
                        self.vagent.message_echo("\n\n================================== AI Message ==================================")
                    msg = data[0]
                    self.vagent.message_echo(self._process_msg_content(msg.content), end="")
                else:
                    index = len(data["messages"])
                    if index == last_msg_index:
                        continue
                    last_msg_index = index
                    msg = data["messages"][-1]
                    self.state_record_mesg(msg)
                    if isinstance(msg, AIMessage):
                        self.vagent.message_echo(get_ai_message_tool_call(msg))
                        self.check_tool_call_error(msg)
                        continue
                    self.vagent.message_echo("\n"+msg.pretty_repr())
        except ValueError as e:
            if "No generations found in stream" not in str(e) or self.agent_non_stream is None:
                raise
            self._stream_fallback_count += 1
            self._stream_disabled_for_run = True
            warning(
                "Streaming returned no generations; disabling streaming for the rest of this run "
                f"(fallback_count={self._stream_fallback_count}) and falling back to non-stream execution"
            )
            self._do_work_values_with_agent(self.agent_non_stream, instructions, config)

    def do_work_values(self, instructions, config):
        self._repair_unmatched_tool_calls(self.agent, config)
        self._do_work_values_with_agent(self.agent, instructions, config)

    def _repair_unmatched_tool_calls(self, agent, config):
        """Add synthetic ToolMessages for tool calls left open by an interrupted run."""
        try:
            values = agent.get_state(config).values
        except Exception as exc:
            warning(f"Failed to inspect agent state before run: {exc}")
            return
        messages = values.get("messages", []) if isinstance(values, dict) else []
        if not messages:
            return

        pending = {}
        for msg in messages:
            if isinstance(msg, AIMessage):
                for call in getattr(msg, "tool_calls", []) or []:
                    call_id = call.get("id")
                    if call_id:
                        pending[call_id] = call
            elif isinstance(msg, ToolMessage):
                pending.pop(getattr(msg, "tool_call_id", None), None)

        if not pending:
            return

        repair_messages: List[ToolMessage] = []
        for call_id, call in pending.items():
            repair_messages.append(ToolMessage(
                content=(
                    "The previous tool call was interrupted before its result was "
                    "recorded. Please inspect the current state and retry the needed "
                    "operation if it is still relevant."
                ),
                tool_call_id=call_id,
                name=call.get("name") or "unknown",
                status="error",
            ))
        agent.update_state(config, {"messages": repair_messages})
        warning(f"Repaired {len(repair_messages)} unmatched tool call(s) in agent state.")

    def _do_work_values_with_agent(self, agent, instructions, config):
        last_msg_index = None
        for _, step in agent.stream(instructions, config, stream_mode=["values"]):
            if self.vagent.is_break():
                break
            index = len(step["messages"])
            if index == last_msg_index:
                continue
            last_msg_index = index
            msg = step["messages"][-1]
            self.check_tool_call_error(msg)
            self.state_record_mesg(msg)
            self.vagent.message_echo(msg.pretty_repr())

    def check_tool_call_error(self, msg):
        if not isinstance(msg, AIMessage):
            return
        if not hasattr(msg, "invalid_tool_calls"):
            return
        if len(msg.invalid_tool_calls) < 1:
            return
        for call in msg.invalid_tool_calls:
            name = call.get("name")
            tool = next((tool for tool in self.vagent.test_tools if tool.name == name), None)
            args = call.get("args") or {}
            status = "success"
            try:
                assert tool is not None, f"Tool {name} not found"
                result = tool._run(*(), **args)
            except Exception as e:
               error(f"Error executing tool {call}: {e}")
               result = str(e)
               status = "error"
            if not isinstance(result, str):
                result = dump_as_json(result)
            self.vagent._tool__call_error.append(ToolMessage(
                content=result,
                tool_call_id=call["id"],
                name=name,
                status=status
            ))
        warning(f"Tool call error: {msg.invalid_tool_calls}, have re-called them in custom way")

    def messages_get_raw(self):
        try:
            values = self.agent.get_state(self.get_work_config()).values
            return values.get("messages", [])
        except Exception as e:
            warning(f"Failed to get messages from agent state: {e}")
        return []

    def get_work_config(self):
        work_config = {
            "configurable": {"thread_id": f"{self.vagent.thread_id}"},
            "recursion_limit": self.config.get_value("recursion_limit", 100000),
        }
        if self.vagent.langfuse_enable:
            work_config["callbacks"] = [self.vagent.langfuse_handler]
            work_config["metadata"] = {
                # "langfuse_user_id": "user_id",
                "langfuse_session_id": self.vagent.session_id.hex,
                # "langfuse_tags": ["some-tag",]
            }
        return work_config

    def model_name(self):
        # ChatOpenAI stores the name as `model_name`; ChatAnthropic uses `model`.
        return getattr(self.model, "model_name", None) or getattr(self.model, "model", "unknown")

    def temperature(self):
        return self.model.temperature

    def get_statistics(self):
        stats = self.message_statistic.get_statistics()
        if self.message_manage_node is not None and hasattr(self.message_manage_node, "get_summary_statistics"):
            stats["summary"] = self.message_manage_node.get_summary_statistics()
        if self.message_manage_node is not None and hasattr(self.message_manage_node, "get_memory_hit_statistics"):
            stats["memory_hit"] = self.message_manage_node.get_memory_hit_statistics()
        if self.performance_callbacks:
            stats["llm_performance"] = {
                callback.role: callback.get_statistics()
                for callback in self.performance_callbacks
            }
        return stats

    def token_speed(self):
        return self.cb_token_speed.get_speed()

    def token_total(self):
        return self.cb_token_speed.total()
