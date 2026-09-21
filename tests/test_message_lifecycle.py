import json

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from ucagent.abackend.langchain.message.conversation import (
    UCMessagesNode,
    _coerce_count,
    _summarize_run_testcases_yaml,
    summarize_messages,
    summarize_messages_structured,
)
from ucagent.abackend.langchain.message.statistic import MessageStatistic


class _UnusedModel:
    def invoke(self, _messages):
        raise AssertionError("summarization should not run in this test")


class _FailingSummaryModel:
    model_name = "failing-summary-model"

    def invoke(self, _messages):
        raise RuntimeError("summary backend unavailable")

    def with_structured_output(self, _schema):
        return self


def test_plain_summary_uses_local_fallback_when_all_models_fail():
    messages = [HumanMessage(content="stage 23 failed test_example.py::test_a")]

    result = summarize_messages(
        messages,
        128,
        _FailingSummaryModel(),
        fallback_model=_FailingSummaryModel(),
    )

    assert isinstance(result, AIMessage)
    assert result.content.strip()


def test_structured_summary_uses_local_json_when_all_models_fail():
    messages = [HumanMessage(content="stage 23 failed test_example.py::test_a")]

    result = summarize_messages_structured(
        messages,
        128,
        _FailingSummaryModel(),
        fallback_model=_FailingSummaryModel(),
    )
    payload = json.loads(result.content)

    assert set(payload) == {
        "stage_info",
        "test_report",
        "coverage_status",
        "bug_tracking",
        "next",
    }


def _node():
    return UCMessagesNode(
        msg_stat=MessageStatistic(),
        max_summary_tokens=128,
        max_keep_msgs=100,
        tail_keep_msgs=10,
        model=_UnusedModel(),
    )


def test_reset_chat_keeps_latest_human_turn_and_removes_older_messages():
    node = _node()
    messages = [
        SystemMessage(content="role", id="system"),
        HumanMessage(content="old request", id="human-old"),
        AIMessage(content="old answer", id="ai-old"),
        HumanMessage(content="new stage", id="human-new"),
    ]

    result = node.reset_chat()(dict(messages=messages))

    assert [message.id for message in result["messages"]] == ["human-old", "ai-old"]
    assert result["llm_input_messages"][-1].id == "human-new"


def test_missing_system_message_is_restored_for_llm_input():
    node = _node()
    node.set_system_message(SystemMessage(content="restored role", id="system"))

    result = node({"messages": [HumanMessage(content="request", id="human")]})

    assert result["llm_input_messages"][0].type == "system"
    assert result["llm_input_messages"][0].content == "restored role"
    assert result["llm_input_messages"][1].content == "request"


def test_token_limit_triggers_summary_before_message_limit(monkeypatch):
    node = UCMessagesNode(
        msg_stat=MessageStatistic(),
        max_summary_tokens=128,
        max_keep_msgs=100,
        tail_keep_msgs=2,
        model=_UnusedModel(),
        max_tokens=100,
    )
    captured = {}

    def fake_summary(messages, *_args, **_kwargs):
        captured["messages"] = list(messages)
        return AIMessage(content="compact summary", id="summary")

    monkeypatch.setattr(
        "ucagent.abackend.langchain.message.conversation.summarize_messages",
        fake_summary,
    )
    messages = [SystemMessage(content="role", id="system")]
    messages.extend(
        HumanMessage(content="large context " * 100, id=f"human-{index}")
        for index in range(5)
    )

    result = node({"messages": messages})

    assert len(captured["messages"]) == 3
    assert [message.id for message in result["llm_input_messages"][-2:]] == ["human-3", "human-4"]
    assert any(isinstance(message, RemoveMessage) for message in result["messages"])


def test_hard_limit_caps_composed_input_without_shrinking_normal_window():
    node = UCMessagesNode(
        msg_stat=MessageStatistic(),
        max_summary_tokens=128,
        max_keep_msgs=100,
        tail_keep_msgs=10,
        model=_UnusedModel(),
        max_tokens=6000,
        hard_max_tokens=7000,
    )
    messages = [SystemMessage(content="role", id="system")]
    messages.extend(
        HumanMessage(content=(f"turn {index} " * 500), id=f"human-{index}")
        for index in range(8)
    )

    bounded, was_trimmed = node._enforce_hard_input_budget(messages)

    assert was_trimmed is True
    assert count_tokens_approximately(bounded) <= 7000
    assert bounded[0].type == "system"
    assert bounded[-1].id == "human-7"


def test_structured_count_fields_accept_lists_without_crashing():
    assert _coerce_count(["test_a", "test_b"]) == 2
    assert _coerce_count({"test_a": "FAILED"}) == 1
    assert _coerce_count("3") == 3
    assert _coerce_count(None) == 0


def test_batch_summary_never_counts_error_cases_as_passed():
    summary = _summarize_run_testcases_yaml({
        "REPORT": {
            "run_test_success": False,
            "tests": {
                "total": 3,
                "fails": 0,
                "test_cases": {
                    "test_setup_a": "ERROR",
                    "test_setup_b": "ERROR",
                    "test_ok": "PASSED",
                },
            },
        },
    })

    assert summary["tests"] == {
        "total": 3,
        "passed": 1,
        "failed": 0,
        "invalid_or_other": 2,
        "status_counts": {"ERROR": 2, "PASSED": 1},
    }
    assert summary["failed_test_cases_top"] == ["test_setup_a", "test_setup_b"]


def test_hard_limit_retains_recent_tool_sequence_when_only_first_turn_is_human():
    node = UCMessagesNode(
        msg_stat=MessageStatistic(),
        max_summary_tokens=128,
        max_keep_msgs=100,
        tail_keep_msgs=10,
        model=_UnusedModel(),
        max_tokens=1500,
        hard_max_tokens=2000,
    )
    messages = [
        SystemMessage(content="role", id="system"),
        HumanMessage(content="initial stage request", id="human-initial"),
    ]
    for index in range(20):
        call_id = f"call-{index}"
        messages.append(AIMessage(
            content="",
            id=f"ai-{index}",
            tool_calls=[{"name": "ReadTextFile", "args": {"path": "x" * 500}, "id": call_id}],
        ))
        messages.append(ToolMessage(
            content=(f"observation {index} " * 100),
            tool_call_id=call_id,
            id=f"tool-{index}",
        ))

    bounded, was_trimmed = node._enforce_hard_input_budget(messages)

    assert was_trimmed is True
    assert count_tokens_approximately(bounded) <= 2000
    assert bounded[0].type == "system"
    assert bounded[1].type == "human"
    assert "CONTEXT_WINDOW_TRIMMED" in bounded[1].content
    assert bounded[-1].id == "tool-19"
    assert any(message.type == "ai" for message in bounded[2:])


def test_stage_state_package_masks_superseded_verifier_observations():
    class _RecordingAgent:
        def __init__(self):
            self.events = []

        def record_structured_event(self, event_type, payload):
            self.events.append((event_type, payload))

    recording_agent = _RecordingAgent()
    node = UCMessagesNode(
        msg_stat=MessageStatistic(),
        max_summary_tokens=128,
        max_keep_msgs=100,
        tail_keep_msgs=20,
        model=_UnusedModel(),
        enable_stage_state_package=True,
        stage_state_package_cfg={"max_chars": 6000},
        enable_observation_masking=True,
        observation_masking_cfg={
            "tool_names": ["RunTestCases", "Check", "Complete"],
            "keep_recent_verifier_observations": 1,
            "min_chars": 100,
        },
        vagent=recording_agent,
    )
    node.update_stage_state_package({
        "stage": {"index": 21, "title": "API test"},
        "unresolved": {"source": "checker", "checker_categories": ["missing_assert"]},
    })
    messages = [
        SystemMessage(content="role", id="system"),
        HumanMessage(content="repair the stage", id="human"),
        AIMessage(content="", id="ai-old", tool_calls=[{
            "name": "Check", "args": {}, "id": "call-old",
        }]),
        ToolMessage(
            content="old checker details " * 40,
            name="Check", tool_call_id="call-old", id="tool-old",
        ),
        AIMessage(content="", id="ai-new", tool_calls=[{
            "name": "RunTestCases", "args": {"targets": "x"}, "id": "call-new",
        }]),
        ToolMessage(
            content="latest targeted result " * 40,
            name="RunTestCases", tool_call_id="call-new", id="tool-new",
        ),
    ]

    llm_messages = node({"messages": messages})["llm_input_messages"]
    old_tool = next(message for message in llm_messages if getattr(message, "id", None) == "tool-old")
    new_tool = next(message for message in llm_messages if getattr(message, "id", None) == "tool-new")

    assert any("STAGE_STATE_PACKAGE" in str(message.content) for message in llm_messages)
    assert old_tool.tool_call_id == "call-old"
    assert "OBSERVATION_MASKED" in old_tool.content
    assert "latest targeted result" in new_tool.content
    assert node.observation_mask_statistics["masked_observations"] == 1
    assert recording_agent.events[0][0] == "observation_masking"


def test_latest_non_verifier_tool_observation_is_bounded_before_token_accounting():
    node = UCMessagesNode(
        msg_stat=MessageStatistic(),
        max_summary_tokens=128,
        max_keep_msgs=100,
        tail_keep_msgs=20,
        model=_UnusedModel(),
        enable_observation_masking=True,
        observation_masking_cfg={"max_tool_observation_chars": 4000},
    )
    messages = [
        SystemMessage(content="role", id="system"),
        HumanMessage(content="list rtl", id="human"),
        AIMessage(content="", id="ai", tool_calls=[{
            "name": "PathList", "args": {"directory": "Adder_RTL"}, "id": "call-path",
        }]),
        ToolMessage(
            content="head\n" + ("generated file entry\n" * 1000) + "tail diagnostic\n",
            name="PathList",
            tool_call_id="call-path",
            id="tool-path",
        ),
    ]

    llm_messages = node({"messages": messages})["llm_input_messages"]
    observation = next(message for message in llm_messages if getattr(message, "id", None) == "tool-path")

    assert len(observation.content) <= 4000
    assert "OVERSIZED_TOOL_OBSERVATION_TRUNCATED" in observation.content
    assert "head" in observation.content
    assert "tail diagnostic" in observation.content
