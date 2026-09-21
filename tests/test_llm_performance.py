# -*- coding: utf-8 -*-

from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from ucagent.abackend.langchain.message.performance import LLMPerformanceCallbackHandler


def test_streaming_request_records_ttft_and_throughput(monkeypatch):
    timestamps = iter([10.0, 10.2, 11.0])
    monkeypatch.setattr(
        "ucagent.abackend.langchain.message.performance.time.perf_counter",
        lambda: next(timestamps),
    )
    callback = LLMPerformanceCallbackHandler(
        role="main",
        model_name="test-model",
        streaming=True,
        stage_getter=lambda: 7,
    )
    run_id = uuid4()

    callback.on_chat_model_start({}, [[HumanMessage(content="hello")]], run_id=run_id)
    callback.on_llm_new_token(
        "ok",
        run_id=run_id,
        chunk=SimpleNamespace(message=SimpleNamespace(content="ok", tool_call_chunks=[])),
    )
    response = LLMResult(
        generations=[[
            ChatGeneration(
                message=AIMessage(
                    content="ok",
                    usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                )
            )
        ]]
    )
    callback.on_llm_end(response, run_id=run_id)

    stats = callback.get_statistics()
    assert stats["requests"] == 1
    assert stats["success"] == 1
    assert stats["errors"] == 0
    assert stats["ttft_ms_p50"] == pytest.approx(200)
    assert stats["prefill_tps_est_p50"] == pytest.approx(50)
    assert stats["decode_tps_est_p50"] == pytest.approx(2.5)


def test_non_streaming_request_has_no_ttft(monkeypatch):
    timestamps = iter([20.0, 22.0])
    monkeypatch.setattr(
        "ucagent.abackend.langchain.message.performance.time.perf_counter",
        lambda: next(timestamps),
    )
    callback = LLMPerformanceCallbackHandler(
        role="summary",
        model_name="summary-model",
        streaming=False,
    )
    run_id = uuid4()

    callback.on_chat_model_start({}, [[HumanMessage(content="summarize")]], run_id=run_id)
    callback.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=AIMessage(content="summary"))]]),
        run_id=run_id,
    )

    stats = callback.get_statistics()
    assert stats["requests"] == 1
    assert stats["latency_ms_p50"] == 2000
    assert stats["ttft_ms_p50"] is None
    assert stats["prefill_tps_est_p50"] is None
    assert stats["decode_tps_est_p50"] is None


def test_failed_request_is_counted(monkeypatch):
    timestamps = iter([30.0, 30.5])
    monkeypatch.setattr(
        "ucagent.abackend.langchain.message.performance.time.perf_counter",
        lambda: next(timestamps),
    )
    callback = LLMPerformanceCallbackHandler(
        role="main",
        model_name="test-model",
        streaming=True,
    )
    run_id = uuid4()

    callback.on_chat_model_start({}, [[HumanMessage(content="hello")]], run_id=run_id)
    callback.on_llm_error(RuntimeError("connection failed"), run_id=run_id)

    stats = callback.get_statistics()
    assert stats["requests"] == 1
    assert stats["success"] == 0
    assert stats["errors"] == 1
