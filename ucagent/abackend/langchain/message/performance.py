# -*- coding: utf-8 -*-
"""Per-request LLM performance telemetry."""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.utils import count_tokens_approximately

from ucagent.util.log import info


def _percentile(values: Iterable[float], percentile: float) -> Optional[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _usage_from_mapping(value: Any) -> Dict[str, int]:
    if not isinstance(value, dict):
        return {}
    prompt = value.get("input_tokens", value.get("prompt_tokens"))
    completion = value.get("output_tokens", value.get("completion_tokens"))
    total = value.get("total_tokens")
    result = {}
    if isinstance(prompt, (int, float)):
        result["prompt_tokens"] = int(prompt)
    if isinstance(completion, (int, float)):
        result["completion_tokens"] = int(completion)
    if isinstance(total, (int, float)):
        result["total_tokens"] = int(total)
    return result


def _extract_usage(response: Any) -> Dict[str, int]:
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        for key in ("token_usage", "usage"):
            usage = _usage_from_mapping(llm_output.get(key))
            if usage:
                return usage

    for generation_group in getattr(response, "generations", []) or []:
        for generation in generation_group or []:
            message = getattr(generation, "message", None)
            if message is None:
                continue
            usage = _usage_from_mapping(getattr(message, "usage_metadata", None))
            if usage:
                return usage
            response_metadata = getattr(message, "response_metadata", None)
            if isinstance(response_metadata, dict):
                for key in ("token_usage", "usage"):
                    usage = _usage_from_mapping(response_metadata.get(key))
                    if usage:
                        return usage
    return {}


def _chunk_payload(token: str, chunk: Any) -> str:
    parts = [token] if token else []
    message = getattr(chunk, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if isinstance(content, str) and content and content not in parts:
            parts.append(content)
        for tool_call in getattr(message, "tool_call_chunks", []) or []:
            if isinstance(tool_call, dict):
                parts.extend(
                    str(tool_call.get(key) or "")
                    for key in ("name", "args")
                    if tool_call.get(key)
                )
    return "".join(parts)


class LLMPerformanceCallbackHandler(BaseCallbackHandler):
    """Measure client-observed latency and token throughput for each LLM call."""

    def __init__(
        self,
        role: str,
        model_name: str,
        streaming: bool,
        stage_getter: Optional[Callable[[], Any]] = None,
    ):
        super().__init__()
        self.role = role
        self.model_name = model_name
        self.streaming = streaming
        self.stage_getter = stage_getter
        self._lock = threading.Lock()
        self._active: Dict[str, Dict[str, Any]] = {}
        self._records = []

    def _stage(self) -> str:
        if not callable(self.stage_getter):
            return "NA"
        try:
            stage = self.stage_getter()
            return "NA" if stage is None else str(stage)
        except Exception:
            return "NA"

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        prompt_messages = []
        for batch in messages or []:
            prompt_messages.extend(batch if isinstance(batch, list) else [batch])
        try:
            prompt_tokens = int(count_tokens_approximately(prompt_messages))
        except Exception:
            prompt_tokens = 0
        with self._lock:
            self._active[str(run_id)] = {
                "start": time.perf_counter(),
                "first_token": None,
                "prompt_tokens_est": prompt_tokens,
                "chunks": [],
                "stage": self._stage(),
            }

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        chunk: Any = None,
        **kwargs: Any,
    ) -> None:
        payload = _chunk_payload(token, chunk)
        if not payload:
            return
        now = time.perf_counter()
        with self._lock:
            state = self._active.get(str(run_id))
            if state is None:
                return
            if state["first_token"] is None:
                state["first_token"] = now
            state["chunks"].append(payload)

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(run_id, response=response, status="success")

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(run_id, status="error", error=str(error))

    def _finish(
        self,
        run_id: UUID,
        response: Any = None,
        status: str = "success",
        error: str = "",
    ) -> None:
        end = time.perf_counter()
        with self._lock:
            state = self._active.pop(str(run_id), None)
        if state is None:
            return

        usage = _extract_usage(response)
        prompt_tokens = usage.get("prompt_tokens", state["prompt_tokens_est"])
        token_source = "provider" if "prompt_tokens" in usage else "approx"
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is None:
            try:
                completion_tokens = int(
                    count_tokens_approximately([AIMessage(content="".join(state["chunks"]))])
                )
            except Exception:
                completion_tokens = 0
        if "completion_tokens" not in usage:
            token_source = "approx"

        latency_s = max(0.0, end - state["start"])
        first_token = state["first_token"]
        ttft_s = max(0.0, first_token - state["start"]) if first_token is not None else None
        decode_s = max(0.0, end - first_token) if first_token is not None else None
        prefill_tps = prompt_tokens / ttft_s if ttft_s and prompt_tokens else None
        decode_tps = completion_tokens / decode_s if decode_s and completion_tokens else None

        record = {
            "status": status,
            "role": self.role,
            "model": self.model_name,
            "stage": state["stage"],
            "stream": self.streaming,
            "latency_ms": latency_s * 1000,
            "ttft_ms": ttft_s * 1000 if ttft_s is not None else None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "prefill_tps_est": prefill_tps,
            "decode_tps_est": decode_tps,
            "token_source": token_source,
        }
        with self._lock:
            self._records.append(record)

        def metric(value: Optional[float], digits: int = 2) -> str:
            return "NA" if value is None else f"{value:.{digits}f}"

        error_text = f" error={error!r}" if error else ""
        info(
            "[data_collection][llm_request] "
            f"role={self.role} model={self.model_name} stage={state['stage']} "
            f"status={status} stream={str(self.streaming).lower()} "
            f"latency_ms={record['latency_ms']:.1f} "
            f"ttft_ms={metric(record['ttft_ms'], 1)} "
            f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} "
            f"prefill_tps_est={metric(prefill_tps)} decode_tps_est={metric(decode_tps)} "
            f"token_source={token_source}{error_text}"
        )

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            records = list(self._records)
        successful = [record for record in records if record["status"] == "success"]
        ttfts = [record["ttft_ms"] for record in successful if record["ttft_ms"] is not None]
        decode_rates = [
            record["decode_tps_est"]
            for record in successful
            if record["decode_tps_est"] is not None
        ]
        prefill_rates = [
            record["prefill_tps_est"]
            for record in successful
            if record["prefill_tps_est"] is not None
        ]
        latencies = [record["latency_ms"] for record in successful]
        return {
            "requests": len(records),
            "success": len(successful),
            "errors": len(records) - len(successful),
            "latency_ms_p50": _percentile(latencies, 0.50),
            "latency_ms_p95": _percentile(latencies, 0.95),
            "ttft_ms_p50": _percentile(ttfts, 0.50),
            "ttft_ms_p95": _percentile(ttfts, 0.95),
            "prefill_tps_est_p50": _percentile(prefill_rates, 0.50),
            "decode_tps_est_p50": _percentile(decode_rates, 0.50),
        }
