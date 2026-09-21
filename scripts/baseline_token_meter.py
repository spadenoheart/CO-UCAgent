#!/usr/bin/env python3
"""Non-invasive LangChain token telemetry for the upstream UCAgent baseline.

The experiment runner loads this module through a narrowly scoped
``sitecustomize`` hook. It adds one callback to each ``ChatOpenAI`` instance
without editing the baseline source tree, prompts, tool schemas, or requests.
Provider usage is preferred; otherwise the same LangChain approximate counter
used by CO-UCAgent's native telemetry is applied.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.messages.utils import count_tokens_approximately


TOKEN_LOG_ENV = "UCAGENT_BASELINE_TOKEN_LOG"
_INSTALL_LOCK = threading.Lock()
_WRITE_LOCK = threading.Lock()
_INSTALLED = False


def _usage_from_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    prompt = value.get("input_tokens", value.get("prompt_tokens"))
    completion = value.get("output_tokens", value.get("completion_tokens"))
    total = value.get("total_tokens")
    result: dict[str, int] = {}
    if isinstance(prompt, (int, float)):
        result["prompt_tokens"] = int(prompt)
    if isinstance(completion, (int, float)):
        result["completion_tokens"] = int(completion)
    if isinstance(total, (int, float)):
        result["total_tokens"] = int(total)
    return result


def _extract_usage(response: Any) -> dict[str, int]:
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
            metadata = getattr(message, "response_metadata", None)
            if isinstance(metadata, dict):
                for key in ("token_usage", "usage"):
                    usage = _usage_from_mapping(metadata.get(key))
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
            if not isinstance(tool_call, dict):
                continue
            for key in ("name", "args"):
                value = tool_call.get(key)
                if value:
                    parts.append(str(value))
    return "".join(parts)


def _append_record(record: dict[str, Any]) -> None:
    raw_path = os.environ.get(TOKEN_LOG_ENV, "")
    if not raw_path:
        return
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, ensure_ascii=True, sort_keys=True)
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()


class BaselineTokenCallback(BaseCallbackHandler):
    """Record one JSONL event for every completed or failed LLM request."""

    def __init__(self, model_instance: int):
        super().__init__()
        self.model_instance = model_instance
        self._lock = threading.Lock()
        self._active: dict[str, dict[str, Any]] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
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
                "model": str(
                    serialized.get("kwargs", {}).get("model_name")
                    or serialized.get("kwargs", {}).get("model")
                    or "unknown"
                ),
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
        ended = time.perf_counter()
        with self._lock:
            state = self._active.pop(str(run_id), None)
        if state is None:
            return
        usage = _extract_usage(response)
        prompt_tokens = usage.get("prompt_tokens", state["prompt_tokens_est"])
        completion_tokens = usage.get("completion_tokens")
        token_source = "provider" if "prompt_tokens" in usage else "approx"
        if completion_tokens is None:
            try:
                completion_tokens = int(
                    count_tokens_approximately(
                        [AIMessage(content="".join(state["chunks"]))]
                    )
                )
            except Exception:
                completion_tokens = 0
        if "completion_tokens" not in usage:
            token_source = "approx"
        first_token = state["first_token"]
        _append_record(
            {
                "schema_version": 1,
                "event": "llm_request",
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "pid": os.getpid(),
                "request_id": str(run_id),
                "model_instance": self.model_instance,
                "model": state["model"],
                "status": status,
                "latency_ms": round(max(0.0, ended - state["start"]) * 1000.0, 3),
                "ttft_ms": (
                    round(max(0.0, first_token - state["start"]) * 1000.0, 3)
                    if first_token is not None
                    else None
                ),
                "prompt_tokens": max(0, int(prompt_tokens)),
                "completion_tokens": max(0, int(completion_tokens)),
                "token_source": token_source,
                "error": error,
            }
        )


def install() -> None:
    """Patch ChatOpenAI construction once in the target Agent process."""
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        from langchain_openai import ChatOpenAI

        original_init = ChatOpenAI.__init__
        instance_counter = {"value": 0}

        def metered_init(self: Any, *args: Any, **kwargs: Any) -> None:
            callbacks = list(kwargs.get("callbacks") or [])
            instance_counter["value"] += 1
            callbacks.append(BaselineTokenCallback(instance_counter["value"]))
            kwargs["callbacks"] = callbacks
            original_init(self, *args, **kwargs)

        ChatOpenAI.__init__ = metered_init
        _INSTALLED = True


def summarize(path: Path) -> dict[str, Any]:
    totals = {
        "requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "provider_records": 0,
        "approx_records": 0,
        "invalid_records": 0,
    }
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                totals["invalid_records"] += 1
                continue
            if not isinstance(record, dict) or record.get("event") != "llm_request":
                continue
            totals["requests"] += 1
            totals["prompt_tokens"] += max(0, int(record.get("prompt_tokens", 0) or 0))
            totals["completion_tokens"] += max(0, int(record.get("completion_tokens", 0) or 0))
            source_key = "provider_records" if record.get("token_source") == "provider" else "approx_records"
            totals[source_key] += 1
    totals["total_tokens"] = totals["prompt_tokens"] + totals["completion_tokens"]
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a baseline token-meter JSONL file")
    parser.add_argument("token_log", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.token_log), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
