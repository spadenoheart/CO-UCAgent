#!/usr/bin/env python3
"""Run extracted LLM latency cases against an Ollama-compatible API."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    return json.loads(body)


def metric_ratio(count: Any, duration_ns: Any) -> float | None:
    if not isinstance(count, (int, float)) or not isinstance(duration_ns, (int, float)):
        return None
    if duration_ns <= 0:
        return None
    return float(count) / (float(duration_ns) / 1_000_000_000.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases_dir", nargs="?", default="benchmark/llm_latency_cases")
    parser.add_argument("--url", default="http://127.0.0.1:11435/api/chat")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--num-predict", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cases_dir = Path(args.cases_dir)
    manifest = json.loads((cases_dir / "manifest.json").read_text(encoding="utf-8"))
    results = []
    for case in manifest["cases"]:
        case_id = case["case_id"]
        request_path = cases_dir / case_id / "ollama_chat_request.json"
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        if args.num_predict is not None:
            payload.setdefault("options", {})["num_predict"] = args.num_predict
        print(f"running {case_id} ...", flush=True)
        started = time.time()
        try:
            response = post_json(args.url, payload, args.timeout)
            wall_s = time.time() - started
            result = {
                "case_id": case_id,
                "ok": True,
                "wall_s": wall_s,
                "prompt_eval_count": response.get("prompt_eval_count"),
                "prompt_eval_duration_ns": response.get("prompt_eval_duration"),
                "eval_count": response.get("eval_count"),
                "eval_duration_ns": response.get("eval_duration"),
                "load_duration_ns": response.get("load_duration"),
                "total_duration_ns": response.get("total_duration"),
                "prompt_tps": metric_ratio(
                    response.get("prompt_eval_count"),
                    response.get("prompt_eval_duration"),
                ),
                "decode_tps": metric_ratio(
                    response.get("eval_count"),
                    response.get("eval_duration"),
                ),
                "source_metadata": case["request"],
            }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            result = {
                "case_id": case_id,
                "ok": False,
                "wall_s": time.time() - started,
                "error": repr(exc),
                "source_metadata": case["request"],
            }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    output_path = Path(args.out) if args.out else cases_dir / "run_results.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
