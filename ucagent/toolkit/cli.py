"""Stable command dispatchers for the seven public CO-UCAgent tools."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping


TOOLS: dict[str, dict[str, tuple[str, str]]] = {
    "co-llm-profiler": {
        "analyze": ("scripts.analyze_llm_performance", "summarize request-level TTFT/TPS logs"),
        "extract": ("scripts.extract_llm_latency_cases", "extract replay cases from exact LLM inputs"),
        "replay": ("scripts.run_llm_latency_cases", "replay latency cases against an OpenAI-compatible endpoint"),
    },
    "co-context": {
        "events": ("scripts.analyze_structured_events", "summarize context and verifier events"),
        "runtime": ("scripts.analyze_ucagent_runtime", "analyze stage, token, and runtime behavior"),
    },
    "co-memory-cache": {
        "inspect": ("scripts.inspect_memory_cache", "inspect memory entries and cache metrics"),
    },
    "co-trace": {
        "build": ("scripts.build_trace_tree", "compile structured events into a trace tree"),
        "visualize": ("scripts.visualize_agent_trajectory", "render or serve interactive trajectories"),
    },
    "co-strategy": {
        "build": ("scripts.build_context_reuse_pack", "extract verifier-grounded repair episodes"),
        "curate": ("scripts.curate_context_reuse_pack", "deduplicate and quality-filter a strategy pack"),
        "analyze": ("scripts.analyze_context_reuse_decisions", "analyze retrieval and injection outcomes"),
        "gate": ("scripts.replay_context_reuse_gate", "run paired admission replay"),
    },
    "co-bench": {
        "run": ("scripts.run_multi_dut_experiments", "run resumable multi-DUT experiments"),
        "freeze": ("scripts.freeze_experiment_baseline", "freeze a reproducible experiment manifest"),
        "stage": ("scripts.run_adder_stage_benchmarks", "run checkpoint-based stage replay"),
    },
    "co-trajectory-data": {
        "build": ("scripts.build_ucagent_finetune_dataset", "build provenance-preserving trajectory data"),
        "export": ("scripts.prepare_llamafactory_ucagent_dataset", "export data for LLaMA-Factory"),
    },
}


def _print_help(tool: str, commands: Mapping[str, tuple[str, str]]) -> None:
    print(f"usage: {tool} <command> [args]\n")
    print("commands:")
    width = max(len(command) for command in commands)
    for command, (_, description) in commands.items():
        print(f"  {command:<{width}}  {description}")
    print(f"\nRun '{tool} <command> --help' for command-specific options.")


def _dispatch(tool: str) -> int:
    commands = TOOLS[tool]
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        _print_help(tool, commands)
        return 0

    command = sys.argv[1]
    target = commands.get(command)
    if target is None:
        available = ", ".join(commands)
        print(f"{tool}: unknown command '{command}' (choose from: {available})", file=sys.stderr)
        return 2

    module_name, _ = target
    module = importlib.import_module(module_name)
    sys.argv = [f"{tool} {command}", *sys.argv[2:]]
    result = module.main()
    return int(result) if isinstance(result, int) else 0


def llm_profiler() -> int:
    return _dispatch("co-llm-profiler")


def context() -> int:
    return _dispatch("co-context")


def memory_cache() -> int:
    return _dispatch("co-memory-cache")


def trace() -> int:
    return _dispatch("co-trace")


def strategy() -> int:
    return _dispatch("co-strategy")


def bench() -> int:
    return _dispatch("co-bench")


def trajectory_data() -> int:
    return _dispatch("co-trajectory-data")
