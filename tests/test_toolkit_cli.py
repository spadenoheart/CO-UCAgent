from __future__ import annotations

import json
import sys

from scripts.inspect_memory_cache import discover, summarize
from ucagent.toolkit import cli


def test_all_public_tool_commands_have_help(monkeypatch, capsys):
    for tool in cli.TOOLS:
        monkeypatch.setattr(sys, "argv", [tool, "--help"])
        assert cli._dispatch(tool) == 0
        output = capsys.readouterr().out
        assert f"usage: {tool}" in output
        assert "commands:" in output


def test_unknown_public_tool_subcommand_is_rejected(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["co-trace", "missing"])
    assert cli._dispatch("co-trace") == 2
    assert "unknown command" in capsys.readouterr().err


def test_memory_cache_discovery_and_summary(tmp_path):
    memory_dir = tmp_path / ".ucagent_memory" / "Adder"
    memory_dir.mkdir(parents=True)
    memory_path = memory_dir / "memory.jsonl"
    memory_path.write_text(
        "\n".join(
            [
                json.dumps({"memory_type": "episode", "stage_name": "api", "dut": "Adder"}),
                json.dumps({"memory_type": "semantic", "stage_name": "test", "dut": "Adder"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    metrics_path = memory_dir / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"queries": 3, "useful_hits": 2}) + "\n",
        encoding="utf-8",
    )

    paths = discover([str(tmp_path)])
    result = summarize(paths)

    assert paths == [memory_path, metrics_path]
    assert result["memory_entries"] == 2
    assert result["by_type"] == {"episode": 1, "semantic": 1}
    assert result["by_dut"] == {"Adder": 2}
    assert result["metric_totals"] == {"queries": 3, "useful_hits": 2}
