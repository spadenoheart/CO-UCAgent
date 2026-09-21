import importlib.util
from pathlib import Path

from ucagent.tools.skill import ArgsRunSkillScript, RunSkillScript


def _load_recordbug_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "ucagent/lang/zh/skills/unitytest/test-case-implementation-in-batch/scripts/recordbug.py"
    )
    spec = importlib.util.spec_from_file_location("recordbug_contract_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_recordbug_contract_accepts_one_complete_entry():
    args = [
        "-BG", "BG-SUM-WIDTH-95",
        "-TC", "TC-unity_test/tests/test_Adder_basic.py::test_add_zero_zero",
        "-BD", "sum width is incorrect",
        "-ROOT", "RTL width mismatch",
        "-FILE", "Adder_RTL/Adder.v:10-14",
        "-FIX", "corrected RTL source",
    ]

    assert RunSkillScript._validate_recordbug_args(args) is None


def test_recordbug_parser_preserves_verilog_apostrophe_in_fix_text():
    parsed, error = RunSkillScript._parse_recordbug_args(
        "-BG 'BG-INF-95' "
        "-TC 'TC-unity_test/tests/test_ALU754.py::test_op_with_inf' "
        "-BD 'positive infinity is decoded as NaN' "
        "-ROOT 'comparison against 32'h7F800000 uses the wrong mask' "
        "-FILE 'ALU754_RTL/ALU754.v:31-35' "
        "-FIX 'preserve 32'h7F800000 and correct the exponent mask'"
    )

    assert error is None
    assert parsed is not None
    assert parsed[parsed.index("-ROOT") + 1] == "comparison against 32'h7F800000 uses the wrong mask"
    assert parsed[parsed.index("-FIX") + 1] == "preserve 32'h7F800000 and correct the exponent mask"
    assert RunSkillScript._validate_recordbug_args(parsed) is None


def test_recordbug_contract_rejects_unsupported_flag_and_multi_source():
    unsupported = RunSkillScript._validate_recordbug_args([
        "-BG", "BG-SUM-WIDTH-95",
        "-TC", "TC-unity_test/tests/test_Adder_basic.py::test_add_zero_zero",
        "-BD", "bug",
        "-FILE", "Adder_RTL/Adder.v:10",
        "-FIX", "fix",
        "-CONFIRM", "yes",
    ])
    multi_source = RunSkillScript._validate_recordbug_args([
        "-BG", "BG-SUM-WIDTH-95",
        "-TC", "TC-unity_test/tests/test_Adder_basic.py::test_add_zero_zero",
        "-BD", "bug",
        "-FILE", "Adder_RTL/Adder.v:10,Adder_RTL/Adder.v:12",
        "-FIX", "fix",
    ])

    assert "-CONFIRM" in unsupported
    assert "exactly one relative source location" in multi_source


def test_recordbug_accepts_legacy_bug_evidence_section_title():
    recordbug = _load_recordbug_module()
    lines = [
        "# DUT Bug Analysis\n",
        "## 检测点与 Bug 对应关系\n",
        "<FG-API>\n",
        "## 缺陷根因分析\n",
    ]

    assert recordbug.locate_section(lines) == (1, 3)


def test_run_skill_script_normalizes_json_encoded_commands():
    parsed = ArgsRunSkillScript.model_validate({
        "commands": (
            '[["unitytest/test-case-implementation-in-batch",'
            '"prunebug.py","-TC \\\"TC-test.py::test_case\\\""]]'
        )
    })

    assert parsed.commands == [[
        "unitytest/test-case-implementation-in-batch",
        "prunebug.py",
        '-TC "TC-test.py::test_case"',
    ]]
