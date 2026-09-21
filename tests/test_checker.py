#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test cases for checker modules."""

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))
import pytest


from ucagent.checkers import *
from ucagent.util.functions import yam_str


def _require_fixture(relative_path):
    workspace = os.path.abspath(os.path.join(current_dir, relative_path))
    if not os.path.exists(workspace):
        pytest.skip(f"legacy checker fixture is not distributed: {relative_path}")
    return workspace


def _template_checker(ignore_ck_prefix="test_api_"):
    return UnityChipCheckerTestTemplate(
        "functions.md",
        "tests",
        "bugs.md",
        ignore_ck_prefix=ignore_ck_prefix,
    )


def test_template_checker_accepts_structured_expected_failures_after_exit_one():
    checker = _template_checker()
    report = {
        "run_test_success": False,
        "tests": {
            "total": 2,
            "fails": 1,
            "test_cases": {
                "tests/test_api_basic.py::test_api_basic": "PASSED",
                "tests/test_template.py::test_add": "FAILED",
            },
        },
    }

    assert checker._has_usable_expected_failure_report(
        report,
        "AssertionError: Not implemented\n1 failed, 1 passed",
        "Pytest process exited with code 1.",
    ) is True


def test_template_checker_rejects_no_tests_and_infrastructure_errors():
    checker = _template_checker()
    no_tests = {
        "run_test_success": False,
        "tests": {"total": 0, "fails": 0, "test_cases": {}},
    }
    collection_error = {
        "run_test_success": False,
        "tests": {
            "total": 1,
            "fails": 1,
            "test_cases": {"tests/test_template.py::test_add": "FAILED"},
        },
    }

    assert checker._has_usable_expected_failure_report(no_tests, "no tests ran", "") is False
    assert checker._has_usable_expected_failure_report(
        collection_error,
        "AssertionError: Not implemented",
        "ERROR collecting tests/test_template.py: ImportError",
    ) is False


def test_template_checker_rejects_error_or_pass_status_for_template_test():
    checker = _template_checker(ignore_ck_prefix="")
    for status in ("ERROR", "PASSED", "XPASS", "SKIPPED"):
        report = {
            "run_test_success": False,
            "tests": {
                "total": 1,
                "fails": 1,
                "test_cases": {"tests/test_template.py::test_add": status},
            },
        }
        assert checker._has_usable_expected_failure_report(
            report,
            "AssertionError: Not implemented",
            "Pytest process exited with code 1.",
        ) is False


def test_template_checker_contract_accepts_expected_exit_one(monkeypatch, tmp_path):
    checker = _template_checker(ignore_ck_prefix="")
    checker.workspace = str(tmp_path)
    report = {
        "run_test_success": False,
        "tests": {
            "total": 1,
            "fails": 1,
            "test_cases": {"tests/test_template.py::test_add": "FAILED"},
        },
        "all_check_point_list": ["FG-ADD/FC-ADD/CK-BASIC"],
        "unmarked_check_points": 0,
        "unmarked_check_point_list": [],
        "test_function_with_no_check_point_mark": 0,
        "test_function_with_no_check_point_mark_list": [],
        "failed_test_case_with_check_point_list": {},
    }
    monkeypatch.setattr(
        BaseUnityChipCheckerTestCase,
        "do_check",
        lambda self, timeout=0, **kw: (
            report,
            "AssertionError: Not implemented\n1 failed",
            "Pytest process exited with code 1.",
        ),
    )
    monkeypatch.setattr(
        "ucagent.checkers.unity_test.fc.get_unity_chip_doc_marks",
        lambda *args, **kwargs: ["FG-ADD/FC-ADD/CK-BASIC"],
    )

    passed, message = checker.do_check(is_complete=True)

    assert passed is True
    assert message["success"][0] == "Test template validation successful!"


def test_template_checker_excludes_prior_api_and_mock_files(monkeypatch, tmp_path):
    checker = _template_checker(ignore_ck_prefix="test_api_")
    checker.workspace = str(tmp_path)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_DUT_api.py").write_text(
        "def test_api(env):\n"
        "    env.dut.fc_cover['FG-API'].mark_function("
        "'FC-OPERATION', test_api, ['CK-DUT-INIT'])\n",
        encoding="utf-8",
    )
    report = {
        "run_test_success": False,
        "tests": {
            "total": 1,
            "fails": 1,
            "test_cases": {"tests/test_DUT_template.py::test_add": "FAILED"},
        },
        "all_check_point_list": [
            "FG-API/FC-OPERATION/CK-DUT-INIT",
            "FG-ADD/FC-ADD/CK-BASIC",
        ],
        "unmarked_check_points": 1,
        "unmarked_check_point_list": ["FG-API/FC-OPERATION/CK-DUT-INIT"],
        "test_function_with_no_check_point_mark": 0,
        "test_function_with_no_check_point_mark_list": [],
        "failed_test_case_with_check_point_list": {},
    }
    captured = {}

    def _run(_self, timeout=0, **kw):
        captured.update(kw)
        return report, "AssertionError: Not implemented\n1 failed", "Pytest process exited with code 1."

    monkeypatch.setattr(BaseUnityChipCheckerTestCase, "do_check", _run)
    monkeypatch.setattr(
        "ucagent.checkers.unity_test.fc.get_unity_chip_doc_marks",
        lambda *args, **kwargs: [
            "FG-API/FC-OPERATION/CK-DUT-INIT",
            "FG-ADD/FC-ADD/CK-BASIC",
        ],
    )

    passed, _message = checker.do_check(is_complete=True)

    assert passed is True
    assert "--ignore-glob=test_*_api*.py" in captured["pytest_args"]
    assert "--ignore-glob=test_*_mock*.py" in captured["pytest_args"]


def test_template_checker_extracts_only_literal_prior_stage_marks(tmp_path):
    checker = _template_checker()
    checker.workspace = str(tmp_path)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_DUT_api.py").write_text(
        "def test_api(env):\n"
        "    env.dut.fc_cover['FG-API'].mark_function("
        "'FC-OPERATION', test_api, ['CK-A', 'CK-B'])\n"
        "    dynamic = 'CK-C'\n"
        "    env.dut.fc_cover['FG-API'].mark_function("
        "'FC-OPERATION', test_api, [dynamic])\n",
        encoding="utf-8",
    )

    assert checker._prior_stage_check_points() == {
        "FG-API/FC-OPERATION/CK-A",
        "FG-API/FC-OPERATION/CK-B",
    }


def test_api_checker_reports_all_mark_function_name_strings_before_pytest(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_DUT_api.py"
    test_file.write_text(
        "class TestApi:\n"
        "    def test_one(self, env):\n"
        "        env.dut.fc_cover['FG-API'].mark_function("
        "'FC-API', self.test_one.__name__, ['CK-ONE'])\n"
        "    def test_two(self, env):\n"
        "        env.dut.fc_cover['FG-API'].mark_function("
        "'FC-API', self.test_two.__name__, ['CK-TWO'])\n",
        encoding="utf-8",
    )
    checker = UnityChipCheckerDutApiTest(
        "api_DUT_",
        "tests/DUT_api.py",
        "tests/test_DUT_api*.py",
        "functions.md",
        "bugs.md",
    ).set_workspace(str(tmp_path))

    passed, message = checker.do_check()

    assert passed is False
    assert "Found 2 calls" in message["error"][0]
    assert "self.test_one.__name__" in message["error"][1]
    assert "self.test_two.__name__" in message["error"][1]


def test_markdown_checker():
    workspace = _require_fixture("../examples/ALU")
    checker = UnityChipCheckerMarkdownFileFormat(
        markdown_file_list=["unity_test/ALU_functions_and_checks.md"]
    ).set_workspace(workspace)
    p, m = checker.do_check()
    print(p)
    print(yam_str(m))


def test_checker_functions_and_checks():
    """Test the UnityChipCheckerFunctionsAndChecks class."""
    workspace = _require_fixture("../examples/ALU")
    checker = UnityChipCheckerLabelStructure("unity_test/ALU_functions_and_checks.md", "FG").set_workspace(workspace)
    p, m = checker.do_check()
    print(p)
    print(yam_str(m))


def test_checker_dut_api():
    """Test the UnityChipCheckerDutApi class."""
    workspace = _require_fixture("../examples/ALU")
    checker = UnityChipCheckerDutApi("api_ALU", "unity_test/tests/ALU_api.py", 1).set_workspace(workspace)
    p, m = checker.do_check()
    print(p)
    print(yam_str(m))
    checker_creation = UnityChipCheckerDutCreation("unity_test/tests/ALU_api.py").set_workspace(workspace)
    p_creation, m_creation = checker_creation.do_check()
    print(p_creation)
    print(yam_str(m_creation))
    checker_fixture = UnityChipCheckerDutFixture("unity_test/tests/ALU_api.py").set_workspace(workspace)
    p_fixture, m_fixture = checker_fixture.do_check()
    print(p_fixture)
    print(yam_str(m_fixture))


def test_coverage():
    workspace = _require_fixture("../examples/ALU")
    checker = UnityChipCheckerCoverageGroup("unity_test/tests",
                                            "unity_test/tests/ALU_function_coverage_def.py",
                                            "unity_test/ALU_functions_and_checks.md", "FG").set_workspace(workspace)
    p, m = checker.do_check()
    print(p)
    print(yam_str(m))
    checker = UnityChipCheckerCoverageGroup("unity_test/tests",
                                            "unity_test/tests/ALU_function_coverage_def.py",
                                            "unity_test/ALU_functions_and_checks.md", "FC").set_workspace(workspace)
    p, m = checker.do_check()
    print(p)
    print(yam_str(m))
    checker = UnityChipCheckerCoverageGroup("unity_test/tests",
                                            "unity_test/tests/ALU_function_coverage_def.py",
                                            "unity_test/ALU_functions_and_checks.md", ["FC", "CK"]).set_workspace(workspace)
    p, m = checker.do_check()
    print(p)
    print(yam_str(m))


def test_checker_test_case():
    """Test the UnityChipCheckerTestCase class."""
    workspace = _require_fixture("../examples/ALU")

    for cls in [UnityChipCheckerTestTemplate,
                UnityChipCheckerTestFree,
                UnityChipCheckerTestCase]:
        print("cls:", cls.__name__)
        checker = cls("unity_test/ALU_functions_and_checks.md",
                      "unity_test/tests",
                      "unity_test/ALU_bug_analysis.md"
                      ).set_workspace(workspace)
        p, m = checker.do_check()
        print(p)
        print(yam_str(m))


def test_checker_api_test():
    workspace = _require_fixture("../examples/ALU")
    Checker = UnityChipCheckerDutApiTest(
        "api_ALU_", "unity_test/tests/ALU_api.py", "unity_test/tests/test_ALU_api*.py", "unity_test/ALU_functions_and_checks.md", "unity_test/ALU_bug_analysis.md"
    ).set_workspace(workspace)
    p, m = Checker.do_check()
    print(p)
    print(yam_str(m))


def test_checker_line_coverage():
    workspace = _require_fixture("../output")
    dut = "Adder"
    kw = {
        "min_line_coverage": 0.8,
        "dut_name": dut,
        "cfg": {
            "_temp_cfg.DUT": dut
        }
    }
    checker = UnityChipCheckerTestCaseWithLineCoverage(
        f"unity_test/{dut}_functions_and_checks.md",
        "unity_test/tests",
        f"unity_test/{dut}_bug_analysis.md",
        **kw
    ).set_workspace(workspace)
    p, m = checker.do_check()
    print(p)
    print(yam_str(m))


if __name__ == "__main__":
    #test_markdown_checker()
    #test_checker_functions_and_checks()
    #test_checker_dut_api()
    #test_coverage()
    #test_checker_test_case()
    #test_checker_api_test()
    test_checker_line_coverage()
