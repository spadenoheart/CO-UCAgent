#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for Stage 21/25 test-execution contracts."""

from types import SimpleNamespace

import pytest

import ucagent.checkers.unity_test as unity_test
import ucagent.checkers.unity_test_random as unity_test_random
from ucagent.checkers.unity_test import (
    UnityChipCheckerBatchTestsImplementation,
    UnityChipCheckerDutApiTest,
    UnityChipCheckerTestCase,
    UnityChipCheckerTestTemplate,
    _test_execution_contract,
)
from ucagent.util.test_result import (
    TEST_OUTCOME_CRASH,
    TEST_OUTCOME_FAILURE,
    TEST_OUTCOME_INFRASTRUCTURE_ERROR,
    TEST_OUTCOME_NO_TESTS,
    TEST_OUTCOME_TIMEOUT,
)
from ucagent.checkers.unity_test_random import RandomTestCasesChecker


class _FakeStageManager:
    def __init__(self):
        self.data = {}

    def get_data(self, key, default=None):
        return self.data.get(key, default)

    def set_data(self, key, value):
        self.data[key] = value


class _FakeRunTest:
    def __init__(self, report, stdout="", stderr=""):
        self.result = (report, stdout, stderr)
        self.pre_call_back = None

    def set_pre_call_back(self, callback):
        self.pre_call_back = callback

    def do(self, *args, **kwargs):
        return self.result


def _report(test_cases, *, run_success=None):
    test_cases = dict(test_cases)
    failed = [name for name, status in test_cases.items() if status == "FAILED"]
    if run_success is None:
        run_success = not failed
    return {
        "run_test_success": run_success,
        "tests": {
            "total": len(test_cases),
            "fails": len(failed),
            "test_cases": test_cases,
        },
        "all_check_point_list": ["FG-API/FC-ADD/CK-ADD"],
        "unmarked_check_points": 0,
        "unmarked_check_point_list": [],
        "failed_check_point_list": [],
        "failed_test_case_with_check_point_list": {
            name: ["FG-API/FC-ADD/CK-ADD"] for name in failed
        },
        "test_function_with_no_check_point_mark": 0,
        "test_function_with_no_check_point_mark_list": [],
    }


def test_contract_accepts_structured_assertion_failures_not_pytest_exit_code():
    report = _report({
        "tests/test_api.py:1-3::test_ok": "PASSED",
        "tests/test_api.py:5-8::test_dut_bug": "FAILED",
    })

    result = _test_execution_contract(
        report,
        stdout="plugins: pytest-timeout-2.4.0",
        stderr="Pytest process exited with code 1.",
    )

    assert result["accepted"] is True
    assert result["outcome"] == TEST_OUTCOME_FAILURE
    assert result["tests_failed"] == 1


def _template_checker_for_source(tmp_path):
    checker = object.__new__(UnityChipCheckerTestTemplate)
    checker.workspace = str(tmp_path)
    checker.test_dir = "tests"
    checker.ignore_tc_prefix = "test_api_DUT_mock_"
    return checker


def test_template_source_preflight_rejects_commented_mark_before_pytest(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_DUT_template.py").write_text(
        "from DUT_api import *\n\n"
        "def test_data_path(env):\n"
        "    # env.dut.fc_cover['FG-DATA'].mark_function('FC-PATH', test_data_path, ['CK-BASIC'])\n"
        "    # TODO: verify the data path\n"
        "    assert False, 'Not implemented'\n",
        encoding="utf-8",
    )
    checker = _template_checker_for_source(tmp_path)

    passed, message = checker.do_check()

    assert passed is False
    assert "Template Source Contract" in message["error"][0]
    assert message["template_source_contract_violations"][0]["test"] == "test_data_path"


def test_template_source_preflight_accepts_callable_mark_target(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_DUT_template.py").write_text(
        "from DUT_api import *\n\n"
        "def test_data_path(env):\n"
        "    env.dut.fc_cover['FG-DATA'].mark_function('FC-PATH', test_data_path, ['CK-BASIC'])\n"
        "    # TODO: verify the data path\n"
        "    assert False, 'Not implemented'\n",
        encoding="utf-8",
    )
    checker = _template_checker_for_source(tmp_path)

    assert checker._template_source_contract_violations() == []
    assert checker._template_source_files() == ["test_DUT_template.py"]


def test_template_source_selection_excludes_prior_passing_env_tests(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_DUT_env.py").write_text(
        "def test_env_is_ready(env):\n    assert True\n",
        encoding="utf-8",
    )
    (tests_dir / "test_DUT_template.py").write_text(
        "def test_data_path(env):\n"
        "    env.dut.fc_cover['FG-DATA'].mark_function('FC-PATH', test_data_path, ['CK-BASIC'])\n"
        "    assert False, 'Not implemented'\n",
        encoding="utf-8",
    )
    checker = _template_checker_for_source(tmp_path)

    assert checker._template_source_files() == ["test_DUT_template.py"]


@pytest.mark.parametrize(
    ("report", "stderr", "expected_outcome"),
    [
        (_report({}, run_success=False), "no tests collected", TEST_OUTCOME_NO_TESTS),
        (_report({}, run_success=False), "Test run timed out after 15 seconds", TEST_OUTCOME_TIMEOUT),
        (_report({}, run_success=False), "Segmentation fault (core dumped)", TEST_OUTCOME_CRASH),
        (_report({"tests/test_api.py:1-3::test_error": "ERROR"}, run_success=False), "", TEST_OUTCOME_INFRASTRUCTURE_ERROR),
        (_report({"tests/test_api.py:1-3::test_xfail": "XFAIL"}, run_success=True), "", TEST_OUTCOME_FAILURE),
    ],
)
def test_contract_rejects_invalid_execution_and_non_evidence_statuses(report, stderr, expected_outcome):
    result = _test_execution_contract(report, stderr=stderr)

    assert result["accepted"] is False
    assert result["outcome"] == expected_outcome
    assert result["error"]


def test_stage21_stability_ignores_line_changes_but_rejects_deletion_and_regression():
    manager = _FakeStageManager()
    checker = UnityChipCheckerDutApiTest(
        "api_DUT_",
        "tests/DUT_api.py",
        "tests/test_DUT_api*.py",
        "functions.md",
        "bugs.md",
    ).set_stage_manager(manager)

    first = _report({"tests/test_DUT_api.py:10-20::TestApi::test_api_DUT_add": "PASSED"})
    assert checker._check_passing_test_stability(first, update=True) == (True, None)
    assert checker.get_template_data()["LOCKED_API_TESTS_COUNT"] == 1

    moved = _report({"tests/test_DUT_api.py:30-40::TestApi::test_api_DUT_add": "PASSED"})
    assert checker._check_passing_test_stability(moved) == (True, None)

    missing_ok, missing_error = checker._check_passing_test_stability(_report({}))
    assert missing_ok is False
    assert "Missing or renamed" in "\n".join(missing_error)

    regressed = _report({"tests/test_DUT_api.py:30-40::TestApi::test_api_DUT_add": "FAILED"})
    regression_ok, regression_error = checker._check_passing_test_stability(regressed)
    assert regression_ok is False
    assert "now fail" in "\n".join(regression_error)


def test_stage21_source_preflight_rejects_literal_and_name_attribute_targets(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_DUT_api.py"
    test_file.write_text(
        "def test_api_DUT_a(env):\n"
        "    env.dut.fc_cover['FG-API'].mark_function('FC-A', 'test_api_DUT_a', ['CK-A'])\n"
        "    assert True\n\n"
        "def test_api_DUT_b(env):\n"
        "    env.dut.fc_cover['FG-API'].mark_function('FC-B', test_api_DUT_b.__name__, ['CK-B'])\n"
        "    assert True\n",
        encoding="utf-8",
    )
    checker = UnityChipCheckerDutApiTest(
        "api_DUT_", "tests/DUT_api.py", "tests/test_DUT_api*.py", "functions.md", "bugs.md"
    ).set_workspace(str(tmp_path))

    invalid = checker._invalid_mark_function_name_references(["tests/test_DUT_api.py"])

    assert len(invalid) == 2
    assert {item["expression"] for item in invalid} == {"'test_api_DUT_a'", "test_api_DUT_b.__name__"}


def test_stage21_accepts_documented_dut_failure_without_rewriting_collection(tmp_path, monkeypatch):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tmp_path / "functions.md").write_text("functions", encoding="utf-8")
    (tmp_path / "bugs.md").write_text("bugs", encoding="utf-8")
    (tests_dir / "DUT_api.py").write_text(
        "def api_DUT_add():\n    return None\n",
        encoding="utf-8",
    )
    (tests_dir / "test_DUT_api_basic.py").write_text(
        "def test_api_DUT_add_bug(env):\n    assert False\n",
        encoding="utf-8",
    )
    report = _report({
        "tests/test_DUT_api_basic.py:1-2::test_api_DUT_add_bug": "FAILED",
    })
    checker = UnityChipCheckerDutApiTest(
        "api_DUT_",
        "tests/DUT_api.py",
        "tests/test_DUT_api*.py",
        "functions.md",
        "bugs.md",
    ).set_workspace(str(tmp_path)).set_stage_manager(_FakeStageManager())
    checker.run_test = _FakeRunTest(report, stderr="Pytest process exited with code 1.")

    monkeypatch.setattr(unity_test, "check_report", lambda *args, **kwargs: (True, "ok", 1))
    monkeypatch.setattr(unity_test.fc, "check_has_assert_in_tc", lambda *args, **kwargs: (True, {}))

    passed, message = checker.do_check()

    assert passed is True
    assert "accepted only after" in message["success"]


def test_stage25_accepts_only_documented_dut_failures(tmp_path, monkeypatch):
    (tmp_path / "tests").mkdir()
    (tmp_path / "functions.md").write_text("functions", encoding="utf-8")
    (tmp_path / "bugs.md").write_text("bugs", encoding="utf-8")
    report = _report({
        "tests/test_full.py:1-3::test_ok": "PASSED",
        "tests/test_full.py:5-8::test_dut_bug": "FAILED",
    })
    checker = UnityChipCheckerTestCase(
        "functions.md",
        "tests",
        "bugs.md",
    ).set_workspace(str(tmp_path))
    checker.run_test = _FakeRunTest(report, stderr="Pytest process exited with code 1.")

    checked = SimpleNamespace(called=False)

    def _check_report(*args, **kwargs):
        checked.called = True
        return True, "ok", 1

    monkeypatch.setattr(unity_test, "check_report", _check_report)
    monkeypatch.setattr(unity_test.fc, "check_has_assert_in_tc", lambda *args, **kwargs: (True, {}))

    passed, message = checker.do_check()

    assert passed is True
    assert checked.called is True
    assert any("retained as DUT-bug evidence" in line for line in message)


def test_stage25_rejects_infrastructure_error_before_bug_document_check(tmp_path, monkeypatch):
    (tmp_path / "tests").mkdir()
    (tmp_path / "functions.md").write_text("functions", encoding="utf-8")
    checker = UnityChipCheckerTestCase(
        "functions.md",
        "tests",
        "bugs.md",
    ).set_workspace(str(tmp_path))
    checker.run_test = _FakeRunTest(
        {"run_test_success": False},
        stderr="ERROR collecting tests/test_full.py",
    )

    def _unexpected_check_report(*args, **kwargs):
        pytest.fail("bug document validation must not run after infrastructure failure")

    monkeypatch.setattr(unity_test, "check_report", _unexpected_check_report)

    passed, message = checker.do_check()

    assert passed is False
    assert "Test Execution Contract" in "\n".join(message["error"])


def _stage23_checker(tmp_path, report):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_batch.py").write_text(
        "def test_ok():\n    assert True\n\n"
        "def test_dut_bug():\n    assert False\n",
        encoding="utf-8",
    )
    (tmp_path / "functions.md").write_text("functions", encoding="utf-8")
    (tmp_path / "bugs.md").write_text("bugs", encoding="utf-8")
    checker = UnityChipCheckerBatchTestsImplementation(
        doc_func_check="functions.md",
        test_dir="tests",
        doc_bug_analysis="bugs.md",
        data_key="TEST_TEMPLATE_IMP_REPORT",
        pre_report_file="pre.json",
        batch_size=10,
    ).set_workspace(str(tmp_path)).set_stage_manager(_FakeStageManager())
    checker.total_test_cases = [
        ("tests/test_batch.py::test_ok", False),
        ("tests/test_batch.py::test_dut_bug", False),
    ]
    checker.current_test_cases = [item[0] for item in checker.total_test_cases]
    checker.run_test = _FakeRunTest(report, stderr="Pytest process exited with code 1.")
    return checker


def test_stage23_batch_accepts_documented_dut_failure(tmp_path, monkeypatch):
    report = _report({
        "tests/test_batch.py:1-2::test_ok": "PASSED",
        "tests/test_batch.py:4-5::test_dut_bug": "FAILED",
    })
    checker = _stage23_checker(tmp_path, report)

    monkeypatch.setattr(unity_test, "check_report", lambda *args, **kwargs: (True, "ok", 1))
    monkeypatch.setattr(unity_test.fc, "check_has_assert_in_tc", lambda *args, **kwargs: (True, {}))

    passed, message = checker.do_check()

    assert passed is True
    assert "accepted only after" in message["success"]
    assert all(completed for _, completed in checker.total_test_cases)


def test_stage23_batch_rejects_no_tests_before_bug_document_check(tmp_path, monkeypatch):
    checker = _stage23_checker(tmp_path, _report({}, run_success=False))

    def _unexpected_check_report(*args, **kwargs):
        pytest.fail("bug document validation must not run after invalid test execution")

    monkeypatch.setattr(unity_test, "check_report", _unexpected_check_report)

    passed, message = checker.do_check()

    assert passed is False
    assert "Test Execution Contract" in "\n".join(message["error"])
    assert "TEST_REPORT" in message


def _random_checker(tmp_path, report):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_random_add.py").write_text(
        "def test_random_add(env):\n    assert False\n",
        encoding="utf-8",
    )
    (tmp_path / "functions.md").write_text("functions", encoding="utf-8")
    (tmp_path / "bugs.md").write_text("bugs", encoding="utf-8")
    checker = RandomTestCasesChecker(
        target_test_file="tests/test_random_*.py",
        mini_file_count=1,
        min_test_count=1,
        must_func_code_snippet={},
        doc_func_check="functions.md",
        test_dir="tests",
        doc_bug_analysis="bugs.md",
    ).set_workspace(str(tmp_path))
    checker.run_test = _FakeRunTest(report, stderr="Pytest process exited with code 1.")
    return checker


def test_random_stage_accepts_documented_dut_failure(tmp_path, monkeypatch):
    checker = _random_checker(tmp_path, _report({
        "tests/test_random_add.py:1-2::test_random_add": "FAILED",
    }))
    monkeypatch.setattr(unity_test_random, "check_report", lambda *args, **kwargs: (True, "ok", 1))
    monkeypatch.setattr(unity_test_random.fc, "check_has_assert_in_tc", lambda *args, **kwargs: (True, {}))

    passed, message = checker.test_check()

    assert passed is True
    assert "accepted only after" in message


def test_random_stage_rejects_invalid_execution_before_bug_document_check(tmp_path, monkeypatch):
    checker = _random_checker(tmp_path, _report({}, run_success=False))

    def _unexpected_check_report(*args, **kwargs):
        pytest.fail("bug document validation must not run after invalid test execution")

    monkeypatch.setattr(unity_test_random, "check_report", _unexpected_check_report)

    passed, message = checker.test_check()

    assert passed is False
    assert "Test Execution Contract" in "\n".join(message["error"])
