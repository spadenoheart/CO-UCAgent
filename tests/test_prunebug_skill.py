import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "ucagent/lang/zh/skills/unitytest/test-case-implementation-in-batch/scripts/prunebug.py"
)
SPEC = importlib.util.spec_from_file_location("ucagent_prunebug", SCRIPT)
prunebug = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prunebug)


def test_scoped_prune_preserves_duplicate_tc_and_root_cause():
    tc = "TC-test_api.py::TestApi::test_zero"
    lines = [
        "# Bugs\n",
        "## 未测试通过检测点分析\n",
        "<FG-ARITHMETIC>\n",
        "#### Add <FC-ADD>\n",
        "- <CK-ZERO-POS> wrong <BG-WRONG-95>\n",
        f"  - <{tc}> wrong path\n",
        "<FG-SPECIAL>\n",
        "#### Zero <FC-ZERO>\n",
        "- <CK-ZERO-POS> correct <BG-CORRECT-95>\n",
        f"  - <{tc}> correct path\n",
        "## 缺陷根因分析\n",
        "### Zero bug\n",
        "**测试用例**: test_api.py::TestApi::test_zero\n",
    ]

    counts = prunebug.prune_evidence(
        lines, tc, "FG-ARITHMETIC/FC-ADD/CK-ZERO-POS"
    )

    text = "".join(lines)
    assert counts["tc"] == 1
    assert counts["checkpoint_blocks"] == 1
    assert "FG-ARITHMETIC" not in text
    assert "BG-WRONG" not in text
    assert "BG-CORRECT" in text
    assert text.count(f"<{tc}>") == 1


def test_scoped_prune_requires_tc_beneath_requested_checkpoint():
    lines = [
        "## 未测试通过检测点分析\n",
        "<FG-A>\n",
        "#### F <FC-A>\n",
        "- <CK-A> bug <BG-A-80>\n",
        "  - <TC-test_a.py::test_a> evidence\n",
        "## 缺陷根因分析\n",
    ]

    try:
        prunebug.prune_evidence(
            lines, "TC-test_b.py::test_b", "FG-A/FC-A/CK-A"
        )
    except ValueError as exc:
        assert "was not found under checkpoint" in str(exc)
    else:
        raise AssertionError("expected scoped prune to reject a mismatched TC")


def test_scoped_prune_is_idempotent_and_removes_empty_ancestors():
    tc = "TC-test_api.py::test_zero"
    lines = [
        "## 未测试通过检测点分析\n",
        "<FG-OLD>\n",
        "#### Empty <FC-OLD>\n",
        "<FG-VALID>\n",
        "#### Valid <FC-VALID>\n",
        "- <CK-VALID> bug <BG-VALID-90>\n",
        f"  - <{tc}> evidence\n",
        "## 缺陷根因分析\n",
    ]

    counts = prunebug.prune_evidence(
        lines, tc, "FG-OLD/FC-OLD/CK-REMOVED"
    )

    text = "".join(lines)
    assert counts["already_absent"] is True
    assert counts["fc"] == 1
    assert counts["fg"] == 1
    assert "FG-OLD" not in text
    assert "FG-VALID" in text
    assert f"<{tc}>" in text
