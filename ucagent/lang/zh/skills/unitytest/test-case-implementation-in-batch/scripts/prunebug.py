#!/usr/bin/env python3
"""Remove stale PASSED test evidence from a UCAgent bug document."""

from __future__ import annotations

import argparse
import os
import re


EVIDENCE_TITLE = "## 未测试通过检测点分析"
ROOT_CAUSE_TITLE = "## 缺陷根因分析"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-TC",
        required=True,
        help="Exact TC tag to remove, e.g. TC-unity_test/tests/test_DUT.py::test_case",
    )
    parser.add_argument(
        "-CK",
        default=None,
        help=(
            "Optional exact FG/FC/CK path. When supplied, remove the TC only "
            "from that checkpoint and preserve duplicate evidence under other checkpoints."
        ),
    )
    return parser.parse_args()


def section_bounds(lines, title):
    start = next((i for i, line in enumerate(lines) if line.strip() == title), -1)
    if start < 0:
        raise ValueError(f"Section not found: {title}")
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")),
        len(lines),
    )
    return start, end


def _prune_empty_blocks(lines, section_start, section_end, marker, child_marker, boundaries):
    index = section_end - 1
    removed = 0
    while index > section_start:
        if not lines[index].startswith(marker):
            index -= 1
            continue
        block_end = index + 1
        while block_end < section_end and not any(lines[block_end].startswith(item) for item in boundaries):
            block_end += 1
        if not any(line.startswith(child_marker) for line in lines[index + 1:block_end]):
            del lines[index:block_end]
            removed += 1
            section_end -= block_end - index
        index -= 1
    return section_end, removed


def _tag(line, prefix):
    match = re.search(rf"<({re.escape(prefix)}[^>]+)>", line)
    return match.group(1) if match else None


def _checkpoint_blocks(lines, start, end):
    current_fg = None
    current_fc = None
    blocks = []
    index = start + 1
    while index < end:
        fg = _tag(lines[index], "FG-")
        fc = _tag(lines[index], "FC-")
        ck = _tag(lines[index], "CK-")
        if fg:
            current_fg = fg
            current_fc = None
        if fc:
            current_fc = fc
        if ck and current_fg and current_fc:
            block_end = index + 1
            while block_end < end:
                if (
                    _tag(lines[block_end], "FG-")
                    or _tag(lines[block_end], "FC-")
                    or _tag(lines[block_end], "CK-")
                    or lines[block_end].startswith("## ")
                ):
                    break
                block_end += 1
            blocks.append((f"{current_fg}/{current_fc}/{ck}", index, block_end))
            index = block_end
            continue
        index += 1
    return blocks


def _prune_empty_hierarchy(lines):
    start, end = section_bounds(lines, EVIDENCE_TITLE)
    removed_fc = 0
    removed_fg = 0

    index = end - 1
    while index > start:
        if not _tag(lines[index], "FC-"):
            index -= 1
            continue
        block_end = index + 1
        while block_end < end and not (
            _tag(lines[block_end], "FC-")
            or _tag(lines[block_end], "FG-")
            or lines[block_end].startswith("## ")
        ):
            block_end += 1
        if not any(_tag(line, "CK-") for line in lines[index:block_end]):
            del lines[index:block_end]
            removed_fc += 1
            end -= block_end - index
        index -= 1

    index = end - 1
    while index > start:
        if not _tag(lines[index], "FG-"):
            index -= 1
            continue
        block_end = index + 1
        while block_end < end and not (
            _tag(lines[block_end], "FG-")
            or lines[block_end].startswith("## ")
        ):
            block_end += 1
        if not any(_tag(line, "FC-") for line in lines[index:block_end]):
            del lines[index:block_end]
            removed_fg += 1
            end -= block_end - index
        index -= 1
    return removed_fc, removed_fg


def _prune_scoped_evidence(lines, start, end, tc_tag, checkpoint_path):
    token = f"<{tc_tag}>"
    matches = [item for item in _checkpoint_blocks(lines, start, end) if item[0] == checkpoint_path]
    if not matches:
        removed_fc, removed_fg = _prune_empty_hierarchy(lines)
        return {
            "tc": 0,
            "checkpoint_blocks": 0,
            "bg": 0,
            "ck": 0,
            "fc": removed_fc,
            "fg": removed_fg,
            "already_absent": True,
        }
    removed_tc = 0
    removed_blocks = 0
    for _path, block_start, block_end in reversed(matches):
        block = lines[block_start:block_end]
        filtered = [line for line in block if token not in line]
        removed_here = len(block) - len(filtered)
        if not removed_here:
            continue
        removed_tc += removed_here
        if not any("<TC-" in line for line in filtered):
            del lines[block_start:block_end]
            removed_blocks += 1
        else:
            lines[block_start:block_end] = filtered
    if removed_tc == 0:
        raise ValueError(
            f"TC {tc_tag} was not found under checkpoint {checkpoint_path}"
        )
    removed_fc, removed_fg = _prune_empty_hierarchy(lines)
    return {
        "tc": removed_tc,
        "checkpoint_blocks": removed_blocks,
        "bg": 0,
        "ck": removed_blocks,
        "fc": removed_fc,
        "fg": removed_fg,
        "already_absent": False,
    }


def prune_evidence(lines, tc_tag, checkpoint_path=None):
    start, end = section_bounds(lines, EVIDENCE_TITLE)
    if checkpoint_path:
        return _prune_scoped_evidence(lines, start, end, tc_tag, checkpoint_path)
    token = f"<{tc_tag}>"
    before = len(lines)
    lines[start + 1:end] = [line for line in lines[start + 1:end] if token not in line]
    removed_tc = before - len(lines)
    end -= removed_tc

    # Remove now-empty ancestors from leaves upward while preserving unrelated evidence.
    end, removed_bg = _prune_empty_blocks(
        lines, start, end, "  - <BG-", "    - <TC-",
        ("  - <BG-", "- <CK-", "#### <FC-", "<FG-", "## "),
    )
    end, removed_ck = _prune_empty_blocks(
        lines, start, end, "- <CK-", "  - <BG-",
        ("- <CK-", "#### <FC-", "<FG-", "## "),
    )
    end, removed_fc = _prune_empty_blocks(
        lines, start, end, "#### <FC-", "- <CK-",
        ("#### <FC-", "<FG-", "## "),
    )
    end, removed_fg = _prune_empty_blocks(
        lines, start, end, "<FG-", "#### <FC-",
        ("<FG-", "## "),
    )
    return {
        "tc": removed_tc,
        "bg": removed_bg,
        "ck": removed_ck,
        "fc": removed_fc,
        "fg": removed_fg,
    }


def prune_root_cause(lines, tc_tag):
    try:
        start, end = section_bounds(lines, ROOT_CAUSE_TITLE)
    except ValueError:
        return 0
    payload = tc_tag.removeprefix("TC-")
    display = os.path.basename(payload.split("::", 1)[0]) + "::" + payload.split("::", 1)[1]
    removed = 0
    index = end - 1
    while index > start:
        if not lines[index].startswith("### "):
            index -= 1
            continue
        block_end = next(
            (i for i in range(index + 1, end) if lines[i].startswith(("### ", "## "))),
            end,
        )
        if any(line.strip() == f"**测试用例**: {display}" for line in lines[index:block_end]):
            del lines[index:block_end]
            removed += 1
            end -= block_end - index
        index -= 1
    return removed


def main():
    args = parse_args()
    if not args.TC.startswith("TC-"):
        raise ValueError("-TC must start with TC-")
    dut = os.environ.get("DUT")
    out = os.environ.get("OUT")
    if not dut or not out:
        raise ValueError("Missing DUT/OUT environment variables")
    target = os.path.join(os.getcwd(), out, f"{dut}_bug_analysis.md")
    with open(target, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    counts = prune_evidence(lines, args.TC, args.CK)
    token = f"<{args.TC}>"
    root_blocks = 0
    if not any(token in line for line in lines):
        root_blocks = prune_root_cause(lines, args.TC)
    if counts["tc"] == 0 and not counts.get("already_absent"):
        raise ValueError(f"TC not found in bug evidence: {args.TC}")
    with open(target, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    scope = f" under {args.CK}" if args.CK else ""
    print(f"Pruned {args.TC}{scope}: {counts}, root_cause_blocks={root_blocks} -> {target}")


if __name__ == "__main__":
    main()
