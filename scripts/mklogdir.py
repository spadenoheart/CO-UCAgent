#!/usr/bin/env python3
import argparse
import os
import re
from datetime import datetime

import yaml


def _load_enable_data_collection(config_path: str) -> bool:
    if not config_path:
        return False
    if not os.path.isfile(config_path):
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return False
    context_upgrade = data.get("context_upgrade", {}) if isinstance(data, dict) else {}
    value = None
    if isinstance(context_upgrade, dict):
        value = context_upgrade.get("enable_data_collection")
    return bool(value)


def _find_latest_log_dir(root: str, dut: str) -> str:
    if not os.path.isdir(root):
        return ""
    pattern = re.compile(rf"^{re.escape(dut)}\+(\d{{8}}_\d{{6}})")
    candidates = []
    for name in os.listdir(root):
        match = pattern.match(name)
        if not match:
            continue
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        try:
            ts = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        candidates.append((ts, path))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _log_finished(log_file: str) -> bool:
    if not os.path.isfile(log_file):
        return False
    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return False
    if "Verify Agent finished at:" in content:
        return True
    if "[data_collection] summary:" in content:
        return True
    if "Total time taken:" in content:
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a log subdirectory for a DUT run and print the path."
    )
    parser.add_argument("--dut", required=True, help="DUT name")
    parser.add_argument("--root", default="log", help="Log root directory")
    parser.add_argument("--config", default=None, help="Config file to decide resume behavior")
    args = parser.parse_args()

    enable_data_collection = _load_enable_data_collection(args.config)
    if enable_data_collection:
        latest_dir = _find_latest_log_dir(args.root, args.dut)
        if latest_dir:
            log_file = os.path.join(latest_dir, "ucagent-log.log")
            if not _log_finished(log_file):
                print(latest_dir)
                return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dirname = f"{args.dut}+{timestamp}"
    log_dir = os.path.join(args.root, log_dirname)
    os.makedirs(log_dir, exist_ok=True)
    print(log_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
