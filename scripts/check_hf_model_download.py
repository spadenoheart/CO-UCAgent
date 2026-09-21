#!/usr/bin/env python3
"""Check whether a local Hugging Face model snapshot is complete enough to use.

This script is intentionally offline. It uses the local safetensors index when
available, otherwise it infers the expected shard list from existing shard
names such as ``model.safetensors-00014-of-00039.safetensors``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TEMP_SUFFIXES = (
    ".incomplete",
    ".lock",
    ".tmp",
    ".part",
    ".partial",
)


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def load_expected_from_index(model_dir: Path) -> set[str]:
    expected: set[str] = set()
    for index_path in sorted(model_dir.glob("*.safetensors.index.json")):
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[warn] cannot read {index_path}: {exc}")
            continue
        weight_map = data.get("weight_map")
        if isinstance(weight_map, dict):
            expected.update(str(name) for name in weight_map.values() if str(name).endswith(".safetensors"))
    return expected


def infer_expected_from_shards(model_dir: Path) -> set[str]:
    """Infer full shard names from local files.

    Supports both common forms:
    - model-00001-of-00039.safetensors
    - model.safetensors-00001-of-00039.safetensors
    """
    shard_re = re.compile(r"^(?P<prefix>.*?)(?P<num>\d+)-of-(?P<total>\d+)(?P<suffix>\.safetensors)$")
    groups: dict[tuple[str, str, int, int], set[int]] = {}
    for path in model_dir.glob("*.safetensors"):
        match = shard_re.match(path.name)
        if not match:
            continue
        width = len(match.group("num"))
        total_width = len(match.group("total"))
        total = int(match.group("total"))
        key = (match.group("prefix"), match.group("suffix"), width, total_width)
        groups.setdefault(key + (total,), set()).add(int(match.group("num")))

    if not groups:
        return set()

    # Prefer the largest shard group in case unrelated safetensors are present.
    (prefix, suffix, width, total_width, total), _seen = max(
        groups.items(), key=lambda item: (item[0][4], len(item[1]))
    )
    total_text = str(total).zfill(total_width)
    return {
        f"{prefix}{idx:0{width}d}-of-{total_text}{suffix}"
        for idx in range(1, total + 1)
    }


def find_temp_files(model_dir: Path) -> list[Path]:
    temp_files: list[Path] = []
    for path in model_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name.endswith(TEMP_SUFFIXES) or any(part.endswith(".lock") for part in path.parts):
            temp_files.append(path)
    return sorted(temp_files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="<DATA_ROOT>/gffiles/cache/hf_models/Qwen3.5-122B-A10B",
        help="Local directory passed to `hf download --local-dir`.",
    )
    parser.add_argument(
        "--repo-id",
        default="Qwen/Qwen3.5-122B-A10B",
        help="Repo id used only for printing retry commands.",
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable name used in printed hf download commands.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser()
    print(f"model_dir: {model_dir}")
    if not model_dir.exists():
        raise SystemExit(f"model_dir does not exist: {model_dir}")

    safetensors = sorted(model_dir.glob("*.safetensors"))
    total_bytes = sum(path.stat().st_size for path in safetensors)
    zero_sized = [path.name for path in safetensors if path.stat().st_size == 0]

    expected = load_expected_from_index(model_dir)
    expected_source = "safetensors index"
    if not expected:
        expected = infer_expected_from_shards(model_dir)
        expected_source = "filename inference"

    present_names = {path.name for path in safetensors}
    missing = sorted(expected - present_names)
    extra = sorted(present_names - expected) if expected else []
    temp_files = find_temp_files(model_dir)

    print(f"safetensors_present: {len(safetensors)}")
    print(f"safetensors_bytes: {human_size(total_bytes)}")
    print(f"expected_source: {expected_source if expected else 'unknown'}")
    print(f"expected_safetensors: {len(expected) if expected else 'unknown'}")
    print(f"missing_safetensors: {len(missing)}")
    print(f"zero_sized_safetensors: {len(zero_sized)}")
    print(f"temp_or_lock_files: {len(temp_files)}")

    if missing:
        print("\n[missing]")
        for name in missing:
            print(name)

    if zero_sized:
        print("\n[zero-sized]")
        for name in zero_sized:
            print(name)

    if extra:
        print("\n[extra safetensors not listed by expected set]")
        for name in extra:
            print(name)

    if temp_files:
        print("\n[temp/lock files]")
        for path in temp_files:
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            print(f"{path.relative_to(model_dir)}\t{human_size(size)}")

    retry_names = missing + zero_sized
    if retry_names:
        print("\n[single-file retry commands]")
        for name in retry_names:
            print(
                "hf download "
                f"{args.repo_id} {name} "
                f"--local-dir {model_dir} "
                "--max-workers 1 "
                f'--token "${args.token_env}"'
            )
    elif expected and len(safetensors) >= len(expected) and not zero_sized:
        print("\nstatus: all expected safetensors are present locally")
    else:
        print(
            "\nstatus: cannot determine the complete shard list. "
            "Let `hf download` continue once, then rerun this script."
        )


if __name__ == "__main__":
    main()
