#!/usr/bin/env python3
"""Summarize row-format res.csv including memory-cache metrics."""

import csv
import re
import sys
from collections import defaultdict


DURATION_RE = re.compile(r"(?:(?P<h>\d+)h)?(?P<m>\d+)min$")
TOKEN_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)?)(?P<scale>[KkMm]?)$")


def parse_duration_minutes(value):
    if not value:
        return None
    m = DURATION_RE.match(value.strip())
    if not m:
        return None
    hours = int(m.group("h") or 0)
    minutes = int(m.group("m") or 0)
    return hours * 60 + minutes


def parse_token_short(value):
    if not value or value == "N/A":
        return None
    m = TOKEN_RE.match(value.strip())
    if not m:
        return None
    num = float(m.group("num"))
    scale = m.group("scale").lower()
    if scale == "k":
        num *= 1_000
    elif scale == "m":
        num *= 1_000_000
    return int(num)


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def avg(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def safe_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_res.py res.csv")
        raise SystemExit(1)
    rows = load_rows(sys.argv[1])
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("Model", ""), row.get("DUT", ""))].append(row)

    print("Model,DUT,runs,avg_time_min,avg_token_in,avg_token_out,avg_stage_cache_hit_rate,avg_prefetch_hit_rate,avg_stage_first_turn_hit_rate,avg_retrieval_hit_rate,avg_recent_fallback_rate,avg_useful_hit_rate,avg_retrieval_useful_hit_rate,avg_prefetch_useful_hit_rate,avg_fallback_useful_hit_rate,avg_stale_hit_rate,avg_memory_pollution_rate,avg_prefetch_pollution_rate")
    for (model, dut), items in sorted(grouped.items()):
        times = [parse_duration_minutes(x.get("time", "")) for x in items]
        token_in = [parse_token_short(x.get("token_in", "")) for x in items]
        token_out = [parse_token_short(x.get("token_out", "")) for x in items]
        stage_cache = [safe_float(x.get("stage_cache_hit_rate", 0)) for x in items]
        prefetch = [safe_float(x.get("prefetch_hit_rate", 0)) for x in items]
        first_turn = [safe_float(x.get("stage_first_turn_hit_rate", 0)) for x in items]
        retrieval = [safe_float(x.get("retrieval_hit_rate", 0)) for x in items]
        recent_fallback = [safe_float(x.get("recent_fallback_rate", 0)) for x in items]
        useful = [safe_float(x.get("useful_hit_rate", 0)) for x in items]
        retrieval_useful = [safe_float(x.get("retrieval_useful_hit_rate", 0)) for x in items]
        prefetch_useful = [safe_float(x.get("prefetch_useful_hit_rate", 0)) for x in items]
        fallback_useful = [safe_float(x.get("fallback_useful_hit_rate", 0)) for x in items]
        stale = [safe_float(x.get("stale_hit_rate", 0)) for x in items]
        pollution = [safe_float(x.get("memory_pollution_rate", 0)) for x in items]
        prefetch_pollution = [safe_float(x.get("prefetch_pollution_rate", 0)) for x in items]
        print(
            ",".join(
                [
                    model,
                    dut,
                    str(len(items)),
                    f"{(avg(times) or 0):.2f}",
                    str(int(avg(token_in) or 0)),
                    str(int(avg(token_out) or 0)),
                    f"{(avg(stage_cache) or 0):.4f}",
                    f"{(avg(prefetch) or 0):.4f}",
                    f"{(avg(first_turn) or 0):.4f}",
                    f"{(avg(retrieval) or 0):.4f}",
                    f"{(avg(recent_fallback) or 0):.4f}",
                    f"{(avg(useful) or 0):.4f}",
                    f"{(avg(retrieval_useful) or 0):.4f}",
                    f"{(avg(prefetch_useful) or 0):.4f}",
                    f"{(avg(fallback_useful) or 0):.4f}",
                    f"{(avg(stale) or 0):.4f}",
                    f"{(avg(pollution) or 0):.4f}",
                    f"{(avg(prefetch_pollution) or 0):.4f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
