#!/usr/bin/env python3
"""Summarize UCAgent per-request LLM performance log records."""

import argparse
import csv
import re
from collections import defaultdict


LINE_RE = re.compile(r"\[data_collection\]\[llm_request\]\s+(.*)$")
FIELD_RE = re.compile(r"(\w+)=('(?:[^']|\\')*'|\S+)")
NUMERIC_FIELDS = {
    "latency_ms",
    "ttft_ms",
    "prompt_tokens",
    "completion_tokens",
    "prefill_tps_est",
    "decode_tps_est",
}


def parse_record(line):
    match = LINE_RE.search(line)
    if not match:
        return None
    record = {}
    for key, raw_value in FIELD_RE.findall(match.group(1)):
        value = raw_value.strip("'")
        if key in NUMERIC_FIELDS:
            record[key] = None if value == "NA" else float(value)
        else:
            record[key] = value
    return record


def percentile(values, fraction):
    ordered = sorted(value for value in values if value is not None)
    if not ordered:
        return None
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def metric(value):
    return "NA" if value is None else f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", help="Path to ucagent-log.log")
    parser.add_argument("--csv", dest="csv_path", help="Write request records to CSV")
    args = parser.parse_args()

    records = []
    with open(args.log_file, encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            record = parse_record(line)
            if record:
                records.append(record)

    if args.csv_path and records:
        fieldnames = sorted({key for record in records for key in record})
        with open(args.csv_path, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    grouped = defaultdict(list)
    for record in records:
        grouped[record.get("role", "unknown")].append(record)

    print(f"requests={len(records)}")
    for role, role_records in sorted(grouped.items()):
        successful = [record for record in role_records if record.get("status") == "success"]
        latencies = [record.get("latency_ms") for record in successful]
        ttfts = [record.get("ttft_ms") for record in successful]
        prefill = [record.get("prefill_tps_est") for record in successful]
        decode = [record.get("decode_tps_est") for record in successful]
        print(
            f"role={role} requests={len(role_records)} success={len(successful)} "
            f"errors={len(role_records) - len(successful)} "
            f"latency_ms_p50={metric(percentile(latencies, 0.50))} "
            f"latency_ms_p95={metric(percentile(latencies, 0.95))} "
            f"ttft_ms_p50={metric(percentile(ttfts, 0.50))} "
            f"ttft_ms_p95={metric(percentile(ttfts, 0.95))} "
            f"prefill_tps_est_p50={metric(percentile(prefill, 0.50))} "
            f"decode_tps_est_p50={metric(percentile(decode, 0.50))}"
        )


if __name__ == "__main__":
    main()
