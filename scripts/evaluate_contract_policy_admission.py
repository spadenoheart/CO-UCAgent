#!/usr/bin/env python3
"""Evaluate a candidate memory policy from matched checkpoint replays.

Admission is quality constrained. Efficiency cannot compensate for a lost
stage completion, a new regression, or an invalid verifier execution. Eligible
candidates additionally need a positive bootstrap lower confidence bound and a
bounded harmful-pair rate.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Iterable


def load_pairs(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = data.get("pairs", data)
    if isinstance(pairs, dict):
        return [item for item in pairs.values() if isinstance(item, dict)]
    if isinstance(pairs, list):
        return [item for item in pairs if isinstance(item, dict)]
    raise ValueError(f"paired result must contain an object or list of pairs: {path}")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def bootstrap_mean_interval(
    values: list[float],
    *,
    confidence: float = 0.95,
    samples: int = 10000,
    seed: int = 20260818,
) -> tuple[float, float] | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    estimates = sorted(
        mean(rng.choice(values) for _ in values)
        for _ in range(max(100, samples))
    )
    alpha = (1.0 - confidence) / 2.0
    lower_index = max(0, min(len(estimates) - 1, int(alpha * len(estimates))))
    upper_index = max(
        0,
        min(len(estimates) - 1, int((1.0 - alpha) * len(estimates)) - 1),
    )
    return estimates[lower_index], estimates[upper_index]


def evaluate_admission(
    pairs: Iterable[dict],
    *,
    policy_id: str,
    min_pairs: int = 9,
    min_utility_lcb: float = 0.0,
    max_harmful_rate: float = 0.10,
    bootstrap_samples: int = 10000,
    seed: int = 20260818,
) -> dict:
    rows = list(pairs)
    def comparison_valid(row: dict) -> bool:
        explicit = row.get("comparison_valid")
        if isinstance(explicit, bool):
            return explicit
        return row.get("control_status") == "completed" and row.get("treatment_status") == "completed"

    valid_rows = [row for row in rows if comparison_valid(row)]
    invalid_rows = [row for row in rows if not comparison_valid(row)]
    quality_violations = [
        row for row in valid_rows if row.get("quality_noninferior") is not True
    ] + [
        row
        for row in invalid_rows
        if row.get("control_status") == "completed"
        and row.get("treatment_status") != "completed"
    ]
    harmful = [
        row
        for row in valid_rows
        if int(row.get("regression_delta", 0) or 0) > 0
        or int(row.get("invalid_test_delta", 0) or 0) > 0
    ]
    utilities = [
        float(row.get("utility_score", 0.0) or 0.0)
        for row in valid_rows
        if row.get("quality_noninferior") is True
        and row.get("utility_score") is not None
    ]
    utility_interval = bootstrap_mean_interval(
        utilities,
        samples=bootstrap_samples,
        seed=seed,
    )
    harm_interval = wilson_interval(len(harmful), len(valid_rows))
    sample_ready = len(valid_rows) >= min_pairs
    utility_ready = utility_interval is not None and utility_interval[0] > min_utility_lcb
    harm_point_rate = len(harmful) / len(valid_rows) if valid_rows else None
    harm_bound_ready = harm_interval is not None and harm_interval[1] <= max_harmful_rate

    if quality_violations or (harm_point_rate is not None and harm_point_rate > max_harmful_rate):
        decision = "reject"
        reason = "quality_constraint_violated"
    elif sample_ready and utility_ready and harm_bound_ready:
        decision = "admit"
        reason = "positive_utility_lcb_and_bounded_harm"
    else:
        decision = "hold"
        reasons = []
        if not sample_ready:
            reasons.append("insufficient_pairs")
        if not utility_ready:
            reasons.append("utility_lcb_not_positive")
        if not harm_bound_ready:
            reasons.append("harm_upper_bound_too_wide")
        reason = ",".join(reasons) or "insufficient_evidence"

    return {
        "schema_version": 1,
        "analysis_type": "matched_checkpoint_policy_admission",
        "policy_id": policy_id,
        "decision": decision,
        "reason": reason,
        "causal_scope": "matched stage-checkpoint replay; not full-DUT or cross-DUT causality",
        "thresholds": {
            "min_pairs": min_pairs,
            "min_utility_lcb": min_utility_lcb,
            "max_harmful_rate_wilson_upper": max_harmful_rate,
        },
        "evidence": {
            "pairs": len(rows),
            "valid_pairs": len(valid_rows),
            "invalid_pairs": len(invalid_rows),
            "quality_violations": len(quality_violations),
            "harmful_pairs": len(harmful),
            "harmful_pair_rate": round(harm_point_rate, 6) if harm_point_rate is not None else None,
            "harmful_rate_wilson_95": (
                [round(harm_interval[0], 6), round(harm_interval[1], 6)]
                if harm_interval
                else None
            ),
            "mean_utility": round(mean(utilities), 6) if utilities else None,
            "utility_bootstrap_95": (
                [round(utility_interval[0], 6), round(utility_interval[1], 6)]
                if utility_interval
                else None
            ),
            "treatment_wins": sum(row.get("winner") == "treatment" for row in rows),
            "control_wins": sum(row.get("winner") == "control" for row in rows),
            "ties": sum(row.get("winner") == "tie" for row in rows),
            "treatment_completion_only": sum(
                row.get("treatment_status") == "completed"
                and row.get("control_status") != "completed"
                for row in rows
            ),
        },
        "failed_pair_ids": [str(row.get("pair_id") or "") for row in quality_violations],
    }


def update_registry(path: Path, admission: dict) -> None:
    registry = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.is_file()
        else {"schema_version": 1, "candidates": {}}
    )
    state = {
        "admit": "admitted",
        "hold": "held",
        "reject": "quarantined",
    }[admission["decision"]]
    registry.setdefault("candidates", {})[admission["policy_id"]] = {
        "state": state,
        "admission": admission,
        "deployment_note": (
            "Admission records eligibility only. Runtime promotion requires an explicit frozen matrix update."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paired_results", type=Path)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--min-pairs", type=int, default=9)
    parser.add_argument("--min-utility-lcb", type=float, default=0.0)
    parser.add_argument("--max-harmful-rate", type=float, default=0.10)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    args = parser.parse_args()

    result = evaluate_admission(
        load_pairs(args.paired_results),
        policy_id=args.policy_id,
        min_pairs=args.min_pairs,
        min_utility_lcb=args.min_utility_lcb,
        max_harmful_rate=args.max_harmful_rate,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.registry:
        update_registry(args.registry, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
