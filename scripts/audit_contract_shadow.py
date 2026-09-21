#!/usr/bin/env python3
"""Build and evaluate blinded audits for shadow contract applicability.

Sampling is balanced over the shadow prediction and observed verifier outcome.
The blinded file never contains the rule prediction or outcome, preventing the
annotator from simply copying the policy decision or judging with hindsight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ucagent.memory.trajectory_contract import (
    build_transition_contract,
    contract_applicability,
)


LABELS = ("applicable", "inapplicable", "uncertain")
BINARY_LABELS = ("applicable", "inapplicable")


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(item, dict):
                records.append(item)
    return records


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_strategy_catalog(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    pack = json.loads(path.read_text(encoding="utf-8"))
    catalog = {}
    for collection in ("effective_items", "episodes"):
        for item in pack.get(collection, []):
            if not isinstance(item, dict):
                continue
            for key in (item.get("strategy_id"), item.get("episode_id")):
                if key:
                    catalog[str(key)] = item
    return catalog


def normalize_failure(failure: dict) -> dict:
    failure = failure if isinstance(failure, dict) else {}
    return {
        "event_type": failure.get("event_type"),
        "tool": failure.get("tool"),
        "failure_signature": failure.get("failure_signature") or failure.get("signature"),
        "failure_pattern": failure.get("failure_pattern") or failure.get("pattern"),
        "failure_group": failure.get("failure_group") or failure.get("group"),
        "failed_cases_top": list(failure.get("failed_cases_top") or [])[:5],
        "failed_checkpoints_top": list(failure.get("failed_checkpoints_top") or [])[:5],
        "error_top": list(failure.get("error_top") or [])[:3],
    }


def _audit_id(record: dict, strategy_id: str) -> str:
    raw = "|".join((
        str(record.get("run_id") or ""),
        str(record.get("node_id") or ""),
        str(record.get("stage_index") or ""),
        strategy_id,
    ))
    return "contract-audit-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def collect_candidates(analysis: dict, catalog: dict[str, dict] | None = None) -> list[dict]:
    """Expand decision records to one auditable row per candidate strategy."""
    catalog = catalog or {}
    candidates = []
    for record in analysis.get("records", []):
        if not isinstance(record, dict):
            continue
        failure = normalize_failure(record.get("failure", {}))
        outcomes = {
            str(item.get("strategy_id") or ""): item
            for item in record.get("contract_outcomes", [])
            if isinstance(item, dict)
        }
        for strategy in record.get("strategies", []):
            if not isinstance(strategy, dict):
                continue
            strategy_id = str(strategy.get("strategy_id") or strategy.get("episode_id") or "")
            source = catalog.get(strategy_id, {})
            contract = strategy.get("transition_contract")
            if not isinstance(contract, dict) or not contract:
                contract = source.get("transition_contract")
            if not isinstance(contract, dict) or not contract:
                contract = build_transition_contract(source) if source else {}
            if not contract:
                continue

            prediction = strategy.get("contract_policy")
            prediction = prediction if isinstance(prediction, dict) else {}
            applicable = prediction.get("applicable")
            if not isinstance(applicable, bool):
                prediction = contract_applicability(
                    contract,
                    failure,
                    stage_name=str(record.get("stage_name") or ""),
                    min_score=float(prediction.get("threshold", 0.55) or 0.55),
                )
            predicted_label = "applicable" if prediction.get("applicable") is True else "inapplicable"
            outcome = outcomes.get(strategy_id, {})
            outcome_status = str(outcome.get("status") or "unobserved")
            candidates.append({
                "audit_id": _audit_id(record, strategy_id),
                "source": {
                    "run_id": record.get("run_id"),
                    "decision_node_id": record.get("node_id"),
                    "stage_index": record.get("stage_index"),
                    "stage_name": record.get("stage_name"),
                },
                "current_failure": failure,
                "candidate": {
                    "strategy_id": strategy_id,
                    "source_dut": strategy.get("source_dut") or source.get("dut"),
                    "source_stage": strategy.get("source_stage") or source.get("stage_name"),
                    "failure_pattern": strategy.get("failure_pattern") or source.get("failure_pattern"),
                    "action_categories": list(
                        strategy.get("action_categories") or source.get("action_categories") or []
                    )[:5],
                    "transition_contract": contract,
                },
                "prediction": {
                    "label": predicted_label,
                    "score": prediction.get("score"),
                    "threshold": prediction.get("threshold"),
                    "reasons": list(prediction.get("reasons") or []),
                    "hard_mismatches": list(prediction.get("hard_mismatches") or []),
                },
                "observation": {
                    "contract_status": outcome_status,
                    "credit_role": record.get("credit_role"),
                },
                "stratum": f"{predicted_label}:{outcome_status}",
            })
    return candidates


def balanced_sample(candidates: Iterable[dict], size: int, seed: int) -> list[dict]:
    """Round-robin strata while avoiding one strategy dominating the sample."""
    if size <= 0:
        return []
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        groups[str(candidate.get("stratum") or "unknown")].append(candidate)
    for rows in groups.values():
        rng.shuffle(rows)

    selected = []
    seen_ids = set()
    strategy_counts: Counter[str] = Counter()
    strata = sorted(groups)
    rng.shuffle(strata)
    while len(selected) < size:
        progressed = False
        minimum_strategy_count = min(strategy_counts.values(), default=0)
        for stratum in strata:
            rows = groups[stratum]
            if not rows:
                continue
            best_index = next(
                (
                    index
                    for index, item in enumerate(rows)
                    if strategy_counts[str(item["candidate"].get("strategy_id") or "")]
                    <= minimum_strategy_count
                ),
                0,
            )
            item = rows.pop(best_index)
            if item["audit_id"] in seen_ids:
                continue
            selected.append(item)
            seen_ids.add(item["audit_id"])
            strategy_counts[str(item["candidate"].get("strategy_id") or "")] += 1
            progressed = True
            if len(selected) >= size:
                break
        if not progressed:
            break
    rng.shuffle(selected)
    return selected


def blind_record(item: dict) -> dict:
    return {
        "audit_id": item["audit_id"],
        "source": item["source"],
        "current_failure": item["current_failure"],
        "candidate": item["candidate"],
        "instructions": (
            "Judge only whether the candidate contract is applicable to the current failure. "
            "Do not infer applicability from whether a later repair happened to pass."
        ),
        "annotations": [
            {
                "annotator": "",
                "applicability": "",
                "confidence": None,
                "rationale": "",
            }
        ],
    }


def key_record(item: dict) -> dict:
    return {
        "audit_id": item["audit_id"],
        "prediction": item["prediction"],
        "observation": item["observation"],
        "stratum": item["stratum"],
    }


def completed_labels(record: dict) -> list[str]:
    labels = []
    for annotation in record.get("annotations", []):
        if not isinstance(annotation, dict):
            continue
        label = str(annotation.get("applicability") or "").strip().lower()
        if label in LABELS:
            labels.append(label)
    return labels


def merge_annotation_sets(record_sets: Iterable[Iterable[dict]]) -> list[dict]:
    """Merge independently labeled blind files by audit id."""
    merged: dict[str, dict] = {}
    seen_labels: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for records in record_sets:
        for source in records:
            audit_id = str(source.get("audit_id") or "")
            if not audit_id:
                continue
            if audit_id not in merged:
                merged[audit_id] = {**source, "annotations": []}
            for annotation in source.get("annotations", []):
                if not isinstance(annotation, dict):
                    continue
                label = str(annotation.get("applicability") or "").strip().lower()
                if label not in LABELS:
                    continue
                key = (
                    str(annotation.get("annotator") or ""),
                    label,
                    str(annotation.get("rationale") or ""),
                )
                if key in seen_labels[audit_id]:
                    continue
                merged[audit_id]["annotations"].append(annotation)
                seen_labels[audit_id].add(key)
    return list(merged.values())


def cohen_kappa(left: list[str], right: list[str]) -> float | None:
    if not left or len(left) != len(right):
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    lc, rc = Counter(left), Counter(right)
    expected = sum((lc[label] / len(left)) * (rc[label] / len(right)) for label in LABELS)
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def adjudicated_label(labels: list[str]) -> str | None:
    if not labels:
        return None
    counts = Counter(labels)
    best = counts.most_common()
    if len(best) > 1 and best[0][1] == best[1][1]:
        return "uncertain"
    return best[0][0]


def evaluate_annotations(records: Iterable[dict], keys: Iterable[dict]) -> dict:
    key_by_id = {str(item.get("audit_id") or ""): item for item in keys}
    pairs = []
    double_left, double_right = [], []
    total = 0
    uncertain = 0
    for record in records:
        total += 1
        labels = completed_labels(record)
        if len(labels) >= 2:
            double_left.append(labels[0])
            double_right.append(labels[1])
        reference = adjudicated_label(labels)
        if reference == "uncertain":
            uncertain += 1
            continue
        key = key_by_id.get(str(record.get("audit_id") or ""), {})
        predicted = str((key.get("prediction") or {}).get("label") or "")
        if reference in BINARY_LABELS and predicted in BINARY_LABELS:
            pairs.append((reference, predicted))

    confusion = {
        reference: {prediction: 0 for prediction in BINARY_LABELS}
        for reference in BINARY_LABELS
    }
    for reference, prediction in pairs:
        confusion[reference][prediction] += 1
    tp = confusion["applicable"]["applicable"]
    fp = confusion["inapplicable"]["applicable"]
    fn = confusion["applicable"]["inapplicable"]
    tn = confusion["inapplicable"]["inapplicable"]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    accuracy = (tp + tn) / len(pairs) if pairs else None
    interval = wilson_interval(tp, tp + fp)
    kappa = cohen_kappa(double_left, double_right)
    return {
        "schema_version": 1,
        "records_total": total,
        "records_binary_labeled": len(pairs),
        "records_uncertain": uncertain,
        "coverage": round(len(pairs) / total, 4) if total else 0.0,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "applicable_precision": round(precision, 4) if precision is not None else None,
        "applicable_recall": round(recall, 4) if recall is not None else None,
        "applicable_f1": round(f1, 4) if f1 is not None else None,
        "applicable_precision_wilson_95": (
            [round(interval[0], 4), round(interval[1], 4)] if interval else None
        ),
        "confusion_matrix": confusion,
        "double_labeled_records": len(double_left),
        "cohen_kappa": round(kappa, 4) if kappa is not None else None,
        "acceptance_gate": {
            "point_precision_at_least_0_90": precision is not None and precision >= 0.90,
            "sample_size_at_least_50": len(pairs) >= 50,
            "note": "A 0.90 point estimate is not a 0.90 lower confidence bound.",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sample = subparsers.add_parser("sample")
    sample.add_argument("analysis", type=Path)
    sample.add_argument("--pack", type=Path)
    sample.add_argument("--size", type=int, default=50)
    sample.add_argument("--seed", type=int, default=20260818)
    sample.add_argument("--blind-out", type=Path, required=True)
    sample.add_argument("--key-out", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("annotations", type=Path, nargs="+")
    evaluate.add_argument("--key", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "sample":
        analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
        candidates = collect_candidates(analysis, load_strategy_catalog(args.pack))
        selected = balanced_sample(candidates, args.size, args.seed)
        write_jsonl((blind_record(item) for item in selected), args.blind_out)
        write_jsonl((key_record(item) for item in selected), args.key_out)
        print(json.dumps({
            "candidates": len(candidates),
            "selected": len(selected),
            "strata": dict(Counter(item["stratum"] for item in selected)),
            "blind_out": str(args.blind_out),
            "key_out": str(args.key_out),
        }, ensure_ascii=False, indent=2))
        return

    merged = merge_annotation_sets(load_jsonl(path) for path in args.annotations)
    metrics = evaluate_annotations(merged, load_jsonl(args.key))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
