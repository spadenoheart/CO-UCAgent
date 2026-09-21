#!/usr/bin/env python3
"""Evaluate manual labels for verifier-grounded Episode credit assignment."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_context_reuse_pack import CREDIT_ROLES


PREDICTION_EVIDENCE_FIELDS = (
    "failure_before",
    "action_sequence",
    "observation_after",
    "stage_advanced",
    "failure_set_delta",
    "regression_count",
    "information_gain",
    "action_cost",
    "credit_role",
    "confidence",
)


def completed_labels(record: dict) -> list[str]:
    labels = []
    for annotation in record.get("annotations", []):
        if not isinstance(annotation, dict):
            continue
        label = str(annotation.get("credit_role") or "").strip()
        if label in CREDIT_ROLES:
            labels.append(label)
    return labels


def cohen_kappa(left: list[str], right: list[str]) -> float | None:
    if not left or len(left) != len(right):
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        (left_counts[role] / len(left)) * (right_counts[role] / len(right))
        for role in CREDIT_ROLES
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def evaluate_records(records: Iterable[dict]) -> dict:
    predictions: list[str] = []
    references: list[str] = []
    second_left: list[str] = []
    second_right: list[str] = []
    total = 0
    for record in records:
        total += 1
        prediction = str(record.get("model_credit_role") or "").strip()
        labels = completed_labels(record)
        if prediction in CREDIT_ROLES and labels:
            predictions.append(prediction)
            references.append(labels[0])
        if len(labels) >= 2:
            second_left.append(labels[0])
            second_right.append(labels[1])

    confusion = {
        role: {predicted: 0 for predicted in CREDIT_ROLES}
        for role in CREDIT_ROLES
    }
    for predicted, reference in zip(predictions, references):
        confusion[reference][predicted] += 1

    per_role = {}
    f1_values = []
    for role in CREDIT_ROLES:
        tp = confusion[role][role]
        fp = sum(confusion[other][role] for other in CREDIT_ROLES if other != role)
        fn = sum(confusion[role][other] for other in CREDIT_ROLES if other != role)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(confusion[role].values())
        per_role[role] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        if support:
            f1_values.append(f1)

    correct = sum(a == b for a, b in zip(predictions, references))
    return {
        "records_total": total,
        "records_labeled": len(references),
        "coverage": round(len(references) / total, 4) if total else 0.0,
        "accuracy": round(correct / len(references), 4) if references else None,
        "macro_f1_present_roles": round(sum(f1_values) / len(f1_values), 4) if f1_values else None,
        "per_role": per_role,
        "confusion_matrix": confusion,
        "double_labeled_records": len(second_left),
        "cohen_kappa": (
            round(cohen_kappa(second_left, second_right), 4)
            if second_left and cohen_kappa(second_left, second_right) is not None
            else None
        ),
    }


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def refresh_predictions(records: Iterable[dict], pack: dict) -> tuple[list[dict], dict]:
    """Refresh rule predictions by stable Episode id without touching labels."""
    episodes = {
        str(item.get("episode_id")): item
        for item in pack.get("episodes", [])
        if isinstance(item, dict) and item.get("episode_id")
    }
    refreshed = []
    matched = 0
    missing_ids = []
    for original in records:
        record = copy.deepcopy(original)
        episode_id = str(record.get("episode_id") or "")
        episode = episodes.get(episode_id)
        if episode is None:
            missing_ids.append(episode_id)
            refreshed.append(record)
            continue
        matched += 1
        record["model_credit_role"] = episode.get("credit_role")
        record["model_confidence"] = episode.get("confidence")
        evidence = record.get("evidence")
        if not isinstance(evidence, dict):
            evidence = {}
        for field in PREDICTION_EVIDENCE_FIELDS:
            evidence[field] = copy.deepcopy(episode.get(field))
        record["evidence"] = evidence
        refreshed.append(record)
    return refreshed, {
        "records_total": len(refreshed),
        "matched": matched,
        "missing": len(missing_ids),
        "missing_episode_ids": missing_ids,
    }


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def merge_annotation_records(base_records: Iterable[dict], other_sets: Iterable[Iterable[dict]]) -> tuple[list[dict], dict]:
    """Merge independently labeled files by Episode id without exposing predictions."""
    merged = [copy.deepcopy(record) for record in base_records]
    by_id = {str(record.get("episode_id") or ""): record for record in merged}
    source_sets = 1
    missing_ids = []

    def valid_annotations(record: dict) -> list[dict]:
        return [
            copy.deepcopy(annotation)
            for annotation in record.get("annotations", [])
            if isinstance(annotation, dict)
            and str(annotation.get("credit_role") or "").strip() in CREDIT_ROLES
        ]

    for record in merged:
        record["annotations"] = valid_annotations(record)
    for records in other_sets:
        source_sets += 1
        for source in records:
            episode_id = str(source.get("episode_id") or "")
            target = by_id.get(episode_id)
            if target is None:
                missing_ids.append(episode_id)
                continue
            existing = {
                (
                    str(item.get("annotator") or ""),
                    str(item.get("credit_role") or ""),
                    str(item.get("notes") or ""),
                )
                for item in target["annotations"]
            }
            for annotation in valid_annotations(source):
                key = (
                    str(annotation.get("annotator") or ""),
                    str(annotation.get("credit_role") or ""),
                    str(annotation.get("notes") or ""),
                )
                if key not in existing:
                    target["annotations"].append(annotation)
                    existing.add(key)
    return merged, {
        "source_sets": source_sets,
        "records_total": len(merged),
        "records_with_labels": sum(bool(record.get("annotations")) for record in merged),
        "records_double_labeled": sum(len(record.get("annotations", [])) >= 2 for record in merged),
        "missing": len(missing_ids),
        "missing_episode_ids": missing_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--predictions-pack",
        type=Path,
        help="Optional v2 candidate pack used to refresh predictions by episode_id.",
    )
    parser.add_argument(
        "--refreshed-out",
        type=Path,
        help="Optional JSONL path for records with refreshed rule predictions.",
    )
    parser.add_argument(
        "--merge-annotations",
        type=Path,
        action="append",
        default=[],
        help="Independent annotation JSONL to merge by episode_id. Repeat as needed.",
    )
    parser.add_argument(
        "--merged-out",
        type=Path,
        help="Optional pre-prediction merged annotation JSONL.",
    )
    args = parser.parse_args()

    records = load_jsonl(args.annotations)
    merge_summary = None
    if args.merge_annotations:
        records, merge_summary = merge_annotation_records(
            records,
            [load_jsonl(path) for path in args.merge_annotations],
        )
    if args.merged_out:
        write_jsonl(records, args.merged_out)
    refresh_summary = None
    if args.predictions_pack:
        pack = json.loads(args.predictions_pack.read_text(encoding="utf-8"))
        records, refresh_summary = refresh_predictions(records, pack)
    if args.refreshed_out:
        write_jsonl(records, args.refreshed_out)

    result = evaluate_records(records)
    if merge_summary is not None:
        result["annotation_merge"] = merge_summary
    if refresh_summary is not None:
        result["prediction_refresh"] = refresh_summary
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
