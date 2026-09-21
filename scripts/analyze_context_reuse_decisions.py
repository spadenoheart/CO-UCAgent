#!/usr/bin/env python3
"""Link context-reuse decisions to the next verifier observation.

The output is observational credit, not a causal effect estimate. A causal
comparison still requires paired runs from the same stage checkpoint with and
without the retrieved context.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_context_reuse_pack import build_credit_episode, is_failure_observation, stage_key
from ucagent.memory.trajectory_contract import (
    build_transition_contract,
    evaluate_transition_contract,
)


def _sort_nodes(nodes: list[dict]) -> list[dict]:
    return sorted(
        nodes,
        key=lambda node: (
            float(node.get("time_unix", 0) or 0),
            int(node.get("round_index", 0) or 0),
            str(node.get("node_id") or ""),
        ),
    )


def _matching_failure(stage_nodes: list[dict], decision_index: int, signature: str) -> dict | None:
    fallback = None
    for node in reversed(stage_nodes[:decision_index]):
        if not is_failure_observation(node):
            continue
        if fallback is None:
            fallback = node
        observation = node.get("observation") if isinstance(node.get("observation"), dict) else {}
        if signature and observation.get("failure_signature") == signature:
            return node
    return fallback


def analyze_trace(trace: dict, strategy_catalog: dict[str, dict] | None = None) -> dict:
    strategy_catalog = strategy_catalog or {}
    by_stage: dict[tuple, list[dict]] = defaultdict(list)
    for node in trace.get("nodes", []):
        by_stage[stage_key(node)].append(node)

    decision_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    utility_gate_counts: Counter[str] = Counter()
    hard_gate_counts: Counter[str] = Counter()
    strategy_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    contract_outcome_counts: Counter[str] = Counter()
    contract_applicability_counts: Counter[str] = Counter()
    records = []
    for (run_id, stage_index, stage_name), raw_nodes in by_stage.items():
        stage_nodes = _sort_nodes(raw_nodes)
        for index, node in enumerate(stage_nodes):
            if node.get("event_type") != "context_reuse_decision":
                continue
            memory = node.get("memory_decision") if isinstance(node.get("memory_decision"), dict) else {}
            decision = str(memory.get("decision") or "unknown")
            decision_counts[decision] += 1
            search_summary = memory.get("search_summary") if isinstance(memory.get("search_summary"), dict) else {}
            utility_gate_counts["match_candidates"] += int(search_summary.get("match_candidates", 0) or 0)
            utility_gate_counts["passed_candidates"] += int(search_summary.get("utility_passed", 0) or 0)
            utility_gate_counts["rejected_candidates"] += int(search_summary.get("utility_rejected", 0) or 0)
            utility_gate_counts["unknown_candidates"] += int(search_summary.get("utility_unknown", 0) or 0)
            hard_gate_counts["rejected_candidates"] += int(search_summary.get("hard_gate_rejected", 0) or 0)
            contract_applicability_counts["candidates"] += int(
                search_summary.get("contract_candidates", 0) or 0
            )
            contract_applicability_counts["applicable"] += int(
                search_summary.get("contract_applicable", 0) or 0
            )
            contract_applicability_counts["inapplicable"] += int(
                search_summary.get("contract_inapplicable", 0) or 0
            )
            contract_applicability_counts["enforced_rejected"] += int(
                search_summary.get("contract_enforced_rejected", 0) or 0
            )
            for reason, count in (search_summary.get("hard_gate_reasons") or {}).items():
                hard_gate_counts[str(reason)] += int(count or 0)
            if memory.get("reason") == "utility_gate_rejected_all":
                utility_gate_counts["all_rejected_decisions"] += 1
            record = {
                "node_id": node.get("node_id"),
                "run_id": run_id,
                "stage_index": stage_index,
                "stage_name": stage_name,
                "decision": decision,
                "reason": memory.get("reason"),
                "failure": memory.get("failure", {}),
                "strategies": memory.get("strategies", []),
                "search_summary": search_summary,
                "prompt_item_count": memory.get("prompt_item_count", 0),
                "credit_role": None,
                "credit_episode": None,
                "contract_outcomes": [],
            }
            if decision == "inject":
                signature = str((memory.get("failure") or {}).get("signature") or "")
                failure_node = _matching_failure(stage_nodes, index, signature)
                after_index = next(
                    (
                        candidate_index
                        for candidate_index in range(index + 1, len(stage_nodes))
                        if stage_nodes[candidate_index].get("event_type") in {"test_run", "check_result"}
                    ),
                    None,
                )
                if failure_node is not None and after_index is not None:
                    actions = [
                        candidate
                        for candidate in stage_nodes[index + 1:after_index]
                        if candidate.get("event_type") == "file_mutation" and candidate.get("success") is True
                    ]
                    episode = build_credit_episode(
                        dut=str(trace.get("dut") or ""),
                        run_id=run_id,
                        stage_index=stage_index,
                        stage_name=stage_name,
                        failure_node=failure_node,
                        after_node=stage_nodes[after_index],
                        after_index=after_index,
                        stage_nodes=stage_nodes,
                        action_nodes=actions,
                    )
                    role = episode.get("credit_role")
                    record["credit_role"] = role
                    record["credit_episode"] = episode
                    outcome_counts[str(role)] += 1
                    for strategy in memory.get("strategies", []):
                        if not isinstance(strategy, dict):
                            continue
                        strategy_id = strategy.get("strategy_id") or strategy.get("episode_id")
                        if strategy_id:
                            strategy_outcomes[str(strategy_id)][str(role)] += 1
                        contract = (
                            strategy.get("transition_contract")
                            if isinstance(strategy.get("transition_contract"), dict)
                            else {}
                        )
                        if not contract and strategy_id in strategy_catalog:
                            source_item = strategy_catalog[str(strategy_id)]
                            contract = (
                                source_item.get("transition_contract")
                                if isinstance(source_item.get("transition_contract"), dict)
                                else build_transition_contract(source_item)
                            )
                        if contract:
                            contract_outcome = evaluate_transition_contract(
                                contract,
                                episode.get("failure_before", {}),
                                episode.get("observation_after", {}),
                                stage_advanced=bool(episode.get("stage_advanced")),
                                scope_relation=str(
                                    episode.get("validation_scope_relation") or "same"
                                ),
                            )
                            contract_outcome["strategy_id"] = strategy_id
                            record["contract_outcomes"].append(contract_outcome)
                            contract_outcome_counts[str(contract_outcome.get("status") or "unknown")] += 1
            records.append(record)

    evaluated = sum(outcome_counts.values())
    progress = outcome_counts.get("progress", 0)
    regression = outcome_counts.get("regression", 0)
    contracts_evaluated = sum(contract_outcome_counts.values())
    contracts_fulfilled = contract_outcome_counts.get("fulfilled", 0)
    return {
        "schema_version": 2,
        "analysis_type": "context_reuse_decision_observational_credit",
        "causal_claim": False,
        "decision_counts": dict(decision_counts),
        "utility_gate_counts": dict(utility_gate_counts),
        "hard_gate_counts": dict(hard_gate_counts),
        "injections_evaluated": evaluated,
        "credit_role_counts": dict(outcome_counts),
        "observed_progress_rate": round(progress / evaluated, 4) if evaluated else None,
        "observed_regression_rate": round(regression / evaluated, 4) if evaluated else None,
        "strategy_outcomes": {
            strategy_id: dict(counts)
            for strategy_id, counts in sorted(strategy_outcomes.items())
        },
        "contract_applicability_counts": dict(contract_applicability_counts),
        "contract_outcome_counts": dict(contract_outcome_counts),
        "contracts_evaluated": contracts_evaluated,
        "observed_contract_fulfillment_rate": (
            round(contracts_fulfilled / contracts_evaluated, 4)
            if contracts_evaluated
            else None
        ),
        "records": records,
    }


def write_markdown(result: dict, path: Path) -> None:
    lines = [
        "# Context Reuse Decision Credit",
        "",
        f"- Decisions: `{json.dumps(result['decision_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- Utility gate: `{json.dumps(result['utility_gate_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- Semantic hard gates: `{json.dumps(result['hard_gate_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- Evaluated injections: {result['injections_evaluated']}",
        f"- Credit roles: `{json.dumps(result['credit_role_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- Observed progress rate: {result['observed_progress_rate']}",
        f"- Observed regression rate: {result['observed_regression_rate']}",
        f"- Contract applicability: `{json.dumps(result.get('contract_applicability_counts', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- Contract outcomes: `{json.dumps(result.get('contract_outcome_counts', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- Observed contract fulfillment rate: {result.get('observed_contract_fulfillment_rate')}",
        "- Causal claim: false; use paired checkpoint replay for causal uplift.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_tree", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument(
        "--pack",
        type=Path,
        help="Optional context-reuse pack used to backfill contracts for older traces.",
    )
    args = parser.parse_args()

    trace = json.loads(args.trace_tree.read_text(encoding="utf-8"))
    strategy_catalog = {}
    if args.pack:
        pack = json.loads(args.pack.read_text(encoding="utf-8"))
        for item in pack.get("effective_items", []):
            if not isinstance(item, dict):
                continue
            for key in (item.get("strategy_id"), item.get("episode_id")):
                if key:
                    strategy_catalog[str(key)] = item
    result = analyze_trace(trace, strategy_catalog=strategy_catalog)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.out_md:
        write_markdown(result, args.out_md)
    print(json.dumps({
        "out_json": str(args.out_json),
        "out_md": str(args.out_md) if args.out_md else None,
        "decision_counts": result["decision_counts"],
        "utility_gate_counts": result["utility_gate_counts"],
        "injections_evaluated": result["injections_evaluated"],
        "credit_role_counts": result["credit_role_counts"],
        "contract_outcome_counts": result["contract_outcome_counts"],
        "observed_contract_fulfillment_rate": result["observed_contract_fulfillment_rate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
