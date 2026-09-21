# -*- coding: utf-8 -*-
"""Conservative long-term memory store for UCAgent.

This module is intentionally isolated from the core execution path:
- when disabled, it does nothing
- when enabled, it only stores compact, high-value events
- retrieval uses a small prompt-facing working set
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import glob
import re
import time
from typing import Dict, Iterable, List, Optional, Tuple

from ucagent.util.log import info

try:
    from langchain_openai import OpenAIEmbeddings
except Exception:  # pragma: no cover - runtime optional
    OpenAIEmbeddings = None


def _safe_int(value, default=0) -> int:
    try:
        return int(value or 0)
    except Exception:
        return default


def _normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def _tokenize(text: str) -> List[str]:
    tokens = []
    buf = []
    for ch in (text or ""):
        if ch.isalnum() or ch in ("_", "-", "/", "."):
            buf.append(ch.lower())
        else:
            if buf:
                tokens.append("".join(buf))
                buf = []
    if buf:
        tokens.append("".join(buf))
    return [t for t in tokens if len(t) >= 3]


def _dedup_keep_order(values: List[str], limit: int) -> List[str]:
    out = []
    seen = set()
    for item in values or []:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


_STAGE_TERM_STOPWORDS = {
    "dut", "test", "tests", "testing", "verify", "verification", "analysis", "analyze",
    "implement", "implementation", "create", "creation", "generate", "generation",
    "function", "functions", "stage", "planning", "current", "basic", "random",
    "summary", "review", "understanding", "task", "tasks", "plan", "plans",
    "write", "writing", "case", "cases", "check", "checks", "prepare", "preparation",
    "code", "codes", "module", "modules", "run", "running", "coverage",
}


def _jaccard(a: List[str], b: List[str]) -> float:
    sa = set(_dedup_keep_order(a or [], 64))
    sb = set(_dedup_keep_order(b or [], 64))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _status_rank(status: str) -> int:
    return {"failure": 3, "mixed": 2, "success": 1}.get(str(status or "").lower(), 0)


def _stage_terms(text: str, dut_name: str = "") -> List[str]:
    raw_tokens = _tokenize(text)
    dut_tokens = set(_tokenize(dut_name))
    out = []
    seen = set()
    for token in raw_tokens:
        parts = [token]
        if any(sep in token for sep in ("_", "-", "/", ".")):
            parts.extend(part for part in re.split(r"[_\-/\.]+", token) if part)
        for part in parts:
            part = part.lower().strip()
            if len(part) < 3:
                continue
            if part in _STAGE_TERM_STOPWORDS:
                continue
            if part in dut_tokens:
                continue
            if part in seen:
                continue
            seen.add(part)
            out.append(part)
    return out


class LongTermMemoryStore:
    """JSONL-based memory store with scoring, merge/promotion, and lazy embeddings."""

    def __init__(
        self,
        workspace: str,
        dut_name: str,
        max_entries: int = 256,
        enable_embed: bool = False,
        embed_config: Optional[Dict] = None,
        options: Optional[Dict] = None,
    ):
        self.workspace = os.path.abspath(workspace)
        self.dut_name = dut_name
        self.enable_embed = enable_embed
        self.embed_config = embed_config or {}
        self.options = options or {}
        if hasattr(self.options, "as_dict"):
            self.options = self.options.as_dict()
        elif not isinstance(self.options, dict):
            self.options = {}
        self.max_entries = int(self.options.get("max_entries", max_entries) or max_entries)
        self.max_candidate_entries = int(self.options.get("max_candidate_entries", 2000) or 2000)
        self.prompt_limit = int(self.options.get("prompt_limit", 3) or 3)
        self.recent_fallback_limit = int(self.options.get("recent_fallback_limit", 2) or 2)
        self.min_write_score = float(self.options.get("min_write_score", 0.58) or 0.58)
        self.episode_support = int(self.options.get("episode_support", 2) or 2)
        self.semantic_support = int(self.options.get("semantic_support", 4) or 4)
        self.keep_candidates = bool(self.options.get("keep_candidates", True))
        self.skip_passed_singleton = bool(self.options.get("skip_passed_singleton", True))
        self.preload_from_completed = bool(self.options.get("preload_from_completed", True))
        self.preload_keep = int(self.options.get("preload_keep", min(self.max_entries, 64)) or min(self.max_entries, 64))
        self.prefetch_min_score = float(self.options.get("prefetch_min_score", 0.22) or 0.22)
        self.prefetch_min_title_match = float(self.options.get("prefetch_min_title_match", 0.12) or 0.12)
        self.prefetch_max_stage_delta = int(self.options.get("prefetch_max_stage_delta", 3) or 3)
        self.prefetch_relative_score = float(self.options.get("prefetch_relative_score", 0.82) or 0.82)
        self.prefetch_inject_min_score = float(self.options.get("prefetch_inject_min_score", 0.25) or 0.25)
        self.prefetch_max_per_source_stage = int(self.options.get("prefetch_max_per_source_stage", 1) or 1)

        self.base_dir = os.path.join(self.workspace, ".ucagent_memory", dut_name)
        self.path = os.path.join(self.base_dir, "memory.jsonl")
        self.candidate_path = os.path.join(self.base_dir, "memory.candidate.jsonl")
        self.vector_path = os.path.join(self.base_dir, "memory.emb.jsonl")
        self.metrics_path = os.path.join(self.base_dir, "metrics.jsonl")
        os.makedirs(self.base_dir, exist_ok=True)

        self._embedder = None
        self._vector_cache: Optional[List[Dict]] = None
        self._vector_dirty = True
        self._version = 0
        self._runtime_metrics = {
            "queries": 0,
            "retrieval_hits": 0,
            "retrieval_misses": 0,
            "recent_fallback_hits": 0,
            "prefetch_prompt_hits": 0,
            "stage_cache_hits": 0,
            "prompt_injections": 0,
            "injected_items": 0,
            "useful_hits": 0,
            "retrieval_useful_hits": 0,
            "prefetch_useful_hits": 0,
            "fallback_useful_hits": 0,
            "stale_hits": 0,
            "memory_pollution": 0,
            "retrieval_pollution": 0,
            "prefetch_pollution": 0,
            "fallback_pollution": 0,
        }
        self._maybe_preload_from_completed()

    def version(self) -> int:
        return self._version

    def clear(self) -> None:
        for path in (self.path, self.candidate_path, self.vector_path, self.metrics_path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        self._touch_memory_changed()

    def save(self, meta: Dict, content: Dict | str) -> bool:
        entry = self._normalize_entry(meta or {}, content)
        if entry is None:
            return False

        if self.keep_candidates:
            self._append_jsonl(self.candidate_path, entry)
            self._prune_jsonl(self.candidate_path, self.max_candidate_entries)

        score = float(entry.get("write_score", 0.0) or 0.0)
        if score < self.min_write_score:
            info(
                f"[long_term_memory][skip] score={score:.3f} "
                f"kind={entry.get('content', {}).get('kind')} status={entry.get('content', {}).get('status')}"
            )
            return False

        entries = self._load_jsonl(self.path)
        best_idx, best_score = self._find_merge_target(entry, entries)
        if best_idx >= 0:
            entries[best_idx] = self._merge_entries(entries[best_idx], entry)
            info(
                f"[long_term_memory][merge] score={best_score:.3f} "
                f"level={entries[best_idx].get('level')} support={entries[best_idx].get('support')}"
            )
        else:
            entry["level"] = self._decide_level(1)
            entries.append(entry)
            info(
                f"[long_term_memory][write] score={score:.3f} "
                f"level={entry.get('level')} kind={entry.get('content', {}).get('kind')} "
                f"status={entry.get('content', {}).get('status')}"
            )

        entries.sort(key=lambda item: (float(item.get("last_seen", 0.0) or 0.0), int(item.get("support", 1) or 1)))
        entries = entries[-self.max_entries :]
        self._write_jsonl(self.path, entries)
        self._touch_memory_changed()
        return True

    def recent(self, limit: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        entries = self._apply_filters(self._load_jsonl(self.path), filters)
        entries.sort(key=lambda item: float(item.get("last_seen", 0.0) or 0.0), reverse=True)
        return entries[:limit]

    def search(self, query: str, limit: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        entries = self._apply_filters(self._load_jsonl(self.path), filters)
        self._runtime_metrics["queries"] += 1
        if not entries:
            self._runtime_metrics["retrieval_misses"] += 1
            self._log_metric_event("query", {"query": query, "limit": limit, "filters": filters or {}, "hits": 0})
            return []

        embed_scores = {}
        if self.enable_embed and self._get_embedder() is not None:
            self._ensure_vector_index(entries)
            embed_scores = self._search_embed_scores(query)

        query_tokens = _tokenize(query)
        scored = []
        for entry in entries:
            lexical = self._lexical_score(query_tokens, entry)
            embed_score = embed_scores.get(entry.get("hash"), 0.0)
            support_bonus = min(int(entry.get("support", 1) or 1), 5) * 0.04
            level_bonus = {"semantic": 0.15, "episode": 0.08, "candidate": 0.02}.get(entry.get("level"), 0.0)
            status_bonus = {"failure": 0.10, "mixed": 0.05, "success": 0.02}.get(
                entry.get("content", {}).get("status"), 0.0
            )
            stale_penalty = min(float(entry.get("stale_score", 0.0) or 0.0), 1.0) * 0.12
            resolved_penalty = 0.18 if bool(entry.get("resolved", False)) else 0.0
            score = lexical * 0.65 + embed_score * 0.55 + support_bonus + level_bonus + status_bonus - stale_penalty - resolved_penalty
            scored.append(
                (
                    score,
                    {
                        **entry,
                        "__retrieval__": {
                            "query": query,
                            "hit_type": "retrieval",
                            "from_prefetch": False,
                            "score": round(score, 4),
                            "score_breakdown": {
                                "lexical": round(lexical, 4),
                                "embed": round(embed_score, 4),
                                "support_bonus": round(support_bonus, 4),
                                "level_bonus": round(level_bonus, 4),
                                "status_bonus": round(status_bonus, 4),
                                "stale_penalty": round(stale_penalty, 4),
                                "resolved_penalty": round(resolved_penalty, 4),
                            },
                        },
                    },
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        results = [entry for score, entry in scored if score > 0.01][:limit]
        if results:
            self._runtime_metrics["retrieval_hits"] += 1
        else:
            self._runtime_metrics["retrieval_misses"] += 1
        self._log_metric_event(
            "query",
            {
                "query": query,
                "limit": limit,
                "filters": filters or {},
                "hits": len(results),
                "top_hashes": [item.get("hash") for item in results[:3]],
                "top_scores": [item.get("__retrieval__", {}).get("score") for item in results[:3]],
            },
        )
        if self.enable_embed:
            top_scores = [round(score, 4) for score, _ in scored[:3]]
            info(
                f"[long_term_memory][embed_search] query='{(query or '')[:80]}' "
                f"hits={len(results)} top3={top_scores}"
            )
        return results

    def prefetch_search(
        self,
        query: str,
        stage_index: Optional[int] = None,
        stage_title: str = "",
        limit: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        entries = self._apply_filters(self._load_jsonl(self.path), filters)
        if not entries:
            self._log_metric_event(
                "prefetch_query",
                {"query": query, "stage_index": stage_index, "stage_title": stage_title, "hits": 0},
            )
            return []
        title_tokens = _stage_terms(stage_title, self.dut_name)
        scored = []
        for entry in entries:
            meta = entry.get("meta", {}) or {}
            entry_title_terms = _stage_terms(meta.get("stage_title", ""), self.dut_name)
            lexical = self._lexical_score(
                title_tokens,
                {"retrieval_text": _normalize_text(" ".join(entry_title_terms or [meta.get("stage_title", "")]))},
            )
            title_match = _jaccard(title_tokens, entry_title_terms) if title_tokens else 0.0
            entry_stage_index = meta.get("stage_index")
            proximity = 0.0
            if isinstance(stage_index, int) and isinstance(entry_stage_index, int):
                delta = abs(stage_index - entry_stage_index)
                if delta == 0:
                    proximity = 0.26
                elif delta == 1:
                    proximity = 0.18
                elif delta == 2:
                    proximity = 0.10
                elif delta <= self.prefetch_max_stage_delta:
                    proximity = 0.04
                else:
                    proximity = 0.0
            level_bonus = {"semantic": 0.18, "episode": 0.10, "candidate": 0.03}.get(entry.get("level"), 0.0)
            support_bonus = min(int(entry.get("support", 1) or 1), 5) * 0.04
            useful_count = int(entry.get("useful_count", 0) or 0)
            pollution_count = int(entry.get("pollution_count", 0) or 0)
            useful_ratio = useful_count / max(1, useful_count + pollution_count)
            historical_bonus = useful_ratio * 0.12 + min(useful_count, 3) * 0.03
            pollution_penalty = min(pollution_count, 3) * 0.06
            resolved_penalty = 0.20 if bool(entry.get("resolved", False)) else 0.0
            stale_penalty = min(float(entry.get("stale_score", 0.0) or 0.0), 1.0) * 0.12
            hard_match = title_match >= self.prefetch_min_title_match or proximity >= 0.10
            if pollution_count > useful_count + 1 and useful_count == 0:
                hard_match = title_match >= max(self.prefetch_min_title_match, 0.20) or proximity >= 0.18
            score = (
                lexical * 0.20
                + title_match * 0.50
                + proximity
                + level_bonus
                + support_bonus
                + historical_bonus
                - pollution_penalty
                - resolved_penalty
                - stale_penalty
            )
            if not hard_match:
                continue
            if score < self.prefetch_min_score:
                continue
            scored.append(
                (
                    score,
                    {
                        **entry,
                        "__retrieval__": {
                            "query": query,
                            "hit_type": "prefetch",
                            "from_prefetch": True,
                            "score": round(score, 4),
                            "score_breakdown": {
                                "lexical": round(lexical, 4),
                                "title_match": round(title_match, 4),
                                "proximity": round(proximity, 4),
                                "level_bonus": round(level_bonus, 4),
                                "support_bonus": round(support_bonus, 4),
                                "historical_bonus": round(historical_bonus, 4),
                                "pollution_penalty": round(pollution_penalty, 4),
                                "stale_penalty": round(stale_penalty, 4),
                                "resolved_penalty": round(resolved_penalty, 4),
                            },
                        },
                    },
                )
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        results = [entry for score, entry in scored][:limit]
        self._log_metric_event(
            "prefetch_query",
            {
                "query": query,
                "stage_index": stage_index,
                "stage_title": stage_title,
                "hits": len(results),
                "top_hashes": [item.get("hash") for item in results[:3]],
                "top_scores": [item.get("__retrieval__", {}).get("score") for item in results[:3]],
            },
        )
        return results

    def mark_prompt_usage(self, query: str, entries: List[Dict], hit_type: str = "retrieval", from_prefetch: bool = False) -> None:
        if not entries:
            return
        store_entries = self._load_jsonl(self.path)
        by_hash = {item.get("hash"): item for item in store_entries}
        now_ts = time.time()
        touched = []
        for result in entries:
            h = result.get("hash")
            if not h or h not in by_hash:
                continue
            entry = by_hash[h]
            entry["hit_count"] = int(entry.get("hit_count", 0) or 0) + 1
            entry["reuse_count"] = int(entry.get("reuse_count", 0) or 0) + 1
            entry["last_hit"] = now_ts
            if from_prefetch:
                entry["prefetched"] = True
            if bool(entry.get("resolved", False)) or float(entry.get("stale_score", 0.0) or 0.0) >= 0.5:
                self._runtime_metrics["stale_hits"] += 1
            touched.append(h)
        if touched:
            self._runtime_metrics["prompt_injections"] += 1
            self._runtime_metrics["injected_items"] += len(touched)
            if hit_type == "recent_fallback":
                self._runtime_metrics["recent_fallback_hits"] += 1
            elif hit_type == "prefetch":
                self._runtime_metrics["prefetch_prompt_hits"] += 1
            elif hit_type == "stage_cache":
                self._runtime_metrics["stage_cache_hits"] += 1
            self._write_jsonl(self.path, list(by_hash.values()))
            self._log_metric_event(
                "prompt_usage",
                {
                    "query": query,
                    "hit_type": hit_type,
                    "from_prefetch": from_prefetch,
                    "hashes": touched,
                },
            )

    def mark_useful_hit(self, entry_hashes: List[str], useful: bool, reason: str = "", hit_type: str = "retrieval") -> None:
        if not entry_hashes:
            return
        store_entries = self._load_jsonl(self.path)
        by_hash = {item.get("hash"): item for item in store_entries}
        now_ts = time.time()
        touched = []
        for h in entry_hashes:
            entry = by_hash.get(h)
            if not entry:
                continue
            touched.append(h)
            if useful:
                self._runtime_metrics["useful_hits"] += 1
                if hit_type == "retrieval":
                    self._runtime_metrics["retrieval_useful_hits"] += 1
                elif hit_type == "prefetch":
                    self._runtime_metrics["prefetch_useful_hits"] += 1
                elif hit_type == "recent_fallback":
                    self._runtime_metrics["fallback_useful_hits"] += 1
                entry["reuse_count"] = int(entry.get("reuse_count", 0) or 0) + 1
                entry["useful_count"] = int(entry.get("useful_count", 0) or 0) + 1
                entry["stale_score"] = round(max(0.0, float(entry.get("stale_score", 0.0) or 0.0) - 0.1), 4)
            else:
                self._runtime_metrics["memory_pollution"] += 1
                if hit_type == "retrieval":
                    self._runtime_metrics["retrieval_pollution"] += 1
                elif hit_type == "prefetch":
                    self._runtime_metrics["prefetch_pollution"] += 1
                elif hit_type == "recent_fallback":
                    self._runtime_metrics["fallback_pollution"] += 1
                entry["pollution_count"] = int(entry.get("pollution_count", 0) or 0) + 1
                entry["stale_score"] = round(min(1.0, float(entry.get("stale_score", 0.0) or 0.0) + 0.08), 4)
            entry["last_hit"] = now_ts
        if touched:
            self._write_jsonl(self.path, list(by_hash.values()))
            self._log_metric_event(
                "useful_hit",
                {"hashes": touched, "useful": useful, "reason": reason, "hit_type": hit_type},
            )

    def get_runtime_metrics(self) -> Dict:
        metrics = dict(self._runtime_metrics)
        queries = max(1, metrics["queries"])
        total_memory_hits = int(metrics["retrieval_hits"] or 0) + int(metrics["recent_fallback_hits"] or 0)
        total_memory_hits += int(metrics["prefetch_prompt_hits"] or 0)
        metrics["retrieval_hit_rate"] = round(metrics["retrieval_hits"] / queries, 4)
        metrics["recent_fallback_rate"] = round(metrics["recent_fallback_hits"] / queries, 4)
        metrics["prefetch_prompt_hit_rate"] = round(metrics["prefetch_prompt_hits"] / queries, 4)
        metrics["stale_hit_rate"] = round(metrics["stale_hits"] / max(1, metrics["injected_items"]), 4)
        metrics["useful_hit_rate"] = round(metrics["useful_hits"] / max(1, total_memory_hits), 4)
        metrics["retrieval_useful_hit_rate"] = round(metrics["retrieval_useful_hits"] / max(1, metrics["retrieval_hits"]), 4)
        metrics["prefetch_useful_hit_rate"] = round(metrics["prefetch_useful_hits"] / max(1, metrics["prefetch_prompt_hits"]), 4)
        metrics["fallback_useful_hit_rate"] = round(metrics["fallback_useful_hits"] / max(1, metrics["recent_fallback_hits"]), 4)
        metrics["memory_pollution_rate"] = round(metrics["memory_pollution"] / max(1, metrics["prompt_injections"]), 4)
        metrics["retrieval_pollution_rate"] = round(metrics["retrieval_pollution"] / max(1, metrics["retrieval_hits"]), 4)
        metrics["prefetch_pollution_rate"] = round(metrics["prefetch_pollution"] / max(1, metrics["prefetch_prompt_hits"]), 4)
        metrics["fallback_pollution_rate"] = round(metrics["fallback_pollution"] / max(1, metrics["recent_fallback_hits"]), 4)
        return metrics

    def archive_completed(self) -> None:
        archived_main = _rename_with_fallback(self.path, os.path.join(self.base_dir, "memory.completed.jsonl"))
        archived_candidate = _rename_with_fallback(
            self.candidate_path, os.path.join(self.base_dir, "memory.candidate.completed.jsonl")
        )
        archived_vec = _rename_with_fallback(self.vector_path, os.path.join(self.base_dir, "memory.emb.completed.jsonl"))
        archived_metrics = _rename_with_fallback(self.metrics_path, os.path.join(self.base_dir, "metrics.completed.jsonl"))
        if archived_main:
            info(f"[long_term_memory] archived {archived_main}")
        if archived_candidate:
            info(f"[long_term_memory] archived {archived_candidate}")
        if archived_vec:
            info(f"[long_term_memory] archived {archived_vec}")
        if archived_metrics:
            info(f"[long_term_memory] archived {archived_metrics}")
        self._touch_memory_changed()

    def _normalize_entry(self, meta: Dict, content: Dict | str) -> Optional[Dict]:
        payload = self._normalize_content(meta, content)
        if payload is None:
            return None
        ts = float(meta.get("timestamp") or time.time())
        summary = payload.get("summary", "") or ""
        raw_hash = json.dumps(
            {
                "meta": {
                    "type": meta.get("type"),
                    "dut": meta.get("dut", self.dut_name),
                    "stage_index": payload.get("stage_info", {}).get("stage_index"),
                },
                "content": {
                    "kind": payload.get("kind"),
                    "status": payload.get("status"),
                    "failed_checkpoints_top": payload.get("failed_checkpoints_top", []),
                    "failed_cases_top": payload.get("failed_cases_top", []),
                    "summary": summary,
                    "root_cause": payload.get("root_cause", ""),
                    "action": payload.get("action", ""),
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        score = self._compute_write_score(payload)
        entry = {
            "hash": hashlib.sha1(raw_hash.encode("utf-8", errors="ignore")).hexdigest(),
            "meta": {
                "type": meta.get("type", payload.get("kind", "")),
                "dut": meta.get("dut", self.dut_name),
                "stage_index": payload.get("stage_info", {}).get("stage_index"),
                "stage_title": payload.get("stage_info", {}).get("stage_title", ""),
                "section_index": payload.get("stage_info", {}).get("section_index", ""),
                "timestamp": ts,
            },
            "content": payload,
            "support": 1,
            "level": "candidate",
            "write_score": round(score, 4),
            "first_seen": ts,
            "last_seen": ts,
            "retrieval_text": self._build_retrieval_text(payload),
            "hit_count": 0,
            "last_hit": 0.0,
            "reuse_count": 0,
            "useful_count": 0,
            "pollution_count": 0,
            "prefetched": False,
            "resolved": False,
            "resolved_at": 0.0,
            "stale_score": 0.0,
        }
        return entry

    def _normalize_content(self, meta: Dict, content: Dict | str) -> Optional[Dict]:
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    content = parsed
            except Exception:
                return None

        if not isinstance(content, dict):
            return None

        kind = str(content.get("kind") or meta.get("type") or "").strip() or "memory"
        stage_info = self._extract_stage_info(meta, content)
        tests = self._extract_tests(content)
        failed_cases = _dedup_keep_order(self._extract_failed_cases(content), 10)
        failed_checkpoints = _dedup_keep_order(self._extract_failed_checkpoints(content), 12)
        root_cause = self._extract_root_cause(content)
        action = self._extract_action(content)
        summary = str(content.get("summary") or "").strip()

        if not summary:
            summary = self._build_summary(kind, stage_info, tests, failed_cases, failed_checkpoints, root_cause, action)

        status = self._infer_status(content, tests, failed_cases, failed_checkpoints)

        if kind == "turn" and status == "success" and not root_cause and not action:
            return None
        if self.skip_passed_singleton and kind == "batch" and status == "success" and _safe_int(tests.get("total")) <= 1:
            return None

        payload = {
            "kind": kind,
            "status": status,
            "stage_info": stage_info,
            "tests": tests,
            "failed_cases_top": failed_cases,
            "failed_checkpoints_top": failed_checkpoints,
            "root_cause": root_cause,
            "action": action,
            "summary": summary,
        }
        return payload

    def _extract_stage_info(self, meta: Dict, content: Dict) -> Dict:
        stage_info = {}
        for source in (content.get("stage_info"), meta):
            if not isinstance(source, dict):
                continue
            if isinstance(source.get("stage_index"), int):
                stage_info["stage_index"] = source.get("stage_index")
            if source.get("stage_title"):
                stage_info["stage_title"] = str(source.get("stage_title"))
            if source.get("section_index"):
                stage_info["section_index"] = str(source.get("section_index"))
            if source.get("progress"):
                stage_info["progress"] = str(source.get("progress"))
        return {
            "stage_index": stage_info.get("stage_index"),
            "stage_title": stage_info.get("stage_title", ""),
            "section_index": stage_info.get("section_index", ""),
            "progress": stage_info.get("progress", ""),
        }

    def _extract_tests(self, content: Dict) -> Dict:
        tests = content.get("tests") if isinstance(content.get("tests"), dict) else {}
        total = _safe_int(tests.get("total"))
        failed = _safe_int(tests.get("failed", tests.get("fails")))
        passed = _safe_int(tests.get("passed", max(0, total - failed)))
        if total == 0 and isinstance(content.get("test_report"), dict):
            report = content.get("test_report") or {}
            total = _safe_int(report.get("total"))
            failed = _safe_int(report.get("failed", report.get("fails")))
            passed = _safe_int(report.get("passed", max(0, total - failed)))
        return {"total": total, "passed": passed, "failed": failed}

    def _extract_failed_cases(self, content: Dict) -> List[str]:
        for key in ("failed_cases_top", "failed_test_cases_top"):
            block = content.get(key)
            if isinstance(block, list):
                return [str(item) for item in block]
        report = content.get("test_report")
        if isinstance(report, dict):
            block = report.get("failed_cases_top")
            if isinstance(block, list):
                cases = []
                for item in block:
                    if isinstance(item, dict):
                        case_name = item.get("case")
                        if case_name:
                            cases.append(str(case_name))
                    else:
                        cases.append(str(item))
                return cases
        return []

    def _extract_failed_checkpoints(self, content: Dict) -> List[str]:
        for key in ("failed_checkpoints_top", "failed_check_points_top", "failed_check_point_list_top"):
            block = content.get(key)
            if isinstance(block, list):
                return [str(item) for item in block]
        report = content.get("test_report")
        if isinstance(report, dict):
            block = report.get("failed_checkpoints_top")
            if isinstance(block, list):
                return [str(item) for item in block]
        coverage = content.get("coverage_status")
        if isinstance(coverage, dict):
            block = coverage.get("failed_check_point_list_top")
            if isinstance(block, list):
                return [str(item) for item in block]
        return []

    def _extract_root_cause(self, content: Dict) -> str:
        if isinstance(content.get("root_cause"), str) and content.get("root_cause").strip():
            return content.get("root_cause").strip()
        bug_tracking = content.get("bug_tracking")
        if isinstance(bug_tracking, dict):
            block = bug_tracking.get("root_cause_hypotheses_top")
            if isinstance(block, list):
                texts = []
                for item in block[:3]:
                    if isinstance(item, dict):
                        hypothesis = item.get("hypothesis")
                        if hypothesis:
                            texts.append(str(hypothesis))
                    elif item:
                        texts.append(str(item))
                if texts:
                    return " | ".join(texts)
        return ""

    def _extract_action(self, content: Dict) -> str:
        if isinstance(content.get("action"), str) and content.get("action").strip():
            return content.get("action").strip()
        next_info = content.get("next")
        if isinstance(next_info, dict):
            for key in ("todo", "blockers", "notes"):
                value = next_info.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ""

    def _infer_status(self, content: Dict, tests: Dict, failed_cases: List[str], failed_checkpoints: List[str]) -> str:
        if str(content.get("status", "")).lower() in ("failure", "mixed", "success"):
            return str(content.get("status")).lower()
        if content.get("run_test_success") is False:
            return "failure"
        if _safe_int(tests.get("failed")) > 0 or failed_cases or failed_checkpoints:
            return "failure"
        if _safe_int(tests.get("total")) > 0:
            return "success"
        return "mixed" if self._extract_root_cause(content) or self._extract_action(content) else "success"

    def _build_summary(
        self,
        kind: str,
        stage_info: Dict,
        tests: Dict,
        failed_cases: List[str],
        failed_checkpoints: List[str],
        root_cause: str,
        action: str,
    ) -> str:
        stage = stage_info.get("stage_title") or f"stage {stage_info.get('stage_index')}"
        parts = [f"{kind} @ {stage}".strip()]
        total = _safe_int(tests.get("total"))
        failed = _safe_int(tests.get("failed"))
        if total > 0:
            parts.append(f"tests={failed}/{total} failed")
        if failed_checkpoints:
            parts.append("checkpoints=" + ", ".join(failed_checkpoints[:4]))
        elif failed_cases:
            parts.append("cases=" + ", ".join(failed_cases[:3]))
        if root_cause:
            parts.append("root_cause=" + root_cause[:120])
        if action:
            parts.append("next=" + action[:120])
        return "; ".join(parts)

    def _compute_write_score(self, payload: Dict) -> float:
        kind = payload.get("kind", "")
        status = payload.get("status", "success")
        tests = payload.get("tests", {})
        failed_cases = payload.get("failed_cases_top", [])
        failed_checkpoints = payload.get("failed_checkpoints_top", [])
        root_cause = payload.get("root_cause", "")
        action = payload.get("action", "")
        stage_index = payload.get("stage_info", {}).get("stage_index")

        score = 0.0
        if status == "failure":
            score += 0.38
        elif status == "mixed":
            score += 0.22
        else:
            score += 0.06
        score += min(len(failed_checkpoints), 5) * 0.08
        score += min(len(failed_cases), 5) * 0.04
        if root_cause:
            score += 0.12
        if action:
            score += 0.08
        if isinstance(stage_index, int):
            score += 0.08
        if kind == "stage":
            score += 0.10
        elif kind == "batch":
            score += 0.06
        elif kind == "turn":
            score -= 0.04
        if self.skip_passed_singleton and kind == "batch" and status == "success" and _safe_int(tests.get("total")) <= 1:
            score -= 0.40
        return max(0.0, min(1.0, score))

    def _build_retrieval_text(self, payload: Dict) -> str:
        stage = payload.get("stage_info", {})
        parts = [
            str(payload.get("kind", "")),
            str(payload.get("status", "")),
            str(stage.get("stage_title", "")),
            str(stage.get("section_index", "")),
            str(payload.get("summary", "")),
            " ".join(payload.get("failed_checkpoints_top", [])[:8]),
            " ".join(payload.get("failed_cases_top", [])[:6]),
            str(payload.get("root_cause", "")),
            str(payload.get("action", "")),
        ]
        return _normalize_text(" ".join(part for part in parts if part))

    def _find_merge_target(self, candidate: Dict, entries: List[Dict]) -> Tuple[int, float]:
        best_idx = -1
        best_score = 0.0
        candidate_content = candidate.get("content", {})
        candidate_status = candidate_content.get("status", "success")
        candidate_stage = candidate.get("meta", {}).get("stage_index")
        for idx, entry in enumerate(entries):
            if entry.get("meta", {}).get("dut") != self.dut_name:
                continue
            current_content = entry.get("content", {})
            score = 0.0
            if entry.get("meta", {}).get("type") == candidate.get("meta", {}).get("type"):
                score += 0.10
            if entry.get("meta", {}).get("stage_index") == candidate_stage and candidate_stage is not None:
                score += 0.18
            score += _jaccard(
                current_content.get("failed_checkpoints_top", []), candidate_content.get("failed_checkpoints_top", [])
            ) * 0.50
            score += _jaccard(
                current_content.get("failed_cases_top", []), candidate_content.get("failed_cases_top", [])
            ) * 0.24
            if current_content.get("root_cause") and current_content.get("root_cause") == candidate_content.get("root_cause"):
                score += 0.08
            if current_content.get("summary") == candidate_content.get("summary"):
                score += 0.12
            if score > best_score:
                best_score = score
                best_idx = idx
        threshold = 0.50 if candidate_status == "failure" else 0.70
        if best_score < threshold:
            return -1, best_score
        return best_idx, best_score

    def _merge_entries(self, existing: Dict, incoming: Dict) -> Dict:
        ex = dict(existing)
        ex_content = dict(existing.get("content", {}))
        in_content = dict(incoming.get("content", {}))

        ex_content["status"] = (
            ex_content.get("status")
            if _status_rank(ex_content.get("status")) >= _status_rank(in_content.get("status"))
            else in_content.get("status")
        )
        ex_content["failed_cases_top"] = _dedup_keep_order(
            list(ex_content.get("failed_cases_top", [])) + list(in_content.get("failed_cases_top", [])), 10
        )
        ex_content["failed_checkpoints_top"] = _dedup_keep_order(
            list(ex_content.get("failed_checkpoints_top", [])) + list(in_content.get("failed_checkpoints_top", [])), 12
        )
        if in_content.get("root_cause") and (
            not ex_content.get("root_cause") or len(in_content.get("root_cause")) > len(ex_content.get("root_cause", ""))
        ):
            ex_content["root_cause"] = in_content.get("root_cause")
        if in_content.get("action") and (
            not ex_content.get("action") or len(in_content.get("action")) > len(ex_content.get("action", ""))
        ):
            ex_content["action"] = in_content.get("action")
        if len(in_content.get("summary", "")) > len(ex_content.get("summary", "")):
            ex_content["summary"] = in_content.get("summary")

        ex["content"] = ex_content
        ex["last_seen"] = incoming.get("last_seen", time.time())
        ex["support"] = int(ex.get("support", 1) or 1) + 1
        ex["level"] = self._decide_level(ex["support"])
        ex["write_score"] = round(max(float(ex.get("write_score", 0.0) or 0.0), float(incoming.get("write_score", 0.0) or 0.0)), 4)
        ex["retrieval_text"] = self._build_retrieval_text(ex_content)
        ex["hit_count"] = int(ex.get("hit_count", 0) or 0) + int(incoming.get("hit_count", 0) or 0)
        ex["reuse_count"] = int(ex.get("reuse_count", 0) or 0) + int(incoming.get("reuse_count", 0) or 0)
        ex["useful_count"] = int(ex.get("useful_count", 0) or 0) + int(incoming.get("useful_count", 0) or 0)
        ex["pollution_count"] = int(ex.get("pollution_count", 0) or 0) + int(incoming.get("pollution_count", 0) or 0)
        ex["stale_score"] = round(max(float(ex.get("stale_score", 0.0) or 0.0), float(incoming.get("stale_score", 0.0) or 0.0)), 4)
        return ex

    def _decide_level(self, support: int) -> str:
        if support >= self.semantic_support:
            return "semantic"
        if support >= self.episode_support:
            return "episode"
        return "candidate"

    def _apply_filters(self, entries: List[Dict], filters: Optional[Dict]) -> List[Dict]:
        if not filters:
            return entries
        out = []
        for entry in entries:
            meta = entry.get("meta", {})
            ok = True
            for key, value in filters.items():
                if meta.get(key) != value:
                    ok = False
                    break
            if ok:
                out.append(entry)
        return out

    def _lexical_score(self, query_tokens: List[str], entry: Dict) -> float:
        if not query_tokens:
            return 0.0
        hay = entry.get("retrieval_text", "")
        if not hay:
            return 0.0
        hit = sum(1 for token in query_tokens if token in hay)
        return hit / max(1, len(set(query_tokens)))

    def _touch_memory_changed(self) -> None:
        self._version += 1
        self._vector_dirty = True
        self._vector_cache = None

    def _maybe_preload_from_completed(self) -> None:
        if not self.preload_from_completed:
            return
        current_entries = self._load_jsonl(self.path)
        if current_entries:
            return
        candidates = []
        patterns = [
            os.path.join(self.base_dir, "memory.completed.jsonl"),
            os.path.join(self.base_dir, "memory.completed.*.jsonl"),
        ]
        for pattern in patterns:
            candidates.extend(glob.glob(pattern))
        candidates = [p for p in candidates if os.path.isfile(p)]
        if not candidates:
            return
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        latest = candidates[0]
        entries = self._load_jsonl(latest)
        if not entries:
            return
        entries.sort(
            key=lambda item: (
                {"semantic": 3, "episode": 2, "candidate": 1}.get(item.get("level"), 0),
                int(item.get("support", 1) or 1),
                float(item.get("last_seen", 0.0) or 0.0),
            ),
            reverse=True,
        )
        preloaded = entries[: self.preload_keep]
        self._write_jsonl(self.path, preloaded)
        info(
            f"[long_term_memory][preload] loaded {len(preloaded)} entries from {os.path.basename(latest)}"
        )
        self._touch_memory_changed()

    def _append_jsonl(self, path: str, item: Dict) -> None:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _load_jsonl(self, path: str) -> List[Dict]:
        if not os.path.exists(path):
            return []
        items = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        items.append(obj)
                except Exception:
                    continue
        return items

    def _write_jsonl(self, path: str, items: List[Dict]) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _log_metric_event(self, event_type: str, payload: Dict) -> None:
        record = {
            "ts": time.time(),
            "event": event_type,
            "dut": self.dut_name,
            "payload": payload or {},
        }
        try:
            self._append_jsonl(self.metrics_path, record)
        except Exception:
            pass

    def _prune_jsonl(self, path: str, keep: int) -> None:
        items = self._load_jsonl(path)
        if len(items) <= keep:
            return
        self._write_jsonl(path, items[-keep:])

    def _get_embedder(self):
        if not self.enable_embed or OpenAIEmbeddings is None:
            return None
        if self._embedder is None:
            model = self.embed_config.get("model_name")
            base_url = self.embed_config.get("openai_api_base")
            api_key = self.embed_config.get("openai_api_key")
            if not model or not base_url:
                return None
            try:
                self._embedder = OpenAIEmbeddings(model=model, base_url=base_url, api_key=api_key)
            except Exception:
                self._embedder = None
        return self._embedder

    def _ensure_vector_index(self, entries: List[Dict]) -> None:
        if not self.enable_embed or self._get_embedder() is None:
            return
        if not self._vector_dirty and self._vector_cache is not None:
            return
        data = []
        embedder = self._get_embedder()
        for entry in entries:
            try:
                vec = embedder.embed_query(entry.get("retrieval_text", ""))
            except Exception:
                vec = []
            data.append(
                {
                    "hash": entry.get("hash"),
                    "vec": vec,
                    "meta": entry.get("meta", {}),
                    "ts": entry.get("last_seen"),
                }
            )
        self._write_jsonl(self.vector_path, data)
        self._vector_cache = data
        self._vector_dirty = False

    def _search_embed_scores(self, query: str) -> Dict[str, float]:
        if self._vector_cache is None:
            self._vector_cache = self._load_jsonl(self.vector_path)
        embedder = self._get_embedder()
        if embedder is None or not self._vector_cache:
            return {}
        try:
            query_vec = embedder.embed_query(query)
        except Exception:
            return {}
        scores = {}
        for item in self._vector_cache:
            score = _cosine_similarity(query_vec, item.get("vec", []))
            scores[item.get("hash")] = score
        return scores


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2:
        return 0.0
    n = min(len(v1), len(v2))
    if n == 0:
        return 0.0
    dot = sum(v1[i] * v2[i] for i in range(n))
    norm1 = math.sqrt(sum(v1[i] * v1[i] for i in range(n)))
    norm2 = math.sqrt(sum(v2[i] * v2[i] for i in range(n)))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def _rename_with_fallback(src: str, dst: str) -> Optional[str]:
    if not os.path.exists(src):
        return None
    if os.path.exists(dst):
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        dst = dst.replace(".jsonl", f".{ts}.jsonl")
    try:
        os.rename(src, dst)
        return dst
    except Exception:
        try:
            import shutil

            shutil.copy(src, dst)
            os.remove(src)
            return dst
        except Exception:
            return None
