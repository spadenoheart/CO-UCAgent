# -*- coding: utf-8 -*-
"""Lightweight long-term memory store (jsonl)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

from ucagent.util.log import info

try:
    from langchain_openai import OpenAIEmbeddings
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAIEmbeddings = None


def _normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def _tokenize(text: str) -> List[str]:
    tokens = []
    buf = []
    for ch in (text or ""):
        if ch.isalnum() or ch in ("_", "-", "/"):
            buf.append(ch.lower())
        else:
            if buf:
                tokens.append("".join(buf))
                buf = []
    if buf:
        tokens.append("".join(buf))
    return [t for t in tokens if len(t) >= 3]


class LongTermMemoryStore:
    """Append-only jsonl store with simple keyword search and optional embeddings."""

    def __init__(
        self,
        workspace: str,
        dut_name: str,
        max_entries: int = 2000,
        enable_embed: bool = False,
        embed_config: Optional[Dict] = None,
    ):
        self.workspace = os.path.abspath(workspace)
        self.dut_name = dut_name
        self.max_entries = max_entries
        self.enable_embed = enable_embed
        self.embed_config = embed_config or {}
        self.base_dir = os.path.join(self.workspace, ".ucagent_memory", dut_name)
        self.path = os.path.join(self.base_dir, "memory.jsonl")
        self.vector_path = os.path.join(self.base_dir, "memory.emb.jsonl")
        self._seen_hash = set()
        self._embedder = None
        self._vector_cache: Optional[List[Dict]] = None
        os.makedirs(self.base_dir, exist_ok=True)
        self._load_seen_hashes()

    def _load_seen_hashes(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        h = obj.get("hash")
                        if h:
                            self._seen_hash.add(h)
                    except Exception:
                        continue
        except Exception:
            return

    def _make_hash(self, content: str) -> str:
        return hashlib.sha1(content.encode("utf-8", errors="ignore")).hexdigest()

    def save(self, meta: Dict, content: Dict | str) -> bool:
        """Save a memory entry. Returns True if written."""
        payload = {
            "meta": meta or {},
            "content": content,
            "timestamp": meta.get("timestamp") if isinstance(meta, dict) else None,
        }
        if not payload["timestamp"]:
            payload["timestamp"] = time.time()
        content_text = json.dumps(content, ensure_ascii=False, sort_keys=True) if not isinstance(content, str) else content
        payload["hash"] = self._make_hash(content_text)
        if payload["hash"] in self._seen_hash:
            return False
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._seen_hash.add(payload["hash"])
            self._prune_if_needed()
            self._maybe_save_embedding(payload)
            return True
        except Exception:
            return False

    def _iter_entries(self) -> Iterable[Dict]:
        if not os.path.exists(self.path):
            return []
        def _gen():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue
        return _gen()

    def _prune_if_needed(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) <= self.max_entries:
                return
            keep = lines[-self.max_entries :]
            with open(self.path, "w", encoding="utf-8") as f:
                f.writelines(keep)
        except Exception:
            return

    def recent(self, limit: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        entries = list(self._iter_entries())
        if filters:
            entries = [e for e in entries if _match_filters(e, filters)]
        return entries[-limit:]

    def search(self, query: str, limit: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        if self.enable_embed and self._get_embedder() is not None:
            emb_hits = self._search_by_embedding(query, limit=limit, filters=filters)
            if emb_hits:
                return emb_hits
        # fallback to keyword search
        tokens = _tokenize(query)
        if not tokens:
            return self.recent(limit=limit, filters=filters)
        scored = []
        for e in self._iter_entries():
            if filters and not _match_filters(e, filters):
                continue
            hay = _normalize_text(json.dumps(e, ensure_ascii=False))
            score = sum(1 for t in tokens if t in hay)
            if score > 0:
                scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:limit]]

    def clear(self) -> None:
        if os.path.exists(self.path):
            try:
                os.remove(self.path)
            except Exception:
                pass
        if os.path.exists(self.vector_path):
            try:
                os.remove(self.vector_path)
            except Exception:
                pass

    def archive_completed(self) -> None:
        """Rename memory.jsonl (and embedding file) for archival."""
        dst_main = os.path.join(self.base_dir, "memory.completed.jsonl")
        dst_vec = os.path.join(self.base_dir, "memory.emb.completed.jsonl")
        archived_main = _rename_with_fallback(self.path, dst_main)
        archived_vec = _rename_with_fallback(self.vector_path, dst_vec)
        if archived_main:
            info(f"[long_term_memory] archived {archived_main}")
        if archived_vec:
            info(f"[long_term_memory] archived {archived_vec}")

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

    def _maybe_save_embedding(self, payload: Dict) -> None:
        if not self.enable_embed or self._get_embedder() is None:
            return
        try:
            text = _normalize_text(json.dumps(payload, ensure_ascii=False))
            vec = self._get_embedder().embed_query(text)
            item = {"hash": payload.get("hash"), "vec": vec, "meta": payload.get("meta", {}), "ts": payload.get("timestamp")}
            with open(self.vector_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            self._vector_cache = None
        except Exception:
            return

    def _load_vector_cache(self) -> List[Dict]:
        if self._vector_cache is not None:
            return self._vector_cache
        if not os.path.exists(self.vector_path):
            self._vector_cache = []
            return self._vector_cache
        data = []
        with open(self.vector_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except Exception:
                    continue
        self._vector_cache = data
        return data

    def _search_by_embedding(self, query: str, limit: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        vecs = self._load_vector_cache()
        if not vecs:
            return []
        embedder = self._get_embedder()
        if embedder is None:
            return []
        query_vec = embedder.embed_query(query)
        scored: List[Tuple[float, Dict]] = []
        for item in vecs:
            if filters and not _match_filters(item.get("meta", {}), filters):
                continue
            score = _cosine_similarity(query_vec, item.get("vec", []))
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_hashes = [item.get("hash") for _, item in scored[:limit] if item.get("hash")]
        if not top_hashes:
            return []
        # fetch entries from jsonl by hash
        results = []
        for e in self._iter_entries():
            if e.get("hash") in top_hashes:
                results.append(e)
        # preserve order by top_hashes
        order = {h: i for i, h in enumerate(top_hashes)}
        results.sort(key=lambda x: order.get(x.get("hash"), 0))
        info(
            f"[long_term_memory][embed_search] query='{query[:80]}' hits={len(results)}"
        )
        return results[:limit]


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


def _match_filters(entry: Dict, filters: Dict) -> bool:
    if not filters:
        return True
    for k, v in filters.items():
        if isinstance(entry, dict) and entry.get("meta") and isinstance(entry.get("meta"), dict):
            meta = entry["meta"]
        else:
            meta = entry
        if meta.get(k) != v:
            return False
    return True


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
