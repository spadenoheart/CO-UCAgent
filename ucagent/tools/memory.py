# -*- coding: utf-8 -*-
"""Memory management tools for UCAgent."""

from mem0 import Memory
import time
from .uctool import UCTool
from langchain_core.tools.base import ArgsSchema

from typing import Optional, List, Union
from pydantic import BaseModel, Field

import os
from ucagent.util.log import info, warning, error


from langgraph.store.memory import InMemoryStore
from langchain_openai import OpenAIEmbeddings
from langmem import utils


class ArgsMemSearch(BaseModel):
    """Arguments for memory search"""
    query: str = Field(..., description="The query string to search")
    limit: int = Field(3, description="The maximum number of results to return, default 3", ge=1, le=100)


def new_embed(config) -> dict:
    info(f"Creating new embedding with model: {config['model_name']}, base_url: {config['openai_api_base']}")
    return {"embed":OpenAIEmbeddings(model   = config["model_name"],
                            base_url= config["openai_api_base"],
                            api_key = config["openai_api_key"]),
            "dims":config["dims"]
            }

class SemanticSearchInGuidDoc(UCTool):
    """Semantic search in the guild documentation for verification definitions and examples."""
    name: str = "SemanticSearchInGuidDoc"
    description: str = (
        "Semantic search in the guild documentation for verification definitions and examples. "
    )
    args_schema: Optional[ArgsSchema] = ArgsMemSearch

    # custom variables
    workspace: str = Field(str, description="The workspace directory to search in")
    doc_path: str = Field(str, description="The path to the documentation directory relative to the workspace")
    store: InMemoryStore = Field(InMemoryStore, description="In-memory store for document references")
    namespace: utils.NamespaceTemplate = Field(utils.NamespaceTemplate,
        description="Namespace template for document references"
    )
    rerank_enabled: bool = Field(False, description="Whether to apply lightweight rerank after semantic search")
    disabled: bool = Field(False, description="Set true if initialization failed")
    disable_reason: str = Field("", description="Reason for disable")

    def __init__(self, config, workspace, doc_path, file_extension: List[str] = [".md", ".py", ".v"], rerank_enabled: bool = False):
        super().__init__()
        self.namespace = utils.NamespaceTemplate("doc_reference")
        self.store = None
        self.rerank_enabled = rerank_enabled
        self.workspace = os.path.abspath(workspace)
        self.doc_path = os.path.abspath(os.path.join(workspace, doc_path))
        assert os.path.exists(self.doc_path), f"Doc path {self.doc_path} does not exist."
        info(f"Initializing SearchInGuidDoc with workspace: {self.workspace}, doc_path: {self.doc_path}")
        try:
            self.store = InMemoryStore(
                index=new_embed(config)
            )
            for root, _, files in os.walk(self.doc_path):
                for file in files:
                    if any(file.endswith(ext) for ext in file_extension):
                        file_path = os.path.abspath(os.path.join(root, file)).removeprefix(self.workspace + os.sep)
                        self.store.put(self.namespace(),
                                       key=str(file_path),
                                       value={
                                           "content": open(os.path.join(root, file), 'r', encoding='utf-8').read(),
                                           "path": file_path,
                                           "source": "guide_doc",
                                           "timestamp": time.time(),
                                       }),
                        info(f"Added file {file_path} to memory.")
        except Exception as e:
            self.disabled = True
            self.disable_reason = f"embedding init failed: {e}"
            warning(f"SemanticSearchInGuidDoc disabled: {self.disable_reason}")

    def _rerank(self, memories: list) -> list:
        """Lightweight rerank: similarity/score + recency bonus if available."""
        def _score(m):
            base = m.get("score") or m.get("similarity") or 0.0
            ts = None
            value = m.get("value") or {}
            if isinstance(value, dict):
                ts = value.get("timestamp")
            recency = 0.0
            if ts:
                # simple recency bonus with half-life ~1 day
                age_hours = max(0.0, (time.time() - ts) / 3600.0)
                recency = 0.1 / (1.0 + age_hours / 24.0)
            return base + recency
        try:
            return sorted(memories, key=_score, reverse=True)
        except Exception:
            return memories

    def _run(self, query: str, limit: int = 3, run_manager = None) -> str:
        if self.disabled or self.store is None:
            warning(f"SemanticSearchInGuidDoc skipped because disabled ({self.disable_reason})")
            return utils.dumps([])
        memories = self.store.search(
            self.namespace(),
            query=query,
            filter=None,
            limit=limit,
            offset=0,
        )
        mem_dicts = [m.dict() for m in memories]
        if self.rerank_enabled:
            try:
                preview_before = []
                for m in mem_dicts[: min(3, len(mem_dicts))]:
                    value = m.get("value") or {}
                    preview_before.append(
                        {
                            "path": value.get("path", ""),
                            "score": m.get("score") or m.get("similarity") or 0.0,
                        }
                    )
                info(
                    f"[context_upgrade][rerank] query='{query[:80]}' "
                    f"limit={limit} hits={len(mem_dicts)} preview_before={preview_before}"
                )
            except Exception:
                pass
            mem_dicts = self._rerank(mem_dicts)
            try:
                preview_after = []
                for m in mem_dicts[: min(3, len(mem_dicts))]:
                    value = m.get("value") or {}
                    preview_after.append(
                        {
                            "path": value.get("path", ""),
                            "score": m.get("score") or m.get("similarity") or 0.0,
                        }
                    )
                info(f"[context_upgrade][rerank] preview_after={preview_after}")
            except Exception:
                pass
        return utils.dumps(mem_dicts)


class ArgsMemoryPut(BaseModel):
    """Arguments for MemoryPut"""
    scope: str = Field(..., description="The scope of the memory, e.g., 'general', 'task-specific'")
    data: str = Field(..., description="The content to save in memory, can be a string or a JSON object")


class ArgsMemoryGet(BaseModel):
    """Arguments for MemoryGet"""
    scope: str = Field(..., description="The scope of the memory, e.g., 'general', 'task-specific'")
    query: str = Field(..., description="The query string to search")
    limit: int = Field(3, description="The maximum number of results to return, default 3", ge=1, le=100)


class MemoryTool(UCTool):
    store: InMemoryStore = Field(None, description="In-memory store for document references")
    def set_store(self, config=None, store: Optional[InMemoryStore] = None):
        super().__init__()
        if store is not None:
            self.store = store
        else:
            self.store = InMemoryStore(
                index=new_embed(config)
            )
        return self

    def get_store(self):
        """Get the in-memory store."""
        return self.store


class MemoryPut(MemoryTool):
    """Save important information to long-term memory."""
    name: str = "MemoryPut"
    description: str = (
        "Save important information to long-term memory. "
        "This tool allows you to store content in the memory store for future reference. "
        "The content can be a string or a JSON object. "
        "You can specify the scope of the memory, such as 'general' or 'task-specific'. "
        "The content will be saved in the memory store and can be retrieved later using the MemoryGet tool."
    )
    args_schema: Optional[ArgsSchema] = ArgsMemoryPut

    def _run(self, scope: str, data, run_manager = None) -> str:
        """Save the content to the memory store."""
        key = str(time.time_ns())
        self.store.put(
            utils.NamespaceTemplate(scope)(),
            key=key,  # Use a unique key based on the current time
            value={
                "content": data,
                "scope": scope,
                "timestamp": time.time(),
                "source": "memory_put",
            }
        )
        return f"Content saved to memory under scope '{scope}' with key '{key}' complete. "


class MemoryGet(MemoryTool):
    """Retrieve information from long-term memory."""
    name: str = "MemoryGet"
    description: str = (
        "Retrieve information from long-term memory. "
        "This tool allows you to search for content in the memory store based on a query. "
        "You can specify the scope of the memory, such as 'general' or 'task-specific'. "
        "The tool will return the most relevant content based on the query."
    )
    args_schema: Optional[ArgsSchema] = ArgsMemoryGet

    def _run(self, scope: str, query: str, limit: int = 3, run_manager = None) -> str:
        """Search for content in the memory store."""
        memories = self.store.search(
            utils.NamespaceTemplate(scope)(),
            query=query,
            filter=None,
            limit=limit,
            offset=0,
        )
        return utils.dumps([m.dict() for m in memories])
