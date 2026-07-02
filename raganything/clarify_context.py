"""Clarify gate v4: KG filtering, context rebuild, and cached query bundles."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.prompt import PROMPTS


def split_source_ids(source_id: str | None) -> set[str]:
    if not source_id:
        return set()
    return {part.strip() for part in str(source_id).split(GRAPH_FIELD_SEP) if part.strip()}


def filter_kg_by_chunk_ids(
    entities: list[dict[str, Any]] | None,
    relationships: list[dict[str, Any]] | None,
    chunk_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep entity/relation rows whose ``source_id`` intersects surviving chunk ids."""
    if not chunk_ids:
        return [], []

    filtered_entities: list[dict[str, Any]] = []
    for entity in entities or []:
        if not isinstance(entity, dict):
            continue
        if split_source_ids(entity.get("source_id")) & chunk_ids:
            filtered_entities.append(entity)

    filtered_relations: list[dict[str, Any]] = []
    for relation in relationships or []:
        if not isinstance(relation, dict):
            continue
        if split_source_ids(relation.get("source_id")) & chunk_ids:
            filtered_relations.append(relation)

    return filtered_entities, filtered_relations


def chunk_ids_from_llm_chunks(chunks: list[dict[str, Any]] | None) -> set[str]:
    ids: set[str] = set()
    for chunk in chunks or []:
        if not isinstance(chunk, dict):
            continue
        cid = chunk.get("chunk_id") or chunk.get("id")
        if cid:
            ids.add(str(cid))
    return ids


def scope_kg_to_llm_chunks(
    raw_data: dict[str, Any] | None,
    context_str: str | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Keep only KG rows backed by surviving LLM document chunks; rebuild context."""
    if not isinstance(raw_data, dict):
        return context_str, raw_data

    data = raw_data.get("data")
    if not isinstance(data, dict):
        return context_str, raw_data

    chunks = [row for row in (data.get("chunks") or []) if isinstance(row, dict)]
    references = [row for row in (data.get("references") or []) if isinstance(row, dict)]
    entities = [row for row in (data.get("entities") or []) if isinstance(row, dict)]
    relationships = [
        row for row in (data.get("relationships") or []) if isinstance(row, dict)
    ]

    chunk_ids = chunk_ids_from_llm_chunks(chunks)
    entities_filtered, relationships_filtered = filter_kg_by_chunk_ids(
        entities, relationships, chunk_ids
    )

    raw_filtered = copy.deepcopy(raw_data)
    filtered_data = raw_filtered.setdefault("data", {})
    filtered_data["entities"] = entities_filtered
    filtered_data["relationships"] = relationships_filtered
    filtered_data["chunks"] = chunks
    filtered_data["references"] = references

    rebuilt_context = rebuild_kg_context_str(
        entities_filtered,
        relationships_filtered,
        chunks,
        references,
    )
    final_context = rebuilt_context.strip() or (context_str or "").strip() or None
    return final_context, raw_filtered


def rebuild_kg_context_str(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> str:
    """Rebuild ``kg_query_context`` body (Document Chunks + filtered KG + references)."""
    entities_str = "\n".join(
        json.dumps(entity, ensure_ascii=False) for entity in entities
    )
    relations_str = "\n".join(
        json.dumps(relation, ensure_ascii=False) for relation in relationships
    )

    chunks_context: list[dict[str, Any]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        entry: dict[str, Any] = {
            "reference_id": chunk.get("reference_id", ""),
            "content": chunk.get("content", ""),
        }
        if chunk.get("content_headings"):
            entry["content_headings"] = chunk["content_headings"]
        chunks_context.append(entry)

    text_units_str = "\n".join(
        json.dumps(text_unit, ensure_ascii=False) for text_unit in chunks_context
    )
    reference_list_str = "\n".join(
        f"[{ref['reference_id']}] {ref['file_path']}"
        for ref in references
        if isinstance(ref, dict) and ref.get("reference_id")
    )

    template = PROMPTS["kg_query_context"]
    return template.format(
        entities_str=entities_str,
        relations_str=relations_str,
        text_chunks_str=text_units_str,
        reference_list_str=reference_list_str,
    )


@dataclass
class CachedQueryBundle:
    query: str
    context_str: str
    raw_data: dict[str, Any]
    document_chunks: list[dict[str, Any]]
    entities_filtered: list[dict[str, Any]]
    relationships_filtered: list[dict[str, Any]]
    reference_list: list[dict[str, Any]]
    chunk_ids: set[str] = field(default_factory=set)

    def to_injection(self) -> dict[str, Any]:
        return {
            "context_str": self.context_str,
            "raw_data": self.raw_data,
        }


@dataclass
class RetrievalProbeResult:
    query: str
    mode: str
    answerable: bool
    chunk_count: int
    max_rerank_score: float | None
    min_rerank_threshold: float
    llm_chunk_total: int
    scores_unavailable: bool
    final_score: float | None = None
    direct_rerank_min: float | None = None
    max_rerank_any: float | None = None
    bundle: CachedQueryBundle | None = None

    def as_stats(self) -> dict[str, Any]:
        return {
            "answerable": self.answerable,
            "chunk_count": self.chunk_count,
            "final_score": self.final_score,
            "max_rerank_score": self.max_rerank_score,
            "max_rerank_any": self.max_rerank_any,
            "min_rerank_threshold": self.min_rerank_threshold,
            "direct_rerank_min": self.direct_rerank_min,
            "mode": self.mode,
            "llm_chunk_total": self.llm_chunk_total,
            "scores_unavailable": self.scores_unavailable,
            "gate_version": "v4",
        }


def build_cached_bundle(
    query: str,
    *,
    context_str: str | None,
    raw_data: dict[str, Any] | None,
) -> CachedQueryBundle | None:
    if not isinstance(raw_data, dict):
        return None

    data = raw_data.get("data")
    if not isinstance(data, dict):
        return None

    chunks = [
        chunk for chunk in (data.get("chunks") or []) if isinstance(chunk, dict)
    ]
    references = [
        ref for ref in (data.get("references") or []) if isinstance(ref, dict)
    ]

    final_context, raw_filtered = scope_kg_to_llm_chunks(raw_data, context_str)
    if not isinstance(raw_filtered, dict):
        return None

    filtered_data = raw_filtered.get("data") or {}
    entities_filtered = filtered_data.get("entities") or []
    relationships_filtered = filtered_data.get("relationships") or []

    return CachedQueryBundle(
        query=(query or "").strip(),
        context_str=final_context or "",
        raw_data=raw_filtered,
        document_chunks=chunks,
        entities_filtered=entities_filtered,
        relationships_filtered=relationships_filtered,
        reference_list=references,
        chunk_ids=chunk_ids_from_llm_chunks(chunks),
    )


def format_chunk_previews(chunks: list[dict[str, Any]], *, limit: int = 3) -> str:
    parts: list[str] = []
    for chunk in chunks[:limit]:
        if not isinstance(chunk, dict):
            continue
        content = (chunk.get("content") or "").strip()
        if content:
            parts.append(content[:400])
    return "\n---\n".join(parts)
