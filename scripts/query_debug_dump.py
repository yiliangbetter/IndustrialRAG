"""Structured query debug dumps when ``RAG_QUERY_DEBUG_DUMP=1``."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from client_paths import get_app_root, get_logs_dir

_SCHEMA = "rag_query_dump_v2"
_MAX_TEXT = 12000
_MAX_DOC_BODY = 4000
_MAX_ANSWER = 8000
_MAX_ENTITY_DESC = 600
_MAX_RELATION_DESC = 600
_MAX_LLM_ENTITIES = 80
_MAX_LLM_RELATIONS = 80
_MAX_LLM_CHUNKS = 32


def is_query_debug_enabled() -> bool:
    return (os.getenv("RAG_QUERY_DEBUG_DUMP") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def get_query_dump_dir() -> Path:
    directory = get_logs_dir() / "query_dumps"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _truncate(text: str | None, limit: int = _MAX_TEXT) -> str | None:
    if text is None:
        return None
    value = str(text)
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n... [truncated {len(value) - limit} chars]"


def _serialize_entities(
    entities: list[dict[str, Any]] | None, *, limit: int = _MAX_LLM_ENTITIES
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in (entities or [])[:limit]:
        if not isinstance(item, dict):
            continue
        row: dict[str, Any] = {}
        for key in ("entity_name", "entity_type", "file_path", "source_id"):
            val = item.get(key)
            if val not in (None, ""):
                row[key] = val
        desc = item.get("description")
        if isinstance(desc, str) and desc.strip():
            row["description"] = _truncate(desc.strip(), _MAX_ENTITY_DESC)
        rows.append(row)
    return rows


def _serialize_relationships(
    relations: list[dict[str, Any]] | None, *, limit: int = _MAX_LLM_RELATIONS
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in (relations or [])[:limit]:
        if not isinstance(item, dict):
            continue
        row: dict[str, Any] = {}
        for key in ("src_id", "tgt_id", "keywords", "weight", "file_path", "source_id"):
            val = item.get(key)
            if val not in (None, ""):
                row[key] = val
        desc = item.get("description")
        if isinstance(desc, str) and desc.strip():
            row["description"] = _truncate(desc.strip(), _MAX_RELATION_DESC)
        rows.append(row)
    return rows


def _serialize_references(
    references: list[dict[str, Any]] | None, *, limit: int = _MAX_LLM_CHUNKS
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in (references or [])[:limit]:
        if not isinstance(item, dict):
            continue
        row: dict[str, Any] = {}
        for key in ("reference_id", "file_path"):
            val = item.get(key)
            if val not in (None, ""):
                row[key] = val
        rows.append(row)
    return rows


def build_llm_input_snapshot(
    raw_data: dict[str, Any] | None,
    *,
    context_str: str | None = None,
) -> dict[str, Any]:
    """Structured KG + document chunks actually bound for the answer LLM."""
    data = raw_data.get("data") if isinstance(raw_data, dict) else None
    if not isinstance(data, dict):
        data = {}

    entities = data.get("entities") if isinstance(data.get("entities"), list) else []
    relationships = (
        data.get("relationships") if isinstance(data.get("relationships"), list) else []
    )
    chunks = data.get("chunks") if isinstance(data.get("chunks"), list) else []
    references = data.get("references") if isinstance(data.get("references"), list) else []

    meta = raw_data.get("metadata") if isinstance(raw_data, dict) else None
    processing = (
        meta.get("processing_info") if isinstance(meta, dict) else None
    )

    snapshot: dict[str, Any] = {
        "context_chars": len(context_str) if isinstance(context_str, str) else 0,
        "counts": {
            "entities": len(entities),
            "relationships": len(relationships),
            "chunks": len(chunks),
            "references": len(references),
        },
        "entities": _serialize_entities(entities),
        "relationships": _serialize_relationships(relationships),
        "chunks": _serialize_docs(chunks, limit=_MAX_LLM_CHUNKS),
        "references": _serialize_references(references),
    }
    if isinstance(processing, dict) and processing:
        snapshot["processing"] = processing
    if isinstance(meta, dict) and meta.get("keywords"):
        snapshot["keywords"] = meta.get("keywords")
    return snapshot


def _serialize_docs(docs: list[dict[str, Any]] | None, *, limit: int = 24) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for doc in (docs or [])[:limit]:
        if not isinstance(doc, dict):
            continue
        row: dict[str, Any] = {}
        for key in (
            "file_path",
            "source",
            "title",
            "rerank_score",
            "score",
            "id",
            "reference_id",
            "chunk_id",
        ):
            if key in doc:
                row[key] = doc[key]
        for key in ("content", "text", "chunk_content", "page_content"):
            val = doc.get(key)
            if isinstance(val, str) and val.strip():
                row["content"] = _truncate(val, _MAX_DOC_BODY)
                break
        rows.append(row)
    return rows


def _slug_query(query: str) -> str:
    slug = re.sub(r"[^\w\u4e00-\u9fff]+", "_", (query or "").strip())[:40]
    return slug.strip("_") or "query"


def build_query_dump(
    *,
    query: str,
    mode: str,
    thinking: str | None = None,
    answer: str | None = None,
    error: str | None = None,
    retrieval_context: str | None = None,
    retrieved_docs: list[dict[str, Any]] | None = None,
    retrieved_docs_text: str | None = None,
    related_images: list[dict[str, Any]] | None = None,
    images_debug: dict[str, Any] | None = None,
    steering_report: dict[str, Any] | None = None,
    naive_relevance: dict[str, Any] | None = None,
    clarify_gate: dict[str, Any] | None = None,
    llm_input: dict[str, Any] | None = None,
    duration_ms: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": _SCHEMA,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "app_root": str(get_app_root()),
        "query": query,
        "mode": mode,
        "duration_ms": duration_ms,
        "error": error,
        "answer": _truncate(answer, _MAX_ANSWER),
        "thinking": _truncate(thinking, _MAX_ANSWER),
        "retrieval": {
            "context": _truncate(retrieval_context),
            "docs_text": _truncate(retrieved_docs_text),
            "docs": _serialize_docs(retrieved_docs),
        },
        "images": {
            "selected": related_images or [],
            "debug": images_debug or {},
        },
        "steering": steering_report or {},
    }
    if isinstance(naive_relevance, dict) and naive_relevance:
        payload["naive_relevance"] = naive_relevance
    if isinstance(clarify_gate, dict) and clarify_gate:
        payload["clarify_gate"] = clarify_gate
    if isinstance(llm_input, dict) and llm_input:
        payload["llm_input"] = llm_input
    return payload


def write_query_dump(payload: dict[str, Any]) -> Path:
    dump_dir = get_query_dump_dir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{_slug_query(str(payload.get('query') or ''))}_{uuid.uuid4().hex[:8]}.json"
    path = dump_dir / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def list_recent_dumps(*, limit: int = 15) -> list[dict[str, Any]]:
    dump_dir = get_query_dump_dir()
    if not dump_dir.is_dir():
        return []
    files = sorted(dump_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    rows: list[dict[str, Any]] = []
    for path in files[:limit]:
        try:
            stat = path.stat()
        except OSError:
            continue
        rows.append(
            {
                "name": path.name,
                "path": str(path),
                "size_bytes": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            }
        )
    return rows
