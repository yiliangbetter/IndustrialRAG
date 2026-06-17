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


def _serialize_docs(docs: list[dict[str, Any]] | None, *, limit: int = 24) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for doc in (docs or [])[:limit]:
        if not isinstance(doc, dict):
            continue
        row: dict[str, Any] = {}
        for key in ("file_path", "source", "title", "rerank_score", "score", "id"):
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
