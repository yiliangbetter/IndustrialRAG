"""Monkeypatch LightRAG query stages to emit real progress events (Web UI SSE)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextvars import ContextVar
from pathlib import Path
from typing import Any

PHASE_RETRIEVE = "retrieve"
PHASE_RERANK = "rerank"
PHASE_GENERATE = "generate"

PHASE_TEXT = {
    PHASE_RETRIEVE: "正在检索知识库…",
    PHASE_RERANK: "正在重排序文档片段…",
    PHASE_GENERATE: "正在生成回答…",
}

_progress_queue: ContextVar[asyncio.Queue[dict[str, str]] | None] = ContextVar(
    "progress_queue", default=None
)
_retrieval_context: ContextVar[str | None] = ContextVar("retrieval_context", default=None)
_retrieved_docs_text: ContextVar[str | None] = ContextVar("retrieved_docs_text", default=None)
_media_roots: ContextVar[list[Path] | None] = ContextVar("media_roots", default=None)
_query_text: ContextVar[str | None] = ContextVar("query_text", default=None)
_retrieved_docs: ContextVar[list[dict] | None] = ContextVar("retrieved_docs", default=None)
_images_emitted: ContextVar[bool] = ContextVar("images_emitted", default=False)
_related_images_selected: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "related_images_selected", default=None
)
_last_image_debug: ContextVar[dict[str, Any] | None] = ContextVar(
    "last_image_debug", default=None
)
_steering_report: ContextVar[dict[str, Any] | None] = ContextVar(
    "steering_report", default=None
)
# Worker tasks copy ContextVar; parent dump reads this shared snapshot instead.
_query_debug_snapshot: dict[str, Any] = {}


def _sync_query_debug_snapshot() -> None:
    _query_debug_snapshot.clear()
    _query_debug_snapshot.update(
        {
            "retrieval_context": _retrieval_context.get(),
            "retrieved_docs_text": _retrieved_docs_text.get(),
            "retrieved_docs": _retrieved_docs.get(),
            "related_images": list(_related_images_selected.get() or []),
            "steering_report": dict(_steering_report.get() or {}),
            "images_debug": dict(_last_image_debug.get() or {}),
        }
    )


def get_query_debug_state() -> dict[str, Any]:
    """Snapshot hook state for query debug dumps."""
    if _query_debug_snapshot:
        return dict(_query_debug_snapshot)
    return {
        "retrieval_context": _retrieval_context.get(),
        "retrieved_docs_text": _retrieved_docs_text.get(),
        "retrieved_docs": _retrieved_docs.get(),
        "related_images": list(_related_images_selected.get() or []),
        "steering_report": dict(_steering_report.get() or {}),
        "images_debug": dict(_last_image_debug.get() or {}),
    }


def _latest_retrieved_docs(explicit: list[dict] | None) -> list[dict] | None:
    if explicit is not None:
        return explicit
    return _retrieved_docs.get()


def set_query_media_roots(roots: list[Path]) -> None:
    _media_roots.set(roots)


def set_query_text_for_images(query: str) -> None:
    _query_text.set(query)


def get_retrieval_context() -> str | None:
    return _retrieval_context.get()


async def _emit_related_images(retrieved_docs: list[dict] | None = None) -> None:
    if _images_emitted.get():
        return
    q = _progress_queue.get()
    roots = _media_roots.get()
    if q is None or not roots:
        return
    try:
        from image_query_refs import (  # noqa: WPS433
            explain_query_images,
            images_for_api,
            merge_context_for_images,
            text_from_retrieved_docs,
        )

        docs = _latest_retrieved_docs(retrieved_docs) or []
        docs_text = text_from_retrieved_docs(docs).strip()
        if not docs_text:
            return
        _retrieved_docs_text.set(docs_text)
        _last_image_debug.set(
            explain_query_images(
                docs_text,
                roots,
                query=_query_text.get(),
                retrieved_docs=docs,
            )
        )
        images = images_for_api(
            docs_text,
            roots,
            query=_query_text.get(),
            retrieved_docs=docs,
            limit=4,
        )
        if images:
            q.put_nowait({"type": "related_images", "images": images})
            _related_images_selected.set(list(images))
            _images_emitted.set(True)
        _sync_query_debug_snapshot()
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning("Failed to emit related images: %s", exc)


async def _emit(phase: str) -> None:
    q = _progress_queue.get()
    if q is None:
        return
    try:
        q.put_nowait(
            {
                "type": "status",
                "phase": phase,
                "text": PHASE_TEXT.get(phase, phase),
            }
        )
    except asyncio.QueueFull:
        pass


@contextlib.asynccontextmanager
async def query_progress_hooks() -> AsyncIterator[asyncio.Queue[dict[str, str]]]:
    """Install hooks; yield a queue of ``{type, phase, text}`` status events."""
    import lightrag.operate as op
    import lightrag.utils as ut

    q: asyncio.Queue[dict[str, str]] = asyncio.Queue(maxsize=32)
    token = _progress_queue.set(q)

    orig_build_ctx = op._build_query_context
    orig_naive_query = op.naive_query
    orig_rerank = ut.apply_rerank_if_enabled
    orig_process_chunks = ut.process_chunks_unified

    async def _build_query_context(*args: Any, **kwargs: Any):
        await _emit(PHASE_RETRIEVE)
        ctx = await orig_build_ctx(*args, **kwargs)
        if isinstance(ctx, str):
            _retrieval_context.set(ctx)
        return ctx

    async def _naive_query(*args: Any, **kwargs: Any):
        await _emit(PHASE_RETRIEVE)
        return await orig_naive_query(*args, **kwargs)

    async def _apply_rerank_if_enabled(
        query: str,
        retrieved_docs: list[dict],
        global_config: dict,
        enable_rerank: bool = True,
        top_n: int | None = None,
    ):
        if enable_rerank and retrieved_docs:
            await _emit(PHASE_RERANK)
        _query_text.set(query)
        docs = await orig_rerank(
            query, retrieved_docs, global_config, enable_rerank, top_n
        )
        try:
            from query_doc_steering import filter_retrieved_docs_by_query  # noqa: WPS433

            filtered = filter_retrieved_docs_by_query(query, docs)
            final_docs = filtered if filtered else docs
        except Exception:
            final_docs = docs
        _retrieved_docs.set(final_docs)
        await _emit_related_images(final_docs)
        _sync_query_debug_snapshot()
        return final_docs

    async def _process_chunks_unified(*args: Any, **kwargs: Any):
        query = args[0] if args else kwargs.get("query", "")
        if isinstance(query, str) and query.strip():
            _query_text.set(query)
        chunks = await orig_process_chunks(*args, **kwargs)
        await _emit(PHASE_GENERATE)
        try:
            import sys
            from pathlib import Path

            scripts_dir = Path(__file__).resolve().parent
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            from query_doc_steering import consume_filter_report  # noqa: WPS433

            report = consume_filter_report()
            if report and report.get("active"):
                _steering_report.set(dict(report))
                q = _progress_queue.get()
                if q is not None:
                    q.put_nowait(report)
        except Exception:
            pass
        return chunks

    op._build_query_context = _build_query_context  # type: ignore[method-assign]
    op.naive_query = _naive_query  # type: ignore[method-assign]
    ut.apply_rerank_if_enabled = _apply_rerank_if_enabled  # type: ignore[method-assign]
    ut.process_chunks_unified = _process_chunks_unified  # type: ignore[method-assign]

    try:
        yield q
    finally:
        op._build_query_context = orig_build_ctx  # type: ignore[method-assign]
        op.naive_query = orig_naive_query  # type: ignore[method-assign]
        ut.apply_rerank_if_enabled = orig_rerank  # type: ignore[method-assign]
        ut.process_chunks_unified = orig_process_chunks  # type: ignore[method-assign]
        _progress_queue.reset(token)
        _retrieval_context.set(None)
        _retrieved_docs_text.set(None)
        _media_roots.set(None)
        _query_text.set(None)
        _images_emitted.set(False)
        _retrieved_docs.set(None)
        _related_images_selected.set(None)
        _steering_report.set(None)
        _last_image_debug.set(None)
        _query_debug_snapshot.clear()


def strip_think_tags(text: str) -> str:
    """Remove chain-of-thought blocks from streamed LLM output."""
    import re

    if not text:
        return text
    tags = (
        ("think", "think"),
        ("redacted_reasoning", "redacted_reasoning"),
    )
    out = text
    for open_name, close_name in tags:
        o = "<" + open_name + ">"
        c = "</" + close_name + ">"
        out = re.sub(re.escape(o) + r"[\s\S]*?" + re.escape(c), "", out, flags=re.I)
    return out.strip()
