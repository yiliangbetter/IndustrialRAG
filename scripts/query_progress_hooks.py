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
_images_emitted: ContextVar[bool] = ContextVar("images_emitted", default=False)


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
            images_for_api,
            merge_context_for_images,
            text_from_retrieved_docs,
        )

        docs_text = text_from_retrieved_docs(retrieved_docs)
        if docs_text:
            _retrieved_docs_text.set(docs_text)
        merged = merge_context_for_images(
            docs_text,
            _retrieved_docs_text.get(),
        )
        images = images_for_api(
            merged,
            roots,
            query=_query_text.get(),
            extra_context=_retrieval_context.get(),
            limit=4,
        )
        if images:
            q.put_nowait({"type": "related_images", "images": images})
            _images_emitted.set(True)
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
        result = await orig_naive_query(*args, **kwargs)
        if isinstance(result, str):
            _retrieval_context.set(result)
            await _emit_related_images()
        return result

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
        await _emit_related_images(docs)
        try:
            from query_doc_steering import filter_retrieved_docs_by_query  # noqa: WPS433

            filtered = filter_retrieved_docs_by_query(query, docs)
            if filtered:
                return filtered
            # If filter removed everything, keep reranked set (avoid empty context).
            return docs
        except Exception:
            return docs

    async def _process_chunks_unified(*args: Any, **kwargs: Any):
        chunks = await orig_process_chunks(*args, **kwargs)
        await _emit(PHASE_GENERATE)
        await _emit_related_images()
        try:
            import sys
            from pathlib import Path

            scripts_dir = Path(__file__).resolve().parent
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            from query_doc_steering import consume_filter_report  # noqa: WPS433

            report = consume_filter_report()
            if report and report.get("active"):
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
