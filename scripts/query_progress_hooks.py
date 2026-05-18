"""Monkeypatch LightRAG query stages to emit real progress events (Web UI SSE)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextvars import ContextVar
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
        return await orig_build_ctx(*args, **kwargs)

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
        docs = await orig_rerank(
            query, retrieved_docs, global_config, enable_rerank, top_n
        )
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
