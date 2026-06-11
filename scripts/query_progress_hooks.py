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
_llm_chunks_for_images: ContextVar[list[dict] | None] = ContextVar(
    "llm_chunks_for_images", default=None
)
_rerank_docs: ContextVar[list[dict] | None] = ContextVar("rerank_docs", default=None)
_llm_chunks_rerank_figure_supplement: ContextVar[int] = ContextVar(
    "llm_chunks_rerank_figure_supplement", default=0
)
_related_images_selected: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "related_images_selected", default=None
)
_answer_text: ContextVar[str | None] = ContextVar("answer_text", default=None)
_inline_placements: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "inline_placements", default=None
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
            "llm_chunks_for_images": _llm_chunks_for_images.get(),
            "llm_chunks_rerank_figure_supplement": _llm_chunks_rerank_figure_supplement.get()
            or 0,
            "related_images": list(_related_images_selected.get() or []),
            "inline_placements": list(_inline_placements.get() or []),
            "steering_report": dict(_steering_report.get() or {}),
            "images_debug": dict(_last_image_debug.get() or {}),
        }
    )


def get_query_debug_state() -> dict[str, Any]:
    """Snapshot hook state for query debug dumps."""
    live = {
        "retrieval_context": _retrieval_context.get(),
        "retrieved_docs_text": _retrieved_docs_text.get(),
        "retrieved_docs": _retrieved_docs.get(),
        "llm_chunks_for_images": _llm_chunks_for_images.get(),
        "related_images": list(_related_images_selected.get() or []),
        "inline_placements": list(_inline_placements.get() or []),
        "steering_report": dict(_steering_report.get() or {}),
        "images_debug": dict(_last_image_debug.get() or {}),
    }
    if not _query_debug_snapshot:
        return live
    snap = dict(_query_debug_snapshot)
    for key, val in live.items():
        if val is None or val == [] or val == {}:
            continue
        if not snap.get(key):
            snap[key] = val
    return snap


def get_answer_text_for_images() -> str | None:
    return _answer_text.get()


def set_query_media_roots(roots: list[Path]) -> None:
    _media_roots.set(roots)


def set_query_text_for_images(query: str) -> None:
    _query_text.set(query)


def set_answer_text_for_images(answer: str) -> None:
    _answer_text.set((answer or "").strip() or None)


def get_retrieval_context() -> str | None:
    return _retrieval_context.get()


def finalize_inline_images(
    *,
    limit: int | None = None,
    answer_text: str | None = None,
) -> dict[str, Any]:
    """Resolve inline figures cited by the generated answer (Plan A + placements)."""
    from image_query_refs import (  # noqa: WPS433
        build_inline_placements,
        default_image_selection_limit,
        filter_docs_cited_by_answer,
        resolve_query_images,
        text_from_retrieved_docs,
    )
    from query_doc_steering import detect_table_filter_signal  # noqa: WPS433

    if limit is None:
        limit = default_image_selection_limit()

    snap = get_query_debug_state()
    docs = list(
        snap.get("llm_chunks_for_images")
        or _llm_chunks_for_images.get()
        or []
    )
    roots = _media_roots.get()
    q = (_query_text.get() or "").strip()
    answer = (answer_text or _answer_text.get() or "").strip()
    if answer:
        set_answer_text_for_images(answer)
    empty: dict[str, Any] = {"images": [], "placements": [], "debug": {}}
    if not roots or not q:
        _related_images_selected.set([])
        _inline_placements.set([])
        return empty

    full_primary = text_from_retrieved_docs(docs).strip()
    if full_primary and detect_table_filter_signal(q, full_primary):
        debug: dict[str, Any] = {
            "gate": {"ok": False, "reason": "table_filter_listing"},
            "llm_chunk_count": len(docs),
        }
        _last_image_debug.set(debug)
        _related_images_selected.set([])
        _inline_placements.set([])
        _sync_query_debug_snapshot()
        return {**empty, "debug": debug}

    cite_meta: dict[str, Any] = {}
    if answer:
        docs, cite_meta = filter_docs_cited_by_answer(answer, docs, query=q)
    docs_text = text_from_retrieved_docs(docs).strip() if docs else ""
    if not docs_text:
        debug = {
            "gate": {"ok": False, "reason": "no_answer_cited_chunks"},
            "answer_citation": cite_meta,
            "llm_chunk_count": len(docs),
        }
        _last_image_debug.set(debug)
        _related_images_selected.set([])
        _inline_placements.set([])
        _sync_query_debug_snapshot()
        return {**empty, "debug": debug}

    images, debug = resolve_query_images(
        docs_text,
        roots,
        query=q,
        retrieved_docs=docs,
        limit=limit,
    )
    if not images:
        gate = dict(debug.get("gate") or {})
        if gate.get("ok") is not False:
            debug["gate"] = {"ok": False, "reason": "no_inline_in_cited"}
        _last_image_debug.set(debug)
        _related_images_selected.set([])
        _inline_placements.set([])
        _sync_query_debug_snapshot()
        return {**empty, "debug": debug}

    placements: list[dict[str, Any]] = []
    if answer:
        images_copy = list(images)
        placements = build_inline_placements(
            answer,
            images_copy,
            retrieved_docs=docs,
        )
        if placements:
            images = images_copy
        else:
            images = []
            debug["gate"] = {"ok": False, "reason": "no_inline_placements"}

    supplement = int(snap.get("llm_chunks_rerank_figure_supplement") or 0)
    if cite_meta.get("mode") == "answer_citation":
        debug["image_source"] = "llm_chunks+answer_citation"
    else:
        debug["image_source"] = (
            "llm_chunks+rerank_inline_figures" if supplement else "llm_chunks"
        )
    debug["answer_citation"] = cite_meta
    debug["llm_chunk_count"] = len(docs)
    debug["inline_placements"] = placements
    if supplement:
        debug["rerank_figure_supplement"] = supplement
    _last_image_debug.set(debug)
    _related_images_selected.set(list(images))
    _inline_placements.set(list(placements))
    _sync_query_debug_snapshot()
    return {"images": images, "placements": placements, "debug": debug}


def finalize_related_images(
    *,
    limit: int | None = None,
    answer_text: str | None = None,
) -> list[dict[str, Any]]:
    """Backward-compatible wrapper returning only the image list."""
    result = finalize_inline_images(limit=limit, answer_text=answer_text)
    return list(result.get("images") or [])


def _sync_llm_chunks_for_images(query: str, chunks: list[dict]) -> None:
    """Persist LLM input chunks; optionally add rerank inline-figure chunks aligned with query."""
    if not chunks:
        return
    from image_query_refs import (  # noqa: WPS433
        supplement_llm_docs_with_rerank_figures,
        text_from_retrieved_docs,
    )

    try:
        from query_doc_steering import filter_retrieved_docs_by_query  # noqa: WPS433

        filtered = filter_retrieved_docs_by_query(query, chunks)
        final = filtered if filtered else chunks
    except Exception:
        final = chunks
    rerank_pool = list(_rerank_docs.get() or [])
    merged, added = supplement_llm_docs_with_rerank_figures(
        query, final, rerank_pool
    )
    _llm_chunks_rerank_figure_supplement.set(added)
    _llm_chunks_for_images.set(merged)
    _retrieved_docs.set(merged)
    docs_text = text_from_retrieved_docs(merged).strip()
    if docs_text:
        _retrieved_docs_text.set(docs_text)
    _sync_query_debug_snapshot()


def _sync_retrieved_docs_after_rerank(final_docs: list[dict]) -> None:
    """Snapshot rerank pool for optional subject-figure supplement at LLM chunk sync."""
    _rerank_docs.set(list(final_docs or []))


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
    # operate.py imports process_chunks_unified at module load; patch both bindings.
    orig_process_chunks = op.process_chunks_unified

    async def _build_query_context(*args: Any, **kwargs: Any):
        await _emit(PHASE_RETRIEVE)
        query = args[0] if args else kwargs.get("query", "")
        if isinstance(query, str) and query.strip():
            _query_text.set(query)
        ctx = await orig_build_ctx(*args, **kwargs)
        if isinstance(ctx, str) and ctx.strip():
            _retrieval_context.set(ctx)
            _sync_query_debug_snapshot()
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
        _sync_retrieved_docs_after_rerank(final_docs)
        return final_docs

    async def _process_chunks_unified(*args: Any, **kwargs: Any):
        query = args[0] if args else kwargs.get("query", "")
        if isinstance(query, str) and query.strip():
            _query_text.set(query)
        chunks = await orig_process_chunks(*args, **kwargs)
        if isinstance(chunks, list) and chunks:
            final_chunks = chunks
            try:
                from image_query_refs import (  # noqa: WPS433
                    narrow_retrieved_docs_to_anchor_sections,
                    sanitize_retrieved_docs_content,
                )

                narrowed, narrow_meta = narrow_retrieved_docs_to_anchor_sections(
                    query, chunks
                )
                final_chunks = narrowed if narrow_meta.get("narrowed") else chunks
                final_chunks = sanitize_retrieved_docs_content(final_chunks)
                if narrow_meta.get("narrowed"):
                    prev = dict(_last_image_debug.get() or {})
                    prev["llm_section_narrow"] = narrow_meta
                    _last_image_debug.set(prev)
            except Exception:
                final_chunks = chunks
            _sync_llm_chunks_for_images(query, final_chunks)
            await _emit(PHASE_GENERATE)
            try:
                import sys

                scripts_dir = Path(__file__).resolve().parent
                if str(scripts_dir) not in sys.path:
                    sys.path.insert(0, str(scripts_dir))
                from query_doc_steering import consume_filter_report  # noqa: WPS433

                report = consume_filter_report()
                if report:
                    _steering_report.set(report)
                    q = _progress_queue.get()
                    if q is not None:
                        q.put_nowait(report)
            except Exception:
                pass
            return final_chunks
        await _emit(PHASE_GENERATE)
        try:
            import sys

            scripts_dir = Path(__file__).resolve().parent
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            from query_doc_steering import consume_filter_report  # noqa: WPS433

            report = consume_filter_report()
            if report:
                _steering_report.set(report)
                q = _progress_queue.get()
                if q is not None:
                    q.put_nowait(report)
        except Exception:
            pass
        return chunks

    op._build_query_context = _build_query_context  # type: ignore[method-assign]
    op.naive_query = _naive_query  # type: ignore[method-assign]
    ut.apply_rerank_if_enabled = _apply_rerank_if_enabled  # type: ignore[method-assign]
    op.process_chunks_unified = _process_chunks_unified  # type: ignore[method-assign]
    ut.process_chunks_unified = _process_chunks_unified  # type: ignore[method-assign]

    try:
        yield q
    finally:
        op._build_query_context = orig_build_ctx  # type: ignore[method-assign]
        op.naive_query = orig_naive_query  # type: ignore[method-assign]
        ut.apply_rerank_if_enabled = orig_rerank  # type: ignore[method-assign]
        op.process_chunks_unified = orig_process_chunks  # type: ignore[method-assign]
        ut.process_chunks_unified = orig_process_chunks  # type: ignore[method-assign]
        _progress_queue.reset(token)
        _retrieval_context.set(None)
        _retrieved_docs_text.set(None)
        _media_roots.set(None)
        _query_text.set(None)
        _answer_text.set(None)
        _retrieved_docs.set(None)
        _llm_chunks_for_images.set(None)
        _rerank_docs.set(None)
        _llm_chunks_rerank_figure_supplement.set(0)
        _related_images_selected.set(None)
        _inline_placements.set(None)
        _steering_report.set(None)
        _last_image_debug.set(None)
        _query_debug_snapshot.clear()


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
