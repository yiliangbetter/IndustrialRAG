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
_rerank_figure_pool: ContextVar[list[dict] | None] = ContextVar(
    "rerank_figure_pool", default=None
)
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
_lightrag_for_relevance: ContextVar[Any | None] = ContextVar(
    "lightrag_for_relevance", default=None
)
_query_mode_for_relevance: ContextVar[str | None] = ContextVar(
    "query_mode_for_relevance", default=None
)
_naive_relevance: ContextVar[dict[str, Any] | None] = ContextVar(
    "naive_relevance", default=None
)
_llm_input: ContextVar[dict[str, Any] | None] = ContextVar("llm_input", default=None)
# Worker tasks copy ContextVar; parent dump reads this shared snapshot instead.
_query_debug_snapshot: dict[str, Any] = {}
# Survives hook teardown so dumps can be written after ``async with`` exits.
_retained_query_debug_snapshot: dict[str, Any] = {}


def _retain_query_debug_snapshot() -> None:
    if not _query_debug_snapshot:
        return
    _retained_query_debug_snapshot.clear()
    _retained_query_debug_snapshot.update(_query_debug_snapshot)


def _active_query_debug_snapshot() -> dict[str, Any]:
    if _query_debug_snapshot:
        return dict(_query_debug_snapshot)
    if _retained_query_debug_snapshot:
        return dict(_retained_query_debug_snapshot)
    return {}


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
            "naive_relevance": dict(_naive_relevance.get() or {})
            if isinstance(_naive_relevance.get(), dict)
            else _naive_relevance.get(),
            "llm_input": dict(_llm_input.get() or {})
            if isinstance(_llm_input.get(), dict)
            else _llm_input.get(),
        }
    )


def get_query_debug_state() -> dict[str, Any]:
    """Snapshot hook state for query debug dumps."""
    snap = _active_query_debug_snapshot()
    live = {
        "retrieval_context": _retrieval_context.get(),
        "retrieved_docs_text": _retrieved_docs_text.get(),
        "retrieved_docs": _retrieved_docs.get(),
        "llm_chunks_for_images": _llm_chunks_for_images.get(),
        "related_images": list(_related_images_selected.get() or []),
        "inline_placements": list(_inline_placements.get() or []),
        "steering_report": dict(_steering_report.get() or {}),
        "images_debug": dict(_last_image_debug.get() or {}),
        "naive_relevance": _naive_relevance.get(),
        "llm_input": _llm_input.get(),
    }
    if not snap:
        return live
    merged = dict(snap)
    for key, val in live.items():
        if val is None or val == [] or val == {}:
            continue
        if not merged.get(key):
            merged[key] = val
    return merged


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


def get_llm_input() -> dict[str, Any] | None:
    val = _llm_input.get()
    return dict(val) if isinstance(val, dict) else None


def _capture_llm_context_from_build_result(ctx: Any) -> None:
    """Persist answer-LLM context from LightRAG ``QueryContextResult`` or legacy str."""
    context_str: str | None = None
    raw_data: dict[str, Any] | None = None
    if isinstance(ctx, str):
        context_str = ctx.strip() or None
    elif ctx is not None:
        maybe_ctx = getattr(ctx, "context", None)
        if isinstance(maybe_ctx, str) and maybe_ctx.strip():
            context_str = maybe_ctx.strip()
        maybe_raw = getattr(ctx, "raw_data", None)
        if isinstance(maybe_raw, dict):
            raw_data = maybe_raw
    if context_str:
        _retrieval_context.set(context_str)
    if raw_data is not None:
        from query_debug_dump import build_llm_input_snapshot  # noqa: WPS433

        _llm_input.set(build_llm_input_snapshot(raw_data, context_str=context_str))
    elif context_str:
        _llm_input.set(
            {
                "context_chars": len(context_str),
                "counts": {
                    "entities": 0,
                    "relationships": 0,
                    "chunks": 0,
                    "references": 0,
                },
                "entities": [],
                "relationships": [],
                "chunks": [],
                "references": [],
            }
        )


def get_naive_relevance() -> dict[str, Any] | None:
    """Latest naive relevance payload for the in-flight query (if scored)."""
    val = _naive_relevance.get()
    return dict(val) if isinstance(val, dict) else None


def set_query_lightrag(lightrag: Any, *, mode: str | None = None) -> None:
    """Attach LightRAG for per-query naive relevance scoring inside hooks."""
    _lightrag_for_relevance.set(lightrag)
    if mode is not None:
        _query_mode_for_relevance.set(mode.strip() or None)


async def _ensure_naive_relevance_scored(query: str) -> dict[str, Any] | None:
    existing = _naive_relevance.get()
    if isinstance(existing, dict) and existing.get("query") == (query or "").strip():
        return existing

    mode = _query_mode_for_relevance.get() or ""
    from raganything.naive_relevance import (  # noqa: WPS433
        is_naive_relevance_enabled,
        score_naive_relevance,
    )

    if not is_naive_relevance_enabled(mode):
        return None

    lightrag = _lightrag_for_relevance.get()
    if lightrag is None:
        return None

    from lightrag import QueryParam

    payload = await score_naive_relevance(
        lightrag,
        query,
        query_param=QueryParam(mode=mode or "mix"),
    )
    _naive_relevance.set(payload)
    _sync_query_debug_snapshot()

    q = _progress_queue.get()
    if q is not None:
        try:
            q.put_nowait({"type": "naive_relevance", "data": payload})
        except asyncio.QueueFull:
            pass
    return payload


def finalize_inline_images(
    *,
    limit: int | None = None,
    answer_text: str | None = None,
) -> dict[str, Any]:
    """Resolve inline figures cited by the generated answer (Plan A + placements)."""
    from image_query_refs import (  # noqa: WPS433
        _dedupe_doc_list,
        _doc_content,
        _is_multi_machine_comparison_query,
        build_inline_placements,
        default_image_selection_limit,
        extract_image_refs_from_context,
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

    docs_before_cite = list(docs)
    cite_pool = list(docs_before_cite)
    anchor_pool = list(docs_before_cite)
    cite_meta: dict[str, Any] = {}
    if answer:
        cite_pool = list(docs_before_cite)
        rerank_pool = list(_rerank_docs.get() or [])
        figure_pool = list(_rerank_figure_pool.get() or [])
        if rerank_pool:
            cite_pool = _dedupe_doc_list(cite_pool + rerank_pool)
        anchor_pool = cite_pool
        if figure_pool:
            anchor_pool = _dedupe_doc_list(cite_pool + figure_pool)
        if _is_multi_machine_comparison_query(q) and rerank_pool:
            multi_fig = [
                doc
                for doc in rerank_pool
                if extract_image_refs_from_context(_doc_content(doc).strip())
            ]
            if multi_fig:
                anchor_pool = _dedupe_doc_list(anchor_pool + multi_fig)
        from image_query_refs import (  # noqa: WPS433
            _cited_manual_hints_from_answer,
            _doc_basename,
            _load_figure_chunks_for_manual_paths,
            _should_expand_cited_manual_kv_pool,
        )

        if _should_expand_cited_manual_kv_pool(q, answer):
            cited = _cited_manual_hints_from_answer(answer)
            allowed: set[str] = set(cited)
            for doc in cite_pool + rerank_pool + figure_pool:
                fp = _doc_basename(doc)
                if fp:
                    allowed.add(fp)
            if allowed:
                kv_extra = _load_figure_chunks_for_manual_paths(allowed, q)
                if kv_extra:
                    anchor_pool = _dedupe_doc_list(anchor_pool + kv_extra)
        docs, cite_meta = filter_docs_cited_by_answer(
            answer,
            docs,
            query=q,
            pool=cite_pool,
            anchor_pool=anchor_pool,
        )
        cite_meta["cite_pool_size"] = len(cite_pool)
        if figure_pool:
            cite_meta["anchor_pool_size"] = len(anchor_pool)
            cite_meta["cite_figure_pool_size"] = len(figure_pool)
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
        figure_pool=anchor_pool,
        limit=limit,
        answer=answer or None,
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
            query=q,
            retrieved_docs=docs,
        )
        if placements:
            images = images_copy
        else:
            images = images_copy
            debug["placement_fallback"] = "gallery_no_inline_anchors"

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
    """Persist LLM input chunks (same batch the answer LLM sees)."""
    if not chunks:
        return
    from image_query_refs import (  # noqa: WPS433
        llm_rerank_figure_supplement_enabled,
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
    if llm_rerank_figure_supplement_enabled() and rerank_pool:
        merged, added = supplement_llm_docs_with_rerank_figures(
            query, final, rerank_pool
        )
    else:
        merged, added = final, 0
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


def get_llm_input_chunks() -> list[dict]:
    """LLM-bound chunks from the latest hooked query (includes ``rerank_score`` when rerank ran)."""
    raw = _llm_chunks_for_images.get() or _retrieved_docs.get() or []
    return list(raw)


@contextlib.asynccontextmanager
async def query_progress_hooks() -> AsyncIterator[asyncio.Queue[dict[str, str]]]:
    """Install hooks; yield a queue of ``{type, phase, text}`` status events."""
    import lightrag.operate as op
    import lightrag.utils as ut

    q: asyncio.Queue[dict[str, str]] = asyncio.Queue(maxsize=32)
    token = _progress_queue.set(q)
    _query_debug_snapshot.clear()
    _retained_query_debug_snapshot.clear()

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
            await _ensure_naive_relevance_scored(query)
        ctx = await orig_build_ctx(*args, **kwargs)
        _capture_llm_context_from_build_result(ctx)
        _sync_query_debug_snapshot()
        return ctx

    async def _naive_query(*args: Any, **kwargs: Any):
        await _emit(PHASE_RETRIEVE)
        query = args[0] if args else kwargs.get("query", "")
        if isinstance(query, str) and query.strip():
            _query_text.set(query)
            await _ensure_naive_relevance_scored(query)
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
        try:
            from image_query_refs import (  # noqa: WPS433
                _chunk_figure_context_aligns_query,
                _chunk_subject_score,
                _doc_content,
                extract_image_refs_from_context,
            )

            fig_candidates: list[tuple[float, dict]] = []
            for doc in retrieved_docs or []:
                content = _doc_content(doc).strip()
                if not content or not extract_image_refs_from_context(content):
                    continue
                if not _chunk_figure_context_aligns_query(query, content):
                    continue
                score = _chunk_subject_score(query, content)
                if score < 0.25:
                    continue
                fig_candidates.append((score, doc))
            fig_candidates.sort(key=lambda pair: pair[0], reverse=True)
            fig_docs = [doc for _, doc in fig_candidates[:16]]
            try:
                from query_doc_steering import _is_cross_manual_listing_query  # noqa: WPS433

                if _is_cross_manual_listing_query(query):
                    from image_query_refs import (  # noqa: WPS433
                        _doc_basename,
                        _load_figure_chunks_for_manual_paths,
                    )

                    allowed = {
                        fp
                        for doc in retrieved_docs or []
                        if (fp := _doc_basename(doc))
                    }
                    kv_fig = _load_figure_chunks_for_manual_paths(
                        allowed, query, max_per_manual=8
                    )
                    if kv_fig:
                        seen_ids = {id(d) for d in fig_docs}
                        for doc in kv_fig:
                            if id(doc) not in seen_ids:
                                fig_docs.append(doc)
                                seen_ids.add(id(doc))
            except Exception:
                pass
            _rerank_figure_pool.set(fig_docs)
        except Exception:
            _rerank_figure_pool.set([])
        return final_docs

    async def _process_chunks_unified(*args: Any, **kwargs: Any):
        query = args[0] if args else kwargs.get("query", "")
        if isinstance(query, str) and query.strip():
            _query_text.set(query)
        chunks = await orig_process_chunks(*args, **kwargs)
        if isinstance(chunks, list) and chunks:
            # Plan B: passthrough chunks to the answer LLM (demo granularity).
            # Snapshot the same batch for Plan A inline images; do not narrow/sanitize return.
            _sync_llm_chunks_for_images(query, chunks)
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
        _rerank_figure_pool.set(None)
        _llm_chunks_rerank_figure_supplement.set(0)
        _related_images_selected.set(None)
        _inline_placements.set(None)
        _steering_report.set(None)
        _last_image_debug.set(None)
        _lightrag_for_relevance.set(None)
        _query_mode_for_relevance.set(None)
        _naive_relevance.set(None)
        _llm_input.set(None)
        _retain_query_debug_snapshot()
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
