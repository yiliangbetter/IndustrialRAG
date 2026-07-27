"""Per-request timing trace for Web query debugging (see ``RAG_QUERY_DEBUG_DUMP``)."""

from __future__ import annotations

import os
import time
from contextvars import ContextVar
from typing import Any

_TRACE_ENABLED: ContextVar[bool] = ContextVar(
    "query_timing_trace_enabled", default=False
)
_TRACE_START: ContextVar[float | None] = ContextVar(
    "query_timing_trace_start", default=None
)
_TRACE_META: ContextVar[dict[str, Any] | None] = ContextVar(
    "query_timing_trace_meta", default=None
)
_TRACE_EVENTS: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "query_timing_trace_events", default=None
)


def is_query_timing_enabled() -> bool:
    flag = (os.getenv("RAG_QUERY_DEBUG_DUMP") or "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return True
    flag = (os.getenv("RAG_QUERY_TIMING_DUMP") or "").strip().lower()
    return flag in ("1", "true", "yes", "on")


def clear_query_trace() -> None:
    _TRACE_ENABLED.set(False)
    _TRACE_START.set(None)
    _TRACE_META.set(None)
    _TRACE_EVENTS.set(None)


def begin_query_trace(**meta: Any) -> None:
    if not is_query_timing_enabled():
        return
    _TRACE_ENABLED.set(True)
    _TRACE_START.set(time.perf_counter())
    _TRACE_META.set(dict(meta))
    _TRACE_EVENTS.set([])


def trace_event(kind: str, **detail: Any) -> None:
    if not _TRACE_ENABLED.get():
        return
    start = _TRACE_START.get()
    elapsed_ms: float | None = None
    if start is not None:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    row: dict[str, Any] = {"kind": kind}
    if elapsed_ms is not None:
        row["t_ms"] = elapsed_ms
    for key, val in detail.items():
        if val is not None:
            row[key] = val
    events = list(_TRACE_EVENTS.get() or [])
    events.append(row)
    _TRACE_EVENTS.set(events)


def _env_snapshot() -> dict[str, str | None]:
    keys = (
        "RAG_CLARIFY_ENABLED",
        "CLARIFY_CANDIDATE_SKIP_PROBE",
        "CLARIFY_DIRECT_RERANK_MIN",
        "MIN_RERANK_SCORE",
        "RERANK_MODEL",
        "RERANK_HF_DEVICE",
        "HF_EMBED_DEVICE",
        "RERANK_RELEASE_AFTER_QUERY",
        "RERANK_BY_DEFAULT",
        "RAG_QUERY_MODE",
    )
    return {key: os.getenv(key) for key in keys}


def _summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    loads = [e for e in events if e.get("kind") == "rerank_load_done"]
    releases = [e for e in events if e.get("kind") == "rerank_release"]
    reuses = [e for e in events if e.get("kind") == "rerank_reuse"]
    predicts = [e for e in events if e.get("kind") == "rerank_predict_done"]
    phases = [e for e in events if e.get("kind") == "phase"]
    predict_s = sum(float(e.get("predict_s") or 0) for e in predicts)
    pairs = sum(int(e.get("pairs") or 0) for e in predicts)
    lines: list[str] = []
    if loads:
        lines.append(f"CrossEncoder 加载 {len(loads)} 次")
    if reuses:
        lines.append(f"CrossEncoder 复用 {len(reuses)} 次（未 release 或同轮复用）")
    if releases:
        lines.append(f"CrossEncoder 释放 {len(releases)} 次")
    if predicts:
        lines.append(
            f"rerank predict {len(predicts)} 次，合计 ~{predict_s:.1f}s，"
            f"约 {pairs} query-chunk 对"
        )
    phase_text = " → ".join(str(e.get("phase")) for e in phases if e.get("phase"))
    if phase_text:
        lines.append(f"答题阶段 UI: {phase_text}")
    injects = [e for e in events if e.get("kind") == "clarify_inject"]
    if injects:
        lines.append("点选后使用了 clarify bundle 注入（跳过检索/rerank）")
    return {
        "rerank_load_count": len(loads),
        "rerank_reuse_count": len(reuses),
        "rerank_release_count": len(releases),
        "rerank_predict_count": len(predicts),
        "rerank_predict_s_sum": round(predict_s, 1) if predict_s else 0.0,
        "rerank_pairs_sum": pairs,
        "phase_sequence": [e.get("phase") for e in phases if e.get("phase")],
        "summary_lines": lines,
    }


def finish_query_trace(**extra: Any) -> dict[str, Any]:
    if not _TRACE_ENABLED.get():
        return {}
    start = _TRACE_START.get()
    events = list(_TRACE_EVENTS.get() or [])
    wall_ms: float | None = None
    if start is not None:
        wall_ms = round((time.perf_counter() - start) * 1000, 1)
    summary = _summarize_events(events)
    payload: dict[str, Any] = {
        "source": "web",
        "wall_ms": wall_ms,
        "meta": dict(_TRACE_META.get() or {}),
        "env": _env_snapshot(),
        "events": events,
        "summary": summary,
    }
    for key, val in extra.items():
        if val is not None:
            payload[key] = val
    clear_query_trace()
    return payload
