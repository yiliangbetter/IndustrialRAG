"""Naive (chunk-vector) relevance scoring for query clarification / threshold tuning.

Coarse retrieval only: embed the query (or keyword strings) and search ``chunks_vdb``.
No rerank, no KG merge. Used as the first gate before mix-mode retrieval and future
clarifying-question flows.

Public API (stable extension surface):

- :func:`is_naive_relevance_enabled` — env / mode switch
- :func:`score_naive_relevance` — score query + high/low keywords
- :func:`probe_chunk_vector_score` — score arbitrary search text
- :func:`primary_relevance_score` — pick query-level score from a payload
- :func:`format_relevance_report` — human-readable report
"""

from __future__ import annotations

import os
from typing import Any

from lightrag import QueryParam
from lightrag.operate import get_keywords_from_query


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def is_naive_relevance_enabled(mode: str | None = None) -> bool:
    """Whether naive relevance scoring runs on each query (``RAG_NAIVE_RELEVANCE``)."""
    flag = (os.getenv("RAG_NAIVE_RELEVANCE") or "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return True
    return (mode or "").strip().lower() == "naive"


def cosine_threshold_from_env() -> float:
    return _env_float("COSINE_THRESHOLD", 0.2)


def probe_top_k_from_env() -> int:
    return max(1, _env_int("NAIVE_RELEVANCE_TOP_K", _env_int("TOP_K", 40)))


def _cosine_from_hit(hit: dict[str, Any]) -> float | None:
    for key in ("distance", "score", "__metrics__"):
        val = hit.get(key)
        if val is None or val == "":
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def _preview_hit(hit: dict[str, Any], *, limit: int = 80) -> str | None:
    for key in ("content", "text", "chunk_content"):
        val = hit.get(key)
        if isinstance(val, str) and val.strip():
            text = val.strip().replace("\n", " ")
            return text[:limit] + ("…" if len(text) > limit else "")
    return None


async def probe_chunk_vector_score(
    chunks_vdb: Any,
    query_text: str,
    *,
    top_k: int | None = None,
    cosine_threshold: float | None = None,
    preview_limit: int = 3,
) -> dict[str, Any]:
    """Vector-search *query_text* against chunk VDB; return max cosine + hits."""
    text = (query_text or "").strip()
    if not text:
        return {
            "text": query_text,
            "max_cosine_similarity": None,
            "hits_above_threshold": 0,
            "probe_hit_count": 0,
            "reason": "empty_text",
        }

    if top_k is None:
        top_k = probe_top_k_from_env()
    if cosine_threshold is None:
        cosine_threshold = cosine_threshold_from_env()

    old_threshold = getattr(chunks_vdb, "cosine_better_than_threshold", cosine_threshold)
    try:
        chunks_vdb.cosine_better_than_threshold = 0.0
        probe_hits = await chunks_vdb.query(text, top_k=top_k)
        probe_scores: list[float] = []
        top_rows: list[dict[str, Any]] = []
        for idx, hit in enumerate(probe_hits or []):
            if not isinstance(hit, dict):
                continue
            cosine = _cosine_from_hit(hit)
            if cosine is not None:
                probe_scores.append(cosine)
            if idx < preview_limit:
                row: dict[str, Any] = {"rank": idx + 1}
                if cosine is not None:
                    row["cosine_similarity"] = round(cosine, 4)
                fp = hit.get("file_path")
                if isinstance(fp, str) and fp.strip():
                    row["file_path"] = fp.strip()
                preview = _preview_hit(hit)
                if preview:
                    row["preview"] = preview
                top_rows.append(row)

        chunks_vdb.cosine_better_than_threshold = cosine_threshold
        filtered = await chunks_vdb.query(text, top_k=top_k)

        max_cosine = max(probe_scores) if probe_scores else None
        return {
            "text": text,
            "max_cosine_similarity": round(max_cosine, 4) if max_cosine is not None else None,
            "probe_top_k": top_k,
            "probe_hit_count": len(probe_hits or []),
            "hits_above_threshold": len(filtered or []),
            "configured_cosine_threshold": cosine_threshold,
            "top_hits": top_rows,
        }
    finally:
        chunks_vdb.cosine_better_than_threshold = old_threshold


def _global_config(lightrag: Any) -> dict[str, Any]:
    builder = getattr(lightrag, "_build_global_config", None)
    if callable(builder):
        return builder()
    return dict(getattr(lightrag, "__dict__", {}))


async def extract_query_keywords(
    lightrag: Any,
    query: str,
    query_param: QueryParam | None = None,
) -> dict[str, list[str]]:
    """LLM keyword extraction (same path as mix/local/global modes)."""
    param = query_param or QueryParam(mode="mix")
    global_config = _global_config(lightrag)
    hl, ll = await get_keywords_from_query(
        query,
        param,
        global_config,
        hashing_kv=lightrag.llm_response_cache,
    )
    return {
        "high_level": list(hl or []),
        "low_level": list(ll or []),
    }


def _keywords_search_text(keywords: list[str]) -> str:
    parts = [str(k).strip() for k in keywords if str(k).strip()]
    return ", ".join(parts)


async def score_naive_relevance(
    lightrag: Any,
    query: str,
    *,
    query_param: QueryParam | None = None,
    top_k: int | None = None,
    cosine_threshold: float | None = None,
) -> dict[str, Any]:
    """Score full query, high_level keywords, and low_level keywords."""
    q = (query or "").strip()
    chunks_vdb = lightrag.chunks_vdb
    keywords = await extract_query_keywords(lightrag, q, query_param)
    hl_text = _keywords_search_text(keywords["high_level"])
    ll_text = _keywords_search_text(keywords["low_level"])

    query_score = await probe_chunk_vector_score(
        chunks_vdb,
        q,
        top_k=top_k,
        cosine_threshold=cosine_threshold,
    )
    high_score = (
        await probe_chunk_vector_score(
            chunks_vdb,
            hl_text,
            top_k=top_k,
            cosine_threshold=cosine_threshold,
        )
        if hl_text
        else {
            "text": "",
            "max_cosine_similarity": None,
            "hits_above_threshold": 0,
            "probe_hit_count": 0,
            "reason": "empty_high_level_keywords",
            "keywords": keywords["high_level"],
        }
    )
    low_score = (
        await probe_chunk_vector_score(
            chunks_vdb,
            ll_text,
            top_k=top_k,
            cosine_threshold=cosine_threshold,
        )
        if ll_text
        else {
            "text": "",
            "max_cosine_similarity": None,
            "hits_above_threshold": 0,
            "probe_hit_count": 0,
            "reason": "empty_low_level_keywords",
            "keywords": keywords["low_level"],
        }
    )

    return {
        "mode": "naive_vector_probe",
        "query": q,
        "keywords": keywords,
        "thresholds": {
            "cosine_threshold": cosine_threshold
            if cosine_threshold is not None
            else cosine_threshold_from_env(),
            "probe_top_k": top_k if top_k is not None else probe_top_k_from_env(),
        },
        "scores": {
            "query": query_score,
            "high_level": high_score,
            "low_level": low_score,
        },
    }


def primary_relevance_score(
    payload: dict[str, Any] | None,
    *,
    which: str = "query",
) -> float | None:
    """Return max_cosine for ``which`` in (``query``, ``high_level``, ``low_level``)."""
    if not isinstance(payload, dict):
        return None
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        return None
    section = scores.get(which)
    if not isinstance(section, dict):
        return None
    val = section.get("max_cosine_similarity")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def format_relevance_report(payload: dict[str, Any]) -> str:
    """Human-readable lines for CLI / debug dumps."""
    lines: list[str] = []
    lines.append(f"query: {payload.get('query', '')}")
    kw = payload.get("keywords") or {}
    lines.append(f"high_level keywords: {kw.get('high_level', [])}")
    lines.append(f"low_level keywords: {kw.get('low_level', [])}")
    thr = payload.get("thresholds") or {}
    lines.append(
        f"thresholds: cosine={thr.get('cosine_threshold')} probe_top_k={thr.get('probe_top_k')}"
    )
    lines.append("")
    scores = payload.get("scores") or {}
    for label in ("query", "high_level", "low_level"):
        section = scores.get(label) or {}
        text = section.get("text") or ""
        max_cos = section.get("max_cosine_similarity")
        above = section.get("hits_above_threshold")
        reason = section.get("reason")
        lines.append(f"=== {label} ===")
        if text:
            lines.append(f"  search_text: {text!r}")
        lines.append(f"  max_cosine_similarity: {max_cos}")
        lines.append(f"  hits_above_threshold: {above}")
        if reason:
            lines.append(f"  reason: {reason}")
        top = section.get("top_hits") or []
        if top:
            lines.append("  top_hits:")
            for row in top:
                preview = (row.get("preview") or "")[:60]
                lines.append(
                    f"    #{row.get('rank')} cosine={row.get('cosine_similarity')} {preview}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
