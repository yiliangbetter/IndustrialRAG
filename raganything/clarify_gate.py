"""Clarification gate before mix/naive aquery (v2).

Naive query vector probe (max cosine vs chunk VDB):
- score < threshold (default 0.6): reject — unrelated to knowledge base, no aquery.
- score >= threshold: show clarification UI with k answerable candidate questions
  (default k=2, each verified via mix retrieval with LLM chunk rerank >= MIN_RERANK_SCORE)
  plus keep-original option.

See ``docs/澄清门控设计方案.md``.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from lightrag import QueryParam

from raganything.naive_relevance import probe_chunk_vector_score

CLARIFY_UNRELATED_MESSAGE = (
    "您的问题与当前知识库内容关联度较低，暂无法基于知识库作答。"
    "请尝试换种说法，或联系技术支持。"
)


class ClarifyGateError(Exception):
    """Base error for clarification gate."""


class ClarifyValidationError(ClarifyGateError):
    """Invalid use_candidate / clarification_id pairing."""


class ClarifyBand(str, Enum):
    UNRELATED = "unrelated"
    CLARIFY = "clarify"


@dataclass(frozen=True)
class ClarifyBypass:
    """Proceed to aquery without showing clarification UI."""

    reason: Literal["disabled", "use_candidate", "keep_original"]


@dataclass(frozen=True)
class ClarifyRequired:
    """Return clarification UI; do not run aquery."""

    data: dict[str, Any]


_ClarifyResult = ClarifyBypass | ClarifyRequired

_STORE_TTL_SEC = 3600
_CLARIFICATION_STORE: dict[str, dict[str, Any]] = {}


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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def is_clarify_gate_enabled(mode: str | None = None) -> bool:
    if not _env_bool("RAG_CLARIFY_ENABLED", False):
        return False
    if _env_bool("CLARIFY_ONLY_NAIVE_MODE", False):
        return (mode or "").strip().lower() == "naive"
    return True


def clarify_threshold() -> float:
    """Single naive-score gate (``CLARIFY_THRESHOLD`` or legacy ``CLARIFY_THRESHOLD_LOWER``)."""
    raw = (os.getenv("CLARIFY_THRESHOLD") or os.getenv("CLARIFY_THRESHOLD_LOWER") or "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return 0.6


def clarify_threshold_lower() -> float:
    """Backward-compatible alias for :func:`clarify_threshold`."""
    return clarify_threshold()


def clarify_threshold_upper() -> float:
    """Deprecated; kept for replay scripts — same as :func:`clarify_threshold`."""
    return clarify_threshold()


def clarify_candidate_k() -> int:
    return max(1, _env_int("CLARIFY_CANDIDATE_K", 2))


def clarify_candidate_max_rounds() -> int:
    return max(1, _env_int("CLARIFY_CANDIDATE_MAX_ROUNDS", 3))


def clarify_candidate_strategy() -> str:
    return (os.getenv("CLARIFY_CANDIDATE_STRATEGY") or "fill_k").strip().lower()


def clarify_candidate_min_rerank_score() -> float:
    """Minimum rerank score for a candidate to count as answerable (defaults to MIN_RERANK_SCORE)."""
    raw = (
        os.getenv("CLARIFY_CANDIDATE_MIN_RERANK_SCORE")
        or os.getenv("MIN_RERANK_SCORE")
        or "0.28"
    ).strip()
    try:
        return float(raw)
    except ValueError:
        return 0.28


def _chunk_rerank_score(doc: dict[str, Any]) -> float | None:
    for key in ("rerank_score", "score"):
        raw = doc.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def _env_bool_rerank_default() -> bool:
    raw = (os.getenv("RERANK_BY_DEFAULT") or "true").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _probe_query_param(mode: str) -> QueryParam:
    return QueryParam(
        mode=(mode or "mix").strip() or "mix",
        only_need_context=True,
        enable_rerank=_env_bool_rerank_default(),
    )


def _summarize_llm_chunks(
    chunks: list[dict[str, Any]], *, min_thr: float
) -> dict[str, Any]:
    scored: list[float] = []
    qualifying: list[dict[str, Any]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        rs = _chunk_rerank_score(chunk)
        if rs is None:
            continue
        scored.append(rs)
        if rs >= min_thr:
            qualifying.append(chunk)

    if not scored:
        return {
            "answerable": False,
            "chunk_count": 0,
            "max_rerank_score": None,
            "llm_chunk_total": len(chunks),
            "scores_unavailable": bool(chunks),
        }

    max_score = max(scored)
    return {
        "answerable": len(qualifying) > 0,
        "chunk_count": len(qualifying),
        "max_rerank_score": round(max_score, 4),
        "llm_chunk_total": len(chunks),
        "scores_unavailable": False,
    }


async def probe_llm_retrieval(
    lightrag: Any,
    query: str,
    *,
    mode: str = "mix",
) -> dict[str, Any]:
    """Run Web-equivalent mix retrieval (no answer LLM) and inspect LLM-bound chunks."""
    q = (query or "").strip()
    min_thr = clarify_candidate_min_rerank_score()
    if not q:
        return {
            "answerable": False,
            "chunk_count": 0,
            "max_rerank_score": None,
            "min_rerank_threshold": min_thr,
            "mode": mode,
            "llm_chunk_total": 0,
            "scores_unavailable": False,
        }

    param = _probe_query_param(mode)
    chunks: list[dict[str, Any]] = []
    try:
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        scripts = root / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        from query_progress_hooks import (  # noqa: WPS433
            get_llm_input_chunks,
            query_progress_hooks,
        )

        async with query_progress_hooks():
            await lightrag.aquery_data(q, param)
            chunks = get_llm_input_chunks()
    except Exception:
        chunks = []

    stats = _summarize_llm_chunks(chunks, min_thr=min_thr)
    return {
        **stats,
        "min_rerank_threshold": min_thr,
        "mode": param.mode,
    }


def classify_query_relevance(
    score: float | None,
    *,
    threshold: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
) -> ClarifyBand:
    """Classify naive probe score against a single threshold."""
    del upper  # legacy kwarg; v2 uses one threshold only
    thr = clarify_threshold() if threshold is None else threshold
    if lower is not None:
        thr = lower
    if score is None or score < thr:
        return ClarifyBand.UNRELATED
    return ClarifyBand.CLARIFY


async def probe_query_score(lightrag: Any, query: str) -> dict[str, Any]:
    """Probe full query text against chunk VDB (query leg only)."""
    chunks_vdb = lightrag.chunks_vdb
    return await probe_chunk_vector_score(chunks_vdb, query)


def _purge_clarification_store() -> None:
    now = time.time()
    expired = [
        key
        for key, rec in _CLARIFICATION_STORE.items()
        if now - float(rec.get("created", 0)) > _STORE_TTL_SEC
    ]
    for key in expired:
        _CLARIFICATION_STORE.pop(key, None)


def _register_clarification(
    *,
    original_query: str,
    candidates: list[dict[str, Any]],
    band: ClarifyBand,
    relevance_score: float | None,
    threshold: float,
    generation: dict[str, Any],
) -> str:
    _purge_clarification_store()
    clarification_id = str(uuid.uuid4())
    _CLARIFICATION_STORE[clarification_id] = {
        "created": time.time(),
        "original_query": original_query,
        "candidates": {c["id"]: c["text"] for c in candidates},
        "band": band.value,
        "threshold": threshold,
    }
    return clarification_id


def validate_use_candidate(
    clarification_id: str | None,
    candidate_id: str | None,
    query_text: str,
) -> bool:
    cid = (clarification_id or "").strip()
    cand = (candidate_id or "").strip()
    text = (query_text or "").strip()
    if not cid or not cand or not text:
        return False
    rec = _CLARIFICATION_STORE.get(cid)
    if not rec:
        return False
    if time.time() - float(rec.get("created", 0)) > _STORE_TTL_SEC:
        _CLARIFICATION_STORE.pop(cid, None)
        return False
    candidates = rec.get("candidates")
    if not isinstance(candidates, dict):
        return False
    expected = candidates.get(cand)
    return isinstance(expected, str) and expected.strip() == text


def _normalize_candidate_line(line: str) -> str:
    text = (line or "").strip()
    text = re.sub(r"^[\s\d]+[.、)）]\s*", "", text)
    text = text.strip("\"'“”‘’ ")
    return text.strip()


def _parse_candidate_lines(raw: str, *, limit: int) -> list[str]:
    rows: list[str] = []
    for line in (raw or "").splitlines():
        cleaned = _normalize_candidate_line(line)
        if len(cleaned) < 4:
            continue
        if cleaned not in rows:
            rows.append(cleaned)
        if len(rows) >= limit:
            break
    return rows


async def _call_lightrag_llm(lightrag: Any, prompt: str, *, system_prompt: str) -> str:
    fn = getattr(lightrag, "llm_model_func", None)
    if not callable(fn):
        raise ClarifyGateError("lightrag.llm_model_func is not available")
    result = fn(
        prompt,
        system_prompt=system_prompt,
        history_messages=[],
        temperature=0.2,
    )
    if asyncio.iscoroutine(result):
        result = await result
    return str(result or "").strip()


def _format_top_hits(probe: dict[str, Any], *, limit: int = 3) -> str:
    parts: list[str] = []
    for row in probe.get("top_hits") or []:
        if not isinstance(row, dict):
            continue
        preview = (row.get("preview") or "").strip()
        if preview:
            parts.append(preview)
        if len(parts) >= limit:
            break
    return "\n".join(parts)


async def _generate_candidate_lines(
    lightrag: Any,
    *,
    query: str,
    probe: dict[str, Any],
    count: int,
    exclude: set[str],
) -> list[str]:
    k = max(count, 1)
    exclude_block = ""
    if exclude:
        exclude_block = "不要重复以下已有问句：\n" + "\n".join(f"- {x}" for x in sorted(exclude))

    context = _format_top_hits(probe)
    prompt = (
        f"用户原问较模糊或与手册表述不完全一致。根据用户原问和下列检索片段，"
        f"生成 {k} 条更明确的中文疑问句。\n"
        "问句须与片段主题一致，且适合在设备手册中检索到依据。\n"
        "不要回答问题，不要解释，每行只输出一条问句。\n"
        f"{exclude_block}\n\n"
        f"用户原问：{query}\n\n"
        f"检索片段：\n{context or '（无）'}"
    )
    system = "你是设备手册问答助手，只输出中文问句，每行一条。"

    raw = await _call_lightrag_llm(lightrag, prompt, system_prompt=system)
    return _parse_candidate_lines(raw, limit=k + 2)


async def _collect_answerable_candidates(
    lightrag: Any,
    *,
    query: str,
    probe: dict[str, Any],
    mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    strategy = clarify_candidate_strategy()
    k_target = clarify_candidate_k()
    max_rounds = clarify_candidate_max_rounds()
    seen: set[str] = {query.strip()}
    answerable: list[dict[str, Any]] = []
    rounds_used = 0
    probes_used = 0

    while rounds_used < max_rounds:
        if len(answerable) >= k_target:
            break

        rounds_used += 1
        if strategy == "first":
            batch_size = 1
        else:
            need = k_target - len(answerable)
            batch_size = max(need + 2, k_target)
        lines = await _generate_candidate_lines(
            lightrag,
            query=query,
            probe=probe,
            count=batch_size,
            exclude=seen,
        )

        for line in lines:
            if line in seen:
                continue
            seen.add(line)
            probes_used += 1
            retrieval = await probe_llm_retrieval(lightrag, line, mode=mode)
            if not retrieval.get("answerable"):
                continue
            answerable.append(
                {
                    "id": f"c{len(answerable) + 1}",
                    "text": line,
                    "chunk_count": int(retrieval.get("chunk_count") or 0),
                    "max_rerank_score": retrieval.get("max_rerank_score"),
                    "min_rerank_threshold": retrieval.get("min_rerank_threshold"),
                    "answerable": True,
                }
            )
            if strategy == "first" and answerable:
                break
            if len(answerable) >= k_target:
                break

    for idx, row in enumerate(answerable, start=1):
        row["id"] = f"c{idx}"

    meta = {
        "k_requested": k_target,
        "k_answerable": len(answerable),
        "rounds_used": rounds_used,
        "probes_used": probes_used,
        "strategy": strategy,
        "candidate_validation": "mix_llm_chunks",
        "min_rerank_threshold": clarify_candidate_min_rerank_score(),
    }
    return answerable, meta


def build_clarification_payload(
    *,
    clarification_id: str,
    original_query: str,
    relevance_score: float | None,
    band: ClarifyBand,
    threshold: float,
    candidates: list[dict[str, Any]],
    generation: dict[str, Any],
    unrelated: bool = False,
) -> dict[str, Any]:
    score_out = round(float(relevance_score), 4) if relevance_score is not None else None
    payload: dict[str, Any] = {
        "clarification_id": clarification_id,
        "original_query": original_query,
        "relevance_score": score_out,
        "band": band.value,
        "threshold": threshold,
        "threshold_lower": threshold,
        "threshold_upper": threshold,
        "candidates": candidates,
        "generation": generation,
    }
    if unrelated:
        payload["unrelated"] = True
        payload["unanswerable"] = True
        payload["message"] = CLARIFY_UNRELATED_MESSAGE
    else:
        payload["keep_original"] = {
            "label": "保持原问题",
            "text": original_query,
            "relevance_score": score_out,
            "clarify_choice": "keep_original",
        }
    return payload


async def evaluate_clarify_gate(
    lightrag: Any,
    query: str,
    *,
    mode: str | None = None,
    clarify_choice: str | None = None,
    clarification_id: str | None = None,
    candidate_id: str | None = None,
) -> _ClarifyResult:
    """Classify query or build clarification UI payload."""
    q = (query or "").strip()
    if not q:
        raise ClarifyGateError("empty query")

    choice = (clarify_choice or "").strip().lower()
    if choice == "keep_original":
        return ClarifyBypass("keep_original")
    if choice == "use_candidate":
        if not validate_use_candidate(clarification_id, candidate_id, q):
            raise ClarifyValidationError(
                "invalid clarification_id / candidate_id / query for use_candidate"
            )
        return ClarifyBypass("use_candidate")

    if not is_clarify_gate_enabled(mode):
        return ClarifyBypass("disabled")

    threshold = clarify_threshold()
    probe = await probe_query_score(lightrag, q)
    score_raw = probe.get("max_cosine_similarity")
    try:
        score = float(score_raw) if score_raw is not None else None
    except (TypeError, ValueError):
        score = None

    band = classify_query_relevance(score, threshold=threshold)
    if band == ClarifyBand.UNRELATED:
        clarification_id = _register_clarification(
            original_query=q,
            candidates=[],
            band=band,
            relevance_score=score,
            threshold=threshold,
            generation={"k_requested": 0, "k_answerable": 0, "reason": "below_threshold"},
        )
        data = build_clarification_payload(
            clarification_id=clarification_id,
            original_query=q,
            relevance_score=score,
            band=band,
            threshold=threshold,
            candidates=[],
            generation={"k_requested": 0, "k_answerable": 0, "reason": "below_threshold"},
            unrelated=True,
        )
        return ClarifyRequired(data)

    candidates, generation = await _collect_answerable_candidates(
        lightrag,
        query=q,
        probe=probe,
        mode=(mode or "mix").strip() or "mix",
    )

    clarification_id = _register_clarification(
        original_query=q,
        candidates=candidates,
        band=band,
        relevance_score=score,
        threshold=threshold,
        generation=generation,
    )
    data = build_clarification_payload(
        clarification_id=clarification_id,
        original_query=q,
        relevance_score=score,
        band=band,
        threshold=threshold,
        candidates=candidates,
        generation=generation,
        unrelated=False,
    )
    return ClarifyRequired(data)
