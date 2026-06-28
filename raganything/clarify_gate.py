"""Clarification gate before mix aquery (v4).

Three-band gate on **final rerank score** (max among qualifying document chunks):
- ``final_score is None`` → reject (no chunk >= MIN_RERANK_SCORE).
- ``final_score > CLARIFY_DIRECT_RERANK_MIN`` (default 7) → direct aquery.
- else → offer k recommendations; each candidate must probe with ``final_score >`` threshold.

See ``docs/澄清门控设计方案_v4.md``.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from lightrag import QueryParam

from raganything.clarify_context import (
    CachedQueryBundle,
    RetrievalProbeResult,
    build_cached_bundle,
    format_chunk_previews,
)

CLARIFY_UNRELATED_MESSAGE = (
    "您的问题与当前知识库内容关联度较低，暂无法基于知识库作答。"
    "请尝试换种说法，或联系技术支持。"
)

CLARIFY_CANNOT_FILL_MESSAGE = (
    "暂无法为您的问法生成足够明确、且能在知识库中找到依据的推荐问法，"
    "请换种说法或联系技术支持。"
)

GATE_VERSION = "v4"


class ClarifyGateError(Exception):
    """Base error for clarification gate."""


class ClarifyValidationError(ClarifyGateError):
    """Invalid use_candidate / clarification_id pairing."""


@dataclass(frozen=True)
class ClarifyBypass:
    """Proceed to aquery without showing clarification UI."""

    reason: Literal["disabled", "direct", "use_candidate"]
    probe: RetrievalProbeResult | None = None


@dataclass(frozen=True)
class ClarifyRequired:
    """Return clarification or reject UI; do not run aquery."""

    data: dict[str, Any]
    gate_outcome: Literal["reject", "offer"] = "offer"


_ClarifyResult = ClarifyBypass | ClarifyRequired

_STORE_TTL_SEC = 3600
_CLARIFICATION_STORE: dict[str, dict[str, Any]] = {}


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
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


def clarify_candidate_k() -> int:
    return max(1, _env_int("CLARIFY_CANDIDATE_K", 2))


def clarify_candidate_max_rounds() -> int:
    return max(1, _env_int("CLARIFY_CANDIDATE_MAX_ROUNDS", 3))


def clarify_candidate_max_probes() -> int | None:
    raw = (os.getenv("CLARIFY_CANDIDATE_MAX_PROBES") or "").strip()
    if not raw:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        return None


def clarify_candidate_strategy() -> str:
    return (os.getenv("CLARIFY_CANDIDATE_STRATEGY") or "fill_k").strip().lower()


def clarify_candidate_min_rerank_score() -> float:
    raw = (
        os.getenv("CLARIFY_CANDIDATE_MIN_RERANK_SCORE")
        or os.getenv("MIN_RERANK_SCORE")
        or "0.28"
    ).strip()
    try:
        return float(raw)
    except ValueError:
        return 0.28


def clarify_direct_rerank_min() -> float:
    return _env_float("CLARIFY_DIRECT_RERANK_MIN", 7.0)


def _chunk_rerank_score(doc: dict[str, Any]) -> float | None:
    """CrossEncoder ``rerank_score`` only (not vector ``score`` fallback)."""
    raw = doc.get("rerank_score")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
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
    qualifying_scores: list[float] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        rs = _chunk_rerank_score(chunk)
        if rs is not None and rs >= min_thr:
            qualifying_scores.append(rs)

    if not qualifying_scores:
        return {
            "answerable": False,
            "chunk_count": 0,
            "final_score": None,
            "max_rerank_score": None,
            "llm_chunk_total": len(chunks),
            "scores_unavailable": bool(chunks),
        }

    final_score = round(max(qualifying_scores), 4)
    return {
        "answerable": True,
        "chunk_count": len(qualifying_scores),
        "final_score": final_score,
        "max_rerank_score": final_score,
        "llm_chunk_total": len(chunks),
        "scores_unavailable": False,
    }


def _probe_passes_direct_threshold(probe: RetrievalProbeResult) -> bool:
    fs = probe.final_score
    return fs is not None and fs > clarify_direct_rerank_min()


async def probe_llm_retrieval(
    lightrag: Any,
    query: str,
    *,
    mode: str = "mix",
) -> dict[str, Any]:
    """Run mix retrieval probe and return chunk stats."""
    result = await probe_llm_retrieval_full(lightrag, query, mode=mode)
    return result.as_stats()


async def probe_llm_retrieval_full(
    lightrag: Any,
    query: str,
    *,
    mode: str = "mix",
) -> RetrievalProbeResult:
    """Run Web-equivalent mix retrieval and build a KG-filtered CachedQueryBundle."""
    q = (query or "").strip()
    min_thr = clarify_candidate_min_rerank_score()
    direct_min = clarify_direct_rerank_min()
    empty = RetrievalProbeResult(
        query=q,
        mode=mode,
        answerable=False,
        chunk_count=0,
        max_rerank_score=None,
        min_rerank_threshold=min_thr,
        llm_chunk_total=0,
        scores_unavailable=False,
        final_score=None,
        direct_rerank_min=direct_min,
        bundle=None,
    )
    if not q:
        return empty

    param = _probe_query_param(mode)
    chunks: list[dict[str, Any]] = []
    context_str: str | None = None
    raw_data: dict[str, Any] | None = None
    try:
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        scripts = root / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        from query_progress_hooks import (  # noqa: WPS433
            get_last_probe_raw_data,
            get_llm_input_chunks,
            get_retrieval_context,
            query_progress_hooks,
        )

        async with query_progress_hooks():
            await lightrag.aquery_data(q, param)
            chunks = get_llm_input_chunks()
            context_str = get_retrieval_context()
            raw_data = get_last_probe_raw_data()
    except Exception:
        chunks = []

    stats = _summarize_llm_chunks(chunks, min_thr=min_thr)
    bundle = build_cached_bundle(q, context_str=context_str, raw_data=raw_data)
    if stats.get("answerable") and bundle is None and raw_data:
        bundle = build_cached_bundle(q, context_str=context_str, raw_data=raw_data)

    final_score = stats.get("final_score")
    return RetrievalProbeResult(
        query=q,
        mode=param.mode,
        answerable=bool(stats.get("answerable")),
        chunk_count=int(stats.get("chunk_count") or 0),
        max_rerank_score=final_score,
        min_rerank_threshold=min_thr,
        llm_chunk_total=int(stats.get("llm_chunk_total") or 0),
        scores_unavailable=bool(stats.get("scores_unavailable")),
        final_score=final_score,
        direct_rerank_min=direct_min,
        bundle=bundle if stats.get("answerable") else None,
    )


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
    options: dict[str, dict[str, Any]],
    generation: dict[str, Any],
) -> str:
    _purge_clarification_store()
    clarification_id = str(uuid.uuid4())
    _CLARIFICATION_STORE[clarification_id] = {
        "created": time.time(),
        "original_query": original_query,
        "options": options,
        "candidates": {c["id"]: c["text"] for c in candidates},
        "generation": generation,
    }
    return clarification_id


def _get_clarification_record(clarification_id: str | None) -> dict[str, Any] | None:
    cid = (clarification_id or "").strip()
    if not cid:
        return None
    rec = _CLARIFICATION_STORE.get(cid)
    if not rec:
        return None
    if time.time() - float(rec.get("created", 0)) > _STORE_TTL_SEC:
        _CLARIFICATION_STORE.pop(cid, None)
        return None
    return rec


def validate_use_candidate(
    clarification_id: str | None,
    candidate_id: str | None,
    query_text: str,
) -> bool:
    rec = _get_clarification_record(clarification_id)
    if not rec:
        return False
    cand = (candidate_id or "").strip()
    text = (query_text or "").strip()
    if not cand or not text:
        return False
    candidates = rec.get("candidates")
    if not isinstance(candidates, dict):
        return False
    expected = candidates.get(cand)
    return isinstance(expected, str) and expected.strip() == text


def resolve_clarify_bundle(
    clarification_id: str | None,
    clarify_choice: str,
    query_text: str,
    candidate_id: str | None = None,
) -> CachedQueryBundle | None:
    choice = (clarify_choice or "").strip().lower()
    if choice != "use_candidate":
        return None
    cid = (clarification_id or "").strip()
    if not validate_use_candidate(cid, candidate_id, query_text):
        return None
    rec = _get_clarification_record(cid)
    if not rec:
        return None
    options = rec.get("options")
    if not isinstance(options, dict):
        return None
    entry = options.get((candidate_id or "").strip())
    if not isinstance(entry, dict):
        return None
    bundle = entry.get("bundle")
    return bundle if isinstance(bundle, CachedQueryBundle) else None


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


async def _generate_candidate_lines(
    lightrag: Any,
    *,
    query: str,
    bundle: CachedQueryBundle | None,
    count: int,
    exclude: set[str],
) -> list[str]:
    k = max(count, 1)
    exclude_block = ""
    if exclude:
        exclude_block = "不要重复以下已有问句：\n" + "\n".join(f"- {x}" for x in sorted(exclude))

    context = ""
    if bundle is not None:
        context = format_chunk_previews(bundle.document_chunks)

    prompt = (
        f"用户原问表述不够清楚。请生成 {k} 条「更具体、但仍与原问同一意图」的中文疑问句，"
        "供用户点选澄清。\n"
        "规则：\n"
        "1. 必须保留原问的核心：问的对象、问的属性（如型号/参数/原因/步骤/地址等）不能更换。\n"
        "2. 每条推荐问应能看作原问的细化或改写，而不是换一个新话题。\n"
        "3. 检索片段仅用于对齐手册中的术语与表述，不得因片段内容而改换用户意图。\n"
        "4. 不要回答问题，不要解释，每行只输出一条完整问句。\n"
        f"{exclude_block}\n\n"
        f"用户原问：{query}\n\n"
        f"检索片段（仅供术语参考）：\n{context or '（无）'}"
    )
    system = (
        "你是设备手册问答助手。只输出中文疑问句，每行一条。"
        "推荐问必须与原问同一意图，仅表述更具体，不得偏离用户要问的事项。"
    )

    raw = await _call_lightrag_llm(lightrag, prompt, system_prompt=system)
    return _parse_candidate_lines(raw, limit=k + 2)


async def _collect_high_confidence_candidates(
    lightrag: Any,
    *,
    query: str,
    original_bundle: CachedQueryBundle | None,
    mode: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    strategy = clarify_candidate_strategy()
    k_target = clarify_candidate_k()
    max_rounds = clarify_candidate_max_rounds()
    max_probes = clarify_candidate_max_probes()
    direct_min = clarify_direct_rerank_min()
    seen: set[str] = {query.strip()}
    answerable: list[dict[str, Any]] = []
    option_entries: dict[str, dict[str, Any]] = {}
    rounds_used = 0
    probes_used = 0
    while rounds_used < max_rounds and len(answerable) < k_target:
        if max_probes is not None and probes_used >= max_probes:
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
            bundle=original_bundle,
            count=batch_size,
            exclude=seen,
        )

        for line in lines:
            if max_probes is not None and probes_used >= max_probes:
                break
            if line in seen:
                continue
            seen.add(line)
            probes_used += 1
            probe = await probe_llm_retrieval_full(lightrag, line, mode=mode)
            if not _probe_passes_direct_threshold(probe) or probe.bundle is None:
                continue

            cid = f"c{len(answerable) + 1}"
            answerable.append(
                {
                    "id": cid,
                    "text": line,
                    "chunk_count": probe.chunk_count,
                    "final_score": probe.final_score,
                    "max_rerank_score": probe.final_score,
                    "min_rerank_threshold": probe.min_rerank_threshold,
                    "direct_rerank_min": direct_min,
                }
            )
            option_entries[cid] = {
                "query": line,
                "bundle": probe.bundle,
            }

            if strategy == "first" and answerable:
                break
            if len(answerable) >= k_target:
                break

    rekeyed_options: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(answerable, start=1):
        old_id = row["id"]
        new_id = f"c{idx}"
        row["id"] = new_id
        if old_id in option_entries:
            rekeyed_options[new_id] = option_entries[old_id]
    option_entries = rekeyed_options

    meta = {
        "k_requested": k_target,
        "k_answerable": len(answerable),
        "rounds_used": rounds_used,
        "probes_used": probes_used,
        "strategy": strategy,
        "candidate_validation": "final_score_gt_direct_min",
        "min_rerank_threshold": clarify_candidate_min_rerank_score(),
        "direct_rerank_min": direct_min,
        "reason": (
            "filled_k"
            if len(answerable) >= k_target
            else "cannot_fill_high_confidence_candidates"
        ),
    }
    return answerable, option_entries, meta


def _probe_summary(probe: RetrievalProbeResult) -> dict[str, Any]:
    return probe.as_stats()


def build_clarification_payload(
    *,
    clarification_id: str,
    original_query: str,
    gate_outcome: Literal["reject", "offer"],
    gate_reason: str,
    original_probe: dict[str, Any],
    candidates: list[dict[str, Any]],
    generation: dict[str, Any],
    message: str | None = None,
) -> dict[str, Any]:
    reject = gate_outcome == "reject"
    payload: dict[str, Any] = {
        "clarification_id": clarification_id,
        "original_query": original_query,
        "gate_version": GATE_VERSION,
        "gate_outcome": gate_outcome,
        "gate_reason": gate_reason,
        "original_probe": original_probe,
        "candidates": candidates,
        "generation": generation,
        "keep_original": None,
    }
    if reject:
        payload["unrelated"] = True
        payload["unanswerable"] = True
        payload["message"] = message or CLARIFY_UNRELATED_MESSAGE
    else:
        payload["unrelated"] = False
        payload["unanswerable"] = False
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
    """Classify query by final rerank score or build clarification UI payload."""
    q = (query or "").strip()
    if not q:
        raise ClarifyGateError("empty query")

    choice = (clarify_choice or "").strip().lower()
    if choice == "use_candidate":
        if not validate_use_candidate(clarification_id, candidate_id, q):
            raise ClarifyValidationError(
                "invalid clarification_id / candidate_id / query for use_candidate"
            )
        return ClarifyBypass("use_candidate")

    if not is_clarify_gate_enabled(mode):
        return ClarifyBypass("disabled")

    query_mode = (mode or "mix").strip() or "mix"
    original_probe = await probe_llm_retrieval_full(lightrag, q, mode=query_mode)
    final_score = original_probe.final_score
    direct_min = clarify_direct_rerank_min()

    if final_score is None:
        generation = {
            "k_requested": clarify_candidate_k(),
            "k_answerable": 0,
            "rounds_used": 0,
            "probes_used": 0,
            "direct_rerank_min": direct_min,
            "reason": "no_final_chunks",
        }
        clarification_id_out = _register_clarification(
            original_query=q,
            candidates=[],
            options={},
            generation=generation,
        )
        data = build_clarification_payload(
            clarification_id=clarification_id_out,
            original_query=q,
            gate_outcome="reject",
            gate_reason="no_final_chunks",
            original_probe=_probe_summary(original_probe),
            candidates=[],
            generation=generation,
            message=CLARIFY_UNRELATED_MESSAGE,
        )
        return ClarifyRequired(data, gate_outcome="reject")

    if final_score > direct_min:
        return ClarifyBypass("direct", probe=original_probe)

    candidates, candidate_options, generation = await _collect_high_confidence_candidates(
        lightrag,
        query=q,
        original_bundle=original_probe.bundle,
        mode=query_mode,
    )

    k_target = clarify_candidate_k()
    if len(candidates) < k_target:
        clarification_id_out = _register_clarification(
            original_query=q,
            candidates=[],
            options={},
            generation=generation,
        )
        data = build_clarification_payload(
            clarification_id=clarification_id_out,
            original_query=q,
            gate_outcome="reject",
            gate_reason="cannot_fill_high_confidence_candidates",
            original_probe=_probe_summary(original_probe),
            candidates=[],
            generation=generation,
            message=CLARIFY_CANNOT_FILL_MESSAGE,
        )
        return ClarifyRequired(data, gate_outcome="reject")

    clarification_id_out = _register_clarification(
        original_query=q,
        candidates=candidates,
        options=candidate_options,
        generation=generation,
    )
    data = build_clarification_payload(
        clarification_id=clarification_id_out,
        original_query=q,
        gate_outcome="offer",
        gate_reason="filled_k",
        original_probe=_probe_summary(original_probe),
        candidates=candidates,
        generation=generation,
    )
    return ClarifyRequired(data, gate_outcome="offer")
