"""Clarification gate before mix aquery (v4, route 2).

Three-band gate on **final rerank score** (max among qualifying document chunks):
- ``final_score is None`` → reject (no chunk >= MIN_RERANK_SCORE).
- ``final_score > CLARIFY_DIRECT_RERANK_MIN`` (default 7) → direct aquery (reuse probe bundle).
- else → offer k LLM recommendations (**no per-candidate rerank**); original probe bundle
  is cached for **keep_original** injection.

See ``docs/澄清门控设计方案_v4.md``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
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
from raganything.pipeline_rerank import (
    release_cross_encoder,
    rerank_release_after_gate,
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

logger = logging.getLogger(__name__)


class ClarifyGateError(Exception):
    """Base error for clarification gate."""


class ClarifyValidationError(ClarifyGateError):
    """Invalid use_candidate / clarification_id pairing."""


@dataclass(frozen=True)
class ClarifyBypass:
    """Proceed to aquery without showing clarification UI."""

    reason: Literal["disabled", "direct", "use_candidate", "keep_original"]
    probe: RetrievalProbeResult | None = None
    gate_timing: dict[str, Any] | None = None
    cached_bundle: CachedQueryBundle | None = None


@dataclass(frozen=True)
class ClarifyRequired:
    """Return clarification or reject UI; do not run aquery."""

    data: dict[str, Any]
    gate_outcome: Literal["reject", "offer"] = "offer"


_ClarifyResult = ClarifyBypass | ClarifyRequired

_STORE_TTL_SEC = 3600
_CLARIFICATION_STORE: dict[str, dict[str, Any]] = {}
_CLARIFICATION_STORE_LOCK = threading.RLock()


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
    all_rerank_scores: list[float] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        rs = _chunk_rerank_score(chunk)
        if rs is not None:
            all_rerank_scores.append(rs)
            if rs >= min_thr:
                qualifying_scores.append(rs)

    max_rerank_any = round(max(all_rerank_scores), 4) if all_rerank_scores else None

    if not qualifying_scores:
        return {
            "answerable": False,
            "chunk_count": 0,
            "final_score": None,
            "max_rerank_score": None,
            "max_rerank_any": max_rerank_any,
            "llm_chunk_total": len(chunks),
            "scores_unavailable": bool(chunks) and not all_rerank_scores,
        }

    final_score = round(max(qualifying_scores), 4)
    return {
        "answerable": True,
        "chunk_count": len(qualifying_scores),
        "final_score": final_score,
        "max_rerank_score": final_score,
        "max_rerank_any": max_rerank_any,
        "llm_chunk_total": len(chunks),
        "scores_unavailable": False,
    }


def _probe_passes_direct_threshold(probe: RetrievalProbeResult) -> bool:
    fs = probe.final_score
    return fs is not None and fs > clarify_direct_rerank_min()


def _elapsed_s(start: float) -> float:
    return round(time.perf_counter() - start, 1)


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
    stats: dict[str, Any] = {}
    context_str: str | None = None
    raw_data: dict[str, Any] | None = None

    hooks_mod: Any | None = None
    try:
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        scripts = root / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        import query_progress_hooks as hooks_mod  # noqa: WPS433
    except ImportError:
        logger.warning(
            "query_progress_hooks not available; "
            "clarify probe uses aquery_data fallback"
        )
        hooks_mod = None

    try:
        if hooks_mod is None:
            raw = await lightrag.aquery_data(q, param)
            raw_data = raw if isinstance(raw, dict) else None
        else:
            gate_probe_scope = hooks_mod.gate_probe_scope
            get_last_probe_raw_data = hooks_mod.get_last_probe_raw_data
            get_llm_input_chunks = hooks_mod.get_llm_input_chunks
            get_rerank_pool_chunks = hooks_mod.get_rerank_pool_chunks
            get_rerank_pool_chunks_raw = hooks_mod.get_rerank_pool_chunks_raw
            get_retrieval_context = hooks_mod.get_retrieval_context
            progress_hooks_active = hooks_mod.progress_hooks_active
            query_progress_hooks = hooks_mod.query_progress_hooks

            def _probe_chunks_and_stats() -> (
                tuple[list[dict[str, Any]], dict[str, Any]]
            ):
                def _pick_best(
                    *candidates: tuple[list[dict[str, Any]], dict[str, Any]],
                ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
                    best_chunks, best_stats = candidates[0]
                    best_final = best_stats.get("final_score")
                    for cand_chunks, cand_stats in candidates[1:]:
                        cand_final = cand_stats.get("final_score")
                        if cand_final is None:
                            continue
                        if best_final is None or float(cand_final) > float(best_final):
                            best_chunks, best_stats = cand_chunks, cand_stats
                            best_final = cand_final
                    return best_chunks, best_stats

                llm_chunks = get_llm_input_chunks()
                pool = get_rerank_pool_chunks()
                raw_pool = get_rerank_pool_chunks_raw()
                candidates: list[tuple[list[dict[str, Any]], dict[str, Any]]] = [
                    (llm_chunks, _summarize_llm_chunks(llm_chunks, min_thr=min_thr)),
                ]
                if pool:
                    candidates.append(
                        (pool, _summarize_llm_chunks(pool, min_thr=min_thr))
                    )
                if raw_pool and raw_pool is not pool:
                    candidates.append(
                        (raw_pool, _summarize_llm_chunks(raw_pool, min_thr=min_thr))
                    )
                return _pick_best(*candidates)

            async def _run_probe() -> None:
                with gate_probe_scope():
                    await lightrag.aquery_data(q, param)

            if progress_hooks_active():
                await _run_probe()
                chunks, stats = _probe_chunks_and_stats()
                context_str = get_retrieval_context()
                raw_data = get_last_probe_raw_data()
            else:
                async with query_progress_hooks():
                    await _run_probe()
                    chunks, stats = _probe_chunks_and_stats()
                    context_str = get_retrieval_context()
                    raw_data = get_last_probe_raw_data()
    except Exception as exc:
        logger.warning("clarify retrieval probe failed: %s", exc, exc_info=True)
        chunks = []
        stats = {}
        context_str = None
        raw_data = None

    if not stats:
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
        max_rerank_any=stats.get("max_rerank_any"),
        bundle=bundle if stats.get("answerable") else None,
    )


def _purge_clarification_store() -> None:
    now = time.time()
    with _CLARIFICATION_STORE_LOCK:
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
    original_bundle: CachedQueryBundle | None = None,
) -> str:
    _purge_clarification_store()
    clarification_id = str(uuid.uuid4())
    with _CLARIFICATION_STORE_LOCK:
        _CLARIFICATION_STORE[clarification_id] = {
            "created": time.time(),
            "original_query": original_query,
            "original_bundle": original_bundle,
            "options": options,
            "candidates": {c["id"]: c["text"] for c in candidates},
            "generation": generation,
        }
    return clarification_id


def _get_clarification_record(clarification_id: str | None) -> dict[str, Any] | None:
    cid = (clarification_id or "").strip()
    if not cid:
        return None
    with _CLARIFICATION_STORE_LOCK:
        rec = _CLARIFICATION_STORE.get(cid)
        if not rec:
            return None
        if time.time() - float(rec.get("created", 0)) > _STORE_TTL_SEC:
            _CLARIFICATION_STORE.pop(cid, None)
            return None
        return rec


def consume_clarification_record(clarification_id: str | None) -> bool:
    """Remove an entire clarification session (TTL purge helper; not used on point-select)."""
    cid = (clarification_id or "").strip()
    if not cid:
        return False
    with _CLARIFICATION_STORE_LOCK:
        return _CLARIFICATION_STORE.pop(cid, None) is not None


def _consume_keep_original_bundle(clarification_id: str | None) -> None:
    """Drop only the cached original-query bundle; keep candidates for further picks."""
    with _CLARIFICATION_STORE_LOCK:
        rec = _get_clarification_record_unlocked(clarification_id)
        if rec is not None:
            rec["original_bundle"] = None
            rec["keep_original_consumed"] = True


def _get_clarification_record_unlocked(
    clarification_id: str | None,
) -> dict[str, Any] | None:
    """Lookup without taking the store lock (caller must hold ``_CLARIFICATION_STORE_LOCK``)."""
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


def _mark_candidate_used(
    clarification_id: str | None, candidate_id: str | None
) -> None:
    with _CLARIFICATION_STORE_LOCK:
        rec = _get_clarification_record_unlocked(clarification_id)
        if rec is None:
            return
        options = rec.get("options")
        if not isinstance(options, dict):
            return
        entry = options.get((candidate_id or "").strip())
        if isinstance(entry, dict):
            entry["used"] = True


def keep_original_consumed(clarification_id: str | None) -> bool:
    with _CLARIFICATION_STORE_LOCK:
        rec = _get_clarification_record_unlocked(clarification_id)
        return bool(rec and rec.get("keep_original_consumed"))


def candidate_already_used(
    clarification_id: str | None, candidate_id: str | None
) -> bool:
    with _CLARIFICATION_STORE_LOCK:
        rec = _get_clarification_record_unlocked(clarification_id)
        if not rec:
            return False
        options = rec.get("options")
        if not isinstance(options, dict):
            return False
        entry = options.get((candidate_id or "").strip())
        return isinstance(entry, dict) and bool(entry.get("used"))


def validate_keep_original(
    clarification_id: str | None,
    query_text: str,
) -> bool:
    rec = _get_clarification_record(clarification_id)
    if not rec:
        return False
    text = (query_text or "").strip()
    if not text:
        return False
    expected = (rec.get("original_query") or "").strip()
    return text == expected


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


def _peek_keep_original_bundle(
    clarification_id: str | None,
    query_text: str,
) -> CachedQueryBundle | None:
    if not validate_keep_original(clarification_id, query_text):
        return None
    rec = _get_clarification_record(clarification_id)
    if not rec:
        return None
    bundle = rec.get("original_bundle")
    return bundle if isinstance(bundle, CachedQueryBundle) else None


def _peek_candidate_bundle(
    clarification_id: str | None,
    candidate_id: str | None,
    query_text: str,
) -> CachedQueryBundle | None:
    if not validate_use_candidate(clarification_id, candidate_id, query_text):
        return None
    rec = _get_clarification_record(clarification_id)
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


def finalize_clarify_bypass_consumption(
    clarification_id: str | None,
    clarify_choice: str | None,
    *,
    candidate_id: str | None = None,
) -> None:
    """Mark keep_original / use_candidate consumed **after** a successful aquery.

    Call this only once the answer path succeeds; ``evaluate_clarify_gate`` only
    peeks bundles and does not consume them (so retries stay possible on failure).
    """
    choice = (clarify_choice or "").strip().lower()
    if choice == "keep_original":
        _consume_keep_original_bundle(clarification_id)
    elif choice == "use_candidate":
        _mark_candidate_used(clarification_id, candidate_id)


def resolve_clarify_bundle(
    clarification_id: str | None,
    clarify_choice: str,
    query_text: str,
    candidate_id: str | None = None,
) -> CachedQueryBundle | None:
    choice = (clarify_choice or "").strip().lower()
    if choice == "keep_original":
        return _peek_keep_original_bundle(clarification_id, query_text)
    if choice == "use_candidate":
        return _peek_candidate_bundle(clarification_id, candidate_id, query_text)
    return None


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
    parse_limit: int | None = None,
) -> list[str]:
    k = max(count, 1)
    exclude_block = ""
    if exclude:
        exclude_block = "不要重复以下已有问句：\n" + "\n".join(
            f"- {x}" for x in sorted(exclude)
        )

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
    return _parse_candidate_lines(
        raw, limit=parse_limit if parse_limit is not None else k + 2
    )


def _candidate_generation_batch_size(
    *,
    strategy: str,
    k_target: int,
    answerable_count: int,
) -> int:
    if strategy == "first":
        return 1
    return max(k_target - answerable_count, 1)


def _append_llm_only_candidate(
    *,
    answerable: list[dict[str, Any]],
    option_entries: dict[str, dict[str, Any]],
    line: str,
    direct_min: float,
) -> None:
    cid = f"c{len(answerable) + 1}"
    answerable.append(
        {
            "id": cid,
            "text": line,
            "chunk_count": None,
            "final_score": None,
            "max_rerank_score": None,
            "min_rerank_threshold": clarify_candidate_min_rerank_score(),
            "direct_rerank_min": direct_min,
        }
    )
    option_entries[cid] = {"query": line, "bundle": None}


async def _collect_high_confidence_candidates(
    lightrag: Any,
    *,
    query: str,
    original_bundle: CachedQueryBundle | None,
    mode: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    del mode  # recommendations are LLM-only; no per-candidate retrieval/rerank
    strategy = clarify_candidate_strategy()
    k_target = clarify_candidate_k()
    max_rounds = clarify_candidate_max_rounds()
    direct_min = clarify_direct_rerank_min()
    try:
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        scripts = root / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        from query_progress_hooks import PHASE_CLARIFY, emit_query_phase  # noqa: WPS433

        await emit_query_phase(PHASE_CLARIFY)
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("clarify phase emit skipped: %s", exc)
    seen: set[str] = {query.strip()}
    answerable: list[dict[str, Any]] = []
    option_entries: dict[str, dict[str, Any]] = {}
    rounds_used = 0
    gen_rounds: list[dict[str, Any]] = []
    while rounds_used < max_rounds and len(answerable) < k_target:
        rounds_used += 1
        batch_size = _candidate_generation_batch_size(
            strategy=strategy,
            k_target=k_target,
            answerable_count=len(answerable),
        )

        t_gen = time.perf_counter()
        lines = await _generate_candidate_lines(
            lightrag,
            query=query,
            bundle=original_bundle,
            count=batch_size,
            exclude=seen,
            parse_limit=batch_size,
        )
        gen_rounds.append(
            {
                "round": rounds_used,
                "duration_s": _elapsed_s(t_gen),
                "lines_requested": batch_size,
                "lines_returned": len(lines),
            }
        )

        for line in lines:
            if line in seen:
                continue
            seen.add(line)
            _append_llm_only_candidate(
                answerable=answerable,
                option_entries=option_entries,
                line=line,
                direct_min=direct_min,
            )
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

    gen_total = round(sum(float(r.get("duration_s") or 0) for r in gen_rounds), 1)
    meta = {
        "k_requested": k_target,
        "k_answerable": len(answerable),
        "rounds_used": rounds_used,
        "probes_used": 0,
        "strategy": strategy,
        "candidate_validation": "llm_only",
        "min_rerank_threshold": clarify_candidate_min_rerank_score(),
        "direct_rerank_min": direct_min,
        "reason": (
            "filled_k"
            if len(answerable) >= k_target
            else "cannot_fill_high_confidence_candidates"
        ),
        "gate_timing": {
            "candidate_gen_rounds": gen_rounds,
            "candidate_gen_total_s": gen_total,
            "clarify_loop_total_s": gen_total,
        },
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
    keep_original: dict[str, Any] | None = None,
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
        if keep_original is not None:
            payload["keep_original"] = keep_original
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
        if candidate_already_used(clarification_id, candidate_id):
            raise ClarifyValidationError("该推荐问已回答，请选择其他选项或重新提问")
        if not validate_use_candidate(clarification_id, candidate_id, q):
            raise ClarifyValidationError(
                "invalid clarification_id / candidate_id / query for use_candidate"
            )
        bundle = _peek_candidate_bundle(clarification_id, candidate_id, q)
        return ClarifyBypass("use_candidate", cached_bundle=bundle)
    if choice == "keep_original":
        if not validate_keep_original(clarification_id, q):
            raise ClarifyValidationError(
                "invalid clarification_id / query for keep_original"
            )
        if keep_original_consumed(clarification_id):
            raise ClarifyValidationError("原问已回答，请选择推荐问或重新提问")
        bundle = _peek_keep_original_bundle(clarification_id, q)
        return ClarifyBypass("keep_original", cached_bundle=bundle)

    if not is_clarify_gate_enabled(mode):
        return ClarifyBypass("disabled")

    try:
        return await _evaluate_clarify_gate_probed(
            lightrag,
            q,
            mode=mode,
        )
    finally:
        if rerank_release_after_gate():
            release_cross_encoder()


async def _evaluate_clarify_gate_probed(
    lightrag: Any,
    q: str,
    *,
    mode: str | None = None,
) -> _ClarifyResult:
    query_mode = (mode or "mix").strip() or "mix"
    t_orig = time.perf_counter()
    original_probe = await probe_llm_retrieval_full(lightrag, q, mode=query_mode)
    orig_probe_s = _elapsed_s(t_orig)
    final_score = original_probe.final_score
    direct_min = clarify_direct_rerank_min()
    gate_timing: dict[str, Any] = {"original_probe_s": orig_probe_s}

    if final_score is None:
        generation = {
            "k_requested": clarify_candidate_k(),
            "k_answerable": 0,
            "rounds_used": 0,
            "probes_used": 0,
            "direct_rerank_min": direct_min,
            "reason": "no_final_chunks",
            "gate_timing": {**gate_timing, "gate_total_s": orig_probe_s},
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
        gate_timing["gate_total_s"] = orig_probe_s
        return ClarifyBypass("direct", probe=original_probe, gate_timing=gate_timing)

    (
        candidates,
        candidate_options,
        generation,
    ) = await _collect_high_confidence_candidates(
        lightrag,
        query=q,
        original_bundle=original_probe.bundle,
        mode=query_mode,
    )
    loop_timing = dict(generation.get("gate_timing") or {})
    gate_timing = {
        **gate_timing,
        **loop_timing,
        "gate_total_s": round(
            orig_probe_s + float(loop_timing.get("clarify_loop_total_s") or 0), 1
        ),
    }
    generation["gate_timing"] = gate_timing

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
        original_bundle=original_probe.bundle,
    )
    keep_original: dict[str, Any] | None = None
    if original_probe.answerable and int(original_probe.chunk_count or 0) > 0:
        keep_original = {
            "query": q,
            "clarify_choice": "keep_original",
            "final_score": original_probe.final_score,
            "bundle_cached": original_probe.bundle is not None,
        }
    data = build_clarification_payload(
        clarification_id=clarification_id_out,
        original_query=q,
        gate_outcome="offer",
        gate_reason="filled_k",
        original_probe=_probe_summary(original_probe),
        candidates=candidates,
        generation=generation,
        keep_original=keep_original,
    )
    return ClarifyRequired(data, gate_outcome="offer")
