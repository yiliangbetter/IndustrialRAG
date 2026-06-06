"""Query clarification gate: assess retrieval fit, rewrite vague queries, user choice."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

StatusCallback = Callable[[str], Awaitable[None] | None]

logger = logging.getLogger(__name__)

Band = Literal["clear", "vague", "unrelated"]
ClarifyAction = Literal["proceed", "clarify", "abort"]

USE_ORIGINAL = "use_original"

_SESSION_TTL_SEC = 1800
_sessions: dict[str, dict[str, Any]] = {}


@dataclass
class ClarifyConfig:
    enabled: bool = True
    threshold_lower: float = 0.28
    threshold_upper: float = 0.45
    k: int = 4
    strategy: str = "all"
    score_top_k: int = 3
    preview_max_chars: int = 2400
    preview_chunk_limit: int = 4
    llm_specificity_check: bool = True


@dataclass
class AssessResult:
    query: str
    score: float
    score_top1: float
    band: Band
    chunk_count: int
    preview_snippets: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class CandidateOption:
    query: str
    score: float
    score_top1: float
    label: str = ""


@dataclass
class ClarificationOutcome:
    action: ClarifyAction
    band: Band
    original_query: str
    original_score: float = 0.0
    score_top1: float = 0.0
    effective_query: str = ""
    options: list[CandidateOption] = field(default_factory=list)
    preview_snippets: list[str] = field(default_factory=list)
    message: str = ""
    session_id: str = ""


def clarify_enabled() -> bool:
    return load_clarify_config().enabled


def load_clarify_config() -> ClarifyConfig:
    def _bool(name: str, default: bool) -> bool:
        raw = (os.getenv(name) or "").strip().lower()
        if not raw:
            return default
        return raw in ("1", "true", "yes", "on")

    def _float(name: str, default: float) -> float:
        raw = (os.getenv(name) or "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    def _int(name: str, default: int) -> int:
        raw = (os.getenv(name) or "").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    lower = _float("QUERY_SCORE_THRESHOLD_LOWER", 0.28)
    upper = _float("QUERY_SCORE_THRESHOLD_UPPER", 0.45)
    if lower >= upper:
        upper = lower + 0.12

    return ClarifyConfig(
        enabled=_bool("QUERY_CLARIFY_ENABLED", True),
        threshold_lower=lower,
        threshold_upper=upper,
        k=max(1, min(8, _int("QUERY_CLARIFY_K", 4))),
        strategy=(os.getenv("QUERY_CLARIFY_STRATEGY") or "all").strip().lower(),
        score_top_k=max(1, _int("QUERY_SCORE_TOP_K", 3)),
        llm_specificity_check=_bool("QUERY_CLARIFY_LLM_SPECIFICITY_CHECK", True),
    )


def band_from_score(score: float, cfg: ClarifyConfig) -> Band:
    if score >= cfg.threshold_upper:
        return "clear"
    if score < cfg.threshold_lower:
        return "unrelated"
    return "vague"


def assess_query_param_from_env() -> Any:
    """Light retrieval for clarify gate only (default naive, smaller top_k)."""
    from raganything.pipeline_rerank import query_param_from_env  # noqa: WPS433

    assess_mode = (os.getenv("QUERY_CLARIFY_ASSESS_MODE") or "naive").strip()
    param = query_param_from_env(mode=assess_mode)

    def _int(name: str, default: int) -> int:
        raw = (os.getenv(name) or "").strip()
        try:
            return int(raw)
        except ValueError:
            return default

    param.top_k = _int("QUERY_CLARIFY_TOP_K", 12)
    param.chunk_top_k = _int("QUERY_CLARIFY_CHUNK_TOP_K", 12)
    return param


async def _emit_status(on_status: StatusCallback | None, text: str) -> None:
    if on_status is None:
        return
    result = on_status(text)
    if asyncio.iscoroutine(result):
        await result


def aggregate_rerank_scores(
    docs: list[dict], *, top_k: int = 3
) -> tuple[float, float]:
    """Return (top3_avg, top1) from chunk dicts."""
    scores: list[float] = []
    for doc in docs:
        raw = doc.get("rerank_score")
        if raw is None:
            continue
        try:
            scores.append(float(raw))
        except (TypeError, ValueError):
            continue
    if not scores:
        return 0.0, 0.0
    scores.sort(reverse=True)
    top1 = scores[0]
    head = scores[:top_k]
    avg = sum(head) / len(head)
    return avg, top1


def _chunks_to_docs(chunks: list[dict]) -> list[dict]:
    docs: list[dict] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        content = str(chunk.get("content") or "").strip()
        if not content:
            continue
        doc = dict(chunk)
        doc.setdefault("content", content)
        docs.append(doc)
    return docs


def build_preview_snippets(
    chunks: list[dict], *, max_chars: int = 2400, limit: int = 4
) -> list[str]:
    out: list[str] = []
    used = 0
    for chunk in chunks[:limit]:
        text = str(chunk.get("content") or "").strip()
        if not text:
            continue
        text = re.sub(r"\s+", " ", text)[:600]
        fp = str(chunk.get("file_path") or "").strip()
        prefix = f"[{Path(fp).name}] " if fp else ""
        piece = prefix + text
        if used + len(piece) > max_chars:
            break
        out.append(piece)
        used += len(piece)
    return out


def _parse_queries_json(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    m = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", text)
    if m:
        text = m.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import json_repair

            data = json_repair.loads(text)
        except Exception:
            return []
    if isinstance(data, dict):
        queries = data.get("queries") or data.get("questions") or []
    elif isinstance(data, list):
        queries = data
    else:
        return []
    out: list[str] = []
    for item in queries:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, dict):
            q = str(item.get("query") or item.get("question") or "").strip()
            if q:
                out.append(q)
    return out


def _parse_specificity_json(text: str) -> bool | None:
    """Parse LLM JSON for specific_enough; None if unparseable."""
    text = (text or "").strip()
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        text = m.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import json_repair

            data = json_repair.loads(text)
        except Exception:
            return None
    if not isinstance(data, dict):
        return None
    raw = data.get("specific_enough")
    if raw is None:
        raw = data.get("specific")
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        low = raw.strip().lower()
        if low in ("true", "yes", "1"):
            return True
        if low in ("false", "no", "0"):
            return False
    return None


async def llm_is_query_specific_enough(
    rag: Any,
    query: str,
    preview_snippets: list[str],
) -> bool | None:
    """LLM gate when retrieval score says clear but wording may still be vague. None = skip override."""
    func = getattr(rag, "llm_model_func", None)
    if func is None:
        return None
    ctx = "\n".join(f"- {s}" for s in preview_snippets[:6]) or "（无检索摘要）"
    system = (
        "你是设备维护保养手册的检索门控助手。"
        "判断用户问题是否已足够具体，可直接在手册中检索作答。"
        "足够具体：已明确设备/系统、部位或具体操作。"
        "不够具体：笼统问保养、清理、加油等，未指明对象或场景。"
        '只输出 JSON：{"specific_enough": true} 或 {"specific_enough": false}。'
    )
    user = f"用户问题：{query}\n\n检索摘要（供参考）：\n{ctx}"
    try:
        raw = await func(
            user,
            system_prompt=system,
            history_messages=[],
        )
    except Exception as exc:
        logger.warning("clarify LLM specificity check failed: %s", exc)
        return None
    if not isinstance(raw, str):
        raw = str(raw or "")
    return _parse_specificity_json(raw)


async def _llm_rewrite_queries(
    rag: Any,
    *,
    user_prompt: str,
    system_prompt: str,
) -> list[str]:
    func = getattr(rag, "llm_model_func", None)
    if func is None:
        return []
    try:
        raw = await func(
            user_prompt,
            system_prompt=system_prompt,
            history_messages=[],
        )
    except Exception as exc:
        logger.warning("clarify LLM rewrite failed: %s", exc)
        return []
    if not isinstance(raw, str):
        raw = str(raw or "")
    return _parse_queries_json(raw)


async def assess_query_fit(
    rag: Any,
    query: str,
    *,
    mode: str | None = None,
) -> AssessResult:
    """Lightweight retrieval + rerank; no answer LLM."""
    from raganything.pipeline_rerank import lightrag_global_config  # noqa: WPS433

    q = (query or "").strip()
    if not q:
        return AssessResult(
            query=q,
            score=0.0,
            score_top1=0.0,
            band="unrelated",
            chunk_count=0,
            error="empty query",
        )

    cfg = load_clarify_config()
    param = assess_query_param_from_env()
    lightrag = rag.lightrag
    global_config = lightrag_global_config(lightrag)

    try:
        data = await lightrag.aquery_data(q, param)
    except Exception as exc:
        logger.warning("assess_query_fit aquery_data failed: %s", exc)
        return AssessResult(
            query=q,
            score=0.0,
            score_top1=0.0,
            band="unrelated",
            chunk_count=0,
            error=str(exc),
        )

    if (data.get("status") or "").lower() not in ("success", "ok", ""):
        msg = str(data.get("message") or data.get("status") or "retrieval failed")
        return AssessResult(
            query=q,
            score=0.0,
            score_top1=0.0,
            band="unrelated",
            chunk_count=0,
            error=msg,
        )

    chunks = (data.get("data") or {}).get("chunks") or []
    docs = _chunks_to_docs(chunks)

    has_rerank = any(d.get("rerank_score") is not None for d in docs)
    if param.enable_rerank and docs and not has_rerank:
        import lightrag.utils as ut

        try:
            docs = await ut.apply_rerank_if_enabled(
                q,
                docs,
                global_config,
                enable_rerank=True,
                top_n=param.chunk_top_k or len(docs),
            )
        except Exception as exc:
            logger.warning("assess rerank failed: %s", exc)

    score, top1 = aggregate_rerank_scores(docs, top_k=cfg.score_top_k)
    band = band_from_score(score, cfg)
    previews = build_preview_snippets(
        docs or chunks,
        max_chars=cfg.preview_max_chars,
        limit=cfg.preview_chunk_limit,
    )

    return AssessResult(
        query=q,
        score=score,
        score_top1=top1,
        band=band,
        chunk_count=len(docs),
        preview_snippets=previews,
    )


async def generate_alternative_queries(
    rag: Any,
    query: str,
    *,
    band: Band,
    preview_snippets: list[str],
    k: int,
) -> list[str]:
    cfg = load_clarify_config()
    k = max(1, min(cfg.k, k))

    if band == "unrelated":
        system = (
            "你是工业设备维护保养手册的检索助手。"
            "用户问题可能与知识库匹配度低。请根据原问题生成更可检索的具体问句，"
            "须包含设备/部位/操作等可搜索信息，不要回答问题。"
            f'只输出 JSON：{{"queries": ["问句1", ...]}}，恰好 {k} 条，使用简体中文。'
        )
        user = f"用户原问题：{query}"
    else:
        ctx = "\n".join(f"- {s}" for s in preview_snippets[:6]) or "（无摘要）"
        system = (
            "你是工业设备维护保养手册的检索助手。用户问题较模糊，但检索到部分相关内容。"
            "请结合摘要，生成更具体、可精准检索手册的问句；不要回答问题。"
            f'只输出 JSON：{{"queries": ["问句1", ...]}}，恰好 {k} 条，使用简体中文。'
        )
        user = f"用户原问题：{query}\n\n检索摘要：\n{ctx}"

    queries = await _llm_rewrite_queries(rag, user_prompt=user, system_prompt=system)
    seen = {query.strip()}
    deduped: list[str] = []
    for item in queries:
        t = item.strip()
        if t and t not in seen:
            seen.add(t)
            deduped.append(t)
        if len(deduped) >= k:
            break
    return deduped


async def score_candidate_queries(
    rag: Any,
    queries: list[str],
    *,
    band: Band,
) -> list[CandidateOption]:
    cfg = load_clarify_config()
    threshold = (
        cfg.threshold_lower if band == "unrelated" else cfg.threshold_upper
    )

    async def _one(cand: str) -> CandidateOption:
        assess = await assess_query_fit(rag, cand)
        return CandidateOption(
            query=cand,
            score=assess.score,
            score_top1=assess.score_top1,
            label=_short_label(cand),
        )

    scored = list(await asyncio.gather(*[_one(c) for c in queries]))
    scored.sort(key=lambda o: o.score, reverse=True)

    if cfg.strategy == "first_hit":
        for opt in scored:
            if opt.score > threshold:
                return [opt]
        return scored[:1] if scored else []

    qualified = [o for o in scored if o.score > threshold]
    if qualified:
        return qualified
    return scored[: cfg.k]


def _short_label(query: str, max_len: int = 48) -> str:
    q = re.sub(r"\s+", " ", (query or "").strip())
    return q if len(q) <= max_len else q[: max_len - 1] + "…"


def _purge_sessions() -> None:
    now = time.time()
    dead = [
        sid
        for sid, row in _sessions.items()
        if now - float(row.get("created_at") or 0) > _SESSION_TTL_SEC
    ]
    for sid in dead:
        _sessions.pop(sid, None)


def store_clarification_session(
    *,
    original_query: str,
    band: Band,
    original_score: float,
    options: list[CandidateOption],
    preview_snippets: list[str],
) -> str:
    _purge_sessions()
    sid = uuid.uuid4().hex
    _sessions[sid] = {
        "original_query": original_query,
        "band": band,
        "original_score": original_score,
        "options": [o.query for o in options],
        "preview_snippets": preview_snippets,
        "created_at": time.time(),
    }
    return sid


def get_clarification_session(session_id: str | None) -> dict[str, Any] | None:
    if not session_id:
        return None
    _purge_sessions()
    return _sessions.get(session_id.strip())


def resolve_query_for_rag(
    original_query: str,
    clarification_choice: str | None,
    session_id: str | None,
) -> tuple[str, str]:
    """Return (effective_query, gate_reason). gate_reason: clear | use_original | chosen."""
    choice = (clarification_choice or "").strip()
    if not choice:
        return original_query.strip(), "clear"

    if choice == USE_ORIGINAL:
        return original_query.strip(), "use_original"

    session = get_clarification_session(session_id)
    if session and choice in session.get("options", []):
        return choice, "chosen"

    return choice, "chosen"


async def run_clarification_gate(
    rag: Any,
    query: str,
    *,
    mode: str | None = None,
    on_status: StatusCallback | None = None,
) -> ClarificationOutcome:
    """Decide whether to clarify or proceed to full RAG."""
    cfg = load_clarify_config()
    q = (query or "").strip()
    if not cfg.enabled:
        return ClarificationOutcome(
            action="proceed",
            band="clear",
            original_query=q,
            effective_query=q,
        )

    await _emit_status(on_status, "正在检索原问与知识库匹配度…")
    assess = await assess_query_fit(rag, q)
    if assess.error and assess.chunk_count == 0:
        logger.info("clarify assess error=%s, treating as unrelated", assess.error)

    effective_band: Band = assess.band
    if effective_band == "clear" and cfg.llm_specificity_check:
        await _emit_status(on_status, "正在判断问题是否足够具体…")
        specific = await llm_is_query_specific_enough(
            rag, q, assess.preview_snippets
        )
        if specific is False:
            effective_band = "vague"
            logger.info(
                "clarify LLM specificity: clear(score) -> vague for %r",
                q[:80],
            )

    logger.info(
        "clarify assess query=%r band=%s effective_band=%s score=%.4f top1=%.4f chunks=%d",
        q[:80],
        assess.band,
        effective_band,
        assess.score,
        assess.score_top1,
        assess.chunk_count,
    )

    if effective_band == "clear":
        return ClarificationOutcome(
            action="proceed",
            band="clear",
            original_query=q,
            original_score=assess.score,
            score_top1=assess.score_top1,
            effective_query=q,
        )

    await _emit_status(on_status, "正在生成更具体的推荐问法…")
    alts = await generate_alternative_queries(
        rag,
        q,
        band=effective_band,
        preview_snippets=assess.preview_snippets,
        k=cfg.k,
    )
    if not alts:
        return ClarificationOutcome(
            action="abort",
            band=effective_band,
            original_query=q,
            original_score=assess.score,
            score_top1=assess.score_top1,
            preview_snippets=assess.preview_snippets,
            message="未能生成更具体的问法，请补充设备型号、保养部位或操作名称后重试。",
            effective_query=q,
        )

    await _emit_status(on_status, "正在评估推荐问法与手册匹配度…")
    options = await score_candidate_queries(rag, alts, band=effective_band)
    min_score = (
        cfg.threshold_lower
        if effective_band == "unrelated"
        else cfg.threshold_upper
    )
    qualified = [o for o in options if o.score > min_score]

    if qualified:
        sid = store_clarification_session(
            original_query=q,
            band=effective_band,
            original_score=assess.score,
            options=qualified,
            preview_snippets=assess.preview_snippets,
        )
        return ClarificationOutcome(
            action="clarify",
            band=effective_band,
            original_query=q,
            original_score=assess.score,
            score_top1=assess.score_top1,
            options=qualified,
            preview_snippets=assess.preview_snippets,
            session_id=sid,
        )

    if options:
        sid = store_clarification_session(
            original_query=q,
            band=effective_band,
            original_score=assess.score,
            options=options,
            preview_snippets=assess.preview_snippets,
        )
        msg = (
            "暂未找到与手册高度匹配的问法，您可选择下列推荐问句，或仍用原问题继续检索。"
        )
        return ClarificationOutcome(
            action="clarify",
            band=effective_band,
            original_query=q,
            original_score=assess.score,
            score_top1=assess.score_top1,
            options=options,
            preview_snippets=assess.preview_snippets,
            session_id=sid,
            message=msg,
        )

    return ClarificationOutcome(
        action="abort",
        band=effective_band,
        original_query=q,
        original_score=assess.score,
        score_top1=assess.score_top1,
        preview_snippets=assess.preview_snippets,
        message="未能找到更贴切的问题，请补充机型、部位或具体操作后重试。",
        effective_query=q,
    )


def clarification_to_sse_payload(outcome: ClarificationOutcome) -> dict[str, Any]:
    return {
        "type": "clarification",
        "band": outcome.band,
        "original_query": outcome.original_query,
        "original_score": round(outcome.original_score, 4),
        "score_top1": round(outcome.score_top1, 4),
        "session_id": outcome.session_id,
        "message": outcome.message or "",
        "preview_snippets": outcome.preview_snippets[:4],
        "options": [
            {
                "query": o.query,
                "score": round(o.score, 4),
                "score_top1": round(o.score_top1, 4),
                "label": o.label or _short_label(o.query),
            }
            for o in outcome.options
        ],
    }
