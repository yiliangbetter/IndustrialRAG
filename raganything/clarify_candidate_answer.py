"""LLM pre-answer validation for clarify gate candidates."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from lightrag import QueryParam

from raganything.clarify_context import CachedQueryBundle

_REFUSAL_MARKERS = (
    "未提及",
    "无法确定",
    "无法回答",
    "没有足够",
    "暂无相关",
    "未能找到",
    "无法基于",
    "不能确定",
    "无相关",
    "未找到",
    "无法提供",
    "不足以",
    "无法从",
    "没有明确",
    "无法判断",
    "无法确认",
)


def clarify_candidate_llm_validate_enabled() -> bool:
    raw = (os.getenv("CLARIFY_CANDIDATE_LLM_VALIDATE") or "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _answer_body_without_references(answer: str) -> str:
    text = (answer or "").strip()
    match = re.search(r"(?im)^#{1,3}\s*References\s*$", text)
    if match:
        return text[: match.start()].strip()
    return text


def is_substantive_clarify_answer(answer: str) -> bool:
    """Return False when the model hedges or only reports missing evidence."""
    body = _answer_body_without_references(answer)
    if len(body) < 30:
        return False

    lines = [
        line.strip()
        for line in body.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not lines:
        return False

    refusal_lines = sum(
        1 for line in lines if any(marker in line for marker in _REFUSAL_MARKERS)
    )
    if len(lines) >= 2 and refusal_lines >= max(2, int(len(lines) * 0.6)):
        return False
    if body.count("未提及") >= 2 and refusal_lines >= 2:
        if not re.search(r"\[\d+\]", body):
            return False
    if refusal_lines == len(lines) and len(lines) >= 1:
        return False
    return True


def _env_bool_rerank_default() -> bool:
    raw = (os.getenv("RERANK_BY_DEFAULT") or "true").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _query_extras(query: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        scripts = root / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        from query_doc_steering import build_user_prompt_for_query  # noqa: WPS433

        steer = build_user_prompt_for_query(query)
        if steer:
            out["user_prompt"] = steer
    except Exception:
        pass
    return out


async def _collect_aquery_text(result: Any) -> str:
    if asyncio.iscoroutine(result):
        result = await result
    if isinstance(result, str):
        return result
    if hasattr(result, "__aiter__"):
        parts: list[str] = []
        async for chunk in result:
            if chunk:
                parts.append(chunk if isinstance(chunk, str) else str(chunk))
        return "".join(parts)
    return str(result or "")


async def probe_candidate_llm_answer(
    lightrag: Any,
    query: str,
    bundle: CachedQueryBundle,
    *,
    mode: str = "mix",
) -> tuple[str, str]:
    """Run mix ``aquery`` with injected bundle; return ``(thinking, answer)``."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from query_progress_hooks import (  # noqa: WPS433
        clear_clarify_context_injection,
        query_progress_hooks,
        set_clarify_context_injection,
    )
    from stream_cot_parser import parse_complete_cot  # noqa: WPS433

    q = (query or "").strip()
    set_clarify_context_injection(bundle)
    raw = ""
    try:
        async with query_progress_hooks():
            result = await lightrag.aquery(
                q,
                param=QueryParam(
                    mode=(mode or "mix").strip() or "mix",
                    enable_rerank=_env_bool_rerank_default(),
                    **_query_extras(q),
                ),
            )
            raw = await _collect_aquery_text(result)
    finally:
        clear_clarify_context_injection()

    thinking, answer = parse_complete_cot(raw or "")
    try:
        from query_doc_steering import strip_manual_circled_step_markers  # noqa: WPS433

        answer = strip_manual_circled_step_markers(answer)
    except Exception:
        pass
    return thinking or "", (answer or "").strip()


async def validate_candidate_with_llm(
    lightrag: Any,
    query: str,
    bundle: CachedQueryBundle,
    *,
    mode: str = "mix",
) -> tuple[bool, str, str]:
    """Return ``(accepted, thinking, answer)`` after a full injected aquery."""
    thinking, answer = await probe_candidate_llm_answer(
        lightrag, query, bundle, mode=mode
    )
    if not answer:
        return False, thinking, ""
    return is_substantive_clarify_answer(answer), thinking, answer
