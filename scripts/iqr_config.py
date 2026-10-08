"""image_query_refs submodule ``iqr_config``.

Environment-driven thresholds and limits for query-time image selection.
"""

from __future__ import annotations

import os
from typing import Any


_MIN_SUBSTANTIVE_TERM_LEN = 3


_MIN_QUERY_CHARS_FOR_IMAGES = 6


def _min_substantive_term_len() -> int:
    raw = os.getenv("RAG_IMAGE_MIN_TERM_LEN") or str(_MIN_SUBSTANTIVE_TERM_LEN)
    try:
        return max(2, int(raw))
    except ValueError:
        return _MIN_SUBSTANTIVE_TERM_LEN


def _min_query_chars_for_images() -> int:
    raw = os.getenv("RAG_IMAGE_MIN_QUERY_CHARS") or str(_MIN_QUERY_CHARS_FOR_IMAGES)
    try:
        return max(2, int(raw))
    except ValueError:
        return _MIN_QUERY_CHARS_FOR_IMAGES


def default_image_selection_limit() -> int:
    """Global cap for selected figures; ``0`` env means no practical cap (24)."""
    raw = os.getenv("RAG_IMAGE_MULTI_LIMIT") or "0"
    try:
        val = int(raw)
    except ValueError:
        return 24
    if val <= 0:
        return 24
    return max(1, min(24, val))


def _multi_figure_image_limit() -> int:
    return default_image_selection_limit()


def _inline_min_place_score() -> float:
    raw = os.getenv("RAG_IMAGE_INLINE_MIN_PLACE_SCORE") or "0.35"
    try:
        return float(raw)
    except ValueError:
        return 0.35


def _env_int_image(*keys: str, default: int, min_v: int = 1, max_v: int = 32) -> int:
    """Read first set env among ``keys`` as bounded int (phase 5 alias unify)."""
    raw = ""
    for key in keys:
        val = os.getenv(key)
        if val is not None and str(val).strip():
            raw = str(val).strip()
            break
    if not raw:
        return default
    try:
        return max(min_v, min(max_v, int(raw)))
    except ValueError:
        return default


def _env_bool_image(key: str, default: bool = True) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _max_rerank_score(retrieved_docs: list[dict[str, Any]] | None) -> float | None:
    scores: list[float] = []
    for doc in retrieved_docs or []:
        raw = doc.get("rerank_score")
        if raw is None:
            continue
        try:
            scores.append(float(raw))
        except (TypeError, ValueError):
            continue
    return max(scores) if scores else None


def _image_min_rerank_score() -> float:
    raw = (
        os.getenv("RAG_IMAGE_MIN_RERANK_SCORE")
        or os.getenv("MIN_RERANK_SCORE")
        or "0.28"
    )
    try:
        return float(raw)
    except ValueError:
        return 0.28


def _image_min_ref_align() -> float:
    raw = os.getenv("RAG_IMAGE_MIN_REF_ALIGN") or "0.28"
    try:
        return float(raw)
    except ValueError:
        return 0.28
