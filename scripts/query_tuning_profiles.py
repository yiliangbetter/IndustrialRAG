"""Named env profiles for comparing rerank threshold vs image-anchor tuning."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

# Keys touched by tuning experiments (restored after each profile run).
_TUNING_KEYS = (
    "MIN_RERANK_SCORE",
    "RAG_IMAGE_MIN_RERANK_SCORE",
    "RAG_IMAGE_ANCHOR_MODE",
    "RAG_USE_CLARIFY_UPPER_AS_MIN_RERANK",
    "QUERY_SCORE_THRESHOLD_UPPER",
    "QUERY_SCORE_THRESHOLD_LOWER",
    "TOP_K",
    "CHUNK_TOP_K",
    "MAX_TOTAL_TOKENS",
)

PROFILES: dict[str, dict[str, str]] = {
    "baseline": {
        "MIN_RERANK_SCORE": "0.28",
        "RAG_IMAGE_MIN_RERANK_SCORE": "0.28",
        "RAG_IMAGE_ANCHOR_MODE": "off",
        "RAG_USE_CLARIFY_UPPER_AS_MIN_RERANK": "0",
    },
    "high_rerank": {
        "MIN_RERANK_SCORE": "0.45",
        "RAG_IMAGE_MIN_RERANK_SCORE": "0.45",
        "RAG_IMAGE_ANCHOR_MODE": "off",
        "RAG_USE_CLARIFY_UPPER_AS_MIN_RERANK": "0",
        "QUERY_SCORE_THRESHOLD_UPPER": "0.45",
    },
    "image_anchor": {
        "MIN_RERANK_SCORE": "0.28",
        "RAG_IMAGE_MIN_RERANK_SCORE": "0.28",
        "RAG_IMAGE_ANCHOR_MODE": "answer",
        "RAG_USE_CLARIFY_UPPER_AS_MIN_RERANK": "0",
    },
    "both": {
        "MIN_RERANK_SCORE": "0.45",
        "RAG_IMAGE_MIN_RERANK_SCORE": "0.45",
        "RAG_IMAGE_ANCHOR_MODE": "answer",
        "RAG_USE_CLARIFY_UPPER_AS_MIN_RERANK": "0",
        "QUERY_SCORE_THRESHOLD_UPPER": "0.45",
    },
    "sync_upper": {
        "RAG_USE_CLARIFY_UPPER_AS_MIN_RERANK": "1",
        "RAG_IMAGE_MIN_RERANK_SCORE": "0.45",
        "RAG_IMAGE_ANCHOR_MODE": "off",
        "QUERY_SCORE_THRESHOLD_UPPER": "0.45",
    },
    # Smaller retrieval budget only (no image anchor). vs baseline 20/30/52000.
    "slim": {
        "MIN_RERANK_SCORE": "0.28",
        "RAG_IMAGE_MIN_RERANK_SCORE": "0.28",
        "RAG_IMAGE_ANCHOR_MODE": "off",
        "RAG_USE_CLARIFY_UPPER_AS_MIN_RERANK": "0",
        "TOP_K": "12",
        "CHUNK_TOP_K": "10",
        "MAX_TOTAL_TOKENS": "18000",
    },
    "slim_tight": {
        "MIN_RERANK_SCORE": "0.28",
        "RAG_IMAGE_MIN_RERANK_SCORE": "0.28",
        "RAG_IMAGE_ANCHOR_MODE": "off",
        "RAG_USE_CLARIFY_UPPER_AS_MIN_RERANK": "0",
        "TOP_K": "8",
        "CHUNK_TOP_K": "6",
        "MAX_TOTAL_TOKENS": "12000",
    },
}


def profile_names() -> list[str]:
    return list(PROFILES.keys())


def snapshot_env(keys: tuple[str, ...] = _TUNING_KEYS) -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in keys}


def restore_env(saved: dict[str, str | None]) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def apply_profile(name: str) -> dict[str, str | None]:
    if name not in PROFILES:
        raise KeyError(f"Unknown profile {name!r}; choose from {profile_names()}")
    saved = snapshot_env()
    for key, value in PROFILES[name].items():
        os.environ[key] = value
    return saved


def sync_lightrag_rerank_threshold(lightrag: object) -> float:
    """Push current env/profile rerank threshold into the live LightRAG instance."""
    from query_doc_steering import _default_min_rerank_score  # noqa: WPS433

    score = _default_min_rerank_score()
    if hasattr(lightrag, "min_rerank_score"):
        lightrag.min_rerank_score = score  # type: ignore[attr-defined]
    return score


@contextmanager
def use_profile(name: str) -> Iterator[dict[str, str]]:
    saved = apply_profile(name)
    try:
        yield dict(PROFILES[name])
    finally:
        restore_env(saved)
