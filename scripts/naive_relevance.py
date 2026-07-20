"""Backward-compatible re-export — use ``raganything.naive_relevance`` in new code."""

from raganything.naive_relevance import (
    cosine_threshold_from_env,
    extract_query_keywords,
    format_relevance_report,
    is_naive_relevance_enabled,
    primary_relevance_score,
    probe_chunk_vector_score,
    probe_top_k_from_env,
    score_naive_relevance,
)

__all__ = [
    "cosine_threshold_from_env",
    "extract_query_keywords",
    "format_relevance_report",
    "is_naive_relevance_enabled",
    "primary_relevance_score",
    "probe_chunk_vector_score",
    "probe_top_k_from_env",
    "score_naive_relevance",
]
