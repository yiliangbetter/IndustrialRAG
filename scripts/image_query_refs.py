"""Query-time image resolution for RAG Q&A (Plan B).

Ingest co-locates ``[图片]`` blocks with anchor text in vector chunks (Route A).
This module parses inline refs from LLM-input chunk bodies at finalize (Plan A).

Thin public facade over the ``iqr_*`` implementation modules.  Only the public
API is re-exported here; internal ``_``-prefixed helpers live in their source
modules (``iqr_align`` / ``iqr_anchor`` / ``iqr_config`` / ``iqr_explain`` /
``iqr_figure_target`` / ``iqr_media`` / ``iqr_placement`` / ``iqr_protocol`` /
``iqr_store`` / ``iqr_terms``) and must be imported from there directly.
"""

from __future__ import annotations

from raganything.utils import (
    build_image_ref_block as build_image_ref_block,
    flatten_image_refs_for_skip_multimodal as flatten_image_refs_for_skip_multimodal,
    substantive_bigrams as substantive_bigrams,
)

from iqr_protocol import (
    extract_image_refs_from_context as extract_image_refs_from_context,
    normalize_context_for_image_parse as normalize_context_for_image_parse,
)
from iqr_media import (
    decode_media_token as decode_media_token,
    encode_media_token as encode_media_token,
    is_safe_media_path as is_safe_media_path,
    resolve_media_path as resolve_media_path,
)
from iqr_config import (
    default_image_selection_limit as default_image_selection_limit,
)
from iqr_store import (
    merge_context_for_images as merge_context_for_images,
    text_from_retrieved_docs as text_from_retrieved_docs,
)
from iqr_anchor import (
    llm_chunk_locality_enabled as llm_chunk_locality_enabled,
    merge_order_neighbors_into_llm_chunks as merge_order_neighbors_into_llm_chunks,
    query_section_anchor_enabled as query_section_anchor_enabled,
    supplement_unique_chunks_with_order_neighbors as supplement_unique_chunks_with_order_neighbors,
)
from iqr_figure_target import (
    detect_table_filter_signal as detect_table_filter_signal,
    extract_figure_targets as extract_figure_targets,
    filter_docs_cited_by_answer as filter_docs_cited_by_answer,
)
from iqr_explain import (
    explain_query_images as explain_query_images,
    explain_retrieval_supports_images as explain_retrieval_supports_images,
    query_wants_kb_images as query_wants_kb_images,
    retrieval_supports_images as retrieval_supports_images,
)
from iqr_placement import (
    build_inline_placements as build_inline_placements,
    images_for_api as images_for_api,
    resolve_query_images as resolve_query_images,
)

__all__ = [
    "build_image_ref_block",
    "flatten_image_refs_for_skip_multimodal",
    "extract_image_refs_from_context",
    "build_inline_placements",
    "images_for_api",
    "resolve_query_images",
    "default_image_selection_limit",
    "query_wants_kb_images",
    "retrieval_supports_images",
    "encode_media_token",
    "decode_media_token",
    "is_safe_media_path",
    "normalize_context_for_image_parse",
]
