"""
Utility functions for RAGAnything

Facade module: generic helpers live here; term alignment, image-text
pairing, ingest coalescing and insertion were split into text_align,
image_context, ingest_coalesce and ingest_insert (stage-1 refactor, see
docs/utils_refactor_schema_induction.md). All previous public and
private names remain importable from raganything.utils.
"""

import base64
from pathlib import Path

from lightrag.utils import logger

from .text_align import (  # noqa: F401
    _COALESCE_HEADING_MAX_CHARS,
    _SECTION_HEADING_LINE_RE,
    _SHORT_LABEL_ANCHOR_RUN_LEN,
    _SHORT_LABEL_MAX_LEN,
    _SHORT_LABEL_MIN_SHARED_BIGRAMS,
    _best_focus_subspan,
    _best_overlap_cjk_run,
    _focus_midsection_bigram_hits,
    _focus_run_for_bag,
    _is_section_heading_line,
    _join_caption_field,
    _longest_cjk_run,
    discriminative_terms,
    image_label_text,
    label_bigram_coverage,
    short_label_bag_aligns,
    substantive_bigrams,
    text_term_alignment,
    text_term_alignment_symmetric,
)
from .image_context import (  # noqa: F401
    _ABOVE_IMAGE_DISTANCE_PENALTY,
    _BBOX_MATCH_MAX_DIST,
    _BULLET_PREFIX_RE,
    _CAPTION_INFER_MAX_GAP,
    _CAPTION_INFER_MAX_LEN,
    _CAPTION_INFER_WINDOW,
    _LABEL_ALIGN_MIN,
    _READING_ORDER_IMAGE_WINDOW,
    _TEXT_AFTER_IMAGE_PENALTY,
    _TEXT_BEFORE_IMAGE_BONUS,
    _anchor_pairing_priority,
    _bbox_bottom,
    _bbox_center,
    _bbox_top,
    _layout_distance_for_text_image_pair,
    _looks_like_section_heading,
    _text_image_layout_distance,
    anchor_context_for_image,
    best_image_for_text_item,
    context_text_for_image,
    image_label_for_item,
    infer_figure_label_from_layout,
    neighbor_context_text,
    plan_text_image_assignments,
    resolve_image_caption,
    resolve_image_footnote,
    separate_content,
)
from .ingest_coalesce import (  # noqa: F401
    _COALESCE_HEADING_LOOKAHEAD,
    _COALESCE_IMAGE_LOOKAHEAD,
    _COALESCE_METADATA_LOOKAHEAD,
    _FIELD_KEY_RE,
    _IMAGE_REF_MARKER,
    _INGEST_IMAGE_PATH_RE,
    _INGEST_SEGMENT_DELIMITER,
    _SCHEMA_MIN_KEYS,
    _SCHEMA_MIN_REPEAT,
    _TABLE_INGEST_MARKER,
    _TABLE_ROW_RE,
    _assemble_record_section_segments,
    _best_heading_body_score_before,
    _coalesce_immediate_text_image_segments,
    _coalesce_sub_parts_by_section,
    _collect_record_section_parts,
    _context_from_ref_segment,
    _dedupe_key_for_image_segment,
    _explode_overmerged_segments,
    _follows_record_subsection,
    _has_backward_procedure_for_heading,
    _heading_body_merge_score,
    _image_label_from_ref_segment,
    _image_match_text_from_ref_segment,
    _ingest_chunk_token_size,
    _is_image_ref_segment,
    _is_orphan_heading_segment,
    _is_orphan_record_field_segment,
    _is_orphan_record_metadata_only_segment,
    _is_thin_section_lead_segment,
    _line_field_key,
    _lookahead_pair_text_image_segments,
    _merge_orphan_heading_segments,
    _merge_orphan_record_metadata_segments,
    _merge_thin_section_with_following_figure,
    _merge_trailing_orphan_headings_backward,
    _merge_trailing_procedure_into_preceding,
    _merge_unheaded_fields_with_heading_sections,
    _part_starts_section_heading,
    _preceding_section_accepts_trailing_fields,
    _record_section_step_closed,
    _record_step_alignment,
    _section_heading_needs_field_merge,
    _segment_field_keys,
    _segment_first_line,
    _segment_has_field_line,
    _segment_has_inline_image,
    _segment_has_initiator,
    _segment_has_closer,
    _segment_has_record_body,
    _segment_has_record_cycle,
    _segment_starts_new_section,
    _segment_token_count,
    _should_collect_record_subsection,
    _split_block_at_trailing_orphan_heading,
    _split_image_blocks_in_segment,
    _split_multi_image_segments,
    _split_overmerged_record_segment,
    _split_sub_parts_at_section_heading_boundaries,
    _split_table_html_segment,
    _split_text_by_token_size,
    DocFieldSchema,
    build_image_ref_block,
    build_table_aware_ingest_segments,
    coalesce_parts_for_embedding_ingest,
    coalesce_text_image_segments,
    compute_table_aware_ingest_segments,
    flatten_image_refs_for_skip_multimodal,
    induce_field_schema,
    prepare_table_aware_ingest_segments,
    table_aware_ingest_enabled,
)
from .ingest_insert import (  # noqa: F401
    _resolve_ingest_segments,
    _status_field,
    compute_ingest_chunk_id,
    get_processor_for_type,
    get_processor_supports,
    insert_doc_scoped_text_content,
    insert_text_content,
    insert_text_content_with_multimodal_content,
)


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image file to base64 string

    Args:
        image_path: Path to the image file

    Returns:
        str: Base64 encoded string, empty string if encoding fails
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return ""


def validate_image_file(image_path: str, max_size_mb: int = 50) -> bool:
    """
    Validate if a file is a valid image file

    Args:
        image_path: Path to the image file
        max_size_mb: Maximum file size in MB

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        path = Path(image_path)

        logger.debug(f"Validating image path: {image_path}")
        logger.debug(f"Resolved path object: {path}")
        logger.debug(f"Path exists check: {path.exists()}")

        # Check if file exists and is not a symlink (for security)
        if not path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return False

        if path.is_symlink():
            logger.warning(f"Blocking symlink for security: {image_path}")
            return False

        # Check file extension
        image_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp",
            ".tiff",
            ".tif",
        ]

        path_lower = str(path).lower()
        has_valid_extension = any(path_lower.endswith(ext) for ext in image_extensions)
        logger.debug(
            f"File extension check - path: {path_lower}, valid: {has_valid_extension}"
        )

        if not has_valid_extension:
            logger.warning(f"File does not appear to be an image: {image_path}")
            return False

        # Check file size
        file_size = path.stat().st_size
        max_size = max_size_mb * 1024 * 1024
        logger.debug(
            f"File size check - size: {file_size} bytes, max: {max_size} bytes"
        )

        if file_size > max_size:
            logger.warning(f"Image file too large ({file_size} bytes): {image_path}")
            return False

        logger.debug(f"Image validation successful: {image_path}")
        return True

    except Exception as e:
        logger.error(f"Error validating image file {image_path}: {e}")
        return False
