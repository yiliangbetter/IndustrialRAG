"""
Utility functions for RAGAnything

Contains helper functions for content separation, text insertion, and other utilities
"""

import base64
import math
from typing import Dict, List, Any, Tuple
from pathlib import Path
from lightrag.utils import logger


def separate_content(
    content_list: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Separate text content and multimodal content

    Args:
        content_list: Content list from MinerU parsing

    Returns:
        (text_content, multimodal_items): Pure text content and multimodal items list
    """
    text_parts = []
    multimodal_items = []

    for item in content_list:
        content_type = item.get("type", "text")

        if content_type == "text":
            # Text content
            text = item.get("text", "")
            if text.strip():
                text_parts.append(text)
        else:
            # Multimodal content (image, table, equation, etc.)
            multimodal_items.append(item)

    # Merge all text content
    text_content = "\n\n".join(text_parts)

    logger.info("Content separation complete:")
    logger.info(f"  - Text content length: {len(text_content)} characters")
    logger.info(f"  - Multimodal items count: {len(multimodal_items)}")

    # Count multimodal types
    modal_types = {}
    for item in multimodal_items:
        modal_type = item.get("type", "unknown")
        modal_types[modal_type] = modal_types.get(modal_type, 0) + 1

    if modal_types:
        logger.info(f"  - Multimodal type distribution: {modal_types}")

    return text_content, multimodal_items


def _join_caption_field(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(x) for x in value if x).strip()
    return str(value or "").strip()


def build_image_ref_block(
    *,
    img_path: str,
    page_idx: int | None = None,
    caption: str = "",
    footnote: str = "",
    context: str = "",
) -> str:
    """Format a lightweight image reference block for vector / KG indexing."""
    lines = ["[图片]", f"图片路径：{img_path}"]
    if page_idx is not None:
        lines.append(f"页码：{page_idx}")
    if caption:
        lines.append(f"图注：{caption}")
    if footnote:
        lines.append(f"脚注：{footnote}")
    if context:
        lines.append(f"关联正文：{context[:800]}")
    return "\n".join(lines)


def _bbox_center(bbox: Any) -> tuple[float, float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return None
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def neighbor_context_text(items: List[Dict[str, Any]], index: int, window: int = 3) -> str:
    """Collect nearby text blocks around an image for semantic association."""
    parts: List[str] = []
    lo = max(0, index - window)
    hi = min(len(items), index + window + 1)
    for j in range(lo, hi):
        if j == index:
            continue
        item = items[j]
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
                continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return " ".join(parts).strip()


def _text_image_layout_distance(
    img_center: tuple[float, float], text_center: tuple[float, float], text_bbox: Any
) -> float:
    """Prefer left-column text when the image sits in the right column (manual layout)."""
    if img_center[0] > 500 and text_center[0] < 520:
        return abs(text_center[1] - img_center[1])
    return math.hypot(text_center[0] - img_center[0], text_center[1] - img_center[1])


def context_text_for_image(
    items: List[Dict[str, Any]], index: int, window: int = 3, max_chars: int = 800
) -> str:
    """Associate image with text via same-page bbox proximity, then local window."""
    if index < 0 or index >= len(items):
        return ""
    item = items[index]
    if not isinstance(item, dict):
        return neighbor_context_text(items, index, window)

    page_idx = item.get("page_idx")
    img_center = _bbox_center(item.get("bbox"))
    ordered: List[str] = []
    seen: set[str] = set()

    def add_text(text: str) -> None:
        cleaned = text.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            ordered.append(cleaned)

    if page_idx is not None and img_center is not None:
        ranked: List[tuple[float, str]] = []
        for j, other in enumerate(items):
            if j == index or not isinstance(other, dict):
                continue
            if other.get("page_idx") != page_idx or other.get("type") != "text":
                continue
            text = other.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            text_bbox = other.get("bbox")
            text_center = _bbox_center(text_bbox)
            if text_center is None:
                continue
            dist = _text_image_layout_distance(img_center, text_center, text_bbox)
            ranked.append((dist, text.strip()))
        ranked.sort(key=lambda pair: pair[0])
        for _, text in ranked[:3]:
            add_text(text)

    if not ordered:
        neighbor = neighbor_context_text(items, index, window)
        if neighbor:
            add_text(neighbor)

    merged = " ".join(ordered).strip()
    return merged[:max_chars] if merged else ""


def flatten_image_refs_for_skip_multimodal(items: List[Dict[str, Any]]) -> str:
    """Build indexable text for images when multimodal LLM processing is skipped."""
    blocks: List[str] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict) or item.get("type") != "image":
            continue
        img_path = (item.get("img_path") or "").strip()
        if not img_path:
            continue
        caption = _join_caption_field(
            item.get("image_caption", item.get("img_caption", ""))
        )
        footnote = _join_caption_field(
            item.get("image_footnote", item.get("img_footnote", ""))
        )
        page_idx = item.get("page_idx")
        context = context_text_for_image(items, idx)
        blocks.append(
            build_image_ref_block(
                img_path=img_path,
                page_idx=page_idx if isinstance(page_idx, int) else None,
                caption=caption,
                footnote=footnote,
                context=context,
            )
        )
    if not blocks:
        return ""
    return "\n\n".join(blocks)


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


async def insert_text_content(
    lightrag,
    input: str | list[str],
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    ids: str | list[str] | None = None,
    file_paths: str | list[str] | None = None,
):
    """
    Insert pure text content into LightRAG

    Args:
        lightrag: LightRAG instance
        input: Single document string or list of document strings
        split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
        chunk_token_size, it will be split again by token size.
        split_by_character_only: if split_by_character_only is True, split the string by character only, when
        split_by_character is None, this parameter is ignored.
        ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
        file_paths: single string of the file path or list of file paths, used for citation
    """
    logger.info("Starting text content insertion into LightRAG...")

    # Use LightRAG's insert method with all parameters
    await lightrag.ainsert(
        input=input,
        file_paths=file_paths,
        split_by_character=split_by_character,
        split_by_character_only=split_by_character_only,
        ids=ids,
    )

    logger.info("Text content insertion complete")


async def insert_text_content_with_multimodal_content(
    lightrag,
    input: str | list[str],
    multimodal_content: list[dict[str, any]] | None = None,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    ids: str | list[str] | None = None,
    file_paths: str | list[str] | None = None,
    scheme_name: str | None = None,
):
    """
    Insert pure text content into LightRAG

    Args:
        lightrag: LightRAG instance
        input: Single document string or list of document strings
        multimodal_content: Multimodal content list (optional)
        split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
        chunk_token_size, it will be split again by token size.
        split_by_character_only: if split_by_character_only is True, split the string by character only, when
        split_by_character is None, this parameter is ignored.
        ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
        file_paths: single string of the file path or list of file paths, used for citation
        scheme_name: scheme name (optional)
    """
    logger.info("Starting text content insertion into LightRAG...")

    # Use LightRAG's insert method with all parameters
    try:
        await lightrag.ainsert(
            input=input,
            multimodal_content=multimodal_content,
            file_paths=file_paths,
            split_by_character=split_by_character,
            split_by_character_only=split_by_character_only,
            ids=ids,
            scheme_name=scheme_name,
        )
    except Exception as e:
        logger.info(f"Error: {e}")
        logger.info(
            "If the error is caused by the ainsert function not having a multimodal content parameter, please update the raganything branch of lightrag"
        )

    logger.info("Text content insertion complete")


def get_processor_for_type(modal_processors: Dict[str, Any], content_type: str):
    """
    Get appropriate processor based on content type

    Args:
        modal_processors: Dictionary of available processors
        content_type: Content type

    Returns:
        Corresponding processor instance
    """
    # Direct mapping to corresponding processor
    if content_type == "image":
        return modal_processors.get("image")
    elif content_type == "table":
        return modal_processors.get("table")
    elif content_type == "equation":
        return modal_processors.get("equation")
    else:
        # For other types, use generic processor
        return modal_processors.get("generic")


def get_processor_supports(proc_type: str) -> List[str]:
    """Get processor supported features"""
    supports_map = {
        "image": [
            "Image content analysis",
            "Visual understanding",
            "Image description generation",
            "Image entity extraction",
        ],
        "table": [
            "Table structure analysis",
            "Data statistics",
            "Trend identification",
            "Table entity extraction",
        ],
        "equation": [
            "Mathematical formula parsing",
            "Variable identification",
            "Formula meaning explanation",
            "Formula entity extraction",
        ],
        "generic": [
            "General content analysis",
            "Structured processing",
            "Entity extraction",
        ],
    }
    return supports_map.get(proc_type, ["Basic processing"])
