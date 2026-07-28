"""Image-text pairing on MinerU content lists (bbox layout + labels).

Split from utils.py (stage-1 refactor, see
docs/utils_refactor_schema_induction.md).
"""

import math
import re
from typing import Any, Dict, List, Tuple

from lightrag.utils import logger

from .text_align import (
    _is_section_heading_line,
    _join_caption_field,
    image_label_text,
    text_term_alignment_symmetric,
)


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


def image_label_for_item(items: List[Dict[str, Any]], item: Dict[str, Any]) -> str:
    """Caption/footnote for an image block, including layout-inferred labels."""
    if not isinstance(item, dict) or item.get("type") != "image":
        return ""
    # Identity match: equal image dicts must not steal another item's index.
    idx = next((i for i, it in enumerate(items) if it is item), None)
    if idx is None:
        return image_label_text(item)
    caption = resolve_image_caption(items, idx)
    footnote = resolve_image_footnote(items, idx)
    return " ".join(part for part in (caption, footnote) if part).strip()


_CAPTION_INFER_MAX_LEN = 24


_CAPTION_INFER_MAX_GAP = 120.0


_CAPTION_INFER_WINDOW = 8


def _looks_like_section_heading(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if re.match(r"^\d+\.\d+(?:\.\d+)?\s+\S", stripped):
        return True
    if stripped.startswith("第") and "节" in stripped[:8]:
        return True
    if stripped.startswith("保养") and "：" in stripped:
        return True
    return False


def infer_figure_label_from_layout(
    items: List[Dict[str, Any]], image_index: int
) -> str:
    """Recover a short caption below an image via MinerU bbox when footnote is empty."""
    if image_index < 0 or image_index >= len(items):
        return ""
    image_item = items[image_index]
    if not isinstance(image_item, dict) or image_item.get("type") != "image":
        return ""

    page_idx = image_item.get("page_idx")
    img_bottom = _bbox_bottom(image_item.get("bbox"))
    img_left = None
    img_right = None
    bbox = image_item.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            img_left = float(bbox[0])
            img_right = float(bbox[2])
        except (TypeError, ValueError):
            pass

    best: tuple[float, str] | None = None
    hi = min(len(items), image_index + _CAPTION_INFER_WINDOW + 1)
    for j in range(image_index + 1, hi):
        other = items[j]
        if not isinstance(other, dict) or other.get("type") != "text":
            continue
        if page_idx is not None and other.get("page_idx") != page_idx:
            break
        body = str(other.get("text") or "").strip()
        if not body or len(body) > _CAPTION_INFER_MAX_LEN:
            continue
        if _looks_like_section_heading(body):
            continue
        if re.search(r"[。；;]", body):
            continue

        text_top = _bbox_top(other.get("bbox"))
        if img_bottom is not None and text_top is not None:
            gap = text_top - img_bottom
            if gap < -20 or gap > _CAPTION_INFER_MAX_GAP:
                continue
            if img_left is not None and img_right is not None:
                text_bbox = other.get("bbox")
                if isinstance(text_bbox, (list, tuple)) and len(text_bbox) >= 4:
                    try:
                        tx0, tx1 = float(text_bbox[0]), float(text_bbox[2])
                        if tx1 < img_left - 80 or tx0 > img_right + 80:
                            continue
                    except (TypeError, ValueError):
                        pass
            score = abs(gap)
        else:
            score = float(j - image_index)

        if best is None or score < best[0]:
            best = (score, body)

    return best[1] if best else ""


def resolve_image_footnote(items: List[Dict[str, Any]], image_index: int) -> str:
    """Parser footnote, or a short caption inferred from layout below the figure."""
    if image_index < 0 or image_index >= len(items):
        return ""
    item = items[image_index]
    footnote = _join_caption_field(
        item.get("image_footnote", item.get("img_footnote", ""))
    )
    if footnote:
        return footnote
    return infer_figure_label_from_layout(items, image_index)


def resolve_image_caption(items: List[Dict[str, Any]], image_index: int) -> str:
    if image_index < 0 or image_index >= len(items):
        return ""
    item = items[image_index]
    return _join_caption_field(item.get("image_caption", item.get("img_caption", "")))


_BULLET_PREFIX_RE = re.compile(r"^[\s\u2022\u25cf\u25aa\u25e6\uF0D8\u2023\-–—*]+")


def _anchor_pairing_priority(anchor: str) -> float:
    """Prefer short procedure lines over section headings (structural, not domain)."""
    text = (anchor or "").strip()
    if not text:
        return 0.0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        return 0.0
    line = lines[0]
    if _is_section_heading_line(line):
        return -0.06
    bonus = 0.0
    if len(line) <= 72:
        bonus += 0.06
    if _BULLET_PREFIX_RE.match(line):
        bonus += 0.08
    return bonus


def plan_text_image_assignments(
    items: List[Dict[str, Any]],
    *,
    label_align_min: float | None = None,
) -> Dict[int, int]:
    """Exclusive text-index -> image-index pairing for ingest (parser fields + term overlap)."""
    align_min = label_align_min if label_align_min is not None else _LABEL_ALIGN_MIN
    image_index_by_id: Dict[int, int] = {
        id(item): idx
        for idx, item in enumerate(items)
        if isinstance(item, dict) and item.get("type") == "image"
    }
    candidates: List[tuple[float, int, int]] = []
    for ti, item in enumerate(items):
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        anchor = str(item.get("text") or "").strip()
        if not anchor:
            continue
        page_idx = item.get("page_idx")
        hi = min(len(items), ti + _READING_ORDER_IMAGE_WINDOW + 1)
        for ii in range(ti + 1, hi):
            img = items[ii]
            if not isinstance(img, dict) or img.get("type") != "image":
                continue
            if page_idx is not None and img.get("page_idx") != page_idx:
                break
            label = image_label_for_item(items, img)
            label_score = text_term_alignment_symmetric(anchor, label) if label else 0.0
            context = context_text_for_image(items, ii, max_chars=400)
            context_score = (
                text_term_alignment_symmetric(anchor, context) if context else 0.0
            )
            pair_score = max(label_score, context_score * 0.92)
            dist = max(1, ii - ti)
            priority = _anchor_pairing_priority(anchor)
            if pair_score >= align_min:
                candidates.append((pair_score + 0.06 / dist + priority, ti, ii))
            elif dist == 1:
                candidates.append((0.32 + 0.05 / dist + priority, ti, ii))
            elif dist <= 3 and context_score >= 0.45:
                candidates.append((context_score + 0.04 / dist + priority, ti, ii))
        best = best_image_for_text_item(items, ti)
        if best is not None:
            ii = image_index_by_id.get(id(best))
            if ii is None or ii <= ti:
                continue
            label = image_label_for_item(items, best)
            label_score = (
                text_term_alignment_symmetric(anchor, label) if label else 0.22
            )
            context = context_text_for_image(items, ii, max_chars=400)
            context_score = (
                text_term_alignment_symmetric(anchor, context) if context else 0.0
            )
            pair_score = max(label_score, context_score * 0.92)
            dist = max(1, ii - ti)
            candidates.append(
                (pair_score + 0.08 / dist + _anchor_pairing_priority(anchor), ti, ii)
            )

    candidates.sort(key=lambda row: (-row[0], row[2], row[1]))
    assignments: Dict[int, int] = {}
    used_images: set[int] = set()
    used_texts: set[int] = set()
    for score, ti, ii in candidates:
        if ti in used_texts or ii in used_images:
            continue
        if score < 0.18 and ii - ti > 3:
            continue
        assignments[ti] = ii
        used_texts.add(ti)
        used_images.add(ii)
    return assignments


def anchor_context_for_image(
    items: List[Dict[str, Any]],
    image_index: int,
    *,
    anchor_text_index: int | None = None,
    max_chars: int = 800,
) -> str:
    """Context for an image ref: prefer the assigned anchor text body."""
    if anchor_text_index is not None:
        if 0 <= anchor_text_index < len(items):
            anchor_item = items[anchor_text_index]
            if isinstance(anchor_item, dict) and anchor_item.get("type") == "text":
                text = str(anchor_item.get("text") or "").strip()
                if text:
                    return text[:max_chars]
    return context_text_for_image(items, image_index, max_chars=max_chars)


def _bbox_center(bbox: Any) -> tuple[float, float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x0, y0, x1, y1 = (
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            float(bbox[3]),
        )
    except (TypeError, ValueError):
        return None
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _bbox_top(bbox: Any) -> float | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 2:
        return None
    try:
        return float(bbox[1])
    except (TypeError, ValueError):
        return None


def _bbox_bottom(bbox: Any) -> float | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        return float(bbox[3])
    except (TypeError, ValueError):
        return None


_READING_ORDER_IMAGE_WINDOW = 15


_ABOVE_IMAGE_DISTANCE_PENALTY = 120.0


_TEXT_AFTER_IMAGE_PENALTY = 150.0


_TEXT_BEFORE_IMAGE_BONUS = 30.0


_BBOX_MATCH_MAX_DIST = 900.0


_LABEL_ALIGN_MIN = 0.35


def neighbor_context_text(
    items: List[Dict[str, Any]], index: int, window: int = 3
) -> str:
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


def _layout_distance_for_text_image_pair(
    items: List[Dict[str, Any]],
    text_index: int,
    image_index: int,
    *,
    max_dist: float = _BBOX_MATCH_MAX_DIST,
) -> float | None:
    """Score how well an image pairs with anchor text (lower is better)."""
    if (
        text_index < 0
        or image_index < 0
        or text_index >= len(items)
        or image_index >= len(items)
    ):
        return None
    text_item = items[text_index]
    image_item = items[image_index]
    if text_item.get("type") != "text" or image_item.get("type") != "image":
        return None
    page_idx = text_item.get("page_idx")
    if page_idx is None or image_item.get("page_idx") != page_idx:
        return None

    text_center = _bbox_center(text_item.get("bbox"))
    img_center = _bbox_center(image_item.get("bbox"))
    if text_center is None or img_center is None:
        return None

    dist = _text_image_layout_distance(img_center, text_center, text_item.get("bbox"))
    text_bottom = _bbox_bottom(text_item.get("bbox"))
    img_top = _bbox_top(image_item.get("bbox"))
    if text_bottom is not None and img_top is not None:
        if img_top >= text_bottom - 20:
            pass
        elif image_index < text_index:
            dist += _ABOVE_IMAGE_DISTANCE_PENALTY
        else:
            dist += _ABOVE_IMAGE_DISTANCE_PENALTY * 0.5

    if image_index > text_index:
        dist += (image_index - text_index) * 2.0
    elif image_index < text_index:
        dist += _ABOVE_IMAGE_DISTANCE_PENALTY + (text_index - image_index) * 3.0

    if dist > max_dist:
        return None
    return dist


def best_image_for_text_item(
    items: List[Dict[str, Any]],
    text_index: int,
    *,
    max_dist: float = _BBOX_MATCH_MAX_DIST,
    label_align_min: float = _LABEL_ALIGN_MIN,
) -> Dict[str, Any] | None:
    """Pair anchor text with a figure: labeled match > reading order > bbox."""
    if text_index < 0 or text_index >= len(items):
        return None
    text_item = items[text_index]
    if not isinstance(text_item, dict) or text_item.get("type") != "text":
        return None
    page_idx = text_item.get("page_idx")
    if page_idx is None:
        return None

    anchor_body = str(text_item.get("text") or "").strip()
    hi = min(len(items), text_index + _READING_ORDER_IMAGE_WINDOW + 1)

    best_labeled: tuple[float, int, Dict[str, Any]] | None = None
    first_unlabeled_after: Dict[str, Any] | None = None

    for j in range(text_index + 1, hi):
        item = items[j]
        if not isinstance(item, dict):
            continue
        if item.get("page_idx") != page_idx:
            break
        if item.get("type") != "image":
            continue

        label = image_label_text(item)
        if label and anchor_body:
            align = text_term_alignment_symmetric(anchor_body, label)
            if align >= label_align_min:
                if (
                    best_labeled is None
                    or align > best_labeled[0]
                    or (align == best_labeled[0] and j < best_labeled[1])
                ):
                    best_labeled = (align, j, item)
        elif first_unlabeled_after is None:
            first_unlabeled_after = item

    if best_labeled is not None:
        return best_labeled[2]
    if first_unlabeled_after is not None:
        return first_unlabeled_after

    best: Dict[str, Any] | None = None
    best_dist = float("inf")
    for j, item in enumerate(items):
        if item.get("type") != "image" or item.get("page_idx") != page_idx:
            continue
        score = _layout_distance_for_text_image_pair(
            items, text_index, j, max_dist=max_dist
        )
        if score is not None and score < best_dist:
            best_dist = score
            best = item
    return best


def context_text_for_image(
    items: List[Dict[str, Any]], index: int, window: int = 3, max_chars: int = 800
) -> str:
    """Associate image with text: label match > reading-order > bbox."""
    if index < 0 or index >= len(items):
        return ""
    item = items[index]
    if not isinstance(item, dict):
        return neighbor_context_text(items, index, window)

    page_idx = item.get("page_idx")
    label = image_label_text(item)
    ordered: List[str] = []
    seen: set[str] = set()

    def add_text(text: str) -> None:
        cleaned = text.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            ordered.append(cleaned)

    if label and page_idx is not None:
        ranked_labels: List[tuple[float, str]] = []
        lo = max(0, index - 8)
        for j in range(lo, index):
            other = items[j]
            if not isinstance(other, dict):
                continue
            if other.get("page_idx") != page_idx or other.get("type") != "text":
                continue
            text = other.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            align = text_term_alignment_symmetric(label, text.strip())
            if align <= 0:
                continue
            proximity = 1.0 + (index - j) * 0.12
            ranked_labels.append((-(align * proximity), text.strip()))
        ranked_labels.sort()
        for _, text in ranked_labels[:1]:
            add_text(text)
        if ordered:
            merged = " ".join(ordered).strip()
            return merged[:max_chars]

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
            if j < index:
                dist = max(0.0, dist - _TEXT_BEFORE_IMAGE_BONUS)
            elif j > index:
                dist += _TEXT_AFTER_IMAGE_PENALTY
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
