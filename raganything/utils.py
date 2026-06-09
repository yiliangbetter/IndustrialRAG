"""
Utility functions for RAGAnything

Contains helper functions for content separation, text insertion, and other utilities
"""

import base64
import math
import os
import re
from typing import Any, Dict, List, Tuple
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


def image_label_text(item: Dict[str, Any]) -> str:
    """MinerU image caption or footnote label (either may be empty)."""
    if not isinstance(item, dict) or item.get("type") != "image":
        return ""
    caption = _join_caption_field(
        item.get("image_caption", item.get("img_caption", ""))
    )
    footnote = _join_caption_field(
        item.get("image_footnote", item.get("img_footnote", ""))
    )
    return " ".join(part for part in (caption, footnote) if part).strip()


def image_label_for_item(items: List[Dict[str, Any]], item: Dict[str, Any]) -> str:
    """Caption/footnote for an image block, including layout-inferred labels."""
    if not isinstance(item, dict) or item.get("type") != "image":
        return ""
    try:
        idx = items.index(item)
    except ValueError:
        return image_label_text(item)
    caption = resolve_image_caption(items, idx)
    footnote = resolve_image_footnote(items, idx)
    return " ".join(part for part in (caption, footnote) if part).strip()


def discriminative_terms(text: str, *, min_len: int = 2) -> List[str]:
    """Length-based terms for any snippet (no domain phrase lists)."""
    terms: List[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        term = term.strip()
        if len(term) < min_len or term in seen:
            return
        seen.add(term)
        terms.append(term)

    for run in re.findall(r"[\u4e00-\u9fff]+", text or ""):
        if min_len <= len(run) <= 24:
            add(run)
        for size in (min_len, min_len + 1):
            if size > len(run):
                continue
            for i in range(len(run) - size + 1):
                add(run[i : i + size])

    for term in re.findall(r"[a-zA-Z0-9]{4,}", (text or "").lower()):
        add(term)
    return terms


def text_term_alignment(left: str, right: str, *, min_len: int = 2) -> float:
    """Share of discriminative terms from ``left`` found in ``right``."""
    terms = discriminative_terms(left, min_len=min_len)
    if not terms or not right.strip():
        return 0.0
    hits = sum(1 for term in terms if term in right)
    return hits / len(terms)


def text_term_alignment_symmetric(left: str, right: str, *, min_len: int = 2) -> float:
    if not left.strip() or not right.strip():
        return 0.0
    if left in right or right in left:
        return 1.0
    return max(
        text_term_alignment(left, right, min_len=min_len),
        text_term_alignment(right, left, min_len=min_len),
    )


def substantive_bigrams(text: str) -> set[str]:
    """Unique 2-character CJK runs (length-based, no domain phrase lists)."""
    bigrams: set[str] = set()
    for run in re.findall(r"[\u4e00-\u9fff]+", text or ""):
        for i in range(len(run) - 1):
            bigrams.add(run[i : i + 2])
    return bigrams


_SHORT_LABEL_MAX_LEN = 20
_SHORT_LABEL_MIN_SHARED_BIGRAMS = 2
_SHORT_LABEL_ANCHOR_RUN_LEN = 4


def _longest_cjk_run(text: str) -> str:
    runs = re.findall(r"[\u4e00-\u9fff]+", text or "")
    return max(runs, key=len, default="")


def _best_overlap_cjk_run(query: str, label: str, *, min_len: int = 4) -> str:
    """CJK span in the query whose bigrams best match the figure label."""
    label_bigrams = substantive_bigrams(label)
    best_run = ""
    best_score = 0
    for run in re.findall(r"[\u4e00-\u9fff]+", query or ""):
        if len(run) < min_len:
            continue
        score = len(substantive_bigrams(run) & label_bigrams)
        if score > best_score or (score == best_score and len(run) > len(best_run)):
            best_score = score
            best_run = run
    return best_run


def _focus_run_for_bag(query: str, label: str, *, min_len: int = 4) -> str:
    """Shortest query CJK span with strong bigram overlap to the label."""
    label_bigrams = substantive_bigrams(label)
    best_run = ""
    best_key: tuple[int, int] = (0, 0)
    for run in re.findall(r"[\u4e00-\u9fff]+", query or ""):
        if len(run) < min_len:
            continue
        overlap = len(substantive_bigrams(run) & label_bigrams)
        if overlap < _SHORT_LABEL_MIN_SHARED_BIGRAMS:
            continue
        key = (overlap, -len(run))
        if key > best_key:
            best_key = key
            best_run = run
    return best_run


def _best_focus_subspan(
    focus: str, label: str, *, min_len: int = _SHORT_LABEL_ANCHOR_RUN_LEN
) -> str:
    """Shortest sub-span with label overlap; prefer tight match at minimum length."""
    label_bigrams = substantive_bigrams(label)
    min_sub_len = max(min_len, 6)
    matches: list[tuple[int, str]] = []
    for start in range(len(focus)):
        for end in range(start + min_sub_len, len(focus) + 1):
            sub = focus[start:end]
            overlap = len(substantive_bigrams(sub) & label_bigrams)
            if overlap < _SHORT_LABEL_MIN_SHARED_BIGRAMS:
                continue
            matches.append((len(sub), sub))
    if not matches:
        return ""
    min_length = min(length for length, _ in matches)
    for length, sub in sorted(matches, key=lambda item: item[0]):
        if length != min_length:
            continue
        if _focus_midsection_bigram_hits(sub, label):
            return sub
    return ""


def _focus_midsection_bigram_hits(focus: str, label: str) -> set[str]:
    """Shared bigrams from the interior of the focus span (excludes edge-only matches)."""
    if len(focus) < 4:
        return set()
    mid = focus[1:-1]
    if len(mid) < 2:
        return set()
    return substantive_bigrams(mid) & substantive_bigrams(label)


def label_bigram_coverage(query: str, label: str) -> float:
    """Share of figure-label bigrams also present in the query."""
    label_bgs = substantive_bigrams(label)
    if not label_bgs:
        return 0.0
    return len(label_bgs & substantive_bigrams(query)) / len(label_bgs)


def short_label_bag_aligns(
    query: str,
    label: str,
    *,
    max_label_len: int = _SHORT_LABEL_MAX_LEN,
    min_shared: int = _SHORT_LABEL_MIN_SHARED_BIGRAMS,
) -> bool:
    """Align short figure labels when word order differs (e.g. 清洁机器床身 vs 机床床身清洁)."""
    label = (label or "").strip()
    query = (query or "").strip()
    if not label or not query or len(label) > max_label_len:
        return False
    if re.search(r"[。；;，,：:]", label):
        return False

    shared = substantive_bigrams(query) & substantive_bigrams(label)
    if len(shared) < min_shared:
        return False

    focus = _focus_run_for_bag(query, label)
    if len(focus) >= _SHORT_LABEL_ANCHOR_RUN_LEN:
        subspan = _best_focus_subspan(focus, label)
        if not subspan:
            return False
        focus = subspan
        if not _focus_midsection_bigram_hits(focus, label):
            return False
        label_bgs = substantive_bigrams(label)
        sub_hits = len(label_bgs & substantive_bigrams(focus))
        min_hits = max(min_shared, int(len(label_bgs) * 0.34))
        if len(label) >= 6:
            min_hits = max(min_hits, int(len(label_bgs) * 0.5))
        if sub_hits < min_hits:
            return False
    else:
        anchor = _best_overlap_cjk_run(query, label)
        if len(anchor) >= _SHORT_LABEL_ANCHOR_RUN_LEN:
            anchor_bigrams = {anchor[i : i + 2] for i in range(len(anchor) - 1)}
            if not any(bigram in label for bigram in anchor_bigrams):
                return False

    return True


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
    return _join_caption_field(
        item.get("image_caption", item.get("img_caption", ""))
    )


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


_IMAGE_REF_MARKER = "[图片]"
_INGEST_IMAGE_PATH_RE = re.compile(
    r"图片路径[：:]\s*(.+?)(?:\n|$)", re.MULTILINE
)


def _is_image_ref_segment(segment: str) -> bool:
    return (segment or "").lstrip().startswith(_IMAGE_REF_MARKER)


def _dedupe_key_for_image_segment(segment: str) -> str:
    match = _INGEST_IMAGE_PATH_RE.search(segment or "")
    if match:
        return Path(match.group(1).strip().strip('"').strip("'")).name
    return (segment or "").strip()


def coalesce_text_image_segments(segments: List[str]) -> List[str]:
    """Merge each text segment with immediately following ``[图片]`` blocks for indexing.

    Keeps figure metadata in the same vector chunk as the anchor body so rerank +
    steering filter text and inline figures together (Route A ingest).
    """
    out: List[str] = []
    i = 0
    n = len(segments)
    while i < n:
        seg = (segments[i] or "").strip()
        if not seg:
            i += 1
            continue
        if _is_image_ref_segment(seg):
            out.append(seg)
            i += 1
            continue
        parts = [seg]
        seen_images: set[str] = set()
        j = i + 1
        while j < n:
            nxt = (segments[j] or "").strip()
            if not nxt or not _is_image_ref_segment(nxt):
                break
            key = _dedupe_key_for_image_segment(nxt)
            if key not in seen_images:
                seen_images.add(key)
                parts.append(nxt)
            j += 1
        out.append("\n\n".join(parts))
        i = j
    return out


_TABLE_INGEST_MARKER = "[Table]"
_INGEST_SEGMENT_DELIMITER = "\n<<<RAG_SEG_BOUNDARY>>>\n"
_TABLE_ROW_RE = re.compile(r"<tr>.*?</tr>", re.IGNORECASE | re.DOTALL)


def table_aware_ingest_enabled() -> bool:
    raw = (os.getenv("RAG_TABLE_AWARE_INGEST") or "true").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _ingest_chunk_token_size(lightrag) -> int:
    for attr in ("chunk_token_size", "max_chunk_tokens"):
        val = getattr(lightrag, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    raw = os.getenv("CHUNK_SIZE") or "1200"
    try:
        return max(256, int(raw))
    except ValueError:
        return 1200


def _segment_token_count(tokenizer, text: str) -> int:
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return len(text.split())


def _split_text_by_token_size(
    text: str,
    tokenizer,
    max_tokens: int,
    *,
    overlap: int = 100,
) -> List[str]:
    if not text.strip():
        return []
    if _segment_token_count(tokenizer, text) <= max_tokens:
        return [text]
    tokens = tokenizer.encode(text)
    out: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        out.append(tokenizer.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start = max(0, end - overlap)
    return out


def _split_table_html_segment(
    segment: str,
    tokenizer,
    max_tokens: int,
) -> List[str]:
    piece = (segment or "").strip()
    if not piece:
        return []
    marker_idx = piece.find(_TABLE_INGEST_MARKER)
    if marker_idx < 0:
        return _split_text_by_token_size(piece, tokenizer, max_tokens)
    caption = piece[:marker_idx].strip()
    caption_prefix = f"{caption}\n\n" if caption else ""
    table_prefix = _TABLE_INGEST_MARKER + "\n"
    body = piece[marker_idx + len(_TABLE_INGEST_MARKER) :].lstrip("\n")
    prefix = caption_prefix + table_prefix

    rows = _TABLE_ROW_RE.findall(body)
    if not rows:
        return _split_text_by_token_size(prefix + body, tokenizer, max_tokens)

    header, data_rows = rows[0], rows[1:]
    if not data_rows:
        chunk = prefix + header
        if _segment_token_count(tokenizer, chunk) <= max_tokens:
            return [chunk]
        return _split_text_by_token_size(chunk, tokenizer, max_tokens)

    out: List[str] = []
    batch: List[str] = []
    for row in data_rows:
        candidate = prefix + header + "".join(batch + [row])
        if _segment_token_count(tokenizer, candidate) > max_tokens and batch:
            out.append(prefix + header + "".join(batch))
            batch = [row]
        elif _segment_token_count(tokenizer, candidate) > max_tokens:
            out.append(prefix + header + row)
            batch = []
        else:
            batch.append(row)
    if batch:
        out.append(prefix + header + "".join(batch))
    return out or [piece]


def build_table_aware_ingest_segments(document_parts: List[str]) -> List[str]:
    """Keep each ``[Table]`` block intact; coalesce adjacent non-table parts."""
    segments: List[str] = []
    buf: List[str] = []

    def flush_buf() -> None:
        nonlocal buf
        if not buf:
            return
        sub_parts = [p.strip() for p in "\n\n".join(buf).split("\n\n") if p.strip()]
        segments.extend(coalesce_text_image_segments(sub_parts))
        buf = []

    for part in document_parts:
        piece = (part or "").strip()
        if not piece:
            continue
        if _TABLE_INGEST_MARKER in piece:
            flush_buf()
            segments.append(piece)
            continue
        buf.append(piece)
    flush_buf()
    return segments


def prepare_table_aware_ingest_segments(
    document_parts: List[str],
    tokenizer,
    max_tokens: int,
) -> List[str]:
    merged = build_table_aware_ingest_segments(document_parts)
    prepared: List[str] = []
    for seg in merged:
        if _TABLE_INGEST_MARKER in seg:
            prepared.extend(_split_table_html_segment(seg, tokenizer, max_tokens))
            continue
        if _segment_token_count(tokenizer, seg) <= max_tokens:
            prepared.append(seg)
        else:
            prepared.extend(_split_text_by_token_size(seg, tokenizer, max_tokens))
    out = [s for s in prepared if s.strip()]
    return out


def compute_table_aware_ingest_segments(lightrag, document_parts: List[str]) -> List[str]:
    """Prepared ingest segments (table-aware split; no flatten when matrix ingest is on)."""
    if not table_aware_ingest_enabled() or not document_parts:
        return list(document_parts or [])
    tokenizer = getattr(lightrag, "tokenizer", None)
    if tokenizer is None:
        return list(document_parts)
    max_tokens = _ingest_chunk_token_size(lightrag)
    return prepare_table_aware_ingest_segments(document_parts, tokenizer, max_tokens)


def _bbox_center(bbox: Any) -> tuple[float, float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
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


def _layout_distance_for_text_image_pair(
    items: List[Dict[str, Any]],
    text_index: int,
    image_index: int,
    *,
    max_dist: float = _BBOX_MATCH_MAX_DIST,
) -> float | None:
    """Score how well an image pairs with anchor text (lower is better)."""
    if text_index < 0 or image_index < 0 or text_index >= len(items) or image_index >= len(items):
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
                if best_labeled is None or align > best_labeled[0] or (
                    align == best_labeled[0] and j < best_labeled[1]
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
        for j, other in enumerate(items):
            if j >= index or not isinstance(other, dict):
                continue
            if other.get("page_idx") != page_idx or other.get("type") != "text":
                continue
            text = other.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            align = text_term_alignment_symmetric(label, text.strip())
            if align <= 0:
                continue
            order_bonus = (index - j) * 0.01
            ranked_labels.append((-(align - order_bonus), text.strip()))
        ranked_labels.sort()
        for _, text in ranked_labels[:3]:
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


def flatten_image_refs_for_skip_multimodal(items: List[Dict[str, Any]]) -> str:
    """Build indexable text for images when multimodal LLM processing is skipped."""
    blocks: List[str] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict) or item.get("type") != "image":
            continue
        img_path = (item.get("img_path") or "").strip()
        if not img_path:
            continue
        caption = resolve_image_caption(items, idx)
        footnote = resolve_image_footnote(items, idx)
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
    *,
    document_parts: List[str] | None = None,
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
        document_parts: optional pre-split parts (``[Table]`` blocks stay atomic when table-aware ingest is on)
    """
    logger.info("Starting text content insertion into LightRAG...")

    if (
        table_aware_ingest_enabled()
        and document_parts
        and isinstance(input, str)
    ):
        tokenizer = getattr(lightrag, "tokenizer", None)
        if tokenizer is not None:
            max_tokens = _ingest_chunk_token_size(lightrag)
            segments = prepare_table_aware_ingest_segments(
                document_parts, tokenizer, max_tokens
            )
            if segments:
                blob = _INGEST_SEGMENT_DELIMITER.join(segments)
                logger.info(
                    "Table-aware ingest: %d segment(s), delimiter split (max_tokens=%d)",
                    len(segments),
                    max_tokens,
                )
                await lightrag.ainsert(
                    input=blob,
                    file_paths=file_paths,
                    split_by_character=_INGEST_SEGMENT_DELIMITER,
                    split_by_character_only=True,
                    ids=ids,
                )
                logger.info("Text content insertion complete")
                return

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
