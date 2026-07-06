"""
Utility functions for RAGAnything

Contains helper functions for content separation, text insertion, and other utilities
"""

import asyncio
import base64
import inspect
import math
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple
from pathlib import Path
from lightrag.utils import compute_mdhash_id, logger


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
_TABLE_INGEST_MARKER = "[Table]"
_INGEST_SEGMENT_DELIMITER = "\n<<<RAG_SEG_BOUNDARY>>>\n"
_SECTION_HEADING_LINE_RE = re.compile(r"^(?:\d+\.){1,3}\d+\s+\S")
_MAINTENANCE_FIELD_PREFIXES = ("保养周期：", "保养内容：", "保养步骤：")
_COALESCE_HEADING_MAX_CHARS = 120
_COALESCE_HEADING_LOOKAHEAD = 8
_COALESCE_IMAGE_LOOKAHEAD = 8
_COALESCE_METADATA_LOOKAHEAD = 8


def _is_image_ref_segment(segment: str) -> bool:
    return (segment or "").lstrip().startswith(_IMAGE_REF_MARKER)


def _dedupe_key_for_image_segment(segment: str) -> str:
    match = _INGEST_IMAGE_PATH_RE.search(segment or "")
    if match:
        return Path(match.group(1).strip().strip('"').strip("'")).name
    return (segment or "").strip()


def _segment_first_line(segment: str) -> str:
    return (segment or "").strip().split("\n", 1)[0].strip()


def _is_section_heading_line(line: str) -> bool:
    line = (line or "").strip()
    if not line or len(line) > _COALESCE_HEADING_MAX_CHARS:
        return False
    return bool(_SECTION_HEADING_LINE_RE.match(line))


def _is_orphan_heading_segment(segment: str) -> bool:
    seg = (segment or "").strip()
    if not seg or _is_image_ref_segment(seg) or _TABLE_INGEST_MARKER in seg:
        return False
    lines = [line.strip() for line in seg.splitlines() if line.strip()]
    if len(lines) != 1:
        return False
    return _is_section_heading_line(lines[0])


def _maintenance_field_prefix(line: str) -> str | None:
    line = (line or "").strip()
    for prefix in _MAINTENANCE_FIELD_PREFIXES:
        if line.startswith(prefix):
            return prefix
    return None


def _is_orphan_maintenance_field_segment(segment: str) -> bool:
    """Single parser field line(s) such as ``保养周期：…`` not yet merged into a section block."""
    seg = (segment or "").strip()
    if not seg or _is_image_ref_segment(seg) or _TABLE_INGEST_MARKER in seg:
        return False
    lines = [line.strip() for line in seg.splitlines() if line.strip()]
    if not lines:
        return False
    if len(lines) == 1:
        return _maintenance_field_prefix(lines[0]) is not None
    return all(_maintenance_field_prefix(line) is not None for line in lines)


def _is_orphan_maintenance_metadata_only_segment(segment: str) -> bool:
    """Short 保养内容/周期 field line(s), not a standalone 保养步骤 block."""
    if not _is_orphan_maintenance_field_segment(segment):
        return False
    lines = [line.strip() for line in (segment or "").splitlines() if line.strip()]
    if not lines:
        return False
    if len(lines) == 1:
        prefix = _maintenance_field_prefix(lines[0])
        return prefix in ("保养周期：", "保养内容：")
    return all(
        _maintenance_field_prefix(line) in ("保养周期：", "保养内容：") for line in lines
    )


def _segment_starts_new_section(segment: str) -> bool:
    return _is_section_heading_line(_segment_first_line(segment))


def _maintenance_section_step_closed(text: str) -> bool:
    """True once a subsection already has a procedure line or inline figure."""
    seg = (text or "").strip()
    return bool(seg) and ("保养步骤：" in seg or _IMAGE_REF_MARKER in seg)


def _section_heading_needs_field_merge(segment: str) -> bool:
    """Heading (+ optional image) block still missing 保养步骤 body."""
    seg = (segment or "").strip()
    if not seg or not _segment_starts_new_section(seg):
        return False
    if _is_orphan_heading_segment(seg):
        return False
    if "保养步骤：" in seg:
        return False
    if seg.count("保养内容：") >= 1 and seg.count("保养周期：") >= 1:
        return False
    return True


def _segment_has_maintenance_cycle(text: str) -> bool:
    return "保养周期：" in (text or "")


def _split_block_at_trailing_orphan_heading(block: str) -> List[str]:
    """Peel a trailing ``X.Y.Z …`` line off a text block (Q3 trailing 2.1.2)."""
    block = (block or "").strip()
    if not block:
        return []
    lines = block.splitlines()
    if len(lines) >= 2 and _is_section_heading_line(lines[-1].strip()):
        body = "\n".join(lines[:-1]).strip()
        heading = lines[-1].strip()
        if body:
            return [body, heading]
        return [heading]
    return [block]


def _split_overmerged_maintenance_segment(segment: str) -> List[str]:
    """Split one ingest segment that bundled multiple implicit subsections (Q7 page-9)."""
    seg = (segment or "").strip()
    if not seg or _is_image_ref_segment(seg) or _TABLE_INGEST_MARKER in seg:
        return [seg] if seg else []
    raw_blocks = [part.strip() for part in seg.split("\n\n") if part.strip()]
    blocks: List[str] = []
    for block in raw_blocks:
        if _is_image_ref_segment(block) or _is_orphan_heading_segment(block):
            blocks.append(block)
        else:
            blocks.extend(_split_block_at_trailing_orphan_heading(block))
    if len(blocks) <= 1:
        return [seg]
    sections: List[List[str]] = []
    current: List[str] = []
    for block in blocks:
        if _is_image_ref_segment(block):
            if current:
                current.append(block)
                sections.append(current)
                current = []
            else:
                sections.append([block])
            continue
        first_line = block.split("\n", 1)[0].strip()
        if _is_orphan_heading_segment(block):
            if current:
                sections.append(current)
            current = [block]
            continue
        prefix = _maintenance_field_prefix(first_line)
        joined = "\n\n".join(current)
        if prefix == "保养周期：" and current and _segment_has_maintenance_cycle(joined):
            sections.append(current)
            current = [block]
            continue
        if prefix == "保养内容：" and current and _maintenance_section_step_closed(joined):
            sections.append(current)
            current = [block]
            continue
        current.append(block)
    if current:
        sections.append(current)
    if len(sections) <= 1:
        return [seg]
    return ["\n\n".join(part for part in section if part) for section in sections if section]


def _explode_overmerged_segments(segments: List[str]) -> List[str]:
    out: List[str] = []
    for segment in segments:
        out.extend(_split_overmerged_maintenance_segment(segment))
    return out


def _segment_has_maintenance_body(segment: str) -> bool:
    seg = (segment or "").strip()
    if not seg:
        return False
    if _segment_starts_new_section(seg):
        return True
    if "保养内容：" in seg or "保养步骤：" in seg or "保养周期：" in seg:
        return True
    return False


def _follows_maintenance_subsection(segments: List[str], index: int) -> bool:
    """True when the next part is a 保养周期/内容 metadata line (not 保养步骤 body)."""
    if index + 1 >= len(segments):
        return False
    nxt = (segments[index + 1] or "").strip()
    if not nxt or _is_image_ref_segment(nxt):
        return False
    prefix = _maintenance_field_prefix(_segment_first_line(nxt))
    return prefix in ("保养周期：", "保养内容：")


def _has_backward_procedure_for_heading(segments: List[str], index: int) -> bool:
    """Orphan 保养步骤 before a heading may still belong to that section (Q7)."""
    for j in range(max(0, index - _COALESCE_METADATA_LOOKAHEAD), index):
        cand = (segments[j] or "").strip()
        if not cand or _is_image_ref_segment(cand):
            continue
        if _is_orphan_maintenance_field_segment(cand):
            if _maintenance_field_prefix(_segment_first_line(cand)) == "保养步骤：":
                return True
        elif "保养步骤：" in cand and not _segment_starts_new_section(cand):
            return True
    return False


def _should_collect_maintenance_subsection(segments: List[str], index: int) -> bool:
    return _follows_maintenance_subsection(segments, index) and not _has_backward_procedure_for_heading(
        segments, index
    )


def _collect_maintenance_section_parts(
    segments: List[str],
    start: int,
) -> tuple[List[str], int]:
    """Collect heading/field lines and optional immediate ``[图片]`` for one subsection."""
    parts = [segments[start]]
    j = start + 1
    while j < len(segments):
        nxt = (segments[j] or "").strip()
        if not nxt:
            j += 1
            continue
        if _is_orphan_heading_segment(nxt):
            break
        if _segment_starts_new_section(nxt) and not _is_orphan_maintenance_field_segment(nxt):
            break
        if _is_orphan_maintenance_field_segment(nxt):
            joined = "\n\n".join(parts)
            nxt_prefix = _maintenance_field_prefix(_segment_first_line(nxt))
            if nxt_prefix == "保养周期：" and _segment_has_maintenance_cycle(joined):
                break
            if nxt_prefix == "保养内容：" and _maintenance_section_step_closed(joined):
                break
            parts.append(nxt)
            j += 1
            continue
        if _is_image_ref_segment(nxt):
            parts.append(nxt)
            j += 1
            break
        if _TABLE_INGEST_MARKER in nxt:
            break
        if len(_segment_first_line(nxt)) <= _COALESCE_HEADING_MAX_CHARS:
            parts.append(nxt)
            j += 1
            continue
        break
    return parts, j


def _assemble_maintenance_section_segments(segments: List[str]) -> List[str]:
    """Group numbered headings with following 保养内容/周期/步骤 fields before the figure."""
    working = [(segment or "").strip() for segment in segments if (segment or "").strip()]
    out: List[str] = []
    i = 0
    while i < len(working):
        seg = working[i]
        if _is_image_ref_segment(seg) or _TABLE_INGEST_MARKER in seg:
            out.append(seg)
            i += 1
            continue
        if _is_orphan_heading_segment(seg):
            if _should_collect_maintenance_subsection(working, i):
                parts, i = _collect_maintenance_section_parts(working, i)
                out.append("\n\n".join(parts))
                continue
            out.append(seg)
            i += 1
            continue
        if _segment_starts_new_section(seg):
            parts, i = _collect_maintenance_section_parts(working, i)
            out.append("\n\n".join(parts))
            continue
        if _is_orphan_maintenance_metadata_only_segment(seg):
            parts, i = _collect_maintenance_section_parts(working, i)
            out.append("\n\n".join(parts))
            continue
        out.append(seg)
        i += 1
    return out


def _merge_trailing_procedure_into_preceding(segments: List[str]) -> List[str]:
    """Attach orphan ``保养步骤：…`` lines to the preceding section block (Q7/Q13)."""
    working = [(segment or "").strip() for segment in segments if (segment or "").strip()]
    max_passes = max(len(working) * 2, 8)
    for _ in range(max_passes):
        changed = False
        i = 1
        while i < len(working):
            seg = working[i]
            if not _is_orphan_maintenance_field_segment(seg):
                i += 1
                continue
            if _maintenance_field_prefix(_segment_first_line(seg)) != "保养步骤：":
                i += 1
                continue
            prev = working[i - 1]
            if _is_image_ref_segment(prev) or _TABLE_INGEST_MARKER in prev:
                i += 1
                continue
            if not _segment_has_maintenance_body(prev):
                i += 1
                continue
            working[i - 1] = f"{prev}\n\n{seg}"
            del working[i]
            changed = True
        if not changed:
            break
    return working


def _merge_unheaded_fields_with_heading_sections(segments: List[str]) -> List[str]:
    """Merge anonymous 保养内容/步骤 blocks into a later thin heading section (Q7)."""
    working = [(segment or "").strip() for segment in segments if (segment or "").strip()]
    skip: set[int] = set()
    for i in range(len(working)):
        if i in skip:
            continue
        seg = working[i]
        if not _section_heading_needs_field_merge(seg):
            continue
        heading = _segment_first_line(seg)
        best_j = -1
        best_score = 0.12
        for j in range(max(0, i - _COALESCE_METADATA_LOOKAHEAD), i):
            if j in skip:
                continue
            cand = working[j]
            if _is_image_ref_segment(cand) or _TABLE_INGEST_MARKER in cand:
                continue
            if _is_orphan_heading_segment(cand):
                continue
            if _segment_starts_new_section(cand) and "保养步骤：" in cand:
                continue
            score = text_term_alignment_symmetric(heading, cand)
            if score > best_score:
                best_score = score
                best_j = j
        if best_j < 0:
            continue
        working[i] = f"{working[best_j]}\n\n{seg}"
        skip.add(best_j)
    return [seg for idx, seg in enumerate(working) if idx not in skip]


def _merge_orphan_maintenance_metadata_segments(segments: List[str]) -> List[str]:
    """Forward-merge orphan field lines into the next body block within a short window."""
    working = [(segment or "").strip() for segment in segments if (segment or "").strip()]
    max_passes = max(len(working) * 2, 8)
    for _ in range(max_passes):
        changed = False
        i = 0
        while i < len(working):
            if not _is_orphan_maintenance_field_segment(working[i]):
                i += 1
                continue
            meta_parts: List[str] = []
            j = i
            while j < len(working) and _is_orphan_maintenance_field_segment(working[j]):
                meta_parts.append(working[j])
                j += 1
            if meta_parts and _maintenance_section_step_closed("\n\n".join(meta_parts)):
                i += len(meta_parts)
                continue
            if j >= len(working):
                break
            best_k = -1
            best_score = 0.0
            for k in range(j, min(j + _COALESCE_METADATA_LOOKAHEAD + 1, len(working))):
                cand = working[k]
                if _is_orphan_heading_segment(cand) or _is_image_ref_segment(cand):
                    continue
                if _segment_starts_new_section(cand) and not _is_orphan_maintenance_field_segment(
                    cand
                ):
                    continue
                anchor = " ".join(meta_parts)
                score = text_term_alignment_symmetric(anchor, cand)
                if score >= 0.08 and score >= best_score:
                    best_score = score
                    best_k = k
            if best_k < 0 and j < len(working):
                cand = working[j]
                if (
                    not _is_orphan_heading_segment(cand)
                    and not _is_image_ref_segment(cand)
                    and not (
                        _segment_starts_new_section(cand)
                        and not _is_orphan_maintenance_field_segment(cand)
                    )
                ):
                    best_k = j
            if best_k < 0:
                i += 1
                continue
            merged_parts = meta_parts + [working[best_k]]
            remove_at = [best_k]
            if (
                best_k + 1 < len(working)
                and _is_image_ref_segment(working[best_k + 1])
                and best_k == j
            ):
                merged_parts.append(working[best_k + 1])
                remove_at.append(best_k + 1)
            working[i] = "\n\n".join(merged_parts)
            for idx in sorted(set(remove_at), reverse=True):
                if idx != i:
                    del working[idx]
            changed = True
            i += 1
        if not changed:
            break
    return working


def _segment_has_inline_image(segment: str) -> bool:
    seg = (segment or "").strip()
    return bool(seg) and (_is_image_ref_segment(seg) or _IMAGE_REF_MARKER in seg)


def _context_from_ref_segment(segment: str) -> str:
    for line in (segment or "").splitlines():
        if line.startswith("关联正文："):
            return line[5:].strip()
    return ""


def _image_label_from_ref_segment(segment: str) -> str:
    caption = ""
    footnote = ""
    for line in (segment or "").splitlines():
        if line.startswith("图注："):
            caption = line[3:].strip()
        elif line.startswith("脚注："):
            footnote = line[3:].strip()
    return " ".join(part for part in (caption, footnote) if part).strip()


def _image_match_text_from_ref_segment(segment: str) -> str:
    """Caption, footnote, or ingest ``关联正文`` for term-overlap pairing."""
    label = _image_label_from_ref_segment(segment)
    context = _context_from_ref_segment(segment)
    return " ".join(part for part in (label, context) if part).strip()


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


def _split_image_blocks_in_segment(segment: str) -> List[str]:
    """Split one segment that contains multiple ``[图片]`` blocks into smaller parts."""
    seg = (segment or "").strip()
    if not seg or _TABLE_INGEST_MARKER in seg:
        return [seg] if seg else []
    parts = [part.strip() for part in seg.split("\n\n") if part.strip()]
    image_parts = [part for part in parts if _is_image_ref_segment(part)]
    if len(image_parts) <= 1:
        return [seg]
    text_parts = [part for part in parts if not _is_image_ref_segment(part)]
    leading_text = "\n\n".join(text_parts).strip()
    out: List[str] = []
    emitted_leading = False
    for img_part in image_parts:
        match_text = _image_match_text_from_ref_segment(img_part)
        if leading_text and text_term_alignment_symmetric(leading_text, match_text) >= 0.28:
            out.append(f"{leading_text}\n\n{img_part}")
            emitted_leading = True
        else:
            if leading_text and not emitted_leading:
                out.append(leading_text)
                emitted_leading = True
            out.append(img_part)
    if leading_text and not emitted_leading:
        out.insert(0, leading_text)
    return out


def _split_multi_image_segments(segments: List[str]) -> List[str]:
    out: List[str] = []
    for segment in segments:
        out.extend(_split_image_blocks_in_segment(segment))
    return out


def _heading_body_merge_score(
    heading: str,
    candidate: str,
    segments: List[str],
    candidate_index: int,
) -> float:
    score = text_term_alignment_symmetric(heading, candidate)
    inline_labels: List[str] = []
    if _IMAGE_REF_MARKER in candidate:
        for part in candidate.split("\n\n"):
            if not _is_image_ref_segment(part):
                continue
            label = _image_label_from_ref_segment(part)
            if label:
                inline_labels.append(label)
    for label in inline_labels:
        img_score = text_term_alignment_symmetric(heading, label)
        score = max(score, img_score * 0.92 + score * 0.08)

    nearest_img = 0.0
    if candidate_index + 1 < len(segments):
        nxt = (segments[candidate_index + 1] or "").strip()
        if _is_image_ref_segment(nxt):
            label = _image_label_from_ref_segment(nxt)
            if label:
                nearest_img = text_term_alignment_symmetric(heading, label)
                score = max(score, nearest_img * 0.92 + score * 0.08)

    if not inline_labels and not nearest_img:
        for k in range(
            candidate_index + 1,
            min(candidate_index + _COALESCE_HEADING_LOOKAHEAD + 1, len(segments)),
        ):
            img_seg = (segments[k] or "").strip()
            if _is_orphan_heading_segment(img_seg):
                continue
            if _is_image_ref_segment(img_seg):
                label = _image_label_from_ref_segment(img_seg)
                if label:
                    align = text_term_alignment_symmetric(heading, label)
                    nearest_img = max(nearest_img, align)
                    score = max(score, align * 0.88 + score * 0.12)
                break
            if _segment_has_maintenance_body(img_seg):
                break

    if inline_labels:
        best_inline = max(
            text_term_alignment_symmetric(heading, label) for label in inline_labels
        )
        if best_inline >= 0.28:
            score = max(score, best_inline * 0.88 + score * 0.12)
        elif score >= 0.18 and best_inline < 0.18:
            score *= 0.25
    elif nearest_img >= 0.28:
        score = max(score, nearest_img * 0.88 + score * 0.12)
    elif nearest_img >= 0.12 and score >= 0.18 and nearest_img < 0.18:
        score *= 0.55
    return score


def _best_heading_body_score_before(
    heading: str,
    working: List[str],
    heading_index: int,
) -> float:
    best = 0.0
    for j in range(max(0, heading_index - _COALESCE_HEADING_LOOKAHEAD), heading_index):
        cand = working[j]
        if _is_orphan_heading_segment(cand) or _is_image_ref_segment(cand):
            continue
        if _TABLE_INGEST_MARKER in cand:
            continue
        if not _segment_has_maintenance_body(cand):
            continue
        best = max(best, _heading_body_merge_score(heading, cand, working, j))
    return best


def _maintenance_step_alignment(heading: str, candidate: str) -> float:
    best = 0.0
    for line in (candidate or "").splitlines():
        if line.startswith("保养步骤："):
            best = max(best, text_term_alignment_symmetric(heading, line))
    return best


def _merge_trailing_orphan_headings_backward(working: List[str]) -> List[str]:
    """Exclusive backward pairing when headings trail their body blocks (Q3)."""
    heading_indices = [
        idx for idx, seg in enumerate(working) if _is_orphan_heading_segment(seg)
    ]
    if not heading_indices:
        return working
    body_indices = [
        idx
        for idx, seg in enumerate(working)
        if not _is_orphan_heading_segment(seg)
        and not _is_image_ref_segment(seg)
        and _TABLE_INGEST_MARKER not in seg
        and _segment_has_maintenance_body(seg)
    ]
    if not body_indices:
        return working
    pairs: List[tuple[float, float, int, int, int]] = []
    for hi in heading_indices:
        heading = working[hi]
        for bi in body_indices:
            score = _heading_body_merge_score(heading, working[bi], working, bi)
            if score >= 0.12:
                step_align = _maintenance_step_alignment(heading, working[bi])
                pairs.append((score, step_align, -abs(hi - bi), hi, bi))
    pairs.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    used_headings: set[int] = set()
    used_bodies: set[int] = set()
    for _score, _step, _dist, hi, bi in pairs:
        if hi in used_headings or bi in used_bodies:
            continue
        working[bi] = f"{working[hi]}\n\n{working[bi]}"
        used_headings.add(hi)
        used_bodies.add(bi)
    return [seg for idx, seg in enumerate(working) if idx not in used_headings]


def _merge_orphan_heading_segments(segments: List[str]) -> List[str]:
    """Attach orphan section headings (e.g. ``2.1.1 …``) to the best-aligned body block."""
    working = [(segment or "").strip() for segment in segments if (segment or "").strip()]
    max_passes = max(len(working) * 2, 8)
    for _pass in range(max_passes):
        changed = False
        i = 0
        while i < len(working):
            if not _is_orphan_heading_segment(working[i]):
                i += 1
                continue
            heading = working[i]
            best_j = -1
            best_score = 0.12
            for j in range(i + 1, min(i + _COALESCE_HEADING_LOOKAHEAD + 1, len(working))):
                cand = working[j]
                if _is_orphan_heading_segment(cand):
                    break
                if _TABLE_INGEST_MARKER in cand or _is_image_ref_segment(cand):
                    continue
                score = _heading_body_merge_score(heading, cand, working, j)
                if score > best_score:
                    best_score = score
                    best_j = j
            backward_best = _best_heading_body_score_before(heading, working, i)
            if best_j >= 0 and backward_best >= 0.12 and backward_best >= best_score * 0.38:
                i += 1
                continue
            if best_j < 0:
                i += 1
                continue
            absorb_image = (
                best_j + 1 < len(working)
                and _is_image_ref_segment(working[best_j + 1])
            )
            merged = f"{heading}\n\n{working[best_j]}"
            if absorb_image:
                merged = f"{merged}\n\n{working[best_j + 1]}"
            working[i] = merged
            remove_at = [best_j]
            if absorb_image:
                remove_at.append(best_j + 1)
            for idx in sorted(remove_at, reverse=True):
                if idx != i:
                    del working[idx]
            changed = True
            i += 1
        if not changed:
            break
    return _merge_trailing_orphan_headings_backward(working)


def _coalesce_immediate_text_image_segments(segments: List[str]) -> List[str]:
    """Merge each text segment with immediately following ``[图片]`` blocks."""
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
            if j > i + 1:
                break
            match_text = _image_match_text_from_ref_segment(nxt)
            if match_text and text_term_alignment_symmetric(seg, match_text) < 0.2:
                break
            key = _dedupe_key_for_image_segment(nxt)
            if key not in seen_images:
                seen_images.add(key)
                parts.append(nxt)
            j += 1
        out.append("\n\n".join(parts))
        i = j
    return out


def _lookahead_pair_text_image_segments(
    segments: List[str],
    *,
    max_lookahead: int = _COALESCE_IMAGE_LOOKAHEAD,
) -> List[str]:
    """Pair text-only segments with label-aligned ``[图片]`` blocks within a short window."""
    skip: set[int] = set()
    out: List[str] = []
    min_score = max(0.22, _LABEL_ALIGN_MIN * 0.8)
    n = len(segments)
    for i in range(n):
        if i in skip:
            continue
        seg = (segments[i] or "").strip()
        if not seg:
            continue
        if _segment_has_inline_image(seg) or _TABLE_INGEST_MARKER in seg:
            out.append(seg)
            continue
        best_k = -1
        best_score = 0.0
        for k in range(i + 1, min(i + max_lookahead + 1, n)):
            if k in skip:
                continue
            img_seg = (segments[k] or "").strip()
            if not _is_image_ref_segment(img_seg):
                continue
            match_text = _image_match_text_from_ref_segment(img_seg)
            score = (
                text_term_alignment_symmetric(seg, match_text) if match_text else 0.0
            )
            if k == i + 1 and score < min_score:
                score = max(score, 0.28)
            score += _anchor_pairing_priority(seg)
            if score >= min_score and score >= best_score:
                best_score = score
                best_k = k
        if best_k >= 0:
            out.append(f"{seg}\n\n{segments[best_k].strip()}")
            skip.add(best_k)
        else:
            out.append(seg)
    return out


def coalesce_text_image_segments(segments: List[str]) -> List[str]:
    """Merge anchor text with inline figures for indexing (P2: headings + lookahead).

    Keeps figure metadata in the same vector chunk as the anchor body so rerank +
    steering filter text and inline figures together (Route A ingest).
    """
    if not segments:
        return []
    exploded = _explode_overmerged_segments(segments)
    assembled = _assemble_maintenance_section_segments(exploded)
    metadata = _merge_orphan_maintenance_metadata_segments(assembled)
    bridged = _merge_unheaded_fields_with_heading_sections(metadata)
    merged = _merge_orphan_heading_segments(bridged)
    split = _split_multi_image_segments(merged)
    paired = _lookahead_pair_text_image_segments(split)
    trailing = _merge_trailing_procedure_into_preceding(paired)
    return _coalesce_immediate_text_image_segments(trailing)


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
            label_score = (
                text_term_alignment_symmetric(anchor, label) if label else 0.0
            )
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


_TABLE_ROW_RE = re.compile(r"<tr>.*?</tr>", re.IGNORECASE | re.DOTALL)


def compute_ingest_chunk_id(
    full_doc_id: str, chunk_order_index: int, content: str
) -> str:
    """Scope chunk ids by document so identical text in different manuals stays distinct."""
    return compute_mdhash_id(
        f"{full_doc_id}:{chunk_order_index}:{content}", prefix="chunk-"
    )


def _status_field(status_doc: Any, name: str, default: Any = "") -> Any:
    if status_doc is None:
        return default
    if isinstance(status_doc, dict):
        return status_doc.get(name, default)
    return getattr(status_doc, name, default)


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


def _part_starts_section_heading(part: str) -> bool:
    return _is_section_heading_line(_segment_first_line(part))


def _split_sub_parts_at_section_heading_boundaries(sub_parts: List[str]) -> List[List[str]]:
    """Split ingest sub-parts so coalesce never crosses numbered section headings."""
    groups: List[List[str]] = []
    current: List[str] = []
    for part in sub_parts:
        piece = (part or "").strip()
        if not piece:
            continue
        if current and _part_starts_section_heading(piece):
            groups.append(current)
            current = [piece]
        else:
            current.append(piece)
    if current:
        groups.append(current)
    return groups


def _coalesce_sub_parts_by_section(sub_parts: List[str]) -> List[str]:
    """Run P2 coalesce within each numbered-section batch (avoids mega-doc cross-merge)."""
    out: List[str] = []
    for group in _split_sub_parts_at_section_heading_boundaries(sub_parts):
        out.extend(coalesce_text_image_segments(group))
    return out


def coalesce_parts_for_embedding_ingest(
    document_parts: List[str] | None,
    *,
    text_content: str = "",
) -> List[str]:
    """Coalesce ingest parts/chunks without token splitting (embedding-only path)."""
    if document_parts:
        if table_aware_ingest_enabled():
            return build_table_aware_ingest_segments(document_parts)
        parts = [p.strip() for p in document_parts if p.strip()]
        return _coalesce_sub_parts_by_section(parts) if parts else []
    segments = [s.strip() for s in (text_content or "").split("\n\n") if s.strip()]
    return _coalesce_sub_parts_by_section(segments) if segments else []


def build_table_aware_ingest_segments(document_parts: List[str]) -> List[str]:
    """Keep each ``[Table]`` block intact; coalesce adjacent non-table parts."""
    segments: List[str] = []
    buf: List[str] = []

    def flush_buf() -> None:
        nonlocal buf
        if not buf:
            return
        sub_parts = [p.strip() for p in buf if (p or "").strip()]
        segments.extend(_coalesce_sub_parts_by_section(sub_parts))
        buf = []

    for part in document_parts:
        piece = (part or "").strip()
        if not piece:
            continue
        if _TABLE_INGEST_MARKER in piece:
            flush_buf()
            segments.append(piece)
            continue
        if buf and _part_starts_section_heading(piece):
            flush_buf()
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


async def _resolve_ingest_segments(
    lightrag,
    *,
    input_text: str,
    document_parts: List[str] | None,
    split_by_character: str | None,
    split_by_character_only: bool,
) -> List[str]:
    if table_aware_ingest_enabled() and document_parts:
        tokenizer = getattr(lightrag, "tokenizer", None)
        if tokenizer is not None:
            max_tokens = _ingest_chunk_token_size(lightrag)
            segments = prepare_table_aware_ingest_segments(
                document_parts, tokenizer, max_tokens
            )
            if segments:
                return segments

    chunking_result = lightrag.chunking_func(
        lightrag.tokenizer,
        input_text,
        split_by_character,
        split_by_character_only,
        lightrag.chunk_overlap_token_size,
        lightrag.chunk_token_size,
    )
    if inspect.isawaitable(chunking_result):
        chunking_result = await chunking_result
    if not isinstance(chunking_result, (list, tuple)):
        raise TypeError(
            f"chunking_func must return a list or tuple of dicts, got {type(chunking_result)}"
        )
    return [
        str(row["content"])
        for row in chunking_result
        if isinstance(row, dict) and str(row.get("content") or "").strip()
    ]


async def insert_doc_scoped_text_content(
    lightrag,
    *,
    enqueue_input: str,
    doc_id: str,
    file_path: str,
    segments: List[str],
    ids: str | list[str] | None,
    file_paths: str | list[str] | None,
) -> None:
    """Insert text chunks with per-document chunk ids; runs KG extract + merge like ``ainsert``."""
    from lightrag.base import DocStatus
    from lightrag.kg.shared_storage import get_namespace_data, get_pipeline_status_lock
    from lightrag.operate import merge_nodes_and_edges

    pipeline_status = await get_namespace_data("pipeline_status")
    pipeline_status_lock = get_pipeline_status_lock()

    await lightrag.apipeline_enqueue_documents(
        enqueue_input, ids=ids, file_paths=file_paths
    )

    status_doc = await lightrag.doc_status.get_by_id(doc_id)
    processing_start_time = int(datetime.now(timezone.utc).timestamp())

    chunks: Dict[str, Dict[str, Any]] = {}
    order = 0
    for seg in segments:
        content = (seg or "").strip()
        if not content:
            continue
        chunk_id = compute_ingest_chunk_id(doc_id, order, content)
        try:
            tokens = len(lightrag.tokenizer.encode(content))
        except Exception:
            tokens = len(content.split())
        chunks[chunk_id] = {
            "content": content,
            "full_doc_id": doc_id,
            "file_path": file_path,
            "chunk_order_index": order,
            "tokens": tokens,
            "llm_cache_list": [],
        }
        order += 1

    if not chunks:
        logger.warning("Doc-scoped ingest: no segments for doc_id=%s", doc_id)
        return

    await lightrag.doc_status.upsert(
        {
            doc_id: {
                "status": DocStatus.PROCESSING,
                "chunks_count": len(chunks),
                "chunks_list": list(chunks.keys()),
                "content_summary": _status_field(status_doc, "content_summary"),
                "content_length": _status_field(status_doc, "content_length"),
                "created_at": _status_field(status_doc, "created_at"),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "file_path": file_path,
                "track_id": _status_field(status_doc, "track_id"),
                "metadata": {"processing_start_time": processing_start_time},
            }
        }
    )

    await asyncio.gather(
        lightrag.chunks_vdb.upsert(chunks),
        lightrag.text_chunks.upsert(chunks),
    )

    chunk_results = await lightrag._process_extract_entities(
        chunks, pipeline_status, pipeline_status_lock
    )

    await merge_nodes_and_edges(
        chunk_results=chunk_results,
        knowledge_graph_inst=lightrag.chunk_entity_relation_graph,
        entity_vdb=lightrag.entities_vdb,
        relationships_vdb=lightrag.relationships_vdb,
        global_config=asdict(lightrag),
        full_entities_storage=lightrag.full_entities,
        full_relations_storage=lightrag.full_relations,
        doc_id=doc_id,
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
        llm_response_cache=lightrag.llm_response_cache,
        entity_chunks_storage=lightrag.entity_chunks,
        relation_chunks_storage=lightrag.relation_chunks,
        current_file_number=1,
        total_files=1,
        file_path=file_path,
    )

    processing_end_time = int(datetime.now(timezone.utc).timestamp())
    await lightrag.doc_status.upsert(
        {
            doc_id: {
                "status": DocStatus.PROCESSED,
                "chunks_count": len(chunks),
                "chunks_list": list(chunks.keys()),
                "content_summary": _status_field(status_doc, "content_summary"),
                "content_length": _status_field(status_doc, "content_length"),
                "created_at": _status_field(status_doc, "created_at"),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "file_path": file_path,
                "track_id": _status_field(status_doc, "track_id"),
                "metadata": {
                    "processing_start_time": processing_start_time,
                    "processing_end_time": processing_end_time,
                },
            }
        }
    )
    await lightrag._insert_done()


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

    if isinstance(input, str) and ids is not None:
        doc_id = ids if isinstance(ids, str) else ids[0]
        file_path = ""
        if file_paths:
            file_path = file_paths if isinstance(file_paths, str) else file_paths[0]
        file_path = file_path or "unknown_source"

        segments = await _resolve_ingest_segments(
            lightrag,
            input_text=input,
            document_parts=document_parts,
            split_by_character=split_by_character,
            split_by_character_only=split_by_character_only,
        )
        if segments:
            enqueue_input = input
            if (
                table_aware_ingest_enabled()
                and document_parts
                and _INGEST_SEGMENT_DELIMITER.join(segments) != input.strip()
            ):
                enqueue_input = _INGEST_SEGMENT_DELIMITER.join(segments)
            logger.info(
                "Doc-scoped chunk ingest: %d segment(s) for %s",
                len(segments),
                file_path,
            )
            await insert_doc_scoped_text_content(
                lightrag,
                enqueue_input=enqueue_input,
                doc_id=doc_id,
                file_path=file_path,
                segments=segments,
                ids=ids,
                file_paths=file_paths,
            )
            logger.info("Text content insertion complete")
            return

    # Fallback: legacy LightRAG path (content-only chunk ids)
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
