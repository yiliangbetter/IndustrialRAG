"""Ingest segment coalescing and table-aware segmentation.

Split from utils.py (stage-1 refactor, see
docs/utils_refactor_schema_induction.md).
"""

import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .text_align import (
    _COALESCE_HEADING_MAX_CHARS,
    _is_section_heading_line,
    text_term_alignment_symmetric,
)
from .image_context import (
    _LABEL_ALIGN_MIN,
    _anchor_pairing_priority,
    context_text_for_image,
    resolve_image_caption,
    resolve_image_footnote,
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


_INGEST_IMAGE_PATH_RE = re.compile(r"图片路径[：:]\s*(.+?)(?:\n|$)", re.MULTILINE)


_TABLE_INGEST_MARKER = "[Table]"


_INGEST_SEGMENT_DELIMITER = "\n<<<RAG_SEG_BOUNDARY>>>\n"


# --- Per-document record-field schema induction -------------------------
#
# Record-style manuals (e.g. maintenance logs) repeat a fixed sequence of
# ``键：值`` field lines such as ``保养周期：… / 保养内容：… / 保养步骤：…``.
# The coalesce record-boundary / step-closed logic used to hardcode those
# three Chinese field names.  Instead we induce each document's field roles
# from its own structure (run-signature induction) so no business-domain
# literal stays in the code.  A field line is any ``键：`` line whose key is
# 2-8 chars long; a run is a maximal sequence of consecutive field lines
# (image-ref segments do not break a run).  A signature repeating >=
# _SCHEMA_MIN_REPEAT times with >= _SCHEMA_MIN_KEYS distinct keys marks a
# record schema: first key = initiator (record boundary), last key of the
# most frequent signature = closer (procedure / step-closed), the rest are
# metadata.  Documents without a dominant signature get schema=None and the
# record-field logic short-circuits to a no-op.
_FIELD_KEY_RE = re.compile(r"^([^\s：:，。、,.]{2,8})：")

_SCHEMA_MIN_REPEAT = 3

_SCHEMA_MIN_KEYS = 2


@dataclass(frozen=True)
class DocFieldSchema:
    """Induced ``键：值`` record-field roles for one document/section batch."""

    initiator: str
    closer: str
    metadata_keys: frozenset

    @property
    def field_keys(self) -> frozenset:
        return frozenset({self.initiator, self.closer, *self.metadata_keys})

    def field_key(self, line: str) -> str | None:
        """Return the leading key when ``line`` starts a schema field line."""
        match = _FIELD_KEY_RE.match((line or "").strip())
        if match and match.group(1) in self.field_keys:
            return match.group(1)
        return None

    def is_initiator(self, key: str | None) -> bool:
        return key == self.initiator

    def is_closer(self, key: str | None) -> bool:
        return key == self.closer

    def is_metadata(self, key: str | None) -> bool:
        return key in self.metadata_keys


def _line_field_key(line: str) -> str | None:
    """Leading ``键：`` key (2-8 chars) of any line, schema-independent."""
    match = _FIELD_KEY_RE.match((line or "").strip())
    return match.group(1) if match else None


def induce_field_schema(segments: List[str]) -> DocFieldSchema | None:
    """Induce record-field roles from a batch of ingest segments.

    Returns ``None`` when no key-sequence signature repeats often enough,
    which disables the record-field coalesce logic for that batch.
    """
    runs: List[tuple] = []
    current: List[str] = []
    for segment in segments:
        seg = (segment or "").strip()
        if not seg:
            continue
        if _is_image_ref_segment(seg) or _TABLE_INGEST_MARKER in seg:
            continue  # image / table segments do not break a field run
        key = _line_field_key(seg.split("\n", 1)[0])
        if key:
            current.append(key)
        elif current:
            runs.append(tuple(current))
            current = []
    if current:
        runs.append(tuple(current))

    counts = Counter(runs)
    candidates = [
        sig
        for sig, count in counts.items()
        if count >= _SCHEMA_MIN_REPEAT and len(set(sig)) >= _SCHEMA_MIN_KEYS
    ]
    if not candidates:
        return None

    families: Dict[str, List[tuple]] = {}
    for sig in candidates:
        families.setdefault(sig[0], []).append(sig)
    family = max(
        families.values(), key=lambda sigs: sum(counts[s] for s in sigs)
    )
    best = max(family, key=lambda s: counts[s])
    initiator = best[0]
    closer = best[-1]
    metadata = frozenset(k for sig in family for k in sig if k not in (initiator, closer))
    return DocFieldSchema(
        initiator=initiator, closer=closer, metadata_keys=metadata
    )


def _segment_field_keys(segment: str, schema: DocFieldSchema | None) -> List[str]:
    """Schema field keys appearing at the start of any line in ``segment``."""
    if schema is None:
        return []
    seg = (segment or "").strip()
    if not seg:
        return []
    keys: List[str] = []
    for line in seg.splitlines():
        key = schema.field_key(line)
        if key:
            keys.append(key)
    return keys


def _segment_has_field_line(segment: str, schema: DocFieldSchema | None) -> bool:
    return bool(_segment_field_keys(segment, schema))


def _segment_has_initiator(segment: str, schema: DocFieldSchema | None) -> bool:
    if schema is None:
        return False
    return any(key == schema.initiator for key in _segment_field_keys(segment, schema))


def _segment_has_closer(segment: str, schema: DocFieldSchema | None) -> bool:
    if schema is None:
        return False
    return any(key == schema.closer for key in _segment_field_keys(segment, schema))


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


def _is_orphan_heading_segment(segment: str) -> bool:
    seg = (segment or "").strip()
    if not seg or _is_image_ref_segment(seg) or _TABLE_INGEST_MARKER in seg:
        return False
    lines = [line.strip() for line in seg.splitlines() if line.strip()]
    if len(lines) != 1:
        return False
    return _is_section_heading_line(lines[0])


def _is_orphan_record_field_segment(
    segment: str, schema: DocFieldSchema | None
) -> bool:
    """Single parser field line(s) such as ``保养周期：…`` not yet merged into a section block."""
    if schema is None:
        return False
    seg = (segment or "").strip()
    if not seg or _is_image_ref_segment(seg) or _TABLE_INGEST_MARKER in seg:
        return False
    lines = [line.strip() for line in seg.splitlines() if line.strip()]
    if not lines:
        return False
    return all(schema.field_key(line) is not None for line in lines)


def _is_orphan_record_metadata_only_segment(
    segment: str, schema: DocFieldSchema | None
) -> bool:
    """Initiator/metadata field line(s) only, not a standalone closer block."""
    if schema is None:
        return False
    if not _is_orphan_record_field_segment(segment, schema):
        return False
    lines = [line.strip() for line in (segment or "").splitlines() if line.strip()]
    if not lines:
        return False
    return all(schema.field_key(line) != schema.closer for line in lines)


def _segment_starts_new_section(segment: str) -> bool:
    return _is_section_heading_line(_segment_first_line(segment))


def _record_section_step_closed(text: str, schema: DocFieldSchema | None) -> bool:
    """True once a subsection already has a closer (procedure) field line."""
    return _segment_has_closer(text, schema)


def _section_heading_needs_field_merge(
    segment: str, schema: DocFieldSchema | None
) -> bool:
    """Heading (+ optional image) block still missing its closer body."""
    seg = (segment or "").strip()
    if not seg or not _segment_starts_new_section(seg):
        return False
    if _is_orphan_heading_segment(seg):
        return False
    if schema is None:
        return False
    if _segment_has_closer(seg, schema):
        return False
    if _segment_has_initiator(seg, schema) and any(
        schema.is_metadata(key) for key in _segment_field_keys(seg, schema)
    ):
        return False
    return True


def _segment_has_record_cycle(text: str, schema: DocFieldSchema | None) -> bool:
    return _segment_has_initiator(text, schema)


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


def _split_overmerged_record_segment(
    segment: str, schema: DocFieldSchema | None
) -> List[str]:
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
            else:
                current = [block]
            continue
        first_line = block.split("\n", 1)[0].strip()
        if _is_orphan_heading_segment(block):
            if current:
                sections.append(current)
            current = [block]
            continue
        first_key = schema.field_key(first_line) if schema else None
        joined = "\n\n".join(current)
        if (
            schema is not None
            and first_key == schema.initiator
            and current
            and _segment_has_initiator(joined, schema)
        ):
            sections.append(current)
            current = [block]
            continue
        if (
            schema is not None
            and schema.is_metadata(first_key)
            and current
            and _record_section_step_closed(joined, schema)
        ):
            sections.append(current)
            current = [block]
            continue
        current.append(block)
    if current:
        sections.append(current)
    if len(sections) <= 1:
        return [seg]
    return [
        "\n\n".join(part for part in section if part) for section in sections if section
    ]


def _explode_overmerged_segments(
    segments: List[str], schema: DocFieldSchema | None
) -> List[str]:
    out: List[str] = []
    for segment in segments:
        out.extend(_split_overmerged_record_segment(segment, schema))
    return out


def _segment_has_record_body(segment: str, schema: DocFieldSchema | None) -> bool:
    seg = (segment or "").strip()
    if not seg:
        return False
    if _segment_starts_new_section(seg):
        return True
    if _segment_has_field_line(seg, schema):
        return True
    return False


def _follows_record_subsection(
    segments: List[str], index: int, schema: DocFieldSchema | None
) -> bool:
    """True when the next part is an initiator/metadata line (not closer body)."""
    if schema is None:
        return False
    if index + 1 >= len(segments):
        return False
    nxt = (segments[index + 1] or "").strip()
    if not nxt or _is_image_ref_segment(nxt):
        return False
    key = schema.field_key(_segment_first_line(nxt))
    return key is not None and key != schema.closer


def _has_backward_procedure_for_heading(
    segments: List[str], index: int, schema: DocFieldSchema | None
) -> bool:
    """Orphan closer block before a heading may still belong to that section (Q7)."""
    if schema is None:
        return False
    for j in range(max(0, index - _COALESCE_METADATA_LOOKAHEAD), index):
        cand = (segments[j] or "").strip()
        if not cand or _is_image_ref_segment(cand):
            continue
        if _is_orphan_record_field_segment(cand, schema):
            if schema.field_key(_segment_first_line(cand)) == schema.closer:
                return True
        elif _segment_has_closer(cand, schema) and not _segment_starts_new_section(cand):
            return True
    return False


def _should_collect_record_subsection(
    segments: List[str], index: int, schema: DocFieldSchema | None
) -> bool:
    return _follows_record_subsection(
        segments, index, schema
    ) and not _has_backward_procedure_for_heading(segments, index, schema)


def _collect_record_section_parts(
    segments: List[str],
    start: int,
    schema: DocFieldSchema | None,
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
        if _segment_starts_new_section(nxt) and not _is_orphan_record_field_segment(
            nxt, schema
        ):
            break
        if _is_orphan_record_field_segment(nxt, schema):
            joined = "\n\n".join(parts)
            nxt_key = schema.field_key(_segment_first_line(nxt)) if schema else None
            if (
                schema is not None
                and nxt_key == schema.initiator
                and _segment_has_initiator(joined, schema)
            ):
                break
            if (
                schema is not None
                and schema.is_metadata(nxt_key)
                and _record_section_step_closed(joined, schema)
            ):
                break
            parts.append(nxt)
            j += 1
            continue
        if _is_image_ref_segment(nxt):
            parts.append(nxt)
            j += 1
            continue
        if _TABLE_INGEST_MARKER in nxt:
            break
        if len(_segment_first_line(nxt)) <= _COALESCE_HEADING_MAX_CHARS:
            parts.append(nxt)
            j += 1
            continue
        break
    return parts, j


def _assemble_record_section_segments(
    segments: List[str], schema: DocFieldSchema | None
) -> List[str]:
    """Group numbered headings with following record field lines before the figure."""
    working = [
        (segment or "").strip() for segment in segments if (segment or "").strip()
    ]
    out: List[str] = []
    i = 0
    while i < len(working):
        seg = working[i]
        if _is_image_ref_segment(seg) or _TABLE_INGEST_MARKER in seg:
            out.append(seg)
            i += 1
            continue
        if _is_orphan_heading_segment(seg):
            if _should_collect_record_subsection(working, i, schema):
                parts, i = _collect_record_section_parts(working, i, schema)
                out.append("\n\n".join(parts))
                continue
            out.append(seg)
            i += 1
            continue
        if _segment_starts_new_section(seg):
            parts, i = _collect_record_section_parts(working, i, schema)
            out.append("\n\n".join(parts))
            continue
        if _is_orphan_record_metadata_only_segment(seg, schema):
            parts, i = _collect_record_section_parts(working, i, schema)
            out.append("\n\n".join(parts))
            continue
        out.append(seg)
        i += 1
    return out


def _preceding_section_accepts_trailing_fields(
    segment: str, schema: DocFieldSchema | None
) -> bool:
    """True when a prior block may absorb following record field lines."""
    seg = (segment or "").strip()
    if not seg or _is_image_ref_segment(seg) or _TABLE_INGEST_MARKER in seg:
        return False
    if _record_section_step_closed(seg, schema):
        return False
    if _is_orphan_record_field_segment(seg, schema):
        return False
    return (
        _segment_starts_new_section(seg)
        or _segment_has_inline_image(seg)
        or _segment_has_record_body(seg, schema)
    )


def _merge_trailing_procedure_into_preceding(
    segments: List[str], schema: DocFieldSchema | None
) -> List[str]:
    """Attach trailing record field blocks to the preceding section (Q7)."""
    working = [
        (segment or "").strip() for segment in segments if (segment or "").strip()
    ]
    if schema is None:
        return working
    max_passes = max(len(working) * 2, 8)
    for _ in range(max_passes):
        changed = False
        i = 1
        while i < len(working):
            seg = working[i]
            if not _is_orphan_record_field_segment(seg, schema):
                i += 1
                continue
            first_key = schema.field_key(_segment_first_line(seg))
            if first_key is None:
                i += 1
                continue
            prev = working[i - 1]
            if not _preceding_section_accepts_trailing_fields(prev, schema):
                i += 1
                continue
            if first_key == schema.initiator and _segment_has_initiator(prev, schema):
                i += 1
                continue
            if schema.is_metadata(first_key) and any(
                key == first_key for key in _segment_field_keys(prev, schema)
            ):
                i += 1
                continue
            if first_key == schema.closer and _segment_has_closer(prev, schema):
                i += 1
                continue
            working[i - 1] = f"{prev}\n\n{seg}"
            del working[i]
            changed = True
        if not changed:
            break
    return working


def _merge_unheaded_fields_with_heading_sections(
    segments: List[str], schema: DocFieldSchema | None
) -> List[str]:
    """Merge anonymous record field blocks into a later thin heading section (Q7)."""
    working = [
        (segment or "").strip() for segment in segments if (segment or "").strip()
    ]
    skip: set[int] = set()
    for i in range(len(working)):
        if i in skip:
            continue
        seg = working[i]
        if not _section_heading_needs_field_merge(seg, schema):
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
            if _segment_starts_new_section(cand) and _segment_has_closer(cand, schema):
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


def _merge_orphan_record_metadata_segments(
    segments: List[str], schema: DocFieldSchema | None
) -> List[str]:
    """Forward-merge orphan field lines into the next body block within a short window."""
    working = [
        (segment or "").strip() for segment in segments if (segment or "").strip()
    ]
    if schema is None:
        return working
    max_passes = max(len(working) * 2, 8)
    for _ in range(max_passes):
        changed = False
        i = 0
        while i < len(working):
            if not _is_orphan_record_field_segment(working[i], schema):
                i += 1
                continue
            meta_parts: List[str] = []
            j = i
            while j < len(working) and _is_orphan_record_field_segment(
                working[j], schema
            ):
                meta_parts.append(working[j])
                j += 1
            if meta_parts and _record_section_step_closed(
                "\n\n".join(meta_parts), schema
            ):
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
                if _segment_starts_new_section(
                    cand
                ) and not _is_orphan_record_field_segment(cand, schema):
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
                        and not _is_orphan_record_field_segment(cand, schema)
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


def _is_thin_section_lead_segment(
    segment: str, schema: DocFieldSchema | None
) -> bool:
    """Numbered section heading + optional short tail, no inline figure (Template B manuals)."""
    seg = (segment or "").strip()
    if not seg or _segment_has_inline_image(seg) or _TABLE_INGEST_MARKER in seg:
        return False
    if not _segment_starts_new_section(seg):
        return False
    lines = [line.strip() for line in seg.splitlines() if line.strip()]
    if not lines or len(lines) > 4:
        return False
    if _segment_has_closer(seg, schema):
        return False
    if _segment_has_initiator(seg, schema) and any(
        schema.is_metadata(key) for key in _segment_field_keys(seg, schema)
    ):
        return False
    return True


def _merge_thin_section_with_following_figure(
    segments: List[str], schema: DocFieldSchema | None
) -> List[str]:
    """Merge thin ``N.N 标题`` blocks with the immediate next figure-bearing segment."""
    working = [
        (segment or "").strip() for segment in segments if (segment or "").strip()
    ]
    out: List[str] = []
    i = 0
    n = len(working)
    while i < n:
        seg = working[i]
        if (
            _is_thin_section_lead_segment(seg, schema)
            and i + 1 < n
            and _segment_has_inline_image(working[i + 1])
            and not _segment_starts_new_section(working[i + 1])
        ):
            out.append(f"{seg}\n\n{working[i + 1]}")
            i += 2
            continue
        out.append(seg)
        i += 1
    return out


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
        if (
            leading_text
            and text_term_alignment_symmetric(leading_text, match_text) >= 0.28
        ):
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
    schema: DocFieldSchema | None,
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
            if _segment_has_record_body(img_seg, schema):
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
    schema: DocFieldSchema | None,
) -> float:
    best = 0.0
    for j in range(max(0, heading_index - _COALESCE_HEADING_LOOKAHEAD), heading_index):
        cand = working[j]
        if _is_orphan_heading_segment(cand) or _is_image_ref_segment(cand):
            continue
        if _TABLE_INGEST_MARKER in cand:
            continue
        if not _segment_has_record_body(cand, schema):
            continue
        best = max(best, _heading_body_merge_score(heading, cand, working, j, schema))
    return best


def _record_step_alignment(
    heading: str, candidate: str, schema: DocFieldSchema | None
) -> float:
    if schema is None:
        return 0.0
    best = 0.0
    for line in (candidate or "").splitlines():
        if schema.field_key(line) == schema.closer:
            best = max(best, text_term_alignment_symmetric(heading, line))
    return best


def _merge_trailing_orphan_headings_backward(
    working: List[str], schema: DocFieldSchema | None
) -> List[str]:
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
        and _segment_has_record_body(seg, schema)
    ]
    if not body_indices:
        return working
    pairs: List[tuple[float, float, int, int, int]] = []
    for hi in heading_indices:
        heading = working[hi]
        for bi in body_indices:
            score = _heading_body_merge_score(heading, working[bi], working, bi, schema)
            if score >= 0.12:
                step_align = _record_step_alignment(heading, working[bi], schema)
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


def _merge_orphan_heading_segments(
    segments: List[str], schema: DocFieldSchema | None
) -> List[str]:
    """Attach orphan section headings (e.g. ``2.1.1 …``) to the best-aligned body block."""
    working = [
        (segment or "").strip() for segment in segments if (segment or "").strip()
    ]
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
            for j in range(
                i + 1, min(i + _COALESCE_HEADING_LOOKAHEAD + 1, len(working))
            ):
                cand = working[j]
                if _is_orphan_heading_segment(cand):
                    break
                if _TABLE_INGEST_MARKER in cand or _is_image_ref_segment(cand):
                    continue
                score = _heading_body_merge_score(heading, cand, working, j, schema)
                if score > best_score:
                    best_score = score
                    best_j = j
            backward_best = _best_heading_body_score_before(heading, working, i, schema)
            if (
                best_j >= 0
                and backward_best >= 0.12
                and backward_best >= best_score * 0.38
            ):
                i += 1
                continue
            if best_j < 0:
                i += 1
                continue
            absorb_image = best_j + 1 < len(working) and _is_image_ref_segment(
                working[best_j + 1]
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
    return _merge_trailing_orphan_headings_backward(working, schema)


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


def coalesce_text_image_segments(
    segments: List[str], schema: DocFieldSchema | None = None
) -> List[str]:
    """Merge anchor text with inline figures for indexing (P2: headings + lookahead).

    Keeps figure metadata in the same vector chunk as the anchor body so rerank +
    steering filter text and inline figures together (Route A ingest).

    ``schema`` is normally induced once over the whole document by the table-aware
    entry points and threaded down so tiny single-record section groups still share
    the document-level field roles.  When omitted (direct callers) it is induced
    locally over ``segments``.
    """
    if not segments:
        return []
    if schema is None:
        schema = induce_field_schema(segments)
    exploded = _explode_overmerged_segments(segments, schema)
    assembled = _assemble_record_section_segments(exploded, schema)
    metadata = _merge_orphan_record_metadata_segments(assembled, schema)
    bridged = _merge_unheaded_fields_with_heading_sections(metadata, schema)
    merged = _merge_orphan_heading_segments(bridged, schema)
    split = _split_multi_image_segments(merged)
    thin_fig = _merge_thin_section_with_following_figure(split, schema)
    paired = _lookahead_pair_text_image_segments(thin_fig)
    trailing = _merge_trailing_procedure_into_preceding(paired, schema)
    return _coalesce_immediate_text_image_segments(trailing)


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


def _part_starts_section_heading(part: str) -> bool:
    return _is_section_heading_line(_segment_first_line(part))


def _split_sub_parts_at_section_heading_boundaries(
    sub_parts: List[str],
) -> List[List[str]]:
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


def _coalesce_sub_parts_by_section(
    sub_parts: List[str], schema: DocFieldSchema | None = None
) -> List[str]:
    """Run P2 coalesce within each numbered-section batch (avoids mega-doc cross-merge)."""
    out: List[str] = []
    for group in _split_sub_parts_at_section_heading_boundaries(sub_parts):
        out.extend(coalesce_text_image_segments(group, schema=schema))
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
        if not parts:
            return []
        return _coalesce_sub_parts_by_section(parts, induce_field_schema(parts))
    segments = [s.strip() for s in (text_content or "").split("\n\n") if s.strip()]
    if not segments:
        return []
    return _coalesce_sub_parts_by_section(segments, induce_field_schema(segments))


def build_table_aware_ingest_segments(document_parts: List[str]) -> List[str]:
    """Keep each ``[Table]`` block intact; coalesce adjacent non-table parts."""
    segments: List[str] = []
    buf: List[str] = []
    # Induce field roles once over the whole document so single-record section
    # groups still share the document-level schema (run-signature needs >=3 reps).
    schema = induce_field_schema([p for p in document_parts if (p or "").strip()])

    def flush_buf() -> None:
        nonlocal buf
        if not buf:
            return
        sub_parts = [p.strip() for p in buf if (p or "").strip()]
        segments.extend(_coalesce_sub_parts_by_section(sub_parts, schema))
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


def compute_table_aware_ingest_segments(
    lightrag, document_parts: List[str]
) -> List[str]:
    """Prepared ingest segments (table-aware split; no flatten when matrix ingest is on)."""
    if not table_aware_ingest_enabled() or not document_parts:
        return list(document_parts or [])
    tokenizer = getattr(lightrag, "tokenizer", None)
    if tokenizer is None:
        return list(document_parts)
    max_tokens = _ingest_chunk_token_size(lightrag)
    return prepare_table_aware_ingest_segments(document_parts, tokenizer, max_tokens)


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
